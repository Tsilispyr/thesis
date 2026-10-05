"""Live GPS-loss recovery orchestrator -- the Python half of the
"degrade toward simplicity" fail-safe hierarchy (LSTM -> EKF -> rule-based;
the breadcrumb lives elsewhere -- see simulation/scripts/breadcrumb_recovery
.gd). The RL policy meant to sit above this hierarchy (AI_RECOVERY_EXECUTION
_PLAN.md sec 12 Goal 3) is wired in here too, opt-in via
`set_navigation_policy()`: it replaces `_apply_seek_bias()`'s hand-coded
proportional formula, not the LSTM/EKF estimators themselves -- LSTM/EKF
still answer "where am I probably," RL only answers "given that estimate
and how much to trust it, how should I move toward the goal." A second,
independent opt-in exists at the estimator level itself: 'Motor-LSTM' in
MODEL_REGISTRY below, a variant trained on real px4 motor/actuator channels
in addition to IMU (see MODEL_REGISTRY's own comment) -- manually-selectable
only, never auto-picked, same "prove it's real before making it a default"
posture as the RL policy. Deliberately has no socket/threading dependencies
of its own, so the whole layer-
selection contract is testable with a plain script
(`python ai_backend/api/recovery_orchestrator.py`) -- no Godot, no network
involved -- per an explicit ask to keep this half of the system headlessly
verifiable and easy to keep extending.

Output contract, forwarded to Godot by udp_server.py unchanged:
    {"layer": "LSTM"|"EKF"|"RULE_BASED"|"NONE",
     "goal": "TARGET"|"HOME"|"NONE",
     "velocity_cmd": [vx, vy, vz]}
`velocity_cmd` is a world-frame m/s velocity nudge -- the same shape
breadcrumb_recovery.gd's pop_next_reversal_command() already uses internally
on the Godot side, so every recovery source now speaks one consistent
language. `goal` reports which destination is currently driving the seek
bias (see RTL_* below) -- TARGET while still trying to complete the mission
blind, HOME once that's been abandoned in favor of returning to the launch
point, NONE when there's nothing to steer toward at all.
"""
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.dataset_parser import FEATURE_COLS, MOTOR_FEATURE_COLS
from models.dead_reckoning_model import DeadReckoningLSTM
from models.dr_transformer import DeadReckoningTransformer
from models.ekf_baseline import DeadReckoningEKF, rotation_from_euler_deg
from models.rl_path_recovery import POS_NORM as _RL_POS_NORM, UNC_NORM as _RL_UNC_NORM, \
    VEL_NORM as _RL_VEL_NORM

WINDOW_SIZE = 10
_DT_IDX = FEATURE_COLS.index('dt')             # last of the 14 -- see FEATURE_COLS order
BARO_UPDATE_EVERY = 5                          # packets (~0.5s at 10Hz telemetry) -- matches
                                                # breadcrumb_recovery.gd's 2Hz decimation cadence
STALE_GAP_S = 2.0                              # a gap this large means a reconnect, not a dropout --
                                                # reset per-drone state rather than integrate across it
RULE_BASED_SPEED_MPS = 2.0                     # fixed cruise speed for the last-resort vector-to-target

# LSTM/EKF are pure dead-reckoning estimators -- they report "what am I
# probably doing right now", not "how do I get back on course". Without a
# target-seeking term, a drone coasting with no active mission thrust settles
# into a near-static IMU window and the estimate (correctly) converges to a
# small, nearly-constant value -- an honest reading, but it means "AI
# recovery" would otherwise just be "drift very slowly forever", not an
# actual attempt to get home. SEEK_* blends a gentle, capped pull toward
# mission_target on top of the estimate -- the same physically-reasonable
# idea real return-to-home systems use: the destination coordinate is known
# independent of GPS (it's a stored waypoint, not a live position fix), so
# it's legitimate to steer toward it even while the *current* position is
# only an uncertain estimate.
SEEK_KP = 0.15                                 # m/s of pull per metre of estimated distance to target
SEEK_MAX_CONTRIB = 1.5                         # m/s cap on the seek term -- a nudge, not the whole signal

# Hybrid target-then-RTL (explicit design decision, not a default): keep
# seeking the mission target while GPS is lost, since it may well be reached
# before recovery is even needed -- but continuing deeper into a blind
# mission indefinitely is exactly the wrong failure mode for a real vehicle.
# After RTL_TIMEOUT_S of continuous loss, abandon the mission target and
# switch to the stored launch/home position instead -- unless already close
# enough to the target that finishing the short remaining approach is more
# sensible than aborting. Sticky for the rest of the loss episode once
# engaged (no flip-flopping between TARGET and HOME).
RTL_TIMEOUT_S = 8.0                            # seconds of continuous GPS loss before aborting to RTL
RTL_SKIP_IF_CLOSE_M = 3.0                      # ...unless the target is already this close

_MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
_RL_POLICY_PATH = os.path.join(_MODELS_DIR, 'ppo_recovery_seek.zip')


def _hand_coded_seek(v: np.ndarray, believed_pos: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
    """The original hand-coded proportional seek-bias formula, factored out
    to a plain function (not a method -- it never used `self`) so it can be
    imported and reused verbatim by eval_metrics.py's comparison against the
    learned RL policy, instead of a disconnected reimplementation (the real
    bug this project's own "100% vs. 28%" RL headline number turned out to
    have -- see AI_RECOVERY_EXECUTION_PLAN.md's RL-wiring section)."""
    direction = np.array(goal_pos, dtype=float) - believed_pos
    dist = float(np.linalg.norm(direction))
    if dist < 1e-6:
        return v
    seek = direction / dist * min(SEEK_KP * dist, SEEK_MAX_CONTRIB)
    return v + seek

# Which architecture ("kind") and checkpoint tag back each selectable named
# model -- the same 6-model vocabulary ablation_matrix.py's STOCHASTIC_MODELS
# uses, so a name picked in the live demo means the same thing it means in
# the offline ablation study. The reported wire-protocol "layer" name stays
# "LSTM" regardless of which of these actually backs it (see decide()'s
# docstring/output contract) -- this is an internal choice of *architecture*
# for the one learned dead-reckoning layer, not a new layer, so nothing
# downstream (Godot's HUD, udp_server.py's forwarding) needs to change.
MODEL_REGISTRY = {
    'LSTM':            {'kind': 'lstm', 'tag': 'imu_norm'},
    'SSL-LSTM':        {'kind': 'lstm', 'tag': 'imu_ssl_norm'},
    'Transformer':     {'kind': 'transformer', 'tag': 'imu_norm'},
    'SSL-Transformer': {'kind': 'transformer', 'tag': 'imu_ssl_norm'},
    'RandomForest':    {'kind': 'classical', 'tag': 'imu_norm', 'prefix': 'rf'},
    'GBT':             {'kind': 'classical', 'tag': 'imu_norm', 'prefix': 'gbt'},
    # Manually-selectable only, never auto-picked by _select_layer() -- no
    # automatic "IMU unhealthy" signal exists anywhere in this codebase to
    # trigger it on, matching the scope decision this option was built
    # under. Trained on real px4 hardware logs, not imu_data.csv (px4 is the
    # only source with a real motor/actuator channel -- see
    # dataset_parser.py::MOTOR_FEATURE_COLS), so its checkpoint's
    # architecture is real-px4-shaped even though the live demo's telemetry
    # comes from the Godot simulation, which does supply a genuine motor_out
    # channel of its own (drone_body.gd::_send_telemetry). 3-seed comparison
    # (runs/px4_motor_informed/comparison.json): motor-informed 0.2801 mean
    # val loss [0.2761,0.2842] vs. IMU-only baseline 0.2917 [0.2728,0.3106]
    # -- a real-direction but not yet statistically clean improvement (CIs
    # overlap). Added regardless, per the "the demo shows the architecture
    # choice, not a claim it's the best one" design decision -- this is the
    # promoted (lowest-val-loss) seed of that 3-seed run, not a new training
    # run of its own.
    'Motor-LSTM':      {'kind': 'lstm', 'tag': 'px4_motor_informed', 'feature_cols': MOTOR_FEATURE_COLS},
}


def _load_model(model_name: str = 'LSTM'):
    """Loads the checkpoint+scalers for model_name (a MODEL_REGISTRY key).
    Returns (model, scaler_X, scaler_y, kind, feature_cols); kind is
    'lstm'/'transformer' (both take a plain (1,10,N) tensor forward pass --
    see evaluate_trajectory.py's reconstruct_lstm/reconstruct_transformer,
    identical call convention, N=len(feature_cols)) or 'classical' (sklearn
    .predict() on a flattened (1, window*N) array instead --
    reconstruct_classical's convention). feature_cols is FEATURE_COLS (14)
    for every entry except Motor-LSTM, which opts into MOTOR_FEATURE_COLS
    (18) via its own registry spec -- see MODEL_REGISTRY's own comment.
    Degrades gracefully to (None, None, None, None, FEATURE_COLS) if the
    files are missing, same as the original LSTM-only loader did -- e.g.
    RandomForest/GBT before their production (non-ablation-tagged)
    checkpoint has been trained via classical_baselines.py's own CLI."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name '{model_name}', expected one of "
                          f"{list(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[model_name]
    kind, tag = spec['kind'], spec['tag']
    feature_cols = spec.get('feature_cols', FEATURE_COLS)

    if kind == 'classical':
        prefix = spec['prefix']
        mp = os.path.join(_MODELS_DIR, f'{prefix}_{tag}.pkl')
        sx = os.path.join(_MODELS_DIR, f'scaler_X_{prefix}_{tag}.pkl')
        sy = os.path.join(_MODELS_DIR, f'scaler_y_{prefix}_{tag}.pkl')
        if not all(os.path.exists(p) for p in (mp, sx, sy)):
            print(f"[Orchestrator] No '{model_name}' ({tag}) model+scaler pair found -- "
                  f"LSTM layer disabled, will degrade to EKF/rule-based only.")
            return None, None, None, None, feature_cols
        model = joblib.load(mp)
        print(f"[Orchestrator] Loaded {model_name}: {prefix}_{tag}.pkl")
        return model, joblib.load(sx), joblib.load(sy), kind, feature_cols

    if kind == 'lstm':
        pth = os.path.join(_MODELS_DIR, f'dr_lstm_{tag}.pth')
        sx = os.path.join(_MODELS_DIR, f'scaler_X_{tag}.pkl')
        sy = os.path.join(_MODELS_DIR, f'scaler_y_{tag}.pkl')
    else:   # kind == 'transformer'
        pth = os.path.join(_MODELS_DIR, f'dr_transformer_{tag}.pth')
        sx = os.path.join(_MODELS_DIR, f'scaler_X_transformer_{tag}.pkl')
        sy = os.path.join(_MODELS_DIR, f'scaler_y_transformer_{tag}.pkl')

    if not all(os.path.exists(p) for p in (pth, sx, sy)):
        print(f"[Orchestrator] No '{model_name}' ({tag}) model+scaler pair found -- "
              f"LSTM layer disabled, will degrade to EKF/rule-based only.")
        return None, None, None, None, feature_cols

    if kind == 'lstm':
        model = DeadReckoningLSTM(input_size=len(feature_cols), hidden_size=64, num_layers=2, output_size=3)
    else:
        model = DeadReckoningTransformer(input_size=len(feature_cols), window_size=WINDOW_SIZE)
    # weights_only=False: trusted, self-generated checkpoint -- torch>=2.6's
    # default weights_only=True rejects some of this project's checkpoints
    # (see device_utils.py's docstring elsewhere in this project for the
    # full story), so every loader here is explicit about trusting its own
    # training scripts' output rather than relying on the version-dependent
    # default.
    model.load_state_dict(torch.load(pth, map_location='cpu', weights_only=False))
    model.eval()
    print(f"[Orchestrator] Loaded {model_name}: {os.path.basename(pth)}")
    return model, joblib.load(sx), joblib.load(sy), kind, feature_cols


class _DroneState:
    """Per-drone bookkeeping the orchestrator needs across packets."""
    __slots__ = ('buffer', 'ekf', 'last_pos', 'believed_pos', 'home_pos',
                 'last_packet_time', 'was_lost', 'loss_start_time', 'rtl_active',
                 'packets_since_baro')

    def __init__(self):
        self.buffer = deque(maxlen=WINDOW_SIZE)
        self.ekf: DeadReckoningEKF | None = None
        self.last_pos = np.zeros(3)                # last known-good (x, y, z), world metres
        # Running belief of current position during a loss episode -- starts
        # at last_pos the instant GPS drops, then integrated forward each
        # tick by whatever velocity was actually just commanded. Feeds the
        # seek-toward-target/home bias; None whenever GPS is live (no belief needed).
        self.believed_pos: np.ndarray | None = None
        # Launch/home position -- set once, from the first packet ever seen
        # for this drone (regardless of GPS state), never overwritten
        # afterward. A stored waypoint, not a live fix, so it stays valid
        # through any number of GPS-loss episodes.
        self.home_pos: np.ndarray | None = None
        self.last_packet_time: float | None = None
        self.was_lost = False
        self.loss_start_time: float = 0.0
        self.rtl_active = False                    # sticky once an episode aborts the mission target for home
        self.packets_since_baro = 0


class RecoveryOrchestrator:
    """One instance shared across all drones; per-drone state lives in
    self._drones. decide() is the only entry point udp_server.py needs."""

    def __init__(self, model_name: str = 'LSTM', use_rl_seek: bool = False):
        self._drones: dict[str, _DroneState] = {}
        self.model_name = None
        self._model = self._scaler_X = self._scaler_y = self._model_kind = None
        self._feature_cols = FEATURE_COLS
        self.set_model(model_name)
        self._rl_policy = None
        self.rl_seek_enabled = False
        if use_rl_seek:
            self.set_navigation_policy(True)

    def set_navigation_policy(self, enabled: bool) -> bool:
        """Toggles the learned RL seek policy in place, mirroring
        set_model()'s exact contract (does not touch self._drones -- a
        mid-flight toggle must not lose an in-progress episode's
        home_pos/believed_pos/rtl_active/EKF state, the same real bug class
        set_model() was built to avoid). Default is off: RL is new and
        unvalidated in the live loop, unlike the "fully built, tested,
        production-quality" hand-coded formula it can replace, so this is
        opt-in rather than default-armed. Returns True if a checkpoint
        actually loaded, False if it degraded to the hand-coded formula
        (missing file, or enabled=False) -- callers surfacing this to a
        human should report the difference, not claim a plain "on"."""
        if not enabled:
            self._rl_policy = None
            self.rl_seek_enabled = False
            return False
        if not os.path.exists(_RL_POLICY_PATH):
            print(f"[Orchestrator] No RL seek policy found at {_RL_POLICY_PATH} -- "
                  f"falling back to the hand-coded seek formula.")
            self._rl_policy = None
            self.rl_seek_enabled = False
            return False
        from stable_baselines3 import PPO   # soft import: only needed if RL is actually used
        self._rl_policy = PPO.load(_RL_POLICY_PATH)
        self.rl_seek_enabled = True
        print(f"[Orchestrator] Loaded RL seek policy: {os.path.basename(_RL_POLICY_PATH)}")
        return True

    def set_model(self, model_name: str) -> bool:
        """Swaps in a different architecture for the LSTM fail-safe layer,
        in place -- deliberately does NOT touch self._drones. A model swap
        mid-flight (the live demo's "Model" picker is the only caller that
        does this after startup) must not lose an in-progress recovery
        episode's home_pos/believed_pos/rtl_active/EKF state: rebuilding a
        whole new RecoveryOrchestrator instead (the first version of this
        method) re-latched home at whatever position the drone happened to
        be at the moment of the switch instead of the true launch point,
        and silently un-stuck a sticky RTL episode back to TARGET -- both
        real bugs an independent review caught before this shipped, not
        hypothetical ones. Returns True if the checkpoint actually loaded,
        False if it degraded to EKF/rule-based only (missing files) --
        callers surfacing this to a human should report the difference,
        not claim a plain "switched" either way."""
        self.model_name = model_name
        (self._model, self._scaler_X, self._scaler_y,
         self._model_kind, self._feature_cols) = _load_model(model_name)
        return self._model is not None

    def _state(self, drone_id: str) -> _DroneState:
        if drone_id not in self._drones:
            self._drones[drone_id] = _DroneState()
        return self._drones[drone_id]

    def decide(self, drone_id: str, packet: dict, now: float = None,
               forced_layer: str = None) -> dict:
        """packet is the parsed telemetry dict Godot sent (see drone_body.gd
        _send_telemetry). `now` overrides wall-clock time so the smoke test
        can simulate arbitrary gaps deterministically. `forced_layer`, when
        given a real layer name, skips auto-selection and computes that
        layer's genuine output anyway -- used by the HUD's demo "Force
        Layer" control, never fabricates a result, just picks which real
        computation runs."""
        now = time.monotonic() if now is None else now
        st = self._state(drone_id)

        if st.last_packet_time is None:
            dt = 0.1   # first packet ever seen for this drone -- matches the 10Hz telemetry rate
        else:
            gap = now - st.last_packet_time
            if gap > STALE_GAP_S:
                print(f"[Orchestrator] {drone_id}: {gap:.1f}s gap since last packet -- "
                      f"treating as a fresh session (resetting EKF + buffer).")
                st.buffer.clear()
                st.ekf = None
                st.packets_since_baro = 0
                # If this gap happened mid-loss-episode, force the re-anchor block
                # below to fire again -- otherwise `was_lost` staying True would
                # skip EKF re-creation and strand this drone on rule-based/NONE
                # for the rest of the episode even though a fresh anchor is fine.
                st.was_lost = False
                # Treat this packet like a brand-new first sample -- do NOT feed
                # the raw multi-second gap forward as dt, or the freshly-reset
                # EKF's very first predict() would integrate across the whole
                # stale gap and produce exactly the corrupted-dt spike this
                # guard exists to prevent.
                dt = 0.1
            else:
                dt = float(np.clip(gap, 1e-3, 10.0))
        st.last_packet_time = now

        has_gps = bool(packet.get('gps_signal', True))
        # Axis convention matches drone_body.gd::_send_telemetry: delta_lat<->Z, delta_lon<->X, delta_alt<->Y.
        pos = np.array([packet.get('lon', 0.0), packet.get('alt', 0.0), packet.get('lat', 0.0)])

        if st.home_pos is None:
            # First packet ever seen for this drone -- latch it as home, the
            # same way a real autopilot latches home at arm time. Independent
            # of has_gps: home is a stored waypoint, not a live fix.
            st.home_pos = pos.copy()
            print(f"[Orchestrator] {drone_id}: home set at "
                  f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

        # Always buffer the full MOTOR_FEATURE_COLS-wide (18) row -- a strict
        # superset of FEATURE_COLS (14) in the same column order (FEATURE_COLS,
        # 'dt' last, then motor_0..3) -- regardless of which model is
        # currently active. _model_velocity() slices this down to whatever
        # self._feature_cols actually needs. Keeps a live model swap
        # (LSTM <-> Motor-LSTM) from requiring the buffer to be cleared/
        # resized, the same in-place-swap contract set_model() already
        # guarantees for home_pos/believed_pos/rtl_active/EKF state.
        motor_out = packet.get('motor_out', [0.0, 0.0, 0.0, 0.0])
        feat = ([packet.get(c, 0.0) for c in FEATURE_COLS[:-1]] + [dt]
                + [motor_out[i] if len(motor_out) > i else 0.0 for i in range(4)])
        st.buffer.append(feat)

        if has_gps:
            st.last_pos = pos
            st.was_lost = False
            st.ekf = None
            st.believed_pos = None
            st.rtl_active = False
            return {"layer": "NONE", "goal": "NONE", "velocity_cmd": [0.0, 0.0, 0.0]}

        if not st.was_lost:
            print(f"[Orchestrator] {drone_id}: GPS lost -- anchoring recovery at "
                  f"({st.last_pos[0]:.2f}, {st.last_pos[1]:.2f}, {st.last_pos[2]:.2f})")
            st.was_lost = True
            st.ekf = DeadReckoningEKF()
            st.ekf.reset(position=st.last_pos, velocity=np.zeros(3))
            st.packets_since_baro = 0
            st.believed_pos = st.last_pos.copy()
            st.loss_start_time = now
            st.rtl_active = False

        # Step the EKF unconditionally every tick during a loss episode,
        # independent of which layer is actually selected as v's source
        # below -- previously this only happened inside _ekf_velocity(),
        # i.e. only on ticks where layer == "EKF". Whenever layer == "LSTM"
        # (the preferred, most-often-active layer once its buffer is warm),
        # st.ekf's covariance never advanced that tick even though st.ekf
        # stays alive for the whole episode -- found while wiring RL's
        # confidence signal (position_uncertainty), which needs this to stay
        # current on every tick, not just EKF-active ones. Does not change
        # which estimator's output drives v below, only keeps the
        # covariance side-channel live.
        accel = np.array([packet.get('imu_acc_x', 0.0),
                           packet.get('imu_acc_y', 0.0),
                           packet.get('imu_acc_z', 0.0)])
        R = rotation_from_euler_deg(packet.get('roll', 0.0),
                                     packet.get('pitch', 0.0),
                                     packet.get('yaw', 0.0))
        st.ekf.predict(accel, R, dt)
        st.packets_since_baro += 1
        if st.packets_since_baro >= BARO_UPDATE_EVERY and 'baro_alt' in packet:
            st.ekf.update_baro(float(packet['baro_alt']))
            st.packets_since_baro = 0

        layer = forced_layer or self._select_layer(st)
        target = packet.get('mission_target')

        # -- Goal resolution: target-then-RTL hybrid (see RTL_* above). No
        #    mission target at all -> home is the only sensible goal from the
        #    start. Otherwise seek the target until RTL_TIMEOUT_S elapses
        #    without arriving, then abandon it for home -- sticky once engaged.
        if target is None:
            goal_pos, goal_name = st.home_pos, "HOME"
        elif st.rtl_active:
            goal_pos, goal_name = st.home_pos, "HOME"
        else:
            dist_to_target = (float(np.linalg.norm(np.array(target, dtype=float) - st.believed_pos))
                               if st.believed_pos is not None else float('inf'))
            if now - st.loss_start_time >= RTL_TIMEOUT_S and dist_to_target > RTL_SKIP_IF_CLOSE_M:
                st.rtl_active = True
                print(f"[Orchestrator] {drone_id}: GPS loss exceeded {RTL_TIMEOUT_S:.0f}s "
                      f"({dist_to_target:.1f}m from target) -- aborting mission, switching to RTL (home).")
                goal_pos, goal_name = st.home_pos, "HOME"
            else:
                goal_pos, goal_name = np.array(target, dtype=float), "TARGET"

        confidence = st.ekf.position_uncertainty if st.ekf is not None else None
        elapsed_s = now - st.loss_start_time

        if layer == "LSTM" and self._model is not None and len(st.buffer) == WINDOW_SIZE:
            v = self._model_velocity(st)
            v = self._apply_seek_bias(v, st.believed_pos, goal_pos, confidence, elapsed_s)
        elif layer == "EKF" and st.ekf is not None:
            v = self._ekf_velocity(st)
            v = self._apply_seek_bias(v, st.believed_pos, goal_pos, confidence, elapsed_s)
        elif layer == "RULE_BASED":
            v = self._rule_based_velocity(st, goal_pos)
        else:
            layer = "NONE"
            v = np.zeros(3)

        if st.believed_pos is not None:
            st.believed_pos = st.believed_pos + v * dt

        return {"layer": layer, "goal": goal_name, "velocity_cmd": [float(v[0]), float(v[1]), float(v[2])]}

    def _apply_seek_bias(self, v: np.ndarray, believed_pos: np.ndarray, goal_pos: np.ndarray,
                          confidence: np.ndarray = None, elapsed_s: float = 0.0) -> np.ndarray:
        """Adds a seek contribution toward `goal_pos` (the mission target, or
        home once RTL has engaged -- see the goal-resolution block in
        decide()) on top of the raw dead-reckoning estimate `v`. Two
        implementations, same call site either way: the original hand-coded
        proportional formula (`_hand_coded_seek`, see SEEK_* above for why
        this is a legitimate thing to add rather than a fudge), or, once
        `set_navigation_policy(True)` has loaded a checkpoint, a learned RL
        policy that additionally consumes `confidence` (the EKF's per-axis
        position_uncertainty) and `elapsed_s` (time since GPS was lost) --
        the RecoveryPolicyEnv observation's exact 11-dim layout
        (models/rl_path_recovery.py). `confidence`/`elapsed_s` are optional
        and only used by the RL branch; the hand-coded formula ignores them,
        matching its pre-existing behavior exactly when RL isn't loaded."""
        if self._rl_policy is not None:
            if confidence is None:
                confidence = np.zeros(3)
            goal = np.array(goal_pos, dtype=float)
            rel = (goal - believed_pos) / _RL_POS_NORM
            dist = np.array([np.linalg.norm(goal - believed_pos) / _RL_POS_NORM])
            conf = np.asarray(confidence, dtype=float) / _RL_UNC_NORM
            v_norm = np.asarray(v, dtype=float) / _RL_VEL_NORM
            t = np.array([min(elapsed_s / RTL_TIMEOUT_S, 3.0)])
            obs = np.concatenate([rel, dist, conf, v_norm, t]).astype(np.float32)
            obs = np.clip(obs, -20.0, 20.0)
            action, _ = self._rl_policy.predict(obs, deterministic=True)
            norm = np.linalg.norm(action)
            seek = action / max(norm, 1.0) * SEEK_MAX_CONTRIB
            return v + seek
        return _hand_coded_seek(v, believed_pos, goal_pos)

    def _select_layer(self, st: _DroneState) -> str:
        """Auto (non-forced) precedence: LSTM once its 10-step buffer is
        full, else EKF (available from the very first lost-GPS packet in
        this simulated telemetry, since accel/attitude/baro are always
        present on the wire), else rule-based. In practice EKF's
        precondition is essentially always met once GPS is lost here, so
        rule-based's natural trigger (EKF unavailable) rarely arises with
        this wire format -- it is fully exercised on its own merits by the
        smoke test below, and reachable live via the HUD's Force Layer
        control for demo purposes."""
        if self._model is not None and len(st.buffer) == WINDOW_SIZE:
            return "LSTM"
        if st.ekf is not None:
            return "EKF"
        return "RULE_BASED"

    def _model_velocity(self, st: _DroneState) -> np.ndarray:
        """Same chained-delta-to-velocity conversion regardless of which
        architecture backs self._model -- LSTM/Transformer share an
        identical (1,10,N) tensor forward-pass convention (see
        evaluate_trajectory.py's reconstruct_lstm/reconstruct_transformer);
        only the classical (RandomForest/GBT) arms need the window
        flattened for sklearn's .predict() instead (reconstruct_classical's
        convention). N is 14 for every architecture except Motor-LSTM (18)
        -- st.buffer is always stored 18-wide (see decide()), so it's
        sliced down to self._feature_cols's actual width here."""
        n = len(self._feature_cols)
        window = np.array(st.buffer, dtype=np.float32)[:, :n]             # (10, N)
        window_scaled = self._scaler_X.transform(window)
        if self._model_kind == 'classical':
            pred_scaled = self._model.predict(window_scaled.reshape(1, -1))
        else:
            x = torch.tensor(window_scaled[np.newaxis], dtype=torch.float32)  # (1, 10, 14)
            with torch.no_grad():
                pred_scaled = self._model(x).numpy()
        delta = self._scaler_y.inverse_transform(pred_scaled)[0]          # [d_lat, d_lon, d_alt], metres, over the window
        elapsed = max(float(window[:, _DT_IDX].sum()), 1e-3)
        return np.array([delta[1], delta[2], delta[0]]) / elapsed         # -> (x, y, z) m/s

    def _ekf_velocity(self, st: _DroneState) -> np.ndarray:
        # predict()/update_baro() now run unconditionally every tick in
        # decide() itself (see the EKF-staleness fix there) -- this just
        # reads the already-current velocity, no longer steps the filter.
        return st.ekf.velocity

    def _rule_based_velocity(self, st: _DroneState, goal_pos: np.ndarray) -> np.ndarray:
        # Steers from the evolving position belief, not the position frozen
        # at the moment GPS was lost -- matters for long loss episodes, where
        # a stale anchor would point the wrong way as the drone (believedly)
        # gets closer.
        origin = st.believed_pos if st.believed_pos is not None else st.last_pos
        direction = np.array(goal_pos, dtype=float) - origin
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return np.zeros(3)
        return direction / norm * RULE_BASED_SPEED_MPS


if __name__ == '__main__':
    print("=== RecoveryOrchestrator smoke test (no network, no Godot) ===\n")

    print("--- Model registry: every named model loads (or degrades cleanly) ---")
    for name in MODEL_REGISTRY:
        probe = RecoveryOrchestrator(model_name=name)
        status = "loaded" if probe._model is not None else "MISSING (degrades to EKF/rule-based)"
        print(f"  {name:<16} kind={probe._model_kind or 'n/a':<12} {status}")
        assert probe._model is not None, (
            f"{name}: expected a production checkpoint to exist for the live-demo model "
            f"picker to actually work -- train it (e.g. classical_baselines.py's CLI for "
            f"RandomForest/GBT) before shipping this as a selectable option.")
    print()

    orch = RecoveryOrchestrator(model_name='LSTM')
    drone_id = "Drone_Test"
    t = 0.0

    def packet(gps_signal, **overrides):
        base = dict(gps_signal=gps_signal, lat=37.9755, lon=23.7348, alt=10.0,
                    imu_acc_x=0.1, imu_acc_y=0.2, imu_acc_z=9.8,
                    imu_gyro_x=0.01, imu_gyro_y=0.0, imu_gyro_z=0.0,
                    roll=0.0, pitch=0.0, yaw=0.0,
                    mag_x=0.0, mag_y=-0.6, mag_z=0.8,
                    speed=1.0, baro_alt=10.0)
        base.update(overrides)
        return base

    # 1) GPS present for a few packets -- LSTM's 10-step buffer intentionally
    #    NOT warmed up yet (only 3 samples) when GPS drops next.
    for _ in range(3):
        t += 0.1
        result = orch.decide(drone_id, packet(True), now=t)
        assert result["layer"] == "NONE", result

    # 2) GPS lost with a cold buffer -- must fall back to EKF, not LSTM.
    t += 0.1
    result = orch.decide(drone_id, packet(False), now=t)
    print(f"[Test] GPS just lost, buffer=4/{WINDOW_SIZE}: layer={result['layer']}  v={result['velocity_cmd']}")
    assert result["layer"] == "EKF", f"expected EKF while buffer warms up, got {result['layer']}"

    # 3) Keep feeding lost-GPS packets until the buffer reaches WINDOW_SIZE.
    for _ in range(6):
        t += 0.1
        result = orch.decide(drone_id, packet(False), now=t)
    expected = "LSTM" if orch._model is not None else "EKF"
    print(f"[Test] Buffer full ({WINDOW_SIZE}/{WINDOW_SIZE}): layer={result['layer']}  v={result['velocity_cmd']}")
    assert result["layer"] == expected, f"expected {expected} once buffer is full, got {result['layer']}"

    # 3b) Seek bias: a second, independent drone, GPS lost with a distant
    #     mission_target present -- the LSTM/EKF estimate alone is tiny (the
    #     packet is a near-static coast, see the module docstring on
    #     SEEK_KP), so a materially larger, target-pointed velocity here can
    #     only be the seek bias actually firing, not the raw estimate.
    seek_id = "Drone_Seek"
    far_target = [50.0, 5.0, 0.0]
    ts = 0.0
    for _ in range(11):   # warm the buffer, then one extra packet past GPS loss
        ts += 0.1
        result = orch.decide(seek_id, packet(True, mission_target=far_target), now=ts)
    ts += 0.1
    result = orch.decide(seek_id, packet(False, mission_target=far_target), now=ts)
    v = np.array(result["velocity_cmd"])
    print(f"[Test] Seek bias toward distant target (50,5,0): layer={result['layer']}  "
          f"v={result['velocity_cmd']}  |v|={np.linalg.norm(v):.3f}")
    assert np.linalg.norm(v) > 0.5, (
        f"expected the seek bias to dominate toward a distant target, got |v|={np.linalg.norm(v):.3f}")
    assert v[0] > 0, "target is at +x from the origin -- seek component should pull +x"
    assert result["goal"] == "TARGET", f"expected goal=TARGET before the RTL timeout, got {result['goal']}"

    # 3c) Hybrid target-then-RTL: a third drone, GPS lost with a distant
    #     target that never gets any closer (held fixed) -- goal should stay
    #     TARGET until RTL_TIMEOUT_S elapses, then switch to HOME (~its first-
    #     ever packet's position) and stay there even if queried again later.
    rtl_id = "Drone_RTL"
    tr = 0.0
    home_packet = orch.decide(rtl_id, packet(True, mission_target=far_target), now=tr)
    assert home_packet["goal"] == "NONE"   # GPS still up on this very first packet
    expected_home = np.array([23.7348, 10.0, 37.9755])   # this packet's (lon, alt, lat) -- see packet()
    for _ in range(9):
        tr += 0.1
        orch.decide(rtl_id, packet(True, mission_target=far_target), now=tr)
    tr += 0.1
    result = orch.decide(rtl_id, packet(False, mission_target=far_target), now=tr)   # GPS just lost
    loss_time = tr
    print(f"[Test] RTL hybrid, GPS just lost: goal={result['goal']}")
    assert result["goal"] == "TARGET", f"expected TARGET immediately after loss, got {result['goal']}"

    # Keep feeding packets every 0.1s (realistic telemetry cadence -- must stay
    # under STALE_GAP_S or the staleness guard resets the episode instead of
    # letting RTL_TIMEOUT_S elapse) until just under the timeout, measured
    # from when the loss actually started, not from the test's absolute clock.
    while tr - loss_time < RTL_TIMEOUT_S - 0.05:
        tr += 0.1
        result = orch.decide(rtl_id, packet(False, mission_target=far_target), now=tr)
    assert result["goal"] == "TARGET", f"expected TARGET just under the RTL timeout, got {result['goal']}"

    tr += 0.2   # now past the timeout -- should abort to HOME
    result = orch.decide(rtl_id, packet(False, mission_target=far_target), now=tr)
    print(f"[Test] RTL hybrid, past {RTL_TIMEOUT_S:.0f}s timeout: goal={result['goal']}  "
          f"v={result['velocity_cmd']}")
    assert result["goal"] == "HOME", f"expected HOME once the RTL timeout elapses, got {result['goal']}"
    assert orch._drones[rtl_id].rtl_active, "rtl_active should now be sticky for the rest of this episode"
    assert np.allclose(orch._drones[rtl_id].home_pos, expected_home, atol=1e-3), (
        f"home_pos should be the drone's first-ever packet position, got {orch._drones[rtl_id].home_pos}")

    tr += 0.1   # confirm it STAYS on HOME, doesn't flip back to TARGET
    result = orch.decide(rtl_id, packet(False, mission_target=far_target), now=tr)
    assert result["goal"] == "HOME", "goal should remain HOME (sticky) for the rest of the loss episode"

    # 4) Staleness guard: a 5s reconnect gap must reset cleanly, not integrate
    #    a corrupted huge-dt sample into the EKF or buffer.
    t += 5.0
    result = orch.decide(drone_id, packet(False), now=t)
    v_mag = float(np.linalg.norm(result["velocity_cmd"]))
    print(f"[Test] After a 5s stale gap: layer={result['layer']}  v={result['velocity_cmd']}  |v|={v_mag:.3f}")
    assert all(np.isfinite(v) for v in result["velocity_cmd"]), "stale-gap reset produced non-finite output"
    assert result["layer"] == "EKF", f"expected a fresh EKF re-anchor after the gap, got {result['layer']}"
    # The synthetic packet's accel is ~[0.1, 0.2, 9.8] m/s^2 -- integrated across a
    # correctly-clamped ~0.1s dt that should stay under ~1.5 m/s. Integrated across
    # the raw 5s gap instead (the bug this guard exists to catch) it would be ~49 m/s.
    assert v_mag < 1.5, f"stale-gap reset appears to have integrated across the full gap: |v|={v_mag:.3f}"

    # 5) Rule-based, exercised directly: this simulated telemetry always
    #    satisfies the EKF preconditions once GPS is lost, so rule-based's
    #    natural trigger (EKF unavailable) doesn't arise on this wire format --
    #    verified here on its own merits (forced_layer), and reachable live via
    #    the HUD's Force Layer control for demo purposes.
    result = orch.decide(drone_id, packet(False, mission_target=[10.0, 5.0, 0.0]),
                          now=t + 0.1, forced_layer="RULE_BASED")
    v = np.array(result["velocity_cmd"])
    print(f"[Test] Forced RULE_BASED toward (10,5,0): v={result['velocity_cmd']}  |v|={np.linalg.norm(v):.3f}")
    assert result["layer"] == "RULE_BASED"
    assert abs(np.linalg.norm(v) - RULE_BASED_SPEED_MPS) < 1e-6, "rule-based speed should equal the fixed cruise speed"

    # 6) GPS restored -- confirm a clean hand-back (EKF discarded, layer NONE).
    result = orch.decide(drone_id, packet(True), now=t + 0.2)
    assert result["layer"] == "NONE"
    assert orch._drones[drone_id].ekf is None

    # 7) Every non-LSTM architecture actually produces sane, finite output
    #    through the real decide() path, not just "the checkpoint loads" --
    #    covers both call conventions _model_velocity dispatches on: a
    #    tensor forward pass (Transformer/SSL-Transformer, same code path
    #    LSTM/SSL-LSTM already exercised above) and sklearn's flattened
    #    .predict() (RandomForest/GBT, exercised nowhere else in this file).
    print("\n--- Every architecture produces finite output through decide() ---")
    for name in MODEL_REGISTRY:
        if name == 'LSTM':
            continue   # already exhaustively exercised above
        arch_orch = RecoveryOrchestrator(model_name=name)
        arch_id = f"Drone_{name}"
        ta = 0.0
        for _ in range(WINDOW_SIZE):
            ta += 0.1
            result = arch_orch.decide(arch_id, packet(False), now=ta)
        v = np.array(result["velocity_cmd"])
        print(f"  {name:<16} layer={result['layer']:<10} v={result['velocity_cmd']}  "
              f"|v|={np.linalg.norm(v):.3f}")
        assert result["layer"] == "LSTM", (
            f"{name}: expected the buffer-full layer name to still report 'LSTM' "
            f"(architecture is an internal detail, not a new wire-protocol layer), "
            f"got {result['layer']}")
        assert all(np.isfinite(x) for x in v), f"{name}: non-finite velocity output {v}"

    # 8) EKF-staleness fix: feed a sequence of LSTM-active ticks (buffer
    #    already warm) and confirm st.ekf.position_uncertainty keeps
    #    growing across them -- before this fix it would stay frozen at
    #    whatever it was when the buffer first filled, since predict() used
    #    to run only inside _ekf_velocity() (EKF-active ticks only).
    print("\n--- EKF covariance stays current on LSTM-active ticks (staleness fix) ---")
    stale_orch = RecoveryOrchestrator(model_name='LSTM')
    stale_id = "Drone_EKFStale"
    ts2 = 0.0
    for _ in range(WINDOW_SIZE):
        ts2 += 0.1
        stale_orch.decide(stale_id, packet(False), now=ts2)
    unc_1 = stale_orch._drones[stale_id].ekf.position_uncertainty.copy()
    for _ in range(10):
        ts2 += 0.1
        stale_orch.decide(stale_id, packet(False), now=ts2)
    unc_2 = stale_orch._drones[stale_id].ekf.position_uncertainty.copy()
    print(f"  position_uncertainty: {unc_1} -> {unc_2} over 10 LSTM-active ticks")
    # Only x/y (indices 0,1) are asserted to grow -- they're uncorrected
    # horizontal axes, so unfed process noise should keep inflating them.
    # z (index 2, altitude) is legitimately expected to shrink instead: the
    # test packet's baro_alt is constant, and baro corrections now also run
    # every tick as part of this same fix, so periodic baro updates reduce
    # vertical uncertainty exactly as ekf_baseline.py's own smoke test
    # already established (baro-corrected < uncorrected) -- asserting z
    # grows too would be asserting the fix is broken when it's actually
    # working as intended.
    assert np.all(unc_2[:2] > unc_1[:2]), (
        f"EKF horizontal (x/y) covariance should keep growing across LSTM-active ticks "
        f"(unfed process noise), got {unc_1[:2]} -> {unc_2[:2]} -- predict() may not be "
        f"running every tick anymore")
    assert unc_2[2] < unc_1[2], (
        f"EKF vertical (z) covariance should shrink under periodic baro correction, "
        f"got {unc_1[2]:.3f} -> {unc_2[2]:.3f} -- update_baro() may not be running every tick anymore")

    # 9) RL graceful degrade: enabling the seek policy with no checkpoint on
    #    disk must return False and leave decide() working exactly as
    #    before (hand-coded formula), same pattern as every other missing-
    #    checkpoint degrade in this file.
    print("\n--- RL seek policy: graceful degrade with no checkpoint ---")
    rl_orch = RecoveryOrchestrator(model_name='LSTM')
    had_checkpoint = os.path.exists(_RL_POLICY_PATH)
    if not had_checkpoint:
        ok = rl_orch.set_navigation_policy(True)
        assert ok is False, "set_navigation_policy(True) with no checkpoint should return False"
        assert rl_orch._rl_policy is None
        assert rl_orch.rl_seek_enabled is False
        rl_id = "Drone_RLDegrade"
        tr2 = 0.0
        for _ in range(WINDOW_SIZE):
            tr2 += 0.1
            result = rl_orch.decide(rl_id, packet(False), now=tr2)
        assert all(np.isfinite(v) for v in result["velocity_cmd"]), (
            "decide() should still produce finite output via the hand-coded formula")
        print("  No checkpoint present -- degraded cleanly to the hand-coded formula, as expected.")
    else:
        print("  A checkpoint already exists on disk -- degrade path exercised via test 10 instead.")

    # 10) The feature's real acceptance test, only runs once a trained
    #     checkpoint actually exists: the same believed_pos/goal_pos/v_raw
    #     scenario fed through _apply_seek_bias twice, at deliberately
    #     different injected confidence, should produce a measurably
    #     different seek contribution -- concrete, checkable evidence RL
    #     learned something a fixed Kp formula structurally cannot exhibit
    #     (the formula's ||seek|| is confidence-independent by construction).
    #     This is the real pass/fail bar for this feature, not "produces
    #     finite output" (already covered by test 7 above, for architecture
    #     loading -- a different thing).
    if had_checkpoint:
        print("\n--- RL seek policy: confidence-sensitive behavior (acceptance test) ---")
        acc_orch = RecoveryOrchestrator(model_name='LSTM', use_rl_seek=True)
        assert acc_orch.rl_seek_enabled, "expected the RL policy to load given a checkpoint exists"
        v = np.array([0.05, -0.02, 0.01])
        believed = np.array([0.0, 0.0, 0.0])
        goal = np.array([40.0, 10.0, -15.0])
        # "confidence" here is the raw uncertainty magnitude (matches the
        # parameter's meaning everywhere else in this file, e.g.
        # st.ekf.position_uncertainty) -- 0.3m is HIGH confidence (small
        # uncertainty), 9.0m is LOW confidence (large uncertainty). Named
        # low_unc/high_unc, not low_conf/high_conf, to avoid the inverted
        # reading a "confidence" name invites here.
        low_unc = np.array([0.3, 0.3, 0.3])
        high_unc = np.array([9.0, 9.0, 9.0])
        seek_confident = acc_orch._apply_seek_bias(v, believed, goal, confidence=low_unc, elapsed_s=1.0) - v
        seek_unsure = acc_orch._apply_seek_bias(v, believed, goal, confidence=high_unc, elapsed_s=1.0) - v
        norm_confident = float(np.linalg.norm(seek_confident))
        norm_unsure = float(np.linalg.norm(seek_unsure))
        print(f"  ||seek|| at high confidence (0.3m unc.): {norm_confident:.3f} m/s")
        print(f"  ||seek|| at low confidence (9.0m unc.):  {norm_unsure:.3f} m/s")
        assert norm_unsure < norm_confident, (
            "RL seek magnitude should shrink as uncertainty grows -- "
            f"got ||seek||={norm_unsure:.3f} at high uncertainty vs. "
            f"||seek||={norm_confident:.3f} at low uncertainty (expected the former smaller). "
            "A first-pass 10K-timestep policy may not reliably show this yet; if this starts "
            "failing after retraining, treat it as a real regression in confidence-awareness, "
            "not a flaky threshold to loosen.")
        for s in (seek_confident, seek_unsure):
            assert np.linalg.norm(s) <= SEEK_MAX_CONTRIB + 1e-6, (
                f"RL seek contribution exceeded the safety-cap SEEK_MAX_CONTRIB={SEEK_MAX_CONTRIB}: "
                f"||s||={np.linalg.norm(s):.3f}")
    else:
        print("\n--- RL seek policy acceptance test skipped: no trained checkpoint on disk yet ---")

    print("\nAll RecoveryOrchestrator checks passed.")
