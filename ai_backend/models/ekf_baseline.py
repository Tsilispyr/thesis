"""Classical Extended Kalman Filter dead-reckoning baseline (Goal 1 item 4,
also Goal 3's tertiary fail-safe layer - AI_RECOVERY_EXECUTION_PLAN.md §12).

A real, reusable estimator -- not a strawman comparison script -- so it can
serve two roles at once: (1) the non-ML baseline in Goal 2's 5-model
ablation matrix (Pure Inertial < EKF < Supervised LSTM < SSL-LSTM <
SSL-Transformer), and (2) the tertiary layer of Goal 3's degrade-toward-
simplicity fail-safe hierarchy, sitting between the SSL/LSTM estimator and
the zero-AI kinematic breadcrumb.
"""
import numpy as np


class DeadReckoningEKF:
    """State x = [px, py, pz, vx, vy, vz] (world frame, metres / m/s).

    Process model: strapdown double-integration of body-frame specific-force
    acceleration, rotated into world frame via the current attitude estimate
    -- the same specific-force convention already used for the simulated
    accelerometer (see drone_body.gd::_physics_process, "a_body = R^-1 *
    ((v_t - v_{t-1})/dt - g)"; here we run that relationship forward instead
    of backward). No learned weights anywhere -- purely classical.

    Measurement model: barometric altitude, when available, corrects pz (and
    via the cross-covariance terms, vz too) -- the standard aviation-grade
    fusion that keeps vertical position from drifting unboundedly under pure
    inertial integration alone. This is not an invented pattern: it's what
    PX4's own EKF2 does, confirmed directly by inspecting the PX4 logs
    downloaded for this project (AI_RECOVERY_EXECUTION_PLAN.md §11.2) --
    `vehicle_local_position.z` there is exactly this kind of baro-corrected
    estimate.

    Exposes the propagated covariance matrix alongside the point estimate --
    Goal 1 item 5's uncertainty requirement, which a Kalman filter provides
    natively (unlike the LSTM/Transformer, which need an explicit second
    output head trained with Gaussian NLL loss to get the same thing).
    """

    STATE_DIM = 6   # [px, py, pz, vx, vy, vz]

    def __init__(self, process_noise_std=0.5, baro_noise_std=0.3,
                 initial_pos_std=1.0, initial_vel_std=1.0):
        self.x = np.zeros(self.STATE_DIM)
        self.P = np.diag([initial_pos_std ** 2] * 3 + [initial_vel_std ** 2] * 3)
        self.q_accel_std = process_noise_std   # accelerometer process-noise density (m/s^2 / sqrt(s))
        self.R_baro = np.array([[baro_noise_std ** 2]])

    def reset(self, position=None, velocity=None):
        self.x = np.zeros(self.STATE_DIM)
        if position is not None:
            self.x[0:3] = position
        if velocity is not None:
            self.x[3:6] = velocity
        self.P = np.diag([1.0] * 3 + [1.0] * 3)

    @property
    def position(self) -> np.ndarray:
        return self.x[0:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:6].copy()

    @property
    def position_covariance(self) -> np.ndarray:
        return self.P[0:3, 0:3].copy()

    @property
    def position_uncertainty(self) -> np.ndarray:
        """1-sigma position uncertainty per axis (m) -- Goal 1 item 5's free
        covariance output; Goal 3's health-check ("confidence below
        threshold") can be built directly on top of this."""
        return np.sqrt(np.clip(np.diag(self.position_covariance), 0.0, None))

    def predict(self, accel_body: np.ndarray, rotation_matrix: np.ndarray, dt: float) -> None:
        """Propagate state forward using body-frame specific-force
        acceleration rotated into world frame via rotation_matrix (3x3,
        world_from_body -- e.g. Godot's global_transform.basis, or
        rotation_from_euler_deg() below)."""
        if dt <= 0.0:
            return
        accel_world = rotation_matrix @ np.asarray(accel_body, dtype=float)

        F = np.eye(self.STATE_DIM)
        F[0:3, 3:6] = np.eye(3) * dt

        B = np.zeros((self.STATE_DIM, 3))
        B[0:3] = 0.5 * dt * dt * np.eye(3)
        B[3:6] = dt * np.eye(3)

        self.x = F @ self.x + B @ accel_world

        # Discretized process noise for a piecewise-white-noise-acceleration
        # model (standard Kalman-filter INS derivation, e.g. Bar-Shalom
        # "Estimation with Applications to Tracking and Navigation" ch.6).
        q = self.q_accel_std ** 2
        dt2, dt3, dt4 = dt ** 2, dt ** 3, dt ** 4
        Q = np.zeros((self.STATE_DIM, self.STATE_DIM))
        for i in range(3):
            pi, vi = i, i + 3
            Q[pi, pi] = q * dt4 / 4.0
            Q[pi, vi] = Q[vi, pi] = q * dt3 / 2.0
            Q[vi, vi] = q * dt2
        self.P = F @ self.P @ F.T + Q

    def update_baro(self, baro_altitude: float) -> None:
        """Measurement update: absolute altitude (world-frame pz). Standard
        linear Kalman correction (H is linear here, so this is the "E" of
        EKF only in the predict step, which is nonlinear through the
        rotation matrix; the baro update itself is exactly linear)."""
        H = np.zeros((1, self.STATE_DIM))
        H[0, 2] = 1.0
        z = np.array([baro_altitude])
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R_baro
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y).flatten()
        self.P = (np.eye(self.STATE_DIM) - K @ H) @ self.P


def rotation_from_euler_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """World_from_body rotation matrix from roll/pitch/yaw in degrees --
    matches the roll/pitch/yaw convention already used throughout
    dataset_parser.FEATURE_COLS and drone_body.gd (ZYX intrinsic Euler,
    i.e. yaw then pitch then roll)."""
    r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


if __name__ == '__main__':
    # Smoke test: demonstrates the two behaviors that motivate using an EKF
    # over pure inertial integration at all (Goal 2 Category B's "why
    # sensor fusion is necessary" figure) --
    #   (a) uncertainty grows unboundedly under pure prediction alone, and
    #   (b) a periodic baro correction bounds vertical uncertainty/drift.
    # Synthetic: level, stationary drone (zero specific-force accel other
    # than the already-subtracted gravity), 10 Hz, noisy baro every step.
    rng = np.random.default_rng(0)
    dt = 0.1
    n_steps = 50

    ekf_no_baro = DeadReckoningEKF()
    ekf_with_baro = DeadReckoningEKF()
    R = rotation_from_euler_deg(0.0, 0.0, 0.0)
    true_alt = 10.0

    for step in range(n_steps):
        accel_body = np.array([0.0, 0.0, 0.0]) + rng.normal(0, 0.05, 3)
        ekf_no_baro.predict(accel_body, R, dt)
        ekf_with_baro.predict(accel_body, R, dt)
        if step % 5 == 0:   # baro update every 0.5s
            noisy_baro = true_alt + rng.normal(0, 0.3)
            ekf_with_baro.update_baro(noisy_baro)

    print("Pure-inertial-only (no baro correction):")
    print(f"  final position   = {ekf_no_baro.position}")
    print(f"  1-sigma pos unc.  = {ekf_no_baro.position_uncertainty}")
    print("EKF with periodic baro correction:")
    print(f"  final position   = {ekf_with_baro.position}")
    print(f"  1-sigma pos unc.  = {ekf_with_baro.position_uncertainty}")

    assert ekf_with_baro.position_uncertainty[2] < ekf_no_baro.position_uncertainty[2], \
        "baro correction should reduce vertical uncertainty relative to pure inertial integration"
    print("\nOK: baro-corrected vertical uncertainty is lower than pure-inertial, as expected.")
