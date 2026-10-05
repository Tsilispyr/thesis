"""Synthetic multi-drone connectivity-network scenario generator, for the
graph-classification connectivity-prediction task (SD-UAV Networks track,
Track B groundwork -- see AI_RECOVERY_EXECUTION_PLAN.md's own "swarm
connectivity analysis" note and the Graph & Network Analysis coursework's
extract_graph_features()/Node2Vec/GCN-GIN methods this reuses in
train_swarm_connectivity.py).

No multi-drone dataset exists anywhere in this project (imu_data.csv and
px4_raw are both single-drone) -- this generates one from first principles,
reusing this project's own already-validated drift model rather than
inventing a new one.

The prediction task, designed to be a genuine (non-deterministic-from-input)
ML problem rather than a redundant graph computation:

  INPUT graph (what the system currently believes): N drones + 1 fixed base
  station. Each drone reports its own BELIEVED position (its own dead-
  reckoning estimate, corrupted by accumulated GPS-loss drift) and its
  current velocity. Edges = pairs within radio range of each other, using
  BELIEVED positions (the only thing actually observable to the system).

  LABEL (what will actually happen): each drone's TRUE position (unknown to
  the system) evolves under its own real velocity for K more seconds; radio
  connectivity is a physical fact, so the future graph is built from TRUE
  future positions. Label = 1 ("fragments") if any drone loses every relay
  path back to the base station, else 0.

This makes believed-vs-true divergence (driven by each drone's own
accumulated GPS-loss uncertainty) the thing a classifier has to learn to
read the risk of, exactly this project's own recurring GPS-loss/drift theme,
not an arbitrary synthetic add-on. Edge criterion (in-range) mirrors
api/swarm_logic.py::SwarmInterlink's own relay-connectivity rule, adapted
from lat/lon+haversine to local metres for consistency with every other
model in this project (all of which already work in local metres, never
GPS degrees, per dataset_parser.py's own established convention).

Usage:
  python ai_backend/data_processing/swarm_network_generator.py
"""
import os
import sys

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from models.rl_path_recovery import DRIFT_SIGMA_LOW, DRIFT_SIGMA_HIGH, RTL_TIMEOUT_S, DT

_OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'runs', 'swarm_connectivity')

# Scenario geometry -- chosen so a typical scenario has a genuine mix of
# connected/fragmented outcomes (checked directly below, not assumed): a
# search area a few multiples of the communication range, so some drone
# pairs are naturally in range and some aren't.
AREA_SIZE_M = 600.0          # square search area, metres per side
COMM_RANGE_M = 250.0         # tuned empirically (checked directly, not guessed): gives a
                              # near-balanced ~50/50 fragmentation rate over 500 scenarios,
                              # not the ~93% degenerate rate an initial 180m guess produced
N_DRONES_MIN, N_DRONES_MAX = 6, 10
K_SECONDS = 6.0              # look-ahead horizon for the fragmentation label
MAX_SPEED_MPS = 2.0          # matches recovery_orchestrator.py's RULE_BASED_SPEED_MPS

# Node feature layout (both base station and drones share this schema, base
# station's dynamic fields are zeroed -- see _node_features()):
#   [x, y, vx, vy, uncertainty_m, elapsed_loss_s_norm]
N_NODE_FEATURES = 6


def _clip_speed(v, max_speed):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    scale = np.minimum(1.0, max_speed / np.maximum(norm, 1e-9))
    return v * scale


def _adjacency_from_positions(positions, comm_range):
    n = positions.shape[0]
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    adj = (dist <= comm_range) & (dist > 0)
    np.fill_diagonal(adj, False)
    return adj.astype(np.float32)


def generate_scenario(rng, n_drones=None, area_size=AREA_SIZE_M, comm_range=COMM_RANGE_M,
                       k_seconds=K_SECONDS):
    """Returns (node_features [n_drones+1, N_NODE_FEATURES], input_adjacency
    [n_drones+1, n_drones+1], label). Node 0 is always the fixed base
    station; nodes 1..n_drones are the drones."""
    if n_drones is None:
        n_drones = int(rng.integers(N_DRONES_MIN, N_DRONES_MAX + 1))

    base_pos = np.array([area_size / 2.0, area_size / 2.0])

    true_pos_now = rng.uniform(0.0, area_size, size=(n_drones, 2))
    velocity = _clip_speed(rng.normal(0.0, 1.0, size=(n_drones, 2)), MAX_SPEED_MPS)

    # Domain-randomized per-drone estimator quality and elapsed GPS-loss
    # time, same log-uniform range this project already validated for the
    # single-drone RL environment (models/rl_path_recovery.py) -- some
    # drones behave LSTM-like, some EKF-like, some have only just lost GPS,
    # some are near the real system's own RTL_TIMEOUT_S give-up boundary.
    drift_sigma = np.exp(rng.uniform(np.log(DRIFT_SIGMA_LOW), np.log(DRIFT_SIGMA_HIGH), size=n_drones))
    elapsed_loss = rng.uniform(0.0, RTL_TIMEOUT_S, size=n_drones)
    uncertainty = drift_sigma * np.sqrt(np.maximum(elapsed_loss / DT, 1.0))
    accumulated_error = rng.normal(0.0, uncertainty[:, None], size=(n_drones, 2))
    believed_pos_now = true_pos_now + accumulated_error

    # --- Input graph: base station + drones, edges from BELIEVED positions ---
    all_believed = np.vstack([base_pos[None, :], believed_pos_now])
    input_adjacency = _adjacency_from_positions(all_believed, comm_range)

    node_features = np.zeros((n_drones + 1, N_NODE_FEATURES), dtype=np.float32)
    node_features[0] = [base_pos[0], base_pos[1], 0.0, 0.0, 0.0, 0.0]
    node_features[1:, 0:2] = believed_pos_now
    node_features[1:, 2:4] = velocity
    node_features[1:, 4] = uncertainty
    node_features[1:, 5] = elapsed_loss / RTL_TIMEOUT_S

    # --- Label: TRUE future positions after k_seconds, physical connectivity ---
    heading_jitter = rng.normal(0.0, 0.15, size=(n_drones, 2))  # small realistic heading drift
    true_pos_future = true_pos_now + (velocity + heading_jitter) * k_seconds
    all_true_future = np.vstack([base_pos[None, :], true_pos_future])
    future_adjacency = _adjacency_from_positions(all_true_future, comm_range)

    G_future = nx.from_numpy_array(future_adjacency)
    reachable_from_base = nx.node_connected_component(G_future, 0)
    fragments = len(reachable_from_base) < (n_drones + 1)
    label = int(fragments)

    return node_features, input_adjacency, label


def generate_dataset(n_scenarios, seed):
    rng = np.random.default_rng(seed)
    scenarios = []
    labels = []
    for _ in range(n_scenarios):
        feats, adj, label = generate_scenario(rng)
        scenarios.append((feats, adj))
        labels.append(label)
    labels = np.array(labels)
    return scenarios, labels


if __name__ == '__main__':
    os.makedirs(_OUT_DIR, exist_ok=True)

    N_TRAIN, N_VAL, N_TEST = 700, 150, 150
    splits = {}
    for name, n, seed in [('train', N_TRAIN, 0), ('val', N_VAL, 1), ('test', N_TEST, 2)]:
        scenarios, labels = generate_dataset(n, seed=seed)
        splits[name] = (scenarios, labels)
        frac_frag = labels.mean()
        print(f"{name:>5}: {n} scenarios, fragmentation rate = {frac_frag:.3f} "
              f"({int(labels.sum())} fragment / {int((1 - labels).sum())} stay connected)")
        # Sanity check before trusting this dataset for anything downstream --
        # same discipline as train_motor_informed.py's own non-degeneracy
        # assertion: a task where every label is identical isn't learnable
        # or meaningful to report a "result" on.
        assert 0.10 < frac_frag < 0.90, (
            f"{name} split is degenerate (fragmentation rate {frac_frag:.3f}) -- "
            f"adjust AREA_SIZE_M/COMM_RANGE_M/K_SECONDS before trusting this dataset.")

    import pickle
    out_path = os.path.join(_OUT_DIR, 'swarm_scenarios.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(splits, f)
    print(f"Saved -> {out_path}")
