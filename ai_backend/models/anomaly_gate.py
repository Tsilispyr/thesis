"""Two independent, unsupervised anomaly detectors for IMU windows, run
side by side (Track A, Phase A4 of the coursework-grounded extension
roadmap): IsolationForest and a PCA-reconstruction-distance threshold.

Ported pattern: the Machine Learning course's Netflix content-recommender
notebook (D:\\ΠΜΣ\\machine learning\\εργασία εξαμήνου machine learning\\
ais2118_Exercise_2_Machine_Learning_Project_.ipynb, cell 51) fits
sklearn.ensemble.IsolationForest directly on a TF-IDF feature matrix
(contamination=0.01), and separately flags rows above the 99th-percentile
PCA-reconstruction distance, two independent unsupervised detectors
compared, not one method presented alone. Applied here to flattened,
scaled IMU windows instead of TF-IDF vectors.

This is a genuinely different signal from the EKF's covariance or the
uncertainty-LSTM's NLL head (dead_reckoning_model_uncertainty.py): those
quantify motion uncertainty (how much does the *position* estimate drift),
this flags *input* pathology (does this window's sensor reading look like
anything the model has seen), independent of whether a delta-position
prediction was even attempted.

Offline analysis only in this phase (see anomaly_gate.py's role in
models/ANALYSIS.md): flagging correlates telemetry windows with high
chained-trajectory error, live wiring into recovery_orchestrator.py's
layer selection is an explicit stretch goal, not part of this phase.

Usage:
  python ai_backend\\models\\anomaly_gate.py --sources imu
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.dataset_parser import load_combined_dataset, create_dead_reckoning_dataset

WINDOW_SIZE = 10


def _flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


class IsolationForestGate:
    """contamination=0.01 (flag the most-anomalous ~1% of windows), same
    setting as the coursework reference. n_jobs=-1 since IsolationForest
    parallelizes cleanly across its tree ensemble."""

    def __init__(self, contamination: float = 0.01, random_state: int = 0):
        self.model = IsolationForest(contamination=contamination,
                                      random_state=random_state, n_jobs=-1)

    def fit(self, X_flat: np.ndarray) -> 'IsolationForestGate':
        self.model.fit(X_flat)
        return self

    def flag(self, X_flat: np.ndarray) -> np.ndarray:
        """True where anomalous (IsolationForest's predict() returns -1 for
        outliers, 1 for inliers, inverted here so True consistently means
        'flagged' across both gates in this module)."""
        return self.model.predict(X_flat) == -1

    def score(self, X_flat: np.ndarray) -> np.ndarray:
        """Continuous anomaly score, higher = more anomalous (decision_function
        is higher for inliers by sklearn's convention, negated here so both
        gates in this module share the same "higher = more anomalous" sign)."""
        return -self.model.decision_function(X_flat)


class PCAReconstructionGate:
    """Fits PCA on the flattened windows, flags the top percentile by
    reconstruction error (||x - inverse_transform(transform(x))||) as
    anomalous, the second, independent detector from the coursework
    reference's dual-detector pattern."""

    def __init__(self, n_components: int = 20, percentile: float = 99.0):
        self.pca = PCA(n_components=n_components)
        self.percentile = percentile
        self.threshold_ = None

    def fit(self, X_flat: np.ndarray) -> 'PCAReconstructionGate':
        self.pca.fit(X_flat)
        errors = self._reconstruction_error(X_flat)
        self.threshold_ = float(np.percentile(errors, self.percentile))
        return self

    def _reconstruction_error(self, X_flat: np.ndarray) -> np.ndarray:
        recon = self.pca.inverse_transform(self.pca.transform(X_flat))
        return np.linalg.norm(X_flat - recon, axis=1)

    def flag(self, X_flat: np.ndarray) -> np.ndarray:
        return self._reconstruction_error(X_flat) > self.threshold_

    def score(self, X_flat: np.ndarray) -> np.ndarray:
        return self._reconstruction_error(X_flat)


def compare_gates(iso_flags: np.ndarray, pca_flags: np.ndarray) -> dict:
    """Do the two independent detectors agree on which windows are
    anomalous, or are they catching different failure modes? Jaccard index
    (intersection over union) plus the raw overlap counts."""
    iso_flags, pca_flags = np.asarray(iso_flags), np.asarray(pca_flags)
    both = int(np.sum(iso_flags & pca_flags))
    either = int(np.sum(iso_flags | pca_flags))
    only_iso = int(np.sum(iso_flags & ~pca_flags))
    only_pca = int(np.sum(pca_flags & ~iso_flags))
    jaccard = both / either if either > 0 else 0.0
    return {
        'n_iso_flagged': int(iso_flags.sum()),
        'n_pca_flagged': int(pca_flags.sum()),
        'n_both': both,
        'n_only_iso': only_iso,
        'n_only_pca': only_pca,
        'jaccard_index': jaccard,
    }


def run(sources=('imu',), out_path: str = None) -> dict:
    print(f"\n[AnomalyGate] Fitting on sources={list(sources)}")
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets')
    df = load_combined_dataset(base_dir, sources=sources)
    X_train, _y_train, X_val, _y_val, _scaler_X, _scaler_y = create_dead_reckoning_dataset(
        df, window_size=WINDOW_SIZE, normalize=True)
    X_train_flat, X_val_flat = _flatten(X_train), _flatten(X_val)

    iso_gate = IsolationForestGate().fit(X_train_flat)
    pca_gate = PCAReconstructionGate().fit(X_train_flat)

    iso_val_flags = iso_gate.flag(X_val_flat)
    pca_val_flags = pca_gate.flag(X_val_flat)
    agreement = compare_gates(iso_val_flags, pca_val_flags)

    print(f"[AnomalyGate] Val set ({len(X_val_flat)} windows): "
          f"IsolationForest flagged {agreement['n_iso_flagged']}, "
          f"PCA-reconstruction flagged {agreement['n_pca_flagged']}, "
          f"agree on {agreement['n_both']} (Jaccard={agreement['jaccard_index']:.3f})")

    results = {'sources': list(sources), 'agreement': agreement}
    if out_path is None:
        out_path = os.path.join(os.path.dirname(__file__), '..', '..', 'runs',
                                 f'anomaly_gate_{"_".join(sources)}.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[AnomalyGate] Saved -> {out_path}")

    return iso_gate, pca_gate, results


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', nargs='+', default=['imu'], choices=['nav', 'imu', 'px4'])
    args = ap.parse_args()
    run(sources=tuple(args.sources))
