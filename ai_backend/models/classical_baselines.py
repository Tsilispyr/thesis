"""Classical (non-deep-learning) ML baselines for the dead-reckoning task:
RandomForestRegressor and gradient-boosted trees, trained on the same
flattened, scaled windows as every deep model in the ablation matrix (Track
A, Phase A4 of the coursework-grounded extension roadmap).

Ported pattern: the KNIME workflow at
D:\\ΠΜΣ\\Data Mining and Recommender Systems\\...\\data mine recom syst.knwf
runs a full classical-classifier bench (Decision Tree, Random Forest, GBT,
SVM, Naive Bayes, MLP) inside a grid-search loop. This module is the same
idea, scoped to the two tree-ensemble methods most directly comparable to
this task's regression target, with a GridSearchCV hyperparameter sweep.

Why HistGradientBoostingRegressor rather than plain GradientBoostingRegressor
(the classical, single-threaded, exact-split sklearn implementation): the
training set here is ~436K rows x 140 flattened features, well beyond the
scale plain GradientBoostingRegressor is practical at (single-threaded,
grows trees on the exact, unbinned data). HistGradientBoostingRegressor is
sklearn's histogram-based GBM (LightGBM's algorithm family), the standard
modern choice at this data scale, still a genuine gradient-boosted-trees
baseline, just the efficient variant. It's single-output, so it's wrapped
in MultiOutputRegressor to predict all 3 position-delta components (fits 3
independent boosters internally, same approach scikit-learn recommends for
any single-output-only regressor on a multi-output target).

Usage:
  python ai_backend\\models\\classical_baselines.py --sources imu
"""
import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.dataset_parser import load_combined_dataset, create_dead_reckoning_dataset

WINDOW_SIZE = 10
GRID_SEARCH_SUBSAMPLE = 20_000   # rows used for the hyperparameter search itself, not final fit


def _flatten(X: np.ndarray) -> np.ndarray:
    """(N, window, features) -> (N, window*features), the standard way to
    hand a sequence window to a non-sequential model (each timestep's 14
    features become 14 more columns, order preserved)."""
    n = X.shape[0]
    return X.reshape(n, -1)


def train_random_forest(sources=('imu',), n_estimators=100, max_depth=20,
                         grid_search=True, random_state=0, n_jobs=-1):
    """Trains a RandomForestRegressor (native multi-output support, no
    wrapper needed) on the same scaled train/val split as every deep model
    in this project. grid_search=True sweeps n_estimators x max_depth on a
    GRID_SEARCH_SUBSAMPLE-row subsample (RF on the full ~436K x 140 table
    per grid point would be prohibitively slow), then refits the best
    config on the full training set.

    n_jobs defaults to -1 (all cores), but is a real parameter (not
    hardcoded) so two seeds can be trained in separate concurrent processes
    without both trying to claim every core -- see ablation_matrix.py's
    --n-jobs flag."""
    print(f"\n[RandomForest] Training  |  sources={list(sources)}  "
          f"grid_search={grid_search}  n_jobs={n_jobs}")

    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets')
    df = load_combined_dataset(base_dir, sources=sources)
    X_train, y_train, X_val, y_val, scaler_X, scaler_y = create_dead_reckoning_dataset(
        df, window_size=WINDOW_SIZE, normalize=True)
    X_train_flat, X_val_flat = _flatten(X_train), _flatten(X_val)

    t0 = time.time()
    if grid_search:
        rng = np.random.default_rng(random_state)
        sub_idx = rng.choice(len(X_train_flat), size=min(GRID_SEARCH_SUBSAMPLE, len(X_train_flat)),
                              replace=False)
        param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
        search = GridSearchCV(
            RandomForestRegressor(random_state=random_state, n_jobs=n_jobs),
            param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=n_jobs)
        search.fit(X_train_flat[sub_idx], y_train[sub_idx])
        best_params = search.best_params_
        print(f"[RandomForest] GridSearchCV (on {len(sub_idx)}-row subsample) "
              f"best params: {best_params}")
    else:
        best_params = {'n_estimators': n_estimators, 'max_depth': max_depth}

    model = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs, **best_params)
    model.fit(X_train_flat, y_train)
    fit_elapsed = time.time() - t0

    val_pred = model.predict(X_val_flat)
    val_mse = float(np.mean((val_pred - y_val) ** 2))
    print(f"[RandomForest] Fit in {fit_elapsed:.1f}s (full {len(X_train_flat)} rows). "
          f"Val MSE: {val_mse:.4f}")

    return model, scaler_X, scaler_y, {
        'best_params': best_params, 'val_mse': val_mse, 'fit_elapsed_s': fit_elapsed,
    }


def train_gbt(sources=('imu',), max_iter=100, max_depth=None,
              grid_search=True, random_state=0, n_jobs=-1):
    """Trains a MultiOutputRegressor(HistGradientBoostingRegressor(...)) on
    the same pipeline as train_random_forest. See module docstring for why
    HistGradientBoostingRegressor rather than plain GradientBoostingRegressor.

    n_jobs controls the GridSearchCV and MultiOutputRegressor parallelism
    (same reasoning as train_random_forest's n_jobs). HistGradientBoosting
    Regressor itself has no n_jobs parameter -- it parallelizes internally
    via OpenMP, controllable only via the OMP_NUM_THREADS environment
    variable if that also needs capping for concurrent processes."""
    print(f"\n[GBT] Training  |  sources={list(sources)}  grid_search={grid_search}  "
          f"n_jobs={n_jobs}")

    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets')
    df = load_combined_dataset(base_dir, sources=sources)
    X_train, y_train, X_val, y_val, scaler_X, scaler_y = create_dead_reckoning_dataset(
        df, window_size=WINDOW_SIZE, normalize=True)
    X_train_flat, X_val_flat = _flatten(X_train), _flatten(X_val)

    t0 = time.time()
    if grid_search:
        rng = np.random.default_rng(random_state)
        sub_idx = rng.choice(len(X_train_flat), size=min(GRID_SEARCH_SUBSAMPLE, len(X_train_flat)),
                              replace=False)
        # GridSearchCV can't directly grid-search params of the estimator
        # wrapped inside MultiOutputRegressor without the estimator__ prefix,
        # search a single output (delta_lat) as a fast proxy, then apply the
        # winning config to all 3 outputs via MultiOutputRegressor. The
        # tree structure that's best for one position-delta axis is a
        # reasonable, cheap proxy for the other two, all three come from the
        # same sensor window and share the same underlying dynamics.
        param_grid = {'max_iter': [50, 100], 'max_depth': [None, 10]}
        search = GridSearchCV(
            HistGradientBoostingRegressor(random_state=random_state),
            param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=n_jobs)
        search.fit(X_train_flat[sub_idx], y_train[sub_idx, 0])
        best_params = search.best_params_
        print(f"[GBT] GridSearchCV (on {len(sub_idx)}-row subsample, delta_lat proxy) "
              f"best params: {best_params}")
    else:
        best_params = {'max_iter': max_iter, 'max_depth': max_depth}

    base_estimator = HistGradientBoostingRegressor(random_state=random_state, **best_params)
    model = MultiOutputRegressor(base_estimator, n_jobs=n_jobs)
    model.fit(X_train_flat, y_train)
    fit_elapsed = time.time() - t0

    val_pred = model.predict(X_val_flat)
    val_mse = float(np.mean((val_pred - y_val) ** 2))
    print(f"[GBT] Fit in {fit_elapsed:.1f}s (full {len(X_train_flat)} rows). "
          f"Val MSE: {val_mse:.4f}")

    return model, scaler_X, scaler_y, {
        'best_params': best_params, 'val_mse': val_mse, 'fit_elapsed_s': fit_elapsed,
    }


def save_random_forest_artifacts(model, scaler_X, scaler_y, info: dict, sources,
                                  tag_suffix: str = '') -> str:
    """Persists the RandomForest model/scalers/info, tagged by sources (+
    optional tag_suffix, same convention as train.py::train_model). Split
    from GBT's save (rather than the combined history_classical_{tag}.json
    __main__ below writes) because ablation_matrix.py trains and saves these
    two models at different times, not together -- previously it called
    train_random_forest() directly and discarded the returned info, so
    nothing reached disk for this arm during the ablation run."""
    tag = "_".join(sources) + "_norm" + tag_suffix
    out_dir = os.path.dirname(os.path.abspath(__file__))
    joblib.dump(model, os.path.join(out_dir, f'rf_{tag}.pkl'))
    joblib.dump(scaler_X, os.path.join(out_dir, f'scaler_X_rf_{tag}.pkl'))
    joblib.dump(scaler_y, os.path.join(out_dir, f'scaler_y_rf_{tag}.pkl'))
    with open(os.path.join(out_dir, f'history_rf_{tag}.json'), 'w') as f:
        json.dump(info, f, indent=2)
    print(f"[RandomForest] Saved model + scalers + info with tag '{tag}' -> {out_dir}")
    return tag


def save_gbt_artifacts(model, scaler_X, scaler_y, info: dict, sources,
                        tag_suffix: str = '') -> str:
    """Same as save_random_forest_artifacts, for the GBT arm."""
    tag = "_".join(sources) + "_norm" + tag_suffix
    out_dir = os.path.dirname(os.path.abspath(__file__))
    joblib.dump(model, os.path.join(out_dir, f'gbt_{tag}.pkl'))
    joblib.dump(scaler_X, os.path.join(out_dir, f'scaler_X_gbt_{tag}.pkl'))
    joblib.dump(scaler_y, os.path.join(out_dir, f'scaler_y_gbt_{tag}.pkl'))
    with open(os.path.join(out_dir, f'history_gbt_{tag}.json'), 'w') as f:
        json.dump(info, f, indent=2)
    print(f"[GBT] Saved model + scalers + info with tag '{tag}' -> {out_dir}")
    return tag


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', nargs='+', default=['imu'], choices=['nav', 'imu', 'px4'])
    ap.add_argument('--no-grid-search', action='store_true',
                     help='Skip GridSearchCV, use fixed default hyperparameters.')
    args = ap.parse_args()

    rf_model, rf_sx, rf_sy, rf_info = train_random_forest(
        sources=tuple(args.sources), grid_search=not args.no_grid_search)
    gbt_model, gbt_sx, gbt_sy, gbt_info = train_gbt(
        sources=tuple(args.sources), grid_search=not args.no_grid_search)

    tag = "_".join(args.sources) + "_norm"
    out_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(rf_model, os.path.join(out_dir, f'rf_{tag}.pkl'))
    joblib.dump(rf_sx, os.path.join(out_dir, f'scaler_X_rf_{tag}.pkl'))
    joblib.dump(rf_sy, os.path.join(out_dir, f'scaler_y_rf_{tag}.pkl'))
    joblib.dump(gbt_model, os.path.join(out_dir, f'gbt_{tag}.pkl'))
    joblib.dump(gbt_sx, os.path.join(out_dir, f'scaler_X_gbt_{tag}.pkl'))
    joblib.dump(gbt_sy, os.path.join(out_dir, f'scaler_y_gbt_{tag}.pkl'))

    with open(os.path.join(out_dir, f'history_classical_{tag}.json'), 'w') as f:
        json.dump({'random_forest': rf_info, 'gbt': gbt_info}, f, indent=2)

    print(f"\nRandomForest val MSE: {rf_info['val_mse']:.4f}")
    print(f"GBT val MSE:          {gbt_info['val_mse']:.4f}")
    print(f"Saved models + scalers with tag '{tag}' -> {out_dir}")
