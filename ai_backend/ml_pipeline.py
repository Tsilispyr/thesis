"""
SD-UAV AI Recovery - Full ML Pipeline
======================================
Εκτελεί ολόκληρη τη ροή ML του project σε ένα script:

  STEP 1  Dataset Audit        - φόρτωση, shapes, NaN, στατιστικά
  STEP 2  Data Cleaning        - ffill, fillna, column remapping
  STEP 3  Normalization        - StandardScaler before/after σύγκριση
  STEP 4  Feature Engineering  - features, targets, correlation matrix
  STEP 5  Sequence Building    - sliding-window visualization
  STEP 6  Run A - nav only, no normalization   (baseline)
  STEP 7  Run B - nav only, normalized
  STEP 8  Run C - imu only, normalized
  STEP 9  Run D - nav + imu combined, normalized  (best)
  STEP 10 All-runs comparison
  STEP 11 RL Path Recovery evaluation (Rule-Based vs PPO)
  STEP 12 Final summary  ->  runs/results.json  +  summary figure

Usage (from AI_Recovery folder):
  python ai_backend/ml_pipeline.py

Outputs:
  runs/results.json         - all numerical results
  runs/figures/*.png        - one figure per step
  runs/models/*.pth         - trained model per run
"""

# --------------------------------------------------------
#  CONFIGURATION  (edit here to change experiment parameters)
# --------------------------------------------------------

EPOCHS       = 10        # αλλάζει για περισσότερα/λιγότερα epochs
BATCH_SIZE   = 64        # αλλάζει για μεγαλύτερα/μικρότερα batches
WINDOW_SIZE  = 10        # αλλάζει για μεγαλύτερο ιστορικό (π.χ. 50 = 5 sec)
IMU_MAX_ROWS = None      # None = όλα τα 544K | αριθμός π.χ. 60_000 για γρήγορο test
SEED         = 42

# --------------------------------------------------------
#  IMPORTS
# --------------------------------------------------------

import os, sys, json, time, warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless - no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


class _Tee:
    """Write simultaneously to multiple file objects (stdout + log file)."""
    def __init__(self, *files): self.files = files
    def write(self, d):
        for f in self.files: f.write(d)
    def flush(self):
        for f in self.files: f.flush()
    def isatty(self): return False

warnings.filterwarnings("ignore")
np.random.seed(SEED)
torch.manual_seed(SEED)

#  Path setup 
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
RUNS_DIR    = os.path.join(BASE_DIR, "runs")
FIG_DIR     = os.path.join(RUNS_DIR, "figures")
MDL_DIR     = os.path.join(RUNS_DIR, "models")

for d in (RUNS_DIR, FIG_DIR, MDL_DIR):
    os.makedirs(d, exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Single source of truth for the 14-feature schema and the LSTM architecture -
# both live under ai_backend/, imported here instead of re-defined.
from data_processing.dataset_parser import (FEATURE_COLS, TARGET_COLS, INPUT_SIZE,
                                             create_dead_reckoning_dataset,
                                             _M_PER_DEG_LAT, _M_PER_DEG_LON)
from models.dead_reckoning_model import DeadReckoningLSTM

IMU_RENAME = {
    "accel_x": "imu_acc_x", "accel_y": "imu_acc_y", "accel_z": "imu_acc_z",
    "gyro_x":  "imu_gyro_x","gyro_y":  "imu_gyro_y","gyro_z":  "imu_gyro_z",
    "pos_x":   "latitude",  "pos_y":   "longitude",  "pos_z":   "altitude",
    # roll, pitch, yaw, mag_x/y/z kept as-is (already canonical names)
}

results = {}   # filled throughout - written to JSON at the end

# --------------------------------------------------------
#  HELPER UTILITIES
# --------------------------------------------------------

def banner(step: int, title: str) -> None:
    sep = "═" * 70
    print(f"\n{sep}")
    print(f"  STEP {step:>2} │ {title}")
    print(f"{sep}")

def save_fig(name: str, fig=None) -> str:
    path = os.path.join(FIG_DIR, name)
    (fig or plt).savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  -> Saved figure: runs/figures/{name}")
    return path

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.spines[["top","right"]].set_visible(False)

# --------------------------------------------------------
#  STEP 1 - DATASET AUDIT
# --------------------------------------------------------

def step1_dataset_audit():
    banner(1, "Dataset Audit - shapes, columns, NaN counts")

    datasets = {
        "uav_navigation_dataset.csv": {
            "label": "nav",
            "path": os.path.join(DATASET_DIR, "uav_navigation_dataset.csv"),
        },
        "imu_data.csv": {
            "label": "imu",
            "path": os.path.join(DATASET_DIR, "imu_data.csv"),
        },
    }

    audit = {}
    for fname, meta in datasets.items():
        try:
            df = pd.read_csv(meta["path"], on_bad_lines="skip",
                             nrows=IMU_MAX_ROWS if meta["label"] == "imu" else None)
            nan_pct = (df.isnull().sum().sum() / df.size * 100)
            has_imu = any(c in df.columns for c in
                          ["imu_acc_x","accel_x","imu_acc_y","accel_y"])
            has_gps = any(c in df.columns for c in
                          ["latitude","longitude","TrackPositionLatitude","pos_x"])
            audit[meta["label"]] = {
                "rows": len(df), "cols": len(df.columns),
                "nan_pct": round(nan_pct, 2),
                "has_imu": has_imu, "has_gps": has_gps,
                "columns": list(df.columns),
            }
            print(f"  {meta['label']:15s} rows={len(df):>7,}  cols={len(df.columns):>2}"
                  f"  NaN={nan_pct:.1f}%  IMU={'✓' if has_imu else '✗'}"
                  f"  GPS={'✓' if has_gps else '✗'}")
        except Exception as e:
            print(f"  {meta['label']:15s} ERROR: {e}")
            audit[meta["label"]] = {"error": str(e)}


    results["step1_audit"] = audit

    #  Figure 1 ─
    labels   = [k for k in audit if "error" not in audit[k]]
    rows_k   = [audit[k]["rows"] / 1000 for k in labels]
    nan_vals = [audit[k]["nan_pct"] for k in labels]
    usable   = [1 if (audit[k]["has_imu"] and audit[k]["has_gps"]) else 0
                for k in labels]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("STEP 1 - Dataset Audit", fontsize=13, fontweight="bold")

    colors = ["#2196F3","#4CAF50","#FF9800","#9C27B0"]
    axes[0].bar(labels, rows_k, color=colors[:len(labels)], edgecolor="white")
    style_ax(axes[0], "Record Count (thousands)", "Dataset", "Records (K)")
    for i, v in enumerate(rows_k):
        axes[0].text(i, v + 1.5, f"{v:.1f}K", ha="center", fontsize=8)

    axes[1].bar(labels, nan_vals, color=colors[:len(labels)], edgecolor="white")
    style_ax(axes[1], "NaN Percentage", "Dataset", "% NaN")
    for i, v in enumerate(nan_vals):
        axes[1].text(i, v + 0.02, f"{v:.1f}%", ha="center", fontsize=8)

    cmap = ["#4CAF50" if u else "#F44336" for u in usable]
    axes[2].bar(labels, usable, color=cmap, edgecolor="white")
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["Not usable\nfor DR", "Usable\nfor DR"])
    style_ax(axes[2], "Usable for Dead Reckoning\n(needs both IMU & GPS)")

    plt.tight_layout()
    save_fig("01_dataset_audit.png")
    return audit


# --------------------------------------------------------
#  STEP 2 - DATA CLEANING
# --------------------------------------------------------

def step2_data_cleaning():
    banner(2, "Data Cleaning - ffill, fillna, column remapping")

    nav_path = os.path.join(DATASET_DIR, "uav_navigation_dataset.csv")
    imu_path = os.path.join(DATASET_DIR, "imu_data.csv")

    # Load raw
    nav_raw = pd.read_csv(nav_path)
    imu_raw = pd.read_csv(imu_path, nrows=IMU_MAX_ROWS)

    nav_nan_before = nav_raw.isnull().sum()
    imu_nan_before = imu_raw.isnull().sum()

    # Clean nav
    nav = nav_raw.copy()
    nav["timestamp"] = pd.to_datetime(nav["timestamp"])
    nav.sort_values("timestamp", inplace=True)
    nav.ffill(inplace=True)
    nav.fillna(0, inplace=True)
    if "speed" not in nav.columns: nav["speed"] = 0.0
    # Drop obstacle-avoidance rows - trajectory contaminated by collision logic
    if "obstacle_detected" in nav.columns:
        _before = len(nav)
        nav = nav[nav["obstacle_detected"] == 0].copy()
        print(f"  nav: dropped {_before - len(nav)} obstacle-avoidance rows "
              f"({(_before - len(nav)) / _before * 100:.1f}%)")
    # dt from timestamp differences (seconds)
    nav["dt"] = (nav["timestamp"].diff()
                 .dt.total_seconds().fillna(0.1).clip(1e-3, 10.0))
    # Orientation & magnetometer not in nav CSV - fill zero
    for _c in ("roll", "pitch", "yaw", "mag_x", "mag_y", "mag_z"):
        if _c not in nav.columns:
            nav[_c] = 0.0

    # Clean imu
    imu = imu_raw.copy()
    imu.sort_values("time", inplace=True)
    imu.dropna(inplace=True)
    imu.rename(columns=IMU_RENAME, inplace=True)
    # dt from time column
    _dt = (imu["time"].diff()
           .fillna(imu["time"].diff().median()).clip(1e-3, 10.0))
    imu["dt"] = _dt
    # Speed: |displacement| / dt  (pos_x/y/z in metres after rename)
    _dx = imu["latitude"].diff().fillna(0.0)
    _dy = imu["longitude"].diff().fillna(0.0)
    _dz = imu["altitude"].diff().fillna(0.0)
    imu["speed"] = (np.sqrt(_dx**2 + _dy**2 + _dz**2) / _dt).fillna(0.0).clip(0.0, 200.0)

    # Targets, computed per-source (before any concatenation) so a delta
    # never spans two unrelated recordings. nav's lat/lon deltas are
    # converted from GPS degrees to metres (equirectangular approximation)
    # so they share imu's already-metric scale - see dataset_parser.py's
    # _M_PER_DEG_LAT/_M_PER_DEG_LON for the canonical constants/rationale.
    _lat_rad = np.radians(nav["latitude"])
    nav["delta_lat"] = nav["latitude"].diff().fillna(0.0) * _M_PER_DEG_LAT
    nav["delta_lon"] = (nav["longitude"].diff().fillna(0.0)
                         * _M_PER_DEG_LON * np.cos(_lat_rad))
    nav["delta_alt"] = nav["altitude"].diff().fillna(0.0)
    imu["delta_lat"], imu["delta_lon"], imu["delta_alt"] = _dx, _dy, _dz

    # Tag each source so create_dead_reckoning_dataset() can skip any
    # window/target that would straddle the (physically meaningless)
    # boundary between these two unrelated recordings once concatenated.
    nav["_src_id"] = 0
    imu["_src_id"] = 1

    nav_nan_after = nav.isnull().sum()
    imu_nan_after = imu.isnull().sum()

    print(f"  nav  rows: {len(nav_raw):>6,} -> {len(nav):>6,}  "
          f"NaN before: {nav_nan_before.sum():>3}  after: {nav_nan_after.sum()}")
    print(f"  imu  rows: {len(imu_raw):>6,} -> {len(imu):>6,}  "
          f"NaN before: {imu_nan_before.sum():>3}  after: {imu_nan_after.sum()}")
    print(f"  IMU column remap applied: {list(IMU_RENAME.keys())[:4]} -> ...")

    results["step2_cleaning"] = {
        "nav_rows_in": len(nav_raw), "nav_rows_out": len(nav),
        "nav_nan_before": int(nav_nan_before.sum()),
        "nav_nan_after":  int(nav_nan_after.sum()),
        "imu_rows_in": len(imu_raw), "imu_rows_out": len(imu),
        "imu_nan_before": int(imu_nan_before.sum()),
        "imu_nan_after":  int(imu_nan_after.sum()),
        "imu_renamed_cols": list(IMU_RENAME.keys()),
    }

    #  Figure 2 ─ 2 panels: row counts + IMU rename table
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    fig.suptitle("STEP 2 - Data Cleaning", fontsize=12, fontweight="bold")

    # Panel 1: rows in vs rows out
    ds_labels = ["nav", "imu"]
    rows_in  = [len(nav_raw), len(imu_raw)]
    rows_out = [len(nav),     len(imu)]
    x = np.arange(len(ds_labels))
    w = 0.35
    axes[0].bar(x - w/2, [r/1000 for r in rows_in],  w, label="Raw",     color="#F44336", alpha=0.85)
    axes[0].bar(x + w/2, [r/1000 for r in rows_out], w, label="Cleaned", color="#4CAF50", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ds_labels)
    for i, (ri, ro) in enumerate(zip(rows_in, rows_out)):
        axes[0].text(i - w/2, ri/1000 + 0.5, f"{ri:,}", ha="center", fontsize=7)
        axes[0].text(i + w/2, ro/1000 + 0.5, f"{ro:,}", ha="center", fontsize=7)
    style_ax(axes[0], "Row Count: Raw vs Cleaned", "Dataset", "Rows (K)")
    axes[0].legend(fontsize=8)

    # Panel 2: IMU column rename map (no arrow column)
    axes[1].axis("off")
    rename_items = list(IMU_RENAME.items())
    table_data = [[src, dst] for src, dst in rename_items]
    tbl = axes[1].table(
        cellText=table_data,
        colLabels=["imu_data.csv", "Canonical name"],
        cellLoc="center", loc="center",
        bbox=[0.05, 0.0, 0.9, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2196F3")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == 0:
            cell.set_facecolor("#fafafa")
        cell.set_edgecolor("#cccccc")
    axes[1].set_title("IMU Column Rename Map", fontsize=10, fontweight="bold", pad=8)

    plt.tight_layout()
    save_fig("02_data_cleaning.png")
    return nav, imu


# --------------------------------------------------------
#  STEP 3 - NORMALIZATION
# --------------------------------------------------------

def step3_normalization(nav: pd.DataFrame, imu: pd.DataFrame):
    banner(3, "Normalization - StandardScaler before/after comparison")

    combined = pd.concat([nav, imu], ignore_index=True).fillna(0)
    feats_raw = combined[FEATURE_COLS].values.astype(np.float32)

    scaler = StandardScaler()
    feats_norm = scaler.fit_transform(feats_raw)

    stats_before = {c: {"mean": round(float(feats_raw[:,i].mean()), 4),
                         "std":  round(float(feats_raw[:,i].std()),  4)}
                    for i, c in enumerate(FEATURE_COLS)}
    stats_after  = {c: {"mean": round(float(feats_norm[:,i].mean()), 4),
                         "std":  round(float(feats_norm[:,i].std()),  4)}
                    for i, c in enumerate(FEATURE_COLS)}

    print(f"  {'Feature':<15} {'Raw mean':>10} {'Raw std':>10} "
          f"{'Norm mean':>10} {'Norm std':>10}")
    print("  " + "-"*58)
    for c in FEATURE_COLS:
        print(f"  {c:<15} {stats_before[c]['mean']:>10.4f} "
              f"{stats_before[c]['std']:>10.4f} "
              f"{stats_after[c]['mean']:>10.4f} "
              f"{stats_after[c]['std']:>10.4f}")

    results["step3_normalization"] = {
        "before": stats_before, "after": stats_after,
        "note": "StandardScaler: mean=0, std=1 per feature after transform",
    }

    #  Figure 3 - 2 rows × N features; width scales with feature count
    fig, axes = plt.subplots(2, len(FEATURE_COLS),
                             figsize=(max(18, len(FEATURE_COLS) * 1.4), 7))
    fig.suptitle(
        "STEP 3 - Normalization (StandardScaler before / after)\n"
        "Shape is identical by design (linear transform)  -  "
        "difference is the X-axis scale. Badge shows std before → 1.00 after.",
        fontsize=11, fontweight="bold"
    )

    sample = min(10_000, len(feats_raw))
    idx    = np.random.choice(len(feats_raw), sample, replace=False)

    # shared x-axis for "after" row: all features normalised to same space
    norm_xlim = (-5, 5)

    for j, col in enumerate(FEATURE_COLS):
        raw_std  = stats_before[col]["std"]
        raw_mean = stats_before[col]["mean"]

        # ── Before ──
        axes[0, j].hist(feats_raw[idx, j], bins=40, color="#2196F3",
                        alpha=0.8, edgecolor="none")
        axes[0, j].set_title(col, fontsize=8, fontweight="bold")
        axes[0, j].set_xlabel(f"raw  (std={raw_std:.2f})", fontsize=6.5)
        axes[0, j].tick_params(labelsize=6)
        axes[0, j].grid(True, alpha=0.3)
        # Annotate the raw std as a badge so scale difference is obvious
        axes[0, j].annotate(
            f"σ={raw_std:.2f}",
            xy=(0.97, 0.96), xycoords="axes fraction",
            fontsize=6.5, ha="right", va="top", fontweight="bold", color="#1565C0",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#1565C0", alpha=0.9),
        )

        # ── After ──
        axes[1, j].hist(feats_norm[idx, j], bins=40, color="#4CAF50",
                        alpha=0.8, edgecolor="none")
        axes[1, j].set_xlabel("normalized  (σ=1.00)", fontsize=6.5)
        axes[1, j].tick_params(labelsize=6)
        axes[1, j].grid(True, alpha=0.3)
        axes[1, j].set_xlim(*norm_xlim)          # force same scale for all "after" plots
        axes[1, j].axvline(0, color="#e53935", linewidth=1.0, alpha=0.7)  # mark μ=0
        axes[1, j].annotate(
            "σ=1.00",
            xy=(0.97, 0.96), xycoords="axes fraction",
            fontsize=6.5, ha="right", va="top", fontweight="bold", color="#2e7d32",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#2e7d32", alpha=0.9),
        )

    axes[0, 0].set_ylabel("Before (count)", fontsize=8)
    axes[1, 0].set_ylabel("After (count - all same scale)",  fontsize=8)
    plt.tight_layout()
    save_fig("03_normalization.png")
    return scaler


# --------------------------------------------------------
#  STEP 4 - FEATURE ENGINEERING
# --------------------------------------------------------

def step4_features(nav: pd.DataFrame, imu: pd.DataFrame):
    banner(4, "Feature Engineering - features, targets, correlations")

    # delta_lat/lon/alt are already computed per-source in step2_data_cleaning()
    # (metre-unified, no diff across the nav/imu boundary) - do NOT recompute
    # via .diff() on the concatenated frame here, that would reintroduce the
    # single-row cross-source outlier documented in AI_RECOVERY_EXECUTION_PLAN.md §7.
    combined = pd.concat([nav, imu], ignore_index=True).fillna(0)

    all_cols = FEATURE_COLS + TARGET_COLS
    corr = combined[all_cols].corr()

    print(f"  Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"  Targets  ({len(TARGET_COLS)}):  {TARGET_COLS}")
    print(f"  Strongest feature->target correlations:")
    for t in TARGET_COLS:
        top = corr[t][FEATURE_COLS].abs().nlargest(3)
        print(f"    {t}: " + ", ".join(f"{k}={v:.3f}" for k, v in top.items()))

    results["step4_features"] = {
        "feature_cols": FEATURE_COLS,
        "target_cols":  TARGET_COLS,
        "top_correlations": {
            t: {k: round(float(v), 4)
                for k, v in corr[t][FEATURE_COLS].abs().nlargest(3).items()}
            for t in TARGET_COLS
        },
    }

    #  Figure 4 ─
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle("STEP 4 - Feature × Target Correlation Matrix",
                 fontsize=13, fontweight="bold")

    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(all_cols)))
    ax.set_yticks(range(len(all_cols)))
    ax.set_xticklabels(all_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(all_cols, fontsize=8)
    for i in range(len(all_cols)):
        for j in range(len(all_cols)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}",
                    ha="center", va="center", fontsize=6.5,
                    color="black" if abs(corr.values[i,j]) < 0.7 else "white")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    save_fig("04_feature_correlations.png")
    return combined


# --------------------------------------------------------
#  STEP 5 - SEQUENCE BUILDING
# --------------------------------------------------------

def step5_sequences(nav: pd.DataFrame, imu: pd.DataFrame):
    banner(5, "Sequence Building - sliding-window visualization")

    nav = nav.copy()
    nav["delta_lat"] = nav["latitude"].diff().fillna(0)
    nav["delta_lon"] = nav["longitude"].diff().fillna(0)
    nav["delta_alt"] = nav["altitude"].diff().fillna(0)

    imu = imu.copy()
    imu["delta_lat"] = imu["latitude"].diff().fillna(0)
    imu["delta_lon"] = imu["longitude"].diff().fillna(0)
    imu["delta_alt"] = imu["altitude"].diff().fillna(0)

    nav_seq = len(nav) - WINDOW_SIZE
    imu_seq = len(imu) - WINDOW_SIZE
    print(f"  Window size    : {WINDOW_SIZE} timesteps")
    print(f"  Nav sequences  : {nav_seq:,}")
    print(f"  IMU sequences  : {imu_seq:,}  (used for visualization - has all 14 features)")
    print(f"  Input shape    : ({WINDOW_SIZE}, {len(FEATURE_COLS)})  ->  ({len(TARGET_COLS)},)")
    print(f"  Example: reading rows 0–{WINDOW_SIZE-1} -> predicts delta at row {WINDOW_SIZE}")

    results["step5_sequences"] = {
        "window_size": WINDOW_SIZE,
        "nav_sequences": nav_seq,
        "imu_sequences": imu_seq,
        "input_shape": [WINDOW_SIZE, len(FEATURE_COLS)],
        "output_shape": [len(TARGET_COLS)],
    }

    # Use IMU for the window visualization - nav zero-fills roll/pitch/yaw/mag,
    # making those subplots uninformative. IMU has all 14 features populated.
    start  = 1000   # pick a mid-flight window, not the very start
    src_df = imu
    window = src_df[FEATURE_COLS].iloc[start : start + WINDOW_SIZE].values

    n_feat  = len(FEATURE_COLS)
    n_rows  = (n_feat + 1) // 2

    fig = plt.figure(figsize=(13, 2.2 * n_rows + 2.5))
    fig.suptitle(
        f"STEP 5 - One IMU Sequence Window (rows {start}–{start+WINDOW_SIZE-1})\n"
        f"Each sub-plot = one input feature over {WINDOW_SIZE} timesteps  "
        f"[source: imu_data.csv - all 14 features populated]",
        fontsize=11, fontweight="bold"
    )
    gs = gridspec.GridSpec(n_rows + 1, 2, figure=fig, hspace=0.55, wspace=0.32)

    feature_axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(n_feat)]
    target_ax    = fig.add_subplot(gs[n_rows, :])

    colors_feat = plt.cm.tab10(np.linspace(0, 1, len(FEATURE_COLS)))
    for j, (col, c) in enumerate(zip(FEATURE_COLS, colors_feat)):
        ax = feature_axes[j]
        col_data = window[:, j]

        ax.plot(range(WINDOW_SIZE), col_data, color=c,
                linewidth=2, marker="o", markersize=4)
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_xlabel("timestep", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        for spine in ax.spines.values():
            spine.set_visible(True)

    pred_row = src_df[TARGET_COLS].iloc[start + WINDOW_SIZE]
    target_ax.axis("off")
    summary = (
        f"Predicted target  (IMU row {start + WINDOW_SIZE}):\n\n"
        f"  delta_lat = {pred_row['delta_lat']:.7f}  m  (Δ relative-x)\n"
        f"  delta_lon = {pred_row['delta_lon']:.7f}  m  (Δ relative-y)\n"
        f"  delta_alt = {pred_row['delta_alt']:.7f}  m  (Δ relative-z)"
    )
    target_ax.text(0.5, 0.5, summary, transform=target_ax.transAxes,
                   fontsize=10, va="center", ha="center", family="monospace",
                   bbox=dict(boxstyle="square,pad=0.6", fc="#f0f4ff", ec="none"))

    save_fig("05_sequence_window.png", fig)


# --------------------------------------------------------
#  TRAINING CORE
# --------------------------------------------------------

def _train_run(run_id: str, df: pd.DataFrame, normalize: bool,
               fig_name: str) -> dict:
    """Core training loop. Returns per-epoch loss dict."""
    print(f"\n  Config: {len(df):,} rows  normalize={normalize}"
          f"  epochs={EPOCHS}  batch={BATCH_SIZE}")

    # create_dead_reckoning_dataset() (data_processing/dataset_parser.py) is the
    # single source of truth for sequence building: it expects delta_lat/lon/alt
    # and _src_id already on df (set in step2_data_cleaning()), keeps sequences
    # in temporal order (no leakage from shuffling overlapping windows), skips
    # any window that would straddle a source boundary, and fits the scaler on
    # the train split only (no leakage from fitting on validation rows).
    X_train, y_train, X_val, y_val, _, _ = create_dead_reckoning_dataset(
        df, window_size=WINDOW_SIZE, normalize=normalize)
    n_seq = len(X_train) + len(X_val)

    train_dl = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                          batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
                          batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DeadReckoningLSTM().to(device)
    crit   = nn.MSELoss()
    opt    = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    train_losses, val_losses = [], []
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tl = sum(
            _fwd(model, bx.to(device), by.to(device), opt, crit)
            for bx, by in train_dl
        ) / len(train_dl)

        model.eval()
        with torch.no_grad():
            vl = sum(
                crit(model(bx.to(device)), by.to(device)).item()
                for bx, by in val_dl
            ) / len(val_dl)

        train_losses.append(round(tl, 6))
        val_losses.append(round(vl, 6))
        print(f"  Epoch [{epoch:>2}/{EPOCHS}]  Train: {tl:.4f}  Val: {vl:.4f}")

    elapsed = round(time.time() - t0, 1)
    mdl_path = os.path.join(MDL_DIR, f"{run_id}.pth")
    torch.save(model.state_dict(), mdl_path)
    print(f"  Model -> runs/models/{run_id}.pth  ({elapsed}s)")

    #  Per-run loss figure 
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs_x = range(1, EPOCHS + 1)
    ax.plot(epochs_x, train_losses, "o-", color="#2196F3",
            linewidth=2, markersize=4, label="Train Loss")
    ax.plot(epochs_x, val_losses,   "s-", color="#F44336",
            linewidth=2, markersize=4, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Run: {run_id}\n"
                 f"rows={len(df):,}  seq={n_seq:,}  "
                 f"norm={'yes' if normalize else 'no'}  "
                 f"final Val={val_losses[-1]:.4f}",
                 fontsize=10, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    save_fig(fig_name)

    return {
        "run_id": run_id,
        "rows": len(df),
        "sequences": n_seq,
        "normalize": normalize,
        "epochs": EPOCHS,
        "train_losses": train_losses,
        "val_losses":   val_losses,
        "final_train":  train_losses[-1],
        "final_val":    val_losses[-1],
        "elapsed_s":    elapsed,
    }


def _fwd(model, bx, by, opt, crit):
    opt.zero_grad()
    loss = crit(model(bx), by)
    loss.backward()
    opt.step()
    return loss.item()


# --------------------------------------------------------
#  STEPS 6–9 - TRAINING RUNS
# --------------------------------------------------------

def step6_run_A(nav: pd.DataFrame):
    banner(6, "Run A - nav only, NO normalization  (baseline)")
    r = _train_run("A_nav_nonorm", nav, normalize=False,
                   fig_name="06_runA_nav_nonorm.png")
    results["runA"] = r
    return r


def step7_run_B(nav: pd.DataFrame):
    banner(7, "Run B - nav only, normalized")
    r = _train_run("B_nav_norm", nav, normalize=True,
                   fig_name="07_runB_nav_norm.png")
    results["runB"] = r
    return r


def step8_run_C(imu: pd.DataFrame):
    banner(8, "Run C - imu only, normalized")
    r = _train_run("C_imu_norm", imu, normalize=True,
                   fig_name="08_runC_imu_norm.png")
    results["runC"] = r
    return r


def step9_run_D(nav: pd.DataFrame, imu: pd.DataFrame):
    banner(9, "Run D - nav + imu combined, normalized")
    combined = pd.concat([nav, imu], ignore_index=True).fillna(0)
    r = _train_run("D_nav_imu_norm", combined, normalize=True,
                   fig_name="09_runD_nav_imu_norm.png")
    results["runD"] = r
    return r


# --------------------------------------------------------
#  STEP 10 - ALL-RUNS COMPARISON
# --------------------------------------------------------

def _select_trustworthy_winner(runs: list) -> dict:
    """Picks the run to report as "best" -- prefers C_imu_norm (the only
    source not tainted by the deprecated 'nav' dataset, see
    AI_RECOVERY_EXECUTION_PLAN.md §10) over naive numeric minimization.
    Runs A/B/D all depend on 'nav', whose position data is not a real
    trajectory -- their val loss numbers aren't meaningful to rank against,
    even now that the leakage/scaler-order/domain-shift bugs that used to
    additionally corrupt them are fixed (a fixed pipeline applied to fake
    data still isn't a real result). Falls back to numeric min only if
    C_imu_norm isn't present in this particular run set.
    """
    for r in runs:
        if r["run_id"] == "C_imu_norm":
            return r
    return min(runs, key=lambda r: r["final_val"])


def step10_comparison(runs: list):
    banner(10, "All-Runs Comparison - Val Loss curves + final bar chart")

    colors  = ["#9E9E9E", "#2196F3", "#4CAF50", "#F44336"]
    markers = ["x",       "o",       "s",        "D"]

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("STEP 10 - All Runs Comparison", fontsize=13, fontweight="bold")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

    # Val loss curves
    ax1 = fig.add_subplot(gs[0, :2])
    for r, c, m in zip(runs, colors, markers):
        ax1.plot(range(1, EPOCHS + 1), r["val_losses"],
                 color=c, marker=m, linewidth=2, markersize=5,
                 label=f"{r['run_id']}  (final={r['final_val']:.4f})")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Val Loss (MSE, log scale)")
    ax1.set_title("Validation Loss - all runs (log scale)"); ax1.legend(fontsize=8)
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3, which="both"); ax1.spines[["top","right"]].set_visible(False)

    # Final val loss bar (log scale so Run A doesn't dwarf B/C/D)
    ax2 = fig.add_subplot(gs[0, 2])
    names  = [r["run_id"].replace("_", "\n") for r in runs]
    finals = [r["final_val"] for r in runs]
    bars   = ax2.bar(names, finals, color=colors, edgecolor="white")
    ax2.set_yscale("log")
    for bar, v in zip(bars, finals):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 v * 1.5,
                 f"{v:.4f}", ha="center", fontsize=8, fontweight="bold")
    ax2.set_ylabel("Final Val Loss (log)"); ax2.set_title("Final Epoch Val Loss (log)")
    ax2.grid(True, alpha=0.3, axis="y", which="both")
    ax2.spines[["top","right"]].set_visible(False)

    save_fig("10_comparison_all_runs.png")

    winner   = _select_trustworthy_winner(runs)
    run_B    = next(r for r in runs if r["run_id"].startswith("B"))
    imp_vs_A = round((runs[0]["final_val"] - winner["final_val"]) /
                      max(runs[0]["final_val"], 1e-9) * 100, 1)
    imp_vs_B = round((run_B["final_val"] - winner["final_val"]) /
                      max(run_B["final_val"], 1e-9) * 100, 1)
    print(f"\n  Winner: {winner['run_id']}  Val Loss={winner['final_val']:.4f}")
    print(f"  Improvement vs Run A (no-norm baseline) : {imp_vs_A}%")
    print(f"  Improvement vs Run B (nav-norm baseline): {imp_vs_B}%")
    results["step10_comparison"] = {
        "winner": winner["run_id"],
        "winner_val_loss": winner["final_val"],
        "improvement_pct_over_A_nonorm":   imp_vs_A,
        "improvement_pct_over_B_nav_norm": imp_vs_B,
    }


# --------------------------------------------------------
#  STEP 11 - RL EVALUATION
# --------------------------------------------------------

def step11_rl_evaluation():
    """Delegates to eval_metrics.py rather than reimplementing this
    comparison a third time -- ml_pipeline.py used to run its own disconnected
    toy ([0,10]x[0,10] grid, fixed target, hardcoded step size) that never
    touched the real system, a separate copy of the exact same problem found
    and fixed in eval_metrics.py's own former run_rule_based_eval(). Both now
    share one source of truth: the real recovery_orchestrator._hand_coded_seek
    formula and the real RecoveryPolicyEnv, imported once here."""
    banner(11, "RL Path Recovery - Rule-Based vs PPO Agent")

    import eval_metrics

    rb_rate, rb_avg, hc_results = eval_metrics.run_rule_based_eval()
    rl_rate_frac, rl_avg, rl_results = eval_metrics.run_rl_eval()

    rb_rate = round(rb_rate * 100, 1)
    rb_avg = round(rb_avg, 2)
    rl_rate = round(rl_rate_frac * 100, 1) if rl_rate_frac is not None else None
    rl_avg = round(rl_avg, 2) if rl_avg is not None else None
    rl_note = (f"loaded from {eval_metrics._RL_POLICY_PATH}" if rl_results is not None
               else f"RL seek policy not found at {eval_metrics._RL_POLICY_PATH}")

    results["step11_rl"] = {
        "rule_based_success_pct": rb_rate, "rule_based_avg_steps": rb_avg,
        "rl_success_pct": rl_rate,         "rl_avg_steps": rl_avg,
        "rl_note": rl_note,
    }
    if rl_results is not None:
        table = eval_metrics.confidence_binned_table(hc_results, rl_results)
        results["step11_rl"]["confidence_binned_mean_seek"] = table

    #  Figure 11 ─
    labels  = ["Rule-Based", "RL (PPO)"]
    s_vals  = [rb_rate, rl_rate if rl_rate is not None else 0]
    st_vals = [rb_avg,  rl_avg  if rl_avg  is not None else 0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("STEP 11 - Path Recovery: Rule-Based vs RL Agent",
                 fontsize=13, fontweight="bold")

    c = ["#2196F3","#F44336"]
    axes[0].bar(labels, s_vals, color=c, edgecolor="white")
    for i, v in enumerate(s_vals):
        axes[0].text(i, v + 1, f"{v}%", ha="center", fontweight="bold")
    style_ax(axes[0], "Success Rate (%)", "", "%")
    axes[0].set_ylim(0, 115)

    axes[1].bar(labels, st_vals, color=c, edgecolor="white")
    for i, v in enumerate(st_vals):
        axes[1].text(i, v + 1, f"{v:.0f}", ha="center", fontweight="bold")
    style_ax(axes[1], "Avg Steps to Recovery", "", "Steps")

    if rl_rate is None:
        axes[0].text(1, 5, "N/A\n(model not\nloaded)", ha="center",
                     fontsize=9, color="#999")
        axes[1].text(1, 5, "N/A", ha="center", fontsize=9, color="#999")

    plt.tight_layout()
    save_fig("11_rl_evaluation.png")


# --------------------------------------------------------
#  STEP 12 - FINAL SUMMARY
# --------------------------------------------------------

def step12_summary(runs: list, run_count: int = 1):
    banner(12, "Final Summary - results.json + summary figure")

    #  Append this run to results.json, keyed by run number ─────────────────
    json_path = os.path.join(RUNS_DIR, "results.json")
    all_results = {}
    if os.path.exists(json_path):
        try:
            with open(json_path) as _f:
                all_results = json.load(_f)
        except Exception:
            all_results = {}
    all_results[f"run_{run_count}"] = {
        "timestamp": datetime.now().isoformat(),
        **results,
    }
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  -> Saved: runs/results.json  (run_{run_count} appended)")

    #  Summary table figure 
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#1a1a2e")
    ax.axis("off")
    fig.suptitle("SD-UAV AI Recovery - ML Pipeline Summary",
                 fontsize=16, fontweight="bold", color="white", y=0.97)

    rows_table = [
        ["Run", "Sources", "Norm", "Sequences", "Final Train Loss",
         "Final Val Loss", "Time (s)"],
    ]
    for r in runs:
        rows_table.append([
            r["run_id"],
            "nav+imu" if "imu" in r["run_id"] and "nav" in r["run_id"]
            else ("imu" if "imu" in r["run_id"] else "nav"),
            "Yes" if r["normalize"] else "No",
            f"{r['sequences']:,}",
            f"{r['final_train']:.4f}",
            f"{r['final_val']:.4f}",
            f"{r['elapsed_s']}",
        ])

    col_widths = [0.18, 0.10, 0.07, 0.12, 0.16, 0.15, 0.10]
    col_colors = ["#cfe2f3"] * len(col_widths)
    header_c   = "#e69138"
    data_c     = [["#ffffff"] * len(col_widths)] * len(rows_table[1:])

    tbl = ax.table(
        cellText  = rows_table[1:],
        colLabels = rows_table[0],
        cellLoc   = "center",
        loc       = "center",
        colWidths = col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 2.2)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#000000")
        if r == 0:
            cell.set_facecolor(header_c)
            cell.set_text_props(color="black", fontweight="bold")
        else:
            # Highlight winner row -- green background alone does the pointing;
            # text stays bold black rather than also being colored, for legibility.
            is_winner = rows_table[r][0] == _select_trustworthy_winner(runs)["run_id"]
            cell.set_facecolor("#d5ffd8" if is_winner else "#ffffff")
            cell.set_text_props(
                color="black",
                fontweight="bold" if is_winner else "normal",
            )

    # Key stats text
    winner = _select_trustworthy_winner(runs)
    rl_data = results.get("step11_rl", {})
    stats_text = (
        f"Best LSTM: {winner['run_id']}  ->  Val Loss = {winner['final_val']:.4f}\n"
        f"(Runs A/B/D use the deprecated 'nav' dataset -- reference only, see "
        f"AI_RECOVERY_EXECUTION_PLAN.md §10)\n"
        f"Rule-Based Path Recovery: "
        f"{rl_data.get('rule_based_success_pct','?')}% success  "
        f"/ {rl_data.get('rule_based_avg_steps','?')} avg steps\n"
        f"RL Agent (10K steps): "
        f"{rl_data.get('rl_success_pct') or 'N/A'}% success  "
        f"/ {rl_data.get('rl_avg_steps') or 'N/A'} avg steps\n"
        f"Production model: ai_backend/models/dr_lstm_imu_norm.pth"
    )
    ax.text(0.5, 0.08, stats_text, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9.5, color="#0f3460",
            family="monospace", fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.5", fc="#dce6f5", ec="none"))

    plt.tight_layout(rect=[0, 0.0, 1, 0.96])
    save_fig("12_summary_dashboard.png")

    #  Console summary ─
    print("\n" + "═"*70)
    print("  PIPELINE COMPLETE")
    print("═"*70)
    for r in runs:
        marker = " ← BEST" if r["run_id"] == winner["run_id"] else ""
        print(f"  {r['run_id']:<22}  Val Loss={r['final_val']:.4f}{marker}")
    print(f"\n  Results JSON : runs/results.json")
    print(f"  Figures      : runs/figures/  ({len(os.listdir(FIG_DIR))} files)")
    print(f"  Models       : runs/models/   ({len(os.listdir(MDL_DIR))} files)")
    print("═"*70)


#  MAIN
# --------------------------------------------------------

def main():
    # ── Run counter ────────────────────────────────────────────────────────────
    counter_path = os.path.join(RUNS_DIR, "run_counter.json")
    run_count = 1
    if os.path.exists(counter_path):
        try:
            with open(counter_path) as _f:
                run_count = json.load(_f).get("count", 0) + 1
        except Exception:
            pass
    with open(counter_path, "w") as _f:
        json.dump({"count": run_count,
                   "last_run": datetime.now().isoformat()}, _f, indent=2)

    # ── Clear previous run outputs (figures, models, results.json) ────────────
    for _f in os.listdir(FIG_DIR):
        if _f.endswith(".png"):
            os.remove(os.path.join(FIG_DIR, _f))
    for _f in os.listdir(MDL_DIR):
        if _f.endswith(".pth"):
            os.remove(os.path.join(MDL_DIR, _f))

    # ── Tee stdout + stderr → runs/pipeline.log (append) ──────────────────────
    log_path  = os.path.join(RUNS_DIR, "pipeline.log")
    _log_file = open(log_path, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _log_file)
    sys.stderr = _Tee(sys.__stderr__, _log_file)

    print("\n" + "═"*70)
    print(f"  RUN #{run_count}  ·  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═"*70)

    print("\n" + "═"*70)
    print("  SD-UAV AI Recovery - Full ML Pipeline")
    print(f"  epochs={EPOCHS}  batch={BATCH_SIZE}  window={WINDOW_SIZE}"
          f"  imu_max_rows={IMU_MAX_ROWS}")
    print("═"*70)

    # Step 1 - audit
    step1_dataset_audit()

    # Step 2 - cleaning  (returns cleaned DataFrames)
    nav, imu = step2_data_cleaning()

    # Step 3 - normalization analysis
    step3_normalization(nav, imu)

    # Step 4 - feature engineering
    step4_features(nav, imu)

    # Step 5 - sequence visualization
    step5_sequences(nav, imu)

    # Steps 6–9 - training runs
    runs = []
    runs.append(step6_run_A(nav))
    runs.append(step7_run_B(nav))
    runs.append(step8_run_C(imu))
    runs.append(step9_run_D(nav, imu))

    # Step 10 - comparison
    step10_comparison(runs)

    # Step 11 - RL evaluation
    step11_rl_evaluation()

    # Step 12 - summary
    step12_summary(runs, run_count)

    # ── Flush and restore stdout/stderr ────────────────────────────────────────
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    _log_file.close()
    print(f"  Log appended -> runs/pipeline.log  (run #{run_count})")


if __name__ == "__main__":
    main()
