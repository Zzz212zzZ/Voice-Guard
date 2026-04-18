#!/usr/bin/env python3
"""Threshold sweep — show how FPR/FNR/accuracy change with threshold.

Currently the Shield uses (0.3, 0.7) as (REAL, SUSPICIOUS, FAKE) thresholds,
but those were picked by the original paper on ASVspoof's training conditions.
This script sweeps threshold over [0, 1] and plots the trade-off — useful
for both the results slide and the future-work discussion.

Usage:
    python scripts/threshold_sweep.py --tier subsample
    python scripts/threshold_sweep.py --tier full
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse styling + colors from analyze_eval
from analyze_eval import (
    _apply_slide_style, COLOR_REAL, COLOR_FAKE,
    COLOR_HIGHLIGHT, COLOR_ACCENT, COLOR_NEUTRAL,
    compute_eer,
)


def sweep_threshold(df, n_points=101):
    """Compute FPR, FNR, accuracy across threshold range [0, 1]."""
    y_true = (df["label"] == "spoof").astype(int).values
    y_score = df["spoof_prob"].values

    thresholds = np.linspace(0, 1, n_points)
    fpr_list, fnr_list, acc_list = [], [], []

    for t in thresholds:
        pred_spoof = y_score >= t
        # FPR: of all bonafide, how many predicted as spoof
        real_mask = y_true == 0
        fpr = pred_spoof[real_mask].sum() / max(real_mask.sum(), 1) * 100
        # FNR: of all spoof, how many predicted as bonafide
        fake_mask = y_true == 1
        fnr = (~pred_spoof[fake_mask]).sum() / max(fake_mask.sum(), 1) * 100
        acc = (pred_spoof == y_true).sum() / len(y_true) * 100
        fpr_list.append(fpr)
        fnr_list.append(fnr)
        acc_list.append(acc)

    return thresholds, np.array(fpr_list), np.array(fnr_list), np.array(acc_list)


def plot_threshold_sweep(df, out_path):
    thresholds, fprs, fnrs, accs = sweep_threshold(df)

    # Find optimal thresholds
    # - Min FPR+FNR (balanced)
    balanced = fprs + fnrs
    best_bal_idx = int(np.argmin(balanced))
    # - EER point
    y_true = (df["label"] == "spoof").astype(int).values
    y_score = df["spoof_prob"].values
    eer_pct, eer_thr, _, _ = compute_eer(y_true, y_score)

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(thresholds, fprs, color=COLOR_REAL, linewidth=2.5,
            label="False Positive Rate (real → FAKE)")
    ax.plot(thresholds, fnrs, color=COLOR_FAKE, linewidth=2.5,
            label="False Negative Rate (fake → REAL)")
    ax.plot(thresholds, accs, color=COLOR_ACCENT, linewidth=2,
            linestyle=":", label="Overall Accuracy")

    # Current threshold markers (from Shield UI)
    ax.axvline(0.3, color=COLOR_NEUTRAL, linestyle="--", alpha=0.5, linewidth=1.2)
    ax.axvline(0.7, color=COLOR_NEUTRAL, linestyle="--", alpha=0.5, linewidth=1.2)
    ax.text(0.3, 102, "current\n(REAL)", ha="center", fontsize=9, color=COLOR_NEUTRAL)
    ax.text(0.7, 102, "current\n(FAKE)", ha="center", fontsize=9, color=COLOR_NEUTRAL)

    # Annotate optimal + EER
    ax.scatter([thresholds[best_bal_idx]], [balanced[best_bal_idx] / 2],
               s=120, color=COLOR_HIGHLIGHT, zorder=5, edgecolors="white", linewidth=2,
               label=f"Balanced-optimal thr = {thresholds[best_bal_idx]:.2f}")

    ax.set_xlim(0, 1)
    ax.set_ylim(-3, 110)
    ax.set_xlabel("Spoof-Probability Threshold for FAKE verdict")
    ax.set_ylabel("Rate (%)")
    ax.set_title(f"Threshold Sensitivity — ASVspoof 2019 LA (N={len(df):,})")
    ax.legend(loc="center right")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")

    print(f"\n  Current thresholds: (0.3 REAL, 0.7 FAKE)")
    print(f"  Balanced-optimal threshold: {thresholds[best_bal_idx]:.2f} "
          f"(FPR={fprs[best_bal_idx]:.2f}%, FNR={fnrs[best_bal_idx]:.2f}%)")
    print(f"  EER: {eer_pct:.3f}% @ threshold {eer_thr:.4f}")

    return {
        "balanced_optimal_threshold": float(thresholds[best_bal_idx]),
        "balanced_optimal_fpr_pct": float(fprs[best_bal_idx]),
        "balanced_optimal_fnr_pct": float(fnrs[best_bal_idx]),
        "eer_pct": float(eer_pct),
        "eer_threshold": float(eer_thr),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=["sample", "subsample", "full"], required=True)
    args = parser.parse_args()

    _apply_slide_style()

    in_csv = os.path.join(ROOT, "scripts", "output", "asvspoof_eval", f"eval_{args.tier}.csv")
    if not os.path.isfile(in_csv):
        print(f"Not found: {in_csv}")
        sys.exit(1)

    out_dir = os.path.join(ROOT, "scripts", "output", "asvspoof_eval", args.tier)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(in_csv)
    print(f"Loaded {len(df):,} rows from {in_csv}")

    out_path = os.path.join(out_dir, "threshold_sweep.png")
    plot_threshold_sweep(df, out_path)


if __name__ == "__main__":
    main()
