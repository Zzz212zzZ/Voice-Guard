#!/usr/bin/env python3
"""Turn eval CSV into slide-ready charts + metrics report.

Takes the output of eval_asvspoof.py and produces:
  - EER + ROC curve (with EER point annotated)
  - Per-attack-type accuracy bar chart (zoomed to highlight differences)
  - Confusion matrix (counts + percentages)
  - Feature distributions at scale (violin plots, 8 features)
  - Spoof-probability histogram (where wrong predictions cluster)
  - Failure case listing (top false-positive / false-negative samples)
  - Metrics JSON for machine-readable summary

All plots use presentation-grade styling: large fonts (readable from back
of a lecture room), high-contrast colors, consistent palette, 200 DPI.

Usage:
    python scripts/analyze_eval.py --tier sample
    python scripts/analyze_eval.py --tier subsample
    python scripts/analyze_eval.py --tier full
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 13 LA 2019 eval attacks, keep in order for chart readability
ATTACK_ORDER = [f"A{i:02d}" for i in range(7, 20)]  # A07..A19

FEATURE_COLS = [
    ("jitter", "Jitter"),
    ("shimmer", "Shimmer"),
    ("f0_cv", "F0 CV"),
    ("energy_cv", "Energy CV"),
    ("spectral_centroid", "Spectral Centroid (Hz)"),
    ("spectral_bandwidth", "Spectral Bandwidth (Hz)"),
    ("spectral_flatness", "Spectral Flatness"),
    ("spectral_rolloff", "Spectral Rolloff (Hz)"),
]

# Consistent palette across all plots
COLOR_REAL = "#2E8B8B"       # teal — bonafide
COLOR_FAKE = "#E53E3E"       # red  — spoof
COLOR_REAL_SCATTER = "#38B2AC"
COLOR_FAKE_SCATTER = "#F56565"
COLOR_NEUTRAL = "#64748B"    # slate gray
COLOR_HIGHLIGHT = "#DD6B20"  # orange — for annotations
COLOR_ACCENT = "#3182CE"     # blue — for ROC, thresholds


def _apply_slide_style():
    """Set global matplotlib rcParams for presentation-quality output."""
    mpl.rcParams.update({
        "font.size": 13,
        "font.family": "sans-serif",
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "axes.labelweight": "normal",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def compute_eer(y_true, y_score):
    """Equal Error Rate: point where FAR == FRR. Returns (eer_pct, threshold, fpr, tpr)."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer * 100, float(thr[idx]), fpr, tpr


# ============================================================
# Plot 1: ROC curve
# ============================================================
def plot_roc(fpr, tpr, eer_pct, n_samples, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))

    # ROC curve
    ax.plot(fpr, tpr, color=COLOR_ACCENT, linewidth=2.5,
            label=f"W2V-AASIST  (EER = {eer_pct:.2f}%)")

    # Random baseline
    ax.plot([0, 1], [0, 1], color=COLOR_NEUTRAL, linestyle="--",
            linewidth=1.2, alpha=0.6, label="Random classifier")

    # EER point
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    ax.scatter([fpr[eer_idx]], [tpr[eer_idx]], s=140,
               color=COLOR_HIGHLIGHT, zorder=5,
               edgecolors="white", linewidth=2,
               label=f"EER point ({eer_pct:.2f}%)")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — ASVspoof 2019 LA eval (N={n_samples:,})")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================
# Plot 2: Per-attack accuracy
# ============================================================
def plot_per_attack(df, out_path):
    """Detection accuracy per attack type. Zoomed y-axis to emphasize differences."""
    attacks = [a for a in ATTACK_ORDER if a in df["attack_type"].values]
    accuracies = []
    counts = []
    for a in attacks:
        sub = df[df["attack_type"] == a]
        correct = (sub["verdict"] != "REAL").sum()
        accuracies.append(correct / len(sub) * 100)
        counts.append(len(sub))

    bona = df[df["attack_type"] == "bonafide"]
    bona_acc = (bona["verdict"] == "REAL").sum() / len(bona) * 100 if len(bona) else 0

    xs = list(range(len(attacks) + 1))
    ys = accuracies + [bona_acc]
    labels = attacks + ["bonafide"]
    colors = [COLOR_FAKE] * len(attacks) + [COLOR_REAL]
    ns = counts + [len(bona)]

    # Zoom if all accuracies > 90%
    y_min = min(ys) if ys else 0
    if y_min > 90:
        y_lo, y_hi = 90, 101
    elif y_min > 75:
        y_lo, y_hi = 70, 102
    else:
        y_lo, y_hi = 0, 108

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(xs, ys, color=colors, width=0.7,
                  edgecolor="white", linewidth=1.0)

    # Value labels on top of bars
    for x, y in zip(xs, ys):
        label_off = (y_hi - y_lo) * 0.015
        ax.text(x, y + label_off, f"{y:.1f}%",
                ha="center", fontsize=11, fontweight="bold")

    # Average line
    avg = np.mean(ys)
    ax.axhline(y=avg, color=COLOR_HIGHLIGHT, linestyle=":", linewidth=1.5,
               alpha=0.7, label=f"Mean = {avg:.1f}%")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=0, fontsize=11)
    ax.set_ylabel("Detection Accuracy (%)")
    ax.set_ylim(y_lo, y_hi)

    # n= caption (summary, not per-bar to avoid overlap)
    n_summary = (
        f"Per-attack n = {counts[0]:,}" if len(set(counts)) == 1
        else f"n ≈ {int(np.mean(counts)):,} per attack"
    )
    ax.set_xlabel(f"Attack Type   ({n_summary},  bonafide n = {len(bona):,})",
                  fontsize=11, labelpad=10)
    ax.set_title(f"Per-Attack Detection Accuracy — ASVspoof 2019 LA (N={len(df):,})")
    ax.legend(loc="lower right")
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================
# Plot 3: Confusion matrix
# ============================================================
def plot_confusion(df, out_path):
    """Confusion matrix with counts + row-normalized percentages."""
    labels_order = ["bonafide", "spoof"]
    verdict_order = ["REAL", "SUSPICIOUS", "FAKE"]

    mat = np.zeros((len(labels_order), len(verdict_order)), dtype=int)
    for i, lbl in enumerate(labels_order):
        for j, v in enumerate(verdict_order):
            mat[i, j] = ((df["label"] == lbl) & (df["verdict"] == v)).sum()

    # Row-normalize for percentage
    row_sums = mat.sum(axis=1, keepdims=True)
    pct = mat / np.maximum(row_sums, 1) * 100

    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(pct, cmap="Blues", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(verdict_order)))
    ax.set_xticklabels(verdict_order, fontsize=13)
    ax.set_yticks(range(len(labels_order)))
    ax.set_yticklabels(labels_order, fontsize=13)
    ax.set_xlabel("Predicted Verdict", fontsize=13)
    ax.set_ylabel("Ground Truth", fontsize=13)
    ax.set_title(f"Confusion Matrix — ASVspoof 2019 LA (N={len(df):,})")

    # Annotate each cell with count + percent
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            count = mat[i, j]
            p = pct[i, j]
            text_color = "white" if p > 50 else "black"
            ax.text(j, i - 0.08, f"{count:,}",
                    ha="center", va="center",
                    color=text_color, fontsize=16, fontweight="bold")
            ax.text(j, i + 0.22, f"({p:.1f}%)",
                    ha="center", va="center",
                    color=text_color, fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("% of row", fontsize=11)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================
# Plot 4: Feature distributions at scale
# ============================================================
def plot_feature_distributions(df, out_path):
    """Violin plots for 8 features — Real vs Fake. Handles large N gracefully."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    real = df[df["label"] == "bonafide"]
    fake = df[df["label"] == "spoof"]

    for ax, (col, pretty) in zip(axes, FEATURE_COLS):
        real_vals = real[col].dropna().values
        fake_vals = fake[col].dropna().values

        # Violin plot — better than boxplot at scale (shows distribution shape)
        parts = ax.violinplot(
            [real_vals, fake_vals],
            positions=[1, 2], widths=0.75,
            showmeans=False, showmedians=True, showextrema=False,
        )
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(COLOR_REAL if i == 0 else COLOR_FAKE)
            pc.set_edgecolor("black")
            pc.set_alpha(0.65)
            pc.set_linewidth(1.0)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(2)

        # Overlay downsampled scatter (150 points per group, so it reads at any N)
        for vals, pos, color in [(real_vals, 1, COLOR_REAL_SCATTER),
                                 (fake_vals, 2, COLOR_FAKE_SCATTER)]:
            if len(vals) > 150:
                sample = np.random.RandomState(0).choice(vals, 150, replace=False)
            else:
                sample = vals
            jitter = np.random.RandomState(1).normal(0, 0.05, size=len(sample))
            ax.scatter(pos + jitter, sample, alpha=0.35, s=10,
                       color=color, zorder=3, edgecolors="none")

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Real", "Fake"], fontsize=12)
        ax.set_title(pretty, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.grid(axis="x", visible=False)

    fig.suptitle(
        f"Acoustic Feature Distributions at Scale — N={len(df):,} "
        f"(n_real={len(real):,}, n_fake={len(fake):,})",
        fontsize=16, fontweight="bold", y=1.00,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================
# Plot 5: Spoof-prob histogram (shows where failures cluster)
# ============================================================
def plot_spoof_prob_histogram(df, out_path):
    """Where do real vs fake audios sit on the spoof_prob axis?"""
    real = df[df["label"] == "bonafide"]["spoof_prob"].values
    fake = df[df["label"] == "spoof"]["spoof_prob"].values

    fig, ax = plt.subplots(figsize=(11, 5.5))

    bins = np.linspace(0, 1, 51)
    ax.hist(real, bins=bins, alpha=0.65, color=COLOR_REAL,
            label=f"Bonafide (n={len(real):,})", edgecolor="white", linewidth=0.5)
    ax.hist(fake, bins=bins, alpha=0.65, color=COLOR_FAKE,
            label=f"Spoof (n={len(fake):,})", edgecolor="white", linewidth=0.5)

    # Threshold markers
    ax.axvline(0.3, color=COLOR_NEUTRAL, linestyle="--", linewidth=1.2,
               alpha=0.7, label="Verdict thresholds (0.3 / 0.7)")
    ax.axvline(0.7, color=COLOR_NEUTRAL, linestyle="--", linewidth=1.2, alpha=0.7)

    # Annotate zones
    ymax = ax.get_ylim()[1]
    ax.text(0.15, ymax * 0.9, "REAL\n(< 0.3)", ha="center",
            fontsize=11, color=COLOR_NEUTRAL, fontweight="bold")
    ax.text(0.50, ymax * 0.9, "SUSPICIOUS\n(0.3 – 0.7)", ha="center",
            fontsize=11, color=COLOR_NEUTRAL, fontweight="bold")
    ax.text(0.85, ymax * 0.9, "FAKE\n(> 0.7)", ha="center",
            fontsize=11, color=COLOR_NEUTRAL, fontweight="bold")

    ax.set_yscale("log")
    ax.set_xlabel("Predicted Spoof Probability")
    ax.set_ylabel("Number of Utterances (log scale)")
    ax.set_title(f"Spoof-Probability Distribution — ASVspoof 2019 LA (N={len(df):,})")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper center")
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================
# Plot 6: Speaker-level false-positive rate (bonafide only)
# ============================================================
def plot_speaker_fpr(df, out_path):
    """Per-speaker FPR for bonafide audio. Shows if errors concentrate on certain speakers."""
    real = df[df["label"] == "bonafide"].copy()
    if len(real) == 0 or "speaker_id" not in real.columns:
        print("  (skipping speaker FPR — no bonafide data)")
        return

    grp = real.groupby("speaker_id").agg(
        n=("utt_id", "count"),
        n_flagged=("verdict", lambda v: (v != "REAL").sum()),
    )
    grp["fpr_pct"] = grp["n_flagged"] / grp["n"] * 100
    grp = grp[grp["n"] >= 5].sort_values("fpr_pct", ascending=False)

    if len(grp) == 0:
        print("  (skipping speaker FPR — no speakers with >=5 utts)")
        return

    fig, ax = plt.subplots(figsize=(12, max(4, len(grp) * 0.18)))
    colors = [COLOR_FAKE if v > 5 else COLOR_REAL if v == 0 else COLOR_HIGHLIGHT
              for v in grp["fpr_pct"]]
    ax.barh(range(len(grp)), grp["fpr_pct"], color=colors,
            edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(grp)))
    ax.set_yticklabels(grp.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("False Positive Rate (%)")
    ax.set_title(f"Per-Speaker FPR on Bonafide Audio "
                 f"({len(grp)} speakers, ≥5 utts each)")
    ax.set_xlim(0, max(5, grp["fpr_pct"].max() * 1.1))
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================
# Metrics + failure cases
# ============================================================
def report_metrics(df):
    y_true = (df["label"] == "spoof").astype(int).values
    y_score = df["spoof_prob"].values

    eer_pct, eer_thr, fpr, tpr = compute_eer(y_true, y_score)

    pred_spoof = (df["verdict"] != "REAL").values
    correct = (pred_spoof == y_true.astype(bool)).sum()
    acc = correct / len(df) * 100

    real = df[df["label"] == "bonafide"]
    fake = df[df["label"] == "spoof"]
    fp_rate = (real["verdict"] != "REAL").sum() / len(real) * 100 if len(real) else 0
    fn_rate = (fake["verdict"] == "REAL").sum() / len(fake) * 100 if len(fake) else 0

    print("\n" + "=" * 60)
    print("METRICS")
    print("=" * 60)
    print(f"  Total utterances:         {len(df):,}")
    print(f"  Bonafide / Spoof:         {len(real):,} / {len(fake):,}")
    print(f"  EER:                      {eer_pct:.3f}% (threshold {eer_thr:.4f})")
    print(f"  Accuracy (thr=0.3):       {acc:.3f}%")
    print(f"  False Positive Rate:      {fp_rate:.3f}% (real flagged non-REAL)")
    print(f"  False Negative Rate:      {fn_rate:.3f}% (fake flagged REAL)")
    print("=" * 60)

    per_attack = {}
    print("\n  Per-attack acc (predicted ≠ REAL on spoof):")
    for a in ATTACK_ORDER:
        sub = df[df["attack_type"] == a]
        if not len(sub):
            continue
        a_correct = (sub["verdict"] != "REAL").sum()
        a_acc = a_correct / len(sub) * 100
        per_attack[a] = {"n": int(len(sub)), "correct": int(a_correct), "acc_pct": a_acc}
        print(f"    {a}: {a_correct:>5,}/{len(sub):>5,} = {a_acc:6.2f}%")

    return {
        "n": int(len(df)),
        "n_bonafide": int(len(real)),
        "n_spoof": int(len(fake)),
        "eer_pct": float(eer_pct),
        "eer_threshold": float(eer_thr),
        "accuracy_pct": float(acc),
        "fp_rate_pct": float(fp_rate),
        "fn_rate_pct": float(fn_rate),
        "per_attack": per_attack,
        "fpr": fpr,
        "tpr": tpr,
    }


def dump_failure_cases(df, out_path, top_n=20):
    real = df[df["label"] == "bonafide"].copy()
    real = real.sort_values("spoof_prob", ascending=False).head(top_n)

    fake = df[df["label"] == "spoof"].copy()
    fake = fake.sort_values("spoof_prob", ascending=True).head(top_n)

    with open(out_path, "w") as f:
        f.write("# Top Failure Cases\n\n")
        f.write(f"## False Positives (real → flagged as fake/suspicious), top {top_n}\n\n")
        f.write("| utt_id | speaker | spoof_prob | verdict | jitter | shimmer |\n")
        f.write("|--------|---------|-----------|---------|--------|----------|\n")
        for _, r in real.iterrows():
            f.write(f"| {r['utt_id']} | {r['speaker_id']} | {r['spoof_prob']:.4f} | "
                    f"{r['verdict']} | {r['jitter']:.4f} | {r['shimmer']:.4f} |\n")

        f.write(f"\n## False Negatives (fake → flagged as real), top {top_n}\n\n")
        f.write("| utt_id | attack | spoof_prob | verdict | jitter | shimmer |\n")
        f.write("|--------|--------|-----------|---------|--------|----------|\n")
        for _, r in fake.iterrows():
            f.write(f"| {r['utt_id']} | {r['attack_type']} | {r['spoof_prob']:.4f} | "
                    f"{r['verdict']} | {r['jitter']:.4f} | {r['shimmer']:.4f} |\n")
    print(f"  Saved: {out_path}")


def save_metrics_json(metrics, out_path):
    """Save machine-readable metrics (without heavy arrays)."""
    to_save = {k: v for k, v in metrics.items() if k not in ("fpr", "tpr")}
    with open(out_path, "w") as f:
        json.dump(to_save, f, indent=2)
    print(f"  Saved: {out_path}")


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

    metrics = report_metrics(df)

    plot_roc(metrics["fpr"], metrics["tpr"], metrics["eer_pct"],
             metrics["n"], os.path.join(out_dir, "roc.png"))
    plot_per_attack(df, os.path.join(out_dir, "per_attack_accuracy.png"))
    plot_confusion(df, os.path.join(out_dir, "confusion_matrix.png"))
    plot_feature_distributions(df, os.path.join(out_dir, "feature_distributions.png"))
    plot_spoof_prob_histogram(df, os.path.join(out_dir, "spoof_prob_histogram.png"))
    plot_speaker_fpr(df, os.path.join(out_dir, "speaker_fpr.png"))

    dump_failure_cases(df, os.path.join(out_dir, "failure_cases.md"))
    save_metrics_json(metrics, os.path.join(out_dir, "metrics.json"))

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
