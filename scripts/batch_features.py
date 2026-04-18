#!/usr/bin/env python3
"""Compare acoustic features between real and fake (cloned) audio samples.

Generates side-by-side box plots and summary statistics for the
Explainability challenge analysis presentation.

Usage:
    conda activate openvoice
    python scripts/batch_features.py
"""

import os
import sys
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modules.shield.spectral import analyze_spectral
from modules.shield.prosody import analyze_prosody

# -------------------------------------------------------------------
# Audio sample paths
# -------------------------------------------------------------------
FAKE_DIR = os.path.join(ROOT, "OpenVoice", "outputs_v2")
OUTPUT_DIR = os.path.join(ROOT, "scripts", "output")

# Curated real audio samples (verified authentic)
REAL_FILES = [
    os.path.join(ROOT, "OpenVoice", "resources", "demo_speaker0.mp3"),
    os.path.join(ROOT, "OpenVoice", "resources", "demo_speaker2.mp3"),
    os.path.join(ROOT, "OpenVoice", "resources", "my_reference_audio.m4a"),
]

FAKE_EXTS = {".wav"}


def collect_fake_files(directory, extensions):
    files = []
    for f in sorted(os.listdir(directory)):
        if os.path.splitext(f)[1].lower() in extensions:
            files.append(os.path.join(directory, f))
    return files


def extract_features(file_list, label):
    rows = []
    for path in file_list:
        fname = os.path.basename(path)
        print(f"  [{label}] {fname} ... ", end="", flush=True)
        try:
            spec = analyze_spectral(path)
            pros = analyze_prosody(path)
            row = {
                "file": fname,
                "label": label,
                "jitter": pros["jitter"],
                "shimmer": pros["shimmer"],
                "f0_mean": pros["f0_mean"],
                "f0_std": pros["f0_std"],
                "f0_cv": pros["f0_cv"],
                "energy_cv": pros["energy_cv"],
                "spectral_centroid": spec["spectral_centroid"],
                "spectral_bandwidth": spec["spectral_bandwidth"],
                "spectral_flatness": spec["spectral_flatness"],
                "spectral_rolloff": spec["spectral_rolloff"],
            }
            rows.append(row)
            print("OK")
        except Exception as e:
            print(f"SKIP ({e})")
    return rows


def plot_comparison(df, features, title, filename):
    n = len(features)
    cols = min(3, n)
    rows_count = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_count, cols, figsize=(5 * cols, 4 * rows_count))
    if rows_count == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i, feat in enumerate(features):
        ax = axes[i]
        real_vals = df[df["label"] == "Real"][feat].values
        fake_vals = df[df["label"] == "Fake"][feat].values

        positions = [1, 2]
        bp = ax.boxplot(
            [real_vals, fake_vals], positions=positions, widths=0.5,
            patch_artist=True, labels=["Real", "Fake"],
        )
        bp["boxes"][0].set_facecolor("#72B7B2")
        bp["boxes"][1].set_facecolor("#E45756")

        # Overlay individual points
        for j, (vals, pos) in enumerate([(real_vals, 1), (fake_vals, 2)]):
            jitter = np.random.normal(0, 0.05, size=len(vals))
            ax.scatter(pos + jitter, vals, alpha=0.6, s=20, zorder=3,
                       color="#4C78A8" if j == 0 else "#F58518")

        ax.set_title(feat.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.3)

    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    print(f"  Saved: {os.path.join(OUTPUT_DIR, filename)}")
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    real_files = REAL_FILES
    fake_files = collect_fake_files(FAKE_DIR, FAKE_EXTS)

    print(f"Real samples: {len(real_files)}")
    print(f"Fake samples: {len(fake_files)}")

    print("\nExtracting features...")
    all_rows = extract_features(real_files, "Real") + extract_features(fake_files, "Fake")

    df = pd.DataFrame(all_rows)

    # Save raw data
    csv_path = os.path.join(OUTPUT_DIR, "feature_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw data: {csv_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("FEATURE COMPARISON: REAL vs FAKE")
    print("=" * 60)
    for feat in ["jitter", "shimmer", "f0_cv", "energy_cv",
                  "spectral_centroid", "spectral_flatness"]:
        real_mean = df[df["label"] == "Real"][feat].mean()
        fake_mean = df[df["label"] == "Fake"][feat].mean()
        diff_pct = ((fake_mean - real_mean) / real_mean * 100) if real_mean != 0 else 0
        print(f"  {feat:>20s}:  Real={real_mean:.4f}  Fake={fake_mean:.4f}  "
              f"({diff_pct:+.1f}%)")
    print("=" * 60)

    # Generate plots
    print("\nGenerating comparison plots...")
    plot_comparison(
        df,
        ["jitter", "shimmer", "f0_cv", "energy_cv"],
        "Voice Quality: Real vs Fake",
        "voice_quality_comparison.png",
    )
    plot_comparison(
        df,
        ["spectral_centroid", "spectral_bandwidth", "spectral_flatness", "spectral_rolloff"],
        "Spectral Features: Real vs Fake",
        "spectral_comparison.png",
    )
    plot_comparison(
        df,
        ["f0_mean", "f0_std"],
        "Pitch (F0): Real vs Fake",
        "pitch_comparison.png",
    )

    print("\nDone! Charts saved to scripts/output/")


if __name__ == "__main__":
    main()
