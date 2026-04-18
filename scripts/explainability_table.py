#!/usr/bin/env python3
"""Per-sample verdict + features table and correlation scatter plot.

Shows the 'explainability gap': features correlate with the model's verdict
but aren't formally connected to it.

Usage:
    conda activate openvoice
    python scripts/explainability_table.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modules.shield.spectral import analyze_spectral
from modules.shield.prosody import analyze_prosody

OUTPUT_DIR = os.path.join(ROOT, "scripts", "output")

# Same samples as batch_features.py
REAL_FILES = [
    os.path.join(ROOT, "OpenVoice", "resources", "demo_speaker0.mp3"),
    os.path.join(ROOT, "OpenVoice", "resources", "demo_speaker2.mp3"),
    os.path.join(ROOT, "OpenVoice", "resources", "my_reference_audio.m4a"),
]

FAKE_DIR = os.path.join(ROOT, "OpenVoice", "outputs_v2")


def collect_fake_files():
    files = []
    for f in sorted(os.listdir(FAKE_DIR)):
        if f.endswith(".wav"):
            files.append(os.path.join(FAKE_DIR, f))
    return files


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Import detector (heavy — loads model)
    from modules.shield.detect import DeepfakeDetector
    detector = DeepfakeDetector(
        ssl_root=os.path.join(ROOT, "SSL_Anti-spoofing"),
        checkpoints_root=os.path.join(ROOT, "checkpoints"),
    )

    all_files = [(f, "Real") for f in REAL_FILES] + [(f, "Fake") for f in collect_fake_files()]

    rows = []
    for path, label in all_files:
        fname = os.path.basename(path)
        print(f"  [{label}] {fname} ... ", end="", flush=True)
        try:
            w2v = detector.detect(path)
            pros = analyze_prosody(path)
            spec = analyze_spectral(path)
            rows.append({
                "file": fname,
                "label": label,
                "verdict": w2v["verdict"],
                "spoof_prob": w2v["spoof_prob"],
                "confidence": w2v["confidence"],
                "shimmer": pros["shimmer"],
                "jitter": pros["jitter"],
                "f0_cv": pros["f0_cv"],
                "energy_cv": pros["energy_cv"],
                "spectral_centroid": spec["spectral_centroid"],
            })
            print(f"{w2v['verdict']} ({w2v['spoof_prob']:.4f})")
        except Exception as e:
            print(f"SKIP ({e})")

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "per_sample_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Print markdown table
    print("\n### Per-Sample Explainability Table\n")
    print("| File | Label | Verdict | Spoof Prob | Shimmer | Energy CV | F0 CV |")
    print("|------|-------|---------|-----------|---------|-----------|-------|")
    for _, r in df.iterrows():
        print(f"| {r['file'][:25]} | {r['label']} | {r['verdict']} | {r['spoof_prob']:.4f} | "
              f"{r['shimmer']:.4f} | {r['energy_cv']:.4f} | {r['f0_cv']:.4f} |")

    # --- Scatter plot: shimmer vs spoof_prob ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    features_to_plot = [
        ("shimmer", "Shimmer (Amplitude Perturbation)"),
        ("energy_cv", "Energy CV (Dynamic Range)"),
        ("f0_cv", "F0 CV (Pitch Variability)"),
    ]

    for ax, (feat, title) in zip(axes, features_to_plot):
        real = df[df["label"] == "Real"]
        fake = df[df["label"] == "Fake"]

        ax.scatter(real[feat], real["spoof_prob"], c="#72B7B2", s=80,
                   edgecolors="black", linewidth=0.5, label="Real", zorder=3)
        ax.scatter(fake[feat], fake["spoof_prob"], c="#E45756", s=80,
                   edgecolors="black", linewidth=0.5, label="Fake", zorder=3)

        ax.axhline(y=0.7, color="gray", linestyle="--", alpha=0.5, label="FAKE threshold (0.7)")
        ax.axhline(y=0.3, color="gray", linestyle=":", alpha=0.5, label="SUSPICIOUS threshold (0.3)")

        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel("Spoof Probability (W2V-AASIST)", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, loc="center right")
        ax.grid(alpha=0.3)

    fig.suptitle("Feature Values vs Model Verdict — Correlation but Not Causation",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "feature_vs_verdict.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close(fig)

    print("\nDone!")


if __name__ == "__main__":
    main()
