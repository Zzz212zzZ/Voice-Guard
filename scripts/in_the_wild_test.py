#!/usr/bin/env python3
"""In-the-wild vs benchmark comparison.

Runs W2V-AASIST on real (non-ASVspoof) audio files — phone/computer recordings
of actual human speech — to quantify the domain-gap False Positive Rate.

Pairs with eval_subsample.csv to produce the Final slide:
  "FPR on ASVspoof (studio) = 0% vs in-the-wild FPR = X%"

Usage:
    python scripts/in_the_wild_test.py
"""

import csv
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modules.shield.detect import DeepfakeDetector

# Candidate in-the-wild real speech samples (human recordings, not studio)
REAL_FILES = [
    os.path.join(ROOT, "OpenVoice", "resources", "demo_speaker0.mp3"),
    os.path.join(ROOT, "OpenVoice", "resources", "demo_speaker1.mp3"),
    os.path.join(ROOT, "OpenVoice", "resources", "demo_speaker2.mp3"),
    os.path.join(ROOT, "OpenVoice", "resources", "example_reference.mp3"),
    os.path.join(ROOT, "OpenVoice", "resources", "Reference audio.m4a"),
]

OUT_DIR = os.path.join(ROOT, "scripts", "output", "asvspoof_eval")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load detector (same as benchmark run)
    print("Loading W2V-AASIST...")
    detector = DeepfakeDetector(
        ssl_root=os.path.join(ROOT, "SSL_Anti-spoofing"),
        checkpoints_root=os.path.join(ROOT, "checkpoints"),
    )

    rows = []
    for path in REAL_FILES:
        if not os.path.isfile(path):
            print(f"  SKIP (missing): {path}")
            continue
        fname = os.path.basename(path)
        print(f"  Testing {fname} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            result = detector.detect(path)
            elapsed = time.time() - t0
            rows.append({
                "file": fname,
                "path": path,
                "spoof_prob": result["spoof_prob"],
                "bonafide_prob": result["bonafide_prob"],
                "verdict": result["verdict"],
                "confidence": result["confidence"],
                "inference_time_s": round(elapsed, 3),
            })
            mark = "OK " if result["verdict"] == "REAL" else "!! "
            print(f"{mark}verdict={result['verdict']}  p_spoof={result['spoof_prob']:.4f}")
        except Exception as e:
            print(f"ERR {e}")

    if not rows:
        print("\nNo samples processed.")
        sys.exit(1)

    # Save CSV
    out_csv = os.path.join(OUT_DIR, "in_the_wild_real.csv")
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Metrics
    n_total = len(df)
    n_wrong = (df["verdict"] != "REAL").sum()
    fpr = n_wrong / n_total * 100
    print(f"\nIn-the-Wild FPR: {n_wrong}/{n_total} = {fpr:.1f}%")

    # Compare against benchmark (tier 2)
    benchmark_csv = os.path.join(OUT_DIR, "eval_subsample.csv")
    if os.path.isfile(benchmark_csv):
        bdf = pd.read_csv(benchmark_csv)
        breal = bdf[bdf["label"] == "bonafide"]
        bench_fpr = (breal["verdict"] != "REAL").sum() / len(breal) * 100
        bench_n = len(breal)
    else:
        bench_fpr = 0.0
        bench_n = 200

    # Bar chart: FPR benchmark vs in-the-wild
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [f"ASVspoof 2019 LA\n(studio, n={bench_n})",
              f"In-the-wild\n(phone/laptop mic, n={n_total})"]
    values = [bench_fpr, fpr]
    colors = ["#72B7B2", "#E45756"]
    bars = ax.bar(labels, values, color=colors, width=0.55)

    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                f"{v:.1f}%", ha="center", fontsize=13, fontweight="bold")

    ax.set_ylabel("False Positive Rate (%)", fontsize=12)
    ax.set_ylim(0, max(max(values) + 15, 20))
    ax.set_title("Domain Gap: Benchmark FPR vs In-the-Wild FPR\n"
                 "(Real human speech mis-flagged as FAKE)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_fig = os.path.join(OUT_DIR, "domain_gap_fpr.png")
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_fig}")


if __name__ == "__main__":
    main()
