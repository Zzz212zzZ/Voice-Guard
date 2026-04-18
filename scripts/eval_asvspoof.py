#!/usr/bin/env python3
"""ASVspoof 2019 LA eval benchmark — tiered pipeline.

Modes:
  sample     10 bonafide + 4 per attack type (62 files)   ~3 min   sanity check
  subsample  200 bonafide + 200 per attack (2800 files)   ~2-3 hr  main analysis
  full       all 71,237 trials                            ~15+ hr  (use --workers)

Each run writes one CSV row per utterance with W2V-AASIST verdict +
acoustic features. Resumable: skips utterances already in output CSV.

Usage:
    python scripts/eval_asvspoof.py --tier sample
    python scripts/eval_asvspoof.py --tier subsample
    python scripts/eval_asvspoof.py --tier full --workers 8 --resume
"""

import argparse
import csv
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modules.shield.detect import DeepfakeDetector
from modules.shield.spectral import analyze_spectral
from modules.shield.prosody import analyze_prosody

# -------------------------------------------------------------------
# Paths (resolved via symlinks in worktree root)
# -------------------------------------------------------------------
PROTOCOL_PATH = os.path.join(
    ROOT, "SSL_Anti-spoofing", "database", "ASVspoof_LA_cm_protocols",
    "ASVspoof2019.LA.cm.eval.trl.txt",
)
AUDIO_DIR = os.path.join(ROOT, "data", "LA", "ASVspoof2019_LA_eval", "flac")
OUT_DIR = os.path.join(ROOT, "scripts", "output", "asvspoof_eval")

TIERS = {
    "sample":    {"bonafide": 10,   "per_attack": 4},
    "subsample": {"bonafide": 200,  "per_attack": 200},
    "full":      None,
}

CSV_COLS = [
    "utt_id", "speaker_id", "attack_type", "label",
    "spoof_prob", "bonafide_prob", "verdict", "confidence",
    "jitter", "shimmer", "f0_mean", "f0_std", "f0_cv", "energy_cv",
    "spectral_centroid", "spectral_bandwidth", "spectral_flatness", "spectral_rolloff",
    "inference_time_s",
]


def load_protocol():
    """Parse ASVspoof2019 LA eval protocol.

    Line format: `speaker_id utt_id - attack_type label`
    attack_type is '-' for bonafide.
    """
    rows = []
    with open(PROTOCOL_PATH) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            speaker, utt, _, attack, label = parts
            rows.append({
                "speaker_id": speaker,
                "utt_id": utt,
                "attack_type": "bonafide" if attack == "-" else attack,
                "label": label,
            })
    return rows


def stratified_sample(rows, tier_cfg, seed=42):
    """Sample N bonafide + M per attack type. Returns list preserving protocol order."""
    by_group = defaultdict(list)
    for r in rows:
        by_group[r["attack_type"]].append(r)

    rng = random.Random(seed)
    selected = set()
    for group, items in by_group.items():
        n = tier_cfg["bonafide"] if group == "bonafide" else tier_cfg["per_attack"]
        sampled = rng.sample(items, min(n, len(items)))
        for r in sampled:
            selected.add(r["utt_id"])

    return [r for r in rows if r["utt_id"] in selected]


def load_existing_results(csv_path):
    """Return set of utt_ids already in the CSV (for resume)."""
    if not os.path.isfile(csv_path):
        return set()
    done = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(row["utt_id"])
    return done


def process_utterance(detector, row):
    """Run detector + features on one utterance. Returns CSV row dict or None on error."""
    audio_path = os.path.join(AUDIO_DIR, row["utt_id"] + ".flac")
    if not os.path.isfile(audio_path):
        return None

    t0 = time.time()
    try:
        w2v = detector.detect(audio_path)
        prosody = analyze_prosody(audio_path)
        spectral = analyze_spectral(audio_path)
    except Exception as e:
        print(f"  ERR {row['utt_id']}: {e}", flush=True)
        return None
    elapsed = time.time() - t0

    return {
        "utt_id": row["utt_id"],
        "speaker_id": row["speaker_id"],
        "attack_type": row["attack_type"],
        "label": row["label"],
        "spoof_prob": w2v["spoof_prob"],
        "bonafide_prob": w2v["bonafide_prob"],
        "verdict": w2v["verdict"],
        "confidence": w2v["confidence"],
        "jitter": prosody["jitter"],
        "shimmer": prosody["shimmer"],
        "f0_mean": prosody["f0_mean"],
        "f0_std": prosody["f0_std"],
        "f0_cv": prosody["f0_cv"],
        "energy_cv": prosody["energy_cv"],
        "spectral_centroid": spectral["spectral_centroid"],
        "spectral_bandwidth": spectral["spectral_bandwidth"],
        "spectral_flatness": spectral["spectral_flatness"],
        "spectral_rolloff": spectral["spectral_rolloff"],
        "inference_time_s": round(elapsed, 3),
    }


# -------------------------------------------------------------------
# Multiprocessing worker (spawn-compatible)
# -------------------------------------------------------------------
_WORKER_DETECTOR = None


def _init_worker(ssl_root, ckpt_root, device):
    """Initializer: runs once per worker process at startup.

    Under macOS spawn method, workers re-import this module and run the
    initializer. Heavy model loading happens here, not per task.
    """
    global _WORKER_DETECTOR
    # Ensure project root is on path for module imports
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from modules.shield.detect import DeepfakeDetector
    _WORKER_DETECTOR = DeepfakeDetector(ssl_root, ckpt_root, device)


def _worker_process(row):
    """Task function: process one utterance using the worker-local detector."""
    return process_utterance(_WORKER_DETECTOR, row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=TIERS.keys(), required=True)
    parser.add_argument("--resume", action="store_true",
                        help="Skip utterances already in output CSV")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cpu, mps, cuda:0)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Hard cap on number of new utterances to process")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes (default 1 = sequential). "
                             "Use 8 on M4 Pro for ~5x speedup.")
    parser.add_argument("--progress-every", type=int, default=None,
                        help="Print progress every N utterances (default: 25 seq / 100 parallel)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, f"eval_{args.tier}.csv")

    # ---- Load + sample ----
    print(f"Loading protocol: {PROTOCOL_PATH}")
    all_rows = load_protocol()
    print(f"  Total trials: {len(all_rows)}")

    attack_counts = Counter(r["attack_type"] for r in all_rows)
    print(f"  Groups: {dict(attack_counts)}")

    if args.tier == "full":
        target_rows = all_rows
    else:
        target_rows = stratified_sample(all_rows, TIERS[args.tier], seed=args.seed)
    print(f"  Selected for tier '{args.tier}': {len(target_rows)}")

    # ---- Resume ----
    already_done = load_existing_results(out_csv) if args.resume else set()
    if already_done:
        print(f"  Resuming — {len(already_done)} already in {out_csv}")
        target_rows = [r for r in target_rows if r["utt_id"] not in already_done]

    if args.limit:
        target_rows = target_rows[:args.limit]
    print(f"  To process: {len(target_rows)}")

    if not target_rows:
        print("Nothing to do.")
        return

    # ---- Sanity: audio dir ----
    if not os.path.isdir(AUDIO_DIR):
        print(f"\n❌ Audio directory missing: {AUDIO_DIR}")
        print("   Download first: LA.zip from datashare.ed.ac.uk/handle/10283/3336")
        sys.exit(1)

    # ---- CSV writer (append mode if resuming) ----
    write_header = not os.path.isfile(out_csv) or not args.resume
    mode = "a" if args.resume and os.path.isfile(out_csv) else "w"
    f_out = open(out_csv, mode, newline="")
    writer = csv.DictWriter(f_out, fieldnames=CSV_COLS)
    if write_header:
        writer.writeheader()

    ssl_root = os.path.join(ROOT, "SSL_Anti-spoofing")
    ckpt_root = os.path.join(ROOT, "checkpoints")

    # ---- Main loop ----
    start = time.time()
    n_ok = 0
    n_err = 0

    def _print_progress(i, row, result, rate_window_s=None):
        elapsed = time.time() - start
        rate = i / elapsed if elapsed > 0 else 0
        eta = (len(target_rows) - i) / rate if rate > 0 else 0
        correct = (
            (result["verdict"] != "REAL" and row["label"] == "spoof") or
            (result["verdict"] == "REAL" and row["label"] == "bonafide")
        )
        print(f"  [{i}/{len(target_rows)}] {row['utt_id']} "
              f"{row['attack_type']:8s} {row['label']:8s} "
              f"verdict={result['verdict']:10s} "
              f"p={result['spoof_prob']:.3f} "
              f"{'OK ' if correct else '!! '}"
              f"({rate:.2f} utt/s, ETA {eta/60:.1f}min)",
              flush=True)

    if args.workers <= 1:
        # ---- Sequential path ----
        print("\nLoading W2V-AASIST (single process)...")
        detector = DeepfakeDetector(
            ssl_root=ssl_root,
            checkpoints_root=ckpt_root,
            device=args.device,
        )
        progress_every = args.progress_every or 25
        for i, row in enumerate(target_rows, 1):
            result = process_utterance(detector, row)
            if result is None:
                n_err += 1
                continue
            writer.writerow(result)
            f_out.flush()
            n_ok += 1
            if i % progress_every == 0 or i == len(target_rows):
                _print_progress(i, row, result)
    else:
        # ---- Parallel path ----
        print(f"\nStarting {args.workers} worker processes "
              f"(each loads its own W2V-AASIST; ~30 s warmup)...",
              flush=True)
        progress_every = args.progress_every or 100

        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(ssl_root, ckpt_root, args.device),
        ) as ex:
            # Submit all tasks; workers pick them up as they become free
            future_to_row = {
                ex.submit(_worker_process, row): row
                for row in target_rows
            }
            completed = 0
            for future in as_completed(future_to_row):
                row = future_to_row[future]
                completed += 1
                try:
                    result = future.result()
                except Exception as e:
                    print(f"  WORKER_EXC {row['utt_id']}: {e}", flush=True)
                    n_err += 1
                    continue
                if result is None:
                    n_err += 1
                    continue
                writer.writerow(result)
                f_out.flush()
                n_ok += 1
                if completed % progress_every == 0 or completed == len(target_rows):
                    _print_progress(completed, row, result)

    f_out.close()
    elapsed = time.time() - start
    print(f"\nDone. {n_ok} ok, {n_err} err, {elapsed/60:.1f} min total "
          f"(rate: {len(target_rows)/elapsed:.2f} utt/s).")
    print(f"Output: {out_csv}")


if __name__ == "__main__":
    main()
