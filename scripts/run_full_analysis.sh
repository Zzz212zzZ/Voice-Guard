#!/bin/bash
# One-shot: runs all analyses on the Tier 3 full-eval CSV.
# Use after Tier 3 completes (eval_full.csv has all 71,237 rows).

set -e

cd "$(dirname "$0")/.."

PYTHON=/opt/anaconda3/envs/openvoice/bin/python

echo "=== Analyze metrics + core charts ==="
"$PYTHON" scripts/analyze_eval.py --tier full

echo ""
echo "=== Threshold sweep ==="
"$PYTHON" scripts/threshold_sweep.py --tier full

echo ""
echo "=== Generate slide update text (drop into final_presentation_draft.md) ==="
"$PYTHON" scripts/generate_slide_updates.py --tier full

echo ""
echo "=== Done. Outputs: ==="
ls -la scripts/output/asvspoof_eval/full/
