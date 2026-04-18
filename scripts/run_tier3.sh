#!/bin/bash
# Tier 3 launcher — runs full ASVspoof 2019 LA eval.
# Usage:
#   bash scripts/run_tier3.sh          # default 6 workers
#   bash scripts/run_tier3.sh 4        # 4 workers
#   bash scripts/run_tier3.sh 8        # 8 workers
#
# Log goes to scripts/output/asvspoof_eval/tier3.log inside the repo.
# Resumable: skips utterances already in eval_full.csv.

set -e

WORKERS="${1:-6}"

# cd to worktree root (this script is in scripts/, so ../ is root)
cd "$(dirname "$0")/.."

PYTHON=/opt/anaconda3/envs/openvoice/bin/python
LOG=scripts/output/asvspoof_eval/tier3.log

# Kick off in background, survive terminal close, redirect stdout+stderr.
nohup "$PYTHON" scripts/eval_asvspoof.py \
    --tier full \
    --workers "$WORKERS" \
    --resume \
    > "$LOG" 2>&1 &

PID=$!
echo "Tier 3 started. PID=$PID  workers=$WORKERS"
echo "Log: $(pwd)/$LOG"
echo ""
echo "Monitor:"
echo "  tail -f $LOG"
echo "  wc -l scripts/output/asvspoof_eval/eval_full.csv  # target: 71238"
echo ""
echo "Kill:"
echo "  kill $PID"
echo "  # or: pkill -f eval_asvspoof"
echo ""
echo "Restart with different workers (will resume where left off):"
echo "  bash scripts/run_tier3.sh 4    # switch to 4 workers"
echo "  bash scripts/run_tier3.sh 8    # switch to 8 workers"
