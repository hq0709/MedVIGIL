#!/usr/bin/env bash
# Loop one API model across all sigma levels for the visual-decay ablation.
# Usage: bash scripts/run_blur_sweep_one_model.sh <model_id> [concurrency]
set -u
MODEL="${1:?model_id required}"
CONC="${2:-8}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs/blur_sweep
LOG="logs/blur_sweep/$(echo "$MODEL" | tr '/' '_')_$(date +%Y%m%d_%H%M%S).log"

echo "[start] $(date) model=$MODEL conc=$CONC log=$LOG"
for sigma in 0 2 4 8 16 32 64 inf; do
  echo "------------- $(date '+%H:%M:%S') sigma=$sigma -------------"
  python3 scripts/bench_blur_sweep_run.py --model "$MODEL" --sigma "$sigma" --concurrency "$CONC"
done 2>&1 | tee -a "$LOG"
echo "[done] $(date) model=$MODEL"
