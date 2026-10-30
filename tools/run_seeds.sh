#!/usr/bin/env bash
# Multi-seed train + D0–D∞ scan + results collection (F2 §6: every method
# 3 seeds unless stated otherwise).
#
#   tools/run_seeds.sh configs/f2/s0_cross_attn_yoloxs.py [seed ...]
#
# Each seed trains into work_dirs/<config>_seed<seed>/ (tools/train.py
# stamps run_meta.txt), then tools/test.py sweeps levels and writes
# pred.<branch>.<level>.json + ap_summary.json into the same directory.
# tools/collect_results.py merges every run into results/results.csv.
set -euo pipefail

CONFIG=${1:?usage: run_seeds.sh <config> [seed ...]}
shift
SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
  SEEDS=(0 1 2)
fi

LEVELS="d0 d1 d2 d3 d4 dinf"
REPO=$(cd "$(dirname "$0")/.." && pwd)
RESULTS=${RESULTS:-$REPO/results/results.csv}

for seed in "${SEEDS[@]}"; do
  name=$(basename "$CONFIG" .py)
  work="$REPO/work_dirs/${name}_seed${seed}"
  echo "=== [$name seed $seed] train ==="
  python "$REPO/tools/train.py" "$CONFIG" --work-dir "$work" --seed "$seed"
  echo "=== [$name seed $seed] scan D0-Dinf ==="
  python "$REPO/tools/test.py" "$CONFIG" "$work/latest.pth" \
    --out-dir "$work" --levels $LEVELS
done

python "$REPO/tools/collect_results.py" "$REPO/work_dirs" --out "$RESULTS"
echo "results -> $RESULTS"
