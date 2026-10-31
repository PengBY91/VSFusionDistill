#!/usr/bin/env bash
# Recording-level leave-one-out protocol (review 2026-10-31 §4.3).
#
#   tools/run_loocv.sh <config> [seed]
#
# The sample has 5 recordings; a fixed train/test split leaves 2 test
# recordings and invites tuning on them (E05-E08 did).  Here each fold
# holds one recording out as the evaluation set and trains on the other
# four (split filter disabled, recording filters on).  Report the
# per-fold numbers and their mean / range; the recording list must match
# the ``recording`` column of frame_meta.csv.
set -euo pipefail

CONFIG=${1:?usage: run_loocv.sh <config> [seed]}
SEED=${2:-0}
REPO=$(cd "$(dirname "$0")/.." && pwd)
RECORDINGS=${RECORDINGS:-"2023-09-20_180722 2023-09-21_100927 2023-09-21_105856 2023-09-21_123500 2023-09-21_124735"}
LEVELS="d0 d1 d2 d3 d4 dinf"
name=$(basename "$CONFIG" .py)

for rec in $RECORDINGS; do
  work="$REPO/work_dirs/${name}_loo_${rec}_seed${SEED}"
  echo "=== [$name fold $rec seed $SEED] train ==="
  python "$REPO/tools/train.py" "$CONFIG" --work-dir "$work" --seed "$SEED" \
    --cfg-options train_dataloader.dataset.split=None \
                  "train_dataloader.dataset.exclude_recordings=['$rec']"
  ckpt=$(cat "$work/last_checkpoint")
  [[ -f "$ckpt" ]] || ckpt="$work/$ckpt"
  echo "=== [$name fold $rec seed $SEED] scan D0-Dinf ==="
  python "$REPO/tools/test.py" "$CONFIG" "$ckpt" --out-dir "$work" \
    --levels $LEVELS \
    --cfg-options test_dataloader.dataset.split=None \
                  "test_dataloader.dataset.recordings=['$rec']"
  python "$REPO/tools/correlation_table.py" "$work" --split all \
    --recordings "$rec" --paired "$REPO/data/paired/uxo"
done
