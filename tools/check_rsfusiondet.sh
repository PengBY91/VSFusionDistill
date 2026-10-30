#!/usr/bin/env bash
# Poll the RSFusionDet official repo for published code/data (F2 TODO 14.2:
# check every two weeks until the authors release). Prints one line per run;
# callers append to doc/logs/rsfusiondet-watch.log.
set -euo pipefail
resp=$(curl -s -m 30 https://api.github.com/repos/LEFTeyex/RSFusionDet)
size=$(printf '%s' "$resp" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("size", -1))')
pushed=$(printf '%s' "$resp" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("pushed_at", "?"))')
# An empty repo makes /commits return a 409 error *object*, not an array —
# check the type, not just the length.
n_commits=$(curl -s -m 30 "https://api.github.com/repos/LEFTeyex/RSFusionDet/commits?per_page=1" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d) if isinstance(d, list) else "error")')
echo "$(date -Is) size=$size pushed_at=$pushed commits_page1=$n_commits"
if [ "$n_commits" != "0" ] && [ "$n_commits" != "error" ]; then
  echo "!! RSFusionDet has commits — record the head commit and data download instructions in the F2 draft TODO."
fi
