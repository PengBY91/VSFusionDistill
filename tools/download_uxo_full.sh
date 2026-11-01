#!/usr/bin/env bash
# E11 prereq 1: download the full Dahn UXO dataset (Zenodo 13778485).
# aria2c with 8 range connections (a single curl connection stalled at
# ~30 MB; the storage host honours ranges, verified 2026-11-02), resume
# + retry + stall detection, md5 verification per file, .ok markers.
# Small packages first, then data_processed (the sample-format main
# package), then the two raw exports.  Run: bash tools/download_uxo_full.sh
# Log:   data/raw/uxo-full/download.log
set -uo pipefail
cd "$(dirname "$0")/.."
DEST=data/raw/uxo-full
mkdir -p "$DEST"
BASE=https://zenodo.org/api/records/13778485/files
# md5 checksums from the Zenodo record metadata (fetched 2026-11-01)
declare -A MD5=(
  [data_export_calibration.7z]=bb7641890899bf424e3eeb242509f261
  [data_export_3dmodels.7z]=4ed8820771bbb39819eacf66b50ff509
  [data_processed.7z]=2b9ef16c763a6964c608ed99a9cd1edb
  [data_export_polar.7z]=6e785ffd07bbab4f491794c2244df478
  [data_export_recordings.7z]=30c45956c0d034c27e3988711195c8c6
)
ORDER=(data_export_calibration.7z data_export_3dmodels.7z
       data_processed.7z data_export_polar.7z
       data_export_recordings.7z)
LOG="$DEST/download.log"

fetch() {  # fetch <file>
  local f=$1
  command -v aria2c >/dev/null 2>&1 || { echo "aria2c missing" >&2; return 9; }
  aria2c --continue=true --allow-overwrite=true --auto-file-renaming=false \
    --max-connection-per-server=16 --split=16 --min-split-size=16M \
    --max-tries=0 --retry-wait=30 --timeout=60 --connect-timeout=60 \
    --lowest-speed-limit=20K --file-allocation=none \
    --console-log-level=warn --summary-interval=300 \
    --dir "$DEST" --out "$f" "$BASE/$f/content" >> "$LOG" 2>&1
}

echo "[$(date '+%m-%d %H:%M')] download started (aria2, 16 connections)" >> "$LOG"
for f in "${ORDER[@]}"; do
  want=${MD5[$f]}
  if [ -f "$DEST/$f.ok" ]; then
    echo "[$(date '+%m-%d %H:%M')] $f already verified, skip" >> "$LOG"
    continue
  fi
  echo "[$(date '+%m-%d %H:%M')] fetching $f" >> "$LOG"
  ok=0
  for attempt in 1 2 3 4 5; do
    if fetch "$f"; then ok=1; break; fi
    echo "[$(date '+%m-%d %H:%M')] $f attempt $attempt failed (rc=$?), retrying" >> "$LOG"
    sleep 60
  done
  if [ "$ok" != 1 ]; then
    echo "[$(date '+%m-%d %H:%M')] $f GIVING UP after 5 attempts" >> "$LOG"
    continue
  fi
  got=$(md5sum "$DEST/$f" | cut -d' ' -f1)
  if [ "$got" = "$want" ]; then
    touch "$DEST/$f.ok"
    echo "[$(date '+%m-%d %H:%M')] $f OK md5=$got" >> "$LOG"
  else
    echo "[$(date '+%m-%d %H:%M')] $f MD5 MISMATCH got=$got want=$want (delete and rerun)" >> "$LOG"
  fi
done
echo "[$(date '+%m-%d %H:%M')] all done" >> "$LOG"
