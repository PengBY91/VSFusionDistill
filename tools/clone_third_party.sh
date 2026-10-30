#!/usr/bin/env bash
# Clone read-only reference repos into third_party/ (setup doc §3.2).
# They are NEVER imported by aodet/ — only consulted as reading material.
#
# Pins: first run clones default branches and records SHAs in
# third_party/PINNED.md. Later runs re-checkout the pinned SHAs, so the
# references stay frozen even as upstream moves.
set -euo pipefail
cd "$(dirname "$0")/.."

declare -A REPOS=(
  [KD-YOLOX-ViT]=https://github.com/remaro-network/KD-YOLOX-ViT
  [YOLOX]=https://github.com/Megvii-BaseDetection/YOLOX
  [uxo-dataset2024]=https://github.com/dfki-ric/uxo-dataset2024
  [ProbEn]=https://github.com/Jamie725/Multimodal-Object-Detection-via-Probabilistic-Ensembling
  [CRKD]=https://github.com/Song-Jingyu/CRKD
  [SCFusion]=https://github.com/NWPU-ASVLab/SCFusion
)
PINNED=third_party/PINNED.md

mkdir -p third_party
for name in "${!REPOS[@]}"; do
  dir="third_party/$name"
  if [ ! -d "$dir/.git" ]; then
    echo ">> cloning $name"
    git clone --filter=blob:none --quiet "${REPOS[$name]}" "$dir"
  fi
  pinned_sha=""
  if [ -f "$PINNED" ]; then
    pinned_sha=$(awk -v n="$name" '$2==n{print $1}' "$PINNED" | head -1)
  fi
  if [ -n "$pinned_sha" ]; then
    git -C "$dir" fetch --quiet origin "$pinned_sha" 2>/dev/null || true
    git -C "$dir" checkout --quiet "$pinned_sha"
    echo "   $name @ $pinned_sha (pinned)"
  else
    sha=$(git -C "$dir" rev-parse HEAD)
    echo "$sha $name" >> "$PINNED"
    echo "   $name @ $sha (newly pinned)"
  fi
done
echo "done. pins in third_party/PINNED.md"
