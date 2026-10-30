#!/usr/bin/env bash
# Debug-machine (CPU-only) environment install for VSFusionDistill / "ao-det".
#
# The plan (doc/topic-f2-codebase-selection-and-setup-2026-10.md §4.1) pins
# torch 2.1.2 / mmcv 2.1.0 / mmengine 0.10.7 / mmdet 3.3.0 / numpy 1.26 with
# CUDA 12.1 for the *training* machine. This box has no visible GPU, so we
# install the CPU builds of the same versions; the cu101..cu121-compatible
# training variant is recorded in env/environment.yml.
#
# Evidence produced by this script (committed): env/pip-freeze.txt,
# lines appended to env/ENVIRONMENT.md (versions + pip check).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

$PY -m pip install --upgrade pip setuptools wheel
# torch 2.1.2 + matching torchvision, CPU wheels
$PY -m pip install torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cpu
# mmcv 2.1.0 prebuilt CPU wheel for torch 2.1.x
$PY -m pip install mmcv==2.1.0 \
  -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html
# framework + pinned deps
$PY -m pip install mmengine==0.10.7 mmdet==3.3.0 "numpy==1.26.4" \
  pycocotools opencv-python-headless scipy pandas pytest matplotlib \
  terminaltables tqdm

{
  echo "### pip check ($(date -Is))"
  $PY -m pip check || true
} >> env/ENVIRONMENT.md

$PY -m pip freeze > env/pip-freeze.txt
echo "OK: environment installed. See env/pip-freeze.txt"
