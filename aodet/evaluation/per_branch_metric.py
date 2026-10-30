"""Per-branch, per-degradation-level COCO AP (F2 §2.3).

Evaluation reads *saved prediction files* — one JSON per (method, run,
branch, level) — so the same predictions can be re-analysed without
re-inference (setup doc §4.4(6)).  This module wraps pycocotools; the
"official example cross-check" test runs COCOeval directly on the same toy
data and compares numbers.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    _HAVE_PYCOCO = True
except ImportError:  # pragma: no cover
    _HAVE_PYCOCO = False

__all__ = [
    "BRANCHES",
    "coco_ap",
    "evaluate_prediction_file",
    "collect_ap_table",
]

# The three output heads of the two-stream student (F2 §5.1), plus the
# camera branch name used by single-stream baselines.
BRANCHES = ("camera", "sonar", "sonar_uncoupled")


def _require_pycoco() -> None:
    if not _HAVE_PYCOCO:
        raise ImportError("pycocotools is required for per-branch metrics")


def coco_ap(
    gt: Dict[str, Any],
    preds: Sequence[Dict[str, Any]],
    img_ids: Optional[Iterable[int]] = None,
) -> Dict[str, float]:
    """COCO AP/AR summary for one branch on one level.

    Args:
        gt: COCO-format ground-truth dict (or a path to one).
        preds: COCO-format result list [{image_id, category_id, bbox,
            score}, ...] (or a path to one).
        img_ids: restrict evaluation to these image ids (e.g. the test
            split's ids).
    """
    _require_pycoco()
    if isinstance(gt, str):
        with open(gt, encoding="utf-8") as f:
            gt = json.load(f)
    if isinstance(preds, str):
        with open(preds, encoding="utf-8") as f:
            preds = json.load(f)

    coco = COCO()
    coco.dataset = gt
    coco.createIndex()
    if preds:
        dt = coco.loadRes(preds)
    else:
        # empty prediction file: "no detections" -> AP 0, not a crash
        dt = COCO()
        dt.dataset = {"images": gt.get("images", []),
                      "annotations": [],
                      "categories": gt.get("categories", [])}
        dt.createIndex()
    ev = COCOeval(coco, dt, iouType="bbox")
    if img_ids is not None:
        ev.params.imgIds = sorted(int(i) for i in img_ids)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = ev.stats
    return {
        "AP": float(s[0]),          # AP@[.50:.95]
        "AP50": float(s[1]),
        "AP75": float(s[2]),
        "AP_small": float(s[3]),
        "AP_medium": float(s[4]),
        "AP_large": float(s[5]),
        "AR@100": float(s[8]),
    }


def evaluate_prediction_file(
    gt_path: str, pred_path: str, img_ids=None
) -> Dict[str, float]:
    return coco_ap(gt_path, pred_path, img_ids=img_ids)


def collect_ap_table(
    results_dir: str,
    gt_path: str,
    img_ids=None,
) -> List[Dict[str, Any]]:
    """Evaluate every ``pred.<branch>.<level>.json`` under ``results_dir``.

    Returns rows suitable for tools/collect_results.py to append to
    results.csv: {branch, level, ...metrics}.
    """
    rows: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(results_dir)):
        if not (name.startswith("pred.") and name.endswith(".json")):
            continue
        _, branch, level = name.split(".")[:3]
        if branch not in BRANCHES:
            continue
        m = evaluate_prediction_file(
            gt_path, os.path.join(results_dir, name), img_ids=img_ids
        )
        rows.append({"branch": branch, "level": level, **m})
    return rows
