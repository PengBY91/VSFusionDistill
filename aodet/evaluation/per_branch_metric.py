"""Per-branch, per-degradation-level COCO AP (F2 §2.3).

Evaluation reads *saved prediction files* — one JSON per (method, run,
branch, level) — so the same predictions can be re-analysed without
re-inference (setup doc §4.4(6)).  This module wraps pycocotools; the
"official example cross-check" test runs COCOeval directly on the same toy
data and compares numbers.

E11 prereg item 5 (doc/experiments.md §E11; rethink §5 rule 7) adds the
class-agnostic view (:func:`collapse_categories`) and the per-recording /
per-class decomposition (:func:`ap_by_group_and_category`) with known-
answer tests: perfect predictions score 1.0, fully swapped classes score
0 class-aware while class-agnostic AP is unchanged.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    "filter_gt_sources",
    "blob_eval_img_ids",
    "evaluate_prediction_file",
    "collect_ap_table",
    "collapse_categories",
    "ap_by_group_and_category",
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


def filter_gt_sources(gt: Dict[str, Any],
                      sources: Sequence[str]) -> Dict[str, Any]:
    """Copy of a COCO GT dict keeping only annotations whose ``source``
    field (written by the projection converter: "blob" = measured echo,
    "projected" = calibrated geometry) is in ``sources``.

    Motivation (E05): projected boxes carry the calibration's range-axis
    error, so the all-GT number under-reports echo-detection quality;
    reporting the blob-only subset alongside keeps a clean reference
    while the calibration is improved.  Images/categories are passed
    through untouched (COCOeval ignores imageless annotations).
    """
    anns = [a for a in gt.get("annotations", [])
            if a.get("source", "blob") in sources]
    return {**gt, "annotations": anns}


def blob_eval_img_ids(gt: Dict[str, Any],
                      img_ids: Iterable[int]) -> List[int]:
    """Image ids for the ``gt=blob`` view: drop projected-only frames.

    A projected-only frame has a physical target whose echo did not pass
    the blob gate.  With its annotation removed but the image kept, every
    detection there — including a correct one on a weak echo — would
    count as a false positive (review 2026-10-31 §4.2).  Frames with a
    blob annotation, and frames with no annotation at all (true
    negatives), stay.
    """
    has_blob = {a["image_id"] for a in gt.get("annotations", [])
                if a.get("source", "blob") == "blob"}
    has_any = {a["image_id"] for a in gt.get("annotations", [])}
    return [int(i) for i in img_ids if i in has_blob or i not in has_any]


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


def collapse_categories(
    gt: Dict[str, Any],
    preds: Optional[Sequence[Dict[str, Any]]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Class-agnostic view of one (gt, preds) pair: a single category.

    Every annotation and prediction keeps its image id, box and score;
    category ids are all rewritten to 1, so "box right, label wrong"
    still matches.  This is the reading the E08/E11 class-axis evidence
    contrasts with class-aware AP (previously only tools/anatomy.py had a
    private copy).  Returns (gt, preds); the inputs are not modified.
    """
    one = [{"id": 1, "name": "object"}]
    gt2 = {**gt, "categories": one,
           "annotations": [dict(a, category_id=1)
                           for a in gt.get("annotations", [])]}
    preds2 = [dict(d, category_id=1) for d in (preds or [])]
    return gt2, preds2


def _ap_row(group: str, category: str, ids: List[int],
            sub_gt: Dict[str, Any],
            sub_preds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One breakdown row: group/category identity + counts + coco_ap."""
    with contextlib.redirect_stdout(io.StringIO()):
        m = coco_ap(sub_gt, sub_preds, img_ids=ids)
    return {"group": group, "category": category, "n_images": len(ids),
            "n_gt": len(sub_gt.get("annotations", [])),
            "n_pred": len(sub_preds), **m}


def ap_by_group_and_category(
    gt: Dict[str, Any],
    preds: Optional[Sequence[Dict[str, Any]]],
    img_ids: Iterable[int],
    group_of_img: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    """COCO AP broken down by image group (e.g. recording) and category.

    E11 prereg item 5: pooled numbers must come with a per-recording,
    per-class decomposition (rethink §5 rule 5).  Returns one row per
    (group, category) for every category present in that group's ground
    truth::

        {"group", "category", "n_images", "n_gt", "n_pred",
         "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
         "AR@100"}

    plus a ``category="all"`` row per group (class-aware AP on the
    group, with an extra ``AP50_class_agnostic`` field) and pooled
    ``group="all"`` rows (one per category, plus the overall one).

    Args:
        gt: COCO-format ground-truth dict (or a path to one).
        preds: COCO-format result list (or a path to one); predictions
            on images outside ``img_ids`` are ignored.
        img_ids: image ids to evaluate on.  Images with no ground truth
            stay in the sets — they are true negatives, cf.
            :func:`blob_eval_img_ids`.
        group_of_img: image id -> group name (e.g. frame_meta's
            ``recording``).  ``None`` (default) yields only the pooled
            rows; images without an entry feed only the pooled rows.

    pycocotools prints one summary table per COCOeval call — with dozens
    of subsets per run that noise is swallowed here.
    """
    _require_pycoco()
    if isinstance(gt, str):
        with open(gt, encoding="utf-8") as f:
            gt = json.load(f)
    if isinstance(preds, str):
        with open(preds, encoding="utf-8") as f:
            preds = json.load(f)
    img_ids = [int(i) for i in img_ids]
    id_set = set(img_ids)
    preds = [d for d in (preds or []) if int(d["image_id"]) in id_set]
    anns = [a for a in gt.get("annotations", [])
            if int(a["image_id"]) in id_set]
    cat_names = {int(c["id"]): str(c.get("name", c["id"]))
                 for c in gt.get("categories", [])}

    group_ids: List[Tuple[str, List[int]]] = []
    if group_of_img:
        by_group: Dict[str, List[int]] = {}
        for i in img_ids:
            g = group_of_img.get(i)
            if g is not None:
                by_group.setdefault(str(g), []).append(i)
        group_ids = sorted(by_group.items())
    group_ids.append(("all", img_ids))

    rows: List[Dict[str, Any]] = []
    for g, ids in group_ids:
        ids_set = set(ids)
        g_anns = [a for a in anns if int(a["image_id"]) in ids_set]
        g_preds = [d for d in preds if int(d["image_id"]) in ids_set]
        g_gt = {**gt, "annotations": g_anns}
        for cid in sorted({int(a["category_id"]) for a in g_anns}):
            rows.append(_ap_row(
                g, cat_names.get(cid, str(cid)), ids,
                {**g_gt, "annotations": [a for a in g_anns
                                         if int(a["category_id"]) == cid]},
                [d for d in g_preds if int(d["category_id"]) == cid]))
        with contextlib.redirect_stdout(io.StringIO()):
            ag_gt, ag_preds = collapse_categories(g_gt, g_preds)
            row = _ap_row(g, "all", ids, g_gt, g_preds)
            row["AP50_class_agnostic"] = coco_ap(
                ag_gt, ag_preds, img_ids=ids)["AP50"]
        rows.append(row)
    return rows
