"""Class-transplant upper bound (E11 A-class amendment, rethink §9.2 c).

The question: how much class-aware sonar AP could the camera's class
information buy at most, if sonar keeps its own localisation?  Keep the
sonar boxes and scores of a sonar-only model (B0), replace their class
with the class distribution of the camera branch on the pi-paired camera
frame (pairs from frame_meta: camera_image_id <-> sonar_image_id), and
re-evaluate class-aware AP on the sonar GT.

Every variant is computed on the same box set (the sonar predictions
after class-agnostic NMS), so differences isolate the class assignment:

* ``original``     -- sonar's own labels (the reference);
* ``camera_hard``  -- every box relabelled with the camera frame's argmax
  class, score unchanged;
* ``camera_soft``  -- every box replicated once per class with score
  ``s * p_cam(c)``, p_cam = per-class max camera score normalised over
  classes;
* ``oracle``       -- every box relabelled with the sonar frame's GT
  class: the ceiling of any class transplant (equals the class-agnostic
  AP of the same boxes when each frame has one target).

Frames whose paired camera frame has no detection keep sonar's own label
in the camera variants (counted as ``n_fallback``); camera information is
never read from GT.  The camera's class distribution is a per-frame
quantity because UXO frames hold a single target.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .per_branch_metric import (ap_by_group_and_category, coco_ap,
                                collapse_categories)

__all__ = [
    "VARIANTS",
    "agnostic_nms",
    "camera_class_dist",
    "transplant",
    "evaluate_transplant",
]

VARIANTS = ("original", "camera_hard", "camera_soft", "oracle")


def _iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _by_image(preds: Iterable[Dict[str, Any]]) -> Dict[int, List[Dict]]:
    out: Dict[int, List[Dict]] = {}
    for d in preds:
        out.setdefault(int(d["image_id"]), []).append(d)
    return out


def agnostic_nms(preds: Iterable[Dict[str, Any]],
                 iou_thr: float) -> List[Dict[str, Any]]:
    """Greedy class-agnostic NMS per image (COCO xywh results).

    YOLOX suppresses per class, so one echo can carry boxes of several
    labels; relabelling them all to one class would turn the extras into
    false positives.  ``iou_thr >= 1`` disables suppression.
    """
    kept: List[Dict[str, Any]] = []
    for _, dets in sorted(_by_image(preds).items()):
        dets = sorted(dets, key=lambda d: -float(d["score"]))
        keep: List[Dict[str, Any]] = []
        for d in dets:
            if iou_thr >= 1 or all(_iou_xywh(d["bbox"], k["bbox"]) < iou_thr
                                   for k in keep):
                keep.append(d)
        kept.extend(keep)
    return kept


def camera_class_dist(camera_preds: Iterable[Dict[str, Any]]
                      ) -> Dict[int, Dict[int, float]]:
    """camera image id -> {category_id: p}, from that frame's detections.

    p(c) = max score among the frame's detections of class c, normalised
    over classes.  Frames with no detection are absent.
    """
    out: Dict[int, Dict[int, float]] = {}
    for iid, dets in _by_image(camera_preds).items():
        best: Dict[int, float] = {}
        for d in dets:
            c = int(d["category_id"])
            best[c] = max(best.get(c, 0.0), float(d["score"]))
        z = sum(best.values())
        if z > 0:
            out[iid] = {c: v / z for c, v in best.items()}
    return out


def transplant(sonar_preds: Sequence[Dict[str, Any]],
               cam_dist_by_sonar_img: Mapping[int, Mapping[int, float]],
               gt_cat_by_sonar_img: Mapping[int, int],
               ) -> Dict[str, List[Dict[str, Any]]]:
    """Build the four variant result lists from NMS'd sonar predictions.

    ``cam_dist_by_sonar_img`` is the camera distribution already moved to
    sonar image ids through pi; ``gt_cat_by_sonar_img`` feeds only the
    oracle variant.
    """
    out: Dict[str, List[Dict[str, Any]]] = {v: [] for v in VARIANTS}
    for d in sonar_preds:
        iid = int(d["image_id"])
        out["original"].append(dict(d))
        dist = cam_dist_by_sonar_img.get(iid)
        if dist:
            top = max(sorted(dist), key=lambda c: dist[c])
            out["camera_hard"].append(dict(d, category_id=top))
            for c, p in sorted(dist.items()):
                out["camera_soft"].append(
                    dict(d, category_id=int(c), score=float(d["score"]) * p))
        else:
            out["camera_hard"].append(dict(d))
            out["camera_soft"].append(dict(d))
        g = gt_cat_by_sonar_img.get(iid)
        out["oracle"].append(dict(d) if g is None
                             else dict(d, category_id=int(g)))
    return out


def evaluate_transplant(
    sonar_gt: Dict[str, Any],
    camera_gt: Dict[str, Any],
    sonar_preds: Sequence[Dict[str, Any]],
    camera_preds: Sequence[Dict[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    sonar_img_ids: Iterable[int],
    group_of_img: Optional[Dict[int, str]] = None,
    nms_iou: float = 0.65,
) -> Dict[str, Any]:
    """Run the transplant on one (sonar run, camera run) combination.

    Args:
        sonar_gt / camera_gt: COCO GT dicts (sonar_gt already filtered to
            the evaluated GT source, e.g. blob).
        sonar_preds / camera_preds: COCO result lists.
        pairs: rows with ``camera_image_id`` and ``sonar_image_id``
            (frame_meta rows of the evaluated split).
        sonar_img_ids: sonar image ids to evaluate on.
        group_of_img: sonar image id -> recording, for the breakdown.
        nms_iou: class-agnostic NMS IoU applied before relabelling.

    Returns a dict with ``pooled`` (variant -> AP metrics), ``breakdown``
    (variant -> ap_by_group_and_category rows), ``delta_vs_original``
    (variant -> AP50 / AP gain) and ``diagnostics`` (pairing and camera
    label accuracy counts).
    """
    ids = sorted(int(i) for i in sonar_img_ids)
    id_set = set(ids)
    son_cat = {int(c["id"]): str(c["name"]) for c in sonar_gt["categories"]}
    cam_name = {int(c["id"]): str(c["name"]) for c in camera_gt["categories"]}
    name_to_son = {n: i for i, n in son_cat.items()}
    missing = sorted(set(cam_name.values()) - set(name_to_son))
    if missing:
        raise ValueError(f"camera categories {missing} absent from sonar GT")

    cam_of_son = {int(r["sonar_image_id"]): int(r["camera_image_id"])
                  for r in pairs}
    cam_dist = camera_class_dist(camera_preds)
    dist_by_son: Dict[int, Dict[int, float]] = {}
    for sid in ids:
        cid = cam_of_son.get(sid)
        if cid is not None and cid in cam_dist:
            dist_by_son[sid] = {name_to_son[cam_name[c]]: p
                                for c, p in cam_dist[cid].items()}

    gt_cat: Dict[int, int] = {}
    for a in sonar_gt.get("annotations", []):
        iid = int(a["image_id"])
        if iid in id_set:
            if iid in gt_cat and gt_cat[iid] != int(a["category_id"]):
                raise ValueError(f"sonar image {iid} has GT of two classes")
            gt_cat[iid] = int(a["category_id"])

    preds = [d for d in sonar_preds if int(d["image_id"]) in id_set]
    nmsd = agnostic_nms(preds, nms_iou)
    variants = transplant(nmsd, dist_by_son, gt_cat)

    pooled: Dict[str, Dict[str, float]] = {}
    breakdown: Dict[str, List[Dict[str, Any]]] = {}
    with contextlib.redirect_stdout(io.StringIO()):
        pooled["raw"] = coco_ap(sonar_gt, preds, img_ids=ids)
        g1, p1 = collapse_categories(sonar_gt, nmsd)
        pooled["class_agnostic"] = coco_ap(g1, p1, img_ids=ids)
        for v, res in variants.items():
            pooled[v] = coco_ap(sonar_gt, res, img_ids=ids)
    for v, res in variants.items():
        breakdown[v] = ap_by_group_and_category(sonar_gt, res, ids,
                                                group_of_img=group_of_img)

    base = pooled["original"]
    delta = {v: {"AP50": pooled[v]["AP50"] - base["AP50"],
                 "AP": pooled[v]["AP"] - base["AP"]} for v in VARIANTS}

    # camera label quality on frames with sonar GT (diagnostic only)
    n_gt = len(gt_cat)
    n_cam = sum(1 for s in gt_cat if s in dist_by_son)
    n_right = sum(1 for s, g in gt_cat.items() if s in dist_by_son
                  and max(sorted(dist_by_son[s]),
                          key=lambda c: dist_by_son[s][c]) == g)
    frames_with_pred = {int(d["image_id"]) for d in nmsd}
    diagnostics = {
        "n_eval_images": len(ids),
        "n_images_with_gt": n_gt,
        "n_sonar_preds_raw": len(preds),
        "n_sonar_preds_nms": len(nmsd),
        "n_fallback_images": len(frames_with_pred - set(dist_by_son)),
        "n_gt_images_with_camera_dist": n_cam,
        "camera_top1_acc_on_gt_images": (n_right / n_cam) if n_cam else None,
        "nms_iou": nms_iou,
    }
    return {"pooled": pooled, "breakdown": breakdown,
            "delta_vs_original": delta, "diagnostics": diagnostics}
