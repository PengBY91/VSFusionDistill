#!/usr/bin/env python
"""Branch-error correlation table (F2 §6.4, TODO W3 deliverable).

Reads the prediction files tools/test.py leaves in a run directory
(``pred.<branch>.<level>.json``, COCO result format) plus the paired
root, computes per-sample existence decisions for the camera and sonar
branches on matched frames (the pairs of ``pairs.csv``), and evaluates
the §6.4 quantities via ``aodet.evaluation.branch_correlation``:
phi, conditional miss lift (Delta P_miss) with bootstrap CIs, and the
Spearman confidence correlation.

A branch "detects" a frame when its best scoring detection that
matches the frame's GT box (IoU >= thr, same category) has score >=
tau; conf is that detection's score (0 when missed).  At D-infinity
the camera branch receives no frame, so every camera decision is a
miss by construction -- exactly the collapse the table is meant to
show.

    python tools/correlation_table.py <run_dir> --paired data/paired/uxo \
        [--levels d0 d1 d2 d3 d4 dinf] [--out run_dir/correlation.csv]
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CAMERA_BRANCHES = ("camera",)
SONAR_BRANCHES = ("sonar", "sonar_uncoupled")


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _decide(dets, gt_box, gt_cat, tau, iou_thr):
    """Existence decision + confidence from one image's detections."""
    best = 0.0
    for d in dets:
        if d["category_id"] == gt_cat and _iou(d["bbox"], gt_box) >= iou_thr:
            best = max(best, d["score"])
    return (1.0 if best >= tau else 0.0), best   # (hit, conf)


def _load(p):
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", help="directory with pred.<branch>.<level>.json")
    ap.add_argument("--paired", default="data/paired/uxo")
    ap.add_argument("--levels", nargs="+",
                    default=["d0", "d1", "d2", "d3", "d4", "dinf"])
    ap.add_argument("--tau", type=float, default=0.25,
                    help="existence score threshold")
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--out", default=None,
                    help="output csv (default <run_dir>/correlation.csv)")
    args = ap.parse_args()
    args.out = args.out or os.path.join(args.run_dir, "correlation.csv")

    camera = _load(os.path.join(args.paired, "camera.json"))
    sonar = _load(os.path.join(args.paired, "sonar.json"))
    gt_cam = {a["image_id"]: a for a in camera["annotations"]}
    gt_son = {a["image_id"]: a for a in sonar["annotations"]}
    with open(os.path.join(args.paired, "frame_meta.csv"), newline="") as f:
        frames = [dict(r) for r in csv.DictReader(f)]

    rows = []
    for level in args.levels:
        preds = {}
        for branch in CAMERA_BRANCHES + SONAR_BRANCHES:
            p = os.path.join(args.run_dir, f"pred.{branch}.{level}.json")
            if os.path.exists(p):
                by_img = {}
                for d in _load(p):
                    by_img.setdefault(d["image_id"], []).append(d)
                preds[branch] = by_img
        cam_b = next((b for b in CAMERA_BRANCHES if b in preds), None)
        son_b = next((b for b in SONAR_BRANCHES if b in preds), None)
        if cam_b is None or son_b is None:
            print(f"[{level}] skipped (need one camera and one sonar "
                  f"branch; found {sorted(preds)})")
            continue

        miss_c, miss_s, conf_c, conf_s = [], [], [], []
        for fr in frames:
            cam_id = int(fr["camera_image_id"])
            son_id = int(fr["sonar_image_id"])
            if cam_id not in gt_cam or son_id not in gt_son:
                continue                     # no GT on both sides -> no pair
            gca, gsa = gt_cam[cam_id], gt_son[son_id]
            hit_c, hc = _decide(preds[cam_b].get(cam_id, []),
                                gca["bbox"], gca["category_id"],
                                args.tau, args.iou_thr)
            hit_s, hs = _decide(preds[son_b].get(son_id, []),
                                gsa["bbox"], gsa["category_id"],
                                args.tau, args.iou_thr)
            miss_c.append(1.0 - hit_c)
            miss_s.append(1.0 - hit_s)
            conf_c.append(hc)
            conf_s.append(hs)

        from aodet.evaluation.branch_correlation import summarize_matched_pairs
        rep = summarize_matched_pairs(miss_c, miss_s, conf_c, conf_s)
        rows.append(dict(level=level, camera_branch=cam_b, sonar_branch=son_b,
                         **rep.as_row()))

    if rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        for r in rows:
            print(f"[{r['level']}] n={r['n_pairs']} "
                  f"phi={r['phi']:+.3f} (CI {r['phi_ci95'][0]:+.3f},"
                  f"{r['phi_ci95'][1]:+.3f}) "
                  f"dP_miss={r['delta_p_miss']:+.3f} "
                  f"(CI {r['delta_p_miss_ci95'][0]:+.3f},"
                  f"{r['delta_p_miss_ci95'][1]:+.3f}) "
                  f"spearman_conf={r['spearman_conf']:+.3f}")
        print(f"correlation table -> {args.out}")
    else:
        print("no levels with both branches; nothing written")


if __name__ == "__main__":
    main()
