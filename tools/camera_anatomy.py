#!/usr/bin/env python3
"""Per-recording anatomy of the camera branch on the UXO paired test split.

Decomposes the pooled camera AP50 the official ap_summary reports into
per-recording, per-class contributions, because the test split is two
recordings with one class each and the pooled number hides which axis
(localisation, classification, annotation domain) failed.

For every test recording it prints:

* GT geometry profile: box count, tall-box share (h > w), bottom-clipped
  share (y2 >= 0.98*H), and the centre trajectory first -> last (the
  124735 target moves; every other recording is static).
* Class-aware AP50 per GT class and class-agnostic AP50 (all-points,
  greedy IoU >= 0.5 matching, same convention as the pooled cross-check).
* Top-1 prediction IoU deciles and the median centre offset (dx, dy)
  between the top-1 prediction and the GT box, per frame.
* Prediction category distribution (the class-confusion readout that
  E10 criterion 5 consumes: cat3 share on a cat3 recording etc.).

The pooled class-aware mean over test classes is re-printed at the end so
the per-recording numbers can be tied back to ap_summary.json (validated
2026-11-01: B3 0.0502 vs official 0.0499, T-L 0.1248 vs 0.1270).

Usage::

    python tools/camera_anatomy.py work_dirs/<run> [--level d0]
"""
import argparse
import csv
import json
import os
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = os.path.join(REPO, "data/paired/uxo/frame_meta.csv")
GT = os.path.join(REPO, "data/paired/uxo/camera.json")


def iou(a, b):
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def ap50(preds, gt_by_img):
    """All-points AP50; preds are (score, img_id, box)."""
    total = sum(len(v) for v in gt_by_img.values())
    if total == 0:
        return None
    if not preds:
        return 0.0
    ranked = sorted(preds, key=lambda x: -x[0])
    used, tp, fp = set(), 0, 0
    precs, recs = [], []
    for s, img, box in ranked:
        hit = False
        for gb in gt_by_img.get(img, []):
            key = (img, id(gb))
            if key not in used and iou(box, gb) >= 0.5:
                used.add(key)
                hit = True
                break
        tp += hit
        fp += not hit
        precs.append(tp / (tp + fp))
        recs.append(tp / total)
    mrec = [0.0] + recs + [1.0]
    mpre = [0.0] + precs + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    return sum((mrec[i] - mrec[i - 1]) * mpre[i]
               for i in range(1, len(mrec)) if mrec[i] != mrec[i - 1])


def load_world():
    rec, split, fname = {}, {}, {}
    with open(META) as f:
        for row in csv.DictReader(f):
            cid = int(row["camera_image_id"])
            rec[cid] = row["recording"].split("_")[-1]
            split[cid] = row["split"]
    with open(GT) as f:
        cj = json.load(f)
    for i in cj["images"]:
        fname[i["id"]] = i["file_name"]
    gt_by = defaultdict(list)
    for a in cj["annotations"]:
        gt_by[a["image_id"]].append((a["category_id"], a["bbox"]))
    return rec, split, fname, gt_by


def analyze(run, level, rec, split, fname, gt_by):
    pred_file = os.path.join(run, f"pred.camera.{level}.json")
    preds = json.load(open(pred_file))
    by_img = defaultdict(list)
    for p in preds:
        by_img[p["image_id"]].append(p)

    test_recs = sorted({rec[i] for i in gt_by if split[i] == "test"})
    print(f"\n===== {os.path.basename(run.rstrip('/'))} @ {level} (test) =====")
    pooled = []
    for r in test_recs:
        gts = {i: v for i, v in gt_by.items() if split[i] == "test" and rec[i] == r}
        rec_preds = [p for p in preds if split.get(p["image_id"]) == "test"
                     and rec[p["image_id"]] == r]
        # GT geometry profile
        boxes = [b for v in gts.values() for _, b in v]
        n = len(boxes)
        tall = sum(1 for b in boxes if b[3] > b[2])
        clip = sum(1 for b in boxes if b[1] + b[3] >= 0.98 * 1080)
        cys = [b[1] + b[3] / 2 for b in boxes]
        print(f"[{r}] GT n={n} cats={sorted({c for v in gts.values() for c, _ in v})} "
              f"tall(h>w)={tall}/{n} bottom-clip={clip}/{n} "
              f"cy {cys[0]:.0f}->{cys[-1]:.0f} (range {min(cys):.0f}..{max(cys):.0f})")
        # AP decomposition
        out = {}
        for c in sorted({c for v in gts.values() for c, _ in v}):
            g = {i: [b for cc, b in v if cc == c] for i, v in gts.items()}
            p = [(x["score"], x["image_id"], x["bbox"])
                 for x in rec_preds if x["category_id"] == c]
            out[f"cat{c}"] = ap50(p, g)
            pooled.append(ap50(p, g))
        g_agn = {i: [b for _, b in v] for i, v in gts.items()}
        p_agn = [(x["score"], x["image_id"], x["bbox"]) for x in rec_preds]
        out["agnostic"] = ap50(p_agn, g_agn)
        # top-1 localisation + centre offset + category distribution
        top1_ious, dxs, dys = [], [], []
        catdist = defaultdict(int)
        for x in rec_preds:
            catdist[x["category_id"]] += 1
        for i, v in gts.items():
            if not by_img.get(i):
                continue
            best = max(by_img[i], key=lambda p: p["score"])
            b = best["bbox"]
            top1_ious.append(max(iou(b, gb) for _, gb in v))
            g = v[0][1]
            dxs.append(b[0] + b[2] / 2 - (g[0] + g[2] / 2))
            dys.append(b[1] + b[3] / 2 - (g[1] + g[3] / 2))
        top1_ious.sort()
        k = len(top1_ious)
        med = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("nan")
        print(f"      AP50 {out}  top1 IoU p50={top1_ious[k // 2]:.2f} "
              f"(p10={top1_ious[k // 10]:.2f} max={top1_ious[-1]:.2f})  "
              f"top1 offset med (dx,dy)=({med(dxs):+.0f},{med(dys):+.0f})px  "
              f"pred cats={dict(sorted(catdist.items()))}")
    if pooled:
        print(f"pooled class-aware AP50 (cross-check vs ap_summary): "
              f"{sum(pooled) / len(pooled):.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work_dirs", nargs="+", help="run work_dirs with pred.camera dumps")
    ap.add_argument("--level", default="d0", help="degradation level tag (default d0)")
    args = ap.parse_args()
    world = load_world()
    for run in args.work_dirs:
        analyze(run, args.level, *world)


if __name__ == "__main__":
    main()
