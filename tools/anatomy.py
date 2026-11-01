#!/usr/bin/env python
"""Prediction anatomy for one run dir (E08/E09 pre-registered analysis).

Decomposes a sonar-branch prediction dump (``pred.<branch>.<level>.json``,
COCO result format, written by tools/test.py) by RECORDING and reports,
per recording (blob-view GT):

* class-aware AP50 (COCOeval restricted to the recording's frames);
* class-agnostic AP50 (all categories collapsed to one class -- "the box
  is right, the label is wrong" view; the E08 class-axis evidence);
* prediction category distribution (softmax-collapse reading);
* near-GT placement + scale: predictions whose centre lies within
  +-near_row rows / +-near_col columns of a GT centre, with the height
  p50 of those predictions vs the GT height p50 (the E07/E08 scale-axis
  mechanism check).

E11 prereg item 5 (doc/experiments.md §E11): ``--split`` takes a comma-
separated list of frame_meta split values (the E11 converter writes
train/val/test/val_id; the column is otherwise opaque here), and a second
CSV ``anatomy.<branch>.<level>_by_class.csv`` decomposes AP50 by
recording x category on the same GT-bearing frames
(aodet.evaluation.per_branch_metric.ap_by_group_and_category), with
"all" rows carrying the pooled and class-agnostic views.

    python tools/anatomy.py <run_dir> [--branch sonar_uncoupled] \
        [--level d0] [--paired data/paired/uxo] [--split test] \
        [--near-row 30] [--near-col 20] [--out run_dir/anatomy.csv] \
        [--out-by-class run_dir/anatomy_by_class.csv]

E08 B0 self-check (2026-10-31 run, old recipe): 100927 class-agnostic
AP50 0.097, near-GT height p50 22 vs GT 23, cat3 0/443 predictions.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(p):
    with open(p) as f:
        return json.load(f)


def _coco_ap50(gt, dets, img_ids):
    """AP50 of dets against gt on img_ids (class-aware or -agnostic:
    pass gt already collapsed for the latter)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco = COCO()
    coco.dataset = gt
    coco.createIndex()
    if not dets:
        return 0.0
    evala = COCOeval(coco, coco.loadRes(dets), iouType="bbox")
    evala.params.imgIds = img_ids
    evala.evaluate()
    evala.accumulate()
    evala.summarize()
    return float(evala.stats[1])          # AP@IoU=0.50


def _by_class_rows(sonar, preds, rec_imgs, anns_by_img):
    """Per-recording x per-category AP50 rows on the GT-bearing frames.

    Same blob-view GT and the same frame sets as the per-recording rows
    in main(), so the ``category=all`` AP50 of a recording here equals
    the AP50 column there (the COCOeval work happens in
    aodet.evaluation.per_branch_metric; "all" rows additionally carry
    the class-agnostic AP50).
    """
    from aodet.evaluation.per_branch_metric import ap_by_group_and_category
    group_of_img = {i: rec for rec, imgs in rec_imgs.items() for i in imgs
                    if i in anns_by_img}
    if not group_of_img:
        return []
    rows = ap_by_group_and_category(
        sonar, preds, sorted(group_of_img), group_of_img=group_of_img)
    return [dict(recording=r["group"], category=r["category"],
                 n_images=r["n_images"], n_gt=r["n_gt"], n_pred=r["n_pred"],
                 AP50=round(r["AP50"], 4),
                 **({"AP50_class_agnostic":
                     round(r["AP50_class_agnostic"], 4)}
                    if "AP50_class_agnostic" in r else {}))
            for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--branch", default="sonar_uncoupled")
    ap.add_argument("--level", default="d0")
    ap.add_argument("--paired", default="data/paired/uxo")
    ap.add_argument("--split", default="test",
                    help="comma-separated frame_meta split values of the "
                         "pred dump ('all' = no filter)")
    ap.add_argument("--near-row", type=int, default=30,
                    help="near-GT window, rows (+-)")
    ap.add_argument("--near-col", type=int, default=20,
                    help="near-GT window, columns (+-)")
    ap.add_argument("--gt-sources", nargs="+", default=["blob"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-by-class", default=None,
                    help="per-recording x per-class CSV path (default: "
                         "<out> with _by_class before the extension)")
    args = ap.parse_args()
    args.out = args.out or os.path.join(
        args.run_dir, f"anatomy.{args.branch}.{args.level}.csv")
    if not args.out_by_class:
        base, ext = os.path.splitext(args.out)
        args.out_by_class = base + "_by_class" + ext

    from aodet.evaluation.per_branch_metric import (
        collapse_categories, filter_gt_sources)

    pred_path = os.path.join(
        args.run_dir, f"pred.{args.branch}.{args.level}.json")
    preds = _load(pred_path)
    sonar = filter_gt_sources(
        _load(os.path.join(args.paired, "sonar.json")),
        tuple(args.gt_sources))
    with open(os.path.join(args.paired, "frame_meta.csv"), newline="") as f:
        frames = [dict(r) for r in csv.DictReader(f)]
    if args.split != "all":
        wanted = {s.strip() for s in args.split.split(",") if s.strip()}
        frames = [fr for fr in frames if fr["split"] in wanted]

    cats = {c["id"]: c["name"] for c in _load(
        os.path.join(args.paired, "sonar.json"))["categories"]}

    # recording -> sonar image ids with (filtered) GT
    rec_imgs = {}
    for fr in frames:
        rec_imgs.setdefault(fr["recording"], set()).add(
            int(fr["sonar_image_id"]))
    anns_by_img = {}
    for a in sonar["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    preds_by_img = {}
    for d in preds:
        preds_by_img.setdefault(d["image_id"], []).append(d)

    rows = []
    for rec, imgs in sorted(rec_imgs.items()):
        imgs = sorted(i for i in imgs if i in anns_by_img)
        if not imgs:
            continue
        gt_rec = {**sonar,
                  "annotations": [a for i in imgs for a in anns_by_img[i]]}
        # COCOeval needs image entries present
        gt_rec = {**gt_rec,
                  "images": [im for im in sonar["images"] if im["id"] in imgs]}
        dets_rec = [d for d in preds if d["image_id"] in imgs]
        gt_agn, dets_agn = collapse_categories(gt_rec, dets_rec)

        ap50 = _coco_ap50(gt_rec, dets_rec, imgs)
        ap50_agn = _coco_ap50(gt_agn, dets_agn, imgs)

        # category distribution over ALL predictions in the recording
        cat_counts = {}
        for d in dets_rec:
            cat_counts[cats.get(d["category_id"], d["category_id"])] = \
                cat_counts.get(cats.get(d["category_id"],
                                        d["category_id"]), 0) + 1

        # near-GT placement + scale: pred centre within the GT window
        near_h, n_near, n_gt = [], 0, 0
        for i in imgs:
            for a in anns_by_img[i]:
                n_gt += 1
                gcx, gcy = (a["bbox"][0] + a["bbox"][2] / 2,
                            a["bbox"][1] + a["bbox"][3] / 2)
                for d in preds_by_img.get(i, []):
                    pcx, pcy = (d["bbox"][0] + d["bbox"][2] / 2,
                                d["bbox"][1] + d["bbox"][3] / 2)
                    if (abs(pcy - gcy) <= args.near_row
                            and abs(pcx - gcx) <= args.near_col):
                        n_near += 1
                        near_h.append(d["bbox"][3])

        # top-1 (max-score) per frame -- the E08 anatomy construction:
        # fraction of frames whose TOP-1 prediction lands near GT, and the
        # height p50 of those top-1s vs the GT height p50 on those frames
        top1_near_h, top1_near_gt_h = [], []
        for i in imgs:
            dp = preds_by_img.get(i, [])
            if not dp:
                continue
            d = max(dp, key=lambda x: x["score"])
            for a in anns_by_img[i]:
                gcx, gcy = (a["bbox"][0] + a["bbox"][2] / 2,
                            a["bbox"][1] + a["bbox"][3] / 2)
                pcx, pcy = (d["bbox"][0] + d["bbox"][2] / 2,
                            d["bbox"][1] + d["bbox"][3] / 2)
                if (abs(pcy - gcy) <= args.near_row
                        and abs(pcx - gcx) <= args.near_col):
                    top1_near_h.append(d["bbox"][3])
                    top1_near_gt_h.append(a["bbox"][3])

        def _p50(xs):
            xs = sorted(xs)
            return xs[len(xs) // 2] if xs else 0.0

        gt_h50 = _p50([a["bbox"][3] for i in imgs for a in anns_by_img[i]])

        rows.append(dict(
            recording=rec, n_gt_frames=len(imgs), n_gt=n_gt,
            n_pred=len(dets_rec), AP50=round(ap50, 4),
            AP50_class_agnostic=round(ap50_agn, 4),
            pred_cats=json.dumps(cat_counts, ensure_ascii=False),
            n_pred_near_gt=n_near,
            pred_h50_near_gt=_p50(near_h), gt_h50=gt_h50,
            n_top1_near=len(top1_near_h),
            top1_h_p50=_p50(top1_near_h),
            top1_gt_h_p50=_p50(top1_near_gt_h)))

    if rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        for r in rows:
            print(f"[{r['recording']}] gt_frames={r['n_gt_frames']} "
                  f"AP50={r['AP50']:.3f} AP50_agnostic="
                  f"{r['AP50_class_agnostic']:.3f} "
                  f"preds={r['n_pred']} cats={r['pred_cats']} "
                  f"nearGT={r['n_pred_near_gt']} "
                  f"h_p50={r['pred_h50_near_gt']:.1f}/{r['gt_h50']:.1f} "
                  f"top1_near={r['n_top1_near']}/{r['n_gt_frames']} "
                  f"top1_h={r['top1_h_p50']:.1f}/{r['top1_gt_h_p50']:.1f}")
        print(f"anatomy -> {args.out}")
        by_class = _by_class_rows(sonar, preds, rec_imgs, anns_by_img)
        if by_class:
            fields = ["recording", "category", "n_images", "n_gt", "n_pred",
                      "AP50", "AP50_class_agnostic"]
            with open(args.out_by_class, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(by_class)
            print(f"anatomy_by_class -> {args.out_by_class}")
    else:
        print("no recordings with GT; nothing written")


if __name__ == "__main__":
    main()
