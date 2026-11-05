#!/usr/bin/env python
"""Class-transplant upper bound: sonar boxes + pi-paired camera classes.

E11 A-class amendment (doc/experiments.md §E11; rethink §9.2 c): keep a
sonar-only model's boxes and scores, replace their class with the class
distribution of a camera branch on the pi-paired frame, and see how far
class-aware sonar AP rises.  E12 starts only if this bound clears B0 by
the frozen effect threshold.  Logic lives in
``aodet.evaluation.class_transplant``; variants: original / camera_hard /
camera_soft / oracle (GT class, the ceiling), all on the same
class-agnostic-NMS'd box set.

Reads saved prediction files only (tools/test.py output), so sonar and
camera may come from different runs (e.g. B0 sonar + Bcam or B2 camera):

    python tools/class_transplant.py \\
        --sonar-pred work_dirs/<b0_run>/eval_val/pred.sonar_uncoupled.d0.json \\
        --camera-pred work_dirs/<bcam_run>/eval_val/pred.camera.d0.json \\
        --paired data/paired/uxo_full --split val \\
        --out work_dirs/<b0_run>/transplant.<tag>.json

``--split`` is required (E11: every evaluation names its split
explicitly); ``test`` is refused unless ``--unseal-test`` is given.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sonar-pred", required=True,
                    help="sonar-branch COCO results (e.g. B0 pred.sonar_uncoupled.d0.json; B2-type fused head: pred.sonar.d0.json)")
    ap.add_argument("--camera-pred", required=True,
                    help="camera-branch COCO results on the same split "
                         "(e.g. Bcam/B2 pred.camera.d0.json)")
    ap.add_argument("--paired", required=True,
                    help="paired root with sonar.json, camera.json, "
                         "frame_meta.csv")
    ap.add_argument("--split", required=True,
                    help="comma-separated frame_meta split values")
    ap.add_argument("--unseal-test", action="store_true",
                    help="allow 'test' in --split (protocol freeze only)")
    ap.add_argument("--sonar-gt-sources", nargs="+", default=["blob"],
                    help="sonar annotation sources kept as GT (default blob)")
    ap.add_argument("--nms-iou", type=float, default=0.65,
                    help="class-agnostic NMS IoU before relabelling "
                         "(>=1 disables)")
    ap.add_argument("--tag", default="",
                    help="free-text label stored in the output")
    ap.add_argument("--out", required=True, help="output json")
    args = ap.parse_args(argv)

    splits = {s.strip() for s in args.split.split(",") if s.strip()}
    if "test" in splits and not args.unseal_test:
        raise SystemExit("refusing --split test: the test split is sealed "
                         "until protocol freeze (pass --unseal-test)")

    from aodet.evaluation.class_transplant import VARIANTS, evaluate_transplant
    from aodet.evaluation.per_branch_metric import (blob_eval_img_ids,
                                                    filter_gt_sources)

    sonar_all = _load(os.path.join(args.paired, "sonar.json"))
    camera_gt = _load(os.path.join(args.paired, "camera.json"))
    sources = tuple(args.sonar_gt_sources)
    sonar_gt = filter_gt_sources(sonar_all, sources)
    with open(os.path.join(args.paired, "frame_meta.csv"), newline="",
              encoding="utf-8") as f:
        frames = [dict(r) for r in csv.DictReader(f)]
    if frames and "split" not in frames[0]:
        raise SystemExit("frame_meta.csv has no 'split' column")
    frames = [fr for fr in frames if fr["split"] in splits]
    if not frames:
        raise SystemExit(f"no frame_meta rows with split in {sorted(splits)}")

    ids = sorted({int(fr["sonar_image_id"]) for fr in frames})
    if sources == ("blob",):
        ids = blob_eval_img_ids(sonar_all, ids)
    rec_of = {int(fr["sonar_image_id"]): fr.get("recording", "")
              for fr in frames if fr.get("recording")}

    res = evaluate_transplant(
        sonar_gt, camera_gt, _load(args.sonar_pred), _load(args.camera_pred),
        frames, ids, group_of_img=rec_of or None, nms_iou=args.nms_iou)
    res["inputs"] = {"sonar_pred": args.sonar_pred,
                     "camera_pred": args.camera_pred,
                     "paired": args.paired, "split": sorted(splits),
                     "sonar_gt_sources": list(sources), "tag": args.tag}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    p, d = res["pooled"], res["delta_vs_original"]
    print(f"split={','.join(sorted(splits))} gt={','.join(sources)} "
          f"images={res['diagnostics']['n_eval_images']}")
    for v in ("raw", "class_agnostic") + VARIANTS:
        extra = (f"  dAP50={d[v]['AP50']:+.4f} dAP={d[v]['AP']:+.4f}"
                 if v in d else "")
        print(f"  {v:<15} AP50={p[v]['AP50']:.4f} AP={p[v]['AP']:.4f}{extra}")
    for v in VARIANTS:
        per = {r["category"]: r["AP50"] for r in res["breakdown"][v]
               if r["group"] == "all" and r["category"] != "all"}
        print(f"  {v:<15} per-class AP50 " + " ".join(
            f"{k}={per[k]:.3f}" for k in sorted(per)))
    dg = res["diagnostics"]
    acc = dg["camera_top1_acc_on_gt_images"]
    print(f"  camera top1 acc on GT frames: "
          f"{'n/a' if acc is None else f'{acc:.3f}'} "
          f"({dg['n_gt_images_with_camera_dist']}/{dg['n_images_with_gt']} "
          f"with camera dets); fallback frames {dg['n_fallback_images']}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
