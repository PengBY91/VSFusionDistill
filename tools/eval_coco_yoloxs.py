#!/usr/bin/env python
"""W1 acceptance: reproduce mmdet YOLOX-S COCO AP on a val2017 subset.

Downloads the first ``--n`` val2017 images (sorted image ids, pinned by
the subset json), runs the official checkpoint with the standard val
pipeline, and prints COCO AP.  The subset keeps the acceptance cheap on
CPU; record the subset size next to the number (sampling noise on the
AP of a random subset shrinks as ~1/sqrt(n)).

    python tools/eval_coco_yoloxs.py --n 1000 --out work/coco_eval/
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMG_URL = "http://images.cocodataset.org/val2017/{}"
CKPT_URL = ("https://download.openmmlab.com/mmdetection/v2.0/yolox/"
            "yolox_s_8x8_300e_coco/"
            "yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth")
# official mmdetection model-zoo number for this checkpoint (full val2017);
# the mmdet 3.3 metafile ships 40.5 (mmdet2 release notes said 40.2)
OFFICIAL_AP = 40.5


def build_subset(ann_json, img_dir, n, subset_json, mode, seed):
    with open(ann_json) as f:
        coco = json.load(f)
    ids = sorted(im["id"] for im in coco["images"])
    if mode == "first":
        ids = ids[:n]
    else:  # seeded random draw: composition-bias-free vs the official number
        rng = random.Random(seed)
        rng.shuffle(ids)
        ids = sorted(ids[:n])
    keep = set(ids)
    images = [im for im in coco["images"] if im["id"] in keep]
    anns = [a for a in coco["annotations"] if a["image_id"] in keep]
    with open(subset_json, "w") as f:
        json.dump(dict(info=coco.get("info", {}),
                       licenses=coco.get("licenses", []),
                       images=images, annotations=anns,
                       categories=coco["categories"]), f)
    print(f"subset: {len(images)} images, {len(anns)} annotations -> "
          f"{subset_json}")
    return [im["file_name"] for im in images]


def fetch_images(file_names, img_dir, max_rounds: int = 5):
    """Fetch missing images; retry in rounds so one transient curl failure
    (xargs exits 123 if ANY invocation fails) doesn't abort the batch."""
    t0 = time.time()
    for _ in range(max_rounds):
        todo = [fn for fn in file_names
                if not os.path.exists(os.path.join(img_dir, fn))]
        if not todo:
            print(f"all {len(file_names)} subset images present "
                  f"({time.time() - t0:.0f}s)")
            return
        print(f"fetching {len(todo)} images ...")
        with open(os.path.join(img_dir, "_urls.txt"), "w") as f:
            for fn in todo:
                f.write(f"{IMG_URL.format(fn)}\n")
        subprocess.run(
            ["bash", "-c",
             f"cd {img_dir} && xargs -P 8 -n1 curl -sfO --retry 3 "
             "--retry-all-errors < _urls.txt || true; rm -f _urls.txt"])
    missing = [fn for fn in file_names
               if not os.path.exists(os.path.join(img_dir, fn))]
    raise RuntimeError(f"{len(missing)} images still missing after "
                       f"{max_rounds} rounds, e.g. {missing[:3]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="data/coco/annotations/"
                                     "instances_val2017.json")
    ap.add_argument("--img-dir", default="data/coco/val2017")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--sample", choices=("first", "random"), default="first",
                    help="'first' = lowest image ids (biased easy, see log "
                         "2026-10-30: first-1000 AP 42.2 vs official 40.5); "
                         "'random' = seeded draw, unbiased for the gate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="work/coco_eval")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    suffix = (f"{args.sample}{args.n}" +
              (f"_s{args.seed}" if args.sample == "random" else ""))
    subset_json = os.path.join(args.out, f"val2017_{suffix}.json")
    ckpt = os.path.join(args.out, "yolox_s.pth")

    names = build_subset(args.ann, args.img_dir, args.n, subset_json,
                         args.sample, args.seed)
    if not args.skip_download:
        fetch_images(names, args.img_dir)
        if not os.path.exists(ckpt):
            subprocess.run(["curl", "-sfL", "--retry", "3", "-o", ckpt,
                            CKPT_URL], check=True)
            print(f"checkpoint -> {ckpt}")

    cfg_text = """
# Mirrors configs/yolox/yolox_s_8xb8-300e_coco.py of the *installed* mmdet
# 3.3.  YOLOX takes raw 0-255 BGR inputs: no mean/std in the
# data_preprocessor, and the test pipeline pads to square with 114 (the
# ImageNet-style normalize used by most other models zeroes the
# objectness branch and collapses AP to ~0).
default_scope = 'mmdet'
custom_imports = dict(imports=['aodet.datasets', 'aodet.models'],
                      allow_failed_imports=False)
model = dict(
    type='YOLOX', data_preprocessor=dict(
        type='DetDataPreprocessor', pad_size_divisor=32),
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5,
                  out_indices=(2, 3, 4)),
    neck=dict(type='YOLOXPAFPN', in_channels=[128, 256, 512],
              out_channels=128, num_csp_blocks=1),
    bbox_head=dict(type='YOLOXHead', num_classes=80, in_channels=128,
                   feat_channels=128),
    test_cfg=dict(nms_pre=1000, min_bbox_size=0, score_thr=0.01,
                  nms=dict(type='nms', iou_threshold=0.65),
                  max_per_img=100))
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', pad_to_square=True,
         pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')]
test_dataloader = dict(
    batch_size=1, num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='CocoDataset',
                 data_root='ROOT',
                 ann_file='ANN', data_prefix=dict(img='val2017/'),
                 test_mode=True, pipeline=test_pipeline,
                 metainfo=dict(classes=['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'])))
test_evaluator = dict(type='CocoMetric', ann_file='ANN', metric='bbox')
test_cfg = dict(type='TestLoop')
"""
    cfg_path = os.path.join(args.out, "yolox_s_val.py")
    with open(cfg_path, "w") as f:
        f.write(cfg_text.replace("ROOT", os.path.abspath("data/coco"))
                .replace("ANN", os.path.abspath(subset_json)))

    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    init_default_scope("mmdet")
    import aodet.datasets, aodet.models  # noqa: F401
    from mmengine.runner import Runner
    cfg = Config.fromfile(cfg_path)
    cfg.load_from = ckpt
    cfg.work_dir = args.out
    cfg.default_hooks = dict(logger=dict(type="LoggerHook", interval=100))
    cfg.log_level = "WARNING"
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    ap = metrics["coco/bbox_mAP"] * 100  # official numbers are in percent
    print(f"\nsubset({args.n}) bbox mAP = {ap:.2f} "
          f"(official full-val = {OFFICIAL_AP}) "
          f"diff = {abs(ap - OFFICIAL_AP):.2f}")
    if abs(ap - OFFICIAL_AP) > 1.0:
        print("ACCEPTANCE FAIL: > 1 AP off")
        sys.exit(1)
    print("ACCEPTANCE PASS: within 1 AP")


if __name__ == "__main__":
    main()
