#!/usr/bin/env python
"""Inference + per-branch evaluation for TwoStreamYOLOX (setup doc §4.4(6)).

Sweeps degradation levels by rebuilding the test dataset per level with a
fixed ``deg_level`` (deterministic per-sample seeds make D0/degraded draws
of the same item pair up).  Writes one prediction JSON per (branch, level)
under ``<out_dir>/pred.<branch>.<level>.json`` in COCO result format, then
prints per-branch AP via aodet.evaluation.per_branch_metric.  Evaluation
reads only saved files, so analysis can be re-run without re-inference.

The DetDataSample metainfo carries the COCO image id of each side (set by
AOPackPairedInputs from frame_meta.csv), and category indices in
predictions are the head's label space = position in the GT json's
categories list (PairedSonarCameraDataset maps them the same way).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# register aodet modules (no Runner.from_cfg here to rely on custom_imports)
import aodet.datasets  # noqa: E402,F401
import aodet.models  # noqa: E402,F401
from aodet.cuda_compat import install_nms_cpu_fallback  # noqa: E402

# branch name -> GT side (both sonar heads evaluate against sonar.json)
_SIDE = {"camera": "camera", "sonar": "sonar", "sonar_uncoupled": "sonar"}


def to_coco_results(pred_samples, cat_id_map):
    """DetDataSample list -> COCO result dicts (xywh, score)."""
    results = []
    for ds in pred_samples:
        image_id = ds.metainfo.get("coco_image_id")
        if image_id is None:
            continue  # not matched to the GT json (shouldn't happen)
        pred = ds.pred_instances
        boxes = pred.bboxes.cpu().numpy() if torch.is_tensor(pred.bboxes) \
            else np.asarray(pred.bboxes)
        scores = pred.scores.cpu().numpy() if torch.is_tensor(pred.scores) \
            else np.asarray(pred.scores)
        labels = pred.labels.cpu().numpy() if torch.is_tensor(pred.labels) \
            else np.asarray(pred.labels)
        for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels):
            results.append(dict(
                image_id=int(image_id),
                category_id=cat_id_map[int(l)],
                bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                score=float(s)))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--levels", nargs="+",
                        default=["d0", "d1", "d2", "d3", "d4", "dinf"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    parser.add_argument(
        "--cfg-options", nargs="+", action=DictAction, default=None,
        help="override config keys, e.g. the LOO-CV recording filter")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    init_default_scope(cfg.get("default_scope", "mmdet"))
    install_nms_cpu_fallback()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    from mmdet.registry import MODELS as DET_MODELS
    model = DET_MODELS.build(cfg.model)
    # torch>=2.6 defaults torch.load to weights_only=True, and mmengine
    # 0.10.7 passes no override; its checkpoints embed HistoryBuffer and
    # numpy objects in meta, so the weights-only unpickler rejects them.
    # These checkpoints are this repo's own local artifacts -> load them
    # with weights_only=False (PyTorch's option (1) for trusted files).
    _orig_load = torch.load

    def _load_trusted(*a, **k):
        k.setdefault("weights_only", False)
        return _orig_load(*a, **k)

    torch.load = _load_trusted
    try:
        load_checkpoint(model, args.checkpoint, map_location="cpu")
    finally:
        torch.load = _orig_load
    model = model.to(args.device).eval()

    dl_cfg = cfg.test_dataloader
    dl_cfg.shuffle = False
    root = dl_cfg.dataset.data_root

    from aodet.evaluation.per_branch_metric import (
        blob_eval_img_ids, evaluate_prediction_file, filter_gt_sources)

    summary = {}
    for level in args.levels:
        dl_cfg.dataset.deg_level = level
        dl = Runner.build_dataloader(dl_cfg)
        dataset = dl.dataset
        # mmengine serializes dataset items (data_list is emptied), so read
        # per-item info through the public accessor in both modes
        infos = [dataset.get_data_info(i) for i in range(len(dataset))]
        # per-side evaluated image ids + label-space -> COCO category ids
        img_ids, cat_maps = {}, {}
        for side in ("camera", "sonar"):
            with open(os.path.join(root, f"{side}.json")) as f:
                g = json.load(f)
            img_ids[side] = sorted(int(d[f"coco_{side}_image_id"])
                                   for d in infos
                                   if d.get(f"coco_{side}_image_id")
                                   is not None)
            cat_maps[side] = {i: c["id"] for i, c in
                              enumerate(g["categories"])}
        preds = None
        for data in dl:
            prepped = model.data_preprocessor(data, False)
            with torch.no_grad():
                out = model.predict(prepped["inputs"],
                                    prepped["data_samples"])
            if preds is None:
                preds = {b: [] for b in out}
            for branch, ds_list in out.items():
                preds[branch].extend(ds_list)
        if preds is None:
            raise RuntimeError("test dataloader produced no batches")
        for branch, samples in preds.items():
            side = _SIDE[branch]
            path = os.path.join(out_dir, f"pred.{branch}.{level}.json")
            with open(path, "w") as f:
                json.dump(to_coco_results(samples, cat_maps[side]), f)
            with open(os.path.join(root, f"{side}.json")) as f:
                gt = json.load(f)
            variants = [("all", gt, img_ids[side])]
            if side == "sonar":
                # blob view: measured echoes only; projected-only frames
                # are excluded, not turned into negatives (review §4.2)
                variants.append(("blob", filter_gt_sources(gt, ("blob",)),
                                 blob_eval_img_ids(gt, img_ids[side])))
            for gt_name, gt_v, ids in variants:
                m = evaluate_prediction_file(gt_v, path, img_ids=ids)
                summary[(branch, level, gt_name)] = m
                tag = f" gt={gt_name}" if gt_name != "all" else ""
                print(f"[{branch} @ {level}{tag}] AP={m['AP']:.4f} "
                      f"AP50={m['AP50']:.4f} AR@100={m['AR@100']:.4f}")

    # machine-readable summary next to the prediction files
    with open(os.path.join(args.out_dir, "ap_summary.json"), "w") as f:
        json.dump([{"branch": b, "level": lv, "gt": g, **m}
                   for (b, lv, g), m in sorted(summary.items())], f, indent=2)


if __name__ == "__main__":
    main()
