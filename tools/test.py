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

E11 prereg item 5 (doc/experiments.md §E11): ``--split`` restricts
evaluation (and the prediction dumps) to frames whose frame_meta ``split``
value is in the given comma-separated list — the CLI set replaces the
config's single-value dataset split filter (all loaded frames are still
inferred).  ``ap_breakdown.json`` is written next to ap_summary.json with
the per-recording, per-class decomposition of every pooled number
(ap_by_group_and_category; recording taken from frame_meta).
"""

import argparse
import json
import os
import sys

import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# register aodet modules (no Runner.from_cfg here to rely on custom_imports)
import aodet.datasets  # noqa: E402,F401
import aodet.models  # noqa: E402,F401
from aodet.cuda_compat import install_nms_cpu_fallback  # noqa: E402
from aodet.evaluation.coco_results import to_coco_results  # noqa: E402,F401
from aodet.torch_load_compat import trusted_torch_load  # noqa: E402

# branch name -> GT side (both sonar heads evaluate against sonar.json)
_SIDE = {"camera": "camera", "sonar": "sonar", "sonar_uncoupled": "sonar"}


def filter_infos_by_split(infos, splits):
    """Keep data-info dicts whose frame_meta ``split`` value is in splits.

    ``split`` is an opaque string column (train/val/test today; the E11
    converter adds val/val_id), so callers pass the raw values through.
    """
    wanted = {str(s) for s in splits}
    return [d for d in infos
            if str((d.get("meta") or {}).get("split", "")) in wanted]


def recording_by_image_id(infos, side):
    """coco image id -> frame_meta recording name for one GT side.

    Frames whose meta row carries no recording value are absent from the
    map; the breakdown then pools them into the "all" rows only.
    """
    out = {}
    for d in infos:
        iid = d.get(f"coco_{side}_image_id")
        rec = str((d.get("meta") or {}).get("recording", "") or "")
        if iid is not None and rec:
            out[int(iid)] = rec
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--levels", nargs="+",
                        default=["d0", "d1", "d2", "d3", "d4", "dinf"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    parser.add_argument("--split", default=None,
                        help="comma-separated frame_meta split values to "
                             "evaluate on (e.g. 'val,val_id'); replaces the "
                             "config's dataset split filter (all loaded "
                             "frames are still inferred; predictions "
                             "outside these splits are dropped from the "
                             "dumps); default: no extra filtering")
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
    # torch>=2.6 defaults torch.load to weights_only=True and mmengine
    # 0.10.7 passes no override; mmengine checkpoints embed HistoryBuffer
    # and numpy objects in meta, which the weights-only unpickler rejects.
    # These checkpoints are this repo's own local artifacts -> scoped
    # weights_only=False (aodet.torch_load_compat, shared with train.py).
    with trusted_torch_load():
        load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = model.to(args.device).eval()

    dl_cfg = cfg.test_dataloader
    dl_cfg.shuffle = False
    root = dl_cfg.dataset.data_root
    split_set = None
    if args.split:
        split_set = {s.strip() for s in args.split.split(",") if s.strip()}
        # the CLI split set replaces the config's single-value dataset
        # filter (split=None loads every split; infos/preds are filtered
        # below, so the dumps only carry the requested frames)
        dl_cfg.dataset.split = None

    from aodet.evaluation.per_branch_metric import (
        ap_by_group_and_category, blob_eval_img_ids, evaluate_prediction_file,
        filter_gt_sources)

    summary = {}
    breakdown = []
    for level in args.levels:
        dl_cfg.dataset.deg_level = level
        dl = Runner.build_dataloader(dl_cfg)
        dataset = dl.dataset
        # mmengine serializes dataset items (data_list is emptied), so read
        # per-item info through the public accessor in both modes
        infos = [dataset.get_data_info(i) for i in range(len(dataset))]
        if split_set is not None:
            infos = filter_infos_by_split(infos, split_set)
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
        rec_maps = {side: recording_by_image_id(infos, side)
                    for side in ("camera", "sonar")}
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
            results = to_coco_results(samples, cat_maps[side])
            if split_set is not None:
                keep = set(img_ids[side])
                results = [r for r in results if r["image_id"] in keep]
            with open(path, "w") as f:
                json.dump(results, f)
            with open(path) as f:  # evaluate the saved file, as above
                saved = json.load(f)
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
                for r in ap_by_group_and_category(
                        gt_v, saved, ids,
                        group_of_img=rec_maps[side] or None):
                    breakdown.append({"branch": branch, "level": level,
                                      "gt": gt_name,
                                      "recording": r.pop("group"), **r})

    # machine-readable summary next to the prediction files
    with open(os.path.join(args.out_dir, "ap_summary.json"), "w") as f:
        json.dump([{"branch": b, "level": lv, "gt": g, **m}
                   for (b, lv, g), m in sorted(summary.items())], f, indent=2)
    # per-recording x per-class decomposition of the same numbers
    # ("all" recording/category rows are the pooled views)
    breakdown.sort(key=lambda r: (r["branch"], r["level"], r["gt"],
                                  r["recording"] == "all", r["recording"],
                                  r["category"] == "all", r["category"]))
    with open(os.path.join(args.out_dir, "ap_breakdown.json"), "w") as f:
        json.dump(breakdown, f, indent=2)


if __name__ == "__main__":
    main()
