"""In-training validation curve for TwoStreamYOLOX (val split only).

The F2 recipe has no in-loop validation (``val_cfg=None``); evaluation is
offline via tools/test.py.  This hook adds a cheap learning-curve probe:
every ``interval`` epochs it runs the model on the val split at one fixed
degradation level and records, per branch,

* val detection losses (same GT convention as training: sonar blob-only),
* COCO AP / AP50 (class-aware) and class-agnostic AP50, sonar on the
  ``gt=blob`` view exactly like tools/test.py,
* top-1 class accuracy and mean top score on frames with GT (the
  highest-scoring detection of the frame decides; no detection = wrong).

Rules it enforces:

* ``split`` may never contain ``test`` (the test split is sealed);
* the averaged weights are evaluated when an EMAHook is present -- they
  are what the checkpoint stores and tools/test.py reads -- through the
  EMA copy, so the training model's parameters and mode are untouched;
* all RNG states (python, numpy, torch, cuda) are saved before and
  restored after, so a run with the hook trains on the same random
  stream as a run without it.

Output: one JSON line per evaluation appended to
``<work_dir>/val_curve.jsonl`` (the file is the record; stdout is not),
plus a log line and ``val/<branch>/<metric>`` scalars to the visualizer.
These numbers are a training-time curve, not the E11 readout: the ledger
reads tools/test.py on the final checkpoint.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from mmdet.registry import HOOKS
from mmengine.hooks import Hook

# branch -> GT side (both sonar heads evaluate against sonar.json)
_SIDE = {"camera": "camera", "sonar": "sonar", "sonar_uncoupled": "sonar"}
# loss-key prefix (TwoStreamYOLOX loss_weights keys) -> branch name
_LOSS_PREFIX = {"camera": "camera", "fused_sonar": "sonar",
                "uncoupled_sonar": "sonar_uncoupled"}


def top1_class_stats(preds: Sequence[Dict[str, Any]],
                     gt_cat_by_img: Dict[int, int]) -> Dict[str, float]:
    """Top-1 class accuracy and mean top score over frames with GT.

    For each frame in ``gt_cat_by_img`` the highest-scoring detection
    decides the class; a frame without detections counts as wrong with
    top score 0.
    """
    best: Dict[int, Dict[str, Any]] = {}
    for d in preds:
        iid = int(d["image_id"])
        if iid in gt_cat_by_img and (iid not in best
                                     or d["score"] > best[iid]["score"]):
            best[iid] = d
    n = len(gt_cat_by_img)
    if n == 0:
        return {"top1_acc": float("nan"), "mean_top_score": float("nan"),
                "n_gt_frames": 0}
    right = sum(1 for i, c in gt_cat_by_img.items()
                if i in best and int(best[i]["category_id"]) == c)
    score = sum(float(best[i]["score"]) for i in best) / n
    return {"top1_acc": right / n, "mean_top_score": score,
            "n_gt_frames": n}


@HOOKS.register_module()
class PeriodicValHook(Hook):
    """Every ``interval`` epochs: val losses + per-branch AP / accuracy.

    Args:
        interval: evaluate after every ``interval``-th epoch (and after
            the last epoch).
        split: frame_meta split value to evaluate (default ``val``).
        level: fixed camera degradation level (default ``d0``).
        batch_size / num_workers: override the test dataloader's values
            (its batch 1 exists for the per-sample D-inf drop; d0 does
            not need it).
        out_file: jsonl file name under the work dir.
    """

    priority = "LOW"

    def __init__(self, interval: int = 5, split: str = "val",
                 level: str = "d0", batch_size: int = 8,
                 num_workers: int = 2, out_file: str = "val_curve.jsonl"):
        if interval < 1:
            raise ValueError(f"interval must be >= 1, got {interval}")
        if "test" in {s.strip() for s in str(split).split(",")}:
            raise ValueError("PeriodicValHook never evaluates the sealed "
                             "test split")
        if level == "dinf":
            raise ValueError("level dinf needs batch-1 camera drop; use "
                             "tools/test.py for the D-inf readout")
        self.interval = interval
        self.split = split
        self.level = level
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.out_file = out_file
        self._loader = None
        self._gt: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------ setup
    def _build(self, runner) -> None:
        from mmengine.runner import Runner

        from aodet.cuda_compat import install_nms_cpu_fallback
        from aodet.evaluation.per_branch_metric import filter_gt_sources
        # model.predict below reaches the compiled mmcv NMS; the GPU env
        # builds mmcv CPU-only (see aodet/cuda_compat), so install the
        # same CPU fallback tools/test.py installs at startup.
        install_nms_cpu_fallback()
        dl = copy.deepcopy(runner.cfg.test_dataloader)
        dl.batch_size = self.batch_size
        dl.num_workers = self.num_workers
        dl.persistent_workers = False
        dl.sampler = dict(type="DefaultSampler", shuffle=False)
        ds = dl.dataset
        ds.split = self.split
        ds.deg_level = self.level
        # val loss uses the training GT convention (sonar blob-only)
        train_ds = runner.cfg.train_dataloader.dataset
        ds.sonar_gt_sources = tuple(train_ds.get("sonar_gt_sources",
                                                 ("blob",)))
        self._loader = Runner.build_dataloader(dl)
        root = ds.data_root
        for side in ("camera", "sonar"):
            with open(os.path.join(root, f"{side}.json"),
                      encoding="utf-8") as f:
                self._gt[side] = json.load(f)
        self._gt["sonar_blob"] = filter_gt_sources(self._gt["sonar"],
                                                   ("blob",))
        dataset = self._loader.dataset
        infos = [dataset.get_data_info(i) for i in range(len(dataset))]
        self._ids = {side: sorted(int(d[f"coco_{side}_image_id"])
                                  for d in infos
                                  if d.get(f"coco_{side}_image_id")
                                  is not None)
                     for side in ("camera", "sonar")}
        self._cat_maps = {side: {i: c["id"] for i, c in
                                 enumerate(self._gt[side]["categories"])}
                          for side in ("camera", "sonar")}

    @staticmethod
    def _eval_model(runner):
        """EMA copy when an EMAHook is registered, else the model."""
        for h in runner.hooks:
            ema = getattr(h, "ema_model", None)
            if ema is not None:
                return ema.module, "ema"
        model = runner.model
        return getattr(model, "module", model), "raw"

    # ------------------------------------------------------------- hook
    def after_train_epoch(self, runner) -> None:
        epoch = runner.epoch + 1          # runner.epoch is 0-based here
        if epoch % self.interval and epoch != runner.max_epochs:
            return
        rng = (random.getstate(), np.random.get_state(),
               torch.get_rng_state(),
               torch.cuda.get_rng_state_all()
               if torch.cuda.is_available() else None)
        try:
            rec = self._evaluate(runner, epoch)
        finally:
            random.setstate(rng[0])
            np.random.set_state(rng[1])
            torch.set_rng_state(rng[2])
            if rng[3] is not None:
                torch.cuda.set_rng_state_all(rng[3])
        with open(os.path.join(runner.work_dir, self.out_file), "a",
                  encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        flat = {}
        for b, m in rec["branches"].items():
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    flat[f"val/{b}/{k}"] = v
        runner.visualizer.add_scalars(flat, step=epoch)
        parts = [f"{b}: AP50 {m['AP50']:.4f} AP {m['AP']:.4f} "
                 f"agn50 {m['AP50_class_agnostic']:.4f} "
                 f"top1 {m['top1_acc']:.3f} loss {m.get('loss', 0.0):.4f}"
                 for b, m in rec["branches"].items()]
        runner.logger.info(f"[val {self.split}@{self.level} epoch {epoch} "
                           f"{rec['weights']}] " + " | ".join(parts)
                           + f" ({rec['seconds']:.0f}s)")

    def _evaluate(self, runner, epoch: int) -> Dict[str, Any]:
        from aodet.evaluation.coco_results import to_coco_results
        from aodet.evaluation.per_branch_metric import (blob_eval_img_ids,
                                                        coco_ap,
                                                        collapse_categories)
        t0 = time.time()
        if self._loader is None:
            self._build(runner)
        model, weights = self._eval_model(runner)
        was_training = model.training
        model.eval()
        loss_sum: Dict[str, float] = {}
        n_batches = 0
        preds: Dict[str, List] = {}
        try:
            with torch.no_grad():
                for data in self._loader:
                    prepped = model.data_preprocessor(data, False)
                    losses = model.loss(prepped["inputs"],
                                        prepped["data_samples"])
                    for k, v in losses.items():
                        if torch.is_tensor(v):
                            v = v.sum()
                        loss_sum[k] = loss_sum.get(k, 0.0) + float(v)
                    n_batches += 1
                    out = model.predict(prepped["inputs"],
                                        prepped["data_samples"])
                    for branch, ds_list in out.items():
                        preds.setdefault(branch, []).extend(ds_list)
        finally:
            model.train(was_training)

        branches: Dict[str, Dict[str, Any]] = {}
        for branch, samples in preds.items():
            side = _SIDE[branch]
            res = to_coco_results(samples, self._cat_maps[side])
            if side == "sonar":
                gt = self._gt["sonar_blob"]
                ids = blob_eval_img_ids(self._gt["sonar"], self._ids[side])
            else:
                gt = self._gt["camera"]
                ids = self._ids[side]
            keep = set(ids)
            res = [r for r in res if r["image_id"] in keep]
            with contextlib.redirect_stdout(io.StringIO()):
                m = coco_ap(gt, res, img_ids=ids)
                g1, p1 = collapse_categories(gt, res)
                m_agn = coco_ap(g1, p1, img_ids=ids)
            gt_cat = {int(a["image_id"]): int(a["category_id"])
                      for a in gt["annotations"]
                      if int(a["image_id"]) in keep}
            branches[branch] = {"AP": m["AP"], "AP50": m["AP50"],
                                "AP50_class_agnostic": m_agn["AP50"],
                                "n_pred": len(res),
                                **top1_class_stats(res, gt_cat)}
        for k, v in loss_sum.items():
            prefix, _, name = k.partition("/")
            b = _LOSS_PREFIX.get(prefix)
            if b in branches:
                branches[b][name] = v / max(n_batches, 1)
        for b in branches.values():
            b["loss"] = sum(v for k, v in b.items() if k.startswith("loss_"))
        return {"epoch": epoch, "split": self.split, "level": self.level,
                "weights": weights, "seconds": time.time() - t0,
                "branches": branches}
