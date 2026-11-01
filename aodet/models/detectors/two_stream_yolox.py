"""Two-stream YOLOX with three heads (F2 §5.1, setup doc R1–R2).

Topology::

                camera img ──> cam backbone ──> cam neck ─┐                    │                                          ├─ fusion ──> fused sonar head
                sonar  img ──> son backbone ──> son neck ─┤                                           └──────────> uncoupled sonar head (never sees camera)
                                                        └──────────> camera head

* The **uncoupled sonar head** is served by pre-fusion sonar features only.
  It carries the "decoupled bookkeeping" of 张力 1 and the D∞ fallback, and
  is reported separately at evaluation time (F2 §2.3).
* The **camera stream is optional** (``camera_backbone=None`` gives B0).
* **No Mosaic/MixUp** — geometric augmentation is shared/synchronized at the
  dataset level (setup doc §4.4(2)); nothing here augments.
* Distillation: an optional frozen teacher (bigger or same-capacity two-
  stream YOLOX) runs online under ``torch.no_grad`` on the *clear* camera
  frames (or the same degraded frames for M1); its head outputs feed the KD
  losses (branch_kd / cross_plane_mil).  Offline caching is a later
  optimisation and must pass an equivalence test first (§4.4(4)).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from mmdet.structures import DetDataSample, SampleList
from mmengine.config import ConfigDict
from mmdet.registry import MODELS as DET_MODELS
from mmdet.models.detectors import BaseDetector
from mmdet.models.utils import unpack_gt_instances

from aodet.models.fusion import build_fusion

__all__ = ["TwoStreamYOLOX"]


def _unpack(data_samples, key: str) -> Optional[SampleList]:
    if data_samples is None:
        return None
    return data_samples.get(key)


@DET_MODELS.register_module()
class TwoStreamYOLOX(BaseDetector):
    """Two-stream detector whose heads are mmdet YOLOX heads.

    Args:
        sonar_backbone/sonar_neck: the always-on sonar stream.
        camera_backbone/camera_neck: optional camera stream (None => B0).
        camera_head: head on camera features (L_det^c).
        fused_sonar_head: head on fused features (L_det^s); requires a
            non-None fusion module and a camera stream.
        uncoupled_sonar_head: head on raw sonar features (L_det^s0).
        fusion: dict building one of aodet.models.fusion.* (or None for
            NoneFusion).
        loss_weights: optional per-head loss weights, e.g.
            dict(camera=1.0, fused_sonar=0.5, uncoupled_sonar=0.5).  The
            two sonar heads share the sonar backbone, so their weights
            should sum to B0's single sonar-head weight (review 2026-10-31
            §1.3).
        fusion_detach_sonar: stop-grad variant — the fusion module sees
            detached sonar features, so the sonar backbone is trained by
            sonar-only supervision and the uncoupled head is independent
            of the camera at training time too (review §2.3).
        distillation: optional dict for the frozen teacher:
            dict(teacher=<model cfg>, teacher_checkpoint=<path>,
                 teacher_input='clear'|'degraded',
                 routes={student branch: teacher branch},
                 obj_weight=..., score_weight=..., bg_floor=...,
                 feat_weight=..., student_feat_channels=[...],
                 teacher_feat_channels=[...],
                 reach=dict(k=..., theta=...) or None,
                 reach_routes=('sonar_uncoupled',)).
            ``teacher_checkpoint`` is required; ``teacher_input='clear'``
            raises when the batch has no clear view (review 2026-11-01).
    """

    def __init__(
        self,
        sonar_backbone: dict,
        sonar_neck: dict,
        camera_backbone: Optional[dict] = None,
        camera_neck: Optional[dict] = None,
        camera_head: Optional[dict] = None,
        fused_sonar_head: Optional[dict] = None,
        uncoupled_sonar_head: Optional[dict] = None,
        fusion: Optional[dict] = None,
        loss_weights: Optional[dict] = None,
        fusion_detach_sonar: bool = False,
        distillation: Optional[dict] = None,
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
        data_preprocessor: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
    ):
        if data_preprocessor is None:
            from aodet.models.data_preprocessors import \
                PairedDataPreprocessor  # lazy: avoid import cycle
            data_preprocessor = dict(type="PairedDataPreprocessor")
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.sonar_backbone = DET_MODELS.build(sonar_backbone)
        self.sonar_neck = DET_MODELS.build(sonar_neck)
        self.camera_backbone = DET_MODELS.build(
            camera_backbone) if camera_backbone else None
        self.camera_neck = DET_MODELS.build(
            camera_neck) if camera_neck else None

        def _build_head(cfg: Optional[dict]):
            if cfg is None:
                return None
            cfg = ConfigDict(cfg)
            # SingleStageDetector injects these into the head; so do we
            cfg.update(train_cfg=ConfigDict(train_cfg or {}),
                       test_cfg=ConfigDict(test_cfg or {}))
            return DET_MODELS.build(cfg)

        self.camera_head = _build_head(camera_head)
        self.fused_sonar_head = _build_head(fused_sonar_head)
        self.uncoupled_sonar_head = _build_head(uncoupled_sonar_head)
        if not any([self.camera_head, self.fused_sonar_head,
                    self.uncoupled_sonar_head]):
            raise ValueError("at least one head must be configured")

        self.fusion = build_fusion(fusion)
        if self.fused_sonar_head is not None and self.camera_backbone is None:
            from aodet.models.fusion import NoneFusion
            if not isinstance(self.fusion, NoneFusion):
                raise ValueError(
                    "fused_sonar_head with a non-trivial fusion module "
                    "needs a camera stream")

        self.loss_weights = dict(camera=1.0, fused_sonar=1.0,
                                 uncoupled_sonar=1.0)
        if loss_weights:
            self.loss_weights.update(loss_weights)
        self.fusion_detach_sonar = fusion_detach_sonar

        self.distillation_cfg = distillation
        self.teacher = None
        self.kd_adapters = None
        if distillation:
            self._init_teacher(distillation)
            if float(distillation.get("feat_weight", 0.0)) > 0:
                s_ch = distillation["student_feat_channels"]
                t_ch = distillation["teacher_feat_channels"]
                # train-only adapters (not part of the deployed student)
                self.kd_adapters = nn.ModuleList(
                    [nn.Conv2d(a, b, 1) for a, b in zip(s_ch, t_ch)])

    # -- teacher -----------------------------------------------------------
    def _init_teacher(self, cfg: dict) -> None:
        from aodet.models.teachers.frozen_teacher import FrozenTeacher
        self.teacher = FrozenTeacher(cfg)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

    # -- features ----------------------------------------------------------
    def extract_camera_feat(self, img: torch.Tensor) -> Optional[List]:
        if self.camera_backbone is None:
            return None
        return self.camera_neck(self.camera_backbone(img))

    def extract_sonar_feat(self, img: torch.Tensor) -> List:
        return self.sonar_neck(self.sonar_backbone(img))

    def extract_feat(self, batch_inputs):
        """BaseDetector compat: the sonar stream is the primary branch."""
        return self.extract_sonar_feat(batch_inputs["sonar"])

    # -- mmdet API ----------------------------------------------------------
    # branch name -> (GT side, loss_weights key, head attribute)
    _BRANCHES = {
        "camera": ("camera", "camera", "camera_head"),
        "sonar": ("sonar", "fused_sonar", "fused_sonar_head"),
        "sonar_uncoupled": ("sonar", "uncoupled_sonar",
                            "uncoupled_sonar_head"),
    }

    def _fuse(self, cam_feats, son_feats, camera_valid=None):
        if self.fusion_detach_sonar:
            son_feats = [f.detach() for f in son_feats]
        return self.fusion(cam_feats, son_feats, camera_valid=camera_valid)

    @staticmethod
    def _det_loss(head, outs, samples) -> dict:
        gts, metas, ignores = unpack_gt_instances(samples)
        return head.loss_by_feat(*outs, gts, metas, ignores)

    def loss(self, batch_inputs: Dict[str, torch.Tensor],
             batch_data_samples: Dict[str, SampleList]) -> dict:
        losses: Dict[str, torch.Tensor] = {}
        cam_imgs = batch_inputs.get("camera")
        son_imgs = batch_inputs["sonar"]
        cam_valid = batch_inputs.get("camera_valid")

        son_feats = self.extract_sonar_feat(son_imgs)
        cam_feats = (self.extract_camera_feat(cam_imgs)
                     if cam_imgs is not None else None)

        # every head runs exactly once; detection and KD share these
        # outputs (review 2026-10-31 §1.6)
        outs: Dict[str, tuple] = {}
        if self.camera_head is not None and cam_feats is not None:
            # at a whole-batch D∞ drop the camera head has nothing to
            # regress on; at D∞-zero its GT is empty (honest negatives)
            outs["camera"] = self.camera_head(cam_feats)
        if self.uncoupled_sonar_head is not None:
            outs["sonar_uncoupled"] = self.uncoupled_sonar_head(son_feats)
        if self.fused_sonar_head is not None:
            outs["sonar"] = self.fused_sonar_head(
                self._fuse(cam_feats, son_feats, cam_valid))

        for name, o in outs.items():
            side, wkey, attr = self._BRANCHES[name]
            head_losses = self._det_loss(getattr(self, attr), o,
                                         _unpack(batch_data_samples, side))
            w = self.loss_weights.get(wkey, 1.0)
            for k, v in head_losses.items():
                losses[f"{wkey}/{k}"] = v * w

        if self.teacher is not None and self.training:
            losses.update(self._kd_losses(batch_inputs, outs, son_feats,
                                          cam_valid))
        return losses

    def _kd_losses(self, batch_inputs, outs, son_feats, cam_valid) -> dict:
        """Online frozen-teacher distillation losses (§4.4(4): online first).

        Every returned key contains "loss": mmengine's parse_losses only
        sums such keys, and the earlier "kd/<branch>/obj" names were logged
        but never back-propagated (review 2026-10-31 §1.1).
        """
        from aodet.models.losses.branch_kd import (branch_kd_losses,
                                                    feature_kd_loss)
        cfg = self.distillation_cfg
        mode = cfg.get("teacher_input", "clear")
        if mode not in ("clear", "degraded"):
            raise ValueError(f"unknown teacher_input {mode!r}")
        if mode == "clear":
            t_in = batch_inputs.get("camera_clear")
            t_valid = None      # the clear view always carries signal
            if t_in is None and \
                    self.teacher.model.camera_backbone is not None:
                # no silent fallback to the degraded view: that would turn
                # an M2/M3 run into M1 (review 2026-11-01 #2)
                raise RuntimeError(
                    "teacher_input='clear' but the batch has no "
                    "camera_clear (is AOStudentCameraView in the pipeline?)")
        else:
            t_in = batch_inputs.get("camera")
            # D∞ black frames must be masked in the teacher's fusion too
            t_valid = cam_valid
        with torch.no_grad():
            t_out, t_feats = self.teacher.forward_head_outputs(
                camera=t_in, sonar=batch_inputs["sonar"], return_feats=True,
                camera_valid=t_valid)

        valid = {}
        if cam_valid is not None:
            valid["camera"] = cam_valid     # no camera KD on D∞ samples
        reach = None
        rcfg = cfg.get("reach")
        if rcfg and "sonar_uncoupled" in t_out:
            # F2 §5.4 reachability: the teacher's SONAR-ONLY head decides
            # whether a target is visible in sonar (review §3.4)
            k = float(rcfg.get("k", 10.0))
            theta = float(rcfg.get("theta", 0.3))
            reach = [torch.sigmoid(k * (o.sigmoid() - theta))
                     for o in t_out["sonar_uncoupled"][2]]
        kd = branch_kd_losses(outs, t_out, cfg=cfg, valid=valid, reach=reach)
        losses = {f"kd/loss_{k}": v for k, v in kd.items()}

        fw = float(cfg.get("feat_weight", 0.0))
        if fw > 0 and self.kd_adapters is not None \
                and "sonar" in t_out:
            fg = [o.sigmoid() for o in t_out["sonar"][2]]
            if reach is not None:
                # the student side is the sonar-only neck: camera-only
                # targets must not be pulled in here either (review
                # 2026-11-01 #4)
                fg = [f * r for f, r in zip(fg, reach)]
            losses["kd/loss_sonar_feat"] = fw * feature_kd_loss(
                son_feats, t_feats["sonar_fused"], self.kd_adapters, fg=fg)
        return losses

    def predict(self, batch_inputs: Dict[str, torch.Tensor],
                batch_data_samples: Dict[str, SampleList],
                rescale: bool = True) -> Dict[str, SampleList]:
        """Per-branch predictions; returns a dict (our tools consume it)."""
        son_imgs = batch_inputs["sonar"]
        cam_imgs = batch_inputs.get("camera")
        cam_valid = batch_inputs.get("camera_valid")

        son_feats = self.extract_sonar_feat(son_imgs)
        cam_feats = (self.extract_camera_feat(cam_imgs)
                     if cam_imgs is not None else None)

        out: Dict[str, SampleList] = {}
        if self.camera_head is not None:
            cam_samples = _unpack(batch_data_samples, "camera")
            if cam_feats is None:
                # D∞ drop: camera branch emits "no detections", not a crash
                out["camera"] = self._empty_predictions(cam_samples)
            else:
                out["camera"] = self._branch_predict(
                    self.camera_head, cam_feats, cam_samples, rescale)
        if self.uncoupled_sonar_head is not None:
            out["sonar_uncoupled"] = self._branch_predict(
                self.uncoupled_sonar_head, son_feats,
                _unpack(batch_data_samples, "sonar"), rescale)
        if self.fused_sonar_head is not None:
            fused = self._fuse(cam_feats, son_feats, cam_valid)
            out["sonar"] = self._branch_predict(
                self.fused_sonar_head, fused,
                _unpack(batch_data_samples, "sonar"), rescale)
        return out

    def _branch_predict(self, head, feats, samples, rescale) -> SampleList:
        # BaseDenseHead.predict returns list[InstanceData]; wrapping onto
        # the data samples keeps metainfo (sample_id, coco_image_id).
        # mmdet's add_pred_to_datasample MUTATES the samples in place and
        # returns them; both sonar heads share one sample list, so writing
        # into the originals would alias the branches (E07: the fused
        # results overwrote the uncoupled ones in the dump, making the two
        # prediction files bit-identical).  Wrap onto fresh samples so each
        # branch owns its output.
        results = head.predict(feats, samples, rescale=rescale)
        fresh = [DetDataSample(metainfo=ds.metainfo) for ds in samples]
        return self.add_pred_to_datasample(fresh, results)

    @staticmethod
    def _empty_predictions(samples: Optional[SampleList]) -> SampleList:
        from mmengine.structures import InstanceData
        out: SampleList = []
        for ds in samples or []:
            new = DetDataSample(metainfo=ds.metainfo)
            pred = InstanceData()
            n = 0
            pred.bboxes = torch.zeros((n, 4), dtype=torch.float32)
            pred.scores = torch.zeros((n,), dtype=torch.float32)
            pred.labels = torch.zeros((n,), dtype=torch.int64)
            new.pred_instances = pred
            out.append(new)
        return out

    # BaseDetector abstract members we do not use
    def _forward(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError
