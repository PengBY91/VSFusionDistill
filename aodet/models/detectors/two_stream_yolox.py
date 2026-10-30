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

from aodet.models.fusion import build_fusion

__all__ = ["TwoStreamYOLOX"]


def _unpack(data_samples, key: str) -> Optional[SampleList]:
    if data_samples is None:
        return None
    return data_samples.get(key)


def _gt_and_metas(samples: Optional[SampleList]):
    """Heads take the DetDataSample list itself (mmdet>=3.3 API); kept as a
    pass-through so the call sites read uniformly."""
    return samples


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
            dict(camera=1.0, fused_sonar=1.0, uncoupled_sonar=1.0).
        distillation: optional dict for the frozen teacher:
            dict(teacher=<model cfg or None if checkpoint-only>,
                 teacher_checkpoint=<path>,
                 teacher_input='clear'|'degraded',
                 obj_weight=..., cls_weight=..., xp_weight=..., tau=...).
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

        self.distillation_cfg = distillation
        self.teacher = None
        if distillation:
            self._init_teacher(distillation)

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
    def loss(self, batch_inputs: Dict[str, torch.Tensor],
             batch_data_samples: Dict[str, SampleList]) -> dict:
        losses: Dict[str, torch.Tensor] = {}
        cam_imgs = batch_inputs.get("camera")
        son_imgs = batch_inputs["sonar"]
        clear_imgs = batch_inputs.get("camera_clear")

        son_feats = self.extract_sonar_feat(son_imgs)
        cam_feats = (self.extract_camera_feat(cam_imgs)
                     if cam_imgs is not None else None)

        head_jobs = []
        if self.camera_head is not None and cam_feats is not None:
            # at D∞-drop the whole batch loses the camera stream: the
            # camera head has nothing to regress on, skip its loss
            head_jobs.append(("camera", self.camera_head, cam_feats,
                              _unpack(batch_data_samples, "camera")))
        if self.uncoupled_sonar_head is not None:
            head_jobs.append(("uncoupled_sonar", self.uncoupled_sonar_head,
                              son_feats,
                              _unpack(batch_data_samples, "sonar")))
        if self.fused_sonar_head is not None:
            fused = self.fusion(cam_feats, son_feats)
            head_jobs.append(("fused_sonar", self.fused_sonar_head, fused,
                              _unpack(batch_data_samples, "sonar")))

        for name, head, feats, samples in head_jobs:
            # mmdet>=3.3 heads consume batch_data_samples directly
            head_losses = head.loss(feats, _gt_and_metas(samples))
            w = self.loss_weights.get(name, 1.0)
            for k, v in head_losses.items():
                losses[f"{name}/{k}"] = v * w

        if self.teacher is not None and self.training:
            losses.update(self._kd_losses(batch_inputs, cam_feats, son_feats,
                                          clear_imgs))

        return losses

    def _kd_losses(self, batch_inputs, cam_feats, son_feats,
                   clear_imgs) -> dict:
        """Online frozen-teacher distillation losses (§4.4(4): online first)."""
        from aodet.models.losses.branch_kd import branch_kd_losses

        t_in = clear_imgs if (self.distillation_cfg.get("teacher_input",
                                                        "clear") == "clear"
                              and clear_imgs is not None) else \
            batch_inputs.get("camera")
        with torch.no_grad():
            t_out = self.teacher.forward_head_outputs(
                camera=t_in, sonar=batch_inputs["sonar"])

        s_out = {}
        if self.camera_head is not None and t_out.get("camera") is not None \
                and cam_feats is not None:
            s_out["camera"] = self.camera_head(cam_feats)
        if self.fused_sonar_head is not None and t_out.get("sonar") is not None:
            fused = self.fusion(cam_feats, son_feats)
            s_out["sonar"] = self.fused_sonar_head(fused)
        elif self.uncoupled_sonar_head is not None and t_out.get("sonar") is not None:
            s_out["sonar"] = self.uncoupled_sonar_head(son_feats)

        kd = branch_kd_losses(s_out, t_out, cfg=self.distillation_cfg)
        return {f"kd/{k}": v for k, v in kd.items()}

    def predict(self, batch_inputs: Dict[str, torch.Tensor],
                batch_data_samples: Dict[str, SampleList],
                rescale: bool = True) -> Dict[str, SampleList]:
        """Per-branch predictions; returns a dict (our tools consume it)."""
        son_imgs = batch_inputs["sonar"]
        cam_imgs = batch_inputs.get("camera")

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
            fused = self.fusion(cam_feats, son_feats)
            out["sonar"] = self._branch_predict(
                self.fused_sonar_head, fused,
                _unpack(batch_data_samples, "sonar"), rescale)
        return out

    def _branch_predict(self, head, feats, samples, rescale) -> SampleList:
        # BaseDenseHead.predict returns list[InstanceData]; wrapping onto
        # the data samples keeps metainfo (sample_id, coco_image_id)
        results = head.predict(feats, samples, rescale=rescale)
        return self.add_pred_to_datasample(samples, results)

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
