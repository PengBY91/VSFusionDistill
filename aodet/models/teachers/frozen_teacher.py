"""Frozen teacher wrapper (setup doc §4.4(4): online first, cache later).

The teacher is a *two-stream fusion* teacher (M2/M3: it sees the clear
camera; M1: it sees the same degraded camera as the student).  It runs
inside ``torch.no_grad``/eval during the student's training; its head
outputs (cls_score, bbox_pred, objectness) per branch feed the KD losses.

The offline teacher cache (§5.2 of the F2 draft) is a pure optimisation:
it must reproduce these online outputs elementwise within tolerance before
it may be enabled (tests/test_teacher_cache.py, not yet written — the cache
itself is post-S0 work).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

__all__ = ["FrozenTeacher"]


class FrozenTeacher(nn.Module):
    """Builds a TwoStreamYOLOX teacher and freezes it.

    cfg keys:
        teacher: model config dict (usually TwoStreamYOLOX with YOLOX-L
            streams) — mutually exclusive with ``teacher_checkpoint`` only
            builds, i.e. either the full model config (with embedded init
            checkpoint) or a config + checkpoint path.
        teacher_checkpoint: optional checkpoint path to load after build.
        teacher_input: 'clear' | 'degraded' (consumed by the student, see
            TwoStreamYOLOX._kd_losses).
    """

    def __init__(self, cfg: dict):
        super().__init__()
        from mmdet.registry import MODELS as DET_MODELS
        model_cfg = cfg["teacher"]
        self.model = DET_MODELS.build(model_cfg)
        ckpt = cfg.get("teacher_checkpoint")
        if ckpt:
            from mmengine.runner import CheckpointLoader
            state = CheckpointLoader.load_checkpoint(
                ckpt, map_location="cpu")
            self.model.load_state_dict(state.get("state_dict", state),
                                       strict=False)
        self.model.eval()
        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):  # stay eval forever
        return super().train(False)

    @torch.no_grad()
    def forward_head_outputs(
        self,
        camera: Optional[torch.Tensor],
        sonar: torch.Tensor,
    ) -> Dict[str, dict]:
        """Head outputs per branch on the teacher's own inputs.

        Returns {'camera': (cls, bbox, obj), 'sonar': ...} with per-level
        lists of tensors — the format ``branch_kd_losses`` consumes.
        """
        model = self.model
        out: Dict[str, dict] = {}
        if camera is not None and model.camera_head is not None:
            feats = model.camera_neck(model.camera_backbone(camera))
            out["camera"] = model.camera_head(feats)
        if model.fused_sonar_head is not None or \
                model.uncoupled_sonar_head is not None:
            son_feats = model.sonar_neck(model.sonar_backbone(sonar))
            cam_feats = None
            if camera is not None and model.camera_backbone is not None:
                cam_feats = model.camera_neck(model.camera_backbone(camera))
            head = model.fused_sonar_head or model.uncoupled_sonar_head
            feats = model.fusion(cam_feats, son_feats) \
                if model.fused_sonar_head is not None else son_feats
            out["sonar"] = head(feats)
        return out
