"""Frozen teacher wrapper (setup doc §4.4(4): online first, cache later).

The teacher is a *two-stream fusion* teacher (M2/M3: it sees the clear
camera; M1: it sees the same degraded camera as the student).  It runs
inside ``torch.no_grad``/eval during the student's training; its head
outputs (cls_score, bbox_pred, objectness) per branch feed the KD losses.

Branches returned: ``camera``, ``sonar`` (fused head if present, else the
uncoupled head) and ``sonar_uncoupled`` (when the teacher has one).  The
reachability proxy of F2 §5.4 must come from ``sonar_uncoupled``: the fused
head has seen the clear camera, so its objectness measures camera
visibility, not "visible in sonar" (review 2026-10-31 §3.4).

The offline teacher cache (§5.2 of the F2 draft) is a pure optimisation:
it must reproduce these online outputs elementwise within tolerance before
it may be enabled (tests/test_teacher_cache.py, not yet written — the cache
itself is post-S0 work).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

__all__ = ["FrozenTeacher"]

# keys a distilled student's checkpoint carries besides its own weights
_FOREIGN_PREFIXES = ("teacher.", "kd_adapters.")


class FrozenTeacher(nn.Module):
    """Builds a TwoStreamYOLOX teacher and freezes it.

    cfg keys:
        teacher: model config dict (usually TwoStreamYOLOX with YOLOX-L
            streams).
        teacher_checkpoint: checkpoint path loaded after build.  Required
            (review 2026-11-01 #1): mmengine's ``init_weights`` never
            recurses into this plain ``nn.Module``, so without a checkpoint
            the teacher would stay randomly initialised — not even the
            backbone's ``init_cfg=Pretrained`` fires.  Loading is strict
            (review 2026-10-31 §3.5): a key mismatch raises.
        allow_random_teacher: unit-test escape hatch; True skips the
            checkpoint requirement.
        teacher_input: 'clear' | 'degraded' (consumed by the student, see
            TwoStreamYOLOX._kd_losses).
    """

    def __init__(self, cfg: dict):
        super().__init__()
        from mmdet.registry import MODELS as DET_MODELS
        self.model = DET_MODELS.build(cfg["teacher"])
        ckpt = cfg.get("teacher_checkpoint")
        if not ckpt and not cfg.get("allow_random_teacher", False):
            raise ValueError(
                "distillation.teacher_checkpoint is required: a teacher "
                "without one is randomly initialised (set "
                "allow_random_teacher=True only in unit tests)")
        if ckpt:
            import torch
            from mmengine.runner import CheckpointLoader
            # torch>=2.6 defaults torch.load to weights_only=True and
            # mmengine 0.10.7 passes no override; our training checkpoints
            # embed HistoryBuffer/numpy objects in meta, so the strict
            # unpickler rejects them (tools/test.py hits the same wall).
            # The checkpoint is this repo's own local artifact -> load with
            # weights_only=False, PyTorch's option (1) for trusted files.
            _orig_load = torch.load

            def _load_trusted(*a, **k):
                k.setdefault("weights_only", False)
                return _orig_load(*a, **k)

            torch.load = _load_trusted
            try:
                state = CheckpointLoader.load_checkpoint(
                    ckpt, map_location="cpu")
            finally:
                torch.load = _orig_load
            self.load_teacher_state(state.get("state_dict", state))
        self.model.eval()
        for p in self.parameters():
            p.requires_grad = False

    def load_teacher_state(self, state_dict: dict) -> None:
        sd = {k: v for k, v in state_dict.items()
              if not k.startswith(_FOREIGN_PREFIXES)}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "teacher checkpoint does not match the teacher config: "
                f"{len(missing)} missing (e.g. {list(missing)[:5]}), "
                f"{len(unexpected)} unexpected (e.g. {list(unexpected)[:5]})")

    def train(self, mode: bool = True):  # stay eval forever
        return super().train(False)

    @torch.no_grad()
    def forward_head_outputs(
        self,
        camera: Optional[torch.Tensor],
        sonar: torch.Tensor,
        return_feats: bool = False,
        camera_valid: Optional[torch.Tensor] = None,
    ):
        """Head outputs per branch on the teacher's own inputs.

        ``camera_valid`` (B,) is forwarded to the fusion module so a
        teacher fed the student's degraded view (M1) masks D∞ black frames
        exactly as it did in its own training (review 2026-11-01 #3).

        Returns {'camera': (cls, bbox, obj), 'sonar': ...,
        'sonar_uncoupled': ...} with per-level lists of tensors — the
        format ``branch_kd_losses`` consumes.  With ``return_feats`` also
        returns {'sonar_fused': [...], 'sonar_raw': [...]}.
        """
        model = self.model
        out: Dict[str, tuple] = {}
        feats: Dict[str, list] = {}
        cam_feats = None
        if camera is not None and model.camera_backbone is not None:
            cam_feats = model.camera_neck(model.camera_backbone(camera))
            if model.camera_head is not None:
                out["camera"] = model.camera_head(cam_feats)
        son_feats = model.sonar_neck(model.sonar_backbone(sonar))
        feats["sonar_raw"] = son_feats
        if model.uncoupled_sonar_head is not None:
            out["sonar_uncoupled"] = model.uncoupled_sonar_head(son_feats)
        if model.fused_sonar_head is not None:
            fused = model.fusion(cam_feats, son_feats,
                                 camera_valid=camera_valid)
            out["sonar"] = model.fused_sonar_head(fused)
            feats["sonar_fused"] = fused
        elif "sonar_uncoupled" in out:
            out["sonar"] = out["sonar_uncoupled"]
            feats["sonar_fused"] = son_feats
        if return_feats:
            return out, feats
        return out
