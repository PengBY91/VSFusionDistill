"""Branch-wise KD losses on YOLOX head outputs (F2 §5.3, L_KD^c / L_KD^s).

mmdet's YOLOX head ``forward`` returns ``(cls_score, bbox_pred, objectness)``
as per-level lists.  Following CMOLD/CMHD's branch selection:

* **objectness only is the default** (CMOLD ablation: objectness alone
  0.794 vs 0.797 full) — MSE on sigmoid maps, averaged over levels;
* **class distribution KL with temperature τ, scaled by τ²** is optional
  (``cls_weight``), off by default;
* **box regression is never distilled** (heterogeneous coordinate systems
  anyway).

Same-modality pairs only: teacher camera head -> student camera head,
teacher sonar head -> student sonar head.  The cross-modal term L_xp lives
in ``cross_plane_mil.py``.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["objectness_mse", "cls_kl", "branch_kd_losses"]


def objectness_mse(s_obj, t_obj) -> Tensor:
    """MSE between sigmoid objectness maps, averaged over levels.

    Args:
        s_obj / t_obj: per-level objectness logits, each B×1×H×W (lists may
            be of different lengths; min length is used).
    """
    n = min(len(s_obj), len(t_obj))
    loss = s_obj[0].new_zeros(())
    for s, t in zip(s_obj[:n], t_obj[:n]):
        if s.shape[-2:] != t.shape[-2:]:
            s = F.interpolate(s, size=t.shape[-2:], mode="bilinear",
                              align_corners=False)
        loss = loss + F.mse_loss(s.sigmoid(), t.sigmoid())
    return loss / max(n, 1)


def cls_kl(s_cls, t_cls, tau: float = 1.0) -> Tensor:
    """Per-location KL(student || teacher) over classes with temperature.

    s_cls / t_cls: per-level logits, B×num_classes×H×W.
    """
    n = min(len(s_cls), len(t_cls))
    loss = s_cls[0].new_zeros(())
    for s, t in zip(s_cls[:n], t_cls[:n]):
        if s.shape[-2:] != t.shape[-2:]:
            s = F.interpolate(s, size=t.shape[-2:], mode="bilinear",
                              align_corners=False)
        B, C, H, W = s.shape
        kl = F.kl_div(
            F.log_softmax(s.flatten(2).transpose(1, 2) / tau, dim=-1),
            F.log_softmax(t.flatten(2).transpose(1, 2) / tau, dim=-1),
            log_target=True, reduction="none",
        ).sum(-1).mean()
        loss = loss + (tau ** 2) * kl
    return loss / max(n, 1)


def branch_kd_losses(
    student_out: Dict[str, tuple],
    teacher_out: Dict[str, tuple],
    cfg: Optional[dict] = None,
) -> Dict[str, Tensor]:
    """Compute L_KD for every branch present on both sides.

    Args:
        student_out / teacher_out: {branch: (cls_score, bbox_pred,
            objectness)}; branches matched by name ("camera"/"sonar").
        cfg: distillation config with obj_weight (default 1.0),
            cls_weight (default 0.0), tau (default 1.0).
    """
    cfg = cfg or {}
    obj_w = float(cfg.get("obj_weight", 1.0))
    cls_w = float(cfg.get("cls_weight", 0.0))
    tau = float(cfg.get("tau", 1.0))
    losses: Dict[str, Tensor] = {}
    for branch in sorted(set(student_out) & set(teacher_out)):
        s_cls, _, s_obj = student_out[branch]
        t_cls, _, t_obj = teacher_out[branch]
        if obj_w > 0:
            losses[f"{branch}/obj"] = obj_w * objectness_mse(s_obj, t_obj)
        if cls_w > 0:
            losses[f"{branch}/cls"] = cls_w * cls_kl(s_cls, t_cls, tau)
    if not losses:
        # surface config mistakes loudly instead of silently distilling
        # nothing
        raise RuntimeError(
            f"no overlapping branches for KD: student={list(student_out)} "
            f"teacher={list(teacher_out)}")
    return losses
