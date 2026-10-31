"""Branch-wise KD losses on YOLOX head outputs (F2 §5.3, L_KD^c / L_KD^s).

mmdet's YOLOX head ``forward`` returns ``(cls_score, bbox_pred, objectness)``
as per-level lists.  Following CMOLD/CMHD's branch selection, box regression
is never distilled (heterogeneous coordinate systems anyway).  Terms
(review 2026-10-31 §3.1–§3.3):

* ``objectness_kd`` — soft-target BCE on objectness logits, weighted by
  the teacher's foreground probability (floor ``bg_floor``) and normalised
  by the total weight.  This keeps the object-level emphasis of CMOLD and
  the same scale as YOLOX's own obj loss (sum / num_fg).  The earlier
  dense MSE on sigmoids was dominated by ~8400 background cells per image
  and sat 2–3 orders of magnitude below the detection loss.
* ``score_kd`` — per-class BCE on the detection score σ(cls)·σ(obj), the
  quantity YOLOX ranks by, weighted by the teacher's max class score.
  YOLOX trains class logits with per-class sigmoids on positives only, so
  a softmax-KL over every location (the earlier ``cls_kl``) distilled
  unsupervised background logits under a wrong mutual-exclusion model.
* ``feature_kd_loss`` — foreground-masked MSE between an adapter of the
  student's (uncoupled) sonar neck and the teacher's fused sonar features:
  the representation-level route for privileged information (Q5).

``branch_kd_losses`` routes student branches to teacher branches
explicitly, e.g. ``{"sonar_uncoupled": "sonar"}`` distils the teacher's
fused (clear-camera) sonar head into the student's sonar-only head.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["objectness_kd", "score_kd", "feature_kd_loss",
           "branch_kd_losses", "DEFAULT_ROUTES"]

# student branch -> teacher branch
DEFAULT_ROUTES = {"camera": "camera", "sonar": "sonar",
                  "sonar_uncoupled": "sonar"}


def _resize_to(x: Tensor, size) -> Tensor:
    if x.shape[-2:] == tuple(size):
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _sample_mask(w: Tensor, valid: Optional[Tensor]) -> Tensor:
    if valid is None:
        return w
    return w * valid.to(w.dtype).view(-1, *([1] * (w.dim() - 1)))


def objectness_kd(s_obj: Sequence[Tensor], t_obj: Sequence[Tensor],
                  bg_floor: float = 0.01,
                  valid: Optional[Tensor] = None,
                  reach: Optional[Sequence[Tensor]] = None) -> Tensor:
    """Foreground-weighted soft-target BCE on objectness logits.

    Args:
        s_obj / t_obj: per-level objectness logits, each B×1×H×W.
        bg_floor: minimum per-cell weight, so background false positives
            are still pushed down.
        valid: (B,) bool, masks whole samples (e.g. camera KD at D∞).
        reach: optional per-level reachability weights B×1×H×W (§5.4),
            multiplied into the weight.
    """
    n = min(len(s_obj), len(t_obj))
    num = s_obj[0].new_zeros(())
    den = s_obj[0].new_zeros(())
    for lvl, (s, t) in enumerate(zip(s_obj[:n], t_obj[:n])):
        s = _resize_to(s, t.shape[-2:]).float()
        p_t = t.detach().float().sigmoid()
        w = p_t.clamp(min=bg_floor)
        if reach is not None:
            w = w * _resize_to(reach[lvl], t.shape[-2:]).detach()
        w = _sample_mask(w, valid)
        loss = F.binary_cross_entropy_with_logits(s, p_t, reduction="none")
        num = num + (loss * w).sum()
        den = den + w.sum()
    return num / den.clamp(min=1.0)


def score_kd(s_cls: Sequence[Tensor], s_obj: Sequence[Tensor],
             t_cls: Sequence[Tensor], t_obj: Sequence[Tensor],
             bg_floor: float = 0.01,
             valid: Optional[Tensor] = None,
             eps: float = 1e-6) -> Tensor:
    """Per-class BCE on the detection score σ(cls)·σ(obj).

    Weighted by the teacher's max class score (floor ``bg_floor``) so the
    background — whose class logits YOLOX never supervises — contributes
    only through the floor.  Computed in fp32 (BCE on probabilities is not
    autocast-safe).
    """
    n = min(len(s_cls), len(t_cls))
    num = s_cls[0].new_zeros(())
    den = s_cls[0].new_zeros(())
    for sc, so, tc, to in zip(s_cls[:n], s_obj[:n], t_cls[:n], t_obj[:n]):
        size = tc.shape[-2:]
        sc = _resize_to(sc, size).float()
        so = _resize_to(so, size).float()
        t_score = (tc.detach().float().sigmoid()
                   * to.detach().float().sigmoid())          # B,C,H,W
        s_score = (sc.sigmoid() * so.sigmoid()).clamp(eps, 1 - eps)
        w = t_score.max(1, keepdim=True).values.clamp(min=bg_floor)
        w = _sample_mask(w, valid)
        loss = F.binary_cross_entropy(s_score, t_score, reduction="none")
        num = num + (loss * w).sum()
        den = den + w.sum() * sc.shape[1]
    return num / den.clamp(min=1.0)


def feature_kd_loss(s_feats: Sequence[Tensor], t_feats: Sequence[Tensor],
                    adapters, fg: Optional[Sequence[Tensor]] = None,
                    bg_floor: float = 0.05) -> Tensor:
    """Foreground-masked MSE between adapter(student) and teacher features.

    Args:
        s_feats: student per-level features (e.g. uncoupled sonar neck).
        t_feats: teacher per-level features (e.g. fused sonar features).
        adapters: per-level modules mapping student to teacher channels.
        fg: per-level teacher foreground probabilities B×1×H×W.
    """
    n = min(len(s_feats), len(t_feats))
    loss = s_feats[0].new_zeros(())
    for i in range(n):
        t = t_feats[i].detach()
        s = _resize_to(adapters[i](s_feats[i]), t.shape[-2:])
        if fg is not None:
            w = _resize_to(fg[i].detach(), t.shape[-2:]).clamp(min=bg_floor)
        else:
            w = torch.ones_like(t[:, :1])
        per_cell = ((s.float() - t.float()) ** 2).mean(1, keepdim=True)
        loss = loss + (per_cell * w).sum() / w.sum().clamp(min=1.0)
    return loss / max(n, 1)


def branch_kd_losses(
    student_out: Dict[str, tuple],
    teacher_out: Dict[str, tuple],
    cfg: Optional[dict] = None,
    valid: Optional[Dict[str, Tensor]] = None,
    reach: Optional[Sequence[Tensor]] = None,
) -> Dict[str, Tensor]:
    """Compute L_KD for every configured route present on both sides.

    Args:
        student_out / teacher_out: {branch: (cls_score, bbox_pred,
            objectness)} per-level lists.
        cfg: obj_weight (default 1.0), score_weight (default 0.0),
            bg_floor (default 0.01), routes (default DEFAULT_ROUTES).
        valid: {student branch: (B,) bool} sample masks.
        reach: per-level reachability weights applied to sonar routes.

    Returns keys ``<student branch>_obj`` / ``<student branch>_score``.
    The caller prefixes them so they contain ``loss`` (mmengine only sums
    keys containing "loss" — review 2026-10-31 §1.1).
    """
    cfg = cfg or {}
    obj_w = float(cfg.get("obj_weight", 1.0))
    score_w = float(cfg.get("score_weight", cfg.get("cls_weight", 0.0)))
    bg = float(cfg.get("bg_floor", 0.01))
    routes = cfg.get("routes") or DEFAULT_ROUTES
    valid = valid or {}
    losses: Dict[str, Tensor] = {}
    for s_name, t_name in routes.items():
        if s_name not in student_out or t_name not in teacher_out:
            continue
        s_cls, _, s_obj = student_out[s_name]
        t_cls, _, t_obj = teacher_out[t_name]
        v = valid.get(s_name)
        r = reach if s_name.startswith("sonar") else None
        if obj_w > 0:
            losses[f"{s_name}_obj"] = obj_w * objectness_kd(
                s_obj, t_obj, bg_floor=bg, valid=v, reach=r)
        if score_w > 0:
            losses[f"{s_name}_score"] = score_w * score_kd(
                s_cls, s_obj, t_cls, t_obj, bg_floor=bg, valid=v)
    if not losses:
        # surface config mistakes loudly instead of silently distilling
        # nothing
        raise RuntimeError(
            f"no KD route matched: student={list(student_out)} "
            f"teacher={list(teacher_out)} routes={routes}")
    return losses
