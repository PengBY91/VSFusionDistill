"""Branch-wise KD losses on YOLOX head outputs (F2 §5.3, L_KD^c / L_KD^s).

mmdet's YOLOX head ``forward`` returns ``(cls_score, bbox_pred, objectness)``
as per-level lists.  Following CMOLD/CMHD's branch selection, box regression
is never distilled (heterogeneous coordinate systems anyway).  Terms
(review 2026-10-31 §3.1–§3.3):

* ``objectness_kd`` — soft-target KL (BCE − H) on objectness logits, weighted by
  the teacher's foreground probability (floor ``bg_floor``) and normalised
  by the total weight.  This keeps the object-level emphasis of CMOLD and
  the same scale as YOLOX's own obj loss (sum / num_fg).  The earlier
  dense MSE on sigmoids was dominated by ~8400 background cells per image
  and sat 2–3 orders of magnitude below the detection loss.
* ``score_kd`` — per-class Bernoulli KL on the detection score σ(cls)·σ(obj), the
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

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["objectness_kd", "score_kd", "feature_kd_loss",
           "branch_kd_losses", "DEFAULT_ROUTES", "DEFAULT_REACH_ROUTES"]

# student branch -> teacher branch
DEFAULT_ROUTES = {"camera": "camera", "sonar": "sonar",
                  "sonar_uncoupled": "sonar"}
# student branches gated by the §5.4 reachability weights
DEFAULT_REACH_ROUTES = ("sonar_uncoupled",)


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
    """Foreground-weighted soft-target KL on objectness logits.

    The per-cell term is BCE(s, p_t) − H(p_t), i.e. the Bernoulli KL: the
    same gradient as soft-target BCE, but 0 at a perfect match, so logged
    KD/detection ratios are not inflated by the teacher's entropy
    (review 2026-11-01 #6).

    Args:
        s_obj / t_obj: per-level objectness logits, each B×1×H×W.
        bg_floor: minimum per-cell weight, so background false positives
            are still pushed down.
        valid: (B,) bool, masks whole samples (e.g. camera KD at D∞).
        reach: optional per-level reachability weights B×1×H×W (§5.4).
            They scale the teacher's foreground weight *before* the floor
            (w = max(p_t·reach, bg_floor)): unreachable targets drop to the
            background weight, while the background push-down keeps the
            same floor on every route (review 2026-11-01 #5).
    """
    n = min(len(s_obj), len(t_obj))
    num = s_obj[0].new_zeros(())
    den = s_obj[0].new_zeros(())
    for lvl, (s, t) in enumerate(zip(s_obj[:n], t_obj[:n])):
        s = _resize_to(s, t.shape[-2:]).float()
        t = t.detach().float()
        p_t = t.sigmoid()
        w = p_t
        if reach is not None:
            w = w * _resize_to(reach[lvl], t.shape[-2:]).detach().float()
        w = _sample_mask(w.clamp(min=bg_floor), valid)
        loss = (F.binary_cross_entropy_with_logits(s, p_t, reduction="none")
                - F.binary_cross_entropy_with_logits(t, p_t,
                                                     reduction="none"))
        num = num + (loss * w).sum()
        den = den + w.sum()
    return num / den.clamp(min=1.0)


def _bernoulli_kl(log_q: Tensor, p: Tensor, eps: float) -> Tensor:
    """KL(Bern(p) || Bern(q)) from log q, without F.binary_cross_entropy
    (which CUDA autocast refuses — review 2026-11-01 #7)."""
    log_1mq = torch.log1p(-log_q.exp().clamp(max=1 - eps))
    log_q = log_q.clamp(min=math.log(eps))
    ce = -(p * log_q + (1 - p) * log_1mq)
    ent = -(torch.special.xlogy(p, p) + torch.special.xlogy(1 - p, 1 - p))
    return ce - ent


def score_kd(s_cls: Sequence[Tensor], s_obj: Sequence[Tensor],
             t_cls: Sequence[Tensor], t_obj: Sequence[Tensor],
             bg_floor: float = 0.01,
             valid: Optional[Tensor] = None,
             reach: Optional[Sequence[Tensor]] = None,
             eps: float = 1e-6) -> Tensor:
    """Per-class Bernoulli KL on the detection score σ(cls)·σ(obj).

    Weighted by the teacher's max class score (floor ``bg_floor``) so the
    background — whose class logits YOLOX never supervises — contributes
    only through the floor.  ``reach`` scales that weight exactly as in
    :func:`objectness_kd`; without it the score term re-imports the
    camera-only targets reach removed from the obj term (review
    2026-11-01 #4).  Computed in fp32 with autocast off.
    """
    n = min(len(s_cls), len(t_cls))
    num = s_cls[0].new_zeros((), dtype=torch.float32)
    den = s_cls[0].new_zeros((), dtype=torch.float32)
    with torch.autocast(device_type=s_cls[0].device.type, enabled=False):
        for lvl, (sc, so, tc, to) in enumerate(
                zip(s_cls[:n], s_obj[:n], t_cls[:n], t_obj[:n])):
            size = tc.shape[-2:]
            sc = _resize_to(sc, size).float()
            so = _resize_to(so, size).float()
            t_score = (tc.detach().float().sigmoid()
                       * to.detach().float().sigmoid())          # B,C,H,W
            log_s = F.logsigmoid(sc) + F.logsigmoid(so)
            w = t_score.max(1, keepdim=True).values
            if reach is not None:
                w = w * _resize_to(reach[lvl], size).detach().float()
            w = _sample_mask(w.clamp(min=bg_floor), valid)
            loss = _bernoulli_kl(log_s, t_score, eps)
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
        reach: per-level reachability weights, applied only to the
            student branches in ``cfg['reach_routes']`` (default: the
            sonar-only head).  The fused student head sees the camera, so
            gating it by "visible in sonar" would discard privileged
            targets it can legitimately learn (review 2026-11-01 #4).

    Returns keys ``<student branch>_obj`` / ``<student branch>_score``.
    The caller prefixes them so they contain ``loss`` (mmengine only sums
    keys containing "loss" — review 2026-10-31 §1.1).
    """
    cfg = cfg or {}
    obj_w = float(cfg.get("obj_weight", 1.0))
    score_w = float(cfg.get("score_weight", cfg.get("cls_weight", 0.0)))
    bg = float(cfg.get("bg_floor", 0.01))
    routes = cfg.get("routes") or DEFAULT_ROUTES
    rr = cfg.get("reach_routes")
    reach_routes = tuple(DEFAULT_REACH_ROUTES if rr is None else rr)
    valid = valid or {}
    losses: Dict[str, Tensor] = {}
    for s_name, t_name in routes.items():
        if s_name not in student_out or t_name not in teacher_out:
            continue
        s_cls, _, s_obj = student_out[s_name]
        t_cls, _, t_obj = teacher_out[t_name]
        v = valid.get(s_name)
        r = reach if s_name in reach_routes else None
        if obj_w > 0:
            losses[f"{s_name}_obj"] = obj_w * objectness_kd(
                s_obj, t_obj, bg_floor=bg, valid=v, reach=r)
        if score_w > 0:
            losses[f"{s_name}_score"] = score_w * score_kd(
                s_cls, s_obj, t_cls, t_obj, bg_floor=bg, valid=v, reach=r)
    if not losses:
        # surface config mistakes loudly instead of silently distilling
        # nothing
        raise RuntimeError(
            f"no KD route matched: student={list(student_out)} "
            f"teacher={list(teacher_out)} routes={routes}")
    return losses
