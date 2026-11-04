"""Cross-attention fusion in the CAFusion *direction*: sonar queries camera.

F2 Q4 asks whether this mechanism drags the sonar branch down when the
camera degrades.  RSFusionDet is unpublished, so this is a clean-room
implementation of the direction (Sonar2RGB: sonar features attend to camera
features), not a reproduction of their deformable operator::

    q_l     = proj_s(son_l)
    fused_l = q_l + γ_l · MHA(LN(q_l) + PE_s, LN(c_l) + PE_c, c_l)

per feature level (review 2026-10-31 §2.1–§2.2):

* **2D sine positional encodings** on both sides — without them the
  attention output is invariant to the order of the camera tokens, so a
  sonar location can only see a bag of camera content and can never learn
  a column↔bearing correspondence.
* **γ (per-channel LayerScale), zero-initialised by default**: the fused
  head starts identical to a sonar-only head and coupling is learned.
* **No LayerNorm on the residual output**, so "no camera" is exactly
  ``fused = q``.  The D∞ drop path (``cam_feats is None``) and the masked
  training path (``camera_valid[i] == False``) are the same function — the
  previous version trained D∞ on black frames through attention + LN and
  tested it on an un-normalised projection the head had never seen.
* Camera features are **not** resampled to the sonar grid: the planes are
  unregistered and attention does not need equal sizes.
* **Sonar query dropout** (E11-4 arm B2b, off by default): in training,
  each sample is selected with probability ``p_sample`` and its projected
  sonar features q get a per-channel Bernoulli(``keep``) mask scaled by
  1/keep, on both the residual and the query.  It only reaches the fused
  sonar head (the uncoupled head reads pre-fusion features), so the fused
  head is pushed to recover from the camera.  Evaluation never drops.  Off
  (``None``) is bit-identical to the plain module and draws no random
  numbers.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional


def sine_pe_2d(h: int, w: int, dim: int, device, dtype) -> Tensor:
    """(h*w, dim) 2D sine positional encoding (DETR-style, normalised)."""
    if dim % 4:
        raise ValueError(f"embed dim {dim} must be divisible by 4")
    d = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(d, device=device,
                                          dtype=torch.float32) / d))
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
    ay = ys[:, None] * omega[None] * 2 * math.pi
    ax = xs[:, None] * omega[None] * 2 * math.pi
    py = torch.cat([ay.sin(), ay.cos()], 1)                  # h, 2d
    px = torch.cat([ax.sin(), ax.cos()], 1)                  # w, 2d
    pe = torch.cat([py[:, None, :].expand(h, w, 2 * d),
                    px[None, :, :].expand(h, w, 2 * d)], -1)
    return pe.reshape(h * w, dim).to(dtype)


class CrossAttnFusion(nn.Module):
    def __init__(
        self,
        cam_channels: List[int],
        sonar_channels: List[int],
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
        layer_scale_init: float = 0.0,
        pos_embed: bool = True,
        sonar_query_dropout: Optional[dict] = None,
    ):
        super().__init__()
        self.sq_drop = None
        if sonar_query_dropout is not None:
            p_s = float(sonar_query_dropout["p_sample"])
            keep = float(sonar_query_dropout["keep"])
            if not (0.0 <= p_s <= 1.0 and 0.0 < keep <= 1.0):
                raise ValueError("sonar_query_dropout needs 0<=p_sample<=1 "
                                 f"and 0<keep<=1, got {sonar_query_dropout}")
            self.sq_drop = (p_s, keep)
        assert len(cam_channels) == len(sonar_channels)
        self.levels = len(cam_channels)
        self.pos_embed = pos_embed
        self.cam_proj = nn.ModuleList(
            [nn.Conv2d(c, embed_dim, 1) for c in cam_channels])
        self.sonar_proj = nn.ModuleList(
            [nn.Conv2d(c, embed_dim, 1) for c in sonar_channels])
        self.norm_q = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for _ in cam_channels])
        self.norm_kv = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for _ in cam_channels])
        self.attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout,
                                  batch_first=True)
            for _ in cam_channels
        ])
        self.gamma = nn.ParameterList([
            nn.Parameter(torch.full((embed_dim,), float(layer_scale_init)))
            for _ in cam_channels])

    def forward(
        self,
        cam_feats: Optional[List[Tensor]],
        son_feats: List[Tensor],
        camera_valid: Optional[Tensor] = None,
    ) -> List[Tensor]:
        qs = [proj(sf) for proj, sf in zip(self.sonar_proj, son_feats)]
        if self.sq_drop is not None and self.training:
            qs = self._drop_sonar_queries(qs)
        if cam_feats is None:
            # D∞ drop: identical to the masked path (attention term = 0)
            return qs
        out: List[Tensor] = []
        for i, (cf, q) in enumerate(zip(cam_feats, qs)):
            b, e, h, w = q.shape
            kv = self.cam_proj[i](cf)
            hc, wc = kv.shape[-2:]
            q_tok = q.flatten(2).transpose(1, 2)                  # B,HW,E
            kv_tok = kv.flatten(2).transpose(1, 2)                # B,HcWc,E
            qn = self.norm_q[i](q_tok)
            kn = self.norm_kv[i](kv_tok)
            if self.pos_embed:
                qn = qn + sine_pe_2d(h, w, e, q.device, qn.dtype)
                kn = kn + sine_pe_2d(hc, wc, e, q.device, kn.dtype)
            att, _ = self.attn[i](qn, kn, kv_tok, need_weights=False)
            att = att * self.gamma[i]
            if camera_valid is not None:
                att = att * camera_valid.to(att.dtype)[:, None, None]
            out.append((q_tok + att).transpose(1, 2).reshape(b, e, h, w))
        return out

    def _drop_sonar_queries(self, qs: List[Tensor]) -> List[Tensor]:
        """Per-sample channel dropout on q (selection shared across
        levels, channel masks drawn per level)."""
        p_s, keep = self.sq_drop
        b, e = qs[0].shape[:2]
        dev, dt = qs[0].device, qs[0].dtype
        sel = (torch.rand(b, device=dev) < p_s)[:, None, None, None]
        out = []
        for q in qs:
            m = (torch.rand(b, e, 1, 1, device=dev) < keep).to(dt) / keep
            out.append(torch.where(sel, q * m, q))
        return out
