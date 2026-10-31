"""Bearing-axial cross-attention: sonar positions attend to camera COLUMNS.

A monocular camera gives no range, so the only geometrically honest
camera→sonar correspondence is "camera column → sonar bearing, range
unknown" (F2 张力 4).  This module encodes exactly that (review
2026-10-31 §2.2, option B)::

    camera (B,C,Hc,Wc) --attention-pool over H--> (B,Wc,E) column tokens
    sonar  (B,C,Hs,Ws) -> queries (B,HsWs,E)
    logits += bias[sonar col, camera col]
    fused = q + γ·MHA(LN(q)+PE_s, LN(cols)+PE_c, cols)

* ``bias`` is a geometry prior ``-(θ_s - θ_c)² / (2σ²)`` (σ learnable) when
  bearing tables are given; otherwise the module relies on 1D positional
  encodings of the two column axes and learns the correspondence.
* Bearing tables are callables mapping a normalised column coordinate in
  [0, 1] to degrees, e.g. ``aris_beam_table.beam_center_deg`` (scaled to
  the beam count) for the sonar and ``atan((x·W − cx)/fx)`` for a pinhole
  camera — never a linear approximation for a wide-angle GoPro.
* Cost is O(HsWs · Wc) instead of O(HsWs · HcWc).
* Same D∞ contract as CrossAttnFusion: γ zero-init, no output LN, and
  ``cam_feats is None`` ≡ ``camera_valid[i] == False`` ≡ ``fused = q``.

Per-recording calibration (UXO calibrates the camera bearing per
recording) is not wired yet: the tables are dataset-level for now.
"""

import math
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def sine_pe_1d(n: int, dim: int, device, dtype) -> Tensor:
    """(n, dim) 1D sine positional encoding over a normalised axis."""
    if dim % 2:
        raise ValueError(f"embed dim {dim} must be even")
    d = dim // 2
    omega = 1.0 / (10000 ** (torch.arange(d, device=device,
                                          dtype=torch.float32) / d))
    x = (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n
    a = x[:, None] * omega[None] * 2 * math.pi
    return torch.cat([a.sin(), a.cos()], 1).to(dtype)


class BearingAxialFusion(nn.Module):
    def __init__(
        self,
        cam_channels: List[int],
        sonar_channels: List[int],
        embed_dim: int = 128,
        num_heads: int = 4,
        layer_scale_init: float = 0.0,
        sonar_col_bearing: Optional[Callable] = None,
        camera_col_bearing: Optional[Callable] = None,
        sigma_deg_init: float = 3.0,
    ):
        super().__init__()
        assert len(cam_channels) == len(sonar_channels)
        if (sonar_col_bearing is None) != (camera_col_bearing is None):
            raise ValueError("give both bearing tables or neither")
        n = len(cam_channels)
        self.num_heads = num_heads
        self.cam_proj = nn.ModuleList(
            [nn.Conv2d(c, embed_dim, 1) for c in cam_channels])
        self.cam_pool = nn.ModuleList(
            [nn.Conv2d(embed_dim, 1, 1) for _ in range(n)])
        self.sonar_proj = nn.ModuleList(
            [nn.Conv2d(c, embed_dim, 1) for c in sonar_channels])
        self.norm_q = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for _ in range(n)])
        self.norm_kv = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for _ in range(n)])
        self.attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            for _ in range(n)])
        self.gamma = nn.ParameterList([
            nn.Parameter(torch.full((embed_dim,), float(layer_scale_init)))
            for _ in range(n)])
        self.sonar_col_bearing = sonar_col_bearing
        self.camera_col_bearing = camera_col_bearing
        self.log_sigma = nn.Parameter(
            torch.tensor(math.log(float(sigma_deg_init))))

    def _bias(self, ws: int, wc: int, device) -> Optional[Tensor]:
        """(Ws, Wc) additive attention bias from the bearing tables."""
        if self.sonar_col_bearing is None:
            return None
        xs = (np.arange(ws) + 0.5) / ws
        xc = (np.arange(wc) + 0.5) / wc
        ts = torch.as_tensor(np.asarray(self.sonar_col_bearing(xs)),
                             device=device, dtype=torch.float32)
        tc = torch.as_tensor(np.asarray(self.camera_col_bearing(xc)),
                             device=device, dtype=torch.float32)
        d2 = (ts[:, None] - tc[None, :]) ** 2
        return -d2 / (2 * self.log_sigma.exp() ** 2)

    def forward(
        self,
        cam_feats: Optional[List[Tensor]],
        son_feats: List[Tensor],
        camera_valid: Optional[Tensor] = None,
    ) -> List[Tensor]:
        qs = [p(s) for p, s in zip(self.sonar_proj, son_feats)]
        if cam_feats is None:
            return qs
        out: List[Tensor] = []
        for i, (cf, q) in enumerate(zip(cam_feats, qs)):
            b, e, hs, ws = q.shape
            kv = self.cam_proj[i](cf)                           # B,E,Hc,Wc
            a = self.cam_pool[i](kv).softmax(dim=2)             # over Hc
            cols = (kv * a).sum(2).transpose(1, 2)              # B,Wc,E
            wc = cols.shape[1]
            q_tok = q.flatten(2).transpose(1, 2)                # B,HsWs,E
            qn = self.norm_q[i](q_tok)
            kn = self.norm_kv[i](cols)
            # 1D bearing-axis encodings: sonar tokens get their column's
            # code (range is unknown to the camera, so no row code)
            pe_s = sine_pe_1d(ws, e, q.device, qn.dtype)        # Ws,E
            qn = qn + pe_s[None].expand(hs, ws, e).reshape(1, hs * ws, e)
            kn = kn + sine_pe_1d(wc, e, q.device, kn.dtype)[None]
            mask = None
            bias = self._bias(ws, wc, q.device)
            if bias is not None:
                bias = bias[None].expand(hs, ws, wc).reshape(hs * ws, wc)
                mask = bias.to(qn.dtype)[None].expand(
                    b * self.num_heads, -1, -1)
            att, _ = self.attn[i](qn, kn, cols, attn_mask=mask,
                                  need_weights=False)
            att = att * self.gamma[i]
            if camera_valid is not None:
                att = att * camera_valid.to(att.dtype)[:, None, None]
            out.append((q_tok + att).transpose(1, 2).reshape(b, e, hs, ws))
        return out
