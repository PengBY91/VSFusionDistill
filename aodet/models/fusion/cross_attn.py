"""Cross-attention fusion in the CAFusion *direction*: sonar queries camera.

F2 Q4 asks whether this mechanism drags the sonar branch down when the
camera degrades.  RSFusionDet is unpublished, so this is a clean-room
implementation of the direction (Sonar2RGB: sonar features attend to camera
features), not a reproduction of their deformable operator:

    fused_l = son_l + MultiHeadAttention(q=son_l, k=v=cam_l)

per feature level, with camera features bilinearly aligned to the sonar
grid.  Under D∞ the camera features become garbage and the residual path is
exactly the mechanism whose damage RQ-0 measures.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional


class CrossAttnFusion(nn.Module):
    def __init__(
        self,
        cam_channels: List[int],
        sonar_channels: List[int],
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert len(cam_channels) == len(sonar_channels)
        self.levels = len(cam_channels)
        self.cam_proj = nn.ModuleList(
            [nn.Conv2d(c, embed_dim, 1) for c in cam_channels])
        self.sonar_proj = nn.ModuleList(
            [nn.Conv2d(c, embed_dim, 1) for c in sonar_channels])
        self.attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout,
                                  batch_first=True)
            for _ in cam_channels
        ])
        self.norm = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for _ in cam_channels])

    def forward(
        self, cam_feats: Optional[List[Tensor]], son_feats: List[Tensor]
    ) -> List[Tensor]:
        if cam_feats is None:
            # D∞ fallback: no camera features at all.  The fused head then
            # sees the sonar features alone through the same projection as
            # the attention queries — i.e. the residual branch with the
            # attention term zeroed.  Channels match the normal path
            # (embed_dim), so the head stays valid.
            return [proj(sf) for proj, sf in zip(self.sonar_proj,
                                                 son_feats)]
        out: List[Tensor] = []
        for i, (cf, sf) in enumerate(zip(cam_feats, son_feats)):
            b, _, h, w = sf.shape
            if cf.shape[-2:] != sf.shape[-2:]:
                cf = nn.functional.interpolate(
                    cf, size=(h, w), mode="bilinear", align_corners=False)
            q = self.sonar_proj[i](sf).flatten(2).transpose(1, 2)  # B,HW,E
            kv = self.cam_proj[i](cf).flatten(2).transpose(1, 2)  # B,HW,E
            attended, _ = self.attn[i](q, kv, kv, need_weights=False)
            fused = self.norm[i](q + attended)                    # residual
            out.append(fused.transpose(1, 2).reshape(b, -1, h, w))
        return out
