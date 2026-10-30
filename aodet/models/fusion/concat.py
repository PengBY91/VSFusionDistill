"""Channel-concatenation fusion: the simplest strong baseline (F2 §5.1)."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional


class ConcatFusion(nn.Module):
    """Per-level 1x1 projections, channel concat, 1x1 reduce.

    Camera and sonar necks may have different channel counts at the same
    level; each side is projected to ``out_channels`` first.  No gate, no
    attention — this is the "simple concat" option named in F2 §5.1.
    """

    def __init__(self, cam_channels: List[int], sonar_channels: List[int],
                 out_channels: int = 256):
        super().__init__()
        assert len(cam_channels) == len(sonar_channels)
        self.cam_proj = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1) for c in cam_channels])
        self.sonar_proj = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1) for c in sonar_channels])
        self.reduce = nn.ModuleList(
            [nn.Conv2d(2 * out_channels, out_channels, 1)
             for _ in cam_channels])

    def forward(
        self, cam_feats: Optional[List[Tensor]], son_feats: List[Tensor]
    ) -> List[Tensor]:
        if cam_feats is None:
            # concat cannot fall back: with one argument missing there is
            # nothing to concatenate.  Train with dinf_mode="zero" (dense
            # batches) or use NoneFusion/CrossAttnFusion for the drop path.
            raise ValueError(
                "ConcatFusion got no camera features (D-inf drop); use "
                "dinf_mode='zero' or a fusion with a sonar-only fallback")
        assert len(cam_feats) == len(son_feats)
        out: List[Tensor] = []
        for i, (cf, sf) in enumerate(zip(cam_feats, son_feats)):
            if cf.shape[-2:] != sf.shape[-2:]:
                # align camera to sonar grid (sizes can differ after
                # independent per-stream resizing)
                cf = nn.functional.interpolate(
                    cf, size=sf.shape[-2:], mode="bilinear",
                    align_corners=False)
            out.append(self.reduce[i](
                torch.cat([self.cam_proj[i](cf), self.sonar_proj[i](sf)],
                          dim=1)))
        return out
