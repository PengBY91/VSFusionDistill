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

    ``cam_pool`` decides how the (unregistered) camera map meets the sonar
    grid (review 2026-10-31 §2.4):

    * ``"global"`` (default): the camera map is globally average-pooled and
      broadcast over the sonar grid — no geometric assumption at all.
    * ``"grid"``: legacy behaviour — the camera map is bilinearly resized
      onto the sonar grid and concatenated pixel-wise.  This assumes camera
      (i, j) and sonar (i, j) see the same scene point, the unverified
      CMHD assumption F2 张力 4 argues against; kept only to reproduce the
      earlier baseline.

    ``camera_valid`` (B,) zeroes the camera half of the concat for samples
    without a camera signal (D∞), so the module can take part in modality
    dropout; a whole-batch drop (``cam_feats is None``) does the same.
    """

    def __init__(self, cam_channels: List[int], sonar_channels: List[int],
                 out_channels: int = 256, cam_pool: str = "global"):
        super().__init__()
        assert len(cam_channels) == len(sonar_channels)
        if cam_pool not in ("global", "grid"):
            raise ValueError(f"unknown cam_pool {cam_pool!r}")
        self.cam_pool = cam_pool
        self.out_channels = out_channels
        self.cam_proj = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1) for c in cam_channels])
        self.sonar_proj = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1) for c in sonar_channels])
        self.reduce = nn.ModuleList(
            [nn.Conv2d(2 * out_channels, out_channels, 1)
             for _ in cam_channels])

    def forward(
        self, cam_feats: Optional[List[Tensor]], son_feats: List[Tensor],
        camera_valid: Optional[Tensor] = None,
    ) -> List[Tensor]:
        out: List[Tensor] = []
        for i, sf in enumerate(son_feats):
            s = self.sonar_proj[i](sf)
            if cam_feats is None:
                c = s.new_zeros(s.shape)
            else:
                c = self.cam_proj[i](cam_feats[i])
                if self.cam_pool == "global":
                    c = c.mean(dim=(2, 3), keepdim=True).expand_as(s)
                elif c.shape[-2:] != s.shape[-2:]:
                    c = nn.functional.interpolate(
                        c, size=s.shape[-2:], mode="bilinear",
                        align_corners=False)
                if camera_valid is not None:
                    c = c * camera_valid.to(c.dtype)[:, None, None, None]
            out.append(self.reduce[i](torch.cat([c, s], dim=1)))
        return out
