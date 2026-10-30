"""Fusion modules for the two-stream detector (F2 §5.1).

Interface: every fusion module maps two multi-level neck feature lists to
one fused multi-level list used by the fused sonar head::

    fused = fusion(cam_feats, son_feats)   # each: list[Tensor B,C,H,W]

RSFusionDet's CAFusion is unpublished, so the cross-attention variant here is
a faithful *directional* implementation of the mechanism F2 Q4 asks about —
sonar features query camera features (Sonar2RGB) — using standard attention,
not a reimplementation of their exact operator.  No gating output is exposed
(F2 §4 张力 5: the gate is structure, not a claim, and never a reliability
probability).

Channel counts are stated explicitly in the config (mmengine-style explicit
over cleverness): e.g. ConcatFusion(cam_channels=[128, 256, 512],
sonar_channels=[128, 256, 512]) for a YOLOX-S × YOLOX-S pair.
"""

from .none_fusion import NoneFusion
from .concat import ConcatFusion
from .cross_attn import CrossAttnFusion

_FUSION_TYPES = {
    "NoneFusion": NoneFusion,
    "ConcatFusion": ConcatFusion,
    "CrossAttnFusion": CrossAttnFusion,
}

__all__ = ["NoneFusion", "ConcatFusion", "CrossAttnFusion", "build_fusion"]


def build_fusion(cfg: dict = None):
    if cfg is None:
        return NoneFusion()
    cfg = dict(cfg)
    t = cfg.pop("type")
    if t not in _FUSION_TYPES:
        raise KeyError(f"unknown fusion {t!r}; have {list(_FUSION_TYPES)}")
    return _FUSION_TYPES[t](**cfg)
