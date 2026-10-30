"""No fusion (B5 late fusion / B0 sonar-only): pass sonar features through."""

import torch.nn as nn
from torch import Tensor
from typing import List, Optional


class NoneFusion(nn.Module):
    """Identity fusion: the fused sonar head sees plain sonar features.

    Used by B5 (late fusion: both streams exist, camera features are ignored
    here, matching happens at the result level) and B0 (sonar-only baseline:
    camera stream is off entirely, ``cam_feats is None``).
    """

    def forward(
        self,
        cam_feats: Optional[List[Tensor]],
        son_feats: List[Tensor],
    ) -> List[Tensor]:
        return son_feats
