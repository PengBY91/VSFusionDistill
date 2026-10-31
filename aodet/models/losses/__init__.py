"""Distillation losses."""

from .branch_kd import (branch_kd_losses, feature_kd_loss, objectness_kd,
                        score_kd)
from .cross_plane_mil import CrossPlaneMILLoss

__all__ = ["branch_kd_losses", "feature_kd_loss", "objectness_kd",
           "score_kd", "CrossPlaneMILLoss"]
