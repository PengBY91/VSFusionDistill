"""Distillation losses."""

from .branch_kd import branch_kd_losses, cls_kl, objectness_mse
from .cross_plane_mil import CrossPlaneMILLoss

__all__ = ["branch_kd_losses", "cls_kl", "objectness_mse", "CrossPlaneMILLoss"]
