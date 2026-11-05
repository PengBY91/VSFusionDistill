"""Training-loop hooks (registered into mmdet's HOOKS registry)."""

from .periodic_val_hook import PeriodicValHook, top1_class_stats

__all__ = ["PeriodicValHook", "top1_class_stats"]
