"""Scoped ``torch.load(weights_only=False)`` for trusted local checkpoints.

torch >= 2.6 flips ``torch.load``'s default to ``weights_only=True``; mmengine
0.10.7 passes no override, and its checkpoints embed ``HistoryBuffer`` /
numpy objects in ``meta`` -- the weights-only unpickler rejects them with
``UnpicklingError: numpy.core.multiarray._reconstruct``.  A class-level
``add_safe_globals([HistoryBuffer])`` does not fix it: numpy globals inside
the buffer's pickled state stay denied (measured 2026-11-02 while resuming
e10_m3_tau4; incident log ``doc/logs/2026-11-02-tau4-resume-weights-only.md``,
fix history fa2777c -> this module).

Every checkpoint these entry points load is either this repo's own work-dir
artifact or an official upstream state dict of plain tensors, i.e. trusted,
so the weights-only sandbox is not the threat model here.  The wrapper only
fills in the default; a caller passing ``weights_only`` explicitly is never
overridden, and an operator who forces the sandbox on with
``TORCH_FORCE_WEIGHTS_ONLY_LOAD=1`` wins over the wrapper (it becomes a
no-op, loads then fail loudly rather than silently bypassing the policy).

Used by ``tools/train.py`` (around ``runner.train()``: resume and load-from
both go through ``torch.load``) and ``tools/test.py`` (checkpoint load).
``tests/test_torch_load_compat.py`` round-trips an mmengine-shaped
checkpoint through the exact ``CheckpointLoader`` path to keep this
guarantee tested (save-then-resume smoke;复盘 §8 机制 5).
"""

import os
from contextlib import contextmanager

import torch


def _sandbox_forced():
    return os.getenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0").lower() in (
        "1", "true", "yes")


@contextmanager
def trusted_torch_load():
    """Within this scope, ``torch.load`` defaults to ``weights_only=False``.

    Explicit ``weights_only`` arguments pass through untouched; exiting the
    scope restores the original function even on exceptions.
    """
    orig = torch.load

    def _load(*args, **kwargs):
        if "weights_only" not in kwargs and not _sandbox_forced():
            kwargs["weights_only"] = False
        return orig(*args, **kwargs)

    torch.load = _load
    try:
        yield
    finally:
        torch.load = orig
