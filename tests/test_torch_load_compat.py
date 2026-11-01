"""Save-then-resume smoke for the torch>=2.6 weights_only flip (2026-11-02).

The tau4 incident (doc/logs/2026-11-02-tau4-resume-weights-only.md):
mmengine 0.10.7 calls ``torch.load`` without a ``weights_only`` override,
torch>=2.6 defaults it to True, and mmengine checkpoints carry
HistoryBuffer/numpy objects in ``meta`` -> ``resume()`` dies with
UnpicklingError before the first resumed epoch.  aodet.torch_load_compat
fixes it with a scoped default flip; these tests keep that guarantee:

- the wrapper injects ``weights_only=False`` only as a default (explicit
  arguments pass through, the sandbox-forcing env var wins, the original
  ``torch.load`` is restored on scope exit);
- an mmengine-shaped checkpoint (HistoryBuffer + numpy objects in meta,
  tensors in state_dict) round-trips through the exact CheckpointLoader
  path ``Runner.resume()`` uses;
- the negative control documents that the bare default rejects such a
  payload on torch>=2.6 (skipped below 2.6, where the default is False).

复盘 §8 机制 5：没执行过的路径先冒烟——resume 路径由此获得测试覆盖。
"""

import pickle

import numpy as np
import pytest
import torch
from mmengine.logging import HistoryBuffer
from mmengine.runner.checkpoint import _load_checkpoint

from aodet.torch_load_compat import trusted_torch_load

_TORCH_GE_26 = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 6)


def _incident_shaped_checkpoint(path):
    """Write a checkpoint shaped like mmengine's own resume artifacts."""
    hb = HistoryBuffer()
    hb.update(np.float64(0.42))
    hb.update(np.float64(0.17))
    ckpt = {
        "state_dict": {"layer.weight": torch.ones(3, 3)},
        "optimizer": {"momentum": np.array([0.9, 0.99])},
        "meta": {"log_scalars": {"loss": hb}, "seed": np.int64(0)},
        "epoch": 10,
    }
    torch.save(ckpt, str(path))
    return ckpt


def test_wrapper_defaults_weights_only_off(tmp_path):
    p = tmp_path / "t.pth"
    torch.save({"a": torch.ones(2)}, str(p))
    with trusted_torch_load():
        out = torch.load(str(p))
    assert torch.equal(out["a"], torch.ones(2))


def test_wrapper_restores_torch_load(tmp_path):
    p = tmp_path / "t.pth"
    torch.save({"a": 1}, str(p))
    orig = torch.load
    with trusted_torch_load():
        pass
    assert torch.load is orig


def test_wrapper_keeps_explicit_argument(tmp_path):
    p = tmp_path / "t.pth"
    _incident_shaped_checkpoint(p)
    with trusted_torch_load():
        with pytest.raises(Exception, match="[Ww]eights only"):
            torch.load(str(p), weights_only=True)


def test_wrapper_respects_forced_sandbox(tmp_path, monkeypatch):
    p = tmp_path / "t.pth"
    _incident_shaped_checkpoint(p)
    monkeypatch.setenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "1")
    if not _TORCH_GE_26:
        pytest.skip("weights_only default is False below torch 2.6")
    with trusted_torch_load():
        with pytest.raises(Exception, match="[Ww]eights only"):
            torch.load(str(p))


def test_resume_path_roundtrip(tmp_path):
    """The exact save-then-load path Runner.resume() takes."""
    p = tmp_path / "resume_smoke.pth"
    ref = _incident_shaped_checkpoint(p)
    with trusted_torch_load():
        ckpt = _load_checkpoint(str(p), map_location="cpu")
    assert torch.equal(ckpt["state_dict"]["layer.weight"],
                       ref["state_dict"]["layer.weight"])
    assert ckpt["epoch"] == 10
    # the two failure-mode cargos survive: HistoryBuffer in log_scalars
    # and numpy objects in optimizer/meta
    hb = ckpt["meta"]["log_scalars"]["loss"]
    assert isinstance(hb, HistoryBuffer)
    assert np.allclose(ckpt["optimizer"]["momentum"], [0.9, 0.99])
    assert int(ckpt["meta"]["seed"]) == 0


@pytest.mark.skipif(not _TORCH_GE_26,
                    reason="weights_only defaults to False below torch 2.6")
def test_bare_default_rejects_mmengine_checkpoint(tmp_path):
    """Negative control: without the wrapper, resume would die (tau4)."""
    p = tmp_path / "resume_smoke.pth"
    _incident_shaped_checkpoint(p)
    with pytest.raises((pickle.UnpicklingError, RuntimeError)) as ei:
        _load_checkpoint(str(p), map_location="cpu")
    assert "weights only" in str(ei.value).lower()
