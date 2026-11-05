"""PeriodicValHook (aodet.engine) and tools/train.py --val-interval.

End-to-end on the tiny synthetic two-stream config of test_tools_e2e:
the curve file gets one line per evaluation with per-branch AP / top-1 /
val loss, the EMA weights are the ones evaluated, the sealed test split
is refused, and a run with the hook ends with exactly the same weights as
a run without it (all RNG states restored).
"""

import json
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from aodet.engine import PeriodicValHook, top1_class_stats  # noqa: E402
from tests.test_tools_e2e import _TINY, _make_toy_paired_all  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EMA = ("\ncustom_hooks = [dict(type='EMAHook', ema_type='ExpMomentumEMA', "
        "momentum=0.0001, update_buffers=True, priority=49)]\n")


def test_top1_class_stats_known_answers():
    gt = {1: 1, 2: 2, 3: 1}
    preds = [
        {"image_id": 1, "category_id": 1, "score": 0.9},
        {"image_id": 1, "category_id": 2, "score": 0.5},   # lower: ignored
        {"image_id": 2, "category_id": 1, "score": 0.8},   # wrong class
        {"image_id": 9, "category_id": 1, "score": 0.99},  # no GT: ignored
    ]                                                      # frame 3: none
    s = top1_class_stats(preds, gt)
    assert s["top1_acc"] == pytest.approx(1 / 3)
    assert s["mean_top_score"] == pytest.approx((0.9 + 0.8) / 3)
    assert s["n_gt_frames"] == 3


@pytest.mark.parametrize("kw", [dict(split="test"), dict(split="val,test"),
                                dict(interval=0), dict(level="dinf")])
def test_hook_rejects_bad_args(kw):
    with pytest.raises(ValueError):
        PeriodicValHook(**kw)


def _train(tmp_path, name, *extra):
    data_root = str(tmp_path / "paired")
    if not os.path.exists(data_root):
        _make_toy_paired_all(data_root)
    cfg = tmp_path / "tiny_ema.py"
    cfg.write_text("DATA_ROOT = %r\n" % data_root + _TINY + _EMA)
    work = tmp_path / name
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "tools", "train.py"), str(cfg),
         "--work-dir", str(work), "--seed", "0",
         "--cfg-options", "train_cfg.max_epochs=2", *extra],
        capture_output=True, text=True, cwd=REPO)
    return r, work


def test_val_interval_writes_curve_and_keeps_training_stream(tmp_path):
    r, work = _train(tmp_path, "with_val", "--val-interval", "1",
                     "--val-split", "train")
    assert r.returncode == 0, r.stdout + r.stderr
    lines = [json.loads(x) for x in
             (work / "val_curve.jsonl").read_text().splitlines()]
    assert [x["epoch"] for x in lines] == [1, 2]
    for rec in lines:
        assert rec["split"] == "train" and rec["level"] == "d0"
        assert rec["weights"] == "ema"
        assert set(rec["branches"]) == {"camera", "sonar", "sonar_uncoupled"}
        for m in rec["branches"].values():
            assert 0.0 <= m["AP50"] <= 1.0
            assert 0.0 <= m["AP50_class_agnostic"] <= 1.0
            assert 0.0 <= m["top1_acc"] <= 1.0
            assert m["n_gt_frames"] == 4
            assert m["loss"] > 0 and "loss_cls" in m

    r0, work0 = _train(tmp_path, "no_val")
    assert r0.returncode == 0, r0.stdout + r0.stderr
    assert not (work0 / "val_curve.jsonl").exists()
    a = torch.load(work / "epoch_2.pth", map_location="cpu",
                   weights_only=False)["state_dict"]
    b = torch.load(work0 / "epoch_2.pth", map_location="cpu",
                   weights_only=False)["state_dict"]
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_val_interval_refuses_test_split(tmp_path):
    r, _ = _train(tmp_path, "bad", "--val-interval", "1",
                  "--val-split", "test")
    assert r.returncode != 0
    assert "sealed" in (r.stdout + r.stderr)
