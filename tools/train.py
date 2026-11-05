#!/usr/bin/env python
"""Train wrapper around mmengine Runner (mmdet-style).

Adds aodet to custom_imports automatically and stamps every run directory
with config, git hash, seed and environment summary (setup doc §4.4(7)).

``--val-interval N`` (default 0 = off) appends aodet.engine.PeriodicValHook:
every N epochs, val losses + per-branch AP50 / mAP / class-agnostic AP50 /
top-1 class accuracy on the val split at d0, written to
``<work_dir>/val_curve.jsonl``.  It never reads test and restores all RNG
states, so the training stream is the same with or without it.
"""

import argparse
import os
import subprocess
import sys
import tempfile

from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cfg-options", nargs="+", action=DictAction, default=None)
    parser.add_argument(
        "--val-interval", type=int, default=0,
        help="evaluate on val every N epochs (0 = off); see module doc")
    parser.add_argument(
        "--val-split", default="val",
        help="frame_meta split for --val-interval (test is refused)")
    args = parser.parse_args()

    # make aodet importable before Config.fromfile — mmengine resolves
    # custom_imports inside fromfile, so the repo root must be on sys.path
    # at that point (matters when launched from another cwd)
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))

    from aodet.torch_load_compat import trusted_torch_load
    import aodet.engine  # noqa: F401  (registers PeriodicValHook)

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get("default_scope", "mmdet"))

    cfg.default_hooks = cfg.get("default_hooks", {})
    cfg.work_dir = args.work_dir or os.path.join(
        "work_dirs", os.path.splitext(os.path.basename(args.config))[0]
        + f"_seed{args.seed}")
    cfg.randomness = dict(seed=args.seed, deterministic=False)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if args.val_interval:
        hooks = list(cfg.get("custom_hooks", []) or [])
        hooks.append(dict(type="PeriodicValHook", interval=args.val_interval,
                          split=args.val_split))
        cfg.custom_hooks = hooks

    runner = Runner.from_cfg(cfg)
    os.makedirs(runner.work_dir, exist_ok=True)
    with open(os.path.join(runner.work_dir, "run_meta.txt"), "w") as f:
        f.write(f"config={args.config}\nseed={args.seed}\n"
                f"git={_git_hash()}\npython={sys.version}\n")
    # resume() and init-weight loads go through torch.load inside
    # runner.train(); mmengine 0.10.7 passes no weights_only override and
    # torch>=2.6 defaults it to True, which rejects the HistoryBuffer /
    # numpy objects in our own checkpoints (2026-11-02 tau4 incident).
    # All loads here are trusted local artifacts -> scoped default flip.
    with trusted_torch_load():
        runner.train()


if __name__ == "__main__":
    main()
