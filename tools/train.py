#!/usr/bin/env python
"""Train wrapper around mmengine Runner (mmdet-style).

Adds aodet to custom_imports automatically and stamps every run directory
with config, git hash, seed and environment summary (setup doc §4.4(7)).
"""

import argparse
import os
import subprocess
import sys
import tempfile

from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

# torch>=2.6 defaults torch.load to weights_only=True, which rejects the
# HistoryBuffer objects mmengine writes into its checkpoints (log_scalars
# meta) -- resume() then dies with UnpicklingError.  Allowlist it for our
# own work-dir checkpoints (trusted; the only other load is the official
# COCO-pretrained state dict, which is plain tensors).
try:
    import torch.serialization
    from mmengine.logging import HistoryBuffer
    torch.serialization.add_safe_globals([HistoryBuffer])
except Exception:  # pragma: no cover - older torch without safe_globals
    pass


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
    args = parser.parse_args()

    # make aodet importable before Config.fromfile — mmengine resolves
    # custom_imports inside fromfile, so the repo root must be on sys.path
    # at that point (matters when launched from another cwd)
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get("default_scope", "mmdet"))

    cfg.default_hooks = cfg.get("default_hooks", {})
    cfg.work_dir = args.work_dir or os.path.join(
        "work_dirs", os.path.splitext(os.path.basename(args.config))[0]
        + f"_seed{args.seed}")
    cfg.randomness = dict(seed=args.seed, deterministic=False)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    runner = Runner.from_cfg(cfg)
    os.makedirs(runner.work_dir, exist_ok=True)
    with open(os.path.join(runner.work_dir, "run_meta.txt"), "w") as f:
        f.write(f"config={args.config}\nseed={args.seed}\n"
                f"git={_git_hash()}\npython={sys.version}\n")
    runner.train()


if __name__ == "__main__":
    main()
