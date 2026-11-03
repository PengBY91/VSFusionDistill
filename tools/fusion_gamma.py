"""Per-layer |gamma| readout of a CrossAttnFusion checkpoint (A-class
amendment: fusion activity = B2 per-layer |gamma| plus, from the same
checkpoint, the fused-sonar-head d0 vs D-infinity gap read by the eval
battery).

Usage:
    python tools/fusion_gamma.py WORK_DIR [--checkpoint PATH]

Prints mean |gamma| and L2 per fusion level from the newest checkpoint
(or an explicit one).  Level order follows the state dict (P3/P4/P5).
Exit 1 if the checkpoint has no fusion gamma keys (e.g. a B0 run).
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from aodet.torch_load_compat import trusted_torch_load  # noqa: E402


def newest_checkpoint(work_dir: str) -> str:
    if os.path.exists(os.path.join(work_dir, "last_checkpoint")):
        with open(os.path.join(work_dir, "last_checkpoint")) as f:
            rel = f.read().strip()
        full = os.path.join(work_dir, rel)
        if os.path.exists(full):
            return full
    cands = sorted(glob.glob(os.path.join(work_dir, "epoch_*.pth")))
    if not cands:
        raise SystemExit(f"no checkpoint under {work_dir}")
    return cands[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("work_dir")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    ckpt = args.checkpoint or newest_checkpoint(args.work_dir)
    with trusted_torch_load():
        sd = torch.load(ckpt, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]

    levels = {}
    for key, val in sd.items():
        # model.fusion.gamma.<i> (ParameterList registers as .gamma.<i>)
        if ".fusion.gamma." in key or key.startswith("fusion.gamma."):
            idx = int(key.rsplit(".", 1)[-1])
            levels.setdefault(idx, []).append(val.detach().float().abs())

    if not levels:
        raise SystemExit(f"{ckpt}: no fusion gamma keys (not a fusion run?)")

    name = os.path.basename(ckpt)
    for idx in sorted(levels):
        parts = torch.cat(levels[idx])
        print(f"{name} level{idx}: mean|gamma|={parts.mean().item():.6f} "
              f"L2={parts.norm().item():.6f} n={parts.numel()}")


if __name__ == "__main__":
    main()
