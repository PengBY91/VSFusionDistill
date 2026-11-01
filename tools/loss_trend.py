#!/usr/bin/env python
"""Export per-component loss descent curves from an mmengine work_dir.

Reads the newest ``<work_dir>/<timestamp>/vis_data/scalars.json`` (JSON
lines; train rows are the ones carrying ``loss`` and ``epoch``), writes a
CSV with every logged step and every loss component (detection branches
plus ``kd/*`` when present), and prints a compact per-epoch mean table.

Usage::

    python tools/loss_trend.py work_dirs/<run> [--out doc/loss_trend/<run>.csv]

The CSV is the durable record of the training loss log (work_dirs/ is
gitignored); E10 keeps one per arm under doc/loss_trend/.
"""
import argparse
import glob
import json
import os
from collections import defaultdict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("work_dir", help="mmengine work_dir of a finished/running run")
    ap.add_argument("--out", help="CSV output path (default: <work_dir>/loss_trend.csv)")
    args = ap.parse_args()

    cands = sorted(glob.glob(os.path.join(args.work_dir, "2*", "vis_data", "*.json")))
    srcs = [p for p in cands if not p.endswith("config.py")]
    if not srcs:
        raise SystemExit(f"no vis_data json under {args.work_dir}")
    path = srcs[-1]  # newest run in the work_dir

    rows = []
    for line in open(path, errors="ignore"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if "loss" in d and "epoch" in d:  # train rows only
            rows.append(d)
    if not rows:
        raise SystemExit(f"no train rows in {path}")

    comps = sorted({k for d in rows for k in d
                    if k.endswith(("loss", "loss_cls", "loss_bbox", "loss_obj"))
                    or k.startswith("kd/")})
    keep = ["loss"] + [c for c in comps if c != "loss"]

    out = args.out or os.path.join(args.work_dir, "loss_trend.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write("step,epoch," + ",".join(k.replace("/", "_") for k in keep) + "\n")
        for d in rows:
            f.write(f"{d.get('step')},{d.get('epoch', '')},"
                    + ",".join("" if d.get(k) is None else f"{d[k]:.4f}"
                               for k in keep) + "\n")

    by_ep = defaultdict(list)
    for d in rows:
        by_ep[d.get("epoch")].append(d)
    eps = sorted(e for e in by_ep if e is not None)
    sel = ([e for e in eps if e == 1 or e == eps[-1] or e % 10 == 0]
           if len(eps) > 8 else eps)

    print(f"# {args.work_dir}  ({len(rows)} logged steps, epochs {eps[0]}..{eps[-1]},"
          f" {os.path.basename(path)})")
    print(f"{'epoch':>5} {'n':>4}  " + " ".join(f"{k.split('/')[-1][:14]:>14}"
                                                for k in keep))
    for e in sel:
        ds = by_ep[e]
        means = [sum(d[k] for d in ds if k in d)
                 / max(1, sum(1 for d in ds if k in d)) for k in keep]
        print(f"{e:>5} {len(ds):>4}  " + " ".join(f"{m:>14.4f}" for m in means))
    print(f"# csv -> {out}")


if __name__ == "__main__":
    main()
