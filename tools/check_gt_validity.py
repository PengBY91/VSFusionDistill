#!/usr/bin/env python
"""GT validity gate for sonar annotations (F2 E06->E07 criterion 1).

For every sonar annotation, measure two image-level quantities that do
not reuse the calibration pipeline:

* ``dy`` -- distance between the annotation box centre row and the
  brightest compact structure inside a +/-60-row stripe over the box
  columns (5x5-smoothed argmax).  Frames whose stripe peak stays below
  ``--peak-min`` are counted as no-echo and excluded from dy percentiles
  (in dark frames the argmax locks onto the pool-bottom band instead of
  the target -- the failure mode behind the retracted E05 diagnosis).
* ``contrast`` -- mean intensity inside the box minus mean of a
  10-pixel ring around it.

Gates (doc/experiments.md, pre-registered): projected-source dy p50 <=
20 px and mean contrast > +10, evaluated per recording and overall.
Exit code 1 if any gate fails.
"""

import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np

PAD_ROWS = 60
RING = 10
SMOOTH = 5


def measure(img, bbox):
    x, y, w, h = bbox
    H, W = img.shape
    x0, x1 = max(0, int(round(x))), min(W, int(round(x + w)))
    y0, y1 = max(0, int(round(y))), min(H, int(round(y + h)))
    if x1 <= x0 or y1 <= y0:
        return None
    box = img[y0:y1, x0:x1].astype(np.float64)
    cy = 0.5 * (y0 + y1)
    s0, s1 = max(0, y0 - PAD_ROWS), min(H, y1 + PAD_ROWS)
    stripe = cv2.blur(img[s0:s1, x0:x1].astype(np.float64), (SMOOTH, SMOOTH))
    peak = float(stripe.max())
    dy = abs(float(np.argmax(stripe.max(axis=1))) + s0 - cy)
    rx0, rx1 = max(0, x0 - RING), min(W, x1 + RING)
    ry0, ry1 = max(0, y0 - RING), min(H, y1 + RING)
    ring = img[ry0:ry1, rx0:rx1].astype(np.float64)
    ring_mask = np.ones(ring.shape, bool)
    ring_mask[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0] = False
    contrast = float(box.mean() - ring[ring_mask].mean()) if ring_mask.any() else 0.0
    return dy, contrast, peak


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paired", default="data/paired/uxo")
    ap.add_argument("--peak-min", type=float, default=40.0,
                    help="stripe peak below this = no-echo frame, "
                         "excluded from dy percentiles")
    ap.add_argument("--dy-max", type=float, default=20.0)
    ap.add_argument("--contrast-min", type=float, default=10.0)
    ap.add_argument("--out", default=None, help="optional JSON report path")
    args = ap.parse_args()

    with open(os.path.join(args.paired, "sonar.json")) as f:
        sonar = json.load(f)
    files = {im["id"]: os.path.join(args.paired, im["file_name"])
             for im in sonar["images"]}
    rows = defaultdict(list)  # (recording, source) -> [(dy, contrast, peak)]
    for ann in sonar["annotations"]:
        img = cv2.imread(files[ann["image_id"]], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SystemExit(f"missing image for ann {ann['id']}")
        m = measure(img, ann["bbox"])
        if m is None:
            continue
        rec = os.path.basename(files[ann["image_id"]]).rsplit("_", 1)[0]
        rows[(rec, ann.get("source", "blob"))].append(m)

    print(f"{'recording':<22} {'src':<10} {'n':>5} {'dy p50':>7} "
          f"{'dy p90':>7} {'no-echo':>7} {'contrast':>9}")
    ok = True
    report = []
    for (rec, src), ms in sorted(rows.items()):
        dys = np.array([m[0] for m in ms if m[2] >= args.peak_min])
        cons = float(np.mean([m[1] for m in ms]))
        no_echo = sum(m[2] < args.peak_min for m in ms)
        p50 = float(np.percentile(dys, 50)) if len(dys) else float("nan")
        p90 = float(np.percentile(dys, 90)) if len(dys) else float("nan")
        print(f"{rec:<22} {src:<10} {len(ms):>5} {p50:>7.1f} "
              f"{p90:>7.1f} {no_echo:>7} {cons:>+9.1f}")
        report.append(dict(recording=rec, source=src, n=len(ms),
                           dy_p50=p50, dy_p90=p90, no_echo=no_echo,
                           contrast=cons))
        if src == "projected" and len(dys):
            ok &= p50 <= args.dy_max and cons > args.contrast_min
    print("gate:", "PASS" if ok else
          f"FAIL (projected dy p50 <= {args.dy_max} and "
          f"contrast > {args.contrast_min} required per recording)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(gate_pass=bool(ok), rows=report), f, indent=2)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
