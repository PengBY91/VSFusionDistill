"""Dahn UXO (Zenodo 13778485) -> paired format converter.

Source layout (one dir per recording, frame index shared by all
modalities -- verified on sample.7z 2026-10-30, see
doc/logs/2026-10-30-f2-codebase-w1-w3.md)::

    <src_root>/<recording>/
      labels/<idx>.json    ONE box per annotated gopro frame, coordinates
                           in 640x360 space: multiply by 3 for the
                           1920x1080 gopro jpgs (class, x_min, y_min,
                           x_max, y_max)
      gopro/<idx>.jpg      1920x1080 camera frames
      aris_raw/<idx>.pgm   rectangular polar sonar: (780, 128) uint8,
                           rows = range samples (SamplesPerChannel),
                           cols = beams (NumRawBeams) -> x is azimuth
      gantry.csv           per-frame sonar pose: aris_frame_idx, x, y, z
      aris_frame_meta.csv  per-frame ARIS header (WindowStart/WindowLength/
                           TargetRange/... 178 cols)
      aris_file_meta.yaml  per-recording sensor constants
      notes.txt            target/trajectory notes (informational)

Facts baked in from the sample verification:

* The "boxes don't line up with raw GoPro frames" issue is a resolution
  mismatch: labels were drawn on the 1920x1080 downscaled frames
  (prep_8_gopro_downsample) and published divided by 3.  Scaling by 3
  aligns with the *shipped* jpgs; it never aligns with 5.3K originals
  (5312/1920 ~= 2.77).  The converter therefore uses the shipped jpgs
  verbatim and never re-encodes originals.
* Sonar ground-truth boxes do not exist; they are produced later by the
  pose-based projection tool (F2 TODO 14.2), which fills sonar.json
  annotations and pairs.csv.  This converter emits empty sonar
  annotations/pairs and carries every per-frame quantity the projection
  needs (window, gantry pose, sound speed) in frame_meta.csv so the
  paired dataset stays self-contained.
* Unlabeled gopro frames are NOT emitted by default: an absent label
  file cannot be distinguished from "annotator stopped" vs "no target",
  and emitting them would fabricate negative ground truth.  Opt in with
  ``include_unlabeled=True`` after a frame-level review confirms them.

Recording-level splits avoid trajectory leakage.  The sample default
is the E07 split (2026-10-31, doc/experiments.md): test = {100927
mortar arc, 124735 grenade line}.  Each test recording is novel in a
single dimension (100927: unseen recording, geometry seen; 124735:
unseen line geometry, class seen), so test results decompose into
cross-recording vs cross-geometry generalisation.  The previous E02
default (105856 + 124735) put every line recording in test and every
arc recording in train -- geometry fully confounded with the split,
which is why E04/E05 collapsed (aim-mode arcs hold the target at a
near-constant range row while line flybys sweep it across the image;
see doc/logs/2026-10-31-e05-test-ap-zero-diagnosis.md).  The bomb
appears in a single sample recording so test cannot cover it -- the
full dataset has no such restriction.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

from ..paired_format import PairedData, read_paired, write_paired

__all__ = ["convert", "DEFAULT_TEST_RECORDINGS"]

# E07 split (2026-10-31): arcs 180722+123500 and line 105856 in train,
# arc 100927 + line 124735 in test -- each test recording novel in one
# dimension only; superseded the E02 default (105856+124735)
DEFAULT_TEST_RECORDINGS = ("2023-09-21_100927", "2023-09-21_124735")

_REC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}$")
_FRAME_RE = re.compile(r"^\d{4}$")
# per-frame aris_frame_meta.csv columns carried into frame_meta.csv (keeps
# the projection tool off the raw csv); SamplePeriod completes the
# range-row mapping (window_start + row * period * c / 2)
_ARIS_META_COLUMNS = ("FrameTime", "sonarTimeStamp", "WindowStart",
                      "WindowLength", "SamplePeriod", "ReceiverGain",
                      "Intensity", "Focus", "SoundSpeed", "TargetRange",
                      "TargetBearing")

# sorted for a stable category order; the source misspelling of
# "incendiary" is kept verbatim to stay diffable against upstream
CATEGORIES = ("100lbs_aircraft_bomb", "incindiary_grenade", "mortar_shell")
_LABEL_SCALE = 3  # 640x360 label space -> 1920x1080 gopro jpgs


def _read_csv_rows(path: str) -> Dict[int, Dict[str, str]]:
    """Index a csv by its integer-valued frame-index column."""
    out: Dict[int, Dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = "aris_frame_idx" if "aris_frame_idx" in row else "FrameIndex"
            try:
                out[int(row[key])] = row
            except (KeyError, ValueError):
                continue
    return out


def _sensor_constants(rec_dir: str) -> Tuple[int, int]:
    """(beams, range_samples) from aris_file_meta.yaml; (128, 780) fallback."""
    try:
        import yaml
        with open(os.path.join(rec_dir, "aris_file_meta.yaml"),
                  encoding="utf-8") as f:
            meta = yaml.safe_load(f) or {}
        return int(meta.get("NumRawBeams", 128)), \
            int(meta.get("SamplesPerChannel", 780))
    except Exception:
        return 128, 780


def _link_or_copy(src: str, dst: str) -> None:
    """Hardlink when possible (paired tree stays on this machine), else copy."""
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def convert(src_root: str, out_root: str,
            test_recordings=DEFAULT_TEST_RECORDINGS,
            include_unlabeled: bool = False,
            stride: int = 1,
            contact_sheet: Optional[str] = None) -> Dict[str, Any]:
    """Convert one or more UXO recordings into the paired format.

    Returns a per-recording report dict (frames kept / skipped by
    reason, class counts); print it or assert on it in tests.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    cat_id = {name: i + 1 for i, name in enumerate(CATEGORIES)}
    test_set = set(test_recordings)

    camera: Dict[str, Any] = {"images": [], "annotations": [],
                              "categories": [{"id": i, "name": n}
                                             for n, i in cat_id.items()]}
    sonar: Dict[str, Any] = {"images": [], "annotations": [],
                             "categories": [{"id": i, "name": n}
                                            for n, i in cat_id.items()]}
    pairs: List[Dict[str, Any]] = []   # filled by the projection tool
    frame_meta: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {"recordings": {}}

    os.makedirs(os.path.join(out_root, "images", "camera"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "images", "sonar"), exist_ok=True)

    recs = sorted(d for d in os.listdir(src_root)
                  if _REC_RE.match(d)
                  and os.path.isdir(os.path.join(src_root, d)))
    if not recs:
        raise ValueError(
            f"no recording dirs (YYYY-MM-DD_HHMMSS) under {src_root}")

    next_id = 1
    next_ann_id = 1
    for rec in recs:
        rd = os.path.join(src_root, rec)
        stats = {"split": "", "kept": 0, "unlabeled_skipped": 0,
                 "label_without_gopro": 0, "annotated_without_aris": 0,
                 "no_gantry": 0, "stride_skipped": 0, "classes": {}}
        stats["split"] = "test" if rec in test_set else "train"

        def _frames(sub: str, ext: str) -> set:
            d = os.path.join(rd, sub)
            if not os.path.isdir(d):
                return set()
            return {f[:-len(ext) - 1] for f in os.listdir(d)
                    if f.endswith("." + ext) and _FRAME_RE.match(
                        f[:-len(ext) - 1])}

        gopro = _frames("gopro", "jpg")
        labels = _frames("labels", "json")
        aris = _frames("aris_raw", "pgm")
        aris_meta = _read_csv_rows(os.path.join(rd, "aris_frame_meta.csv"))
        gantry = _read_csv_rows(os.path.join(rd, "gantry.csv"))
        n_beams, n_samples = _sensor_constants(rd)

        for idx in sorted(gopro & aris):  # a pair needs both modalities
            has_label = idx in labels
            if not has_label and not include_unlabeled:
                stats["unlabeled_skipped"] += 1
                continue
            if int(idx) % stride != 0:
                if has_label:  # keep the annotated-frame loss visible
                    stats["stride_skipped"] += 1
                continue
            if int(idx) not in gantry:  # gantry keys are int frame indices
                stats["no_gantry"] += 1
                continue

            sample_id = f"{rec}_{idx}"
            cam_rel = f"images/camera/{sample_id}.jpg"
            son_rel = f"images/sonar/{sample_id}.pgm"
            _link_or_copy(os.path.join(rd, "gopro", idx + ".jpg"),
                          os.path.join(out_root, cam_rel))
            _link_or_copy(os.path.join(rd, "aris_raw", idx + ".pgm"),
                          os.path.join(out_root, son_rel))
            camera["images"].append(
                {"id": next_id, "file_name": cam_rel,
                 "width": 1920, "height": 1080})
            # rows = range samples -> sonar height; cols = beams -> width
            sonar["images"].append(
                {"id": next_id, "file_name": son_rel,
                 "width": n_beams, "height": n_samples})

            row: Dict[str, Any] = {
                "sample_id": sample_id,
                "camera_image_id": next_id,
                "sonar_image_id": next_id,
                "split": stats["split"],
                "recording": rec,
                "frame_idx": int(idx),
                "source_path_camera": f"{rec}/gopro/{idx}.jpg",
                "source_path_sonar": f"{rec}/aris_raw/{idx}.pgm",
                "gantry_x": gantry[int(idx)].get("x", ""),
                "gantry_y": gantry[int(idx)].get("y", ""),
                "gantry_z": gantry[int(idx)].get("z", ""),
            }
            for col in _ARIS_META_COLUMNS:
                row["aris_" + col.lower()] = aris_meta.get(
                    int(idx), {}).get(col, "")
            frame_meta.append(row)

            if has_label:
                with open(os.path.join(rd, "labels", idx + ".json"),
                          encoding="utf-8") as f:
                    lab = json.load(f)
                x1 = lab["x_min"] * _LABEL_SCALE
                y1 = lab["y_min"] * _LABEL_SCALE
                x2 = lab["x_max"] * _LABEL_SCALE
                y2 = lab["y_max"] * _LABEL_SCALE
                camera["annotations"].append({
                    "id": next_ann_id, "image_id": next_id,
                    "category_id": cat_id[lab["class"]],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
                    "segmentation": []})
                next_ann_id += 1
                stats["classes"][lab["class"]] = \
                    stats["classes"].get(lab["class"], 0) + 1
            next_id += 1
            stats["kept"] += 1

        stats["label_without_gopro"] = len(labels - gopro)
        stats["annotated_without_aris"] = len((gopro & labels) - aris)
        report["recordings"][rec] = stats

    write_paired(out_root, PairedData(camera=camera, sonar=sonar,
                                      pairs=pairs, frame_meta=frame_meta,
                                      root=out_root))
    # validate what we just wrote; catches id/reference mistakes early
    read_paired(out_root)
    report["total_kept"] = len(frame_meta)
    report["total_annotations"] = len(camera["annotations"])
    if contact_sheet:
        _write_contact_sheet(out_root, contact_sheet)
    return report


def _write_contact_sheet(out_root: str, dst: str, n: int = 16) -> None:
    """Grid of annotated camera frames with x3 boxes for eyeball checks.

    Deterministic: evenly-spaced samples over the sorted annotated order.
    """
    import cv2
    import numpy as np

    with open(os.path.join(out_root, "camera.json"), encoding="utf-8") as f:
        coco = json.load(f)
    ann_by_img = {a["image_id"]: a for a in coco["annotations"]}
    imgs = [im for im in coco["images"] if im["id"] in ann_by_img]
    if not imgs:
        return
    step = max(1, len(imgs) // n)
    tiles = []
    for im in imgs[::step][:n]:
        img = cv2.imread(os.path.join(out_root, im["file_name"]))
        if img is None:
            continue
        x, y, w, h = ann_by_img[im["id"]]["bbox"]
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)),
                      (0, 0, 255), 3)
        tiles.append(cv2.resize(img, (480, 270)))
    if not tiles:
        return
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    tile_h, tile_w = tiles[0].shape[:2]
    grid = np.zeros((rows * tile_h, cols * tile_w, 3), np.uint8)
    for i, t in enumerate(tiles):
        grid[(i // cols) * tile_h:(i // cols + 1) * tile_h,
             (i % cols) * tile_w:(i % cols + 1) * tile_w] = t
    cv2.imwrite(dst, grid, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True,
                    help="dir containing the recording dirs")
    ap.add_argument("--out", required=True, help="paired-format output dir")
    ap.add_argument("--test-recs", default=",".join(DEFAULT_TEST_RECORDINGS),
                    help="comma-separated recordings assigned to test")
    ap.add_argument("--include-unlabeled", action="store_true",
                    help="also emit gopro frames without labels (as "
                         "negatives; only after manual review)")
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth frame by frame index (default 1)")
    ap.add_argument("--contact-sheet", default=None,
                    help="write a jpg grid of sampled annotated frames "
                         "to this path")
    args = ap.parse_args()
    test_recs = tuple(r for r in
                      (s.strip() for s in args.test_recs.split(",")) if r)
    report = convert(args.src, args.out, test_recordings=test_recs,
                     include_unlabeled=args.include_unlabeled,
                     stride=args.stride,
                     contact_sheet=args.contact_sheet)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
