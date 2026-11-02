"""Dahn UXO (Zenodo 13778485) -> paired format converter.

Source layout (one dir per recording, frame index shared by all
modalities -- verified on sample.7z 2026-10-30, see
doc/logs/2026-10-30-f2-codebase-w1-w3.md).  Both upstream layouts are
accepted: the flat sample layout ``<src>/<recording>/`` and the
class-grouped main-package layout ``<src>/<category_dir>/<recording>/``
(data_export_recordings.7z, verified 2026-11-02 -- the category dir
lands in the report as ``category_dir`` and prefixes
``source_path_*``; the label ``class`` field stays authoritative)::

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

E11 additions (rethink 2026-11-01, 代码与数据前置项 item 3): a
three-way split via ``split_manifest`` (recording -> train|val|test,
plus a per-recording ``moving_target`` flag carried into frame_meta.csv),
a per-recording frame cap sampled evenly over the trajectory
(``max_frames_per_recording``; long recordings must not dominate), and
a same-distribution holdout carved from each train recording's time
tail (``val_id_tail_frac`` of frames become split=val_id, preceded by
``val_id_gap_frac`` unused frames so the blocks do not touch).  With
the new arguments at their defaults the converter's output is the
legacy two-way behaviour plus the new ``moving_target`` column.
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
from .uxo_layout import discover_recordings

__all__ = ["convert", "DEFAULT_TEST_RECORDINGS"]

# E07 split (2026-10-31): arcs 180722+123500 and line 105856 in train,
# arc 100927 + line 124735 in test -- each test recording novel in one
# dimension only; superseded the E02 default (105856+124735)
DEFAULT_TEST_RECORDINGS = ("2023-09-21_100927", "2023-09-21_124735")
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


def _even_subsample(items: List[str], cap: Optional[int]) -> List[str]:
    """``cap`` entries evenly spaced over ``items`` (sorted frame order =
    trajectory order), both ends included.  Deterministic; the long
    middle of a recording must not dominate short ones (E11 rule:
    按录音封顶的时间均匀采样)."""
    if cap is None or len(items) <= cap:
        return items
    step = (len(items) - 1) / (cap - 1)
    return [items[int(round(i * step))] for i in range(cap)]


def _train_roles(n: int, tail: float, gap: float) -> List[Optional[str]]:
    """Per-frame split labels for a train recording's kept frames: the
    time-tail ``tail`` fraction becomes ``val_id`` (the
    same-distribution holdout that measures the ceiling), preceded by
    ``gap`` fraction of unused frames (None) so the blocks do not touch
    (E11 rule: train 录音时间尾块留出 + 空 5%).  Recordings too short to
    carve (fewer than 8 kept frames, or a carve that would empty the
    train block) stay fully train -- visible in the report, not
    silent."""
    if tail <= 0.0 or n < 8:
        return ["train"] * n
    n_tail = max(1, int(round(n * tail)))
    n_gap = int(round(n * gap))
    if n_tail + n_gap >= n:
        return ["train"] * n
    return (["train"] * (n - n_tail - n_gap) + [None] * n_gap
            + ["val_id"] * n_tail)


def convert(src_root: str, out_root: str,
            test_recordings=DEFAULT_TEST_RECORDINGS,
            include_unlabeled: bool = False,
            stride: int = 1,
            contact_sheet: Optional[str] = None,
            split_manifest: Optional[Dict[str, Any]] = None,
            max_frames_per_recording: Optional[int] = None,
            val_id_tail_frac: float = 0.0,
            val_id_gap_frac: float = 0.05) -> Dict[str, Any]:
    """Convert one or more UXO recordings into the paired format.

    ``split_manifest`` (E11): a json dict mapping each recording to
    {"split": train|val|test, "moving_target": bool, "excluded": bool,
    ...} -- either bare or under a "recordings" key.  It overrides
    ``test_recordings``; ``excluded`` recordings are skipped entirely
    (calibration-quality filter, rethink 划分规则).  ``moving_target``
    lands in frame_meta.csv as a per-frame column.

    Returns a per-recording report dict (frames kept / skipped by
    reason, class counts); print it or assert on it in tests.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if max_frames_per_recording is not None and max_frames_per_recording < 1:
        raise ValueError("max_frames_per_recording must be >= 1")
    if not 0.0 <= val_id_tail_frac < 1.0 or not 0.0 <= val_id_gap_frac < 1.0:
        raise ValueError("val_id fractions must be in [0, 1)")
    man = split_manifest or {}
    man = man.get("recordings", man)
    for rec, m in man.items():
        if m.get("split") not in (None, "train", "val", "test"):
            raise ValueError(f"{rec}: bad manifest split {m.get('split')!r}")
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

    recs = discover_recordings(src_root)
    if not recs:
        raise ValueError(
            f"no recording dirs (YYYY-MM-DD_HHMMSS) under {src_root}")

    next_id = 1
    next_ann_id = 1
    for rec, rd, group in recs:
        prefix = f"{group}/" if group else ""
        stats = {"split": "", "kept": 0, "unlabeled_skipped": 0,
                 "label_without_gopro": 0, "annotated_without_aris": 0,
                 "no_gantry": 0, "stride_skipped": 0, "classes": {},
                 "category_dir": group or ""}
        m = man.get(rec, {})
        if m.get("excluded"):
            stats.update(split="excluded",
                         exclude_reason=m.get("exclude_reason", ""),
                         moving_target="true" if m.get("moving_target")
                         else "false")
            report["recordings"][rec] = stats
            continue
        if m.get("split"):
            stats["split"] = m["split"]
            stats["split_source"] = "manifest"
        else:
            stats["split"] = "test" if rec in test_set else "train"
            stats["split_source"] = "default"
        moving_flag = ("true" if m.get("moving_target")
                       else "false" if rec in man else "")
        stats["moving_target"] = moving_flag

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

        cand: List[str] = []
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
            cand.append(idx)

        kept = _even_subsample(cand, max_frames_per_recording)
        stats["capped_skipped"] = len(cand) - len(kept)
        roles = (_train_roles(len(kept), val_id_tail_frac, val_id_gap_frac)
                 if stats["split"] == "train" else [stats["split"]] * len(kept))
        stats["val_id_frames"] = sum(1 for r in roles if r == "val_id")
        stats["gap_unused"] = sum(1 for r in roles if r is None)

        for idx, role in zip(kept, roles):
            if role is None:  # the unused gap between train and val_id
                continue
            has_label = idx in labels
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
                "split": role,
                "recording": rec,
                "frame_idx": int(idx),
                "moving_target": moving_flag,
                "source_path_camera": f"{prefix}{rec}/gopro/{idx}.jpg",
                "source_path_sonar": f"{prefix}{rec}/aris_raw/{idx}.pgm",
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
    ap.add_argument("--splits-json", default=None,
                    help="recording split manifest json (each recording: "
                         "split train|val|test, moving_target bool, "
                         "excluded bool); overrides --test-recs (E11)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="per-recording cap on kept frames, sampled "
                         "evenly over the trajectory (E11 rule: 150)")
    ap.add_argument("--val-id-tail", type=float, default=0.0,
                    help="fraction of each train recording's kept frames "
                         "reserved as the same-distribution val_id tail "
                         "(E11 rule: 0.15)")
    ap.add_argument("--val-id-gap", type=float, default=0.05,
                    help="unused gap fraction between the train and "
                         "val_id blocks (E11 rule: 0.05)")
    ap.add_argument("--contact-sheet", default=None,
                    help="write a jpg grid of sampled annotated frames "
                         "to this path")
    args = ap.parse_args()
    test_recs = tuple(r for r in
                      (s.strip() for s in args.test_recs.split(",")) if r)
    manifest = None
    if args.splits_json:
        with open(args.splits_json, encoding="utf-8") as f:
            manifest = json.load(f)
    report = convert(args.src, args.out, test_recordings=test_recs,
                     include_unlabeled=args.include_unlabeled,
                     stride=args.stride,
                     contact_sheet=args.contact_sheet,
                     split_manifest=manifest,
                     max_frames_per_recording=args.max_frames,
                     val_id_tail_frac=args.val_id_tail,
                     val_id_gap_frac=args.val_id_gap)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
