#!/usr/bin/env python
"""Offline degraded test sets with hashes (setup doc §4.4(3)).

For each level D1–D4, materialises a full paired-format root whose camera
images are the *pre-degraded* pixels (deterministic per (sample_id, level,
seed) via aodet.datasets.transforms.camera_degradation), sonar images are
hardlinked unchanged, and frame_meta.csv pins ``deg_level``.  D0 is
materialised too (identity copy) so every level has the same shape.
D∞ needs no root — it is the camera-absent endpoint.

A manifest.json with sha256 of every written file pins the exact pixels:
all methods/ablations evaluated on an offline set see identical inputs.
Run inference on such a root with a test pipeline that has *no*
AOStudentCameraView (otherwise degradation would be applied twice).
"""

import argparse
import hashlib
import json
import os
import shutil
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aodet.datasets.paired_format import PairedData, read_paired, write_paired
from aodet.datasets.transforms.camera_degradation import degrade_camera

LEVELS = ("d0", "d1", "d2", "d3", "d4")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_level(paired: PairedData, root: str, level: str,
               out_root: str, split_filter=None) -> dict:
    level_dir = os.path.join(out_root, level)
    os.makedirs(os.path.join(level_dir, "images", "camera"), exist_ok=True)
    os.makedirs(os.path.join(level_dir, "images", "sonar"), exist_ok=True)

    # image id -> sample_id / deg_seed / split straight from frame_meta
    cam2sample = {m["camera_image_id"]: m["sample_id"]
                  for m in paired.frame_meta}
    son2sample = {m["sonar_image_id"]: m["sample_id"]
                  for m in paired.frame_meta}
    seeds = {m["sample_id"]: int(m.get("deg_seed", 0))
             for m in paired.frame_meta}
    keep_samples = None
    if split_filter:
        keep_samples = {m["sample_id"] for m in paired.frame_meta
                        if m["split"] in split_filter}

    n_written, n_linked = 0, 0
    manifest = {"level": level, "files": {}}

    # camera images: degrade (D0 = verbatim copy)
    for img in paired.camera["images"]:
        sid = cam2sample.get(img["id"])
        if sid is None or (keep_samples is not None
                           and sid not in keep_samples):
            continue
        src = os.path.join(root, img["file_name"])
        dst = os.path.join(level_dir, img["file_name"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if level == "d0":
            shutil.copyfile(src, dst)
        else:
            bgr = cv2.imread(src, cv2.IMREAD_COLOR)
            if bgr is None:
                raise IOError(f"failed to read {src}")
            out = degrade_camera(bgr, sample_id=sid, level=level,
                                 seed=seeds[sid])
            cv2.imwrite(dst, out)
        n_written += 1
        manifest["files"][img["file_name"]] = _sha256(dst)

    # sonar images: hardlink (same inode, near-free, read-only usage)
    for img in paired.sonar["images"]:
        sid = son2sample.get(img["id"])
        if sid is None or (keep_samples is not None
                           and sid not in keep_samples):
            continue
        src = os.path.join(root, img["file_name"])
        dst = os.path.join(level_dir, img["file_name"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            os.remove(dst)
        os.link(src, dst)
        n_linked += 1

    # metadata: pin deg_level in frame_meta
    meta = [dict(m, deg_level=level) for m in paired.frame_meta
            if keep_samples is None or m["sample_id"] in keep_samples]
    write_paired(level_dir, PairedData(
        camera=paired.camera, sonar=paired.sonar,
        pairs=paired.pairs, frame_meta=meta))
    for name in ("camera.json", "sonar.json", "pairs.csv", "frame_meta.csv"):
        p = os.path.join(level_dir, name)
        if os.path.exists(p):
            manifest["files"][name] = _sha256(p)
    manifest["n_camera"], manifest["n_sonar"] = n_written, n_linked
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paired_root")
    parser.add_argument("out_root")
    parser.add_argument("--levels", nargs="+", default=list(LEVELS))
    parser.add_argument("--splits", nargs="+", default=["test"],
                        help="only materialise samples of these splits "
                             "(default: test)")
    args = parser.parse_args()

    paired = read_paired(args.paired_root, validate=True)
    os.makedirs(args.out_root, exist_ok=True)
    manifests = {}
    for level in args.levels:
        assert level in LEVELS, f"level {level} not in {LEVELS}"
        print(f"[{level}] materialising ...")
        manifests[level] = make_level(paired, args.paired_root, level,
                                      args.out_root,
                                      split_filter=set(args.splits))
        print(f"[{level}] camera: {manifests[level]['n_camera']}, "
              f"sonar: {manifests[level]['n_sonar']}")
    with open(os.path.join(args.out_root, "manifest.json"), "w") as f:
        json.dump(manifests, f, indent=2, sort_keys=True)
    print(f"wrote {os.path.join(args.out_root, 'manifest.json')}")


if __name__ == "__main__":
    main()
