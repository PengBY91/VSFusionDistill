"""The unified paired sonar–camera dataset format (setup doc §4.4(1)).

Every source dataset (RSFusion, UXO, UMOD, H1) is converted exactly once
into this layout; all training/evaluation code reads only this layout::

    <paired_root>/
      camera.json      COCO-format annotations of the camera frames
      sonar.json       COCO-format annotations of the sonar frames
      pairs.csv        camera_ann_id,sonar_ann_id,sample_id
                       (cross-modal correspondence π, one row per object
                       pair; unpaired objects are absent)
      frame_meta.csv   one row per frame pair:
                       sample_id,camera_image_id,sonar_image_id,split
                       + free-form metadata columns (light, range_m, gain,
                       ntu, deg_level, source_path_*, ...)
      images/camera/   camera frames (file_name in camera.json is relative
      images/sonar/    to <paired_root>; sonar likewise)

Design invariants enforced by :func:`validate_paired`:

* ``sample_id`` is the join key; frame_meta must cover every sample exactly
  once, and every annotation/image id referenced by pairs.csv must exist in
  the respective COCO file.
* Camera and sonar keep *separate* id spaces (same numeric id may mean
  different things on the two sides — never join on the raw id).
* Categories are per-modality (the two image domains have different class
  granularities in some sources); correspondence happens at the object level
  through pairs.csv, not through category ids.

The module deliberately depends only on the stdlib + numpy so converters and
tests run without the mm* stack.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

__all__ = [
    "PAIRS_COLUMNS",
    "FRAME_META_REQUIRED_COLUMNS",
    "PairedData",
    "write_paired",
    "read_paired",
    "validate_paired",
    "PairedFormatError",
]

PAIRS_COLUMNS = ("camera_ann_id", "sonar_ann_id", "sample_id")
FRAME_META_REQUIRED_COLUMNS = ("sample_id", "camera_image_id", "sonar_image_id",
                               "split")


class PairedFormatError(ValueError):
    """Raised when a paired dataset violates the format invariants."""


@dataclass
class PairedData:
    camera: Dict[str, Any]            # COCO dict
    sonar: Dict[str, Any]             # COCO dict
    pairs: List[Dict[str, Any]]       # rows of pairs.csv
    frame_meta: List[Dict[str, Any]]  # rows of frame_meta.csv
    root: Optional[str] = None

    # -- convenient views -------------------------------------------------
    def samples(self) -> List[Dict[str, Any]]:
        """Merge frame_meta + per-modality annotations by sample_id."""
        cam_by_img = _group_anns(self.camera)
        son_by_img = _group_anns(self.sonar)
        pair_by_sample: Dict[str, List[Dict[str, Any]]] = {}
        for p in self.pairs:
            pair_by_sample.setdefault(str(p["sample_id"]), []).append(p)
        out = []
        for row in self.frame_meta:
            sid = str(row["sample_id"])
            cam_anns = cam_by_img.get(int(row["camera_image_id"]), [])
            son_anns = son_by_img.get(int(row["sonar_image_id"]), [])
            out.append({
                "sample_id": sid,
                "split": row.get("split", ""),
                "camera_image_id": int(row["camera_image_id"]),
                "sonar_image_id": int(row["sonar_image_id"]),
                "camera_file": _file_name(self.camera,
                                          int(row["camera_image_id"])),
                "sonar_file": _file_name(self.sonar,
                                         int(row["sonar_image_id"])),
                "camera_annotations": cam_anns,
                "sonar_annotations": son_anns,
                "pairs": pair_by_sample.get(sid, []),
                "meta": row,
            })
        return out

    def path(self, rel: str) -> str:
        if self.root is None:
            raise PairedFormatError("PairedData.root not set")
        return os.path.join(self.root, rel)


def _group_anns(coco: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    by_img: Dict[int, List[Dict[str, Any]]] = {}
    for a in coco.get("annotations", []):
        by_img.setdefault(int(a["image_id"]), []).append(a)
    return by_img


def _file_name(coco: Dict[str, Any], image_id: int) -> Optional[str]:
    for im in coco.get("images", []):
        if int(im["id"]) == image_id:
            return im.get("file_name")
    return None


def write_paired(out_dir: str, data: PairedData) -> None:
    """Write the four files (images are expected to be already in place)."""
    os.makedirs(out_dir, exist_ok=True)
    for name, payload in (("camera.json", data.camera),
                          ("sonar.json", data.sonar)):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            json.dump(payload, f)
    with open(os.path.join(out_dir, "pairs.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(PAIRS_COLUMNS))
        w.writeheader()
        for row in data.pairs:
            w.writerow({k: row[k] for k in PAIRS_COLUMNS})
    if data.frame_meta:
        cols = sorted(
            {k for row in data.frame_meta for k in row}
            | set(FRAME_META_REQUIRED_COLUMNS)
        )
        with open(os.path.join(out_dir, "frame_meta.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for row in data.frame_meta:
                w.writerow(row)


def read_paired(root: str, validate: bool = True) -> PairedData:
    with open(os.path.join(root, "camera.json"), encoding="utf-8") as f:
        camera = json.load(f)
    with open(os.path.join(root, "sonar.json"), encoding="utf-8") as f:
        sonar = json.load(f)
    with open(os.path.join(root, "pairs.csv"), newline="",
              encoding="utf-8") as f:
        pairs = [dict(r) for r in csv.DictReader(f)]
    frame_meta: List[Dict[str, Any]] = []
    meta_path = os.path.join(root, "frame_meta.csv")
    if os.path.exists(meta_path):
        with open(meta_path, newline="", encoding="utf-8") as f:
            frame_meta = [dict(r) for r in csv.DictReader(f)]

    # CSV round-trips everything as str: normalise id columns so consumers
    # never see "5" where the json side has 5
    for p in pairs:
        p["camera_ann_id"] = int(p["camera_ann_id"])
        p["sonar_ann_id"] = int(p["sonar_ann_id"])
        p["sample_id"] = str(p["sample_id"])
    for m in frame_meta:
        m["sample_id"] = str(m["sample_id"])
        m["camera_image_id"] = int(m["camera_image_id"])
        m["sonar_image_id"] = int(m["sonar_image_id"])

    data = PairedData(camera=camera, sonar=sonar, pairs=pairs,
                      frame_meta=frame_meta, root=root)
    if validate:
        validate_paired(data)
    return data


def validate_paired(data: PairedData) -> None:
    """Check the format invariants; raise PairedFormatError on violation."""
    errors: List[str] = []

    def _coco_sanity(name: str, coco: Dict[str, Any]) -> Dict[int, int]:
        ids = [int(im["id"]) for im in coco.get("images", [])]
        if len(ids) != len(set(ids)):
            errors.append(f"{name}.json: duplicate image ids")
        ann_ids = [int(a["id"]) for a in coco.get("annotations", [])]
        if len(ann_ids) != len(set(ann_ids)):
            errors.append(f"{name}.json: duplicate annotation ids")
        if not coco.get("categories"):
            errors.append(f"{name}.json: no categories")
        for a in coco.get("annotations", []):
            if int(a["image_id"]) not in set(ids):
                errors.append(
                    f"{name}.json: ann {a['id']} references unknown image "
                    f"{a['image_id']}")
        return {int(i) for i in ids}

    cam_ids = _coco_sanity("camera", data.camera)
    son_ids = _coco_sanity("sonar", data.sonar)
    cam_ann_ids = {int(a["id"]) for a in data.camera.get("annotations", [])}
    son_ann_ids = {int(a["id"]) for a in data.sonar.get("annotations", [])}

    for p in data.pairs:
        missing = [k for k in PAIRS_COLUMNS if k not in p]
        if missing:
            errors.append(f"pairs.csv row missing {missing}: {p}")
            continue
        if int(p["camera_ann_id"]) not in cam_ann_ids:
            errors.append(f"pairs.csv: unknown camera_ann_id {p['camera_ann_id']}")
        if int(p["sonar_ann_id"]) not in son_ann_ids:
            errors.append(f"pairs.csv: unknown sonar_ann_id {p['sonar_ann_id']}")

    if data.frame_meta:
        cols = set(data.frame_meta[0].keys())
        for req in FRAME_META_REQUIRED_COLUMNS:
            if req not in cols:
                errors.append(f"frame_meta.csv missing column {req!r}")
        sids = [str(r["sample_id"]) for r in data.frame_meta]
        if len(sids) != len(set(sids)):
            errors.append("frame_meta.csv: duplicate sample_id rows")
        pair_sids = {str(p["sample_id"]) for p in data.pairs}
        unknown = pair_sids - set(sids)
        if unknown:
            errors.append(
                f"pairs.csv references sample_ids absent from frame_meta: "
                f"{sorted(unknown)[:5]}...")
        for r in data.frame_meta:
            if int(r["camera_image_id"]) not in cam_ids:
                errors.append(
                    f"frame_meta: unknown camera_image_id {r['camera_image_id']}")
            if int(r["sonar_image_id"]) not in son_ids:
                errors.append(
                    f"frame_meta: unknown sonar_image_id {r['sonar_image_id']}")
    elif data.pairs:
        errors.append("pairs.csv present but frame_meta.csv is missing")

    if errors:
        raise PairedFormatError(
            "paired dataset validation failed:\n  " + "\n  ".join(errors)
        )
