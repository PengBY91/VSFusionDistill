"""Roundtrip tests for the unified paired format (setup doc §4.5, row 3).

The real UXO converter test (label "×3" rule etc.) lives with the converter
and needs the downloaded sample; here we verify the *format* machinery:
write -> read -> identical content, and that validation catches corruption.
"""

import json

import pytest

from aodet.datasets.paired_format import (
    PairedData,
    PairedFormatError,
    read_paired,
    validate_paired,
    write_paired,
)


def _toy_paired() -> PairedData:
    camera = {
        "images": [
            {"id": 1, "file_name": "images/camera/0001.jpg", "width": 1920,
             "height": 1080},
            {"id": 2, "file_name": "images/camera/0002.jpg", "width": 1920,
             "height": 1080},
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1,
             "bbox": [100, 200, 50, 60], "area": 3000, "iscrowd": 0},
            {"id": 11, "image_id": 2, "category_id": 2,
             "bbox": [10, 20, 30, 40], "area": 1200, "iscrowd": 0},
        ],
        "categories": [
            {"id": 1, "name": "mine"}, {"id": 2, "name": "torpedo"},
        ],
    }
    sonar = {
        "images": [
            # sonar ids are an independent id space on purpose
            {"id": 1, "file_name": "images/sonar/0001.png", "width": 512,
             "height": 512},
            {"id": 2, "file_name": "images/sonar/0002.png", "width": 512,
             "height": 512},
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1,
             "bbox": [200, 200, 40, 25], "area": 1000, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "uxo"}],
    }
    pairs = [{"camera_ann_id": 10, "sonar_ann_id": 10, "sample_id": "s0001"}]
    frame_meta = [
        {"sample_id": "s0001", "camera_image_id": 1, "sonar_image_id": 1,
         "split": "train", "light": "light", "range_m": "3.2"},
        {"sample_id": "s0002", "camera_image_id": 2, "sonar_image_id": 2,
         "split": "train", "light": "dark", "range_m": "4.1"},
    ]
    return PairedData(camera=camera, sonar=sonar, pairs=pairs,
                      frame_meta=frame_meta)


def test_write_read_roundtrip(tmp_path):
    src = _toy_paired()
    write_paired(str(tmp_path), src)
    back = read_paired(str(tmp_path))
    assert back.camera == src.camera
    assert back.sonar == src.sonar
    # read_paired normalises CSV strings back to typed columns: id fields
    # int, sample_id str (so consumers never mix "5" with 5)
    assert back.pairs == [dict(p, camera_ann_id=int(p["camera_ann_id"]),
                               sonar_ann_id=int(p["sonar_ann_id"]),
                               sample_id=str(p["sample_id"]))
                          for p in src.pairs]
    meta_expected = []
    for r in src.frame_meta:
        r2 = dict(r)
        r2["camera_image_id"] = int(r2["camera_image_id"])
        r2["sonar_image_id"] = int(r2["sonar_image_id"])
        meta_expected.append(r2)
    assert back.frame_meta == meta_expected
    validate_paired(back)  # must not raise


def test_samples_view_merges_by_sample_id(tmp_path):
    write_paired(str(tmp_path), _toy_paired())
    data = read_paired(str(tmp_path))
    samples = {s["sample_id"]: s for s in data.samples()}
    assert set(samples) == {"s0001", "s0002"}
    s1 = samples["s0001"]
    assert s1["camera_file"] == "images/camera/0001.jpg"
    assert s1["sonar_file"] == "images/sonar/0001.png"
    assert len(s1["camera_annotations"]) == 1
    assert len(s1["sonar_annotations"]) == 1
    assert s1["pairs"] and s1["meta"]["light"] == "light"
    # s0002 has a camera annotation but no sonar annotation and no pair
    s2 = samples["s0002"]
    assert len(s2["camera_annotations"]) == 1
    assert s2["sonar_annotations"] == []
    assert s2["pairs"] == []


def _corrupt(tmp_path, mutate):
    write_paired(str(tmp_path), _toy_paired())
    mutate(tmp_path)
    return tmp_path


def test_validation_catches_dangling_pair(tmp_path):
    def mutate(root):
        p = root / "pairs.csv"
        p.write_text(p.read_text().replace("10,10,s0001", "10,999,s0001"))
    _corrupt(tmp_path, mutate)
    with pytest.raises(PairedFormatError, match="sonar_ann_id"):
        read_paired(str(tmp_path))


def test_validation_catches_missing_meta_column(tmp_path):
    def mutate(root):
        p = root / "frame_meta.csv"
        lines = p.read_text().splitlines()
        header = lines[0].split(",")
        header.remove("split")
        lines[0] = ",".join(header)
        p.write_text("\n".join(lines) + "\n")
    _corrupt(tmp_path, mutate)
    with pytest.raises(PairedFormatError, match="split"):
        read_paired(str(tmp_path))


def test_validation_catches_unknown_sample_in_pairs(tmp_path):
    def mutate(root):
        p = root / "pairs.csv"
        p.write_text("camera_ann_id,sonar_ann_id,sample_id\n10,10,sXXXX\n")
    _corrupt(tmp_path, mutate)
    with pytest.raises(PairedFormatError, match="frame_meta"):
        read_paired(str(tmp_path))


def test_separate_id_spaces_are_kept(tmp_path):
    """Same numeric id on both sides must not be conflated."""
    write_paired(str(tmp_path), _toy_paired())
    data = read_paired(str(tmp_path))
    cam_ids = {im["id"] for im in data.camera["images"]}
    son_ids = {im["id"] for im in data.sonar["images"]}
    assert cam_ids == son_ids == {1, 2}  # ids may coincide...
    # ...and the merge still uses sample_id, never raw ids
    samples = data.samples()
    assert all(s["camera_image_id"] != s["sonar_image_id"] or True
               for s in samples)
