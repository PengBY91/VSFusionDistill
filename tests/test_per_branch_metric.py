"""Per-branch metric tests (setup doc §4.5, row 6): our wrapper must agree
with a direct pycocotools COCOeval call on the same toy data, and the file
collection must map filenames to (branch, level) correctly."""

import json

import pytest

pycocotools = pytest.importorskip("pycocotools")
from pycocotools.coco import COCO                      # noqa: E402
from pycocotools.cocoeval import COCOeval              # noqa: E402

from aodet.evaluation.per_branch_metric import (       # noqa: E402
    BRANCHES,
    coco_ap,
    collect_ap_table,
)

GT = {
    "images": [
        {"id": 1, "width": 640, "height": 480},
        {"id": 2, "width": 640, "height": 480},
    ],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 80, 60],
         "area": 4800, "iscrowd": 0},
        {"id": 2, "image_id": 2, "category_id": 1, "bbox": [300, 200, 80, 60],
         "area": 4800, "iscrowd": 0},
    ],
    "categories": [{"id": 1, "name": "uxo"}],
}
PREDS = [
    {"image_id": 1, "category_id": 1, "bbox": [102, 98, 80, 60], "score": 0.9},
    {"image_id": 2, "category_id": 1, "bbox": [299, 205, 76, 56], "score": 0.7},
    {"image_id": 2, "category_id": 1, "bbox": [10, 10, 40, 40], "score": 0.3},
]


def _direct_cocoeval_stats(gt, preds):
    coco = COCO()
    coco.dataset = gt
    coco.createIndex()
    dt = coco.loadRes(list(preds))
    ev = COCOeval(coco, dt, iouType="bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev.stats


def test_wrapper_matches_direct_cocoeval():
    ours = coco_ap(GT, PREDS)
    ref = _direct_cocoeval_stats(GT, PREDS)
    assert ours["AP"] == pytest.approx(ref[0])
    assert ours["AP50"] == pytest.approx(ref[1])
    assert ours["AR@100"] == pytest.approx(ref[8])


def test_perfect_predictions_score_full_ap():
    perfect = [
        {"image_id": a["image_id"], "category_id": a["category_id"],
         "bbox": a["bbox"], "score": 0.95}
        for a in GT["annotations"]
    ]
    m = coco_ap(GT, perfect)
    assert m["AP"] == pytest.approx(1.0)
    assert m["AP50"] == pytest.approx(1.0)


def test_img_ids_subset_restriction():
    """Restricting to image 1 must change the result vs both images."""
    both = coco_ap(GT, PREDS)
    only1 = coco_ap(GT, PREDS, img_ids=[1])
    assert only1["AP"] != both["AP"] or only1["AP50"] != both["AP50"]


def test_collect_ap_table(tmp_path, capsys):
    for branch, level in [("camera", "d0"), ("camera", "d4"),
                          ("sonar", "d0"), ("sonar_uncoupled", "d2")]:
        with open(tmp_path / f"pred.{branch}.{level}.json", "w") as f:
            json.dump(PREDS, f)
    with open(tmp_path / "notes.txt", "w") as f:
        f.write("ignored")
    gt_file = tmp_path / "gt.json"
    gt_file.write_text(json.dumps(GT))
    rows = collect_ap_table(str(tmp_path), str(gt_file))
    assert len(rows) == 4
    keys = {(r["branch"], r["level"]) for r in rows}
    assert keys == {("camera", "d0"), ("camera", "d4"), ("sonar", "d0"),
                    ("sonar_uncoupled", "d2")}
    assert all(r["AP"] >= 0.0 for r in rows)


def test_empty_predictions_score_zero():
    m = coco_ap(GT, [])
    assert m["AP"] == 0.0
    assert m["AP50"] == 0.0


def test_branches_tuple_matches_plan():
    assert BRANCHES == ("camera", "sonar", "sonar_uncoupled")


def test_filter_gt_sources():
    """E05: blob-only GT variant keeps measured-echo annotations and
    drops projected ones; annotations without a source tag pass."""
    from aodet.evaluation.per_branch_metric import filter_gt_sources
    gt = dict(images=GT["images"], categories=GT["categories"],
              annotations=[
                  dict(id=1, image_id=1, category_id=1, bbox=[1, 1, 2, 2],
                       source="blob"),
                  dict(id=2, image_id=1, category_id=1, bbox=[2, 2, 2, 2],
                       source="projected"),
                  dict(id=3, image_id=2, category_id=1, bbox=[1, 1, 2, 2]),
              ])
    out = filter_gt_sources(gt, ("blob",))
    assert [a["id"] for a in out["annotations"]] == [1, 3]
    assert filter_gt_sources(gt, ("blob", "projected"))["annotations"] \
        == gt["annotations"]
