"""Known-answer tests for the evaluation instruments (E11 prereg item 5,
doc/experiments.md §E11; rethink 2026-11-01 §5 rule 7).

The E04–E10 post-mortem found metric bugs only after the fact, so the
instruments now carry answers known in advance:

* perfect predictions must score AP50 = 1.0 exactly;
* predictions with every class id swapped (boxes all right, labels all
  wrong) must score class-aware AP = 0.0 while class-agnostic AP is
  unchanged vs the correct-class case;
* the per-recording / per-class breakdown (ap_by_group_and_category) and
  the class-agnostic view (collapse_categories) inherit both properties.

Toy world: two recordings (recA = split "val", recB = split "val_id"),
two categories, one annotation per frame — the E11 split scheme, small
enough that every number is derivable by hand.
"""

import pytest

pycocotools = pytest.importorskip("pycocotools")

from aodet.evaluation.per_branch_metric import (       # noqa: E402
    ap_by_group_and_category,
    coco_ap,
    collapse_categories,
)

_IMAGES = [{"id": i, "width": 640, "height": 480} for i in range(1, 7)]
_CATEGORIES = [{"id": 1, "name": "mortar"}, {"id": 2, "name": "grenade"}]

# (image_id, category_id, bbox): images 1-3 = recA/val, 4-6 = recB/val_id
GT = {
    "images": _IMAGES,
    "annotations": [
        {"id": i, "image_id": img, "category_id": cat, "bbox": bbox,
         "area": bbox[2] * bbox[3], "iscrowd": 0}
        for i, (img, cat, bbox) in enumerate([
            (1, 1, [100, 100, 80, 60]),
            (2, 2, [200, 120, 80, 60]),
            (3, 1, [300, 140, 80, 60]),
            (4, 2, [120, 300, 80, 60]),
            (5, 1, [220, 320, 80, 60]),
            (6, 2, [320, 340, 80, 60]),
        ], start=1)
    ],
    "categories": _CATEGORIES,
}
GROUP = {1: "recA", 2: "recA", 3: "recA", 4: "recB", 5: "recB", 6: "recB"}
SPLIT = {1: "val", 2: "val", 3: "val", 4: "val_id", 5: "val_id", 6: "val_id"}
ALL_IDS = [1, 2, 3, 4, 5, 6]

PERFECT = [
    {"image_id": a["image_id"], "category_id": a["category_id"],
     "bbox": a["bbox"], "score": 0.9}
    for a in GT["annotations"]
]
SWAPPED = [dict(d, category_id=3 - d["category_id"]) for d in PERFECT]


def _rows(preds, img_ids=ALL_IDS, group_of_img=GROUP):
    return ap_by_group_and_category(GT, preds, img_ids, group_of_img)


def test_perfect_predictions_score_exactly_one():
    m = coco_ap(GT, PERFECT)
    assert m["AP50"] == pytest.approx(1.0)
    assert m["AP"] == pytest.approx(1.0)      # IoU = 1 >= 0.95


def test_swapped_classes_score_zero_class_aware():
    m = coco_ap(GT, SWAPPED)
    assert m["AP50"] == 0.0
    assert m["AP"] == 0.0


def test_swapped_classes_leave_class_agnostic_unchanged():
    correct = coco_ap(*collapse_categories(GT, PERFECT))["AP50"]
    swapped = coco_ap(*collapse_categories(GT, SWAPPED))["AP50"]
    assert correct == pytest.approx(1.0)
    assert swapped == correct            # the known answer: not just != 0


def test_collapse_categories_keeps_boxes_and_ids():
    gt2, preds2 = collapse_categories(GT, PERFECT)
    assert [c["id"] for c in gt2["categories"]] == [1]
    assert {a["category_id"] for a in gt2["annotations"]} == {1}
    assert {d["category_id"] for d in preds2} == {1}
    assert ([a["bbox"] for a in gt2["annotations"]] ==
            [a["bbox"] for a in GT["annotations"]])
    assert [d["bbox"] for d in preds2] == [d["bbox"] for d in PERFECT]
    # inputs are not modified
    assert GT["categories"] == _CATEGORIES
    assert {d["category_id"] for d in PERFECT} == {1, 2}


def test_breakdown_perfect_predictions_all_ones():
    rows = _rows(PERFECT)
    keys = {(r["group"], r["category"]) for r in rows}
    assert keys == {
        ("recA", "mortar"), ("recA", "grenade"), ("recA", "all"),
        ("recB", "mortar"), ("recB", "grenade"), ("recB", "all"),
        ("all", "mortar"), ("all", "grenade"), ("all", "all")}
    for r in rows:
        assert r["AP50"] == pytest.approx(1.0), r
        if r["category"] == "all":
            assert r["AP50_class_agnostic"] == pytest.approx(1.0), r


def test_breakdown_swapped_classes_known_answer():
    rows = _rows(SWAPPED)
    by_key = {(r["group"], r["category"]): r for r in rows}
    # every class-aware number drops to 0 ...
    for r in rows:
        assert r["AP50"] == 0.0, r
    # ... while the class-agnostic view is unchanged vs the perfect case
    for r in _rows(PERFECT):
        if r["category"] == "all":
            assert by_key[(r["group"], "all")]["AP50_class_agnostic"] \
                == pytest.approx(r["AP50_class_agnostic"]) == 1.0


def test_breakdown_split_restriction_selects_recordings():
    val_ids = [i for i in ALL_IDS if SPLIT[i] == "val"]
    rows = _rows(PERFECT, img_ids=val_ids, group_of_img=GROUP)
    groups = {r["group"] for r in rows}
    assert groups == {"recA", "all"}       # recB is val_id, not selected
    by_key = {(r["group"], r["category"]): r for r in rows}
    assert by_key[("all", "all")]["n_images"] == 3
    assert by_key[("all", "all")]["n_gt"] == 3
    # recA alone must equal the val-restricted pooled row
    assert by_key[("recA", "all")]["AP50"] \
        == by_key[("all", "all")]["AP50"]


def test_breakdown_counts_and_pred_filtering():
    rows = {(r["group"], r["category"]): r for r in _rows(PERFECT)}
    assert rows[("recA", "mortar")]["n_gt"] == 2       # images 1 and 3
    assert rows[("recA", "grenade")]["n_gt"] == 1
    assert rows[("recB", "grenade")]["n_pred"] == 2
    # predictions on images outside img_ids never enter the breakdown
    stray = _rows([*PERFECT, {"image_id": 99, "category_id": 1,
                              "bbox": [0, 0, 10, 10], "score": 1.0}])
    pooled = {(r["group"], r["category"]): r for r in stray}
    assert pooled[("all", "all")]["n_pred"] == 6
    assert pooled[("recA", "all")]["n_pred"] == 3
    assert all(r["AP50"] == pytest.approx(1.0) for r in stray)


def test_breakdown_without_groups_pools_only():
    rows = _rows(PERFECT, group_of_img=None)
    assert {r["group"] for r in rows} == {"all"}
    assert len(rows) == 3                  # mortar, grenade, all
