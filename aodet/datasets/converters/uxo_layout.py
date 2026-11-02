"""Source-layout discovery for the Dahn UXO set (Zenodo 13778485).

Two source layouts exist (verified against the archive listings
2026-11-02, see doc/f2-rethink-2026-11-01.md E11 prereqs):

* sample.7z (development set, flat): ``<src>/<recording>/`` with
  recording dir names ``YYYY-MM-DD_HHMMSS`` directly under the root.
* data_export_recordings.7z (the main annotated set, class-grouped):
  one extra level, ``<src>/<category_dir>/<recording>/``, e.g.
  ``recordings/15cm_mortar/2023-09-21_100927`` -- 94 recordings under
  5 category dirs (100lbs_bomb x20, 100lbs_bomb_(floor) x9,
  15cm_mortar x20, 20lbs_incendiary x23, test_cylinder x22).

The category dir is upstream's grouping, not the label namespace: the
authoritative per-frame class is the ``class`` field in labels/*.json,
whose names differ from the dir names (15cm_mortar holds
``mortar_shell`` frames).  ``test_cylinder`` recordings carry no
labels at all (a calibration reference object, not detection data).
Stratification therefore stays on the label-derived category; the dir
is carried as an audit field only.
"""

import os
import re

__all__ = ["REC_RE", "discover_recordings", "CATEGORY_DIR_TO_LABEL_CLASS"]

REC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}$")

# verified by locating the five sample recordings inside the full
# archive listing (their label classes are known from sample.7z);
# 100lbs_bomb_(floor) verified on the extracted set 2026-11-02: all
# 4691 of its boxes are 100lbs_aircraft_bomb -- the floor/water-column
# distinction lives only in the dir name.  test_cylinder carries no
# labels at all, so it audits as None (no claim, no mismatch).
CATEGORY_DIR_TO_LABEL_CLASS = {
    "100lbs_bomb": "100lbs_aircraft_bomb",
    "100lbs_bomb_(floor)": "100lbs_aircraft_bomb",
    "15cm_mortar": "mortar_shell",
    "20lbs_incendiary": "incindiary_grenade",
}


def discover_recordings(src_root):
    """Yield ``(recording, recording_path, category_dir)`` triples for
    both layouts: category_dir is None in the flat sample layout, the
    upstream category dir name (e.g. ``15cm_mortar``) in the grouped
    one.  Non-matching roots and class dirs holding no recordings are
    skipped silently; an empty result is the caller's error to raise.
    """
    found = []
    for name in sorted(os.listdir(src_root)):
        path = os.path.join(src_root, name)
        if not os.path.isdir(path):
            continue
        if REC_RE.match(name):
            found.append((name, path, None))
            continue
        for sub in sorted(os.listdir(path)):
            sub_path = os.path.join(path, sub)
            if REC_RE.match(sub) and os.path.isdir(sub_path):
                found.append((sub, sub_path, name))
    return found
