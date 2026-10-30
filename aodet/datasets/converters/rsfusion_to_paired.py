"""RSFusionDet -> paired format converter (STUB, route a).

The official repo (github.com/RSFusionDet) was still empty on
2026-10-29/30 (doc/logs/rsfusiondet-watch.log, tools/check_rsfusiondet.sh).
This converter stays a stub until the authors release data/code or reply
to the S0 route-a email (F2 TODO 14.1).

When the release lands:
  1. re-verify the archive layout against the paper's description,
  2. map camera/sonar images + boxes into the unified paired format
     (aodet.datasets.paired_format.write_paired),
  3. record the sonar image convention (raw matrix vs fan/polar) and set
     sonar_zero_centred accordingly (§4.4(2) gates sync flips),
  4. write frame_meta deg_level='d0' (clear-water source) and split per
     the official protocol or a documented re-split.
"""

from __future__ import annotations


def convert(rsfusion_root: str, out_root: str) -> None:  # pragma: no cover
    raise NotImplementedError(
        "RSFusionDet public release is still empty (watch log "
        "doc/logs/rsfusiondet-watch.log); this converter activates on "
        "route a")
