"""UMOD -> paired format converter (STUB).

UMOD's public data is incomplete (~259 usable pairs per the codebase
selection doc §3.2) — not enough for the S0 scan, so it is a secondary
verification source, not the main corpus.  Convert only if/when the
authors supply the missing recordings or we decide a small extra
verification set is worth it.

Layout notes (from the selection doc): binaural-beam watermarked frames
+ ROI-style labels; the watermark overlay must be documented before any
conversion, since it alters pixel statistics under degradation.
"""

from __future__ import annotations


def convert(umod_root: str, out_root: str) -> None:  # pragma: no cover
    raise NotImplementedError(
        "UMOD public data incomplete (~259 pairs); secondary source only — "
        "see codebase selection doc §3.2")
