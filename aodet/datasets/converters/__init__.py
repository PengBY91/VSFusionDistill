"""Source-dataset -> paired-format converters.

uxo_to_paired is the primary path (real paired sonar–camera data,
Zenodo 13778485, BSD-3).  The other two are stubs pending their
upstream releases (see the codebase selection doc).
"""

from .rsfusion_to_paired import convert as convert_rsfusion
from .umod_to_paired import convert as convert_umod

__all__ = ["convert_rsfusion", "convert_umod"]
