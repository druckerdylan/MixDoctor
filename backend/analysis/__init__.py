"""MixDoctor deep analysis engine.

Layering, outermost first:

    engine.analyze_mix()      orchestrates everything, returns MixAnalysis
      detectors.*             turn measurements into evidence-backed Findings
        dsp.*                 pure DSP -> measured numbers, no judgement
          core / targets      shared framing, band math, genre reference data

Nothing in `dsp` knows about genres or thresholds; nothing in `detectors`
touches raw audio. Keep it that way — it's what makes the numbers trustworthy.
"""

from .types import (
    Band,
    Dimension,
    DIMENSIONS,
    Evidence,
    Finding,
    MixAnalysis,
    Moment,
    Severity,
)

__all__ = [
    "Band",
    "Dimension",
    "DIMENSIONS",
    "Evidence",
    "Finding",
    "MixAnalysis",
    "Moment",
    "Severity",
]
