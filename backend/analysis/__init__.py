"""MixDoctor deep analysis engine.

Layering, outermost first:

    engine.analyze_mix()      orchestrates everything, returns MixAnalysis
      detectors.*             turn measurements into evidence-backed Findings
        dsp.*                 pure DSP -> measured numbers, no judgement
          core / targets      shared framing, band math, genre reference data

Nothing in `dsp` knows about genres or thresholds; nothing in `detectors`
touches raw audio. Keep it that way — it's what makes the numbers trustworthy.
"""

from . import clarify
from .clarify import ASK_LIMIT as CLARIFY_ASK_LIMIT
from .clarify import apply_answers as apply_clarification_answers
from .clarify import pending as pending_clarifications
from .clarify import select as select_clarifications
from .types import (
    Band,
    Clarification,
    ClarificationAnswer,
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
    "Clarification",
    "ClarificationAnswer",
    "Dimension",
    "DIMENSIONS",
    "Evidence",
    "Finding",
    "MixAnalysis",
    "Moment",
    "Severity",
    # The clarification layer. `select_clarifications` is the one the API and
    # the UI must both go through: it decides which of the questions on a
    # report are actually put to the user, and a second implementation of that
    # rule anywhere else is a second answer to "what am I being asked".
    "clarify",
    "CLARIFY_ASK_LIMIT",
    "select_clarifications",
    "pending_clarifications",
    "apply_clarification_answers",
]
