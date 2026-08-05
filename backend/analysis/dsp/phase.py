"""Phase / mono-compatibility measurement.

This module is deliberately cheap: `measure_stereo` has already done every
filter pass and every dot product, so re-deriving correlation or mono-sum loss
here would double the cost of the analysis for identical numbers. What phase
adds is the *interpretation-free reading* of that data which the detector layer
needs but shouldn't have to recompute:

  * a dedicated polarity flag, because "someone flipped a cable / wired a
    speaker backwards" is a single-click fix and deserves not to be buried
    inside a general width complaint;
  * the worst macro band, which is where a low-frequency phase problem shows
    up long before the broadband figure moves;
  * timestamped spans where the image collapses, so the UI timeline is
    clickable.

Thresholds here are physical, not stylistic. -0.5 correlation is not "a bit
wide", it is more energy cancelling than surviving; 0.2 is the point below
which a fold-down is audibly hollow on any material in any genre. Genre
windows stay in targets.py and are the detector's business.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core import AudioBuffer, merge_spans
from ..types import Moment, PhaseMeasurement, StereoMeasurement

__all__ = ["measure_phase"]

# Below this, the channels are fighting rather than differing: a fold-down
# removes more than it keeps. This is the polarity-inversion signature.
POLARITY_CORRELATION = -0.5

# A fold-down is "hollow" here regardless of style. Used only for moments.
PROBLEM_CORRELATION = 0.2

# Mono-compatibility gate. Two fully decorrelated channels already lose
# 3.01 dB when summed — that is arithmetic, not a fault — so the gate sits
# just below it and catches actual cancellation.
MONO_LOSS_ACCEPTABLE_DB = -4.0
MONO_CORRELATION_ACCEPTABLE = 0.0

# Merge flagged frames within this gap so a flickering reading becomes one
# span the user can scrub to, not forty markers.
MOMENT_MERGE_GAP_SEC = 0.5
MOMENT_MIN_LEN_SEC = 0.25


def _finite(value: float, default: float = 0.0) -> float:
    v = float(value)
    return v if math.isfinite(v) else float(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(min(max(value, lo), hi))


def _intensity(worst_corr: float) -> float:
    """0 at the flag threshold, 1 at full anti-phase. Drives marker size."""
    span = PROBLEM_CORRELATION - (-1.0)
    return _clamp((PROBLEM_CORRELATION - worst_corr) / span, 0.0, 1.0)


def _problem_moments(times: np.ndarray, corr: np.ndarray) -> List[Moment]:
    """Spans where the image collapses, each labelled with its worst reading."""
    if times.size == 0 or corr.size == 0:
        return []
    n = min(times.size, corr.size)
    times, corr = times[:n], corr[:n]

    mask = corr < PROBLEM_CORRELATION
    spans: List[Tuple[float, float]] = merge_spans(
        times, mask, min_gap=MOMENT_MERGE_GAP_SEC, min_len=MOMENT_MIN_LEN_SEC
    )

    moments: List[Moment] = []
    for t0, t1 in spans:
        inside = mask & (times >= t0) & (times <= t1)
        worst = float(corr[inside].min()) if np.any(inside) else PROBLEM_CORRELATION
        worst = _clamp(_finite(worst, PROBLEM_CORRELATION), -1.0, 1.0)
        moments.append(
            Moment(
                t_start=round(float(t0), 3),
                t_end=round(float(t1), 3),
                intensity=round(_intensity(worst), 3),
                value=round(worst, 3),
                label=f"correlation {worst:+.2f}",
            )
        )
    return moments


def _worst_band(band_correlation: Dict[str, float]) -> Tuple[Optional[str], float]:
    """The macro band whose channels agree least.

    Bands with no measurable content report 1.0 from the stereo layer, so they
    can never win this and an empty air band never reads as a phase fault.
    """
    if not band_correlation:
        return None, 1.0
    name = min(band_correlation, key=lambda k: band_correlation[k])
    value = _clamp(_finite(band_correlation[name], 1.0), -1.0, 1.0)
    return name, value


def measure_phase(buf: AudioBuffer, stereo: StereoMeasurement) -> PhaseMeasurement:
    """Read phase health off an existing `StereoMeasurement`.

    `stereo` is the `StereoMeasurement` for the same buffer. Nothing it already
    measured is recomputed here.
    """
    # A mono source has no inter-channel relationship to break. Report the
    # honest neutral case rather than inventing a perfect stereo field.
    if buf.is_mono or getattr(stereo, "is_mono_source", False):
        return PhaseMeasurement(
            correlation=1.0,
            polarity_inverted=False,
            mono_compatible=True,
            mono_sum_loss_db=0.0,
            worst_band=None,
            worst_band_correlation=1.0,
            band_mono_loss_db={k: 0.0 for k in (stereo.band_mono_loss_db or {})},
            problem_moments=[],
        )

    correlation = _clamp(_finite(stereo.correlation, 1.0), -1.0, 1.0)
    mono_sum_loss_db = _clamp(_finite(stereo.mono_sum_loss_db, 0.0), -60.0, 60.0)

    polarity_inverted = bool(correlation < POLARITY_CORRELATION)

    mono_compatible = bool(
        correlation >= MONO_CORRELATION_ACCEPTABLE
        and mono_sum_loss_db >= MONO_LOSS_ACCEPTABLE_DB
    )

    worst_band, worst_band_correlation = _worst_band(dict(stereo.band_correlation or {}))

    band_mono_loss_db = {
        name: round(_clamp(_finite(value, 0.0), -60.0, 60.0), 2)
        for name, value in (stereo.band_mono_loss_db or {}).items()
    }

    times = np.asarray(stereo.times or [], dtype=np.float64)
    corr_series = np.asarray(stereo.correlation_series or [], dtype=np.float64)
    problem_moments = _problem_moments(times, corr_series)

    return PhaseMeasurement(
        correlation=round(correlation, 4),
        polarity_inverted=polarity_inverted,
        mono_compatible=mono_compatible,
        mono_sum_loss_db=round(mono_sum_loss_db, 2),
        worst_band=worst_band,
        worst_band_correlation=round(worst_band_correlation, 3),
        band_mono_loss_db=band_mono_loss_db,
        problem_moments=problem_moments,
    )
