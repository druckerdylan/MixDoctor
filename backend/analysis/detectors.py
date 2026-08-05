"""Measurements -> evidence-backed Findings, and the per-dimension roll-up.

This is the layer that has opinions. Everything under it (`dsp/`) measures and
refuses to judge; everything over it (the AI layer) writes prose and is not
allowed to invent a number. What happens here is the join: a measured value is
compared against the window `targets.py` says this genre lives in, and if it
falls outside, a `Finding` is emitted carrying the figure, the window, and a
sentence that stands on its own without any model in the loop.

Five rules hold the whole file together.

1. **No finding without a number.** Every detector calls `targets.range_miss`
   (or an equivalent explicit comparison) against a window before it emits
   anything. There is no path that produces a Finding from a hunch.

2. **Severity is not chosen, it is computed.** Each metric declares a `scale` —
   "one tolerance unit in this metric's own units" — and severity comes from
   `miss / scale` through one shared mapping (`_severity`). A detector cannot
   decide it is important; it can only say how far out of range its number is.

3. **Confidence tracks how the number was obtained.** True peak, LUFS and
   inter-channel correlation are direct: 0.9+. Gain-reduction estimates,
   sidechain detection and anything derived from centre extraction are
   inferences from a mixed-down file with no stems, and are scored accordingly
   (0.4-0.6). Short files lower confidence further, because most of these
   measurements are statistics over time.

4. **Genre decides the window, not the physics.** -9 LUFS with 4 LU of range is
   a correct trap master and a ruined folk record; 0.75 of width is the point of
   an ambient mix and wrong in hip-hop; lo-fi's missing top end is the genre.
   Every stylistic threshold is read from `targets.get_profile()`. The handful
   of thresholds that are *arithmetic* rather than taste — a mono fold-down
   losing 8 dB, a limiter emitting flat tops — are genre-independent on purpose
   and are marked as such where they appear.

5. **Say it once.** Mud and a hot low-mid band are the same energy described
   twice, so the frequency-balance detector stands down where a more specific
   detector has already spoken. The list is ranked worst-first and capped, and
   every dimension keeps at least its own worst finding.

Optional depth follows the same five rules, and rules 3 and 5 are what shape
it. `Measurements.stems` and `Measurements.sections` do not add a parallel set
of findings; they make the existing ones better:

* **Stems replace inferences with measurements, and the confidence says so.**
  Vocal balance goes from a centre estimate (0.55 — a centre estimate cannot
  tell a singer from a centred synth) to the vocal stem's own loudness against
  the actual instrumental (0.90). Kick versus bass goes from a spectrum
  reconstructed by subtraction (0.70) to two separated objects compared
  directly (0.93). The two-track path stays exactly as it was and is used
  whenever separation was not run or did not find the source.
* **A masking pair is claimed exactly once.** `ctx.take_masking` hands each
  measured masker/maskee pair to the one detector that describes the problem
  best — vocal balance if it is burying the vocal, mud if it is in the low
  mids, clarity for whatever is left — so "the guitars are on top of the
  vocal" is one finding, not three.
* **Sections beat whole-file statistics where they overlap.** A chorus that
  fails to lift and a low end that collapses in one part of the arrangement are
  both invisible to any number averaged over the whole file, and the first of
  them is the same observation as a low loudness range with the timestamps
  attached — so the arrangement detector speaks and the loudness-range arm of
  `dynamic_range` stands down.
* **Genre still decides.** Nothing here is a new global threshold. Whether a
  genre expects a chorus to lift is derived from `vocal_expected`, `punch_min`
  and the loudness-range window it already has (`_expects_arrangement_lift`);
  how far its low end may swing between sections comes from
  `low_end_mono_min`; a drum stem's crest floor is the genre's own crest
  window. Ambient and classical are never told their chorus should be louder.

Two derivations are worth calling out because they are what let this file stay
genre-aware without inventing a second copy of `targets.py`:

* **Region targets are read off the genre's own curve.** `mud_to_mid_db` and
  `boxiness_db` have no window in `targets.py`, so instead of hard-coding one,
  the same ratio is computed *from `targets.target_curve(genre)`* and used as
  the target. Because the DSP layer reports those two as power **densities**
  while the curve is in the 1/3-octave summed convention, the conversion
  applies the -10log10(f) density correction (`_curve_density_db`). Sanity
  check: reconstructing `mud_ratio_db` this way for pop gives -8.7 dB, which
  lands inside pop's hand-written (-14.0, -5.5) window — the two agree.

* **Spectral tilt is compared against the tilt of the genre's target curve**,
  fitted over exactly the 1/3-octave bands the DSP layer used (recoverable from
  `third_octave_db`). Lo-fi's steep roll-off is therefore expected rather than
  penalised, with no lo-fi special case anywhere in this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import targets
from .core import THIRD_OCTAVE_CENTERS
from .types import (
    DIMENSION_LABELS,
    DIMENSIONS,
    DimensionScore,
    Evidence,
    Finding,
    MaskingPair,
    Measurements,
    Moment,
    Section,
    SectionAnalysis,
    Severity,
    StemAnalysis,
    StemMeasurement,
    Verdict,
)

__all__ = ["detect_all", "score_dimensions"]


# ---------------------------------------------------------------------------
# Severity, impact and confidence scales
# ---------------------------------------------------------------------------

# `miss / scale` -> severity. One shared mapping for every detector, so
# "how bad is this" is never a per-detector opinion. A ratio of 1.0 means the
# value sits one full tolerance unit outside the genre's window.
_MAJOR_AT = 1.5
_CRITICAL_AT = 3.0

# Health-score points recoverable if a dimension's worst problem is fixed.
# These are ceiling values, reached only by a fully critical finding; the actual
# impact scales with the miss. Ordered by how much a listener notices.
_IMPACT_WEIGHT: Dict[str, float] = {
    "clipping": 12.0,
    "phase": 12.0,
    "loudness": 6.0,
    "limiter": 9.0,
    "dynamic_range": 7.0,
    "compression": 6.0,
    "frequency_balance": 9.0,
    "mud": 9.0,
    "harshness": 8.0,
    "low_end": 9.0,
    "vocal_balance": 8.0,
    "stereo_width": 5.0,
    "transients": 6.0,
    "clarity": 6.0,
}

# A second finding in the same dimension is mostly the same fix, so it recovers
# less; a third, less again.
_REPEAT_DISCOUNT = (1.0, 0.45, 0.25)

# Total recoverable points across the whole report. `ceiling_score` is built
# from these, so they must never be able to sum past 100 — 60 leaves the
# ceiling honest (fixing everything does not make an average mix a 100).
_IMPACT_TOTAL_CAP = 60.0

MAX_FINDINGS = 12

# A miss smaller than a tenth of one tolerance unit is a rounding artefact, not
# a problem: reporting "0.00 narrower than the genre" is worse than silence.
# This is a significance floor on the miss, not a second window.
MIN_REPORTABLE_RATIO = 0.10

# Delivery ceiling. Lossy encoders reconstruct between the samples and overshoot
# by up to ~1 dB; every major platform asks for -1.0 dBTP or lower. This is
# codec arithmetic, not taste, so it does not vary by genre.
TRUE_PEAK_CEILING_DBTP = -1.0

# Two fully decorrelated channels lose 3.01 dB when summed. That is arithmetic.
# Anything past this is cancellation, in any genre.
MONO_LOSS_FLOOR_DB = -3.5

# Below 25 Hz, at this analysis resolution, a strong 40-60 Hz fundamental leaks
# into the measurement window. Under -20 dB the rumble figure cannot be
# separated from that leakage, so nothing is claimed there. This is the *floor*
# of the ceiling — see `_rumble_ceiling_db`, which raises it for the genres
# whose own target curve asks for energy down there.
RUMBLE_CEILING_DB = -20.0

# How far above the level its own target curve implies a genre is allowed to sit
# before the sub-25 Hz energy is called rumble. The measurement is a ratio of two
# wide-band sums, so a couple of dB of it is the curve interpolation and the
# 5.86 Hz analysis grid rather than content.
RUMBLE_TOLERANCE_DB = 3.0


def _rumble_ceiling_db(genre: str) -> float:
    """The sub-25 Hz ceiling for one genre, derived from that genre's own curve.

    A flat -20 dB ceiling contradicts `targets.py`. The trap/hip-hop anchor curve
    carries +2.0 dB at 20 Hz and +3.9 dB at 25 Hz, which works out to a sub-25 Hz
    share of -16.0 dB for a mix sitting *exactly* on the curve — 4 dB past the
    flat ceiling. The consequence was a major "inaudible energy below 25 Hz"
    finding against mixes whose only crime was matching the target the same
    module handed them, systematically, on every bass-forward genre (trap and
    hip-hop at -16.0 dB implied, EDM and D&B at -17.5).

    So the ceiling is the genre's own implied share plus a tolerance, and never
    below the flat floor: this only ever relaxes the test for a genre that asks
    for sub energy, and leaves folk, rock and lo-fi exactly where they were.
    """
    try:
        centers = np.asarray(targets.THIRD_OCTAVE_CENTERS, dtype=np.float64)
        curve = targets.target_curve(genre, targets.THIRD_OCTAVE_CENTERS)
        power = np.power(10.0, np.asarray(curve, dtype=np.float64) / 10.0)
        total = float(power.sum())
        below = float(power[centers < 25.0].sum())
        if total <= 0.0 or below <= 0.0:
            return RUMBLE_CEILING_DB
        implied = 10.0 * math.log10(below / total)
        if not math.isfinite(implied):
            return RUMBLE_CEILING_DB
        return max(RUMBLE_CEILING_DB, implied + RUMBLE_TOLERANCE_DB)
    except Exception:
        return RUMBLE_CEILING_DB

# Minimum duration, in seconds, before a measurement means anything. Most of
# this module is statistics over time; on a 1.2 s file a loudness range or a
# pumping index is measuring the file's edges.
_MIN_SEC: Dict[str, float] = {
    "loudness": 3.0,
    "limiter": 5.0,
    "dynamics": 5.0,
    "arrangement": 30.0,   # loudness range needs an arrangement to range over
    "transients": 5.0,
    "vocal": 8.0,
    "clarity": 3.0,
    "kick": 5.0,
    # A separated source needs far less material to be trustworthy than a
    # centre estimate does: the 8 s under "vocal" buys enough syllabic
    # modulation to decide whether a voice is *there*, and a stem has already
    # answered that. What is left is the ordinary "is this statistic settled"
    # question, which four seconds covers.
    "stem": 4.0,
}

# Under this integrated loudness there is no programme to have an opinion about.
NO_PROGRAMME_LUFS = -50.0

_CENTERS = np.asarray(THIRD_OCTAVE_CENTERS, dtype=np.float64)


# ---------------------------------------------------------------------------
# Stem-derived thresholds
#
# None of these have a home in `targets.py` — it predates separation and is
# shared with concurrent work — so they live here and are derived from figures
# that file *does* carry wherever a genre has an opinion.
# ---------------------------------------------------------------------------

# Two sources this close in pitch beat rather than stack. At 3 semitones a
# 55 Hz 808 and the kick above it differ by 10 Hz, which the ear still resolves
# as two pitches; inside that, the difference tone lands in the same critical
# band as both of them and the low end reads as one thick object. This is the
# comparison a two-track cannot make honestly — `lowend.py` reconstructs the
# kick's spectrum by subtraction — and it is the whole reason separation earns
# its runtime here.
_SAME_NOTE_SEMITONES = 3.0
_SEMITONE_SCALE = 1.5

# A separated source's level over the frames it is actually sounding on. Same
# figure `separation._ACTIVE_BELOW_STEM_DB` gates its own statistics with, so a
# spread computed here describes the same frames the stem's crest does.
_STEM_ACTIVE_BELOW_PEAK_DB = 35.0
_STEM_MIN_SERIES = 8

# Per-element compression needs the gain-reduction estimate to agree before it
# is claimed on anything but the drums: a vocal with little crest may simply be
# a quiet, even performance.
_STEM_GR_CORROBORATION_DB = 3.0

# Source-against-source masking. `clarity.py` fades a band from audible to
# fully masked over 12 dB under the level its neighbourhood spreads onto it;
# the same 12 dB is what "buried" means between two sources, so the ceiling
# sits below it and one tolerance unit is half of it.
_MASK_PAIR_CEILING_DB = 9.0
_MASK_PAIR_SCALE_DB = 6.0
_MASK_PAIR_MIN_OVERLAP_DB = 6.0
# Never critical on its own: this is a model's output about a model's output,
# and a masking pair alone should not be able to pin the health score.
_MASK_PAIR_MAX_RATIO = 2.4
# The bands the mud detector already owns. A masking pair in here is the same
# problem as the low-mid buildup, described from the other side, so mud
# absorbs it instead of the report saying it twice.
_MASK_MUD_BANDS: Tuple[str, ...] = ("upper_bass", "low_mid", "mid")

# `clarity.masking_index` is the loudness-weighted share of the mix's audible
# loudness sitting under its own masking threshold. Shares of loudness are
# small numbers: measured on the calibration fixtures, the three on-target
# reference tracks land at 1.5-1.8% and the deliberately congested mixes at
# 6.5-7.2%, which matches the DSP module's own documented range (~2% for an
# on-target master, 10-15% for a badly arranged one) and its 12% "as masked as
# it gets".
#
# `profile.masking_max` is 0.55 for every genre, written on the 0-1 "share of
# the 1/3-octave grid" scale the coarser model used before the ERB/spreading
# rewrite. Comparing a number that tops out near 0.15 against a 0.55 ceiling is
# a window that can never be missed, so the ceiling is converted here rather
# than in `targets.py`, which is shared with concurrent work. The conversion is
# written to be idempotent: a `masking_max` already on the new scale (anything
# at or under `_MASKING_INDEX_FULL`) is taken as-is, so whoever owns
# `targets.py` can move it without this silently squaring the correction.
_MASKING_INDEX_FULL = 0.12

# What a master with nothing to say about its masking buries. One tolerance
# unit is the distance from there to the genre's ceiling — "how much room is
# there between a clean mix and a congested one" — rather than an arbitrary
# fraction of the ceiling itself.
_MASKING_ON_TARGET = 0.02


# ---------------------------------------------------------------------------
# Section-derived thresholds
# ---------------------------------------------------------------------------

# Below three sections there is no form to have an opinion about, and under
# ~100 s there is no room for three sections long enough to measure.
_MIN_SECTIONS_FOR_FORM = 3
_MIN_FORM_SEC = 100.0

# What "the chorus lifts" means as a number. Most records put 2-4 LU between
# the verse and the hook; under 1.5 LU the difference is measurable and not
# audible, which is the specific complaint this catches.
_CHORUS_LIFT_MIN_LU = 1.5
_CHORUS_LIFT_SCALE = 1.0

# How much the sub+low-bass *share* of a section may move across the sections
# carrying the record before the bottom is falling out of one of them. Scaled
# by genre below: a house record whose drop loses its sub has lost the record.
_LOW_SWING_BASS_LED_DB = 3.0
_LOW_SWING_SONG_DB = 4.5
_LOW_SWING_FREE_DB = 7.0
_LOW_SWING_SCALE_DB = 3.0
_LOW_CORE_LU = 9.0          # matches sections.LOW_CORE_LU
_AUDIBLE_FLOOR_LUFS = -60.0  # matches sections.AUDIBLE_FLOOR_LUFS


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _fin(value, default: float = 0.0) -> float:
    """Coerce to a finite float. NaN/inf serialise to invalid JSON."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(min(max(_fin(value, lo), lo), hi))


def _severity(ratio: float) -> Severity:
    """Shared miss-ratio -> severity mapping. See rule 2 in the module docstring."""
    r = _fin(ratio, 0.0)
    if r >= _CRITICAL_AT:
        return "critical"
    if r >= _MAJOR_AT:
        return "major"
    if r > 0.0:
        return "minor"
    return "clean"


def _ratio(miss: float, scale: float) -> float:
    """|miss| in units of one tolerance step."""
    return abs(_fin(miss, 0.0)) / max(abs(_fin(scale, 1.0)), 1e-9)


def _verdict(value: float, window: Tuple[float, float], scale: float) -> Verdict:
    miss = targets.range_miss(_fin(value), window)
    if miss == 0.0:
        return "good"
    return "problem" if _ratio(miss, scale) >= _MAJOR_AT else "watch"


def _num(value: float, nd: int = 1) -> str:
    """Format a figure for a sentence: no trailing '.0' noise, no -0.0."""
    v = _fin(value, 0.0)
    if abs(v) < 5e-3:
        v = 0.0
    text = f"{v:.{nd}f}"
    return "0" if text in ("-0", "-0.0", "-0.00") else text


def _win(window: Tuple[float, float], nd: int = 1, unit: str = "") -> str:
    return f"{_num(window[0], nd)} to {_num(window[1], nd)}{unit}"


def _pm(value: float) -> str:
    """'+/-2.2' — the tolerance band around a target, for a sentence."""
    return f"+/-{_num(value, 1)}"


def _ev(
    label: str,
    value: float,
    unit: str = "",
    target: Optional[float] = None,
    target_range: Optional[Tuple[float, float]] = None,
    verdict: Verdict = "good",
    detail: str = "",
) -> Evidence:
    return Evidence(
        label=label,
        value=round(_fin(value), 4),
        unit=unit,
        target=None if target is None else round(_fin(target), 4),
        target_range=(
            None
            if target_range is None
            else (round(_fin(target_range[0]), 4), round(_fin(target_range[1]), 4))
        ),
        verdict=verdict,
        detail=detail,
    )


def _moments(source: Optional[Sequence[Moment]], limit: int = 6) -> List[Moment]:
    """Copy the DSP layer's moments onto a finding, worst first."""
    if not source:
        return []
    ranked = sorted(source, key=lambda mo: -_fin(mo.intensity, 0.0))[:limit]
    return sorted(ranked, key=lambda mo: _fin(mo.t_start, 0.0))


def _clock(seconds: float) -> str:
    s = max(_fin(seconds, 0.0), 0.0)
    return f"{int(s // 60)}:{s % 60:04.1f}"


def _moment_span(moments: Sequence[Moment]) -> str:
    """'worst at 0:12.1' / 'worst at 0:12.1 and 3 other spans'."""
    if not moments:
        return ""
    worst = max(moments, key=lambda mo: _fin(mo.intensity, 0.0))
    if len(moments) == 1:
        return f" Worst at {_clock(worst.t_start)}."
    return f" Worst at {_clock(worst.t_start)}, across {len(moments)} flagged spans."


# ---------------------------------------------------------------------------
# Genre-curve derivations
#
# `targets.py` windows the metrics it names. For the two region ratios it does
# not name (`mud_to_mid_db`, `boxiness_db`) the target is derived from the
# genre's own curve rather than invented here, so a new genre added to
# targets.py automatically gets correct windows for both.
# ---------------------------------------------------------------------------


def _curve_region_db(curve: np.ndarray, lo: float, hi: float) -> float:
    """Power-mean of a target curve over a region, 1/3-octave summed convention.

    Identical maths to `targets.macro_target`, but taking the already-computed
    curve so a detector can ask about arbitrary regions.
    """
    mask = (_CENTERS >= lo) & (_CENTERS < hi)
    if not np.any(mask):
        return 0.0
    return float(10.0 * np.log10(np.mean(10.0 ** (curve[mask] / 10.0))))


def _log_center(lo: float, hi: float) -> float:
    return float(math.sqrt(max(lo, 1e-9) * max(hi, 1e-9)))


def _curve_density_db(curve: np.ndarray, region: Tuple[float, float],
                      reference: Tuple[float, float]) -> float:
    """Target for a power-*density* ratio between two regions of the curve.

    The DSP layer reports `mud_ratio_db`, `mud_to_mid_db` and friends as mean
    power **per FFT bin**, so region width cancels. A target curve is written in
    the 1/3-octave **summed** convention, where band power grows with centre
    frequency for flat spectral density. Converting one to the other is a single
    -10log10(f) term:

        density(A)/density(B) = level(A)/level(B) * f_B/f_A

    Validated against the one density ratio `targets.py` does window: for pop
    this reproduces -8.7 dB for `mud_ratio_db`, inside pop's (-14.0, -5.5).
    """
    level = _curve_region_db(curve, *region) - _curve_region_db(curve, *reference)
    width = 10.0 * math.log10(_log_center(*reference) / max(_log_center(*region), 1e-9))
    return _fin(level + width, 0.0)


def _curve_tilt(curve: np.ndarray, third_octave_db: Sequence[float]) -> float:
    """Tilt of the genre curve, fitted over the bands the DSP layer actually used.

    `spectral.py` fits its tilt over 40 Hz-16 kHz, excluding bands more than
    45 dB under the loudest one, so a brick-walled top end does not invent a
    slope. Reconstructing that mask here from `third_octave_db` makes the two
    numbers directly comparable.
    """
    measured = np.asarray(third_octave_db, dtype=np.float64)
    if measured.size != _CENTERS.size or not np.any(np.isfinite(measured)):
        return 0.0
    mask = (
        (_CENTERS >= 40.0)
        & (_CENTERS <= 16_000.0)
        & (measured > float(np.max(measured)) - 45.0)
    )
    if int(np.count_nonzero(mask)) < 5:
        return 0.0
    return _fin(np.polyfit(np.log10(_CENTERS[mask]), curve[mask], 1)[0], 0.0)


@dataclass
class _BandView:
    """One macro band re-scored against the genre asked for now.

    `SpectralMeasurement.bands` already carries a target, but it was computed at
    whatever genre the measurement pass ran under. Recomputing here makes the
    detector correct regardless, and is the only place band verdicts come from.
    """

    name: str
    low_hz: float
    high_hz: float
    center_hz: float
    level_db: float
    target_db: float
    tolerance_db: float
    deviation_db: float
    miss_db: float

    @property
    def ratio(self) -> float:
        return _ratio(self.miss_db, self.tolerance_db)

    @property
    def label(self) -> str:
        return _BAND_LABELS.get(self.name, self.name.replace("_", " "))

    @property
    def span(self) -> str:
        return f"{_num(self.low_hz, 0)}-{_num(self.high_hz, 0)} Hz"


_BAND_LABELS: Dict[str, str] = {
    "sub": "sub",
    "low_bass": "low bass",
    "upper_bass": "upper bass",
    "low_mid": "low mids",
    "mid": "mids",
    "upper_mid": "upper mids",
    "presence": "presence",
    "brilliance": "brilliance",
    "air": "air",
}


# ---------------------------------------------------------------------------
# Detection context
# ---------------------------------------------------------------------------


@dataclass
class _Hit:
    """A finding plus the miss ratio that produced it.

    The ratio drives impact, and impact has to be assigned globally (it is
    capped across the whole report), so it cannot be baked in at the point the
    detector runs.
    """

    finding: Finding
    ratio: float


def _expects_arrangement_lift(profile: targets.GenreProfile) -> bool:
    """Does this genre's form put a measurable lift into its loudest section?

    A chorus that arrives at the same level as the verse is a real complaint in
    pop, hip-hop, rock and everything shaped like them, and meaningless advice
    for ambient, classical, jazz or a lo-fi loop — those records are not built
    out of a hook that has to land. `targets.py` has no field saying which is
    which and cannot be edited from here, so the answer is derived from three
    that are already there and that already encode it:

    * `vocal_expected` — a song with a lead has a form for the lead to sit in.
      Classical, orchestral, cinematic, ambient and lo-fi are all False.
    * `punch_min` — beat-driven production. Ambient's 0.10 and jazz's 0.28 say
      the material is not built on hits arriving on a grid.
    * `loudness_range_lu` upper bound — where a genre is *allowed* 14-24 LU of
      range, the level movement in it is performance dynamics, not arrangement,
      and a "the chorus does not lift" verdict has nothing to attach to.

    Result: pop, hip-hop, trap, EDM, house, techno, D&B, rock, punk, metal,
    R&B, soul, country, indie and alternative expect a lift. Acoustic, folk,
    jazz, classical, orchestral, cinematic, ambient, lo-fi and the unknown-genre
    "other" fall-back do not.
    """
    return bool(
        getattr(profile, "vocal_expected", False)
        and _fin(profile.punch_min, 0.0) >= 0.30
        and _fin(profile.loudness_range_lu[1], 99.0) <= 10.5
    )


def _low_swing_ceiling(profile: targets.GenreProfile) -> float:
    """How far a section's low-end share may drift from its neighbours'.

    Bass-led genres are identified by the one figure in `targets.py` that says
    the bottom octave is load-bearing rather than decorative: hip-hop, trap,
    EDM, house, techno and D&B all raise `low_end_mono_min` to 0.90 or above
    because the sub is the instrument. A drop that loses its sub has lost the
    record there, so they get the tightest window.
    """
    if _fin(profile.low_end_mono_min, 0.80) >= 0.88:
        return _LOW_SWING_BASS_LED_DB
    return _LOW_SWING_SONG_DB if _expects_arrangement_lift(profile) else _LOW_SWING_FREE_DB


def _stem_level_spread(stem: StemMeasurement) -> Optional[float]:
    """90th minus 10th percentile of a stem's level, over the frames it sounds on.

    The two-track equivalent (`VocalMeasurement.consistency_db`) is the spread
    of a centre estimate, which moves when anything centred moves. This is the
    spread of the source itself, so a vocal riding under a centred synth no
    longer reads as an uneven vocal.
    """
    series = np.asarray(
        [_fin(x, -120.0) for x in (stem.level_series or [])], dtype=np.float64
    )
    if series.size < _STEM_MIN_SERIES:
        return None
    active = series[series > float(series.max()) - _STEM_ACTIVE_BELOW_PEAK_DB]
    if active.size < _STEM_MIN_SERIES:
        return None
    return _fin(float(np.percentile(active, 90.0) - np.percentile(active, 10.0)), 0.0)


def _section_low_shares(
    sections: Sequence[Section],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(low share dB, integrated LUFS, core mask) per section.

    "Low share" is sub+low-bass against the section's own total, which is what
    makes this independent of the section simply being quieter — a bass-free
    intro is an arrangement, a bass-free chorus is a fault. Same derivation
    `sections.py` uses for `low_end_swing_db`, recomputed here because the
    per-section figures are what a finding has to name.
    """
    n = len(sections)
    low = np.zeros(n, dtype=np.float64)
    lufs = np.zeros(n, dtype=np.float64)
    for i, section in enumerate(sections):
        levels = section.band_levels_db or {}
        sub = 10.0 ** (_fin(levels.get("sub", -120.0), -120.0) / 10.0)
        low_bass = 10.0 ** (_fin(levels.get("low_bass", -120.0), -120.0) / 10.0)
        total = sum(10.0 ** (_fin(v, -120.0) / 10.0) for v in levels.values())
        low[i] = _clamp(
            10.0 * math.log10(max(sub + low_bass, 1e-12) / max(total, 1e-12)), -40.0, 0.0
        )
        lufs[i] = _fin(section.integrated_lufs, -70.0)

    audible = lufs > _AUDIBLE_FLOOR_LUFS
    if not np.any(audible):
        audible = np.ones(n, dtype=bool)
    core = audible & (lufs >= float(np.max(lufs[audible])) - _LOW_CORE_LU)
    return low, lufs, core


def _masking_window(profile: targets.GenreProfile) -> Tuple[Tuple[float, float], float]:
    """(window, tolerance unit) for `clarity.masking_index` in this genre.

    See `_MASKING_INDEX_FULL`. `profile.masking_max` still decides how much
    this genre absorbs relative to every other one; only the scale moves.
    """
    raw = _fin(profile.masking_max, 0.55)
    ceiling = _clamp(raw * _MASKING_INDEX_FULL if raw > _MASKING_INDEX_FULL else raw,
                     0.005, 1.0)
    return (0.0, ceiling), max(ceiling - _MASKING_ON_TARGET, 0.01)


def _mask_ratio(pair: MaskingPair) -> float:
    """Miss ratio for one masking pair, from its measured overlap."""
    miss = targets.range_miss(_fin(pair.overlap_db, 0.0), (0.0, _MASK_PAIR_CEILING_DB))
    return min(_ratio(miss, _MASK_PAIR_SCALE_DB), _MASK_PAIR_MAX_RATIO)


def _mask_sentence(
    pairs: Sequence[MaskingPair], lead: str = "Separating the sources names who is doing it:"
) -> str:
    """One clause per claimed masking pair, in the DSP layer's own words."""
    if not pairs:
        return ""
    text = f" {lead.strip()} " + pairs[0].detail.strip()
    if len(pairs) > 1:
        text += " " + " ".join(p.detail.strip() for p in pairs[1:])
    return text


def _mask_evidence(pairs: Sequence[MaskingPair]) -> List[Evidence]:
    return [
        _ev(
            f"{p.masker.capitalize()} over {p.maskee} ({p.band.replace('_', ' ')})",
            _fin(p.overlap_db, 0.0), "dB",
            target_range=(0.0, _MASK_PAIR_CEILING_DB),
            verdict=_verdict(_fin(p.overlap_db, 0.0), (0.0, _MASK_PAIR_CEILING_DB),
                             _MASK_PAIR_SCALE_DB),
            detail=f"Measured between separated sources over "
                   f"{_num(p.low_hz, 0)}-{_num(p.high_hz, 0)} Hz, not inferred "
                   f"from the two-track.",
        )
        for p in pairs
    ]


@dataclass
class _Ctx:
    m: Measurements
    genre_key: str
    profile: targets.GenreProfile
    curve: np.ndarray
    bands: Dict[str, _BandView]
    duration: float
    is_mono: bool
    no_programme: bool
    # Optional depth. Both are always present as objects (never None), so a
    # detector reads `.available` rather than null-checking every access.
    stems: StemAnalysis = field(default_factory=StemAnalysis)
    sections: SectionAnalysis = field(default_factory=SectionAnalysis)
    stems_by_kind: Dict[str, StemMeasurement] = field(default_factory=dict)
    masking_pairs: List[MaskingPair] = field(default_factory=list)
    # Masking pairs already folded into a finding. A pair is claimed exactly
    # once, by whichever detector describes the problem best (rule 5).
    masking_used: set = field(default_factory=set)
    # Tags set by detectors that have already run, so later ones can stand down
    # rather than describe the same energy twice.
    tags: set = field(default_factory=set)

    def has(self, key: str) -> bool:
        return self.duration >= _MIN_SEC.get(key, 0.0)

    @property
    def has_stems(self) -> bool:
        """True only when separation ran *and* produced at least one real source."""
        return bool(self.stems.available and self.stems_by_kind)

    def stem(self, kind: str) -> Optional[StemMeasurement]:
        """The named source, or None if it was absent or never separated."""
        return self.stems_by_kind.get(kind)

    def take_masking(
        self,
        bands: Optional[Sequence[str]] = None,
        maskee: Optional[str] = None,
        limit: int = 2,
    ) -> List[MaskingPair]:
        """Claim the strongest unclaimed masking pairs, worst first.

        Claiming is what keeps one problem to one finding: vocal balance takes
        the pairs burying the vocal, mud takes the ones in its own bands, and
        whatever is left over goes to clarity.
        """
        out: List[MaskingPair] = []
        for pair in self.masking_pairs:
            key = (str(pair.masker), str(pair.maskee), str(pair.band))
            if key in self.masking_used:
                continue
            if bands is not None and str(pair.band) not in bands:
                continue
            if maskee is not None and str(pair.maskee) != maskee:
                continue
            self.masking_used.add(key)
            out.append(pair)
            if len(out) >= limit:
                break
        return out

    def trust(self, base: float, key: Optional[str] = None) -> float:
        """Confidence, reduced for files too short for the statistic to settle.

        A 6 s window of a 20 s file is a third of the record; the same window of
        a 6-minute track is a sample. Anything time-averaged loses confidence on
        short material even when it clears the hard minimum.
        """
        need = _MIN_SEC.get(key or "", 0.0)
        factor = 1.0
        if need > 0.0:
            factor = _clamp(self.duration / (need * 4.0), 0.55, 1.0)
        return round(_clamp(base * factor, 0.05, 0.99), 2)


def _build_ctx(m: Measurements, genre: str) -> _Ctx:
    key = targets.normalise_genre(genre)
    profile = targets.get_profile(key)
    curve = np.asarray(targets.target_curve(key, THIRD_OCTAVE_CENTERS), dtype=np.float64)

    bands: Dict[str, _BandView] = {}
    for band in m.spectral.bands:
        target = _fin(targets.macro_target(key, band.name, band.low_hz, band.high_hz))
        tol = max(_fin(targets.band_tolerance(key, band.name), 3.0), 0.1)
        level = _fin(band.level_db)
        bands[band.name] = _BandView(
            name=band.name,
            low_hz=_fin(band.low_hz),
            high_hz=_fin(band.high_hz),
            center_hz=_fin(band.center_hz),
            level_db=level,
            target_db=target,
            tolerance_db=tol,
            deviation_db=level - target,
            miss_db=targets.range_miss(level, (target - tol, target + tol)),
        )

    stems = m.stems or StemAnalysis()
    # Only sources the separator actually found. An instrumental's `vocals`
    # array comes back full of bleed at -50 dB, and `present=False` is how the
    # DSP layer says so — a detector that read it anyway would report a level
    # for a singer who is not on the record.
    stems_by_kind: Dict[str, StemMeasurement] = {}
    masking_pairs: List[MaskingPair] = []
    if stems.available:
        stems_by_kind = {str(s.kind): s for s in (stems.stems or []) if s.present}
        masking_pairs = sorted(
            (
                p for p in (stems.masking_pairs or [])
                if _fin(p.overlap_db, 0.0) >= _MASK_PAIR_MIN_OVERLAP_DB
                and str(p.masker) in stems_by_kind
                and str(p.maskee) in stems_by_kind
            ),
            key=lambda p: -_mask_ratio(p),
        )

    return _Ctx(
        m=m,
        genre_key=key,
        profile=profile,
        curve=curve,
        bands=bands,
        duration=max(_fin(m.duration_seconds, 0.0), 0.0),
        is_mono=bool(m.is_mono or m.stereo.is_mono_source),
        no_programme=(
            _fin(m.loudness.integrated_lufs, -70.0) <= NO_PROGRAMME_LUFS
            or _fin(m.loudness.true_peak_dbtp, -120.0) <= NO_PROGRAMME_LUFS
        ),
        stems=stems,
        sections=m.sections or SectionAnalysis(),
        stems_by_kind=stems_by_kind,
        masking_pairs=masking_pairs,
    )


# ---------------------------------------------------------------------------
# 1. Clipping & true peak
# ---------------------------------------------------------------------------


def _detect_clipping(ctx: _Ctx) -> List[_Hit]:
    """Pinned samples and delivery headroom.

    Genre-independent by design: 0 dBFS is arithmetic, and a lossy encoder
    overshoots a trap master exactly as far as it overshoots a string quartet.
    """
    c = ctx.m.clipping
    hits: List[_Hit] = []

    clip_pct = _fin(c.clip_percentage, 0.0)
    longest = int(_fin(c.longest_flat_run, 0))
    runs = int(_fin(c.flat_run_count, 0))
    distortion = _fin(c.distortion_index, 0.0)
    true_peak = _fin(c.true_peak_dbtp, -120.0)
    overs = int(_fin(c.inter_sample_overs, 0))

    tp_window = (-120.0, TRUE_PEAK_CEILING_DBTP)
    tp_miss = targets.range_miss(true_peak, tp_window)

    # -- flat tops: the waveform itself is damaged ---------------------------
    # Window is (0, 0): a mix that has been clipped has samples pinned at its
    # own ceiling, and the correct number of those is zero.
    if clip_pct > 0.0 and runs >= 1:
        ratio = max(
            _ratio(clip_pct, 0.03),          # 0.03 % pinned is already audible
            _ratio(longest, 12.0),           # 12 consecutive samples is a plateau
            _ratio(distortion, 0.12),        # HF products the clean frames lack
        )
        ratio = min(ratio, 6.0)
        evidence = [
            _ev("Clipped samples", c.clipped_samples, "samples",
                target=0.0, verdict="problem",
                detail=f"{clip_pct:.4f}% of all samples sit at the file's ceiling."),
            _ev("Longest flat run", longest, "samples", target=0.0,
                verdict="problem" if longest >= 12 else "watch",
                detail="Consecutive samples with no movement — a plateau, not a peak."),
            _ev("Flat-topped runs", runs, "runs", target=0.0, verdict="problem"),
            _ev("Distortion index", distortion, "",
                target_range=(0.0, _distortion_ceiling(ctx.profile)),
                verdict=_verdict(distortion, (0.0, _distortion_ceiling(ctx.profile)), 0.06),
                detail="High-frequency energy in clipped frames vs clean frames "
                       "from this same track."),
            _ev("True peak", true_peak, "dBTP", target=TRUE_PEAK_CEILING_DBTP,
                target_range=tp_window, verdict=_verdict(true_peak, tp_window, 1.0)),
            _ev("Sample peak", _fin(c.sample_peak_dbfs), "dBFS",
                target=TRUE_PEAK_CEILING_DBTP, verdict="problem"),
        ]
        moments = _moments(c.events)
        detail = (
            f"{c.clipped_samples:,} samples ({clip_pct:.4f}% of the file) are pinned at the "
            f"ceiling in {runs} flat-topped runs, the longest {longest} samples long, and the "
            f"clipped frames carry {distortion:.3f} on the distortion index, measured against "
            f"this track's own unclipped frames — the waveform tops are squared off, so that is "
            f"harmonic content the material never had, baked into the render rather than gain "
            f"that can be turned back down."
            + _moment_span(moments)
        )
        if bool(c.is_float_over_unity):
            detail += (
                " The file is float and peaks above unity, so the ceiling measured against "
                "is the material's own, not 0 dBFS."
            )
        hits.append(_Hit(
            Finding(
                id="clipping.hard_clipping",
                dimension="clipping",
                title="Hard clipping: the waveform tops are squared off",
                severity=_severity(ratio),
                confidence=0.97,   # counted samples, not an inference
                detail=detail,
                evidence=evidence,
                moments=moments,
            ),
            ratio,
        ))
        ctx.tags.add("clipping")

    # -- delivery headroom ---------------------------------------------------
    # Folded into the clipping finding above when both are true: a mix pinned at
    # its ceiling and a mix over -1 dBTP are the same fact stated twice.
    elif tp_miss > 0.0:
        ratio = _ratio(tp_miss, 1.0)
        detail = (
            f"True peak reaches {_num(true_peak, 2)} dBTP, {_num(tp_miss, 2)} dB above the "
            f"{_num(TRUE_PEAK_CEILING_DBTP, 1)} dBTP ceiling every major platform asks for "
            f"(Amazon Music wants -2.0). Nothing is clipped in the file itself — sample peak "
            f"is {_num(c.sample_peak_dbfs, 2)} dBFS — but MP3 and AAC reconstruct the waveform "
            f"between the samples, and the decoder will clip what the file does not."
        )
        if overs > 0:
            detail += (
                f" 4x oversampling already finds {overs:,} inter-sample overs that are not "
                f"over at base rate."
            )
        hits.append(_Hit(
            Finding(
                id="clipping.true_peak_over",
                dimension="clipping",
                title="True peak above the delivery ceiling",
                severity=_severity(ratio),
                confidence=0.96,
                detail=detail,
                evidence=[
                    _ev("True peak", true_peak, "dBTP", target=TRUE_PEAK_CEILING_DBTP,
                        target_range=tp_window, verdict="problem"),
                    _ev("Sample peak", _fin(c.sample_peak_dbfs), "dBFS",
                        target=TRUE_PEAK_CEILING_DBTP,
                        verdict=_verdict(_fin(c.sample_peak_dbfs), tp_window, 1.0)),
                    _ev("Inter-sample overs", overs, "samples", target=0.0,
                        verdict="problem" if overs else "good"),
                ],
                moments=_moments(c.events),
            ),
            ratio,
        ))
        ctx.tags.add("clipping")

    return hits


# ---------------------------------------------------------------------------
# 2. Phase & mono compatibility
# ---------------------------------------------------------------------------


def _detect_phase(ctx: _Ctx) -> List[_Hit]:
    """Polarity, correlation and what survives a fold-down.

    Two different kinds of threshold meet here and it matters which is which.
    *How wide a field the genre wants* is taste, and comes from
    `profile.correlation_min` — ambient's 0.10 against hip-hop's 0.40 is why a
    deliberately decorrelated ambient bed is never called a fault. *How much a
    mono fold-down may lose* is arithmetic: two fully decorrelated channels lose
    3.01 dB when summed and nothing legitimately loses more, in any genre.

    A mono source has no inter-channel relationship to break, so nothing is
    reported at all.
    """
    if ctx.is_mono:
        return []

    p = ctx.m.phase
    hits: List[_Hit] = []

    corr = _clamp(_fin(p.correlation, 1.0), -1.0, 1.0)
    loss = _fin(p.mono_sum_loss_db, 0.0)
    corr_min = _fin(ctx.profile.correlation_min, 0.30)
    corr_window = (corr_min, 1.0)
    loss_window = (MONO_LOSS_FLOOR_DB, 60.0)
    corr_miss = targets.range_miss(corr, corr_window)
    loss_miss = targets.range_miss(loss, loss_window)
    moments = _moments(p.problem_moments)

    worst_band = p.worst_band or ""
    worst_band_corr = _fin(p.worst_band_correlation, 1.0)
    band_losses = {k: _fin(v, 0.0) for k, v in (p.band_mono_loss_db or {}).items()}
    worst_loss_band, worst_loss = ("", 0.0)
    if band_losses:
        worst_loss_band = min(band_losses, key=lambda k: band_losses[k])
        worst_loss = band_losses[worst_loss_band]

    def _shared_evidence() -> List[Evidence]:
        return [
            _ev("Correlation", corr, "", target_range=corr_window,
                verdict=_verdict(corr, corr_window, 0.35),
                detail=f"{ctx.profile.label} sits above {_num(corr_min, 2)}."),
            _ev("Mono sum loss", loss, "dB", target_range=loss_window,
                verdict=_verdict(loss, loss_window, 2.0),
                detail="Level of (L+R)/2 against the average channel level; "
                       "-3.0 dB is the arithmetic cost of full decorrelation."),
            _ev("Worst band correlation", worst_band_corr, "",
                target_range=corr_window,
                verdict=_verdict(worst_band_corr, corr_window, 0.35),
                detail=f"Macro band: {worst_band or 'n/a'}."),
            _ev(f"Mono loss in {worst_loss_band or 'worst band'}", worst_loss, "dB",
                target_range=loss_window,
                verdict=_verdict(worst_loss, loss_window, 2.0)),
        ]

    # -- polarity inversion --------------------------------------------------
    # Severity comes from the fold-down loss rather than from the correlation
    # miss, precisely so it does not soften for a genre with a wide window:
    # ambient is allowed a decorrelated field, not a cancelling one.
    if bool(p.polarity_inverted):
        ratio = _ratio(loss_miss if loss_miss else (corr - (-1.0)) - 1.0, 2.0)
        ratio = max(ratio, _ratio(corr_miss, 0.35) * 0.5, 1.6)
        evidence = _shared_evidence()
        low_mono = _fin(ctx.m.low_end.low_end_mono_ratio, 1.0)
        evidence.append(
            _ev("Low-end mono ratio", low_mono, "",
                target_range=(_fin(ctx.profile.low_end_mono_min, 0.8), 1.0),
                verdict=_verdict(low_mono,
                                 (_fin(ctx.profile.low_end_mono_min, 0.8), 1.0), 0.06),
                detail="Share of sub-120 Hz energy that is mono.")
        )
        detail = (
            f"The channels are in opposite polarity: correlation reads {_num(corr, 2)} and "
            f"summing to mono costs {_num(loss, 1)} dB, against the {_num(MONO_LOSS_FLOOR_DB, 1)} dB "
            f"that full decorrelation alone would cost. {worst_loss_band.replace('_', ' ') or 'The worst band'} "
            f"loses {_num(worst_loss, 1)} dB and only {low_mono * 100:.0f}% of the energy below "
            f"120 Hz survives the fold-down. This is a flipped cable or an inverted plugin, not "
            f"a width choice — on any mono playback (club sub, phone speaker, most Bluetooth "
            f"speakers) the mix largely disappears."
            + _moment_span(moments)
        )
        hits.append(_Hit(
            Finding(
                id="phase.polarity_inverted",
                dimension="phase",
                title="Right channel is polarity-inverted",
                severity=_severity(ratio),
                confidence=0.96,   # a correlation coefficient, directly computed
                detail=detail,
                evidence=evidence,
                moments=moments,
            ),
            ratio,
        ))
        ctx.tags.add("polarity")
        return hits

    # -- correlation below what the genre wants ------------------------------
    if corr_miss < 0.0 or loss_miss < 0.0:
        ratio = max(_ratio(corr_miss, 0.35), _ratio(loss_miss, 2.0))
        driver = "correlation" if _ratio(corr_miss, 0.35) >= _ratio(loss_miss, 2.0) else "fold-down"
        detail = (
            f"Inter-channel correlation is {_num(corr, 2)}, under the {_num(corr_min, 2)} floor "
            f"{ctx.profile.label} mixes hold, and folding to mono costs {_num(loss, 1)} dB "
            f"(decorrelation alone costs 3.0). The weakest band is "
            f"{worst_band.replace('_', ' ') or 'n/a'} at {_num(worst_band_corr, 2)}. "
            f"Anything played back in mono loses that much of this mix."
            + _moment_span(moments)
        )
        hits.append(_Hit(
            Finding(
                id="phase.mono_incompatible",
                dimension="phase",
                title=f"Mono fold-down loses level ({driver})",
                severity=_severity(ratio),
                confidence=0.93,
                detail=detail,
                evidence=_shared_evidence(),
                moments=moments,
            ),
            ratio,
        ))
        ctx.tags.add("phase")
        return hits

    # -- one band cancelling under an otherwise healthy broadband figure -----
    # This is the case the headline number cannot catch: the sub is a small
    # share of the energy, so it can be smeared across the field while the
    # broadband correlation still reads +0.8.
    if worst_loss_band and targets.range_miss(worst_loss, (-6.0, 60.0)) < 0.0:
        ratio = _ratio(targets.range_miss(worst_loss, (-6.0, 60.0)), 2.0)
        detail = (
            f"Broadband correlation is fine at {_num(corr, 2)}, but the "
            f"{worst_loss_band.replace('_', ' ')} band loses {_num(worst_loss, 1)} dB when summed "
            f"to mono and correlates at {_num(_fin((ctx.m.stereo.band_correlation or {}).get(worst_loss_band, 1.0)), 2)}. "
            f"A single band cancelling under a healthy headline figure is what makes a mix "
            f"sound complete on headphones and hollow on a mono system."
        )
        hits.append(_Hit(
            Finding(
                id="phase.band_cancellation",
                dimension="phase",
                title=f"{worst_loss_band.replace('_', ' ').title()} cancels on fold-down",
                severity=_severity(ratio),
                confidence=0.88,
                detail=detail,
                evidence=_shared_evidence(),
                moments=moments,
            ),
            ratio,
        ))
        ctx.tags.add("phase")

    return hits


# ---------------------------------------------------------------------------
# 3. Loudness
# ---------------------------------------------------------------------------


def _detect_loudness(ctx: _Ctx) -> List[_Hit]:
    """Integrated level against where this genre's releases actually sit.

    The quiet side deliberately does not mirror the loud side. A master that is
    under the window but still has headroom is not a defective mix — it is a
    fader move away from the window, and every platform leaves quiet tracks
    alone. So "too quiet" only fires when the track *cannot* reach the window
    with clean gain: `integrated + (ceiling - true_peak)` still under it. That
    is the case where getting to level costs limiting, which is a real problem.
    """
    if ctx.no_programme or not ctx.has("loudness"):
        return []

    lo = ctx.m.loudness
    window = ctx.profile.integrated_lufs
    integrated = _fin(lo.integrated_lufs, -70.0)
    true_peak = _fin(lo.true_peak_dbtp, -120.0)
    miss = targets.range_miss(integrated, window)
    if miss == 0.0:
        return []

    spotify_delta = integrated - (-14.0)
    headroom = TRUE_PEAK_CEILING_DBTP - true_peak
    attainable = integrated + headroom

    evidence = [
        _ev("Integrated loudness", integrated, "LUFS", target_range=window,
            verdict=_verdict(integrated, window, 1.5),
            detail=f"{ctx.profile.label} releases sit at {_win(window, 1, ' LUFS')}."),
        _ev("True peak", true_peak, "dBTP", target=TRUE_PEAK_CEILING_DBTP,
            verdict=_verdict(true_peak, (-120.0, TRUE_PEAK_CEILING_DBTP), 1.0)),
        _ev("Clean gain available", headroom, "dB", target=None, verdict="good",
            detail="Level that can be added before hitting -1.0 dBTP, with no limiting."),
        _ev("Loudness reachable with clean gain", attainable, "LUFS",
            target_range=window, verdict=_verdict(attainable, window, 1.5)),
        _ev("Delta vs Spotify reference", spotify_delta, "LU", target=0.0,
            verdict="problem" if spotify_delta > 1.0 else "good",
            detail="Spotify, YouTube and Tidal normalise to -14 LUFS."),
        _ev("Peak to loudness ratio", _fin(lo.plr_db), "dB",
            target_range=ctx.profile.psr_p10_db,
            verdict=_verdict(_fin(lo.plr_db), ctx.profile.psr_p10_db, 1.5)),
    ]

    if miss > 0.0:
        ratio = _ratio(miss, 1.5)
        detail = (
            f"Integrated loudness is {_num(integrated, 2)} LUFS, {_num(miss, 2)} LU above the "
            f"{_win(window, 1, ' LUFS')} window {ctx.profile.label} masters sit in. "
            f"Spotify, YouTube and Tidal all normalise to -14 LUFS, so this is turned down "
            f"{_num(spotify_delta, 1)} LU on playback — the loudness is handed back, while "
            f"whatever was done to reach it (PLR is {_num(lo.plr_db, 1)} dB, loudness range "
            f"{_num(lo.loudness_range_lu, 1)} LU) stays in the file."
        )
        return [_Hit(
            Finding(
                id="loudness.too_loud",
                dimension="loudness",
                title=f"Louder than {ctx.profile.label} masters run",
                severity=_severity(ratio),
                confidence=ctx.trust(0.96, "loudness"),
                detail=detail,
                evidence=evidence,
            ),
            ratio,
        )]

    # Quiet side: only a problem if clean gain cannot get there.
    attain_miss = targets.range_miss(attainable, window)
    if attain_miss >= 0.0:
        return []

    ratio = _ratio(attain_miss, 2.0)
    detail = (
        f"Integrated loudness is {_num(integrated, 2)} LUFS against a {_win(window, 1, ' LUFS')} "
        f"window for {ctx.profile.label}, and there is only {_num(headroom, 1)} dB of clean gain "
        f"left before -1.0 dBTP — turning it up as far as the peaks allow still lands at "
        f"{_num(attainable, 2)} LUFS, {_num(abs(attain_miss), 2)} LU short. Reaching level from "
        f"here costs limiting, so this is a gain-staging problem in the mix rather than a "
        f"mastering fader move."
    )
    return [_Hit(
        Finding(
            id="loudness.cannot_reach_level",
            dimension="loudness",
            title="Cannot reach genre level without limiting",
            severity=_severity(ratio),
            confidence=ctx.trust(0.92, "loudness"),
            detail=detail,
            evidence=evidence,
        ),
        ratio,
    )]


# ---------------------------------------------------------------------------
# 4. Limiter behaviour
# ---------------------------------------------------------------------------


def _distortion_ceiling(profile: targets.GenreProfile) -> float:
    """How much limiter-generated harmonic content this genre absorbs.

    `targets.py` has no window for `distortion_index`, and rather than invent a
    second table this is derived from the one figure that already encodes how
    hard the genre drives its ceiling: the low edge of the PSR window. A genre
    that accepts 4 dB of short-term peak-to-loudness is a genre mastered into
    the limiter; one that expects 12 dB is not, and a limiter throwing harmonics
    there is a mistake. The mapping puts trap and EDM at 0.16, pop at 0.13, rock
    at 0.12, folk at 0.09 and classical at 0.05.
    """
    psr_low = _fin(profile.psr_p10_db[0], 6.0)
    return _clamp(0.22 - 0.015 * psr_low, 0.05, 0.18)


def _detect_limiter(ctx: _Ctx) -> List[_Hit]:
    """How the mix reaches its ceiling, as distinct from whether it goes over it.

    `clipping` owns the digital fact (samples pinned, true peak over spec).
    This owns the consequence: short-term peak-to-loudness crushed flat, and a
    limiter generating harmonics instead of holding a ceiling. Both can be true
    of the same file and they have different fixes — one is the master fader,
    the other is the limiter's input drive.
    """
    if ctx.no_programme or not ctx.has("limiter"):
        return []

    lo = ctx.m.loudness
    c = ctx.m.clipping
    d = ctx.m.dynamics

    psr = _fin(lo.psr_p10_db, 0.0)
    psr_window = ctx.profile.psr_p10_db
    psr_miss = min(targets.range_miss(psr, psr_window), 0.0)   # only the low side

    distortion = _fin(c.distortion_index, 0.0)
    dist_max = _distortion_ceiling(ctx.profile)
    dist_window = (0.0, dist_max)
    dist_miss = targets.range_miss(distortion, dist_window)

    clip_pct = _fin(c.clip_percentage, 0.0)
    runs = int(_fin(c.flat_run_count, 0))
    flat_drive = _ratio(clip_pct, 0.03) if runs >= 3 else 0.0

    ratio = min(max(_ratio(psr_miss, 1.5), _ratio(dist_miss, 0.06), flat_drive), 6.0)
    if ratio <= 0.0:
        return []

    gr = _fin(d.gain_reduction_estimate_db, 0.0)
    evidence = [
        _ev("Short-term PSR (10th pct)", psr, "dB", target_range=psr_window,
            verdict=_verdict(psr, psr_window, 1.5),
            detail=f"{ctx.profile.label} holds {_win(psr_window, 1, ' dB')} of peak over "
                   f"short-term loudness."),
        _ev("Median PSR", _fin(lo.psr_median_db), "dB", target_range=psr_window,
            verdict=_verdict(_fin(lo.psr_median_db), psr_window, 1.5)),
        _ev("Distortion index", distortion, "", target_range=dist_window,
            verdict=_verdict(distortion, dist_window, 0.06),
            detail="High-frequency products in ceiling-pinned frames, referenced to "
                   "this track's own clean frames."),
        _ev("Flat-topped runs at the ceiling", runs, "runs", target=0.0,
            verdict="problem" if runs >= 3 else "good",
            detail=f"{clip_pct:.4f}% of samples pinned; longest run "
                   f"{int(_fin(c.longest_flat_run))} samples."),
        _ev("Estimated gain reduction", gr, "dB", target=None,
            verdict="watch" if gr >= 3.0 else "good",
            detail="Inferred from crest collapse and peak pinning in 50 ms windows — "
                   "an estimate, not a reading off the limiter's meter."),
        _ev("Peak to loudness ratio", _fin(lo.plr_db), "dB", target_range=psr_window,
            verdict=_verdict(_fin(lo.plr_db), psr_window, 1.5)),
    ]

    parts: List[str] = []
    if psr_miss < 0.0:
        parts.append(
            f"short-term peak-to-loudness bottoms out at {_num(psr, 2)} dB against a "
            f"{_win(psr_window, 1, ' dB')} window for {ctx.profile.label} "
            f"({_num(abs(psr_miss), 2)} dB under)"
        )
    if dist_miss > 0.0:
        parts.append(
            f"the distortion index is {distortion:.3f} against {dist_max:.2f} for this genre"
        )
    if flat_drive > 0.0:
        parts.append(
            f"{runs} runs of samples are pinned flat at the ceiling ({clip_pct:.4f}% of the "
            f"file, longest {int(_fin(c.longest_flat_run))} samples)"
        )
    if not parts:
        parts.append(f"short-term peak-to-loudness sits at {_num(psr, 2)} dB")

    detail = (
        "The limiter is being driven past holding a ceiling and into generating one: "
        + "; ".join(parts)
        + f". Estimated gain reduction is {_num(gr, 1)} dB and micro-dynamics are down to "
        f"{_num(d.micro_dynamics_db, 1)} dB. Back the input drive off and let the ceiling "
        f"do less work — the level lost is recoverable, the harmonics are not."
    )

    # Confidence follows the weakest link in the case being made. PSR is
    # measured directly; the distortion index and the gain-reduction figure are
    # inferences from the mixed file.
    confidence = 0.88 if psr_miss < 0.0 else 0.72

    ctx.tags.add("limiter")
    return [_Hit(
        Finding(
            id="limiter.over_driven",
            dimension="limiter",
            title="Limiter driven past its ceiling",
            severity=_severity(ratio),
            confidence=ctx.trust(confidence, "limiter"),
            detail=detail,
            evidence=evidence,
            moments=_moments(c.events, 4),
        ),
        ratio,
    )]


# ---------------------------------------------------------------------------
# 5. Arrangement: what the record does across its own length
#
# Runs before dynamic range on purpose. Both can be looking at the same
# flatness, and this one is the more specific description of it — it names the
# section, the timestamp and the lift in LU instead of reporting a loudness
# range — so it speaks first and `_detect_dynamic_range` stands down on its LRA
# arm afterwards.
# ---------------------------------------------------------------------------


def _detect_arrangement(ctx: _Ctx) -> List[_Hit]:
    """The chorus that does not lift, and the section whose bottom drops out.

    Neither of these moves a whole-file statistic enough to see. A hook that
    arrives at verse level leaves integrated loudness untouched; a drop that
    loses its sub while everything else stays loud leaves the long-term
    spectrum untouched. Both are obvious the moment somebody plays the record
    end to end, which is what `sections.py` measures and this reads.

    Both are genre-gated, and the gates are derived rather than invented — see
    `_expects_arrangement_lift` and `_low_swing_ceiling`. Ambient and classical
    are not told their chorus should be 6 dB louder.
    """
    sa = ctx.sections
    if ctx.no_programme or not sa.available:
        return []

    sections = list(sa.sections or [])
    n = len(sections)
    if n < _MIN_SECTIONS_FOR_FORM or ctx.duration < _MIN_FORM_SEC:
        # One span, or a track too short to have parts. `sections.py` already
        # says so in its own notes; inventing a verdict on top would be worse
        # than the silence.
        return []

    hits: List[_Hit] = []
    lift = _fin(sa.peak_lift_db, 0.0)
    spread = _fin(sa.loudness_spread_lu, 0.0)
    loudest = next((s for s in sections if s.is_loudest), sections[0])
    quietest = next((s for s in sections if s.is_quietest), sections[-1])
    lra = _fin(ctx.m.loudness.loudness_range_lu, 0.0)

    # -- the loudest section does not arrive ---------------------------------
    lift_window = (_CHORUS_LIFT_MIN_LU, 60.0)
    lift_miss = targets.range_miss(lift, lift_window)
    if _expects_arrangement_lift(ctx.profile) and lift_miss < 0.0:
        ratio = _ratio(lift_miss, _CHORUS_LIFT_SCALE)
        part = loudest.label if not loudest.label.startswith("section ") else "loudest section"
        detail = (
            f"Across {n} measured sections the loudest one ({part} at "
            f"{_clock(loudest.t_start)}) sits only {_num(lift, 1)} LU above the median "
            f"section, against the {_num(_CHORUS_LIFT_MIN_LU, 1)} LU a {ctx.profile.label} "
            f"arrangement needs before a listener reads it as the payoff — most records "
            f"put 2-4 LU there. Loudest to quietest across the whole record is "
            f"{_num(spread, 1)} LU and EBU loudness range is {_num(lra, 1)} LU. "
            f"Every part arrives at the same size, so nothing in the arrangement lands; "
            f"this is a fader and an arrangement problem, not a mastering one, and no "
            f"amount of limiting on the master will put the lift back."
        )
        hits.append(_Hit(
            Finding(
                id="dynamic_range.no_section_lift",
                dimension="dynamic_range",
                title="The loudest section does not lift",
                # Every LUFS figure here is a direct BS.1770 measurement; the
                # uncertainty is in where the boundaries were placed, which is
                # a segmenter's judgement.
                confidence=ctx.trust(0.82, "arrangement"),
                severity=_severity(ratio),
                detail=detail,
                evidence=[
                    _ev("Loudest section vs median", lift, "LU", target_range=lift_window,
                        verdict=_verdict(lift, lift_window, _CHORUS_LIFT_SCALE),
                        detail=f"{part} at {_clock(loudest.t_start)}, "
                               f"{_num(loudest.integrated_lufs, 1)} LUFS."),
                    _ev("Loudest to quietest section", spread, "LU", target=None,
                        verdict="watch" if spread < 2.0 else "good",
                        detail=f"{loudest.label} at {_clock(loudest.t_start)} against "
                               f"{quietest.label} at {_clock(quietest.t_start)}."),
                    _ev("Loudness range", lra, "LU",
                        target_range=ctx.profile.loudness_range_lu,
                        verdict=_verdict(lra, ctx.profile.loudness_range_lu, 1.5),
                        detail=f"EBU 3342. {ctx.profile.label} runs "
                               f"{_win(ctx.profile.loudness_range_lu, 1, ' LU')}."),
                    _ev("Sections measured", float(n), "sections", verdict="good"),
                ],
                moments=[
                    Moment(
                        t_start=round(_fin(loudest.t_start), 3),
                        t_end=round(_fin(loudest.t_end), 3),
                        intensity=round(_clamp(1.0 - lift / _CHORUS_LIFT_MIN_LU, 0.0, 1.0), 3),
                        value=round(lift, 2),
                        label=f"{part}: +{_num(lift, 1)} LU",
                    )
                ],
            ),
            ratio,
        ))
        ctx.tags.add("section_lift")

    # -- one section's bottom end falls out ----------------------------------
    low_rel, lufs, core = _section_low_shares(sections)
    swing = _fin(sa.low_end_swing_db, 0.0)
    swing_max = _low_swing_ceiling(ctx.profile)
    swing_window = (0.0, swing_max)
    swing_miss = targets.range_miss(swing, swing_window)
    core_idx = np.flatnonzero(core)
    if swing_miss > 0.0 and core_idx.size >= 2:
        ratio = min(_ratio(swing_miss, _LOW_SWING_SCALE_DB), 6.0)
        weak_i = int(core_idx[int(np.argmin(low_rel[core_idx]))])
        strong_i = int(core_idx[int(np.argmax(low_rel[core_idx]))])
        weakest, strongest = sections[weak_i], sections[strong_i]
        weak_share = _fin(low_rel[weak_i], 0.0)
        strong_share = _fin(low_rel[strong_i], 0.0)
        detail = (
            f"Sub and low bass carry {_num(weak_share, 1)} dB of {weakest.label}'s own "
            f"energy at {_clock(weakest.t_start)} against {_num(strong_share, 1)} dB in "
            f"{strongest.label} at {_clock(strongest.t_start)} — a {_num(swing, 1)} dB swing "
            f"across the {int(core_idx.size)} sections carrying this record, over the "
            f"{_num(swing_max, 1)} dB {ctx.profile.label} holds. This is measured as each "
            f"section's *share* of its own level, so it is not the quiet parts being quiet: "
            f"{weakest.label} is within {_num(_LOW_CORE_LU, 0)} LU of the loudest section and "
            f"still has no bottom under it. On a full-range system that part of the "
            f"arrangement drops out from underneath."
        )
        hits.append(_Hit(
            Finding(
                id="low_end.section_collapse",
                dimension="low_end",
                title=f"Low end collapses in {weakest.label}",
                severity=_severity(ratio),
                confidence=ctx.trust(0.84, "arrangement"),
                detail=detail,
                evidence=[
                    _ev("Low-end swing across sections", swing, "dB",
                        target_range=swing_window,
                        verdict=_verdict(swing, swing_window, _LOW_SWING_SCALE_DB),
                        detail="Sub+low-bass share of each section's own total, over the "
                               "sections within 9 LU of the loudest."),
                    _ev(f"Low-end share in {weakest.label}", weak_share, "dB",
                        target=strong_share, verdict="problem",
                        detail=f"{_clock(weakest.t_start)}-{_clock(weakest.t_end)}, "
                               f"{_num(weakest.integrated_lufs, 1)} LUFS."),
                    _ev(f"Low-end share in {strongest.label}", strong_share, "dB",
                        target=None, verdict="good",
                        detail=f"{_clock(strongest.t_start)}-{_clock(strongest.t_end)}, "
                               f"{_num(strongest.integrated_lufs, 1)} LUFS."),
                    _ev("Sections carrying the record", float(core_idx.size), "sections",
                        verdict="good"),
                ],
                band_hz=(20.0, 120.0),
                moments=[
                    Moment(
                        t_start=round(_fin(weakest.t_start), 3),
                        t_end=round(_fin(weakest.t_end), 3),
                        intensity=round(_clamp(swing / 12.0, 0.0, 1.0), 3),
                        value=round(weak_share, 2),
                        label=f"{weakest.label}: no low end",
                    )
                ],
            ),
            ratio,
        ))
        ctx.tags.add("section_low_end")

    return hits


# ---------------------------------------------------------------------------
# 6. Dynamic range (macro: whole-file and section-to-section)
# ---------------------------------------------------------------------------


def _detect_dynamic_range(ctx: _Ctx) -> List[_Hit]:
    """Crest, TT-DR and loudness range against the genre's windows.

    Loudness range is only consulted on material long enough to have an
    arrangement — on a 20 s loop it measures the loop, not the record.
    """
    if ctx.no_programme or not ctx.has("dynamics"):
        return []

    d = ctx.m.dynamics
    lo = ctx.m.loudness

    crest = _fin(d.crest_factor_db, 0.0)
    crest_window = ctx.profile.crest_factor_db
    crest_miss = targets.range_miss(crest, crest_window)

    lra = _fin(lo.loudness_range_lu, 0.0)
    lra_window = ctx.profile.loudness_range_lu
    lra_miss = targets.range_miss(lra, lra_window) if ctx.has("arrangement") else 0.0

    # The arrangement detector has already reported this flatness, and it did it
    # better: it names the section that fails to lift, when it happens, and by
    # how much. A loudness range is the same observation with the timestamps
    # thrown away, so it is dropped rather than said twice (rule 5). Crest is a
    # different measurement and keeps its arm.
    lra_owned = "section_lift" in ctx.tags
    if lra_owned:
        lra_miss = 0.0

    evidence = [
        _ev("Crest factor", crest, "dB", target_range=crest_window,
            verdict=_verdict(crest, crest_window, 1.5),
            detail=f"{ctx.profile.label} runs {_win(crest_window, 1, ' dB')}."),
        _ev("TT-DR value", _fin(d.dr_value), "dB", target_range=crest_window,
            verdict=_verdict(_fin(d.dr_value), crest_window, 2.0),
            detail="Peak minus RMS over the loudest 20% of 3 s blocks."),
        _ev("Loudness range", lra, "LU", target_range=lra_window,
            verdict=_verdict(lra, lra_window, 1.5) if ctx.has("arrangement") else "good",
            detail=(
                "EBU 3342 range, reported under 'The loudest section does not lift', "
                "which measures the same flatness section by section."
                if lra_owned else
                "EBU 3342 range." if ctx.has("arrangement")
                else f"Not scored: {_num(ctx.duration, 0)} s of audio is too short for a "
                     f"section-to-section range to mean anything.")),
        _ev("Macro dynamics", _fin(d.macro_dynamics_lu), "LU", target_range=lra_window,
            verdict="good"),
        _ev("Integrated loudness", _fin(lo.integrated_lufs), "LUFS",
            target_range=ctx.profile.integrated_lufs,
            verdict=_verdict(_fin(lo.integrated_lufs), ctx.profile.integrated_lufs, 1.5)),
    ]

    # -- squashed ------------------------------------------------------------
    if crest_miss < 0.0 or lra_miss < 0.0:
        ratio = max(_ratio(crest_miss, 1.5), _ratio(lra_miss, 1.5))
        parts = []
        if crest_miss < 0.0:
            parts.append(
                f"crest factor is {_num(crest, 1)} dB against {_win(crest_window, 1, ' dB')} "
                f"for {ctx.profile.label} and TT-DR reads {_num(d.dr_value, 1)}"
            )
        if lra_miss < 0.0:
            parts.append(
                f"loudness range is {_num(lra, 1)} LU against {_win(lra_window, 1, ' LU')}"
            )
        detail = (
            "There is less level movement here than the genre lives on: "
            + "; ".join(parts)
            + f". At {_num(lo.integrated_lufs, 1)} LUFS integrated, the difference between the "
            f"quietest and loudest moment of this mix is small enough that nothing lands — "
            f"every section arrives at the same size."
        )
        return [_Hit(
            Finding(
                id="dynamic_range.squashed",
                dimension="dynamic_range",
                title=f"Flatter than {ctx.profile.label} needs",
                severity=_severity(ratio),
                confidence=ctx.trust(0.90, "dynamics"),
                detail=detail,
                evidence=evidence,
            ),
            ratio,
        )]

    # -- more dynamic than the genre's masters -------------------------------
    # Only reported when the level corroborates it: a wide crest at genre level
    # is a well-made record, a wide crest 6 LU under it is an unmastered mix.
    # A 1.5 dB grace keeps borderline cases quiet.
    level_miss = targets.range_miss(_fin(lo.integrated_lufs, -70.0), ctx.profile.integrated_lufs)
    if crest_miss > 1.5 and level_miss < 0.0:
        ratio = min(_ratio(crest_miss - 1.5, 2.0), 2.4)   # never critical: not damage
        detail = (
            f"Crest factor is {_num(crest, 1)} dB and TT-DR {_num(d.dr_value, 1)}, above the "
            f"{_win(crest_window, 1, ' dB')} that {ctx.profile.label} masters carry, and "
            f"integrated loudness is {_num(lo.integrated_lufs, 1)} LUFS — "
            f"{_num(abs(level_miss), 1)} LU under the {_win(ctx.profile.integrated_lufs, 1, ' LUFS')} "
            f"window. Nothing is broken; this reads as a mix that has not been mastered yet, "
            f"and it will sit quiet and soft next to anything else in the genre."
        )
        return [_Hit(
            Finding(
                id="dynamic_range.unmastered",
                dimension="dynamic_range",
                title=f"More dynamic range than {ctx.profile.label} masters carry",
                severity=_severity(ratio),
                confidence=ctx.trust(0.85, "dynamics"),
                detail=detail,
                evidence=evidence,
            ),
            ratio,
        )]

    return []


# ---------------------------------------------------------------------------
# 7. Compression (micro: inside the hit)
# ---------------------------------------------------------------------------


def _detect_compression(ctx: _Ctx) -> List[_Hit]:
    """What is left of the transient inside each hit, and whether the mix breathes.

    Pumping is the least trustworthy number in the whole analysis: percussive
    material modulates its own envelope at the beat rate with no compressor
    anywhere near it, which is why the DSP layer's own docstring warns about it.
    So it is never reported on its own — it needs the gain-reduction estimate
    and the micro-dynamics figure to corroborate, and even then its confidence
    says plainly that it is an inference.
    """
    if ctx.no_programme or not ctx.has("dynamics"):
        return []

    d = ctx.m.dynamics
    hits: List[_Hit] = []

    micro = _fin(d.micro_dynamics_db, 0.0)
    micro_window = ctx.profile.micro_dynamics_db
    micro_miss = min(targets.range_miss(micro, micro_window), 0.0)
    gr = _fin(d.gain_reduction_estimate_db, 0.0)
    pumping = _fin(d.pumping_index, 0.0)
    rate = _fin(d.pumping_rate_hz, 0.0)

    evidence = [
        _ev("Micro dynamics", micro, "dB", target_range=micro_window,
            verdict=_verdict(micro, micro_window, 1.5),
            detail=f"Crest inside a 50 ms window. {ctx.profile.label} keeps "
                   f"{_win(micro_window, 1, ' dB')}."),
        _ev("Estimated gain reduction", gr, "dB", target=None,
            verdict="watch" if gr >= 3.0 else "good",
            detail="Inferred from crest collapse in loud sections vs moderate ones."),
        _ev("Pumping index", pumping, "", target_range=(0.0, 0.55),
            verdict="watch" if pumping > 0.55 else "good",
            detail=f"Tempo-locked envelope modulation at {_num(rate, 2)} Hz. Percussive "
                   f"material modulates its own envelope at the beat rate, so this alone "
                   f"is not evidence of a compressor."),
        _ev("Crest factor", _fin(d.crest_factor_db), "dB",
            target_range=ctx.profile.crest_factor_db,
            verdict=_verdict(_fin(d.crest_factor_db), ctx.profile.crest_factor_db, 1.5)),
        _ev("Transient to sustain", _fin(ctx.m.transients.transient_to_sustain_db), "dB",
            target=None, verdict="good"),
    ]

    if micro_miss < 0.0:
        ratio = _ratio(micro_miss, 1.5)
        detail = (
            f"Inside a single hit there is only {_num(micro, 1)} dB between peak and RMS, "
            f"against the {_win(micro_window, 1, ' dB')} {ctx.profile.label} keeps, and the "
            f"crest collapse between loud and moderate sections puts estimated gain reduction "
            f"at {_num(gr, 1)} dB. Macro level movement can survive this "
            f"(crest factor is still {_num(d.crest_factor_db, 1)} dB) while every individual "
            f"hit has had its attack flattened — which is the mix that measures fine and "
            f"sounds dead."
        )
        hits.append(_Hit(
            Finding(
                id="compression.micro_dynamics_lost",
                dimension="compression",
                title="Transients flattened inside each hit",
                severity=_severity(ratio),
                confidence=ctx.trust(0.82, "dynamics"),
                detail=detail,
                evidence=evidence,
                moments=_moments(ctx.m.transients.weak_moments, 4),
            ),
            ratio,
        ))

    # Pumping: three independent signals must agree before this is claimed.
    pump_window = (0.0, 0.60)
    pump_miss = targets.range_miss(pumping, pump_window)
    if pump_miss > 0.0 and gr >= 2.5 and micro <= micro_window[0] + 1.0:
        ratio = min(_ratio(pump_miss, 0.15), 3.0)
        detail = (
            f"The broadband envelope is modulating at {_num(rate, 2)} Hz "
            f"({_num(rate * 60.0, 0)} BPM, against a detected tempo of "
            f"{_num(ctx.m.transients.estimated_tempo, 0)}) with a pumping index of "
            f"{pumping:.2f}, and it is corroborated: estimated gain reduction is "
            f"{_num(gr, 1)} dB and micro-dynamics are {_num(micro, 1)} dB against "
            f"{_win(micro_window, 1, ' dB')}. The whole mix breathes on the beat instead of "
            f"individual elements moving."
        )
        hits.append(_Hit(
            Finding(
                id="compression.pumping",
                dimension="compression",
                title="Whole mix breathing on the beat",
                severity=_severity(ratio),
                # Inferred from envelope periodicity, and the DSP layer cannot
                # separate a compressor from a groove. Said plainly.
                confidence=ctx.trust(0.45, "dynamics"),
                detail=detail,
                evidence=evidence,
            ),
            ratio,
        ))

    return hits


def _detect_stem_compression(ctx: _Ctx) -> List[_Hit]:
    """Compression on one source, which a two-track cannot see at all.

    Every compression figure above this is measured on the sum. That answers
    "has the master been crushed" and cannot answer "which element has been
    crushed", because a flattened drum bus under an untouched vocal and a
    flattened vocal over untouched drums produce the same crest factor on the
    master. With stems the question separates.

    Two cases, and the second is deliberately harder to trigger:

    * **Drums.** Crest factor against the genre's window for the *whole
      master*. That is a strong test on purpose: the drums are the most
      transient thing on the record, so a drum stem carrying less peak-to-RMS
      than the finished master is supposed to carry has been flattened. Trap's
      floor is 5.5 dB, pop's 8.0, classical's 14.0 — the genre does the work.
    * **Vocals.** Micro-dynamics against the genre's window, and only with the
      gain-reduction estimate agreeing. A vocal with little crest may just be
      an even performance, and calling that over-compression would be a taste
      judgement dressed up as a measurement.

    One finding, worst first: "the drums and the vocal are both squashed" is
    one session's work and one entry in the list.
    """
    if ctx.no_programme or not ctx.has_stems or not ctx.has("stem"):
        return []

    crest_window = ctx.profile.crest_factor_db
    micro_window = ctx.profile.micro_dynamics_db

    candidates: List[Tuple[float, StemMeasurement, str, float, Tuple[float, float], float]] = []
    for kind in ("drums", "vocals"):
        stem = ctx.stem(kind)
        if stem is None:
            continue
        gr = _fin(stem.gain_reduction_estimate_db, 0.0)
        if kind == "drums":
            value, window, scale, metric = (
                _fin(stem.crest_factor_db, 0.0), crest_window, 1.5, "crest factor"
            )
        else:
            if gr < _STEM_GR_CORROBORATION_DB:
                continue
            value, window, scale, metric = (
                _fin(stem.micro_dynamics_db, 0.0), micro_window, 1.5, "micro-dynamics"
            )
        miss = min(targets.range_miss(value, window), 0.0)   # only the flat side
        if miss >= 0.0:
            continue
        candidates.append((_ratio(miss, scale), stem, metric, value, window, gr))

    if not candidates:
        return []

    ratio, stem, metric, value, window, gr = max(candidates, key=lambda c: c[0])
    ratio = min(ratio, 6.0)
    if ratio < MIN_REPORTABLE_RATIO:
        return []

    kind = str(stem.kind)
    label = {"drums": "drum", "vocals": "vocal"}.get(kind, kind)
    consequence = (
        "The hits arrive without landing, and nothing on the master bus puts an attack "
        "back once it has been compressed off the source."
        if kind == "drums" else
        "The lead sits at one level with no push behind the loud lines, which reads as "
        "distant no matter how far the fader comes up."
    )
    detail = (
        f"The separated {label} stem carries {_num(value, 1)} dB of {metric} against the "
        f"{_win(window, 1, ' dB')} window {ctx.profile.label} holds on the finished master, "
        f"and its own crest collapse and peak pinning put an estimated {_num(gr, 1)} dB of "
        f"gain reduction on it. This is a per-element reading: the two-track's crest factor "
        f"is {_num(ctx.m.dynamics.crest_factor_db, 1)} dB and its micro-dynamics "
        f"{_num(ctx.m.dynamics.micro_dynamics_db, 1)} dB, which cannot tell a flattened "
        f"{label} under an untouched mix from a flattened master — separating them can. "
        f"{consequence} The fix is on the {label} bus, not the master."
    )

    return [_Hit(
        Finding(
            id=f"compression.stem_{kind}_flat",
            dimension="compression",
            title=f"The {label} stem is over-compressed",
            severity=_severity(ratio),
            # Directly measured on the separated source; the uncertainty that
            # is left belongs to the separation, not to the statistic.
            confidence=ctx.trust(0.86, "stem"),
            detail=detail,
            evidence=[
                _ev(f"{label.title()} stem {metric}", value, "dB", target_range=window,
                    verdict=_verdict(value, window, 1.5),
                    detail=f"Measured on the separated {label}, over the frames it is "
                           f"sounding on."),
                _ev(f"{label.title()} stem gain reduction", gr, "dB", target=None,
                    verdict="problem" if gr >= _STEM_GR_CORROBORATION_DB else "watch",
                    detail="Crest collapse between loud and moderate 50 ms frames, plus "
                           "peak pinning, on this source alone."),
                _ev(f"{label.title()} stem crest factor",
                    _fin(stem.crest_factor_db), "dB", target_range=crest_window,
                    verdict=_verdict(_fin(stem.crest_factor_db), crest_window, 1.5)),
                _ev(f"{label.title()} stem transient punch",
                    _fin(stem.transient_punch), "",
                    target_range=(_fin(ctx.profile.punch_min, 0.35), 1.0),
                    verdict=_verdict(_fin(stem.transient_punch),
                                     (_fin(ctx.profile.punch_min, 0.35), 1.0), 0.07)),
                _ev("Whole-mix crest factor", _fin(ctx.m.dynamics.crest_factor_db), "dB",
                    target_range=crest_window,
                    verdict=_verdict(_fin(ctx.m.dynamics.crest_factor_db), crest_window, 1.5),
                    detail="For comparison: the figure the two-track can produce."),
            ],
            moments=_moments(ctx.m.transients.weak_moments, 4),
        ),
        ratio,
    )]


# ---------------------------------------------------------------------------
# 8. Low end (kick / 808 relationship)
# ---------------------------------------------------------------------------


def _detect_low_end(ctx: _Ctx) -> List[_Hit]:
    """Kick against bass, sub weight, and whether the bottom is mono.

    The kick-versus-bass finding has two sources and prefers the better one.
    Without stems everything kick-related is gated on `kick_detected`, because
    the DSP layer applies regularity, definition and band-energy gates
    precisely so a pad's filter noise does not become a drum machine, and it
    reconstructs the kick's spectrum by subtracting a between-hits spectrum
    from an at-hits one — a good estimate of one object inside a waveform that
    contains two.

    With stems there is no reconstruction: the kick is isolated inside the drum
    stem and the bass is a separate array, so the two fundamentals and the
    level between them are direct readings. Same finding, real numbers, and the
    confidence moves from 0.70 to 0.93 to say so.
    """
    if ctx.no_programme:
        return []

    le = ctx.m.low_end
    hits: List[_Hit] = []

    # -- kick and bass on the same fundamental -------------------------------
    collision = _fin(le.kick_bass_collision_db, -40.0)
    collision_window = (-40.0, _fin(ctx.profile.kick_bass_collision_max_db, -6.0))
    collision_miss = targets.range_miss(collision, collision_window)

    sk = ctx.stems
    stem_kick_hz = _fin(sk.kick_fundamental_hz, 0.0) if sk.kick_fundamental_hz else 0.0
    stem_bass_hz = _fin(sk.bass_fundamental_hz, 0.0) if sk.bass_fundamental_hz else 0.0
    kick_to_bass = None if sk.kick_to_bass_db is None else _fin(sk.kick_to_bass_db, 0.0)
    from_stems = bool(
        ctx.has_stems and ctx.has("stem") and stem_kick_hz > 0.0 and stem_bass_hz > 0.0
    )

    kick_hz = stem_kick_hz if from_stems else _fin(le.kick_fundamental_hz)
    bass_hz = stem_bass_hz if from_stems else _fin(le.bass_fundamental_hz)
    semitones = (
        abs(12.0 * math.log2(max(kick_hz, 1e-6) / max(bass_hz, 1e-6)))
        if kick_hz > 0 and bass_hz > 0 else 0.0
    )

    # The pitch distance is only a *trigger* when it was measured on two
    # separated objects. On a two-track both figures come out of the same
    # waveform, and firing on them would be firing on the reconstruction's
    # own error.
    note_window = (_SAME_NOTE_SEMITONES, 120.0)
    note_miss = targets.range_miss(semitones, note_window) if from_stems else 0.0
    gate = (bool(le.kick_detected) and ctx.has("kick")) or from_stems

    if gate and (collision_miss > 0.0 or note_miss < 0.0):
        ratio = max(_ratio(collision_miss, 2.0), _ratio(note_miss, _SEMITONE_SCALE))
        moments = _moments(le.collision_moments)
        duck = _fin(le.ducking_depth_db, 0.0)
        # The overlap figure compares spectral *shapes* over 25-250 Hz, which can
        # be high even when the two fundamentals are far apart. Only claim a
        # shared fundamental when the pitches actually say so.
        same_note = 0.0 < semitones <= 4.0 and kick_hz > 0 and bass_hz > 0
        consequence = (
            "Every kick re-triggers the note the bass is already holding, so the low end "
            "reads as one thick sound instead of a hit and a pitch."
            if same_note else
            "The two fundamentals are far enough apart, but the energy either side of them "
            "is spread across the same shape, so the kick lands inside the bass's body "
            "rather than underneath it."
        )

        if from_stems:
            level = (
                f"the kick sits {_num(abs(kick_to_bass), 1)} dB "
                f"{'over' if kick_to_bass >= 0 else 'under'} the bass over its own hit windows"
                if kick_to_bass is not None else
                "their levels could not be compared"
            )
            detail = (
                f"Separated, the kick inside the drum stem lands on {_num(kick_hz, 1)} Hz and "
                f"the bass stem's fundamental is {_num(bass_hz, 1)} Hz — {_num(semitones, 1)} "
                f"semitones apart, against the {_num(_SAME_NOTE_SEMITONES, 0)} semitones two "
                f"low-frequency sources need before the ear reads them as separate pitches — "
                f"and {level}. Their 25-250 Hz distributions overlap at {_num(collision, 1)} dB "
                f"against a {_num(collision_window[1], 1)} dB ceiling for {ctx.profile.label}. "
                f"These are two measured objects, not one waveform split by an onset gate: the "
                f"kick and the bass were separated before either was measured. Across "
                f"{le.kick_count} detected hits the sub moves {_num(duck, 1)} dB when the kick "
                f"lands"
                + (", so there is no sidechain getting the bass out of the way"
                   if not bool(le.has_sidechain) else ", which is a sidechain working")
                + f". {consequence}"
                + _moment_span(moments)
            )
        else:
            detail = (
                f"The kick's fundamental is {_num(kick_hz, 1)} Hz and the bass sits at "
                f"{_num(bass_hz, 1)} Hz — {_num(semitones, 1)} semitones apart — and their "
                f"25-250 Hz energy distributions overlap at {_num(collision, 1)} dB against a "
                f"{_num(collision_window[1], 1)} dB ceiling for {ctx.profile.label}. Across "
                f"{le.kick_count} detected hits the sub moves {_num(duck, 1)} dB when the kick "
                f"lands"
                + (", so there is no sidechain getting the bass out of the way"
                   if not bool(le.has_sidechain) else ", which is a sidechain working")
                + f". {consequence}"
                + _moment_span(moments)
            )

        evidence = [
            _ev("Kick/bass spectral overlap", collision, "dB",
                target_range=collision_window,
                verdict=_verdict(collision, collision_window, 2.0),
                detail="0 dB means identical shape in the same place."),
            _ev("Kick fundamental", kick_hz, "Hz", verdict="watch",
                detail=("Measured inside the separated drum stem."
                        if from_stems else
                        "At-onset minus between-onset spectrum of the two-track.")),
            _ev("Bass fundamental", bass_hz, "Hz", verdict="watch",
                detail=f"{_num(semitones, 1)} semitones from the kick"
                       + (", both measured on separated sources." if from_stems else ".")),
        ]
        if from_stems:
            evidence.append(
                _ev("Pitch distance", semitones, "semitones", target_range=note_window,
                    verdict=_verdict(semitones, note_window, _SEMITONE_SCALE),
                    detail=f"Under {_num(_SAME_NOTE_SEMITONES, 0)} semitones the difference "
                           f"tone lands in the same critical band as both sources.")
            )
            if kick_to_bass is not None:
                evidence.append(
                    _ev("Kick vs bass level", kick_to_bass, "dB", target=None,
                        verdict="watch",
                        detail="Drum stem's low band over its hit windows against the "
                               "bass stem — two sources, directly compared.")
                )
        evidence.extend([
            _ev("Sidechain ducking depth", duck, "dB", target=None,
                verdict="problem" if duck < 1.5 else "good",
                detail="Sub level drop at each kick hit. Inferred, not read "
                       "off a plugin."),
            _ev("Kick definition", _fin(le.kick_definition_db), "dB",
                target=None, verdict="good",
                detail="Kick transient above the surrounding low-end floor."),
            _ev("Detected kick hits", le.kick_count, "hits", verdict="good"),
        ])

        hits.append(_Hit(
            Finding(
                id="low_end.kick_bass_collision",
                dimension="low_end",
                title=("Kick and bass share a fundamental" if same_note
                       else "Kick and bass occupy the same low-end range"),
                severity=_severity(ratio),
                # With stems these are two separated objects compared directly.
                # Without them, the kick's spectrum is at-onset minus
                # between-onset — a good estimate, and not a stem.
                confidence=(ctx.trust(0.93, "stem") if from_stems
                            else ctx.trust(0.70, "kick")),
                detail=detail,
                evidence=evidence,
                band_hz=(
                    round(min(kick_hz, bass_hz) * 0.8, 1),
                    round(max(kick_hz, bass_hz) * 1.25, 1),
                ) if kick_hz > 0 and bass_hz > 0 else None,
                moments=moments,
            ),
            ratio,
        ))
        ctx.tags.add("low_end_collision")

    # -- sub weight ----------------------------------------------------------
    sub = _fin(le.sub_energy_db, -90.0)
    sub_window = ctx.profile.sub_energy_db
    sub_miss = targets.range_miss(sub, sub_window)
    if sub_miss != 0.0 and sub > -60.0:
        ratio = _ratio(sub_miss, 3.0)
        hot = sub_miss > 0.0
        band = ctx.bands.get("sub")
        detail = (
            f"Energy below 60 Hz is {_num(sub, 1)} dB relative to the whole band, "
            f"{_num(abs(sub_miss), 1)} dB {'above' if hot else 'below'} the "
            f"{_win(sub_window, 1, ' dB')} window {ctx.profile.label} works in"
            + (f", and the sub macro band measures {_num(band.deviation_db, 1)} dB "
               f"{'over' if band.deviation_db > 0 else 'under'} its target curve"
               if band else "")
            + (". That much weight under 60 Hz eats the limiter's headroom and will not "
               "reproduce on anything smaller than a full-range system."
               if hot else
               ". The bottom octave is not carrying its share, so the mix will feel small "
               "on a system that can actually play it.")
        )
        hits.append(_Hit(
            Finding(
                id="low_end.sub_energy_hot" if hot else "low_end.sub_energy_thin",
                dimension="low_end",
                title="Too much energy below 60 Hz" if hot else "Sub octave is underweight",
                severity=_severity(ratio),
                confidence=0.90,   # a band energy ratio, directly measured
                detail=detail,
                evidence=[
                    _ev("Sub energy (20-60 Hz)", sub, "dB", target_range=sub_window,
                        verdict=_verdict(sub, sub_window, 3.0)),
                    _ev("Sub band vs target curve",
                        band.deviation_db if band else 0.0, "dB",
                        target=0.0,
                        target_range=(-(band.tolerance_db if band else 4.5),
                                      (band.tolerance_db if band else 4.5)),
                        verdict="problem" if band and band.miss_db else "good"),
                    _ev("Sub-25 Hz rumble", _fin(le.sub_rumble_db), "dB",
                        target_range=(-90.0, _rumble_ceiling_db(ctx.genre_key)),
                        verdict=_verdict(_fin(le.sub_rumble_db),
                                         (-90.0, _rumble_ceiling_db(ctx.genre_key)),
                                         4.0)),
                ],
                band_hz=(20.0, 60.0),
            ),
            ratio,
        ))
        ctx.tags.add("sub_energy")

    # -- the bottom is not mono ----------------------------------------------
    # Suppressed under a polarity inversion: it is the same fault, and the phase
    # finding already carries the low-end mono ratio as evidence.
    mono_ratio = _fin(le.low_end_mono_ratio, 1.0)
    mono_window = (_fin(ctx.profile.low_end_mono_min, 0.80), 1.0)
    mono_miss = targets.range_miss(mono_ratio, mono_window)
    if mono_miss < 0.0 and not ctx.is_mono and "polarity" not in ctx.tags:
        ratio = _ratio(mono_miss, 0.06)
        side_db = _fin(ctx.m.stereo.low_end_side_energy_db, -60.0)
        detail = (
            f"Only {mono_ratio * 100:.0f}% of the energy below 120 Hz is mono, against the "
            f"{mono_window[0] * 100:.0f}% {ctx.profile.label} needs, and side energy down there "
            f"measures {_num(side_db, 1)} dB relative to mid. Stereo bass cancels unpredictably "
            f"on fold-down, wanders on a club rig where much of the low end is summed, and is "
            f"the single most common reason a cut gets rejected for vinyl."
        )
        hits.append(_Hit(
            Finding(
                id="low_end.not_mono",
                dimension="low_end",
                title="Low end is not mono",
                severity=_severity(ratio),
                confidence=0.92,
                detail=detail,
                evidence=[
                    _ev("Low-end mono ratio", mono_ratio, "", target_range=mono_window,
                        verdict=_verdict(mono_ratio, mono_window, 0.06)),
                    _ev("Side energy below 120 Hz", side_db, "dB", target=-12.0,
                        verdict="problem" if side_db > -12.0 else "good"),
                    _ev("Sub band width",
                        _fin((ctx.m.stereo.band_width or {}).get("sub", 0.0)), "",
                        target=0.0, verdict="watch"),
                ],
                band_hz=(20.0, 120.0),
            ),
            ratio,
        ))

    # -- inaudible energy under 25 Hz ---------------------------------------
    rumble = _fin(le.sub_rumble_db, -90.0)
    rumble_ceiling = _rumble_ceiling_db(ctx.genre_key)
    rumble_window = (-90.0, rumble_ceiling)
    rumble_miss = targets.range_miss(rumble, rumble_window)
    if rumble_miss > 0.0:
        ratio = _ratio(rumble_miss, 4.0)
        detail = (
            f"{_num(rumble, 1)} dB of the total energy sits below 25 Hz, over the "
            f"{_num(rumble_ceiling, 1)} dB point where the figure stops being spectral "
            f"leakage from the bass fundamental and starts being real content — a ceiling "
            f"set from {ctx.profile.label}'s own target curve, which is why it is not the "
            f"same number for every genre. None of it is "
            f"audible on any playback system, and all of it is costing limiter headroom — a "
            f"24 dB/octave high-pass at 25 Hz gives that headroom back for free."
        )
        hits.append(_Hit(
            Finding(
                id="low_end.subsonic_rumble",
                dimension="low_end",
                title="Inaudible energy below 25 Hz",
                severity=_severity(ratio),
                confidence=0.80,
                detail=detail,
                evidence=[
                    _ev("Sub-25 Hz energy", rumble, "dB", target_range=rumble_window,
                        verdict="problem"),
                    _ev("Sub energy (20-60 Hz)", sub, "dB", target_range=sub_window,
                        verdict=_verdict(sub, sub_window, 3.0)),
                ],
                band_hz=(3.0, 25.0),
            ),
            ratio,
        ))

    return hits


# ---------------------------------------------------------------------------
# 9. Mud & low-mid buildup
# ---------------------------------------------------------------------------


def _detect_mud(ctx: _Ctx) -> List[_Hit]:
    """150-400 Hz against the low bass, against the mids, and 300-600 Hz boxiness.

    `mud_ratio_db` is the primary number because `targets.py` windows it per
    genre. The other two have their targets derived from the genre's own curve
    (see `_curve_density_db`), and because they are derived rather than
    hand-set, the detector only fires on them when *both* agree — one derived
    metric on its own is not enough to call a mix muddy.
    """
    if ctx.no_programme:
        return []

    s = ctx.m.spectral

    mud_ratio = _fin(s.mud_ratio_db, 0.0)
    mud_window = ctx.profile.mud_ratio_db
    mud_miss = targets.range_miss(mud_ratio, mud_window)

    tol_low_mid = _fin(targets.band_tolerance(ctx.genre_key, "low_mid"), 2.2)
    tol_mid = _fin(targets.band_tolerance(ctx.genre_key, "mid"), 2.0)
    tol_upper_mid = _fin(targets.band_tolerance(ctx.genre_key, "upper_mid"), 2.0)

    # A ratio between two bands can be wrong by either band's tolerance, so the
    # tolerance on the ratio is the sum of the two.
    m2m = _fin(s.mud_to_mid_db, 0.0)
    m2m_target = _curve_density_db(ctx.curve, (150.0, 400.0), (1000.0, 3000.0))
    m2m_tol = tol_low_mid + tol_upper_mid
    m2m_window = (m2m_target - m2m_tol, m2m_target + m2m_tol)
    m2m_miss = targets.range_miss(m2m, m2m_window)

    boxy = _fin(s.boxiness_db, 0.0)
    curve_median = float(np.median(ctx.curve[(_CENTERS >= 40.0) & (_CENTERS <= 16_000.0)]))
    boxy_target = _curve_region_db(ctx.curve, 300.0, 600.0) - curve_median
    boxy_tol = tol_low_mid + tol_mid
    boxy_window = (boxy_target - boxy_tol, boxy_target + boxy_tol)
    boxy_miss = targets.range_miss(boxy, boxy_window)

    primary = mud_miss > 0.0
    corroborated = m2m_miss > 0.0 and boxy_miss > 0.0
    if not (primary or corroborated):
        return []

    ratio = min(
        max(_ratio(mud_miss, 2.5), _ratio(m2m_miss, m2m_tol), _ratio(boxy_miss, boxy_tol)),
        6.0,
    )

    # A masking pair in the low mids is this same energy described from the
    # other side — the spectrum says the region is heavy, the stems say which
    # source is putting it there and which source it is burying. One problem,
    # one finding: mud claims those pairs so the clarity detector does not
    # report them again, and the pairs make this finding *specific* rather than
    # making it a second finding.
    mask_pairs = ctx.take_masking(_MASK_MUD_BANDS, limit=2)

    resonances = [r for r in (s.resonances or []) if 150.0 <= _fin(r.freq_hz) <= 500.0]
    resonances.sort(key=lambda r: -_fin(r.prominence_db))
    moments: List[Moment] = []
    for r in resonances[:2]:
        moments.extend(r.moments or [])
    moments = _moments(moments)

    res_text = ""
    if resonances:
        r = resonances[0]
        res_text = (
            f" The narrowest offender is a Q {_num(r.q, 1)} peak at {_num(r.freq_hz, 0)} Hz "
            f"standing {_num(r.prominence_db, 1)} dB above the octave around it"
        )
        if len(resonances) > 1:
            res_text += f", with a second at {_num(resonances[1].freq_hz, 0)} Hz"
        res_text += "."

    masking_text = ""
    if ctx.m.vocal.vocal_present and ctx.m.vocal.masked_bands:
        masking_text = (
            f" The centre vocal is being crowded in "
            f"{', '.join(b.replace('_', ' ') for b in ctx.m.vocal.masked_bands)}."
        )

    detail = (
        f"150-400 Hz is carrying {_num(mud_ratio, 1)} dB relative to 60-120 Hz against a "
        f"{_win(mud_window, 1, ' dB')} window for {ctx.profile.label}, and it sits "
        f"{_num(m2m, 1)} dB over 1-3 kHz where this genre's target curve puts it at "
        f"{_num(m2m_target, 1)} dB. 300-600 Hz reads {_num(boxy, 1)} dB against the mix's own "
        f"broadband median, {_num(boxy - boxy_target, 1)} dB more than the curve wants."
        f"{res_text} Everything above the bass is being covered by the region directly under "
        f"it — that is the blanket."
        f"{masking_text}"
        f"{_mask_sentence(mask_pairs)}"
        + _moment_span(moments)
    )

    ctx.tags.add("mud")
    return [_Hit(
        Finding(
            id="mud.low_mid_buildup",
            dimension="mud",
            title="Low-mid buildup covering the mix",
            severity=_severity(ratio),
            # A long-term spectrum, directly measured. With a separated source
            # named as the masker it stops being "this region is heavy" and
            # becomes "this source is heavy here", which is a stronger claim
            # about the same numbers.
            confidence=0.94 if mask_pairs else 0.88,
            detail=detail,
            evidence=_mask_evidence(mask_pairs) + [
                _ev("150-400 Hz vs 60-120 Hz", mud_ratio, "dB", target_range=mud_window,
                    verdict=_verdict(mud_ratio, mud_window, 2.5),
                    detail=f"Windowed per genre in targets.py for {ctx.profile.label}."),
                _ev("150-400 Hz vs 1-3 kHz", m2m, "dB", target=m2m_target,
                    target_range=m2m_window, verdict=_verdict(m2m, m2m_window, m2m_tol),
                    detail="Target derived from this genre's own curve, density-corrected."),
                _ev("Boxiness (300-600 Hz)", boxy, "dB", target=boxy_target,
                    target_range=boxy_window, verdict=_verdict(boxy, boxy_window, boxy_tol)),
                _ev("Low-mid band vs target",
                    ctx.bands["low_mid"].deviation_db if "low_mid" in ctx.bands else 0.0,
                    "dB", target=0.0,
                    target_range=(-tol_low_mid, tol_low_mid),
                    verdict="problem" if ctx.bands.get("low_mid")
                    and ctx.bands["low_mid"].miss_db else "good"),
                _ev("Worst 150-500 Hz resonance",
                    _fin(resonances[0].prominence_db) if resonances else 0.0, "dB",
                    target_range=(0.0, 3.0),
                    verdict="problem" if resonances else "good",
                    detail=(f"{_num(resonances[0].freq_hz, 0)} Hz, Q {_num(resonances[0].q, 1)}"
                            if resonances else "No narrow peak found in this region.")),
            ],
            band_hz=(150.0, 400.0),
            moments=moments,
        ),
        ratio,
    )]


# ---------------------------------------------------------------------------
# 10. Harshness & sibilance
# ---------------------------------------------------------------------------


def _detect_harshness(ctx: _Ctx) -> List[_Hit]:
    """2-5 kHz edge and 5-9 kHz sibilance, against the genre's ceilings.

    Both indices already answer "does this region stick out of the curve its own
    neighbours draw" rather than "is this region loud", so a legitimately bright
    mix scores near zero on them. The genre ceilings come straight from
    `targets.py`: rock tolerates 0.46, R&B 0.34, lo-fi 0.28.
    """
    if ctx.no_programme:
        return []

    s = ctx.m.spectral
    hits: List[_Hit] = []

    def _res_in(lo: float, hi: float):
        found = [r for r in (s.resonances or []) if lo <= _fin(r.freq_hz) <= hi]
        found.sort(key=lambda r: -_fin(r.prominence_db))
        return found

    # -- 2-5 kHz -------------------------------------------------------------
    harsh = _fin(s.harshness_index, 0.0)
    harsh_window = (0.0, _fin(ctx.profile.harshness_max, 0.42))
    harsh_miss = targets.range_miss(harsh, harsh_window)
    sharp = _fin(s.sharpness_acum, 0.0)
    sharp_window = (0.0, _fin(ctx.profile.sharpness_max_acum, 2.6))
    sharp_miss = targets.range_miss(sharp, sharp_window)

    if harsh_miss > 0.0 or sharp_miss > 0.0:
        ratio = min(max(_ratio(harsh_miss, 0.15), _ratio(sharp_miss, 0.5)), 6.0)
        res = _res_in(1800.0, 5200.0)
        moments = _moments([mo for r in res[:2] for mo in (r.moments or [])])
        band = ctx.bands.get("presence")
        res_text = ""
        if res:
            r = res[0]
            res_text = (
                f" The peak driving it is a Q {_num(r.q, 1)} resonance at {_num(r.freq_hz, 0)} Hz "
                f"standing {_num(r.prominence_db, 1)} dB above the octave around it"
                + (f", with another at {_num(res[1].freq_hz, 0)} Hz." if len(res) > 1 else ".")
            )
        detail = (
            f"2-5 kHz scores {harsh:.3f} on the harshness index against a "
            f"{_num(harsh_window[1], 2)} ceiling for {ctx.profile.label}, and psychoacoustic "
            f"sharpness reads {_num(sharp, 2)} acum against {_num(sharp_window[1], 1)}. "
            f"This is not brightness — the index measures how far the region stands above the "
            f"line its own neighbours draw, so a mix with an ordinary downward tilt scores near "
            f"zero.{res_text}"
            + (f" The presence band sits {_num(band.deviation_db, 1)} dB "
               f"{'over' if band.deviation_db > 0 else 'under'} the {ctx.profile.label} curve."
               if band else "")
            + " This is the region that becomes fatiguing at volume and unlistenable on "
              "earbuds."
            + _moment_span(moments)
        )
        hits.append(_Hit(
            Finding(
                id="harshness.upper_mid_edge",
                dimension="harshness",
                title="Harsh 2-5 kHz edge",
                severity=_severity(ratio),
                # A modelled 0-1 index over a directly measured spectrum.
                confidence=0.78,
                detail=detail,
                evidence=[
                    _ev("Harshness index", harsh, "", target_range=harsh_window,
                        verdict=_verdict(harsh, harsh_window, 0.15),
                        detail=f"{ctx.profile.label} ceiling from targets.py."),
                    _ev("Sharpness", sharp, "acum", target_range=sharp_window,
                        verdict=_verdict(sharp, sharp_window, 0.5),
                        detail="Zwicker-style, level-independent."),
                    _ev("Presence band vs target",
                        band.deviation_db if band else 0.0, "dB", target=0.0,
                        target_range=(-(band.tolerance_db if band else 2.2),
                                      (band.tolerance_db if band else 2.2)),
                        verdict="problem" if band and band.miss_db else "good"),
                    _ev("Worst 2-5 kHz resonance",
                        _fin(res[0].prominence_db) if res else 0.0, "dB",
                        target_range=(0.0, 3.0), verdict="problem" if res else "good",
                        detail=(f"{_num(res[0].freq_hz, 0)} Hz, Q {_num(res[0].q, 1)}"
                                if res else "No narrow peak in this region.")),
                    _ev("Spectral centroid", _fin(s.spectral_centroid_hz), "Hz",
                        verdict="good"),
                ],
                band_hz=(2000.0, 5000.0),
                moments=moments,
            ),
            ratio,
        ))
        ctx.tags.add("harsh")

    # -- 5-9 kHz -------------------------------------------------------------
    sib = _fin(s.sibilance_index, 0.0)
    sib_window = (0.0, _fin(ctx.profile.sibilance_max, 0.40))
    sib_miss = targets.range_miss(sib, sib_window)
    if sib_miss > 0.0:
        ratio = min(_ratio(sib_miss, 0.15), 6.0)
        res = _res_in(5000.0, 9000.0)
        moments = _moments([mo for r in res[:2] for mo in (r.moments or [])])
        voc_sib = _fin(ctx.m.vocal.sibilance_db, -60.0)
        detail = (
            f"5-9 kHz scores {sib:.3f} on the sibilance index against a "
            f"{_num(sib_window[1], 2)} ceiling for {ctx.profile.label}. The index separates "
            f"sibilance from air by the frame-to-frame burstiness of the band, so a steady "
            f"shimmer does not register and 'ess' sounds do"
            + (f"; in the centre channel the 95th-percentile 5-9 kHz level sits "
               f"{_num(voc_sib, 1)} dB against the median vocal-band level."
               if ctx.m.vocal.vocal_present else ".")
            + (f" The narrowest peak is {_num(res[0].freq_hz, 0)} Hz at "
               f"{_num(res[0].prominence_db, 1)} dB prominence." if res else "")
            + _moment_span(moments)
        )
        hits.append(_Hit(
            Finding(
                id="harshness.sibilance",
                dimension="harshness",
                title="Sibilance in the 5-9 kHz band",
                severity=_severity(ratio),
                confidence=0.72,
                detail=detail,
                evidence=[
                    _ev("Sibilance index", sib, "", target_range=sib_window,
                        verdict=_verdict(sib, sib_window, 0.15)),
                    _ev("Centre 5-9 kHz vs vocal band", voc_sib, "dB", target=-12.0,
                        verdict="watch" if voc_sib > -12.0 else "good",
                        detail="Derived from centre extraction — an inference, not a stem."),
                    _ev("Brilliance band vs target",
                        ctx.bands["brilliance"].deviation_db
                        if "brilliance" in ctx.bands else 0.0, "dB", target=0.0,
                        verdict="watch"),
                ],
                band_hz=(5000.0, 9000.0),
                moments=moments,
            ),
            ratio,
        ))
        ctx.tags.add("sibilance")

    return hits


# ---------------------------------------------------------------------------
# 11. Frequency balance
# ---------------------------------------------------------------------------

# Which macro bands a more specific detector has already spoken for. The
# frequency-balance detector stands down on those rather than describing the
# same energy a second time (rule 5).
_COVERED_BY: Dict[str, Tuple[str, ...]] = {
    "mud": ("upper_bass", "low_mid"),
    "harsh": ("presence",),
    "sibilance": ("brilliance",),
    "sub_energy": ("sub", "low_bass"),
}


def _detect_frequency_balance(ctx: _Ctx) -> List[_Hit]:
    """Macro bands against the genre curve — at most one hot and one thin.

    Nine bands times two directions is a wall, and a wall is not a report. Only
    the worst deviation in each direction is emitted, measured in units of that
    band's own tolerance so a 3 dB miss in the mids outranks a 5 dB miss in the
    air band (which `targets.py` allows twice as much room).
    """
    if ctx.no_programme or not ctx.bands:
        return []

    covered: set = set()
    for tag, names in _COVERED_BY.items():
        if tag in ctx.tags:
            covered.update(names)

    candidates = [b for b in ctx.bands.values() if b.miss_db != 0.0 and b.name not in covered]
    if not candidates:
        return []

    measured_tilt = _fin(ctx.m.spectral.spectral_tilt_db_per_decade, 0.0)
    target_tilt = _curve_tilt(ctx.curve, ctx.m.spectral.third_octave_db)
    tilt_window = (target_tilt - 4.0, target_tilt + 4.0)
    tilt_miss = targets.range_miss(measured_tilt, tilt_window)

    hits: List[_Hit] = []
    for direction, sign in (("hot", 1.0), ("thin", -1.0)):
        pool = [b for b in candidates if b.miss_db * sign > 0.0]
        if not pool:
            continue
        band = max(pool, key=lambda b: b.ratio)
        ratio = min(band.ratio, 6.0)

        neighbours = [
            ctx.bands[n] for n in ctx.bands
            if n != band.name and abs(ctx.bands[n].center_hz / max(band.center_hz, 1e-9) - 1.0) < 3.0
        ]
        neighbours.sort(key=lambda b: abs(math.log2(max(b.center_hz, 1e-9) /
                                                    max(band.center_hz, 1e-9))))
        near = neighbours[0] if neighbours else None

        detail = (
            f"The {band.label} band ({band.span}) measures {_num(band.level_db, 1)} dB against "
            f"a {_num(band.target_db, 1)} dB target for {ctx.profile.label} — "
            f"{_num(abs(band.deviation_db), 1)} dB "
            f"{'over' if sign > 0 else 'under'}, which is {_num(abs(band.miss_db), 1)} dB "
            f"outside the {_pm(band.tolerance_db)} dB this genre allows there"
            + (f", while {near.label} next to it sits {_num(near.deviation_db, 1)} dB "
               f"from its own target" if near else "")
            + f". Overall tilt is {_num(measured_tilt, 1)} dB/decade against "
            f"{_num(target_tilt, 1)} dB/decade for the {ctx.profile.label} curve"
            + (f", so the whole balance is {'darker' if tilt_miss < 0 else 'brighter'} than the "
               f"genre and not just this one band." if tilt_miss != 0.0
               else ", so the overall slope is right and this is a single-band problem.")
        )

        hits.append(_Hit(
            Finding(
                id=f"frequency_balance.{band.name}_{direction}",
                dimension="frequency_balance",
                title=f"{band.label[:1].upper()}{band.label[1:]} "
                      f"{'too hot' if sign > 0 else 'underweight'} for {ctx.profile.label}",
                severity=_severity(ratio),
                confidence=0.88,
                detail=detail,
                evidence=[
                    _ev(f"{band.label.title()} level", band.level_db, "dB",
                        target=band.target_db,
                        target_range=(band.target_db - band.tolerance_db,
                                      band.target_db + band.tolerance_db),
                        verdict="problem" if band.ratio >= _MAJOR_AT else "watch",
                        detail=f"{band.span}, relative to the mix's own 1 kHz octave."),
                    _ev("Deviation from target", band.deviation_db, "dB", target=0.0,
                        target_range=(-band.tolerance_db, band.tolerance_db),
                        verdict="problem"),
                    _ev("Spectral tilt", measured_tilt, "dB/decade", target=target_tilt,
                        target_range=tilt_window,
                        verdict=_verdict(measured_tilt, tilt_window, 4.0),
                        detail="Target tilt fitted from this genre's own curve over the "
                               "same bands the measurement used."),
                ] + ([
                    _ev(f"{near.label.title()} level", near.level_db, "dB",
                        target=near.target_db,
                        target_range=(near.target_db - near.tolerance_db,
                                      near.target_db + near.tolerance_db),
                        verdict="problem" if near.ratio >= _MAJOR_AT else "good")
                ] if near else []),
                band_hz=(band.low_hz, band.high_hz),
            ),
            ratio,
        ))

    return hits



# ---------------------------------------------------------------------------
# 12. Vocal balance
# ---------------------------------------------------------------------------


def _detect_vocal(ctx: _Ctx) -> List[_Hit]:
    """Lead vocal level, consistency and intelligibility.

    Two sources, and this detector prefers the better one.

    *Without stems* everything here comes from a centre estimate, and the DSP
    layer's own `vocal_present` test (syllabic modulation specific to the
    centre) is the gate: if it says there is no voice, this reports nothing
    rather than describing a centred synth pad as a badly balanced singer. A
    centre estimate cannot tell a lead vocal from a centred synth, a snare or a
    mono bass, so confidences top out around 0.55 and the sentences say the
    figure describes "everything centred".

    *With stems* the same two questions are answered by measurement.
    `vocal_to_instrument_db` is the vocal stem's gated loudness against the
    other three summed, and consistency is the spread of the vocal's own level
    over the frames it is actually singing on. Neither moves when a synth moves.
    Confidence goes to 0.90/0.84, the sentence cites the stem ratio, and the
    detector stops needing a stereo field at all — a mono file has no centre
    channel and still has a vocal stem.
    """
    v = ctx.m.vocal
    if ctx.no_programme:
        return []

    stem = ctx.stem("vocals")
    from_stems = bool(
        stem is not None
        and ctx.has_stems
        and ctx.has("stem")
        and ctx.stems.vocal_to_instrument_db is not None
    )
    if not from_stems:
        if ctx.is_mono or not ctx.has("vocal") or not bool(v.vocal_present):
            return []

    hits: List[_Hit] = []

    # The centre measurement stays usable as *supporting* evidence whenever it
    # was meaningful — it is what carries the timeline moments and the
    # intelligibility index, neither of which the stem pass produces.
    centre_valid = bool(v.vocal_present) and not ctx.is_mono

    v2i_window = ctx.profile.vocal_to_instrument_db
    if from_stems:
        v2i = _fin(ctx.stems.vocal_to_instrument_db, 0.0)
        spread = _stem_level_spread(stem) if stem is not None else None
        consistency = spread if spread is not None else _fin(v.consistency_db, 0.0)
        consistency_source = "stem" if spread is not None else "centre"
    else:
        v2i = _fin(v.vocal_to_instrument_db, 0.0)
        consistency = _fin(v.consistency_db, 0.0)
        consistency_source = "centre"
    v2i_miss = targets.range_miss(v2i, v2i_window)

    # How uneven a vocal may be follows the genre's own tolerance for level
    # movement: a folk record with 14 LU of range earns a wider vocal spread
    # than a trap record with 6.
    consistency_max = 6.0 + 0.4 * _fin(ctx.profile.loudness_range_lu[1], 8.0)
    consistency_window = (0.0, consistency_max)
    consistency_miss = targets.range_miss(consistency, consistency_window)

    intel = _fin(v.intelligibility_index, 0.5)
    intel_window = (max(_fin(ctx.profile.clarity_min, 0.45) - 0.05, 0.1), 1.0)
    intel_miss = targets.range_miss(intel, intel_window) if centre_valid else 0.0

    # Only cited when the centre measurement was meaningful: a masked-band list
    # from a file with no detected centre voice describes whatever else is
    # centred, not the singer.
    masked = list(v.masked_bands or []) if centre_valid else []
    # Sources the separator says are burying the vocal. Claimed here so the
    # clarity detector does not report the same burial a second time — "the
    # guitars are on top of the vocal" is a vocal-balance problem, and that is
    # where a producer will look for it.
    vocal_masking = ctx.take_masking(maskee="vocals", limit=2) if from_stems else []

    if from_stems:
        v2i_evidence = _ev(
            "Vocal vs instruments", v2i, "dB", target_range=v2i_window,
            verdict=_verdict(v2i, v2i_window, 1.5),
            detail="Gated loudness of the separated vocal against drums, bass and "
                   "everything else summed. A measurement, not a centre estimate.",
        )
        second = _ev(
            "Vocal stem level", _fin(stem.level_ratio_db if stem else 0.0), "dB",
            target=None, verdict="good",
            detail=f"The vocal stem against the full mix, and it sounds on "
                   f"{_fin(stem.active_ratio if stem else 0.0) * 100:.0f}% of the track.",
        )
        spread_detail = (
            f"90th minus 10th percentile of the vocal stem's own level, over the frames "
            f"it is singing on. Allowance scales with {ctx.profile.label}'s "
            f"{_win(ctx.profile.loudness_range_lu, 1, ' LU')} loudness range."
            if consistency_source == "stem" else
            f"90th minus 10th percentile over time. Allowance scales with "
            f"{ctx.profile.label}'s {_win(ctx.profile.loudness_range_lu, 1, ' LU')} "
            f"loudness range."
        )
    else:
        v2i_evidence = _ev(
            "Vocal vs instruments", v2i, "dB", target_range=v2i_window,
            verdict=_verdict(v2i, v2i_window, 1.5),
            detail="A-weighted centre 300 Hz-6 kHz against everything else. Centre "
                   "extraction, not a stem.",
        )
        second = _ev("Centre energy ratio", _fin(v.center_energy_ratio), "",
                     target_range=(0.12, 1.0), verdict="good")
        spread_detail = (
            f"90th minus 10th percentile over time. Allowance scales with "
            f"{ctx.profile.label}'s {_win(ctx.profile.loudness_range_lu, 1, ' LU')} "
            f"loudness range."
        )

    evidence = [
        v2i_evidence,
        second,
        _ev("Vocal level spread", consistency, "dB", target_range=consistency_window,
            verdict=_verdict(consistency, consistency_window, 2.0),
            detail=spread_detail),
        _ev("Intelligibility index", intel, "", target_range=intel_window,
            verdict=_verdict(intel, intel_window, 0.12) if centre_valid else "good",
            detail="" if centre_valid else
                   "Not scored: it is derived from centre extraction, which needs a "
                   "stereo field and a detected centre voice."),
        _ev("Presence balance", _fin(v.presence_balance_db), "dB", verdict="good",
            detail="2-6 kHz against 300 Hz-1 kHz within the centre."),
    ] + _mask_evidence(vocal_masking)

    if v2i_miss != 0.0:
        ratio = _ratio(v2i_miss, 1.5)
        buried = v2i_miss < 0.0
        moments = _moments(v.buried_moments if buried else v.loud_moments) if centre_valid else []
        if not moments and vocal_masking:
            moments = _moments([mo for p in vocal_masking for mo in (p.moments or [])])
        if from_stems:
            detail = (
                f"The separated vocal sits {_num(v2i, 1)} dB against the drums, bass and "
                f"everything else summed, {_num(abs(v2i_miss), 1)} dB "
                f"{'below' if buried else 'above'} the {_win(v2i_window, 1, ' dB')} window "
                f"{ctx.profile.label} places a lead vocal in. That is the vocal's own gated "
                f"loudness against the actual instrumental — not the centre channel, so a "
                f"centred synth or a mono bass is on the other side of the ratio where it "
                f"belongs"
                + (f", and it sings on {_fin(stem.active_ratio if stem else 0.0) * 100:.0f}% "
                   f"of the track" if stem is not None else "")
                + "."
                + (f" Intelligibility scores {intel:.2f}." if centre_valid else "")
                + _mask_sentence(vocal_masking)
                + _moment_span(moments)
            )
        else:
            detail = (
                f"The centre 300 Hz-6 kHz sits {_num(v2i, 1)} dB against everything else "
                f"(A-weighted), {_num(abs(v2i_miss), 1)} dB "
                f"{'below' if buried else 'above'} the {_win(v2i_window, 1, ' dB')} window "
                f"{ctx.profile.label} places a lead vocal in"
                + (f", and {', '.join(b.replace('_', ' ') for b in masked)} carry non-centre "
                   f"content within 3 dB of the vocal there" if masked and buried else "")
                + f". Intelligibility scores {intel:.2f}. This is measured from a centre "
                f"estimate rather than a stem, so read it as the balance of everything "
                f"centred, not of the vocal alone."
                + _moment_span(moments)
            )
        hits.append(_Hit(
            Finding(
                id="vocal_balance.buried" if buried else "vocal_balance.too_loud",
                dimension="vocal_balance",
                title="Vocal sits under the mix" if buried else "Vocal sits over the mix",
                severity=_severity(ratio),
                # 0.90 against a separated source, 0.55 against a centre
                # estimate that cannot tell a singer from a centred synth.
                confidence=(ctx.trust(0.90, "stem") if from_stems
                            else ctx.trust(0.55, "vocal")),
                detail=detail,
                evidence=evidence,
                band_hz=(300.0, 6000.0),
                moments=moments,
            ),
            ratio,
        ))

    if consistency_miss > 0.0 or intel_miss < 0.0:
        ratio = max(_ratio(consistency_miss, 2.0), _ratio(intel_miss, 0.12))
        moments = (
            _moments(list(v.buried_moments or []) + list(v.loud_moments or []))
            if centre_valid else []
        )
        parts = []
        if consistency_miss > 0.0:
            source = (
                "the separated vocal's level swings" if consistency_source == "stem"
                else "the vocal level swings"
            )
            parts.append(
                f"{source} {_num(consistency, 1)} dB between its 10th and 90th "
                f"percentile, against the {_num(consistency_max, 1)} dB "
                f"{ctx.profile.label} absorbs"
            )
        if intel_miss < 0.0:
            parts.append(
                f"intelligibility scores {intel:.2f} against a {_num(intel_window[0], 2)} floor"
            )
        detail = (
            "The lead is not holding a steady place in the mix: "
            + "; ".join(parts)
            + (f". 1-4 kHz consonant energy is competing with non-centre content in "
               f"{', '.join(b.replace('_', ' ') for b in masked)}." if masked else ".")
            + (" Measured on the separated vocal, over the frames it is singing on."
               if consistency_source == "stem" else
               " Inferred from centre extraction, so treat the figure as directional.")
            + _moment_span(moments)
        )
        hits.append(_Hit(
            Finding(
                id="vocal_balance.inconsistent",
                dimension="vocal_balance",
                title="Vocal level and intelligibility wander",
                severity=_severity(ratio),
                confidence=(ctx.trust(0.84, "stem") if consistency_source == "stem"
                            else ctx.trust(0.50, "vocal")),
                detail=detail,
                evidence=evidence,
                band_hz=(300.0, 6000.0),
                moments=moments,
            ),
            ratio,
        ))

    return hits


# ---------------------------------------------------------------------------
# 13. Stereo width
# ---------------------------------------------------------------------------


def _detect_stereo_width(ctx: _Ctx) -> List[_Hit]:
    """Width against the genre's window, and L/R balance.

    A mono source is never reported as a stereo defect — a file that never had a
    stereo field cannot have lost one. Under a polarity inversion the width
    reading is a symptom of that fault, not an independent one, so it is
    suppressed and the dimension cross-references the phase finding instead.
    """
    if ctx.no_programme or ctx.is_mono:
        return []

    st = ctx.m.stereo
    hits: List[_Hit] = []

    width = _fin(st.width, 0.0)
    width_window = ctx.profile.stereo_width
    width_miss = targets.range_miss(width, width_window)

    balance = _fin(st.balance_db, 0.0)
    balance_window = (-1.5, 1.5)
    balance_miss = targets.range_miss(balance, balance_window)

    band_width = {k: _fin(vv) for k, vv in (st.band_width or {}).items()}
    widest = max(band_width, key=lambda k: band_width[k]) if band_width else ""
    narrowest = min(band_width, key=lambda k: band_width[k]) if band_width else ""

    evidence = [
        _ev("Stereo width", width, "", target_range=width_window,
            verdict=_verdict(width, width_window, 0.08),
            detail=f"Side/Mid RMS ratio. {ctx.profile.label} sits at {_win(width_window, 2)}."),
        _ev("Correlation", _fin(st.correlation), "",
            target_range=(_fin(ctx.profile.correlation_min), 1.0),
            verdict=_verdict(_fin(st.correlation),
                             (_fin(ctx.profile.correlation_min), 1.0), 0.35)),
        _ev("L/R balance", balance, "dB", target_range=balance_window,
            verdict=_verdict(balance, balance_window, 1.0),
            detail="Positive is right-heavy."),
        _ev(f"Widest band ({widest.replace('_', ' ') or 'n/a'})",
            band_width.get(widest, 0.0), "", target_range=width_window, verdict="good"),
        _ev("Mono sum loss", _fin(st.mono_sum_loss_db), "dB",
            target_range=(MONO_LOSS_FLOOR_DB, 60.0),
            verdict=_verdict(_fin(st.mono_sum_loss_db), (MONO_LOSS_FLOOR_DB, 60.0), 2.0)),
    ]

    if width_miss != 0.0 and "polarity" not in ctx.tags:
        ratio = _ratio(width_miss, 0.08)
        wide = width_miss > 0.0
        detail = (
            f"Side/Mid measures {_num(width, 2)} against the {_win(width_window, 2)} window "
            f"{ctx.profile.label} works in — {_num(abs(width_miss), 2)} "
            f"{'wider' if wide else 'narrower'} than the genre. The widest band is "
            f"{widest.replace('_', ' ') or 'n/a'} at {_num(band_width.get(widest, 0.0), 2)} and "
            f"the narrowest {narrowest.replace('_', ' ') or 'n/a'} at "
            f"{_num(band_width.get(narrowest, 0.0), 2)}, with correlation at "
            f"{_num(st.correlation, 2)} and a {_num(st.mono_sum_loss_db, 1)} dB mono fold-down "
            f"cost."
            + (" Width past the genre's window costs centre density and mono translation "
               "without buying any more space."
               if wide else
               " The mix is collapsing toward the centre, so nothing has anywhere to sit.")
        )
        hits.append(_Hit(
            Finding(
                id="stereo_width.too_wide" if wide else "stereo_width.too_narrow",
                dimension="stereo_width",
                title=f"{'Wider' if wide else 'Narrower'} than {ctx.profile.label} sits",
                severity=_severity(ratio),
                confidence=0.92,   # second-order statistics, directly computed
                detail=detail,
                evidence=evidence,
            ),
            ratio,
        ))

    if balance_miss != 0.0:
        ratio = _ratio(balance_miss, 1.0)
        detail = (
            f"The right channel runs {_num(balance, 2)} dB "
            f"{'hotter' if balance > 0 else 'quieter'} than the left across the whole file, "
            f"outside the {_pm(1.5)} dB a centred image tolerates. A constant offset "
            f"like this is a gain-staging or pan-law problem rather than an arrangement "
            f"choice — it pulls the phantom centre off axis for every listener."
        )
        hits.append(_Hit(
            Finding(
                id="stereo_width.channel_imbalance",
                dimension="stereo_width",
                title="Left and right are not level-matched",
                severity=_severity(ratio),
                confidence=0.94,
                detail=detail,
                evidence=evidence,
            ),
            ratio,
        ))

    return hits


# ---------------------------------------------------------------------------
# 14. Transient impact
# ---------------------------------------------------------------------------


def _detect_transients(ctx: _Ctx) -> List[_Hit]:
    """Punch against the genre floor, with attack time and smearing as support."""
    if ctx.no_programme or not ctx.has("transients"):
        return []

    t = ctx.m.transients
    if _fin(t.onset_density, 0.0) <= 0.0:
        # Nothing struck anything. A drone has no transients to lose.
        return []

    punch = _fin(t.punch_index, 0.0)
    punch_window = (_fin(ctx.profile.punch_min, 0.35), 1.0)
    punch_miss = targets.range_miss(punch, punch_window)
    if punch_miss >= 0.0:
        return []

    ratio = _ratio(punch_miss, 0.07)
    band_punch = {k: _fin(v) for k, v in (t.band_punch or {}).items()}
    weakest = min(band_punch, key=lambda k: band_punch[k]) if band_punch else ""
    moments = _moments(t.weak_moments)

    detail = (
        f"Punch scores {punch:.2f} against a {_num(punch_window[0], 2)} floor for "
        f"{ctx.profile.label}: across {int(_fin(t.onset_density) * ctx.duration)} detected "
        f"onsets the average hit is only {_num(t.transient_to_sustain_db, 1)} dB above its own "
        f"level 50 ms later, with a {_num(t.attack_time_ms, 1)} ms 10-90% attack and a "
        f"smearing index of {t.smearing_index:.2f}. The weakest band is "
        f"{weakest.replace('_', ' ') or 'n/a'} at {_num(band_punch.get(weakest, 0.0), 2)}. "
        f"The ear reads the drop after a hit, not the peak, so with this little fall-off the "
        f"drums arrive without landing."
        + _moment_span(moments)
    )

    return [_Hit(
        Finding(
            id="transients.no_punch",
            dimension="transients",
            title=f"Hits do not land for {ctx.profile.label}",
            severity=_severity(ratio),
            confidence=ctx.trust(0.80, "transients"),
            detail=detail,
            evidence=[
                _ev("Punch index", punch, "", target_range=punch_window,
                    verdict=_verdict(punch, punch_window, 0.07),
                    detail="Transient peak against the level 50 ms later."),
                _ev("Transient to sustain", _fin(t.transient_to_sustain_db), "dB",
                    target=None, verdict="problem"),
                _ev("Attack time", _fin(t.attack_time_ms), "ms", target=None,
                    verdict="watch" if _fin(t.attack_time_ms) > 15.0 else "good"),
                _ev("Smearing index", _fin(t.smearing_index), "",
                    target_range=(0.0, 0.5),
                    verdict=_verdict(_fin(t.smearing_index), (0.0, 0.5), 0.15)),
                _ev(f"Weakest band punch ({weakest.replace('_', ' ') or 'n/a'})",
                    band_punch.get(weakest, 0.0), "", target_range=punch_window,
                    verdict="problem"),
                _ev("Onset density", _fin(t.onset_density), "onsets/s", verdict="good"),
            ],
            moments=moments,
        ),
        ratio,
    )]


# ---------------------------------------------------------------------------
# 15. Clarity
# ---------------------------------------------------------------------------


def _detect_clarity(ctx: _Ctx) -> List[_Hit]:
    """Masking and congestion — is the energy arranged badly rather than distributed badly.

    Runs last of the fourteen so it inherits whatever masking pairs the more
    specific detectors did not claim. A source burying the vocal is a vocal
    problem and a source burying the mids is the low-mid buildup; what is left
    over — bass under drums, a pad under a lead — has no better home, and this
    is it.

    The two-track model and the separated one are complementary rather than
    redundant, which is exactly why they are folded into one finding. From a
    spectrum you can see energy that is buried; you cannot see what buried it.
    From stems you can name the masker; you cannot see the bands where two
    parts of the *same* source stack inside one auditory filter. Reporting both
    as two findings would be reporting one problem twice.
    """
    if ctx.no_programme or not ctx.has("clarity"):
        return []

    c = ctx.m.clarity

    clarity = _fin(c.clarity_index, 0.5)
    clarity_window = (_fin(ctx.profile.clarity_min, 0.45), 1.0)
    clarity_miss = targets.range_miss(clarity, clarity_window)

    masking = _fin(c.masking_index, 0.0)
    masking_window, masking_scale = _masking_window(ctx.profile)
    masking_ceiling = masking_window[1]
    masking_miss = targets.range_miss(masking, masking_window)

    pairs = ctx.take_masking(limit=3)
    pair_ratio = max((_mask_ratio(p) for p in pairs), default=0.0)

    if clarity_miss >= 0.0 and masking_miss <= 0.0 and pair_ratio <= 0.0:
        return []

    ratio = min(
        max(_ratio(clarity_miss, 0.07), _ratio(masking_miss, masking_scale), pair_ratio),
        6.0,
    )
    congestion = {k: _fin(v) for k, v in (c.band_congestion or {}).items()}
    worst = c.worst_congested_band or (
        max(congestion, key=lambda k: congestion[k]) if congestion else ""
    )
    moments = _moments(c.congested_moments)
    if not moments and pairs:
        moments = _moments([mo for p in pairs for mo in (p.moments or [])])

    spectrum_says_nothing = clarity_miss >= 0.0 and masking_miss <= 0.0
    if spectrum_says_nothing:
        # The two-track is happy and the stems are not. That is the case the
        # spectrum structurally cannot see: two sources stacked inside one
        # band look like one well-behaved band from outside.
        detail = (
            f"Nothing in the two-track's own spectrum is congested — clarity scores "
            f"{clarity:.2f} against a {_num(clarity_window[0], 2)} floor for "
            f"{ctx.profile.label} and {masking * 100:.1f}% of its audible loudness sits under "
            f"its own masking threshold, inside the {masking_ceiling * 100:.1f}% ceiling. "
            f"Separating the sources shows why that is not the whole story — two of them "
            f"stacked inside one auditory filter read as one well-behaved band from "
            f"outside."
            + _mask_sentence(pairs, "Measured against each other:")
            + " Both elements are present and neither is separately audible."
            + _moment_span(moments)
        )
        title = "One source is burying another"
        finding_id = "clarity.stem_masking"
        confidence = ctx.trust(0.80, "stem")
    else:
        detail = (
            f"Clarity scores {clarity:.2f} against a {_num(clarity_window[0], 2)} floor for "
            f"{ctx.profile.label}, with {masking * 100:.1f}% of the mix's audible loudness "
            f"sitting under its own masking threshold (ceiling "
            f"{masking_ceiling * 100:.1f}%). The worst band is "
            f"{worst.replace('_', ' ') or 'n/a'} at "
            f"{congestion.get(worst, 0.0):.2f} congestion, and transient frames are only "
            f"{_num(c.definition_db, 1)} dB above sustained ones. Energy is stacked rather "
            f"than spread: elements are present but not separately audible."
            + _mask_sentence(pairs)
            + _moment_span(moments)
        )
        title = "Elements are masking each other"
        finding_id = "clarity.congested"
        # A psychoacoustic spreading model over a measured spectrum, and with
        # separated sources naming the masker it is no longer only a model.
        confidence = ctx.trust(0.86 if pairs else 0.72, "clarity")

    return [_Hit(
        Finding(
            id=finding_id,
            dimension="clarity",
            title=title,
            severity=_severity(ratio),
            confidence=confidence,
            detail=detail,
            evidence=_mask_evidence(pairs) + [
                _ev("Clarity index", clarity, "", target_range=clarity_window,
                    verdict=_verdict(clarity, clarity_window, 0.07)),
                _ev("Masking index", masking, "", target_range=masking_window,
                    verdict=_verdict(masking, masking_window, masking_scale),
                    detail="Loudness-weighted share of the mix sitting under its own "
                           "masking threshold, from an ERB/spreading-function model."),
                _ev(f"Worst band congestion ({worst.replace('_', ' ') or 'n/a'})",
                    congestion.get(worst, 0.0), "", target_range=(0.0, 0.5),
                    verdict=_verdict(congestion.get(worst, 0.0), (0.0, 0.5), 0.15)),
                _ev("Definition", _fin(c.definition_db), "dB", target=None,
                    verdict="watch" if _fin(c.definition_db) < 4.0 else "good",
                    detail="Transient frames against sustained frames."),
                _ev("Spectral contrast", _fin(c.spectral_contrast), "dB", verdict="good"),
            ],
            moments=moments,
        ),
        ratio,
    )]


# ---------------------------------------------------------------------------
# Public API: detect_all
# ---------------------------------------------------------------------------

# Order matters. Specific detectors run before general ones so the general ones
# can stand down (`ctx.tags`) or find their evidence already claimed
# (`ctx.take_masking`):
#
#   phase        before stereo width and the low-end mono check
#   arrangement  before dynamic range — it describes the same flatness with the
#                section, the timestamp and the lift in LU attached
#   vocal        before mud and clarity — a source burying the vocal is a vocal
#                problem, and that is where a producer will look for it
#   mud          before clarity — a masking pair in the low mids is the low-mid
#                buildup seen from the other side
#   mud / harshness / low end   before frequency balance
#   clarity      last, so it inherits whatever masking pairs are left over
_DETECTORS = (
    _detect_clipping,
    _detect_phase,
    _detect_loudness,
    _detect_limiter,
    _detect_arrangement,
    _detect_dynamic_range,
    _detect_compression,
    _detect_stem_compression,
    _detect_low_end,
    _detect_vocal,
    _detect_mud,
    _detect_harshness,
    _detect_frequency_balance,
    _detect_stereo_width,
    _detect_transients,
    _detect_clarity,
)

_SEVERITY_RANK: Dict[str, int] = {"critical": 0, "major": 1, "minor": 2, "clean": 3}


def _rank_key(hit: _Hit) -> Tuple[int, float]:
    """Severity first, then recoverable points.

    Ranking the second key by `impact` rather than by the raw miss ratio is
    deliberate. The ratio says how many tolerance units a number is out by,
    which is not the same question as how much the listener gets back — a
    30 dB hole in the air band is a larger ratio than a clipped master and a
    smaller problem. `_IMPACT_WEIGHT` is where "how much this matters" lives,
    so the ordering the user sees is driven by it.
    """
    return (_SEVERITY_RANK.get(hit.finding.severity, 3), -_fin(hit.finding.impact))


def _cap(hits: List[_Hit], limit: int = MAX_FINDINGS) -> List[_Hit]:
    """Worst first, capped, with every dimension keeping its own worst finding.

    A naive top-N drops a dimension's only finding to make room for a second
    finding of a louder dimension, which leaves that dimension scored as clean
    when it is not. So the first pass takes one per dimension and the second
    fills whatever room is left.
    """
    ordered = sorted(hits, key=_rank_key)
    primary: List[_Hit] = []
    extra: List[_Hit] = []
    seen: set = set()
    for hit in ordered:
        if hit.finding.dimension in seen:
            extra.append(hit)
        else:
            seen.add(hit.finding.dimension)
            primary.append(hit)

    kept = primary[:limit]
    for hit in extra:
        if len(kept) >= limit:
            break
        kept.append(hit)
    return sorted(kept, key=_rank_key)


def _assign_impact(hits: List[_Hit], rescale: bool = True) -> None:
    """Health-score points recoverable, capped so the set can never sum past 100.

    Per finding: the dimension's ceiling weight, scaled by how far outside the
    window the value sits (a fully critical miss earns the whole weight), then
    discounted for repeats within a dimension because the second finding there
    is mostly the same fix.

    Called twice. The first pass is unscaled and exists only to rank and select
    (`_cap` needs an impact to sort by); the second runs on the surviving set
    and applies `_IMPACT_TOTAL_CAP`, so dropped findings cannot deflate the
    points attributed to the ones the user actually sees.
    """
    per_dimension: Dict[str, int] = {}
    raw: List[float] = []
    for hit in sorted(hits, key=lambda h: -h.ratio):
        dim = hit.finding.dimension
        n = per_dimension.get(dim, 0)
        per_dimension[dim] = n + 1
        discount = _REPEAT_DISCOUNT[min(n, len(_REPEAT_DISCOUNT) - 1)]
        weight = _IMPACT_WEIGHT.get(dim, 5.0)
        hit.finding.impact = round(
            _clamp(weight * _clamp(hit.ratio / _CRITICAL_AT, 0.18, 1.0) * discount, 0.0, 100.0),
            2,
        )
        raw.append(hit.finding.impact)

    total = sum(raw)
    if rescale and total > _IMPACT_TOTAL_CAP:
        scale = _IMPACT_TOTAL_CAP / total
        for hit in hits:
            hit.finding.impact = round(_clamp(hit.finding.impact * scale, 0.0, 100.0), 2)


def detect_all(m: Measurements, genre: str) -> List[Finding]:
    """Turn a `Measurements` into evidence-backed `Finding`s for one genre.

    Every finding carries the figure that produced it, the window it missed,
    and a `detail` sentence that is complete without any AI layer. Nothing is
    emitted for a file with no measurable programme, and nothing is emitted for
    a measurement whose value sits inside this genre's window.
    """
    ctx = _build_ctx(m, genre)
    if ctx.no_programme:
        return []

    hits: List[_Hit] = []
    for detector in _DETECTORS:
        try:
            hits.extend(detector(ctx))
        except Exception:
            # One detector failing must not cost the user the other thirteen.
            # The dimension simply scores as unassessed.
            continue

    # Drop misses too small to state honestly at the precision we report in.
    hits = [h for h in hits if h.ratio >= MIN_REPORTABLE_RATIO]

    _assign_impact(hits, rescale=False)   # gives _cap something meaningful to rank by
    hits = _cap(hits)
    _assign_impact(hits)                  # final points, capped across the kept set
    return [hit.finding for hit in sorted(hits, key=_rank_key)]


# ---------------------------------------------------------------------------
# Public API: score_dimensions
# ---------------------------------------------------------------------------

# Points removed from a dimension by its worst finding, plus a smaller amount
# for each additional one (a second problem in the same dimension is usually
# the same session's work).
_PENALTY: Dict[str, float] = {"critical": 58.0, "major": 34.0, "minor": 15.0}
_REPEAT_PENALTY = 8.0
_MAX_REPEAT_PENALTY = 16.0

# Score for a dimension with nothing to report but nothing to praise either —
# a measurement that could not be made rather than one that came back healthy.
_UNASSESSED_SCORE = 90.0


def _comfort(value: float, window: Tuple[float, float]) -> float:
    """0-1: how centrally a value sits inside its window. Edge = 0, middle = 1."""
    lo, hi = float(window[0]), float(window[1])
    if hi <= lo:
        return 0.5
    mid = 0.5 * (lo + hi)
    return _clamp(1.0 - 2.0 * abs(_fin(value, mid) - mid) / (hi - lo), 0.0, 1.0)


def _headroom_comfort(value: float, edge: float, scale: float) -> float:
    """0-1 for a one-sided window: how far inside the edge the value sits."""
    return _clamp(abs(_fin(value, edge) - edge) / max(scale, 1e-9), 0.0, 1.0)


def _elsewhere(
    what: str, value: float, unit: str, window: Tuple[float, float],
    genre_label: str, owner: str, nd: int = 1,
) -> Tuple[str, float, bool]:
    """Headline for a dimension with no findings whose figure is still out of window.

    A dimension can end up empty because a more specific detector claimed the
    problem (a polarity inversion owns the low end's mono ratio) or because the
    miss sits under the significance floor. Either way the clean headline must
    not turn that silence into praise, so it states the figure, states the
    window it is outside, and names where the problem is actually reported.
    """
    return (
        f"{what} measures {_num(value, nd)}{unit}, outside {genre_label}'s "
        f"{_win(window, nd, unit)} window — reported under {owner}, not here.",
        0.0,
        False,
    )


def _clean_report(dim: str, ctx: _Ctx) -> Tuple[str, float, bool]:
    """(headline, comfort 0-1, assessed) for a dimension with no findings.

    The headline says what is *good* and cites the figure that makes it good,
    because "no issues found" tells a producer nothing they can act on or trust.
    Where a measurement genuinely could not be made — a mono file's stereo
    field, a vocal that is not there, a 1.2 s file's loudness range — it says
    that instead of claiming health it cannot support.
    """
    m = ctx.m
    p = ctx.profile
    label = p.label

    if ctx.no_programme:
        return ("No measurable programme in this file — nothing assessed.",
                0.0, False)

    if dim == "clipping":
        tp = _fin(m.clipping.true_peak_dbtp, -120.0)
        head = TRUE_PEAK_CEILING_DBTP - tp
        return (
            f"No clipped samples and no inter-sample overs; true peak sits at {_num(tp, 2)} dBTP, "
            f"{_num(head, 1)} dB under the -1.0 dBTP delivery ceiling.",
            _headroom_comfort(tp, TRUE_PEAK_CEILING_DBTP, 6.0), True,
        )

    if dim == "phase":
        if ctx.is_mono:
            return ("Mono source: it folds down perfectly because there is nothing to fold. "
                    "No phase relationship exists to break.", 1.0, False)
        corr = _fin(m.phase.correlation, 1.0)
        loss = _fin(m.phase.mono_sum_loss_db, 0.0)
        return (
            f"Mono-compatible: correlation {corr:+.2f} against a {_num(p.correlation_min, 2)} "
            f"floor for {label}, and summing to mono costs only {_num(loss, 2)} dB.",
            _headroom_comfort(corr, _fin(p.correlation_min), 0.5), True,
        )

    if dim == "loudness":
        if not ctx.has("loudness"):
            return (f"Not assessed: {_num(ctx.duration, 1)} s is shorter than the gating "
                    f"window integrated loudness needs.", 0.0, False)
        lufs = _fin(m.loudness.integrated_lufs, -70.0)
        window = p.integrated_lufs
        if targets.in_range(lufs, window):
            return (
                f"{_num(lufs, 1)} LUFS lands inside the {_win(window, 1, ' LUFS')} window "
                f"{label} releases sit in, with {_num(m.loudness.plr_db, 1)} dB of "
                f"peak-to-loudness in hand.",
                _comfort(lufs, window), True,
            )
        head = TRUE_PEAK_CEILING_DBTP - _fin(m.loudness.true_peak_dbtp, -120.0)
        return (
            f"Quiet at {_num(lufs, 1)} LUFS but not stuck: {_num(head, 1)} dB of clean gain "
            f"reaches {_num(lufs + head, 1)} LUFS, inside {label}'s "
            f"{_win(window, 1, ' LUFS')} window, with no limiting needed.",
            0.6, True,
        )

    if dim == "limiter":
        if not ctx.has("limiter"):
            return (f"Not assessed: {_num(ctx.duration, 1)} s is too short to judge how the "
                    f"ceiling is being reached.", 0.0, False)
        psr = _fin(m.loudness.psr_p10_db, 0.0)
        runs = int(_fin(m.clipping.flat_run_count))
        clean_ceiling = (
            f"{runs} flat-topped runs and a distortion index of "
            f"{_fin(m.clipping.distortion_index):.3f}"
        )
        if psr > p.psr_p10_db[1]:
            # More peak headroom than the genre's masters keep. That is not a
            # limiter fault — there is barely a limiter — and dynamic_range
            # owns the observation.
            return (
                f"Barely limited: short-term PSR holds {_num(psr, 1)} dB, above "
                f"{label}'s {_win(p.psr_p10_db, 1, ' dB')} window, with {clean_ceiling}. "
                f"Nothing is being crushed; how far under genre level that leaves the mix "
                f"is reported under Dynamic Range.",
                0.5, True,
            )
        return (
            f"The ceiling is being held rather than generated: short-term PSR holds "
            f"{_num(psr, 1)} dB against {_win(p.psr_p10_db, 1, ' dB')} for {label}, with "
            f"{clean_ceiling}.",
            _comfort(psr, p.psr_p10_db), True,
        )

    if dim == "dynamic_range":
        if not ctx.has("dynamics"):
            return (f"Not assessed: {_num(ctx.duration, 1)} s is too short for a dynamic-range "
                    f"statistic to settle.", 0.0, False)
        crest = _fin(m.dynamics.crest_factor_db, 0.0)
        if not targets.in_range(crest, p.crest_factor_db):
            return (
                f"Crest factor is {_num(crest, 1)} dB and TT-DR {_num(m.dynamics.dr_value, 1)}, "
                f"outside {label}'s {_win(p.crest_factor_db, 1, ' dB')} window but by too "
                f"little, or at too healthy a level, to be worth a finding.",
                0.2, True,
            )
        return (
            f"Level moves the way {label} needs it to: {_num(crest, 1)} dB crest factor and "
            f"TT-DR {_num(m.dynamics.dr_value, 1)}, inside the "
            f"{_win(p.crest_factor_db, 1, ' dB')} window.",
            _comfort(crest, p.crest_factor_db), True,
        )

    if dim == "compression":
        if not ctx.has("dynamics"):
            return (f"Not assessed: {_num(ctx.duration, 1)} s is too short to measure crest "
                    f"inside the hits.", 0.0, False)
        micro = _fin(m.dynamics.micro_dynamics_db, 0.0)
        inside = targets.in_range(micro, p.micro_dynamics_db)
        return (
            f"The transient survives the processing: {_num(micro, 1)} dB of crest in each "
            f"50 ms frame, "
            + (f"inside {label}'s " if inside else f"above {label}'s ")
            + f"{_win(p.micro_dynamics_db, 1, ' dB')} window, with an estimated "
            f"{_num(m.dynamics.gain_reduction_estimate_db, 1)} dB of gain reduction.",
            _comfort(micro, p.micro_dynamics_db) if inside else 0.6, True,
        )

    if dim == "frequency_balance":
        inside = [b for b in ctx.bands.values() if b.miss_db == 0.0]
        outside = sorted((b for b in ctx.bands.values() if b.miss_db != 0.0),
                         key=lambda b: -b.ratio)
        if outside:
            names = ", ".join(b.label for b in outside[:3])
            return (
                f"{len(inside)} of {len(ctx.bands)} macro bands land inside their "
                f"{label} tolerance; the deviations in {names} are reported under the "
                f"dimension that owns them.",
                0.35, True,
            )
        worst = max(ctx.bands.values(), key=lambda b: abs(b.deviation_db)) if ctx.bands else None
        return (
            f"All {len(ctx.bands)} macro bands sit inside their {label} tolerance"
            + (f"; the largest deviation is {worst.label} at {_num(worst.deviation_db, 1)} dB "
               f"against a {_pm(worst.tolerance_db)} dB allowance." if worst else "."),
            _clamp(1.0 - (abs(worst.deviation_db) / max(worst.tolerance_db, 1e-9)
                          if worst else 0.0), 0.0, 1.0),
            True,
        )

    if dim == "mud":
        mud = _fin(m.spectral.mud_ratio_db, 0.0)
        if mud < p.mud_ratio_db[0]:
            return (
                f"Nothing muddy here — if anything the opposite: 150-400 Hz sits "
                f"{_num(mud, 1)} dB against 60-120 Hz, under {label}'s "
                f"{_win(p.mud_ratio_db, 1, ' dB')} window, so the mix may be missing body "
                f"rather than carrying too much. Band levels are reported under "
                f"Frequency Balance.",
                0.5, True,
            )
        return (
            f"The low mids stay out of the way: 150-400 Hz sits {_num(mud, 1)} dB against "
            f"60-120 Hz, inside {label}'s {_win(p.mud_ratio_db, 1, ' dB')} window, with "
            f"{_num(m.spectral.boxiness_db, 1)} dB of 300-600 Hz boxiness.",
            _comfort(mud, p.mud_ratio_db), True,
        )

    if dim == "harshness":
        harsh = _fin(m.spectral.harshness_index, 0.0)
        return (
            f"No edge in the ear's most sensitive region: harshness scores {harsh:.2f} against "
            f"a {_num(p.harshness_max, 2)} ceiling for {label} and sibilance "
            f"{_fin(m.spectral.sibilance_index):.2f} against {_num(p.sibilance_max, 2)}.",
            _headroom_comfort(harsh, _fin(p.harshness_max), _fin(p.harshness_max)), True,
        )

    if dim == "low_end":
        le = m.low_end
        mono = _fin(le.low_end_mono_ratio, 1.0)
        mono_window = (_fin(p.low_end_mono_min, 0.80), 1.0)
        if not ctx.is_mono and targets.range_miss(mono, mono_window) < 0.0:
            return (
                f"Only {mono * 100:.0f}% of the energy below 120 Hz is mono, against a "
                f"{mono_window[0] * 100:.0f}% floor for {label}. That is the channels being "
                f"out of polarity rather than an independent low-end fault — it is reported "
                f"under Phase & Mono Compatibility, and fixing that fixes this.",
                0.0, False,
            )
        base = (
            f"{mono * 100:.0f}% of the energy below 120 Hz is mono against a "
            f"{_fin(p.low_end_mono_min) * 100:.0f}% floor, and sub energy is "
            f"{_num(le.sub_energy_db, 1)} dB inside {label}'s "
            f"{_win(p.sub_energy_db, 1, ' dB')} window"
        )
        comfort = _comfort(_fin(le.sub_energy_db), p.sub_energy_db)

        if bool(le.kick_detected) and ctx.has("kick"):
            return (
                f"Low end is tight and mono below 120 Hz: {base}, and the kick at "
                f"{_num(le.kick_fundamental_hz, 0)} Hz clears the bass at "
                f"{_num(le.bass_fundamental_hz, 0)} Hz with "
                f"{_num(le.kick_bass_collision_db, 1)} dB of overlap against a "
                f"{_num(p.kick_bass_collision_max_db, 1)} dB ceiling.",
                comfort, True,
            )
        if bool(le.kick_detected):
            # A kick was found but the file is too short to judge how it sits
            # against the bass. Saying "tight" here would be praising a
            # measurement that was never scored.
            return (
                f"The bottom is clean where it could be measured: {base}. Kick-versus-bass "
                f"separation is not assessed — {_num(ctx.duration, 1)} s is too few hits to "
                f"compare the at-onset and between-onset spectra.",
                comfort * 0.5, True,
            )
        return (
            f"No kick pattern to separate from the bass, and the bottom is clean: {base}.",
            comfort, True,
        )

    if dim == "vocal_balance":
        v = m.vocal
        stem = ctx.stem("vocals")
        if ctx.has_stems and stem is not None and ctx.stems.vocal_to_instrument_db is not None:
            v2i = _fin(ctx.stems.vocal_to_instrument_db)
            spread = _stem_level_spread(stem)
            return (
                f"The lead holds its place, measured on the separated vocal rather than the "
                f"centre channel: it sits {_num(v2i, 1)} dB against drums, bass and everything "
                f"else summed, inside {label}'s {_win(p.vocal_to_instrument_db, 1, ' dB')} "
                f"window, and sings on {_fin(stem.active_ratio) * 100:.0f}% of the track"
                + (f" with {_num(spread, 1)} dB of level spread while it does."
                   if spread is not None else "."),
                _comfort(v2i, p.vocal_to_instrument_db), True,
            )
        if ctx.stems.available and stem is None:
            return (
                "Separation found no vocal source in this file, so there is no lead to "
                "balance. That is a measurement rather than an inference — the vocals stem "
                "came back at the separator's artefact floor"
                + (f", which is unusual for {label}." if p.vocal_expected else "."),
                # Absence is not health: there is nothing here to score well.
                1.0, False,
            )
        if ctx.is_mono:
            return ("Mono source: centre extraction is meaningless without a side channel, so "
                    "vocal balance is not assessed.", 0.0, False)
        if not ctx.has("vocal"):
            return (f"Not assessed: {_num(ctx.duration, 1)} s is too short to test for the "
                    f"syllabic modulation that identifies a voice.", 0.0, False)
        if not bool(v.vocal_present):
            if not p.vocal_expected:
                return (
                    f"No lead vocal, which is how {label} usually works — "
                    f"{_fin(v.center_energy_ratio) * 100:.0f}% of the 300 Hz-6 kHz energy is "
                    f"centred but it does not modulate at a syllabic rate.",
                    1.0, False,
                )
            return (
                f"No sustained lead vocal detected: {_fin(v.center_energy_ratio) * 100:.0f}% of "
                f"300 Hz-6 kHz is centred, but its envelope does not modulate at 2-8 Hz any more "
                f"than the sides do. Vocal balance is not assessed rather than guessed.",
                0.0, False,
            )
        return (
            f"The lead holds its place: centre sits {_num(v.vocal_to_instrument_db, 1)} dB "
            f"against everything else, inside {label}'s "
            f"{_win(p.vocal_to_instrument_db, 1, ' dB')} window, with "
            f"{_num(v.consistency_db, 1)} dB of level spread and intelligibility at "
            f"{_fin(v.intelligibility_index):.2f}.",
            _comfort(_fin(v.vocal_to_instrument_db), p.vocal_to_instrument_db), True,
        )

    if dim == "stereo_width":
        if ctx.is_mono:
            return ("Mono source: there is no stereo field to be too wide or too narrow, and "
                    "nothing here is a defect.", 1.0, False)
        width = _fin(m.stereo.width, 0.0)
        inside = targets.in_range(width, p.stereo_width)
        return (
            (f"Width is right for the genre: Side/Mid measures {_num(width, 2)} inside "
             if inside else
             f"Width is within rounding of the genre: Side/Mid measures {_num(width, 2)} "
             f"against ")
            + f"{label}'s {_win(p.stereo_width, 2)} window, with L/R matched to "
            f"{_num(m.stereo.balance_db, 2)} dB and a {_num(m.stereo.mono_sum_loss_db, 2)} dB "
            f"fold-down cost.",
            _comfort(width, p.stereo_width) if inside else 0.5, True,
        )

    if dim == "transients":
        t = m.transients
        if not ctx.has("transients"):
            return (f"Not assessed: {_num(ctx.duration, 1)} s is too short to average punch "
                    f"over enough onsets.", 0.0, False)
        if _fin(t.onset_density, 0.0) <= 0.0:
            return ("No onsets detected — this is sustained material with no transients to "
                    "lose.", 0.0, False)
        return (
            f"Hits land: punch scores {_fin(t.punch_index):.2f} against a "
            f"{_num(p.punch_min, 2)} floor for {label}, with "
            f"{_num(t.transient_to_sustain_db, 1)} dB between the transient and its level "
            f"50 ms later and a {_num(t.attack_time_ms, 1)} ms attack.",
            _headroom_comfort(_fin(t.punch_index), _fin(p.punch_min), 0.25), True,
        )

    if dim == "clarity":
        if not ctx.has("clarity"):
            return (f"Not assessed: {_num(ctx.duration, 1)} s is too short to measure masking "
                    f"across an arrangement.", 0.0, False)
        clarity = _fin(m.clarity.clarity_index, 0.5)
        mask_window, _ = _masking_window(p)
        stem_note = ""
        if ctx.has_stems:
            stem_note = (
                " Separating the sources finds nothing burying anything else either."
                if not ctx.masking_pairs else
                f" Separation does find {len(ctx.masking_pairs)} source-against-source "
                f"overlap{'s' if len(ctx.masking_pairs) > 1 else ''}, reported under the "
                f"dimension that owns each one."
            )
        return (
            f"Elements stay separately audible: clarity scores {clarity:.2f} against a "
            f"{_num(p.clarity_min, 2)} floor for {label}, with "
            f"{_fin(m.clarity.masking_index) * 100:.1f}% of the mix's audible loudness "
            f"sitting under its own masking threshold against a "
            f"{mask_window[1] * 100:.1f}% ceiling." + stem_note,
            _headroom_comfort(clarity, _fin(p.clarity_min), 0.25), True,
        )

    return ("Measured and inside the window for this genre.", 0.5, True)


def score_dimensions(
    findings: List[Finding], m: Measurements, genre: str
) -> List[DimensionScore]:
    """Roll findings up into one score per dimension — all fourteen, always.

    A dimension with no findings is not silent: it gets 90-100 and a headline
    naming the figure that makes it healthy. A dimension whose measurement could
    not be made (mono file's stereo field, a vocal that is not there, a file too
    short for the statistic) says so plainly and scores neutral rather than
    claiming health it cannot support.
    """
    ctx = _build_ctx(m, genre)
    by_dimension: Dict[str, List[Finding]] = {}
    for finding in findings or []:
        by_dimension.setdefault(str(finding.dimension), []).append(finding)

    # A polarity inversion drives the width reading; the stereo dimension is not
    # independently healthy just because the width finding was suppressed.
    polarity = next(
        (f for f in (findings or []) if f.id == "phase.polarity_inverted"), None
    )

    scores: List[DimensionScore] = []
    for dim in DIMENSIONS:
        hits = sorted(
            by_dimension.get(dim, []),
            key=lambda f: (_SEVERITY_RANK.get(f.severity, 3), -_fin(f.impact)),
        )

        if hits:
            worst = hits[0]
            penalty = _PENALTY.get(worst.severity, 15.0)
            penalty += min(_REPEAT_PENALTY * (len(hits) - 1), _MAX_REPEAT_PENALTY)
            score = _clamp(100.0 - penalty, 2.0, 100.0)
            headline = worst.title
            if len(hits) > 1:
                headline += f" (+{len(hits) - 1} more in this dimension)"
            scores.append(DimensionScore(
                dimension=dim,  # type: ignore[arg-type]
                label=DIMENSION_LABELS.get(dim, dim),
                score=round(score, 1),
                severity=worst.severity,
                headline=headline,
                finding_ids=[f.id for f in hits],
            ))
            continue

        if dim == "stereo_width" and polarity is not None and not ctx.is_mono:
            width = _fin(m.stereo.width, 0.0)
            scores.append(DimensionScore(
                dimension=dim,  # type: ignore[arg-type]
                label=DIMENSION_LABELS.get(dim, dim),
                score=45.0,
                severity="major",
                headline=(
                    f"Width reads {_num(width, 2)} only because the channels are in opposite "
                    f"polarity — fix the phase problem and re-measure, do not narrow the mix."
                ),
                # Cross-reference, not a second charge: the impact stays on the
                # phase finding, so nothing is double-counted in the health score.
                finding_ids=[polarity.id],
            ))
            continue

        headline, comfort, assessed = _clean_report(dim, ctx)
        score = 90.0 + 10.0 * comfort if assessed else _UNASSESSED_SCORE
        scores.append(DimensionScore(
            dimension=dim,  # type: ignore[arg-type]
            label=DIMENSION_LABELS.get(dim, dim),
            score=round(_clamp(score, 0.0, 100.0), 1),
            severity="clean",
            headline=headline,
            finding_ids=[],
        ))

    return scores
