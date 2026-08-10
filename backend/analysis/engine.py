"""Orchestration: one decoded buffer in, one `MixAnalysis` out.

This is the only module that is allowed to know about all three layers at once.
It decodes the file exactly once, hands the *same* `AudioBuffer` to all eleven
measurements, joins the result against the genre's windows via `detectors`, and
composes the payload the API serialises.

Five rules it holds to:

1. **Decode once.** `core.load_audio` resamples and (for anything but wav/flac)
   shells out to ffmpeg. Doing that per measurement would dominate the runtime.
2. **Never raise on valid audio.** Every stage runs under `_stage`, which logs a
   failure, records a warning and substitutes a neutral measurement. A vocal
   detector blowing up must not cost the user their clipping report.
3. **Measure, then judge, then write.** Numbers come from `dsp`, verdicts from
   `detectors`, prose from the AI layer — and the AI layer is optional at every
   point, because a report built from the findings alone is worse but never
   wrong.
4. **Nothing leaves here that JSON cannot carry.** `_sanitize` walks the whole
   payload and replaces NaN/inf before the model is re-validated. A single NaN
   in a series field is a 500 on the endpoint and an empty screen for the user.
5. **Depth is opt-in and never load-bearing.** Source separation is the only
   stage that costs more than the rest of the analysis put together, so
   `separate_stems` defaults to False and the default path stays around three
   seconds. When it is on and it fails, the report loses four upgraded findings
   and gains a warning — it does not lose the analysis.
"""

from __future__ import annotations

import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import clarify, core, detectors, targets
from .core import AudioBuffer
from .types import (
    ClarificationAnswer,
    ClarityMeasurement,
    ClippingMeasurement,
    DimensionScore,
    DynamicsMeasurement,
    EngineerReport,
    Finding,
    LoudnessMeasurement,
    LowEndMeasurement,
    Measurements,
    MixAnalysis,
    PhaseMeasurement,
    PlatformTarget,
    ReferenceDelta,
    ScoreCard,
    SectionAnalysis,
    SpectralMeasurement,
    StemAnalysis,
    StereoMeasurement,
    TransientMeasurement,
    VocalMeasurement,
)

logger = logging.getLogger(__name__)

__all__ = [
    "analyze_mix",
    "analyze_mix_detailed",
    "apply_answers",
    "build_score_card",
    "grade_for",
    "measure_stems_stage",
    "reference_match",
    "technical_score",
    "WAVEFORM_BUCKETS",
]


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Enough resolution for a 1400 px timeline at 1 px per bucket. More is wasted
# bytes on the wire; less and a 4-minute track loses its transients.
WAVEFORM_BUCKETS = 1400

# Delivery ceiling for mastering readiness. Same number the clipping detector
# uses — codec arithmetic, not taste.
TRUE_PEAK_CEILING_DBTP = detectors.TRUE_PEAK_CEILING_DBTP
# Readiness has to use the same slack the detector does, or a file passes the
# clipping check and is still blocked from mastering by the same hundredth of
# a decibel.
TRUE_PEAK_TOLERANCE_DB = detectors.TRUE_PEAK_TOLERANCE_DB

# The eight independent measurements run on a thread pool. numpy, scipy's
# filters and the FFTs all drop the GIL for arrays this size, so this is real
# parallelism, not bookkeeping — measured on a 4-minute stereo track: 17.9 s
# serial, 7.1 s at three workers. More workers is *slower*, because BLAS is
# already threading underneath and the two layers start fighting for cores.
def _default_workers() -> int:
    override = os.environ.get("MIXDOCTOR_ANALYSIS_WORKERS")
    if override and override.strip().isdigit():
        return max(1, int(override.strip()))
    return max(1, min(3, os.cpu_count() or 1))


ANALYSIS_WORKERS = _default_workers()

# How long the optional separation stage may run before it gives back whatever
# it finished. `separation.py` enforces this cooperatively between chunks and
# reports the truncated span in its own warnings, so a blown budget degrades to
# "we separated the first two minutes" rather than to nothing. The module's own
# default (300 s) is right for a batch job and far too long for a web request
# that a proxy will cut at 60-120 s, so the deployment gets to say.
def _separation_budget() -> float:
    override = os.environ.get("MIXDOCTOR_SEPARATION_TIMEOUT_S")
    if override:
        try:
            value = float(override.strip())
        except ValueError:
            value = 0.0
        if math.isfinite(value) and value > 0.0:
            return value
    from .dsp.separation import SEPARATION_TIMEOUT_S

    return float(SEPARATION_TIMEOUT_S)

# How much each dimension is allowed to move the overall score.
#
# These are "how much does this ruin the record", not "how hard is it to fix".
# Clipping and phase are at the top because they are damage: a squared-off
# waveform or a cancelling fold-down is broken on every playback system and no
# amount of taste makes it acceptable. Clarity and frequency balance come next
# because they decide whether the mix reads at all. Loudness and limiter sit in
# the middle: they are recoverable with a fader and a re-render. Stereo width
# and transients are at the bottom because they are the difference between a
# good mix and a great one, not between a usable and an unusable one — the air
# band being 2 dB shy has never lost anyone a release.
_HEALTH_WEIGHTS: Dict[str, float] = {
    "clipping": 1.00,
    "phase": 1.00,
    "clarity": 0.85,
    "frequency_balance": 0.85,
    "loudness": 0.80,
    "limiter": 0.80,
    "mud": 0.75,
    "low_end": 0.75,
    "dynamic_range": 0.70,
    "vocal_balance": 0.70,
    "harshness": 0.65,
    "compression": 0.55,
    "transients": 0.50,
    "stereo_width": 0.40,
}

# The roll-up is a weighted mean pulled toward its worst dimension. A pure mean
# over fourteen dimensions buries a single catastrophic one (one dimension at 42
# and thirteen at 95 averages to 91, which is a lie); a pure minimum throws away
# everything else that was measured. 65/35 keeps both readable.
_MEAN_SHARE = 0.65
_WORST_SHARE = 0.35

# Problems compound. The worst critical dimension is already fully expressed in
# the floor term above, so only the *additional* ones are charged here, scaled
# by their own weight: a second critical in clipping costs 9 points, a second
# critical in stereo width costs 3.6. Same idea, smaller number, for majors.
_EXTRA_CRITICAL_PENALTY = 9.0
_EXTRA_MAJOR_PENALTY = 3.0
_MAX_COMPOUND_PENALTY = 24.0

# There is deliberately no deviation discount here.
#
# A deviation is treated more gently than a defect at the same distance, but
# that happens once, in `detectors.deviation_penalty`, where the size of the
# miss is still in scope. A second flat multiplier at this level — applied to a
# penalty table that had already collapsed every deviation into one band —
# is what flattened the cross-genre gradient to three identical scores. If you
# are tempted to add one back, the thing to change is the curve.

# A soft ceiling, not a hard cap. Hard caps look right on one file and destroy
# the ordering across a catalogue: three mixes with three criticals each all
# pin to exactly the cap and become indistinguishable, which is what a score is
# for. Excess above the ceiling is compressed to 20% instead, so a clipped
# master can never grade well but a clipped master with nothing else wrong
# still outscores a clipped master with six other problems.
_CRITICAL_CEILING_SPAN = 45.0    # ceiling = 100 - span * weight of worst critical
_MAJOR_CEILING_SPAN = 22.0
_CEILING_COMPRESSION = 0.20

# The most of the gap to 100 that applying every prescription is allowed to close.
# See `compute_ceiling`: the remainder is the part of a record a fix list cannot
# reach — arrangement, performance, and the quality of the sources themselves.
_MAX_GAP_RECOVERY = 0.85

# Grade bands. A+ is reserved for a mix with nothing to say about it.
_GRADES: Tuple[Tuple[float, str], ...] = (
    (95.0, "A+"), (90.0, "A"), (87.0, "A-"),
    (83.0, "B+"), (80.0, "B"), (77.0, "B-"),
    (73.0, "C+"), (70.0, "C"), (67.0, "C-"),
    (63.0, "D+"), (60.0, "D"), (55.0, "D-"),
)

# Above this delta a platform's normalisation is audibly handing loudness back.
_TURNDOWN_LU = 0.5

# Largest/smallest value `_sanitize` will substitute for an infinity. Well
# inside float64 and inside anything the UI will try to plot.
_INF_SUBSTITUTE = 1.0e6


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _fin(value: Any, default: float = 0.0) -> float:
    """Coerce anything to a finite float."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(min(max(_fin(value, lo), lo), hi))


def _num(value: float, nd: int = 1) -> str:
    v = _fin(value, 0.0)
    if abs(v) < 5e-3:
        v = 0.0
    text = f"{v:.{nd}f}"
    return "0" if text in ("-0", "-0.0", "-0.00") else text


def _sanitize(obj: Any) -> Any:
    """Recursively replace anything JSON cannot carry with a finite stand-in.

    NaN and +/-Infinity are legal float64 and illegal JSON: `json.dumps` emits
    the bare tokens `NaN` / `Infinity`, which `JSON.parse` rejects, so one bad
    value in one series field blanks the whole report in the browser. Everything
    that leaves this module goes through here.
    """
    if isinstance(obj, float):
        if math.isnan(obj):
            return 0.0
        if math.isinf(obj):
            return _INF_SUBSTITUTE if obj > 0 else -_INF_SUBSTITUTE
        return obj
    if isinstance(obj, (int, bool, str)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {key: _sanitize(val) for key, val in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(item) for item in obj]
    if isinstance(obj, np.generic):
        return _sanitize(obj.item())
    if isinstance(obj, np.ndarray):
        return [_sanitize(item) for item in obj.tolist()]
    return obj


def load_or_explain(path: str, label: str = "file") -> AudioBuffer:
    """`core.load_audio`, with every failure turned into something a user can act on.

    `core.load_audio` raises `ValueError` subclasses for the cases it
    anticipates — unsupported extension, too short, silent — and those messages
    are already written for the person who uploaded the file, so they pass
    straight through. What it does *not* wrap is the decoder itself: a truncated
    MP3, a `.wav` that is really a renamed PDF, or a host with no ffmpeg on the
    PATH all surface as `RuntimeError`/`FileNotFoundError` from deep inside
    soundfile or pydub. Those are still the upload's problem, not a bug, so they
    are re-raised as `ValueError` and the API answers 422 with a sentence
    instead of 500 with nothing.
    """
    try:
        return core.load_audio(path)
    except ValueError:
        raise
    except FileNotFoundError as exc:
        # Not the upload: soundfile refused the container, so the decode fell
        # through to pydub, which shells out to ffmpeg/ffprobe — and those are
        # not on this host's PATH. Naming the exception here would tell the user
        # their file is missing, which is both wrong and unactionable for them.
        logger.error(
            "engine: ffmpeg/ffprobe is not installed, so %s could not be decoded "
            "via the fallback path: %s", path, exc
        )
        raise ValueError(
            f"This {label} is in a format this server cannot currently decode. "
            f"Re-export it as a 16- or 24-bit WAV and try again."
        ) from exc
    except Exception as exc:
        logger.warning("engine: could not decode %s (%s): %s", path, type(exc).__name__, exc)
        raise ValueError(
            f"This {label} could not be decoded. It may be corrupt, truncated, or in a "
            f"container this server cannot open. Re-export it as a 16- or 24-bit WAV "
            f"and try again."
        ) from exc


def grade_for(score: float) -> str:
    """A+ .. F from a 0-100 health score."""
    value = _clamp(score, 0.0, 100.0)
    for threshold, letter in _GRADES:
        if value >= threshold:
            return letter
    return "F"


# ---------------------------------------------------------------------------
# Measurement pass
# ---------------------------------------------------------------------------


def _stage(
    name: str,
    fn: Callable[[], Any],
    fallback: Callable[[], Any],
    timings: Dict[str, float],
    warnings: List[str],
) -> Any:
    """Run one measurement, timing it, and never let it take the request down.

    Called from worker threads, so `timings` and `warnings` are mutated
    concurrently. Both are plain list/dict appends and item sets under CPython,
    which are atomic — no lock needed, and a lock here would serialise the pool.
    """
    start = time.perf_counter()
    try:
        result = fn()
    except Exception:
        logger.exception("engine: %s measurement failed; substituting a neutral result", name)
        warnings.append(
            f"The {name} measurement could not be completed on this file; that dimension "
            f"is reported as unassessed."
        )
        result = fallback()
    timings[name] = round((time.perf_counter() - start) * 1000.0, 1)
    return result


# Which dimensions become meaningless when a given measurement stage dies.
# A neutral fallback measurement is all zeros, and zeros are indistinguishable
# from a genuinely calm reading — so the detector scores a crashed stage as
# "clean". Production hit exactly that: librosa could not JIT on the host, the
# transient stage returned zeros, and the report told the user their drums were
# fine. A dimension nobody measured has to say so.
_STAGE_DIMENSIONS: Dict[str, Tuple[str, ...]] = {
    "loudness": ("loudness", "limiter"),
    "clipping": ("clipping",),
    "spectral": ("frequency_balance", "mud", "harshness"),
    "stereo": ("stereo_width",),
    "phase": ("phase",),
    "dynamics": ("dynamic_range", "compression"),
    "transients": ("transients",),
    "low_end": ("low_end",),
    "vocal": ("vocal_balance",),
    "clarity": ("clarity",),
}

_UNASSESSED_SCORE = 70.0


def _mark_unassessed(
    dimensions: List[DimensionScore],
    timings: Dict[str, float],
    warnings: List[str],
) -> None:
    """Downgrade any dimension whose measurement stage failed.

    `_stage` already appended a warning naming the stage; this makes the
    dimension itself honest so the UI and the AI layer both see "unassessed"
    rather than a confident green tick. The score is neutral rather than zero:
    a stage we could not run is not evidence of a problem, and should not drag
    the health score down any more than it should prop it up.
    """
    failed = {
        name for name in _STAGE_DIMENSIONS
        if any(f"The {name} measurement could not be completed" in w for w in warnings)
    }
    if not failed:
        return

    affected = {dim for stage in failed for dim in _STAGE_DIMENSIONS[stage]}
    for dimension in dimensions:
        if dimension.dimension in affected:
            dimension.score = _UNASSESSED_SCORE
            dimension.severity = "minor"
            dimension.headline = (
                "Not assessed — this measurement could not be completed on this file."
            )
            dimension.finding_ids = []


def _neutral_loudness() -> LoudnessMeasurement:
    return LoudnessMeasurement(
        integrated_lufs=-70.0, momentary_max_lufs=-70.0, short_term_max_lufs=-70.0,
        loudness_range_lu=0.0, true_peak_dbtp=-120.0, sample_peak_dbfs=-120.0,
        plr_db=0.0, psr_p10_db=0.0, psr_median_db=0.0,
    )


def _neutral_clipping() -> ClippingMeasurement:
    return ClippingMeasurement(
        sample_peak_dbfs=-120.0, true_peak_dbtp=-120.0, clipped_samples=0,
        clip_percentage=0.0, longest_flat_run=0, flat_run_count=0,
        inter_sample_overs=0, is_float_over_unity=False, distortion_index=0.0,
    )


def _neutral_spectral(genre: str) -> SpectralMeasurement:
    centers = list(core.THIRD_OCTAVE_CENTERS)
    return SpectralMeasurement(
        third_octave_centers=centers,
        third_octave_db=[0.0] * len(centers),
        bands=[],
        spectral_tilt_db_per_decade=0.0, spectral_centroid_hz=0.0,
        mud_ratio_db=0.0, mud_to_mid_db=0.0, boxiness_db=0.0,
        harshness_index=0.0, sibilance_index=0.0, sharpness_acum=0.0,
    )


def _neutral_stereo(is_mono: bool) -> StereoMeasurement:
    return StereoMeasurement(
        is_mono_source=bool(is_mono), correlation=1.0, width=0.0,
        mono_sum_loss_db=0.0, low_end_side_energy_db=-90.0, balance_db=0.0,
    )


def _neutral_phase() -> PhaseMeasurement:
    return PhaseMeasurement(
        correlation=1.0, polarity_inverted=False, mono_compatible=True,
        mono_sum_loss_db=0.0, worst_band=None, worst_band_correlation=1.0,
    )


def _neutral_dynamics() -> DynamicsMeasurement:
    return DynamicsMeasurement(
        crest_factor_db=0.0, peak_to_loudness_db=0.0, macro_dynamics_lu=0.0,
        micro_dynamics_db=0.0, dr_value=0.0, rms_db=-120.0,
        pumping_index=0.0, pumping_rate_hz=0.0, gain_reduction_estimate_db=0.0,
    )


def _neutral_transients() -> TransientMeasurement:
    return TransientMeasurement(
        onset_density=0.0, estimated_tempo=0.0, attack_time_ms=0.0,
        punch_index=0.0, transient_to_sustain_db=0.0, smearing_index=0.0,
    )


def _neutral_low_end() -> LowEndMeasurement:
    return LowEndMeasurement(
        kick_detected=False, kick_fundamental_hz=0.0, kick_count=0,
        bass_fundamental_hz=0.0, sub_energy_db=-90.0, kick_bass_collision_db=-90.0,
        ducking_depth_db=0.0, has_sidechain=False, kick_definition_db=0.0,
        low_end_mono_ratio=1.0, sub_rumble_db=-90.0,
    )


def _neutral_vocal() -> VocalMeasurement:
    return VocalMeasurement(
        vocal_present=False, center_energy_ratio=0.0, vocal_to_instrument_db=0.0,
        intelligibility_index=0.0, presence_balance_db=0.0, sibilance_db=-90.0,
        consistency_db=0.0,
    )


def _neutral_clarity() -> ClarityMeasurement:
    return ClarityMeasurement(
        clarity_index=0.5, spectral_flatness=0.0, spectral_contrast=0.0,
        masking_index=0.0, worst_congested_band=None, definition_db=0.0,
    )


def _neutral_sections(note: str = "") -> SectionAnalysis:
    """Structural segmentation that could not be produced.

    `sections.measure_sections` already swallows its own failures and falls
    back to a single section, so this is only reached if the call itself blew
    up — an import error, or the buffer being unusable.
    """
    return SectionAnalysis(
        available=False,
        sections=[],
        loudness_spread_lu=0.0,
        peak_lift_db=0.0,
        low_end_swing_db=0.0,
        notes=[note] if note else [],
    )


def _neutral_stems(note: str = "") -> StemAnalysis:
    """A fully-populated "no stems" result. Never None where the model forbids it."""
    return StemAnalysis(
        available=False,
        model_name="",
        separation_ms=0,
        stems=[],
        masking_pairs=[],
        vocal_to_instrument_db=None,
        kick_to_bass_db=None,
        kick_fundamental_hz=None,
        bass_fundamental_hz=None,
        warnings=[note] if note else [],
    )


def _measurements_shell(buf: AudioBuffer, **parts: Any) -> Measurements:
    """A full `Measurements` with neutral placeholders for anything not supplied.

    The reference pass only needs four of the ten measurements, but
    `build_reference_delta` takes a `Measurements` so it stays symmetric with
    the mix side. This fills the rest with honest neutrals rather than making
    the model's fields optional.
    """
    defaults: Dict[str, Any] = {
        "loudness": _neutral_loudness(),
        "clipping": _neutral_clipping(),
        "spectral": _neutral_spectral(""),
        "stereo": _neutral_stereo(buf.is_mono),
        "phase": _neutral_phase(),
        "dynamics": _neutral_dynamics(),
        "transients": _neutral_transients(),
        "low_end": _neutral_low_end(),
        "vocal": _neutral_vocal(),
        "clarity": _neutral_clarity(),
    }
    defaults.update({k: v for k, v in parts.items() if v is not None})
    return Measurements(
        duration_seconds=round(_fin(buf.duration), 3),
        sample_rate=int(buf.sr),
        original_sample_rate=int(buf.original_sr),
        is_mono=bool(buf.is_mono),
        bit_depth=buf.source_bit_depth,
        **defaults,
    )


def measure_all(
    buf: AudioBuffer,
    genre: str,
    timings: Dict[str, float],
    warnings: List[str],
    workers: int = ANALYSIS_WORKERS,
) -> Measurements:
    """Run all eleven measurements against one buffer.

    Nine of the eleven are independent and go out on the thread pool at once.
    The other two are chained because they consume a result rather than the
    audio: `phase` reads the `StereoMeasurement` instead of recomputing the
    correlations, and `dynamics` takes the `LoudnessMeasurement` (PLR comes
    straight off it) plus the detected tempo, so the pumping search knows which
    modulation rate to look at. Both are cheap; the pool covers the expensive
    ones.

    `sections` is on the pool rather than after it because it is the one
    measurement whose *output* is not a scalar summary of the whole file — it
    is what lets a detector say "the chorus at 2:41 does not lift" instead of
    "loudness range is low". It is cheap enough to be unconditional (0.1-1.3 s
    against transients' 2.7 s on the same material) and it never raises: the
    module's own fallback is one section covering the file.

    Note that the per-stage timings below overlap in wall-clock terms — they
    are each stage's own duration, not a partition of the total.
    """
    from .dsp import (  # `analysis.dsp` resolves lazily; force it before threading
        measure_clarity,
        measure_clipping,
        measure_dynamics,
        measure_loudness,
        measure_low_end,
        measure_phase,
        measure_sections,
        measure_spectral,
        measure_stereo,
        measure_transients,
        measure_vocal,
    )

    jobs: List[Tuple[str, Callable[[], Any], Callable[[], Any]]] = [
        # Longest first: the pool drains fastest when the big jobs start first.
        ("stereo", lambda: measure_stereo(buf), lambda: _neutral_stereo(buf.is_mono)),
        ("transients", lambda: measure_transients(buf), _neutral_transients),
        ("sections", lambda: measure_sections(buf, genre),
         lambda: _neutral_sections(
             "Structural segmentation could not be run on this file, so nothing is "
             "reported about how the arrangement moves from section to section.")),
        ("low_end", lambda: measure_low_end(buf), _neutral_low_end),
        ("clipping", lambda: measure_clipping(buf), _neutral_clipping),
        ("loudness", lambda: measure_loudness(buf), _neutral_loudness),
        ("clarity", lambda: measure_clarity(buf), _neutral_clarity),
        ("vocal", lambda: measure_vocal(buf), _neutral_vocal),
        ("spectral", lambda: measure_spectral(buf, genre), lambda: _neutral_spectral(genre)),
    ]

    started = time.perf_counter()
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mixdoc") as pool:
            futures = [
                (name, pool.submit(_stage, name, fn, fallback, timings, warnings))
                for name, fn, fallback in jobs
            ]
            done = {name: future.result() for name, future in futures}
    else:
        done = {
            name: _stage(name, fn, fallback, timings, warnings)
            for name, fn, fallback in jobs
        }

    stereo = done["stereo"]
    loudness = done["loudness"]
    transients = done["transients"]

    phase = _stage("phase", lambda: measure_phase(buf, stereo),
                   _neutral_phase, timings, warnings)
    tempo = _fin(getattr(transients, "estimated_tempo", 0.0), 0.0)
    dynamics = _stage(
        "dynamics",
        lambda: measure_dynamics(buf, loudness, tempo_bpm=tempo if tempo > 0 else None),
        _neutral_dynamics, timings, warnings,
    )
    timings["measure_wall"] = round((time.perf_counter() - started) * 1000.0, 1)

    clipping = done["clipping"]
    spectral = done["spectral"]
    low_end = done["low_end"]
    vocal = done["vocal"]
    clarity = done["clarity"]
    sections = done["sections"]

    return Measurements(
        duration_seconds=round(_fin(buf.duration), 3),
        sample_rate=int(buf.sr),
        original_sample_rate=int(buf.original_sr),
        is_mono=bool(buf.is_mono),
        bit_depth=buf.source_bit_depth,
        loudness=loudness,
        clipping=clipping,
        spectral=spectral,
        stereo=stereo,
        phase=phase,
        dynamics=dynamics,
        transients=transients,
        low_end=low_end,
        vocal=vocal,
        clarity=clarity,
        sections=sections,
    )


def measure_stems_stage(
    buf: AudioBuffer,
    genre: str,
    timings: Dict[str, float],
    warnings: List[str],
    timeout_s: Optional[float] = None,
) -> StemAnalysis:
    """The optional per-source pass. Opt-in, bounded, and never fatal.

    This is the only heavyweight stage in the analysis — a neural separator and
    four more measurement passes over its output — which is why it is off by
    default and why every failure path here ends in an `available=False`
    `StemAnalysis` rather than an exception. `separation.measure_stems` already
    promises not to raise; the try/except is the second line of defence for the
    cases it cannot promise about (no torch, an OOM, a killed worker).

    The warnings it produces are already written for the person who uploaded
    the file, so the ones explaining a *failure* are lifted onto the report.
    The ones that merely qualify a success ("no vocal stem", "ran on the CPU")
    stay on `measurements.stems.warnings`, where the UI can show them next to
    the numbers they qualify instead of at the top of the page.
    """
    started = time.perf_counter()
    try:
        from .dsp.separation import measure_stems

        budget = _fin(timeout_s, 0.0)
        if budget <= 0.0:
            budget = _separation_budget()
        stems = measure_stems(buf, genre, timeout_s=budget)
    except Exception as exc:
        logger.exception("engine: the stem separation stage failed")
        stems = _neutral_stems(
            f"Stem separation could not be run on this host ({type(exc).__name__}: {exc}). "
            f"Every finding below is measured from the two-track."
        )
    timings["stems"] = round((time.perf_counter() - started) * 1000.0, 1)

    if not stems.available:
        # Deduplicated and capped: a broken separator must not push the real
        # findings off the top of the report with three paragraphs about itself.
        seen: set = set()
        for note in list(stems.warnings)[:3]:
            text = str(note or "").strip()
            if text and text not in seen and text not in warnings:
                seen.add(text)
                warnings.append(text)
        if not seen:
            warnings.append(
                "Stem separation was requested but produced no usable sources; the mix "
                "was analysed from the two-track."
            )
    return stems


def measure_reference(
    buf: AudioBuffer,
    genre: str,
    timings: Dict[str, float],
    warnings: List[str],
    workers: int = ANALYSIS_WORKERS,
    tempo_hint: Optional[float] = None,
) -> Measurements:
    """The four measurements a `ReferenceDelta` actually reads.

    A reference track is not being diagnosed, only compared, so running the
    other six on it would roughly double the request for nothing: no finding is
    ever emitted about the reference's clipping or its vocal balance.

    `tempo_hint` is the *mix's* tempo, which is not necessarily the reference's.
    It is safe to pass anyway: the only thing the hint changes is which
    modulation rate the pumping search looks at, and the only figure read off
    this measurement is `crest_factor_db`. Without it, `measure_dynamics` runs
    its own beat tracker, which on a 4-minute reference costs ~3.7 s to produce
    a number nothing downstream reads.
    """
    from .dsp import (
        measure_dynamics,
        measure_loudness,
        measure_spectral,
        measure_stereo,
    )

    jobs: List[Tuple[str, Callable[[], Any], Callable[[], Any]]] = [
        ("ref_stereo", lambda: measure_stereo(buf), lambda: _neutral_stereo(buf.is_mono)),
        ("ref_loudness", lambda: measure_loudness(buf), _neutral_loudness),
        ("ref_spectral", lambda: measure_spectral(buf, genre), lambda: _neutral_spectral(genre)),
    ]
    if workers > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            futures = [
                (name, pool.submit(_stage, name, fn, fallback, timings, warnings))
                for name, fn, fallback in jobs
            ]
            done = {name: future.result() for name, future in futures}
    else:
        done = {
            name: _stage(name, fn, fallback, timings, warnings)
            for name, fn, fallback in jobs
        }

    hint = _fin(tempo_hint, 0.0)
    dynamics = _stage(
        "ref_dynamics",
        lambda: measure_dynamics(buf, done["ref_loudness"],
                                 tempo_bpm=hint if hint > 0 else None),
        _neutral_dynamics, timings, warnings,
    )
    return _measurements_shell(
        buf,
        loudness=done["ref_loudness"],
        spectral=done["ref_spectral"],
        stereo=done["ref_stereo"],
        dynamics=dynamics,
    )


# ---------------------------------------------------------------------------
# Health score / grade / ceiling — the legacy composite
# ---------------------------------------------------------------------------
#
# **This is no longer what the UI leads with. `build_score_card` is.**
#
# `health_score` and `grade` conflate two unrelated questions — "is anything
# actually wrong with this render?" and "does this sound like the genre?" — and
# the second one dominates, because thirteen of the fourteen dimensions are
# stylistic and only one is damage. That produced the pair that made the number
# indefensible: `mix_clean` (zero defects, ready to master) at 57.7/D-, and
# `reference_trap` (two real defects, not ready) at 90.5/A.
#
# They are kept, and kept *bit-identical*, for one reason: they are the wire
# contract for every stored `AnalysisHistory` row and every client already in a
# browser, and `tools/check_regressions.py` pins them against the fixtures so a
# retune of the penalty curves cannot drift them silently. Fix the score by
# reading `MixAnalysis.scores`, not by re-tuning what is below.


def compute_health(
    dimensions: Sequence[DimensionScore], findings: Sequence[Finding]
) -> float:
    """Weighted roll-up of the dimension scores into one 0-100 figure.

    Legacy. See the section comment above and `ScoreCard`: this answers "how far
    is this from a finished record of this kind", which is not the same question
    as "is anything broken", and the report now asks both separately.

    Three terms, in order:

    1. The weighted mean of all fourteen dimension scores.
    2. A pull 35% of the way toward the worst dimension, itself weighted, so a
       catastrophic clipping score drags harder than a catastrophic width score.
    3. A compounding penalty for every critical/major dimension *past the worst
       one*, and a compressed ceiling while anything critical is outstanding.

    Step 3 is what makes the number usable across a catalogue rather than only
    on one file: a mix with one critical problem and a mix with four must not
    land on the same score.

    **Defects and deviations pull differently, and that is handled upstream.**
    A deviation's gentler treatment is applied once, in
    `detectors.deviation_penalty`, where the measured distance from the
    reference is still in scope: it costs less than a defect at the same
    distance and can never cost as much as a critical. By the time a dimension
    score arrives here that discount is already inside the number, so this
    function takes every dimension at face value.

    Discounting again here — as a flat multiplier on the deficit — is what
    flattened the cross-genre gradient: the penalty table had already collapsed
    every deviation into "minor", and a second multiplier on top made a trap
    master score the same against ambient as against rock. Soften once, at the
    point where the magnitude is still visible. The severity-class discount in
    `_weights_for` below is a different thing and does still apply: it is about
    which *class* of finding may pull a ceiling down, not how far out it is.
    """
    if not dimensions:
        return 0.0

    weighted_sum = 0.0
    weight_total = 0.0
    worst_adjusted = 100.0
    for dim in dimensions:
        weight = _HEALTH_WEIGHTS.get(str(dim.dimension), 0.6)
        score = _clamp(dim.score, 0.0, 100.0)
        weighted_sum += weight * score
        weight_total += weight
        # A dimension's pull on the floor is scaled by its weight: a width score
        # of 45 lands at 78 here, a clipping score of 45 lands at 45.
        worst_adjusted = min(worst_adjusted, 100.0 - weight * (100.0 - score))

    mean = weighted_sum / max(weight_total, 1e-9)
    health = _MEAN_SHARE * mean + _WORST_SHARE * worst_adjusted

    def _weights_for(severity: str) -> List[float]:
        """Dimension weights carrying a finding of this grade, worst first.

        Graded by `detectors.scoring_grade`, which reads what the finding costs
        rather than the label it shows the user. A deviation's label is capped
        at "major" for the producer's benefit; reading that cap here would take
        the critical ceiling off every stylistic gap, however large, and the
        ceiling is most of what separates a mix judged against its own genre
        from the same mix judged against a distant one.
        """
        weights: Dict[str, float] = {}
        for f in findings or []:
            if str(detectors.scoring_grade(f)) != severity:
                continue
            dim_key = str(getattr(f, "dimension", ""))
            base = _HEALTH_WEIGHTS.get(dim_key, 0.6)
            # A dimension is charged once, at the heaviest thing in it.
            weights[dim_key] = max(weights.get(dim_key, 0.0), base)
        return sorted(weights.values(), reverse=True)

    critical = _weights_for("critical")
    major = _weights_for("major")

    penalty = (
        _EXTRA_CRITICAL_PENALTY * sum(critical[1:])
        + _EXTRA_MAJOR_PENALTY * sum(major[1:])
    )
    health -= min(penalty, _MAX_COMPOUND_PENALTY)

    if critical:
        ceiling = 100.0 - _CRITICAL_CEILING_SPAN * critical[0]
    elif major:
        ceiling = 100.0 - _MAJOR_CEILING_SPAN * major[0]
    else:
        ceiling = 100.0
    if health > ceiling:
        health = ceiling + (health - ceiling) * _CEILING_COMPRESSION

    return round(_clamp(health, 0.0, 100.0), 1)


def compute_ceiling(health: float, findings: Sequence[Finding]) -> float:
    """Where the mix lands if every prescription is applied.

    The per-report impact cap in `detectors._assign_impact` (60 points total) is
    not on its own enough to keep this honest: a mix at 47 with 52 points of
    recoverable impact adds up to 99.8, i.e. the report tells someone holding an
    F that a evening's EQ makes it a flawless record. It doesn't. Rebalancing a
    broken mix gets you a good one, not a perfect one — the arrangement, the
    performances and the source material all still cap what is reachable, and
    none of those are things a prescription list can fix.

    So recovery closes at most `_MAX_GAP_RECOVERY` of the distance to 100. That
    keeps the number monotonic — a mix with fewer problems still ceilings higher
    than one with more — while never promising the last few points, which are
    exactly the ones that are not available from a fix list.
    """
    recoverable = sum(_fin(getattr(f, "impact", 0.0), 0.0) for f in findings)
    base = _clamp(_fin(health), 0.0, 100.0)
    reachable = (100.0 - base) * _MAX_GAP_RECOVERY
    return round(_clamp(base + min(recoverable, reachable), 0.0, 100.0), 1)


# ---------------------------------------------------------------------------
# Mastering readiness
# ---------------------------------------------------------------------------


def mastering_readiness(
    m: Measurements, findings: Sequence[Finding], intent: str = "full_mix"
) -> Tuple[bool, List[str]]:
    """Ready only when nothing is critical, nothing is clipped, and TP <= -1 dBTP.

    Every blocker names the figure that produced it. "Not ready" with no reason
    is a worse answer than no answer.

    On a reference track the question is not asked. A released master is not
    waiting to be mastered, so "not ready — re-export with the fader down" is
    an instruction aimed at the wrong person about the wrong file. The same
    facts are still measured and still reported, as readings, on the findings
    list; here they are stated without the imperative.
    """
    is_reference = str(intent) == "reference"
    blockers: List[str] = []

    clip = m.clipping
    true_peak = _fin(clip.true_peak_dbtp, -120.0)
    clipped = int(_fin(clip.clipped_samples, 0))
    clip_pct = _fin(clip.clip_percentage, 0.0)

    if clipped > 0:
        blockers.append(
            f"{clipped:,} samples ({clip_pct:.4f}% of the file) are pinned at the ceiling in "
            f"{int(_fin(clip.flat_run_count))} flat-topped runs, the longest "
            f"{int(_fin(clip.longest_flat_run))} samples."
            + ("" if is_reference else
               " A mastering engineer cannot undo clipping that is already rendered into the "
               "file — re-export with the master fader down.")
        )

    if true_peak > TRUE_PEAK_CEILING_DBTP + TRUE_PEAK_TOLERANCE_DB:
        overs = int(_fin(clip.inter_sample_overs, 0))
        blockers.append(
            f"True peak is {_num(true_peak, 2)} dBTP, "
            f"{_num(true_peak - TRUE_PEAK_CEILING_DBTP, 2)} dB above the "
            f"{_num(TRUE_PEAK_CEILING_DBTP, 1)} dBTP every platform asks for"
            + (f"; 4x oversampling already finds {overs:,} inter-sample overs." if overs
               else "." if is_reference
               else ". Leave the mastering engineer headroom to work in.")
        )

    for finding in findings:
        if getattr(finding, "severity", "") != "critical":
            continue
        sentence = str(getattr(finding, "detail", "") or "").split(". ")[0].strip()
        if sentence and not sentence.endswith("."):
            sentence += "."
        blockers.append(
            f"Critical — {finding.title}"
            + (f": {sentence}" if sentence else "")
        )

    # Deduplicate while keeping order: clipping can be both its own blocker and
    # the critical finding that names it.
    seen: set = set()
    unique: List[str] = []
    for blocker in blockers:
        key = blocker[:80]
        if key in seen:
            continue
        seen.add(key)
        unique.append(blocker)

    return (not unique), unique


# ---------------------------------------------------------------------------
# The score card — two numbers, each answering one question
# ---------------------------------------------------------------------------
#
# See `types.ScoreCard`. The split exists because the composite above could not
# be read: zero defects and ready to master scored D-, two defects and not ready
# scored A. `technical` answers "is anything wrong with this render", reads
# nothing but defects, and is the only figure here that carries a letter.
# `reference_match` answers "how close is this to the genre", reads nothing but
# deviations, and never carries one — sitting away from a reference is a
# description of a record, not a fault in it.

# What each additional defect costs, as a share of its own weight. The worst one
# is charged in full; a second and a third compound but not linearly, on the same
# reasoning as `_REPEAT_PENALTY` in the dimension roll-up — two defects in one
# render are usually one bad export, not two independent disasters.
_DEFECT_EXTRA_SHARE = 0.5

# A defect with no distance recorded anywhere is still a defect. Charge it as
# one tolerance unit out, which is what the shared curve calls a "minor". Only
# reachable from an older payload, since a defect on the ruler always carries a
# measured miss.
_DEFECT_UNMEASURED_RATIO = 1.0

# The profile defect magnitudes are measured against, so that changing the genre
# dropdown cannot move the technical score.
#
# This is not a stylistic choice dressed up as a neutral one: "other" is the
# profile `targets.normalise_genre` already falls back to when nobody has said
# what kind of record this is, which is exactly the frame a defect score wants.
#
# It is needed because one defect's window is genre-relative in the detector:
# `limiter.over_driven` reads the PSR window and a distortion ceiling derived
# from it. On the same unmodified file that moved two things — the magnitude
# (0.24 tolerance units against trap, 2.85 against ambient, worth 30 points of
# technical) and, on `beat_tucked_hook`, whether the defect exists at all (no at
# trap, yes at folk). Both are the genre talking, and neither belongs in the one
# score that must not move with it. See `_counted_defects`.
_TECHNICAL_YARDSTICK = "other"

# Where the match crosses 50: the weighted distance at which a record is as much
# its own thing as it is the genre's. The mapping is `100 / (1 + (D/D50)^2)`.
#
# Shape matters more than the constant. Near zero it is flat, so a mix a hair
# outside one window does not stop being pop; through the middle it falls fast,
# which is where the wording actually changes; and it has a long tail, so two
# records 8 and 14 units out still order correctly instead of both reading zero.
# A straight line fails the first and third of those, and the third is the
# flattening this codebase has been bitten by twice.
#
# Fitted against the one thing that is not a matter of opinion: the same trap
# master measured against five genres, D = 0.50 / 1.30 / 1.95 / 3.29 / 3.93 for
# trap / pop / rock / folk / ambient. At 3.6 that reads 98 / 88 / 77 / 55 / 46 —
# close to its own reference, distinctly different from ambient, and no two
# adjacent genres indistinguishable.
_REFERENCE_HALF_MATCH_DISTANCE = 3.6

# Neutral, descriptive, and deliberately not a verdict. "Distinctly different"
# is a true thing to say about a record; "F" is not.
_REFERENCE_BANDS: Tuple[Tuple[float, str], ...] = (
    (90.0, "Sits close to the {genre} reference"),
    (70.0, "Recognisably {genre}, with its own character"),
    (50.0, "Departs from the {genre} reference in several places"),
)
_REFERENCE_BAND_FLOOR = "Distinctly different from the {genre} reference"

# Below this the reference label is worth putting in the headline of an
# otherwise clean mix — not as a problem, as the one other thing worth knowing.
_HEADLINE_REFERENCE_AT = 70.0


def _measured_out(finding: Finding) -> bool:
    """Did the measurement put this finding outside where it should be?

    True when it carries a verdict *or* a measured miss. See
    `_counts_against_the_score`, which is this plus the user's answer, for why
    it takes both.
    """
    return (
        str(finding.severity) != "clean"
        or _fin(getattr(finding, "miss_ratio", 0.0), 0.0) > 0.0
    )


def _counts_against_the_score(finding: Finding) -> bool:
    """Does this finding still put distance between the file and where it should be?

    Acknowledged is the one thing that removes it outright: the user was asked
    and said the choice was deliberate, so it stays in the report with all its
    numbers and stops being a distance to close.

    Otherwise it counts if it either carries a verdict or carries a measured
    miss, and both halves earn their place:

    * *A verdict without a miss* is an older payload — `miss_ratio` post-dates
      the first schema — and dropping those would silently score a clipped
      render from last month as flawless.
    * *A miss without a verdict* is `intent="reference"`, which strips every
      finding to an observation because nothing on somebody else's record is a
      work item. The distance is still real and still worth reporting: it is
      what the person measuring a record they admire came for. Keying purely on
      severity would hand them 100/100 and tell them nothing.

    Nothing else falls in that gap. Exactly one detector emits a genuinely
    positive observation — `vocal_balance.topline_headroom`, a beat's open
    centre — and it reports a miss of zero, so it is excluded by both halves.
    """
    return _measured_out(finding) and not bool(finding.acknowledged)


def _yardstick_distances(m: Measurements) -> Optional[Dict[str, float]]:
    """Defect id -> how far out it measures on the fixed ruler, or None if the pass failed.

    A second detector pass against `_TECHNICAL_YARDSTICK`, at `full_mix`, so
    neither the genre nor what the file is can change what counts as a defect or
    how far out it is. It costs ~1.4 ms and runs no DSP — the measurements are
    already in hand.

    None rather than an empty dict on failure, because those two mean opposite
    things: an empty ruler says the render is clean, and a scorer that cannot
    tell that from a crashed detector pass will happily print 100 over a clipped
    master.
    """
    try:
        neutral = detectors.detect_all(m, _TECHNICAL_YARDSTICK, "full_mix")
    except Exception:
        logger.exception("engine: the technical yardstick pass failed")
        return None
    return {
        str(f.id): _fin(getattr(f, "miss_ratio", 0.0), 0.0)
        for f in neutral
        if str(getattr(f, "kind", "deviation")) == "defect" and str(f.severity) != "clean"
    }


def _counted_defects(
    findings: Sequence[Finding], distances: Optional[Dict[str, float]]
) -> List[Finding]:
    """The defects `technical` charges for — and the number the headline states.

    Two tests, and the second one is the definition from `types.FindingKind`
    read literally: a defect is wrong *no matter the genre*, so if the
    genre-neutral ruler does not call it one, it never was one — that was the
    genre talking. `limiter.over_driven` is the case that makes this matter: its
    window comes from the genre's PSR range, so the same beat picks up a
    second "defect" at folk and ambient that it does not have at trap. Counting
    that would put the genre dropdown back inside the one score that must not
    move with it.

    The intersection only ever removes; nothing can be counted here that the
    report is not also showing, so the headline can never name a defect the user
    cannot find. With no ruler at all (the pass failed) it falls back to what the
    report says — degraded and genre-shaped, but not silent.

    **The asymmetry is a known hole, not a claim.** Because this is an
    intersection, it removes a defect the genre invented and it also removes one
    the genre *suppressed*: `reference_pop.wav` raises `limiter.over_driven` at
    pop and does not raise it at trap, so the ruler calls it a defect both times
    and this counts it only once — 93.7 and "2 defects" at pop, 95.0 and "1
    defect" at trap, on an unmodified file. Closing it properly means the
    detector emitting the defect arm genre-neutrally so the report can show what
    the ruler sees; until then, `technical` is genre-independent for every
    defect except `limiter.over_driven`, whose window is the only genre-relative
    one in the defect set.
    """
    shown = [
        f for f in findings or []
        if str(getattr(f, "kind", "deviation")) == "defect" and _counts_against_the_score(f)
    ]
    if distances is None:
        return shown
    return [f for f in shown if str(f.id) in distances]


def _defect_cost(finding: Finding, distances: Optional[Dict[str, float]]) -> float:
    """Points off `technical` for one defect, before the compounding discount.

    Continuous in the measured distance, via the same distance-to-points curve
    the dimension scorer uses — 1.0 unit out is 15 points, 1.6 is 34, 3.75 and
    beyond is 58 — then scaled by how much that kind of damage actually ruins a
    record. The weights are the ones `_HEALTH_WEIGHTS` already carries, so a
    polarity inversion (1.00) costs two and a half times a channel imbalance
    (0.40) at the same distance, which is the right ratio: one makes the mix
    disappear on a phone speaker, the other makes it lean right.

    Continuity is the point. `reference_trap` clips 309 samples and `mix_problem`
    clips 1,232 in runs four times as long; a severity table calls both of those
    "clipping" and charges them the same, and a technical score that cannot tell
    a hairline over from a squared-off master is not worth printing.
    """
    source = distances if distances is not None else {}
    ratio = _fin(
        source.get(str(finding.id), _fin(getattr(finding, "miss_ratio", 0.0), 0.0)), 0.0
    )
    if ratio <= 0.0:
        ratio = _DEFECT_UNMEASURED_RATIO
    weight = _HEALTH_WEIGHTS.get(str(finding.dimension), 0.6)
    return detectors.deviation_penalty(ratio) * weight


def _technical(findings: Sequence[Finding], distances: Optional[Dict[str, float]]) -> float:
    costs = sorted(
        (_defect_cost(f, distances) for f in _counted_defects(findings, distances)),
        reverse=True,
    )
    if not costs:
        return 100.0
    total = costs[0] + _DEFECT_EXTRA_SHARE * sum(costs[1:])
    return round(_clamp(100.0 - total, 0.0, 100.0), 1)


def technical_score(findings: Sequence[Finding], m: Measurements) -> float:
    """0-100 on defects alone. 100 means nothing measurably wrong with the render.

    The only findings it can see are the ones `detectors.finding_kind` calls
    defects — clipping, inter-sample overs, polarity inversion, mono
    cancellation, limiter over-drive, channel imbalance — and every one of those
    is wrong in every genre, on every record, in every year. Nothing about the
    spectrum reaches this number, and no file scores below 100 for sounding
    unlike something else. That is the whole fix: `mix_clean` has nothing wrong
    with it and was being told it was a D-.

    How far out each defect is comes from `_yardstick_distances` rather than
    from the report's own findings, because one defect's window is
    genre-relative. Across the whole fixture set at 24 genres x 6 intents, this
    figure does not move with the genre dropdown on any file — with one
    exception, `limiter.over_driven` on `reference_pop.wav`, which moves it 1.3
    points. `_counted_defects` documents why and what closing it would take.
    """
    return _technical(findings, _yardstick_distances(m))


def reference_match(findings: Sequence[Finding]) -> float:
    """0-100 on deviations alone. How close the track sits to its genre reference.

    **Not a quality judgement, and it never gets a grade.** A record can be a
    long way from the reference and be exactly what it was meant to be — that is
    most of what "having a sound" means.

    Distance is the L2 norm of the per-deviation misses, each in tolerance units
    and each weighted by how much that dimension defines the genre's identity.
    L2 rather than a sum because deviations are not independent charges to be
    added up: a record 5 units off in one place is further from the reference
    than one 1 unit off in five, and a plain sum says the opposite.

    Acknowledged deviations are gone from this entirely. Once the user has said
    a choice was deliberate, it is not a distance left to close — it is the
    record. That is what makes answering the questions change the number.
    """
    squared = 0.0
    for finding in findings or []:
        if str(getattr(finding, "kind", "deviation")) != "deviation":
            continue
        if not _counts_against_the_score(finding):
            continue
        weight = _HEALTH_WEIGHTS.get(str(finding.dimension), 0.6)
        squared += (weight * _fin(getattr(finding, "miss_ratio", 0.0), 0.0)) ** 2

    distance = math.sqrt(squared) / max(_REFERENCE_HALF_MATCH_DISTANCE, 1e-9)
    return round(_clamp(100.0 / (1.0 + distance * distance), 0.0, 100.0), 1)


def reference_label(match: float, genre: str) -> str:
    """The plain sentence for a match figure. Never a grade, never a verdict."""
    name = targets.get_profile(genre).label
    for threshold, template in _REFERENCE_BANDS:
        if match >= threshold:
            return template.format(genre=name)
    return _REFERENCE_BAND_FLOOR.format(genre=name)


def _blocker_clause(blocker: str) -> str:
    """One mastering blocker, cut down to something that fits in a headline."""
    text = str(blocker or "").strip()
    if text.startswith("Critical — "):
        text = text[len("Critical — "):]
    sentence = text.split(". ")[0].strip().rstrip(".").rstrip(":")
    sentence = " ".join(sentence.split())[:110]
    return sentence[:1].lower() + sentence[1:] if sentence else "one blocker outstanding"


def score_headline(
    defects: int,
    match: float,
    *,
    mastering_ready: bool,
    blockers: Sequence[str],
    genre: str,
    intent: str = "full_mix",
) -> str:
    """The one line worth reading, in priority order.

    Never a letter, never an "F". A grade is an answer to a question nobody
    asked; what a producer opening this needs is the next thing to do, and there
    are only three of those: fix the defects, clear the blocker, or go and
    master it.
    """
    if str(intent) == "reference":
        # Somebody else's finished record. There is no next thing to do.
        return "Measured as a reference — nothing here is a work item"

    if defects > 0:
        noun = "defect" if defects == 1 else "defects"
        return f"{defects} {noun} to fix before mastering"

    if not mastering_ready:
        return f"Not ready to master — {_blocker_clause(blockers[0] if blockers else '')}"

    if match < _HEADLINE_REFERENCE_AT:
        label = reference_label(match, genre)
        return f"Ready to master — {label[:1].lower()}{label[1:]}"
    return "Ready to master"


def build_score_card(
    findings: Sequence[Finding],
    m: Measurements,
    genre: str,
    *,
    mastering_ready: bool,
    blockers: Sequence[str] = (),
    intent: str = "full_mix",
) -> ScoreCard:
    """Compose the two scores, their wording, and the counts behind them."""
    distances = _yardstick_distances(m)
    defects = _counted_defects(findings, distances)

    # `deviations` counts every measured deviation on the report, including the
    # ones already acknowledged; `acknowledged` says how many of those the user
    # has confirmed. The outstanding count is the subtraction, and the total does
    # not shrink as questions get answered — the record still does that thing,
    # it is just no longer a thing to close.
    all_deviations = [
        f for f in findings or []
        if str(getattr(f, "kind", "deviation")) == "deviation" and _measured_out(f)
    ]
    acknowledged = [f for f in all_deviations if bool(f.acknowledged)]

    technical = _technical(findings, distances)
    match = reference_match(findings)

    return ScoreCard(
        technical=technical,
        technical_grade=grade_for(technical),
        reference_match=match,
        reference_label=reference_label(match, genre),
        headline=score_headline(
            len(defects), match,
            mastering_ready=mastering_ready,
            blockers=blockers,
            genre=genre,
            intent=intent,
        ),
        defects=len(defects),
        deviations=len(all_deviations),
        acknowledged=len(acknowledged),
    )


# ---------------------------------------------------------------------------
# Platform targets
# ---------------------------------------------------------------------------


def platform_targets(m: Measurements) -> List[PlatformTarget]:
    """Where this master lands against every platform's normalisation."""
    integrated = _fin(m.loudness.integrated_lufs, -70.0)
    true_peak = _fin(m.loudness.true_peak_dbtp, -120.0)

    out: List[PlatformTarget] = []
    for platform in targets.PLATFORMS:
        delta = integrated - platform.target_lufs
        turned_down = delta > _TURNDOWN_LU
        peak_ok = true_peak <= platform.max_true_peak_dbtp + 1e-9

        if not peak_ok:
            verdict = "problem"
        elif delta > 3.0:
            verdict = "problem"
        elif delta > _TURNDOWN_LU or delta < -4.0:
            verdict = "watch"
        else:
            verdict = "good"

        parts: List[str] = []
        if turned_down:
            parts.append(
                f"Played back {_num(delta, 1)} LU quieter than the file — the level gained "
                f"in the limiter is handed straight back, while the dynamics spent to get "
                f"it stay gone."
            )
        elif delta < -4.0:
            parts.append(
                f"{_num(abs(delta), 1)} LU under the {_num(platform.target_lufs, 1)} LUFS "
                f"reference and not turned up, so it will sit quiet against anything else "
                f"in a playlist."
            )
        else:
            parts.append(
                f"Lands within {_num(abs(delta), 1)} LU of the "
                f"{_num(platform.target_lufs, 1)} LUFS reference — plays back at its own level."
            )

        if peak_ok:
            parts.append(
                f"True peak {_num(true_peak, 2)} dBTP clears the "
                f"{_num(platform.max_true_peak_dbtp, 1)} dBTP ceiling."
            )
        else:
            parts.append(
                f"True peak {_num(true_peak, 2)} dBTP is "
                f"{_num(true_peak - platform.max_true_peak_dbtp, 2)} dB over the "
                f"{_num(platform.max_true_peak_dbtp, 1)} dBTP ceiling; the lossy transcode "
                f"will clip what the file does not."
            )
        parts.append(platform.note)

        out.append(PlatformTarget(
            platform=platform.name,
            target_lufs=round(_fin(platform.target_lufs), 2),
            max_true_peak_dbtp=round(_fin(platform.max_true_peak_dbtp), 2),
            delta_lufs=round(_fin(delta), 2),
            will_be_turned_down=bool(turned_down),
            peak_ok=bool(peak_ok),
            verdict=verdict,  # type: ignore[arg-type]
            note=" ".join(parts),
        ))
    return out


# ---------------------------------------------------------------------------
# Waveform
# ---------------------------------------------------------------------------


def waveform(buf: AudioBuffer, buckets: int = WAVEFORM_BUCKETS) -> Tuple[List[float], List[float]]:
    """Downsampled 0-1 peak and RMS envelopes for the timeline.

    Reshape-and-reduce, not a loop: a 6-minute stereo track is 17 M samples per
    channel, and the Python-level loop version of this took longer than every
    measurement in the analysis put together.
    """
    n = int(buf.n_samples)
    if n <= 0:
        return [], []

    peak_src = np.maximum(np.abs(buf.left), np.abs(buf.right))
    rms_src = buf.mono

    buckets = max(1, min(int(buckets), n))
    size = n // buckets
    if size < 1:
        size, buckets = 1, n
    usable = size * buckets

    peaks = peak_src[:usable].reshape(buckets, size).max(axis=1)
    rms = np.sqrt(np.square(rms_src[:usable].reshape(buckets, size)).mean(axis=1))

    # Both normalise against the same peak so the RMS trace sits inside the
    # peak trace, which is what a waveform view is supposed to show.
    norm = float(np.max(peaks)) if peaks.size else 0.0
    if not math.isfinite(norm) or norm <= 0.0:
        return [0.0] * int(buckets), [0.0] * int(buckets)

    peaks = np.clip(peaks / norm, 0.0, 1.0)
    rms = np.clip(rms / norm, 0.0, 1.0)
    peaks = np.nan_to_num(peaks, nan=0.0, posinf=1.0, neginf=0.0)
    rms = np.nan_to_num(rms, nan=0.0, posinf=1.0, neginf=0.0)
    return (
        [round(float(v), 3) for v in peaks],
        [round(float(v), 3) for v in rms],
    )


# ---------------------------------------------------------------------------
# Reference comparison
# ---------------------------------------------------------------------------

_GAP_LABELS: Dict[str, str] = {
    "sub": "sub (20-60 Hz)",
    "low_bass": "low bass (60-120 Hz)",
    "upper_bass": "upper bass (120-250 Hz)",
    "low_mid": "low mids (250-500 Hz)",
    "mid": "mids (500 Hz-1 kHz)",
    "upper_mid": "upper mids (1-2 kHz)",
    "presence": "presence (2-5 kHz)",
    "brilliance": "brilliance (5-10 kHz)",
    "air": "air (10-20 kHz)",
}


def _reference_similarity(
    toct_delta: np.ndarray, d_lufs: float, d_width: float, d_dr: float
) -> float:
    """0-100. Tonal balance dominates; level and width are secondary."""
    centers = np.asarray(core.THIRD_OCTAVE_CENTERS, dtype=np.float64)
    mask = (centers >= 40.0) & (centers <= 16_000.0)
    band = toct_delta[mask] if toct_delta.size == centers.size else toct_delta
    band = band[np.isfinite(band)] if band.size else band
    spectral_rms = float(np.sqrt(np.mean(np.square(band)))) if band.size else 0.0

    spectral_term = _clamp(100.0 - 6.0 * spectral_rms, 0.0, 100.0)
    loudness_term = _clamp(100.0 - 8.0 * abs(_fin(d_lufs)), 0.0, 100.0)
    width_term = _clamp(100.0 - 120.0 * abs(_fin(d_width)), 0.0, 100.0)
    dr_term = _clamp(100.0 - 6.0 * abs(_fin(d_dr)), 0.0, 100.0)

    return _clamp(
        0.55 * spectral_term + 0.18 * loudness_term + 0.09 * width_term + 0.18 * dr_term,
        0.0, 100.0,
    )


def build_reference_delta(mix: Measurements, ref: Measurements) -> ReferenceDelta:
    """Measured difference between the mix and an uploaded reference.

    Every level figure compared here is already normalised to the track's own
    1 kHz band, so a reference that is simply louder does not read as brighter.
    """
    d_lufs = _fin(mix.loudness.integrated_lufs) - _fin(ref.loudness.integrated_lufs)
    d_dr = _fin(mix.dynamics.crest_factor_db) - _fin(ref.dynamics.crest_factor_db)
    d_width = _fin(mix.stereo.width) - _fin(ref.stereo.width)
    d_tp = _fin(mix.loudness.true_peak_dbtp) - _fin(ref.loudness.true_peak_dbtp)

    mix_bands = {b.name: _fin(b.level_db) for b in mix.spectral.bands}
    ref_bands = {b.name: _fin(b.level_db) for b in ref.spectral.bands}
    band_deltas = {
        name: round(mix_bands[name] - ref_bands.get(name, mix_bands[name]), 2)
        for name in mix_bands
    }

    mix_toct = np.asarray([_fin(v) for v in mix.spectral.third_octave_db], dtype=np.float64)
    ref_toct = np.asarray([_fin(v) for v in ref.spectral.third_octave_db], dtype=np.float64)
    if mix_toct.size and mix_toct.size == ref_toct.size:
        toct_delta = mix_toct - ref_toct
    else:
        toct_delta = np.zeros(len(core.THIRD_OCTAVE_CENTERS), dtype=np.float64)

    similarity = _reference_similarity(toct_delta, d_lufs, d_width, d_dr)

    gaps: List[str] = []
    for name, delta in sorted(band_deltas.items(), key=lambda kv: -abs(kv[1]))[:4]:
        if abs(delta) < 1.0:
            continue
        label = _GAP_LABELS.get(name, name.replace("_", " "))
        direction = "above" if delta > 0 else "below"
        gaps.append(
            f"Your {label} sits {_num(abs(delta), 1)} dB {direction} the reference."
        )
    if abs(d_lufs) >= 1.0:
        gaps.append(
            f"Integrated loudness is {_num(abs(d_lufs), 1)} LU "
            f"{'louder' if d_lufs > 0 else 'quieter'} than the reference "
            f"({_num(mix.loudness.integrated_lufs, 1)} vs "
            f"{_num(ref.loudness.integrated_lufs, 1)} LUFS)."
        )
    if abs(d_dr) >= 1.5:
        gaps.append(
            f"Crest factor is {_num(abs(d_dr), 1)} dB "
            f"{'wider' if d_dr > 0 else 'flatter'} than the reference "
            f"({_num(mix.dynamics.crest_factor_db, 1)} vs "
            f"{_num(ref.dynamics.crest_factor_db, 1)} dB) — "
            + ("the reference is worked harder than this mix."
               if d_dr > 0 else "this mix is worked harder than the reference.")
        )
    if abs(d_width) >= 0.08:
        gaps.append(
            f"Stereo width is {_num(abs(d_width), 2)} "
            f"{'wider' if d_width > 0 else 'narrower'} than the reference "
            f"(Side/Mid {_num(mix.stereo.width, 2)} vs {_num(ref.stereo.width, 2)})."
        )
    if not gaps:
        gaps.append(
            "Nothing separates this mix from the reference by more than a decibel — "
            "tonal balance, level and width all match."
        )

    return ReferenceDelta(
        integrated_lufs=round(d_lufs, 2),
        dynamic_range_db=round(d_dr, 2),
        stereo_width=round(d_width, 3),
        true_peak_dbtp=round(d_tp, 2),
        band_deltas=band_deltas,
        third_octave_delta_db=[round(float(v), 2) for v in toct_delta],
        similarity=round(similarity, 1),
        biggest_gaps=gaps[:6],
    )


# ---------------------------------------------------------------------------
# AI layer
# ---------------------------------------------------------------------------


def _plugin_names(user_plugins: Optional[Sequence[Any]]) -> List[str]:
    """Flatten whatever the caller has (ORM rows, dicts, strings) into labels."""
    names: List[str] = []
    for item in user_plugins or []:
        if item is None:
            continue
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            maker = str(item.get("manufacturer") or "").strip()
            name = str(item.get("name") or "").strip()
            category = str(item.get("category") or "").strip()
            text = " ".join(p for p in (maker, name) if p)
            if category:
                text = f"{text} ({category})" if text else category
        else:
            maker = str(getattr(item, "manufacturer", "") or "").strip()
            name = str(getattr(item, "name", "") or "").strip()
            category = str(getattr(item, "category", "") or "").strip()
            text = " ".join(p for p in (maker, name) if p)
            if category:
                text = f"{text} ({category})" if text else category
        if text:
            names.append(text)
    return names[:120]


def _ai_available() -> bool:
    key = os.environ.get("ANTHROPIC_API_KEY") or ""
    return bool(key.strip())


def _consult(
    analysis: MixAnalysis,
    genre: str,
    notes: Optional[str],
    plugins: List[str],
) -> Optional[EngineerReport]:
    """Call the AI layer. Any failure here costs prose, never the report."""
    import engineer as engineer_module  # top-level module in backend/

    ctx = engineer_module.EngineerContext(
        measurements=analysis.measurements,
        findings=list(analysis.findings),
        dimensions=list(analysis.dimensions),
        genre=genre,
        # The detectors have already gated on intent; the write-up has to know
        # too, or it prescribes a topline for a beat the findings deliberately
        # stayed quiet about.
        intent=str(analysis.intent),
        platform_targets=list(analysis.platform_targets),
        reference=analysis.reference,
        user_notes=notes,
        plugins=plugins,
        filename=analysis.filename,
        health_score=analysis.health_score,
        grade=analysis.grade,
        # The split score is what the brief leads with; the composite above is
        # only still passed for a caller that has not been updated.
        scores=analysis.scores,
        mastering_ready=analysis.mastering_ready,
        mastering_blockers=list(analysis.mastering_blockers),
    )
    return engineer_module.consult_engineer(ctx)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def analyze_mix_detailed(
    path: str,
    genre: str,
    *,
    intent: str = "full_mix",
    filename: str = "",
    notes: Optional[str] = None,
    reference_path: Optional[str] = None,
    user_plugins: Optional[List[Any]] = None,
    run_ai: bool = True,
    separate_stems: bool = False,
    separation_timeout_s: Optional[float] = None,
) -> Tuple[MixAnalysis, Dict[str, float]]:
    """`analyze_mix`, plus the per-stage timing map. Used for profiling."""
    started = time.perf_counter()
    timings: Dict[str, float] = {}
    warnings: List[str] = []

    genre_key = targets.normalise_genre(genre)
    intent_key = detectors._normalise_intent(intent)

    # 1. Decode once. Everything below shares this buffer.
    t0 = time.perf_counter()
    buf = load_or_explain(path, "audio file")
    timings["load"] = round((time.perf_counter() - t0) * 1000.0, 1)

    if buf.is_mono:
        warnings.append(
            "This file is mono. Stereo width, phase and vocal balance need two "
            "channels to measure, so those dimensions are reported as unassessed "
            "rather than scored."
        )
    if buf.duration < 3.0:
        warnings.append(
            f"This file is {buf.duration:.1f} s. Loudness, dynamics and transient "
            f"statistics need several seconds of programme to settle, so the "
            f"dimensions that depend on them are not scored."
        )

    # 2. Measure.
    measurements = measure_all(buf, genre_key, timings, warnings)

    # 2b. Optional depth: separate the sources and measure each on its own.
    #     Off by default, because it is the one stage that turns a ~3 s request
    #     into a ~10 s one. It runs after the pool rather than on it: the
    #     separator saturates every core by itself, and handing it a worker
    #     alongside eight numpy jobs makes both slower.
    if separate_stems:
        measurements.stems = measure_stems_stage(
            buf, genre_key, timings, warnings, timeout_s=separation_timeout_s
        )

    # 3. Judge.
    t0 = time.perf_counter()
    findings = detectors.detect_all(measurements, genre_key, intent_key)
    dimensions = detectors.score_dimensions(
        findings, measurements, genre_key, intent_key
    )
    _mark_unassessed(dimensions, timings, warnings)
    timings["detect"] = round((time.perf_counter() - t0) * 1000.0, 1)

    # 4. Roll up. `health`/`grade` are the legacy composite and are kept as they
    #    were; `scores` is the split the report actually leads with.
    t0 = time.perf_counter()
    health = compute_health(dimensions, findings)
    ceiling = compute_ceiling(health, findings)
    ready, blockers = mastering_readiness(measurements, findings, intent_key)
    scores = build_score_card(
        findings, measurements, genre_key,
        mastering_ready=ready, blockers=blockers, intent=intent_key,
    )
    platforms = platform_targets(measurements)
    timings["score"] = round((time.perf_counter() - t0) * 1000.0, 1)

    # 5. Waveform.
    t0 = time.perf_counter()
    peaks, rms = waveform(buf)
    timings["waveform"] = round((time.perf_counter() - t0) * 1000.0, 1)

    # 6. Reference, if one was uploaded.
    reference: Optional[ReferenceDelta] = None
    if reference_path:
        t0 = time.perf_counter()
        try:
            ref_buf = load_or_explain(reference_path, "reference track")
            ref_measurements = measure_reference(
                ref_buf, genre_key, timings, [],
                tempo_hint=_fin(measurements.transients.estimated_tempo, 0.0),
            )
            reference = build_reference_delta(measurements, ref_measurements)
        except Exception as exc:
            logger.exception("engine: reference analysis failed")
            warnings.append(
                f"The reference track could not be analysed ({exc}); the rest of the "
                f"report is unaffected."
            )
        timings["reference"] = round((time.perf_counter() - t0) * 1000.0, 1)

    analysis = MixAnalysis(
        filename=filename or os.path.basename(path),
        genre=genre_key,
        intent=intent_key,
        health_score=health,
        grade=grade_for(health),
        scores=scores,
        ceiling_score=ceiling,
        mastering_ready=ready,
        mastering_blockers=blockers,
        dimensions=dimensions,
        findings=findings,
        measurements=measurements,
        platform_targets=platforms,
        reference=reference,
        engineer=None,
        waveform_peaks=peaks,
        waveform_rms=rms,
        analysis_ms=0,
        warnings=warnings,
    )

    # 7. The AI layer, last, and never fatal.
    if run_ai:
        t0 = time.perf_counter()
        if not _ai_available():
            warnings.append(
                "ANTHROPIC_API_KEY is not configured, so the engineer's write-up was "
                "skipped. Every finding below is measured and stands on its own."
            )
        else:
            try:
                report = _consult(analysis, genre_key, notes, _plugin_names(user_plugins))
                if report is None:
                    warnings.append(
                        "The engineer's write-up was unavailable for this run. Every "
                        "finding below is measured and stands on its own."
                    )
                analysis.engineer = report
            except Exception as exc:
                logger.exception("engine: the AI layer failed")
                warnings.append(
                    f"The engineer's write-up failed ({type(exc).__name__}); the measured "
                    f"findings below are unaffected."
                )
                analysis.engineer = None
        timings["engineer"] = round((time.perf_counter() - t0) * 1000.0, 1)

    analysis.warnings = warnings
    analysis.analysis_ms = int(round((time.perf_counter() - started) * 1000.0))
    timings["total"] = float(analysis.analysis_ms)

    # 8. Last line of defence before the wire.
    clean = MixAnalysis.model_validate(_sanitize(analysis.model_dump()))
    card = clean.scores
    logger.info(
        "engine: %s (%s) -> tech %.0f (%s) / ref %.0f, legacy %.1f (%s) in %d ms | %s",
        clean.filename, clean.genre,
        card.technical if card else -1.0, card.technical_grade if card else "?",
        card.reference_match if card else -1.0,
        clean.health_score, clean.grade, clean.analysis_ms,
        " ".join(f"{k}={v:.0f}ms" for k, v in timings.items()),
    )
    return clean, timings


def analyze_mix(
    path: str,
    genre: str,
    *,
    intent: str = "full_mix",
    filename: str = "",
    notes: Optional[str] = None,
    reference_path: Optional[str] = None,
    user_plugins: Optional[List[Any]] = None,
    run_ai: bool = True,
    separate_stems: bool = False,
    separation_timeout_s: Optional[float] = None,
) -> MixAnalysis:
    """Analyse one mix end to end.

    Raises `core.AudioTooShortError`, `core.SilentAudioError` or `ValueError`
    for a file that cannot be analysed at all — the API turns those into a 422
    carrying the message. Everything else is caught and downgraded to a warning
    on the returned report.

    `genre` sets the reference every stylistic measurement is compared against;
    `intent` sets which measurements are worth comparing at all. They are
    different questions and the second one is the one that stops a beat being
    told its hi-hats are vocal sibilance: a `beat` is an instrumental built for
    somebody else's topline, so its absent lead is correct, its open mid-range
    is the brief, and its 5-9 kHz burstiness is percussion. See
    `types.TrackIntent` for what each value means and `detectors.detect_all`
    for what each one changes.

    `separate_stems` adds the per-source pass: Demucs splits the mix into
    vocals/drums/bass/other and each source is measured on its own. That turns
    four inferences into measurements — vocal level against the actual
    instrumental rather than a centre proxy, kick against 808 as two objects
    rather than one waveform, per-element compression, and source-against-source
    masking — and the findings built on them carry visibly higher confidence.

    It is off by default because it is expensive, and it is the one stage that
    does not parallelise with anything. Measured on Apple Silicon with MPS: a
    20 s clip goes from 1.9 s to 5.4 s, a four-minute track from 10.2 s to
    52.4 s, and a 5:16 track from 12.5 s to 69.0 s — roughly 9 s of separation
    per minute of audio, several times that on a CPU-only host, against ~3 s
    for the whole default analysis of a short clip.

    It is bounded and optional at every point: a missing model, no network on
    the first run, no GPU, or a blown `separation_timeout_s` all come back as
    `measurements.stems.available == False` plus a warning, and every other
    number in the report is unaffected.
    """
    analysis, _ = analyze_mix_detailed(
        path,
        genre,
        intent=intent,
        filename=filename,
        notes=notes,
        reference_path=reference_path,
        user_plugins=user_plugins,
        run_ai=run_ai,
        separate_stems=separate_stems,
        separation_timeout_s=separation_timeout_s,
    )
    return analysis


# ---------------------------------------------------------------------------
# Reassessment
# ---------------------------------------------------------------------------


def _answerable_ids(findings: Sequence[Finding]) -> set:
    """The findings a user is allowed to answer for.

    A deviation that was actually asked about — it carries a `Clarification` —
    and nothing else. This is a guard, not a formality: the reassess endpoint is
    stateless, so the analysis and the answers both arrive from the client, and
    without this a posted `{"finding_id": "clipping.hard_clipping", "intended":
    true}` would acknowledge a defect straight out of the technical score.
    Clipping is not a decision anybody made, so there is no yes to give.
    """
    return {
        str(f.id) for f in findings or []
        if f.clarification is not None
        and str(getattr(f, "kind", "deviation")) == "deviation"
    }


def apply_answers(
    analysis: MixAnalysis, answers: Sequence[ClarificationAnswer]
) -> MixAnalysis:
    """Fold the user's answers into an existing report and re-score it.

    **No DSP runs.** Nothing about the audio changed — the mix is 5.2 dB thin at
    2 kHz whether or not that was the plan — so every measurement is carried
    through untouched and only the judgement on top of them moves. That is what
    keeps this a few milliseconds instead of a few seconds, and it is also why
    it is honest: an answer cannot alter a number, only what the report makes of
    one.

    What moves: the acknowledged findings stop pulling on their dimension, so
    `dimensions`, the legacy composite and `scores.reference_match` all update,
    and the headline can change with them. What does not move: `technical`,
    unless a defect somehow got acknowledged — which `_answerable_ids` makes
    impossible — and `mastering_ready`, because only defects and criticals can
    block a master and neither is answerable. Saying "yes, the intro is meant to
    be thin" is not allowed to make a clipped file ready to master.

    The input is left alone; the updated report is a copy.
    """
    updated = analysis.model_copy(deep=True)

    allowed = _answerable_ids(updated.findings)
    accepted = [a for a in answers or [] if str(a.finding_id) in allowed]
    rejected = [a for a in answers or [] if str(a.finding_id) not in allowed]
    if rejected:
        logger.info(
            "engine: ignoring %d answer(s) for findings that were never asked about: %s",
            len(rejected), ", ".join(sorted({str(a.finding_id) for a in rejected}))[:200],
        )
    clarify.apply_answers(updated.findings, accepted)

    # Acknowledged findings keep their place in the report — with their numbers,
    # their question and now an answer — and come out of the scoring input. The
    # alternative, leaving them in with the penalty zeroed, means two places that
    # have to agree about what a yes is worth.
    live = [f for f in updated.findings if not f.acknowledged]

    dimensions = detectors.score_dimensions(
        live, updated.measurements, updated.genre, str(updated.intent)
    )
    _mark_unassessed(dimensions, {}, list(updated.warnings))
    updated.dimensions = dimensions

    health = compute_health(dimensions, live)
    updated.health_score = health
    updated.grade = grade_for(health)
    updated.ceiling_score = compute_ceiling(health, live)
    updated.scores = build_score_card(
        updated.findings, updated.measurements, updated.genre,
        mastering_ready=updated.mastering_ready,
        blockers=updated.mastering_blockers,
        intent=str(updated.intent),
    )
    return updated
