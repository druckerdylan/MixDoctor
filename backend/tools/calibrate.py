#!/usr/bin/env python3
"""Fit MixDoctor's genre targets to a corpus of real commercial masters.

The honest weakness of this product is `analysis/targets.py`. Every number in it
— the 1/3-octave anchor curves, the loudness windows, the mud and harshness caps
— was hand-set from engineering practice. That is a defensible starting point and
it is not the same thing as *measured*. A hand-set curve encodes what a good
engineer believes trap should look like; it does not encode what the trap records
people actually release look like, and where those two disagree the product is
confidently telling producers to move toward a number nobody is hitting.

This tool closes that gap for any operator who has a library of masters:

    python tools/calibrate.py --genre trap --input ~/Music/trap-references/

It walks the directory, measures every usable file with the *existing* DSP layer
(`analysis.engine.measure_reference` — nothing here reimplements measurement),
aggregates the corpus, and writes a markdown report that puts the fitted values
next to the hand-set ones and flags every place the hand-set value falls outside
the corpus interquartile range. Those flags are the list of things that are wrong.

`--apply` additionally writes `analysis/targets_fitted.py`: a generated, importable
module holding the fitted anchors in the exact tuple-of-tuples format `targets.py`
uses, plus a ready-to-paste `_register(...)` call. It never touches `targets.py`
itself. Replacing a hand-set opinion with a fitted one is a decision a human makes
by reading the report, not a decision a script makes by overwriting a file.

Design notes
------------
* **Reuse, never reimplement.** The per-file measurement is exactly the four
  measurements `measure_reference` runs for the reference-track comparison —
  spectral, loudness, stereo, dynamics — which is precisely the set the fitted
  values are drawn from. If the DSP changes, the calibration changes with it,
  which is the only way the fitted numbers stay comparable with the live ones.
* **Median, not mean.** One badly-mastered file in a corpus of thirty moves a mean
  and does not move a median. The interquartile range is reported alongside so the
  operator can see how much the corpus actually agrees with itself.
* **Refuse small corpora.** Below `--min-files` (default 5) the tool stops. A
  "fitted" curve from two tracks is worse than the hand-set one, because it looks
  empirical.
* **Process pool, not threads.** Unlike the request path — where the measurements
  share one decoded buffer and numpy drops the GIL — here every file is an
  independent decode plus measurement, so processes win outright.

Known measurement bias
----------------------
`core.stft_power` runs an 8192-point FFT at 48 kHz: 5.86 Hz bins. A 1/3-octave
band's level is the *summed* power of the bins inside it, so a band only a few bins
wide reads hot or shy purely because a bin count is an integer and a band width is
not. `_grid_bias_db()` derives the size of that effect from the grid rather than
guessing at it: +2.06 dB at 31.5 Hz, -1.99 dB at 40 Hz, -0.92 dB at 125 Hz, and
under 0.01 dB from 160 Hz up. `analysis/dsp/spectral.py` documents the same effect
qualitatively ("~2 dB" in the lowest three bands); the derivation here also catches
125 Hz, which the prose does not mention and which `--self-test` measured at
-0.9 dB before the model existed to explain it.

Every file carries the identical bias, so it never widens the interquartile range —
it offsets the fitted median of six of the 31 bands by a fixed, known amount. The
report prints that amount next to each affected band, and `--self-test` holds those
bands to a separate, looser tolerance instead of pretending they pass.
"""

from __future__ import annotations

# Set before numpy/scipy are imported anywhere. Each pool worker measures one
# file on one core; letting BLAS spin up ten threads per worker on a ten-core
# machine oversubscribes it by 10x and runs slower than serial.
import os as _os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    _os.environ.setdefault(_var, "1")

# Third-party import noise that would otherwise be printed once per pool worker
# and drown the progress output. Both are known and neither is actionable here.
import warnings as _warnings

_warnings.filterwarnings("ignore", message=r'Field "model_\w+" has conflict.*')
_warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated.*")
# pydub complains once per undecodable file; the skip line already says it better.
_warnings.filterwarnings("ignore", message=r".*ffprobe or avprobe.*")

import argparse
import datetime
import importlib.util
import math
import os
import shutil
import sys
import tempfile
import textwrap
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# `python tools/calibrate.py` from anywhere, and `python -m tools.calibrate`,
# both have to resolve `import analysis...`. This runs at module scope on
# purpose: spawn-based pool workers re-import this file and need it too.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from analysis import core, targets  # noqa: E402
from analysis.core import SUPPORTED_FORMATS  # noqa: E402

__all__ = [
    "FileResult",
    "CorpusFit",
    "discover_audio",
    "measure_file",
    "fit_corpus",
    "render_report",
    "render_fitted_module",
    "main",
]

TOOL_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Corpus admission rules
# ---------------------------------------------------------------------------

# A commercial master is a finished record. Anything under a minute is an
# interlude, a stem bounce, a loop export or a preview clip, and none of those
# carry a representative long-term spectrum.
DEFAULT_MIN_DURATION_SEC = 60.0

# Fewer than this and the median is not a median, it is an anecdote.
DEFAULT_MIN_FILES = 5

# Analysis is capped at the same 10 minutes the request path uses, so a fitted
# curve is drawn from the same window of audio a user's upload would be.
MAX_SECONDS = 600.0

# Quieter than this and the file is a rough, a stem, or an unmastered bounce.
# The most conservative genre window in targets.py is classical at -23 LUFS.
NOT_A_MASTER_LUFS = -35.0

# Side/Mid below this is a dual-mono file: two identical channels, no stereo
# information. It decodes as stereo so `is_mono` never catches it.
DUAL_MONO_WIDTH = 0.005

# `core.stft_power`'s default transform, mirrored here so the grid bias below can
# be derived rather than guessed. If core ever changes it, this recomputes to
# match — the constant is the only thing that has to be kept in step.
_STFT_N_FFT = 8192

# A band whose predicted bias exceeds this is reported separately: at that point
# the number describes the FFT grid more than it describes the record.
GRID_BIAS_THRESHOLD_DB = 0.5

# --self-test tolerances: what the harness must recover from files synthesised
# to sit exactly on a known curve.
SELF_TEST_TOL_DB = 1.0
SELF_TEST_GRID_TOL_DB = 3.5

# How far outside the corpus median a hand-set band value must sit before the
# tool is willing to call it wrong — a half-width, applied symmetrically.
#
# A corpus can agree with itself far more tightly than this tool can measure it.
# At the 1 kHz normalisation anchor the IQR collapses to hundredths of a decibel
# *by construction* — that is where the 0 dB reference is — so a strict "is the
# hand-set value inside the IQR" test flags every band near 1 kHz whose hand-set
# value is not the median to two decimal places. That is noise dressed as a
# finding. The test interval is therefore widened symmetrically about the median
# to at least +/- this much before the comparison.
#
# The value is the harness's own per-band accuracy, which `--self-test` measures
# and holds itself to (`SELF_TEST_TOL_DB`, a one-sided bound on |error|). A
# disagreement smaller than the tool's own error is not evidence about anything.
BAND_AGREEMENT_FLOOR_DB = SELF_TEST_TOL_DB


def _grid_bias_db() -> Dict[float, float]:
    """Predicted per-band bias of the 1/3-octave measurement, from the FFT grid alone.

    `spectral.py` defines a band's level as the *summed* power of the FFT bins
    whose centres fall inside it. For a locally flat power density that sum is
    proportional to the bin count, and the bin count is an integer while the band
    width is not — so a band 1.25 bins wide that captures 2 bins reads 2.06 dB
    hot, and one 1.58 bins wide that captures 1 reads 1.99 dB shy. Nothing about
    the audio changes that; it is arithmetic on the grid.

    Below 160 Hz the ISO bands are narrower than a handful of 5.86 Hz bins and
    the rounding dominates. Above it there are dozens of bins per band and the
    error falls below a hundredth of a decibel.

    The bias is identical for every file measured through this analyser, so it
    never widens the corpus IQR — it offsets the fitted median of the affected
    bands by a fixed amount, which is why the report and the self-test hold those
    bands to a different standard instead of silently averaging the error in.
    """
    freqs = np.fft.rfftfreq(_STFT_N_FFT, 1.0 / core.ANALYSIS_SR)
    spacing = float(freqs[1] - freqs[0])
    out: Dict[float, float] = {}
    for center, (lo, hi) in zip(core.THIRD_OCTAVE_CENTERS, core.band_edges()):
        actual = int(np.count_nonzero((freqs >= lo) & (freqs < hi)))
        expected = max((hi - lo) / spacing, 1e-9)
        # 0 bins is the `_patch_empty` case: spectral.py assigns the nearest bin
        # rather than reporting silence, so the effective count is 1.
        out[float(center)] = 10.0 * math.log10(max(actual, 1) / expected)
    return out


GRID_BIAS_DB: Dict[float, float] = _grid_bias_db()

# Bands the fitted curve cannot speak for. Two sources, both structural:
#   * FFT-grid quantisation, above — 20, 31.5, 40, 63 and 125 Hz at 8192/48 kHz.
#   * The two ends of the spectrum, where `targets.target_curve()` clamps: the
#     20 Hz and 20 kHz bands extend past the outermost anchor, so the hand-set
#     value there is an extrapolation and the fitted one is not comparable to it.
GRID_LIMITED_HZ: Tuple[float, ...] = tuple(sorted(
    {hz for hz, bias in GRID_BIAS_DB.items() if abs(bias) >= GRID_BIAS_THRESHOLD_DB}
    | {float(core.THIRD_OCTAVE_CENTERS[0]), float(core.THIRD_OCTAVE_CENTERS[-1])}
))


# ---------------------------------------------------------------------------
# What gets fitted
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalarSpec:
    """One scalar the corpus is fitted for, and how to compare it to targets.py.

    `kind` decides both the comparison and the suggested replacement:

        window   profile holds a (low, high) "this is fine" range
        min      profile holds a floor the mix must stay above
        max      profile holds a cap the mix must stay under
        ceiling  no profile field; compared against a fixed delivery limit
    """

    key: str
    label: str
    unit: str
    kind: str
    profile_attr: Optional[str] = None
    ceiling: Optional[float] = None
    register_kwarg: Optional[str] = None
    nd: int = 2
    higher_is_hotter: bool = True


SCALAR_SPECS: Tuple[ScalarSpec, ...] = (
    ScalarSpec("integrated_lufs", "Integrated loudness", "LUFS", "window",
               "integrated_lufs", register_kwarg="lufs", nd=1),
    ScalarSpec("loudness_range_lu", "Loudness range (LRA)", "LU", "window",
               "loudness_range_lu", register_kwarg="lra", nd=1),
    ScalarSpec("true_peak_dbtp", "True peak", "dBTP", "ceiling",
               None, ceiling=-1.0, nd=2),
    ScalarSpec("crest_factor_db", "Crest factor", "dB", "window",
               "crest_factor_db", register_kwarg="crest", nd=1),
    ScalarSpec("micro_dynamics_db", "Micro-dynamics (50 ms)", "dB", "window",
               "micro_dynamics_db", register_kwarg="micro", nd=1),
    ScalarSpec("psr_p10_db", "PSR p10", "dB", "window",
               "psr_p10_db", register_kwarg="psr", nd=1),
    ScalarSpec("stereo_width", "Stereo width (Side/Mid)", "", "window",
               "stereo_width", register_kwarg="width", nd=3),
    ScalarSpec("correlation", "Correlation", "", "min",
               "correlation_min", register_kwarg="corr_min", nd=3),
    ScalarSpec("mud_ratio_db", "Mud ratio (150-400 vs 60-120 Hz)", "dB", "window",
               "mud_ratio_db", register_kwarg="mud_ratio_db", nd=1),
    ScalarSpec("harshness_index", "Harshness index", "", "max",
               "harshness_max", register_kwarg="harshness_max", nd=3),
)

# The anchor frequencies every curve in targets.py is written at. All fifteen
# are ISO 1/3-octave centres, so each one reads straight off the fitted curve
# with no interpolation on the way in.
ANCHOR_HZ: Tuple[float, ...] = (
    20.0, 40.0, 63.0, 100.0, 160.0, 250.0, 400.0, 630.0, 1000.0,
    2000.0, 3150.0, 5000.0, 8000.0, 12500.0, 20000.0,
)


def _is_grid_limited(hz: float) -> bool:
    return any(abs(float(hz) - g) < 1e-6 for g in GRID_LIMITED_HZ)


# Four of those fifteen anchors (20, 40, 63 Hz and 20 kHz) sit on grid-limited
# bands, and an anchor carries its own bias into every band it interpolates
# across — so the 25 Hz band reads correctly but the *anchor-reduced* curve is
# ~1.9 dB wrong there, inherited from its neighbours at 20 and 40 Hz. The span
# below is where every bracketing anchor is clean, and is therefore the only
# region where the anchor format can be held to account for its own fidelity.
_CLEAN_ANCHORS: Tuple[float, ...] = tuple(
    hz for hz in ANCHOR_HZ if not _is_grid_limited(hz)
)
CLEAN_ANCHOR_SPAN_HZ: Tuple[float, float] = (
    (min(_CLEAN_ANCHORS), max(_CLEAN_ANCHORS)) if _CLEAN_ANCHORS else (0.0, 0.0)
)


def _clean_anchor_mask(centers: np.ndarray) -> np.ndarray:
    lo, hi = CLEAN_ANCHOR_SPAN_HZ
    return (centers >= lo - 1e-6) & (centers <= hi + 1e-6)


# ---------------------------------------------------------------------------
# Per-file result (crosses a process boundary — keep it plain and small)
# ---------------------------------------------------------------------------


@dataclass
class FileResult:
    path: str
    name: str
    ok: bool
    skip_reason: str = ""
    duration_sec: float = 0.0
    elapsed_ms: int = 0
    third_octave_db: List[float] = field(default_factory=list)
    scalars: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def _f(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def measure_file(
    path: str,
    genre_key: str,
    min_duration: float = DEFAULT_MIN_DURATION_SEC,
    require_stereo: bool = True,
) -> FileResult:
    """Decode and measure one candidate. Never raises — every failure is a skip.

    Runs at module scope so a spawn-based `ProcessPoolExecutor` can pickle it by
    name. Returns plain floats rather than the pydantic models so the object
    crossing the pipe back to the parent is a few hundred bytes, not a serialised
    `Measurements` with every time series in it.
    """
    name = os.path.basename(path)
    started = time.perf_counter()

    def _skip(reason: str, duration: float = 0.0) -> FileResult:
        return FileResult(
            path=path, name=name, ok=False, skip_reason=reason,
            duration_sec=round(duration, 2),
            elapsed_ms=int(round((time.perf_counter() - started) * 1000.0)),
        )

    try:
        buf = core.load_audio(path, max_seconds=MAX_SECONDS)
    except core.AudioTooShortError:
        return _skip("too short to decode")
    except core.SilentAudioError:
        return _skip("silent or near-silent")
    except ValueError as exc:
        return _skip(f"could not decode: {exc}")
    except FileNotFoundError:
        # Not the corpus file: libsndfile refused the container, the decode fell
        # through to pydub, and pydub shells out to ffmpeg/ffprobe — which are not
        # on this host's PATH. Naming the exception here would say the master is
        # missing, which is both wrong and unactionable. Same reasoning as
        # `engine.load_or_explain`.
        return _skip("libsndfile cannot open this container and ffmpeg is not "
                     "installed, so the fallback decoder is unavailable")
    except Exception as exc:  # noqa: BLE001 — a corpus is other people's files
        return _skip(f"could not decode ({type(exc).__name__}): {exc}")

    if buf.duration < min_duration:
        return _skip(
            f"{buf.duration:.1f} s, under the {min_duration:.0f} s minimum "
            f"(not a full master)",
            buf.duration,
        )
    if require_stereo and buf.is_mono:
        return _skip("mono file, and this genre is mastered in stereo", buf.duration)

    # The exact four measurements the fitted values are read off. workers=1:
    # this process is already one slot in the pool, and nesting a thread pool
    # inside it only fights the other workers for cores.
    from analysis.engine import measure_reference  # local: keeps parent import cheap

    warnings: List[str] = []
    try:
        m = measure_reference(buf, genre_key, {}, warnings, workers=1, tempo_hint=None)
    except Exception as exc:  # noqa: BLE001
        return _skip(f"measurement failed ({type(exc).__name__}): {exc}", buf.duration)

    if any("ref_spectral" in w for w in warnings):
        return _skip("the spectral measurement failed on this file", buf.duration)

    curve = np.asarray([_f(v) for v in m.spectral.third_octave_db], dtype=np.float64)
    if curve.size != len(core.THIRD_OCTAVE_CENTERS) or not np.all(np.isfinite(curve)):
        return _skip("spectral measurement returned an unusable curve", buf.duration)

    width = _f(m.stereo.width, 0.0)
    if require_stereo and not buf.is_mono and width < DUAL_MONO_WIDTH:
        return _skip(
            f"dual-mono (Side/Mid {width:.4f}) — two identical channels, "
            f"no stereo information",
            buf.duration,
        )

    lufs = _f(m.loudness.integrated_lufs)
    if math.isfinite(lufs) and lufs < NOT_A_MASTER_LUFS:
        return _skip(
            f"{lufs:.1f} LUFS — far too quiet to be a released master "
            f"(stem, rough or unmastered bounce)",
            buf.duration,
        )

    scalars = {
        "integrated_lufs": lufs,
        "loudness_range_lu": _f(m.loudness.loudness_range_lu),
        "true_peak_dbtp": _f(m.loudness.true_peak_dbtp),
        "crest_factor_db": _f(m.dynamics.crest_factor_db),
        "micro_dynamics_db": _f(m.dynamics.micro_dynamics_db),
        "psr_p10_db": _f(m.loudness.psr_p10_db),
        "stereo_width": width,
        "correlation": _f(m.stereo.correlation),
        "mud_ratio_db": _f(m.spectral.mud_ratio_db),
        "harshness_index": _f(m.spectral.harshness_index),
    }

    return FileResult(
        path=path,
        name=name,
        ok=True,
        duration_sec=round(buf.duration, 2),
        elapsed_ms=int(round((time.perf_counter() - started) * 1000.0)),
        third_octave_db=[float(v) for v in curve],
        scalars={k: (v if math.isfinite(v) else float("nan")) for k, v in scalars.items()},
        notes=list(warnings),
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _is_hidden(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    return any(part.startswith(".") or part == "__MACOSX" for part in rel.parts)


def discover_audio(root: Path, limit: int = 0) -> List[Path]:
    """Every supported audio file under `root`, recursively, deterministically ordered.

    Hidden files and directories are skipped outright rather than reported as
    skips: `.DS_Store`, `._track.wav` resource forks and `__MACOSX/` are noise
    from the filesystem, not candidates the operator chose to include.
    """
    found: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            continue
        if _is_hidden(path, root):
            continue
        found.append(path.resolve())

    unique = sorted(set(found), key=lambda p: str(p).lower())
    return unique[:limit] if limit and limit > 0 else unique


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class Stats:
    n: int
    minimum: float
    p10: float
    p25: float
    median: float
    p75: float
    p90: float
    maximum: float
    mean: float
    std: float


def _stats(values: Sequence[float]) -> Optional[Stats]:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return None
    q = np.percentile(arr, [10.0, 25.0, 50.0, 75.0, 90.0])
    return Stats(
        n=int(arr.size),
        minimum=float(np.min(arr)),
        p10=float(q[0]), p25=float(q[1]), median=float(q[2]),
        p75=float(q[3]), p90=float(q[4]),
        maximum=float(np.max(arr)),
        mean=float(np.mean(arr)),
        std=float(np.std(arr)) if arr.size > 1 else 0.0,
    )


@dataclass
class BandFit:
    center_hz: float
    median_db: float
    q25_db: float
    q75_db: float
    current_db: float
    delta_db: float           # current - fitted median; + means targets.py is hot
    test_lo_db: float         # IQR, widened to BAND_AGREEMENT_FLOOR_DB if narrower
    test_hi_db: float
    outside_iqr: bool
    is_grid_limited: bool
    grid_bias_db: float


@dataclass
class ScalarFit:
    spec: ScalarSpec
    stats: Stats
    current: Optional[Tuple[float, float]]   # window, or (bound, bound) for min/max
    suggested: Optional[Tuple[float, float]]
    verdict: str                             # "ok" | "watch" | "wrong" | "info"
    note: str
    in_window_pct: float


@dataclass
class CorpusFit:
    genre_key: str
    genre_label: str
    curve_key: str
    root: str
    generated: str
    n_used: int
    n_found: int
    used: List[FileResult]
    skipped: List[FileResult]
    bands: List[BandFit]
    scalars: List[ScalarFit]
    anchors: List[Tuple[float, float]]
    anchor_residual_db: float
    anchor_residual_hz: float
    anchor_residual_core_db: float
    total_seconds: float

    @property
    def flagged_bands(self) -> List[BandFit]:
        return [b for b in self.bands if b.outside_iqr]

    @property
    def flagged_scalars(self) -> List[ScalarFit]:
        return [s for s in self.scalars if s.verdict == "wrong"]


def _current_window(profile: Any, spec: ScalarSpec) -> Optional[Tuple[float, float]]:
    if spec.kind == "ceiling":
        return (float(spec.ceiling), float(spec.ceiling)) if spec.ceiling is not None else None
    if not spec.profile_attr:
        return None
    value = getattr(profile, spec.profile_attr, None)
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    return (float(value), float(value))


def _judge_scalar(spec: ScalarSpec, st: Stats, values: Sequence[float],
                  current: Optional[Tuple[float, float]]) -> ScalarFit:
    """Compare one fitted distribution against its hand-set target.

    Window targets are judged two ways, because they can fail two ways: the
    window can be centred in the wrong place (the corpus median falls outside
    it) or it can be the wrong shape (it is centred correctly but so narrow that
    most real records sit outside it). Either one means the detector fires on
    records that are fine, so either one is `wrong`.
    """
    finite = [v for v in values if math.isfinite(v)]
    suggested: Optional[Tuple[float, float]] = None
    verdict = "info"
    note = ""
    in_pct = 0.0

    if current is None:
        return ScalarFit(spec, st, None, None, "info",
                         "No hand-set target for this measurement.", 0.0)

    lo, hi = current

    if spec.kind == "window":
        inside = sum(1 for v in finite if lo <= v <= hi)
        in_pct = 100.0 * inside / max(len(finite), 1)
        # p10/p90, not the IQR: targets.py windows are "this is fine" ranges and
        # should contain most of a healthy corpus, not merely its middle half.
        suggested = (round(st.p10, spec.nd), round(st.p90, spec.nd))
        median_outside = not (lo <= st.median <= hi)
        disjoint = st.p75 < lo or st.p25 > hi
        if median_outside:
            direction = "below" if st.median < lo else "above"
            verdict = "wrong"
            note = (
                f"Corpus median {st.median:.{spec.nd}f} sits {direction} the hand-set "
                f"window ({lo:.{spec.nd}f}, {hi:.{spec.nd}f}); only {in_pct:.0f}% of the "
                f"corpus falls inside it."
            )
        elif disjoint:
            verdict = "wrong"
            note = (
                f"The corpus interquartile range does not overlap the hand-set window "
                f"at all ({in_pct:.0f}% of files inside)."
            )
        elif in_pct < 50.0:
            verdict = "wrong"
            note = (
                f"Window is centred correctly but too narrow — only {in_pct:.0f}% of the "
                f"corpus falls inside it, so the detector fires on records that are fine."
            )
        elif in_pct < 75.0:
            verdict = "watch"
            note = f"{in_pct:.0f}% of the corpus falls inside the hand-set window."
        else:
            verdict = "ok"
            note = f"{in_pct:.0f}% of the corpus falls inside the hand-set window."

    elif spec.kind == "max":
        cap = hi
        inside = sum(1 for v in finite if v <= cap)
        in_pct = 100.0 * inside / max(len(finite), 1)
        suggested = (round(st.p90, spec.nd), round(st.p90, spec.nd))
        if st.median > cap:
            verdict = "wrong"
            note = (
                f"Over half the corpus already exceeds the {cap:.{spec.nd}f} cap "
                f"(median {st.median:.{spec.nd}f}) — the cap, not the records, is wrong."
            )
        elif st.p75 > cap:
            verdict = "watch"
            note = (
                f"The corpus p75 ({st.p75:.{spec.nd}f}) is over the {cap:.{spec.nd}f} cap; "
                f"{in_pct:.0f}% of files clear it."
            )
        else:
            verdict = "ok"
            note = f"{in_pct:.0f}% of the corpus clears the {cap:.{spec.nd}f} cap."

    elif spec.kind == "min":
        floor_ = lo
        inside = sum(1 for v in finite if v >= floor_)
        in_pct = 100.0 * inside / max(len(finite), 1)
        suggested = (round(st.p10, spec.nd), round(st.p10, spec.nd))
        if st.median < floor_:
            verdict = "wrong"
            note = (
                f"Over half the corpus sits under the {floor_:.{spec.nd}f} floor "
                f"(median {st.median:.{spec.nd}f})."
            )
        elif st.p25 < floor_:
            verdict = "watch"
            note = (
                f"The corpus p25 ({st.p25:.{spec.nd}f}) is under the "
                f"{floor_:.{spec.nd}f} floor; {in_pct:.0f}% of files clear it."
            )
        else:
            verdict = "ok"
            note = f"{in_pct:.0f}% of the corpus clears the {floor_:.{spec.nd}f} floor."

    else:  # ceiling — a delivery limit, not a genre opinion. Never "wrong".
        cap = hi
        inside = sum(1 for v in finite if v <= cap + 1e-9)
        in_pct = 100.0 * inside / max(len(finite), 1)
        verdict = "info"
        note = (
            f"{in_pct:.0f}% of the corpus clears the {cap:.1f} dBTP delivery ceiling "
            f"(median {st.median:.2f} dBTP). This is a codec limit, not a genre target."
        )

    return ScalarFit(spec, st, current, suggested, verdict, note, in_pct)


def fit_corpus(
    genre: str,
    root: Path,
    used: Sequence[FileResult],
    skipped: Sequence[FileResult],
    n_found: int,
) -> CorpusFit:
    """Turn per-file measurements into the fitted curve, the scalars and the anchors."""
    profile = targets.get_profile(genre)
    genre_key = targets.normalise_genre(genre)
    centers = np.asarray(core.THIRD_OCTAVE_CENTERS, dtype=np.float64)

    matrix = np.asarray([r.third_octave_db for r in used], dtype=np.float64)
    q = np.percentile(matrix, [25.0, 50.0, 75.0], axis=0)
    q25, median, q75 = q[0], q[1], q[2]

    current = np.asarray(targets.target_curve(genre_key, core.THIRD_OCTAVE_CENTERS),
                         dtype=np.float64)

    bands: List[BandFit] = []
    for i, hz in enumerate(centers):
        cur = float(current[i])
        mid = float(median[i])
        # Widen the test interval to at least the harness's own accuracy before
        # asking whether the hand-set value is outside it. See the constant.
        half = max(BAND_AGREEMENT_FLOOR_DB, 0.0)
        test_lo = min(float(q25[i]), mid - half)
        test_hi = max(float(q75[i]), mid + half)
        bands.append(BandFit(
            center_hz=float(hz),
            median_db=mid,
            q25_db=float(q25[i]),
            q75_db=float(q75[i]),
            current_db=cur,
            delta_db=cur - mid,
            test_lo_db=test_lo,
            test_hi_db=test_hi,
            outside_iqr=bool(cur < test_lo - 1e-9 or cur > test_hi + 1e-9),
            is_grid_limited=_is_grid_limited(float(hz)),
            grid_bias_db=float(GRID_BIAS_DB.get(float(hz), 0.0)),
        ))

    scalars: List[ScalarFit] = []
    for spec in SCALAR_SPECS:
        values = [r.scalars.get(spec.key, float("nan")) for r in used]
        st = _stats(values)
        if st is None:
            continue
        scalars.append(_judge_scalar(spec, st, values, _current_window(profile, spec)))

    # -- anchor reduction ----------------------------------------------------
    # targets.py stores 15 anchors and log-interpolates them onto the 31 bands.
    # Sampling the fitted median at those 15 frequencies and interpolating back
    # is lossy; report how lossy so the operator can see what the anchor format
    # costs on this corpus rather than assuming it is free.
    anchor_db: List[float] = []
    for hz in ANCHOR_HZ:
        idx = int(np.argmin(np.abs(centers - hz)))
        anchor_db.append(round(float(median[idx]), 1))
    anchors = [(float(hz), float(db)) for hz, db in zip(ANCHOR_HZ, anchor_db)]

    rebuilt = np.interp(
        np.log10(centers),
        np.log10(np.asarray(ANCHOR_HZ, dtype=np.float64)),
        np.asarray(anchor_db, dtype=np.float64),
    )
    residual = np.abs(rebuilt - median)
    worst = int(np.argmax(residual))
    # The "clean" residual is measured only where every bracketing anchor is
    # itself grid-free — see `_CLEAN_ANCHORS`. Outside that span the residual is
    # inherited measurement bias, not a shortcoming of the anchor format.
    clean = _clean_anchor_mask(centers)

    return CorpusFit(
        genre_key=genre_key,
        genre_label=profile.label,
        curve_key=profile.curve_key,
        root=str(root),
        generated=datetime.date.today().isoformat(),
        n_used=len(used),
        n_found=n_found,
        used=list(used),
        skipped=list(skipped),
        bands=bands,
        scalars=scalars,
        anchors=anchors,
        anchor_residual_db=float(residual[worst]),
        anchor_residual_hz=float(centers[worst]),
        anchor_residual_core_db=float(np.max(residual[clean])) if np.any(clean) else 0.0,
        total_seconds=float(sum(r.duration_sec for r in used)),
    )


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _hz(value: float) -> str:
    if value >= 1000.0:
        text = f"{value / 1000.0:.2f}".rstrip("0").rstrip(".")
        return f"{text} kHz"
    text = f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{text} Hz"


def _sd(value: float, nd: int = 1) -> str:
    """Signed, and never a '-0.00' that reads as a real negative."""
    if abs(value) < 0.5 * 10.0 ** (-nd):
        return f"{0.0:.{nd}f}"
    return f"{value:+.{nd}f}"


def render_report(fit: CorpusFit) -> str:
    lines: List[str] = []
    add = lines.append

    flagged_b = fit.flagged_bands
    flagged_s = fit.flagged_scalars
    watch_s = [s for s in fit.scalars if s.verdict == "watch"]

    add(f"# Calibration report — {fit.genre_label} (`{fit.genre_key}`)")
    add("")
    add(f"Fitted **{fit.generated}** by `tools/calibrate.py` v{TOOL_VERSION} "
        f"against `analysis/targets.py`.")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| Corpus | `{fit.root}` |")
    add(f"| Audio files found | {fit.n_found} |")
    add(f"| Usable masters measured | **{fit.n_used}** |")
    add(f"| Skipped | {len(fit.skipped)} |")
    add(f"| Total audio measured | {fit.total_seconds / 60.0:.1f} min |")
    add(f"| Curve currently shared with | `_CURVES[\"{fit.curve_key}\"]` |")
    add("")

    add("## Verdict")
    add("")
    if not flagged_b and not flagged_s:
        add("Nothing to change. Every hand-set band value falls inside the corpus "
            "interquartile range and every scalar window contains the corpus median. "
            "The hand-set targets and this corpus agree.")
    else:
        marginal = (f" {len(watch_s)} more {'is' if len(watch_s) == 1 else 'are'} "
                    f"marginal." if watch_s else "")
        add(f"**{len(flagged_b)} of {len(fit.bands)} bands** carry a hand-set value "
            f"outside the corpus interquartile range, and **{len(flagged_s)} of "
            f"{len(fit.scalars)} scalars** disagree with the corpus outright.{marginal}")
        add("")
        add("Those are the ones that are wrong. Everything else below is confirmation.")
        grid_only = flagged_b and all(b.is_grid_limited for b in flagged_b)
        if grid_only:
            add("")
            add("**Every flagged band is one this analyser cannot measure without a known "
                "bias.** On this corpus the hand-set curve is not contradicted anywhere "
                "the measurement can speak for itself.")
        if flagged_b:
            worst = sorted(flagged_b, key=lambda b: -abs(b.delta_db))[:6]
            add("")
            add("Largest band disagreements:")
            add("")
            for b in worst:
                direction = "hotter than" if b.delta_db > 0 else "quieter than"
                edge = "  _(grid-limited — see the bias note)_" if b.is_grid_limited else ""
                add(f"- **{_hz(b.center_hz)}** — hand-set target is "
                    f"{abs(b.delta_db):.1f} dB {direction} the corpus median "
                    f"({b.current_db:+.1f} vs {b.median_db:+.1f} dB, "
                    f"IQR {b.q25_db:+.1f} … {b.q75_db:+.1f}).{edge}")
        if flagged_s:
            add("")
            add("Scalars the corpus contradicts:")
            add("")
            for s in flagged_s:
                add(f"- **{s.spec.label}** — {s.note}")
    add("")

    # -- curve ---------------------------------------------------------------
    add("## Fitted target curve")
    add("")
    add(_para("Median 1/3-octave level across the corpus, normalised to the 800/1000/1250 Hz "
        "power mean — the same 0 dB reference `analysis/dsp/spectral.py` uses, so these "
        "numbers drop straight onto `targets.target_curve()` without conversion. "
        "**This is the empirical target curve.**"))
    add("")
    add(_para("`Δ` is `hand-set − fitted`: positive means `targets.py` is asking for more "
        "energy in that band than the corpus actually has. `!` marks a hand-set value "
        "outside the corpus interquartile range — widened, where the corpus agrees with "
        f"itself more tightly than that, to at least ±{BAND_AGREEMENT_FLOOR_DB:.1f} dB "
        "around the median, which is the per-band accuracy `--self-test` holds this tool "
        "to. A smaller disagreement than that is not evidence about the hand-set value."))
    add("")
    add("| Band | Fitted median | IQR (p25 … p75) | Spread | Hand-set | Δ | Grid bias | |")
    add("|---|---:|---|---:|---:|---:|---:|:-:|")
    for b in fit.bands:
        flag = "**!**" if b.outside_iqr else ""
        star = " \\*" if b.is_grid_limited else ""
        if abs(b.grid_bias_db) >= GRID_BIAS_THRESHOLD_DB:
            bias = f"{_sd(b.grid_bias_db, 2)} dB"
        elif b.is_grid_limited:
            bias = "_end band_"
        else:
            bias = "—"
        add(f"| {_hz(b.center_hz)}{star} | {b.median_db:+.2f} dB | "
            f"{b.q25_db:+.2f} … {b.q75_db:+.2f} | {b.q75_db - b.q25_db:.2f} dB | "
            f"{b.current_db:+.2f} dB | {_sd(b.delta_db, 2)} | {bias} | {flag} |")
    add("")
    add(_para("\\* Band the fitted curve cannot speak for. `Grid bias` gives the amount the "
        "FFT grid contributes — subtract it before believing the number. `_end band_` "
        "means the band extends past the outermost anchor, where the hand-set value is "
        "an extrapolation rather than a target. See *Known measurement bias*."))
    add("")

    # -- anchors -------------------------------------------------------------
    add("### As anchors")
    add("")
    add(_para(f"`targets.py` stores {len(ANCHOR_HZ)} anchor points and log-interpolates them "
        f"onto the 31 bands. Reducing the fitted 31-point curve to those anchors and "
        f"interpolating back costs at most **{fit.anchor_residual_db:.2f} dB** "
        f"(at {_hz(fit.anchor_residual_hz)}), or "
        f"**{fit.anchor_residual_core_db:.2f} dB** across "
        f"{_hz(CLEAN_ANCHOR_SPAN_HZ[0])}–{_hz(CLEAN_ANCHOR_SPAN_HZ[1])}, the span where "
        f"every bracketing anchor is grid-free. The second figure is the honest one: "
        f"below it, four of the anchor frequencies are themselves grid-limited and carry "
        f"their bias into every band they interpolate across. If even that residual is "
        f"large, the corpus has structure the anchor format cannot represent and the "
        f"full curve above is the better record of it."))
    add("")
    add("```python")
    add(f'"{fit.genre_key}_fitted": (')
    for chunk in _chunk(fit.anchors, 6):
        add("    " + " ".join(f"({int(hz)}, {db:.1f})," for hz, db in chunk))
    add("),")
    add("```")
    add("")

    # -- scalars -------------------------------------------------------------
    add("## Scalars")
    add("")
    add(_para("`Suggested` is p10 … p90 of the corpus for window targets (a \"this is fine\" "
        "range should contain most real records, not just the middle half), p90 for "
        "caps and p10 for floors."))
    add("")
    add("| Measurement | n | p25 | median | p75 | Hand-set | In range | Suggested | |")
    add("|---|---:|---:|---:|---:|---|---:|---|:-:|")
    for s in fit.scalars:
        nd = s.spec.nd
        st = s.stats
        cur = _window_text(s.spec, s.current, nd)
        sug = _window_text(s.spec, s.suggested, nd)
        mark = {"wrong": "**!**", "watch": "~", "ok": "", "info": ""}[s.verdict]
        pct = "—" if s.verdict == "info" and s.spec.kind == "ceiling" else f"{s.in_window_pct:.0f}%"
        add(f"| {s.spec.label} | {st.n} | {st.p25:.{nd}f} | **{st.median:.{nd}f}** | "
            f"{st.p75:.{nd}f} | {cur} | {pct} | {sug} | {mark} |")
    add("")
    for s in fit.scalars:
        if s.note:
            prefix = {"wrong": "**!**", "watch": "~", "ok": "·", "info": "·"}[s.verdict]
            add(f"- {prefix} **{s.spec.label}** — {s.note}")
    add("")

    # -- skipped -------------------------------------------------------------
    add("## Skipped")
    add("")
    if not fit.skipped:
        add("Nothing was skipped — every audio file found was usable.")
    else:
        add(f"{len(fit.skipped)} of {fit.n_found} files were not admitted to the corpus.")
        add("")
        add("| File | Reason |")
        add("|---|---|")
        for r in fit.skipped:
            add(f"| `{r.name}` | {r.skip_reason} |")
    add("")

    stage_notes = [(r.name, n) for r in fit.used for n in r.notes]
    if stage_notes:
        add("Files that were used but had a non-spectral measurement stage degrade "
            "(the affected scalar is excluded from its own distribution, so the `n` "
            "column above may differ per row):")
        add("")
        for name, note in stage_notes[:20]:
            add(f"- `{name}` — {note}")
        add("")

    # -- bias ----------------------------------------------------------------
    add("## Known measurement bias")
    add("")
    spacing = core.ANALYSIS_SR / _STFT_N_FFT
    quantised = [b for b in fit.bands
                 if b.is_grid_limited and abs(b.grid_bias_db) >= GRID_BIAS_THRESHOLD_DB]
    add(_para(
        f"`core.stft_power` runs an {_STFT_N_FFT}-point FFT at "
        f"{core.ANALYSIS_SR // 1000} kHz, so the bins are {spacing:.2f} Hz apart, and "
        f"`spectral.py` defines a band's level as the summed power of the bins whose "
        f"centres fall inside it. That sum is proportional to an integer bin count "
        f"while the band width is not, so every 1/3-octave band narrow enough to hold "
        f"only a few bins reads systematically hot or shy — arithmetic on the grid, "
        f"nothing to do with the audio."
    ))
    add("")
    add("Predicted bias, from the grid alone:")
    add("")
    for b in quantised:
        add(f"- **{_hz(b.center_hz)}** — {_sd(b.grid_bias_db, 2)} dB "
            f"(band is {b.center_hz * (2 ** (1 / 6) - 2 ** (-1 / 6)):.1f} Hz wide, "
            f"i.e. {b.center_hz * (2 ** (1 / 6) - 2 ** (-1 / 6)) / spacing:.2f} bins)")
    add("")
    add(_para(
        f"Above {_hz(max([b.center_hz for b in quantised] + [0.0]))} there are dozens of "
        f"bins per band and the effect falls below 0.01 dB. The two end bands "
        f"({_hz(fit.bands[0].center_hz)} and {_hz(fit.bands[-1].center_hz)}) are starred "
        f"for a different reason: they extend past the outermost anchor in `targets.py`, "
        f"where `np.interp` clamps, so the hand-set value there is an extrapolation and "
        f"is not comparable with a measured one."
    ))
    add("")
    add(_para(
        "None of this widens the interquartile range — it is identical for every file, "
        "so it offsets the fitted median of the starred bands by a fixed amount and "
        "leaves the spread untouched. **Do not move `targets.py` toward the fitted value "
        "in a starred band without subtracting the bias first.** The hand-set anchors "
        "describe acoustic reality; the fitted ones describe what this analyser "
        "measures, and in those bands the two are not the same statement. Everywhere "
        "else they are directly comparable and the fitted value is the better number."
    ))
    add("")

    # -- how to apply --------------------------------------------------------
    add("## Applying this")
    add("")
    add(textwrap.dedent(
        f"""\
        Re-run with `--apply` to write `analysis/targets_fitted.py`:

        ```bash
        python tools/calibrate.py --genre {fit.genre_key} --input "{fit.root}" \\
            --out calibration-{fit.genre_key}.md --apply
        ```

        That file is generated, importable, and never merged into `targets.py`
        automatically. It carries the fitted anchors in the exact tuple format
        `_CURVES` uses and a ready-to-paste `_register(...)` call. Read this report
        first, decide band by band which fitted values you believe, and paste those.
        """
    ).strip())
    add("")

    return "\n".join(lines) + "\n"


def _window_text(spec: ScalarSpec, window: Optional[Tuple[float, float]], nd: int) -> str:
    if window is None:
        return "—"
    lo, hi = window
    if spec.kind == "window":
        return f"{lo:.{nd}f} … {hi:.{nd}f}"
    if spec.kind == "max":
        return f"≤ {hi:.{nd}f}"
    if spec.kind == "min":
        return f"≥ {lo:.{nd}f}"
    return f"≤ {hi:.{nd}f}"


def _para(text: str, width: int = 88) -> str:
    """One markdown paragraph, hard-wrapped so the raw file is readable too."""
    return textwrap.fill(" ".join(text.split()), width=width,
                         break_long_words=False, break_on_hyphens=False)


def _chunk(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# ---------------------------------------------------------------------------
# --apply: analysis/targets_fitted.py
# ---------------------------------------------------------------------------

_FITTED_HEADER = '''"""Fitted genre reference data — GENERATED by tools/calibrate.py. Do not hand-edit.

Every value here was measured from a corpus of commercial masters with the same
DSP layer the product runs on requests, then aggregated as a per-band median.
Nothing in it is applied automatically: `analysis/targets.py` is still the single
source of truth and is only ever changed by a human pasting from here after
reading the calibration report.

    FITTED_CURVES[genre]            15 anchors, the exact format _CURVES uses
    FITTED_THIRD_OCTAVE[genre]      the full 31-band fitted median, unreduced
    FITTED_IQR[genre]               (p25, p75) per band — how much the corpus agrees
    FITTED_PROFILE_KWARGS[genre]    scalar windows, keyed as _register() takes them
    FITTED_META[genre]              corpus size, date, source directory
    SUGGESTED_REGISTER_CALLS[genre] ready-to-paste _register(...) source text

Re-running the tool for a different genre merges into this file rather than
replacing it.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

Anchors = Sequence[Tuple[float, float]]

'''


def _py_repr(value: Any) -> str:
    """Source text for a generated literal.

    `repr()` alone is not usable here: it renders 20000.0 as `2e+04` once the
    value has been through `%g`, single-quotes strings where the rest of the file
    double-quotes them, and happily emits `-0.0`. The generated module is meant to
    be read and pasted from, so it gets consistent formatting.
    """
    if isinstance(value, tuple):
        return "(" + ", ".join(_py_repr(v) for v in value) + ")"
    if isinstance(value, list):
        return "[" + ", ".join(_py_repr(v) for v in value) + "]"
    if isinstance(value, bool):          # before int: bool is a subclass of it
        return "True" if value else "False"
    if isinstance(value, float):
        text = f"{value + 0.0:.6f}".rstrip("0")
        return text + "0" if text.endswith(".") else text
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return repr(value)


def _anchor_line(pairs: Sequence[Tuple[Any, Any]]) -> str:
    """One line of `(hz, db),` anchor tuples, with no `-0.0` in it."""
    return " ".join(
        f"({int(hz)}, {float(db) + 0.0:.1f})," for hz, db in pairs
    )


def _register_call_text(fit: CorpusFit) -> str:
    """The `_register(...)` line for targets.py, built from the fitted scalars."""
    profile = targets.get_profile(fit.genre_key)
    by_key = {s.spec.key: s for s in fit.scalars}

    def sug(key: str, default: Any) -> Any:
        s = by_key.get(key)
        if s is None or s.suggested is None:
            return default
        spec = s.spec
        if spec.kind == "window":
            return (round(s.suggested[0], spec.nd), round(s.suggested[1], spec.nd))
        return round(s.suggested[0], spec.nd)

    lines = [
        "_register(",
        f'    "{fit.genre_key}", "{profile.label}", "{fit.curve_key}",',
        f"    lufs={_py_repr(sug('integrated_lufs', profile.integrated_lufs))}, "
        f"lra={_py_repr(sug('loudness_range_lu', profile.loudness_range_lu))},",
        f"    crest={_py_repr(sug('crest_factor_db', profile.crest_factor_db))}, "
        f"micro={_py_repr(sug('micro_dynamics_db', profile.micro_dynamics_db))}, "
        f"psr={_py_repr(sug('psr_p10_db', profile.psr_p10_db))},",
        f"    width={_py_repr(sug('stereo_width', profile.stereo_width))}, "
        f"corr_min={_py_repr(sug('correlation', profile.correlation_min))},",
        f"    mud_ratio_db={_py_repr(sug('mud_ratio_db', profile.mud_ratio_db))}, "
        f"harshness_max={_py_repr(sug('harshness_index', profile.harshness_max))},",
        f'    notes={_py_repr(f"Fitted {fit.generated} from {fit.n_used} masters. {profile.notes}".strip())},',
        ")",
    ]
    return "\n".join(lines)


def _load_existing_fitted(path: Path) -> Dict[str, Dict[str, Any]]:
    """Read back a previously generated module so other genres survive a re-run."""
    store: Dict[str, Dict[str, Any]] = {
        "FITTED_CURVES": {}, "FITTED_THIRD_OCTAVE": {}, "FITTED_IQR": {},
        "FITTED_PROFILE_KWARGS": {}, "FITTED_META": {}, "SUGGESTED_REGISTER_CALLS": {},
    }
    if not path.exists():
        return store
    try:
        spec = importlib.util.spec_from_file_location("_targets_fitted_prev", str(path))
        if spec is None or spec.loader is None:
            raise ImportError("no loader")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # generated by this tool; trusted
        for key in store:
            value = getattr(module, key, None)
            if isinstance(value, dict):
                store[key] = dict(value)
    except Exception:
        backup = path.with_suffix(".py.bak")
        shutil.copy2(path, backup)
        print(f"  ! could not read the existing {path.name}; backed it up to "
              f"{backup.name} and starting fresh", file=sys.stderr)
    return store


def render_fitted_module(fit: CorpusFit, existing: Dict[str, Dict[str, Any]]) -> str:
    """The full text of `analysis/targets_fitted.py`, with `fit` merged in."""
    centers = [float(c) for c in core.THIRD_OCTAVE_CENTERS]

    curves = dict(existing.get("FITTED_CURVES") or {})
    toct = dict(existing.get("FITTED_THIRD_OCTAVE") or {})
    iqr = dict(existing.get("FITTED_IQR") or {})
    kwargs_map = dict(existing.get("FITTED_PROFILE_KWARGS") or {})
    meta = dict(existing.get("FITTED_META") or {})
    calls = dict(existing.get("SUGGESTED_REGISTER_CALLS") or {})

    curves[fit.genre_key] = tuple((int(hz), round(db, 1)) for hz, db in fit.anchors)
    toct[fit.genre_key] = [round(b.median_db, 2) for b in fit.bands]
    iqr[fit.genre_key] = [(round(b.q25_db, 2), round(b.q75_db, 2)) for b in fit.bands]

    by_key = {s.spec.key: s for s in fit.scalars}
    kw: Dict[str, Any] = {}
    for spec in SCALAR_SPECS:
        if not spec.register_kwarg:
            continue
        s = by_key.get(spec.key)
        if s is None or s.suggested is None:
            continue
        if spec.kind == "window":
            kw[spec.register_kwarg] = (round(s.suggested[0], spec.nd),
                                       round(s.suggested[1], spec.nd))
        else:
            kw[spec.register_kwarg] = round(s.suggested[0], spec.nd)
    kwargs_map[fit.genre_key] = kw

    meta[fit.genre_key] = {
        "genre_label": fit.genre_label,
        "curve_key": fit.curve_key,
        "corpus_files": fit.n_used,
        "corpus_files_found": fit.n_found,
        "corpus_minutes": round(fit.total_seconds / 60.0, 1),
        "corpus_dir": fit.root,
        "fitted_on": fit.generated,
        "tool_version": TOOL_VERSION,
        "anchor_residual_db": round(fit.anchor_residual_db, 2),
        "bands_outside_handset_iqr": len(fit.flagged_bands),
        "scalars_contradicted": [s.spec.key for s in fit.flagged_scalars],
    }
    calls[fit.genre_key] = _register_call_text(fit)

    out: List[str] = [_FITTED_HEADER]
    out.append("# The grid FITTED_THIRD_OCTAVE and FITTED_IQR are indexed by, "
               "mirrored from core.py")
    out.append("THIRD_OCTAVE_CENTERS: Tuple[float, ...] = (")
    for chunk in _chunk(centers, 8):
        out.append("    " + " ".join(f"{_py_repr(v)}," for v in chunk))
    out.append(")\n")

    for key in sorted(meta):
        m = meta[key]
        out.append(
            f"# {'-' * 74}\n"
            f"# {key} — fitted {m['fitted_on']} from {m['corpus_files']} masters "
            f"({m['corpus_minutes']} min)\n"
            f"#   corpus: {m['corpus_dir']}\n"
            f"#   {m['bands_outside_handset_iqr']} of {len(centers)} hand-set bands fell "
            f"outside the corpus IQR\n"
            f"#   anchor-reduction residual: {m['anchor_residual_db']} dB\n"
            f"# {'-' * 74}"
        )
        out.append("")

    out.append("FITTED_CURVES: Dict[str, Anchors] = {")
    for key in sorted(curves):
        out.append(f'    "{key}": (')
        for chunk in _chunk(list(curves[key]), 6):
            out.append("        " + _anchor_line(chunk))
        out.append("    ),")
    out.append("}\n")

    out.append("FITTED_THIRD_OCTAVE: Dict[str, List[float]] = {")
    for key in sorted(toct):
        out.append(f'    "{key}": [')
        for chunk in _chunk(list(toct[key]), 8):
            out.append("        " + " ".join(f"{float(v) + 0.0:.2f}," for v in chunk))
        out.append("    ],")
    out.append("}\n")

    out.append("FITTED_IQR: Dict[str, List[Tuple[float, float]]] = {")
    for key in sorted(iqr):
        out.append(f'    "{key}": [')
        for chunk in _chunk(list(iqr[key]), 4):
            out.append("        " + " ".join(
                f"({float(lo) + 0.0:.2f}, {float(hi) + 0.0:.2f})," for lo, hi in chunk))
        out.append("    ],")
    out.append("}\n")

    out.append("FITTED_PROFILE_KWARGS: Dict[str, Dict[str, object]] = {")
    for key in sorted(kwargs_map):
        out.append(f'    "{key}": {{')
        for name, value in kwargs_map[key].items():
            out.append(f'        "{name}": {_py_repr(value)},')
        out.append("    },")
    out.append("}\n")

    out.append("FITTED_META: Dict[str, Dict[str, object]] = {")
    for key in sorted(meta):
        out.append(f'    "{key}": {{')
        for name, value in meta[key].items():
            out.append(f'        "{name}": {_py_repr(value)},')
        out.append("    },")
    out.append("}\n")

    out.append("# Paste-ready. Each entry is the exact _register(...) call for targets.py,")
    out.append("# with the fitted scalar windows substituted in.")
    out.append("SUGGESTED_REGISTER_CALLS: Dict[str, str] = {")
    for key in sorted(calls):
        out.append(f'    "{key}": """\\')
        out.append(str(calls[key]).strip("\n"))
        out.append('""",')
    out.append("}\n\n")

    out.append(textwrap.dedent(
        '''
        def paste_block(genre: str) -> str:
            """The anchor tuple and the _register(...) call for one fitted genre."""
            anchors = FITTED_CURVES[genre]
            rows = [f\'    "{genre}": (\']
            for i in range(0, len(anchors), 6):
                rows.append("        " + " ".join(
                    f"({int(hz)}, {float(db):.1f})," for hz, db in anchors[i:i + 6]))
            rows.append("    ),")
            return "\\n".join(rows) + "\\n\\n" + SUGGESTED_REGISTER_CALLS[genre]
        ''').strip() + "\n")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Running the corpus
# ---------------------------------------------------------------------------


def _run_measurements(
    paths: Sequence[Path],
    genre_key: str,
    min_duration: float,
    require_stereo: bool,
    workers: int,
    quiet: bool,
) -> List[FileResult]:
    total = len(paths)
    results: List[FileResult] = []
    started = time.perf_counter()

    def report(index: int, r: FileResult) -> None:
        if quiet:
            return
        tag = "ok  " if r.ok else "skip"
        detail = (f"{r.duration_sec:6.1f}s  {r.elapsed_ms / 1000.0:5.1f}s"
                  if r.ok else f"— {r.skip_reason}")
        print(f"  [{index:>3}/{total}] {tag} {r.name[:52]:<52} {detail}", file=sys.stderr)

    if workers <= 1 or total <= 1:
        for i, path in enumerate(paths, 1):
            r = measure_file(str(path), genre_key, min_duration, require_stereo)
            results.append(r)
            report(i, r)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(measure_file, str(p), genre_key, min_duration, require_stereo): p
                for p in paths
            }
            for i, future in enumerate(as_completed(futures), 1):
                path = futures[future]
                try:
                    r = future.result()
                except Exception as exc:  # noqa: BLE001 — a worker died, not the run
                    r = FileResult(
                        path=str(path), name=path.name, ok=False,
                        skip_reason=f"worker crashed ({type(exc).__name__}): {exc}",
                    )
                results.append(r)
                report(i, r)

    if not quiet:
        elapsed = time.perf_counter() - started
        ok = sum(1 for r in results if r.ok)
        print(f"  measured {ok}/{total} usable in {elapsed:.1f}s "
              f"({workers} worker{'s' if workers != 1 else ''})", file=sys.stderr)

    results.sort(key=lambda r: r.name.lower())
    return results


def calibrate(
    genre: str,
    input_dir: Path,
    *,
    min_files: int = DEFAULT_MIN_FILES,
    min_duration: float = DEFAULT_MIN_DURATION_SEC,
    workers: int = 0,
    limit: int = 0,
    allow_mono: bool = False,
    quiet: bool = False,
) -> Tuple[Optional[CorpusFit], str]:
    """Measure a corpus and fit it. Returns (fit, error_message).

    `fit` is None exactly when `error_message` is non-empty; the caller decides
    whether that is a hard exit or, in `--self-test`, a failed assertion.
    """
    genre_key = targets.normalise_genre(genre)
    profile = targets.get_profile(genre_key)

    if not input_dir.is_dir():
        return None, f"{input_dir} is not a directory."

    paths = discover_audio(input_dir, limit=limit)
    if not paths:
        return None, (
            f"No audio found under {input_dir}. Supported extensions: "
            f"{', '.join(sorted(SUPPORTED_FORMATS))}."
        )

    require_stereo = (not allow_mono) and profile.stereo_width[1] > 0.0
    if workers <= 0:
        workers = max(1, min(len(paths), (os.cpu_count() or 2)))

    if not quiet:
        print(f"\n  {profile.label} ({genre_key}) — {len(paths)} audio file"
              f"{'s' if len(paths) != 1 else ''} under {input_dir}", file=sys.stderr)

    results = _run_measurements(paths, genre_key, min_duration, require_stereo,
                                workers, quiet)
    used = [r for r in results if r.ok]
    skipped = [r for r in results if not r.ok]

    if len(used) < min_files:
        detail = "\n".join(f"    - {r.name}: {r.skip_reason}" for r in skipped[:12])
        more = f"\n    … and {len(skipped) - 12} more" if len(skipped) > 12 else ""
        # Point at the rule that actually did the rejecting, not a generic one.
        hints = ["add more masters"]
        n_short = sum(1 for r in skipped if "under the" in r.skip_reason)
        n_mono = sum(1 for r in skipped if r.skip_reason.startswith("mono file"))
        if n_short and n_short >= len(skipped) / 2:
            hints.append(f"lower --min-duration (currently {min_duration:.0f} s; "
                         f"{n_short} file{'s' if n_short != 1 else ''} failed on it)")
        if n_mono:
            hints.append(f"pass --allow-mono ({n_mono} mono file"
                         f"{'s' if n_mono != 1 else ''} were rejected)")
        hints.append(f"lower --min-files (currently {min_files}) if you accept a "
                     f"weaker fit")
        return None, (
            f"Only {len(used)} usable master{'s' if len(used) != 1 else ''} out of "
            f"{len(paths)} file{'s' if len(paths) != 1 else ''} — at least {min_files} "
            f"are required.\n"
            f"  A target curve fitted from {len(used)} track"
            f"{'s' if len(used) != 1 else ''} would look empirical and would be worse "
            f"than the hand-set one it replaced, so nothing was written."
            + (f"\n\n  Skipped:\n{detail}{more}\n" if skipped else "\n")
            + "\n  Options: " + "; ".join(hints) + "."
        )

    return fit_corpus(genre_key, input_dir, used, skipped, len(paths)), ""


def _write_fitted(fit: CorpusFit, path: Path, quiet: bool) -> None:
    existing = _load_existing_fitted(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_fitted_module(fit, existing), encoding="utf-8")
    if not quiet:
        others = sorted(set(existing.get("FITTED_CURVES") or {}) - {fit.genre_key})
        kept = f" (kept {', '.join(others)})" if others else ""
        print(f"  wrote {path}{kept}", file=sys.stderr)
        print(f"  targets.py was NOT modified — paste from "
              f"SUGGESTED_REGISTER_CALLS['{fit.genre_key}'] once you agree with it",
              file=sys.stderr)


# ---------------------------------------------------------------------------
# --self-test
# ---------------------------------------------------------------------------
#
# The fixtures in testdata/reference_*.wav are noise shaped to sit exactly on a
# known genre target curve, which makes them ground truth: whatever this tool
# fits from them must come back as that curve. Testing against the single files
# directly would only exercise the measurement, so instead each one is expanded
# into a small synthetic corpus — five variants at different levels and different
# spectral tilts, plus three files that must be rejected — written to a temp
# directory and run through the *entire* production path: directory walk, skip
# rules, process pool, median/IQR aggregation, comparison, and --apply.
#
# The tilts are symmetric about zero (-0.6 … +0.6 dB/decade), so their median is
# the untilted curve. That is the real assertion: not "one file measures right"
# but "the median of a corpus that disagrees with itself recovers the truth".

_SELF_TEST_GENRES: Tuple[Tuple[str, str], ...] = (
    ("trap", "reference_trap.wav"),
    ("pop", "reference_pop.wav"),
    ("folk", "reference_folk.wav"),
)

_SELF_TEST_TILTS: Tuple[float, ...] = (-0.6, -0.3, 0.0, 0.3, 0.6)   # dB/decade
_SELF_TEST_GAINS: Tuple[float, ...] = (0.0, -2.5, 1.0, -5.0, -1.2)  # dB


def _apply_tilt(x: np.ndarray, sr: int, db_per_decade: float) -> np.ndarray:
    """Zero-phase spectral tilt, `db_per_decade` dB per decade pivoting at 1 kHz."""
    if abs(db_per_decade) < 1e-9:
        return x
    n = x.shape[0]
    nfft = 1 << int(math.ceil(math.log2(max(n, 2))))
    spec = np.fft.rfft(x, n=nfft, axis=0)
    freqs = np.clip(np.fft.rfftfreq(nfft, 1.0 / sr), 20.0, 20_000.0)
    gain = 10.0 ** ((db_per_decade * np.log10(freqs / 1000.0)) / 20.0)
    spec *= gain[:, None] if x.ndim == 2 else gain
    return np.asarray(np.fft.irfft(spec, n=nfft, axis=0)[:n], dtype=np.float64)


def _crossfade_concat(a: np.ndarray, b: np.ndarray, sr: int, fade_ms: float = 10.0) -> np.ndarray:
    """Join two buffers with a short equal-power crossfade, so the splice is silent."""
    fade = max(1, int(fade_ms * 1e-3 * sr))
    fade = min(fade, a.shape[0] // 4, b.shape[0] // 4)
    ramp = np.linspace(0.0, 1.0, fade, endpoint=False)[:, None]
    head = a[:-fade]
    join = a[-fade:] * np.cos(ramp * np.pi / 2.0) + b[:fade] * np.sin(ramp * np.pi / 2.0)
    return np.concatenate([head, join, b[fade:]], axis=0)


def _lengthen(x: np.ndarray, sr: int, shift_sec: int) -> np.ndarray:
    """Tile a 30 s fixture out past the 60 s admission rule.

    Circular shifts and crossfaded joins, so the result is a different waveform
    with an identical long-term spectrum — which is the only property the fit
    reads. The trailing partial copy is what puts it clear of exactly 60 s.
    """
    a = _crossfade_concat(x, np.roll(x, shift_sec * sr, axis=0), sr)
    return _crossfade_concat(a, x[: sr * 5], sr)


def _build_self_test_corpus(source: Path, out_dir: Path) -> Tuple[int, int]:
    """Write one genre's synthetic corpus. Returns (n_expected_usable, n_expected_skipped)."""
    import soundfile as sf

    data, sr = sf.read(str(source), always_2d=True, dtype="float64")
    stem = source.stem

    (out_dir / "nested").mkdir(parents=True, exist_ok=True)

    for i, (tilt, gain_db) in enumerate(zip(_SELF_TEST_TILTS, _SELF_TEST_GAINS)):
        shaped = _apply_tilt(data, sr, tilt) * (10.0 ** (gain_db / 20.0))
        tiled = _lengthen(shaped, sr, (i + 1) * 3)
        peak = float(np.max(np.abs(tiled))) if tiled.size else 0.0
        if peak > 0.98:
            tiled *= 0.98 / peak
        # Two of the five live in a subdirectory: the walk has to be recursive.
        target = (out_dir / "nested" if i >= 3 else out_dir) / f"{stem}_v{i + 1}.wav"
        sf.write(str(target), tiled, sr, subtype="PCM_16")

    # Three files that must be rejected, one per admission rule.
    sf.write(str(out_dir / "_decoy_short.wav"), data[: sr * 12], sr, subtype="PCM_16")
    sf.write(str(out_dir / "_decoy_silent.wav"),
             np.zeros((sr * 65, 2), dtype=np.float64), sr, subtype="PCM_16")
    sf.write(str(out_dir / "_decoy_mono.wav"),
             _lengthen(data, sr, 7).mean(axis=1), sr, subtype="PCM_16")

    # Not audio: proves the extension filter, and must not appear as a skip.
    (out_dir / "liner-notes.txt").write_text("not audio\n", encoding="utf-8")
    (out_dir / ".DS_Store").write_bytes(b"\x00")

    return len(_SELF_TEST_TILTS), 3


def run_self_test(workers: int = 0, keep: bool = False, quiet: bool = False) -> int:
    """Prove the harness recovers a known curve. Returns a process exit code."""
    # Results go to stdout, per-file progress to stderr. Piping the pair makes
    # stdout block-buffered and the two interleave out of order, which turns a
    # readable transcript into a puzzle — so line-buffer stdout for the duration.
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

    testdata = _BACKEND_DIR / "testdata"
    print("=" * 78)
    print("calibrate.py --self-test")
    print("=" * 78)
    print(f"fixtures      : {testdata}")
    print(f"tolerance     : {SELF_TEST_TOL_DB:.1f} dB per band, "
          f"{SELF_TEST_GRID_TOL_DB:.1f} dB in the {len(GRID_LIMITED_HZ)} grid-limited "
          f"bands ({', '.join(_hz(h) for h in GRID_LIMITED_HZ)})")
    print(f"corpus/genre  : {len(_SELF_TEST_TILTS)} variants "
          f"(tilts {', '.join(f'{t:+.1f}' for t in _SELF_TEST_TILTS)} dB/decade, "
          f"median 0) + 3 files that must be rejected")
    print()

    failures: List[str] = []
    workdir = Path(tempfile.mkdtemp(prefix="mixdoctor-calib-selftest-"))
    fits: Dict[str, CorpusFit] = {}

    try:
        for genre, fixture in _SELF_TEST_GENRES:
            source = testdata / fixture
            print("-" * 78)
            print(f"[{genre}]  source fixture: {fixture}")
            if not source.exists():
                failures.append(f"{genre}: fixture {source} is missing")
                print(f"  FAIL  fixture missing: {source}")
                continue

            corpus = workdir / genre
            corpus.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            n_expect_ok, n_expect_skip = _build_self_test_corpus(source, corpus)
            print(f"  built {n_expect_ok + n_expect_skip} files in {corpus} "
                  f"({time.perf_counter() - t0:.1f}s)")

            fit, error = calibrate(
                genre, corpus,
                min_files=DEFAULT_MIN_FILES,
                min_duration=DEFAULT_MIN_DURATION_SEC,
                workers=workers,
                quiet=quiet,
            )
            if fit is None:
                failures.append(f"{genre}: calibration refused — {error.splitlines()[0]}")
                print(f"  FAIL  {error.splitlines()[0]}")
                continue
            fits[genre] = fit

            # 1. admission
            ok_admission = fit.n_used == n_expect_ok and len(fit.skipped) == n_expect_skip
            print(f"  admission   : {fit.n_used} used / {len(fit.skipped)} skipped "
                  f"of {fit.n_found} found "
                  f"{'OK' if ok_admission else 'FAIL'}")
            for r in fit.skipped:
                print(f"                skipped {r.name}: {r.skip_reason}")
            if not ok_admission:
                failures.append(
                    f"{genre}: expected {n_expect_ok} used / {n_expect_skip} skipped, "
                    f"got {fit.n_used} / {len(fit.skipped)}"
                )

            # 2. curve recovery — the actual point of the self-test
            truth = np.asarray(targets.target_curve(genre, core.THIRD_OCTAVE_CENTERS),
                               dtype=np.float64)
            fitted = np.asarray([b.median_db for b in fit.bands], dtype=np.float64)
            err = fitted - truth
            is_grid = np.asarray([b.is_grid_limited for b in fit.bands], dtype=bool)

            free_worst = float(np.max(np.abs(err[~is_grid])))
            free_worst_hz = float(
                fit.bands[int(np.argmax(np.where(is_grid, -1.0, np.abs(err))))].center_hz
            )
            grid_worst = float(np.max(np.abs(err[is_grid])))
            rms = float(np.sqrt(np.mean(np.square(err[~is_grid]))))

            free_ok = free_worst <= SELF_TEST_TOL_DB
            grid_ok = grid_worst <= SELF_TEST_GRID_TOL_DB
            print(f"  curve       : {int(np.count_nonzero(~is_grid))} grid-free bands, "
                  f"worst |error| {free_worst:.2f} dB @ {_hz(free_worst_hz)}, "
                  f"rms {rms:.2f} dB  {'OK' if free_ok else 'FAIL'}")
            print(f"                {int(np.count_nonzero(is_grid))} grid-limited bands, "
                  f"worst |error| {grid_worst:.2f} dB  {'OK' if grid_ok else 'FAIL'}")
            if not free_ok:
                failures.append(
                    f"{genre}: band error {free_worst:.2f} dB at {_hz(free_worst_hz)} "
                    f"exceeds {SELF_TEST_TOL_DB:.1f} dB"
                )
            if not grid_ok:
                failures.append(
                    f"{genre}: grid-limited band error {grid_worst:.2f} dB exceeds "
                    f"{SELF_TEST_GRID_TOL_DB:.1f} dB"
                )

            order = np.argsort(np.abs(err))[::-1][:6]
            print("                worst bands: " + ", ".join(
                f"{_hz(fit.bands[i].center_hz)} {err[i]:+.2f}"
                f"{'*' if fit.bands[i].is_grid_limited else ''}"
                for i in order
            ) + "   (* = grid-limited)")
            print("                predicted grid bias vs measured error: " + ", ".join(
                f"{_hz(fit.bands[i].center_hz)} {fit.bands[i].grid_bias_db:+.2f}/"
                f"{err[i]:+.2f}"
                for i in range(len(fit.bands))
                if abs(fit.bands[i].grid_bias_db) >= GRID_BIAS_THRESHOLD_DB
            ))

            # 3. the IQR has to be a real interval, and has to bracket the median
            widths = np.asarray([b.q75_db - b.q25_db for b in fit.bands])
            brackets = all(b.q25_db - 1e-9 <= b.median_db <= b.q75_db + 1e-9
                           for b in fit.bands)
            iqr_ok = brackets and float(np.max(widths)) > 0.1
            print(f"  spread      : IQR width {float(np.min(widths)):.2f}-"
                  f"{float(np.max(widths)):.2f} dB, median inside IQR in all bands "
                  f"{'OK' if iqr_ok else 'FAIL'}")
            if not iqr_ok:
                failures.append(f"{genre}: IQR is degenerate or does not bracket the median")

            # 4. anchor reduction, and 5. --apply round trip
            apply_path = workdir / "targets_fitted.py"
            _write_fitted(fit, apply_path, quiet=True)
            reread = _load_existing_fitted(apply_path)
            anchors = reread.get("FITTED_CURVES", {}).get(genre)
            round_ok = False
            if anchors:
                rebuilt = np.interp(
                    np.log10(np.asarray(core.THIRD_OCTAVE_CENTERS, dtype=np.float64)),
                    np.log10(np.asarray([a[0] for a in anchors], dtype=np.float64)),
                    np.asarray([a[1] for a in anchors], dtype=np.float64),
                )
                resid = np.abs(rebuilt - fitted)
                clean = _clean_anchor_mask(
                    np.asarray(core.THIRD_OCTAVE_CENTERS, dtype=np.float64))
                free_resid = float(np.max(resid[clean]))
                span = CLEAN_ANCHOR_SPAN_HZ
                round_ok = free_resid <= SELF_TEST_TOL_DB
                print(f"  --apply     : {len(anchors)} anchors re-read from "
                      f"{apply_path.name}, residual {free_resid:.2f} dB over "
                      f"{_hz(span[0])}-{_hz(span[1])}  "
                      f"{'OK' if round_ok else 'FAIL'}")
                print(f"                worst overall {float(np.max(resid)):.2f} dB @ "
                      f"{_hz(float(core.THIRD_OCTAVE_CENTERS[int(np.argmax(resid))]))} "
                      f"— inherited from the grid-limited anchors, not the format")
                if not round_ok:
                    failures.append(
                        f"{genre}: anchor round-trip residual {free_resid:.2f} dB "
                        f"exceeds {SELF_TEST_TOL_DB:.1f} dB"
                    )
            else:
                failures.append(f"{genre}: --apply did not write anchors for this genre")
                print("  --apply     : FAIL — no anchors written")

            # 6. no false accusations. These files sit exactly on the hand-set
            #    curve, so the only bands the tool is entitled to call wrong are
            #    the ones where its own measurement is biased. A flag anywhere
            #    else would mean the comparison invents disagreements.
            wrong = [b for b in fit.flagged_bands if not b.is_grid_limited]
            accuse_ok = not wrong
            print(f"  flags       : {len(fit.flagged_bands)} band(s) flagged, all "
                  f"grid-limited  {'OK' if accuse_ok else 'FAIL'}")
            if not accuse_ok:
                failures.append(
                    f"{genre}: flagged {len(wrong)} grid-free band(s) that sit on the "
                    f"target curve — " + ", ".join(
                        f"{_hz(b.center_hz)} (Δ {b.delta_db:+.2f} dB)" for b in wrong[:5])
                )

            # 7. the report renders and says something
            report = render_report(fit)
            report_ok = "Fitted target curve" in report and len(report) > 2000
            print(f"  report      : {len(report):,} chars, "
                  f"{len(fit.flagged_scalars)} scalar(s) contradicted  "
                  f"{'OK' if report_ok else 'FAIL'}")
            if not report_ok:
                failures.append(f"{genre}: report did not render")

        # All three genres must have merged into one generated module.
        apply_path = workdir / "targets_fitted.py"
        merged = sorted((_load_existing_fitted(apply_path).get("FITTED_CURVES") or {}))
        expected = sorted(g for g, _ in _SELF_TEST_GENRES if g in fits)
        merge_ok = merged == expected
        print("-" * 78)
        print(f"[merge]     : targets_fitted.py holds {merged or 'nothing'}  "
              f"{'OK' if merge_ok else 'FAIL'}")
        if not merge_ok:
            failures.append(f"merge: expected {expected}, got {merged}")

        # And the generated module has to be valid, importable Python.
        try:
            spec = importlib.util.spec_from_file_location("_fitted_check", str(apply_path))
            module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(module)                 # type: ignore[union-attr]
            block = module.paste_block(expected[0]) if expected else ""
            import_ok = "_register(" in block
            print(f"[generated] : imports cleanly, paste_block('{expected[0]}') returns "
                  f"{len(block)} chars  {'OK' if import_ok else 'FAIL'}")
            if not import_ok:
                failures.append("generated module: paste_block() produced no _register call")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"generated module does not import: {type(exc).__name__}: {exc}")
            print(f"[generated] : FAIL — {type(exc).__name__}: {exc}")

        # Refusal below the floor is a feature; prove it fires.
        thin = workdir / "_thin"
        thin.mkdir(exist_ok=True)
        first = next((workdir / _SELF_TEST_GENRES[0][0]).glob("*.wav"), None)
        if first is not None:
            shutil.copy2(first, thin / first.name)
        fit_thin, err_thin = calibrate(_SELF_TEST_GENRES[0][0], thin,
                                       workers=1, quiet=True)
        refuse_ok = fit_thin is None and "at least 5 are required" in err_thin
        print(f"[min-files] : 1-file corpus refused  {'OK' if refuse_ok else 'FAIL'}")
        if not refuse_ok:
            failures.append("min-files: a 1-file corpus was not refused")

    finally:
        if keep:
            print(f"\nkept working directory: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    print("=" * 78)
    if failures:
        print(f"SELF-TEST FAILED — {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        print("=" * 78)
        return 1
    print("SELF-TEST PASSED — the harness recovers the known target curve for "
          f"{len(fits)} genres.")
    print("=" * 78)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calibrate.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Fit MixDoctor's genre targets to a corpus of commercial masters.\n"
            "Measures every usable file with the existing DSP layer, aggregates the\n"
            "corpus, and reports where analysis/targets.py disagrees with it."
        ),
        epilog=textwrap.dedent(
            """\
            examples:
              python tools/calibrate.py --genre trap --input ~/Music/trap-references/
              python tools/calibrate.py --genre pop  --input ./pop/ --out pop.md --apply
              python tools/calibrate.py --self-test

            targets.py is never modified. --apply writes analysis/targets_fitted.py,
            which is generated, importable, and meant to be read before anything is
            pasted out of it.
            """
        ),
    )
    parser.add_argument("--genre", help="Genre key or loose name: trap, Pop, 'Hip Hop'.")
    parser.add_argument("--input", type=Path,
                        help="Directory of masters, searched recursively.")
    parser.add_argument("--out", type=Path,
                        help="Write the markdown report here (default: stdout).")
    parser.add_argument("--apply", action="store_true",
                        help="Also write analysis/targets_fitted.py. Never touches targets.py.")
    parser.add_argument("--fitted-out", type=Path, default=None,
                        help="Override where --apply writes "
                             "(default: analysis/targets_fitted.py).")
    parser.add_argument("--min-files", type=int, default=DEFAULT_MIN_FILES,
                        help=f"Refuse to fit below this many usable masters "
                             f"(default: {DEFAULT_MIN_FILES}).")
    parser.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION_SEC,
                        help=f"Skip anything shorter, in seconds "
                             f"(default: {DEFAULT_MIN_DURATION_SEC:.0f}).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Measure at most N files (0 = no limit).")
    parser.add_argument("--workers", type=int, default=0,
                        help="Process pool size (0 = one per core, 1 = serial).")
    parser.add_argument("--allow-mono", action="store_true",
                        help="Admit mono files instead of skipping them.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-file progress on stderr.")
    parser.add_argument("--self-test", action="store_true",
                        help="Verify the harness against testdata/reference_*.wav "
                             "and exit.")
    parser.add_argument("--keep-temp", action="store_true",
                        help="With --self-test, keep the generated corpus for inspection.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test(workers=args.workers, keep=args.keep_temp, quiet=args.quiet)

    if not args.genre or not args.input:
        parser.error("--genre and --input are both required (or use --self-test).")

    genre_key = targets.normalise_genre(args.genre)
    if not args.quiet and genre_key != str(args.genre).strip().lower():
        print(f"  genre '{args.genre}' resolved to '{genre_key}'", file=sys.stderr)

    try:
        fit, error = calibrate(
            args.genre,
            args.input.expanduser(),
            min_files=max(1, int(args.min_files)),
            min_duration=float(args.min_duration),
            workers=int(args.workers),
            limit=int(args.limit),
            allow_mono=bool(args.allow_mono),
            quiet=bool(args.quiet),
        )
    except KeyboardInterrupt:
        print("\n  interrupted; nothing was written.", file=sys.stderr)
        return 130

    if fit is None:
        print(f"\n  {error}\n", file=sys.stderr)
        return 2

    report = render_report(fit)
    if args.out:
        out = args.out.expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        if not args.quiet:
            print(f"  wrote {out}", file=sys.stderr)
    else:
        sys.stdout.write(report)

    if args.apply:
        fitted_path = (args.fitted_out.expanduser() if args.fitted_out
                       else _BACKEND_DIR / "analysis" / "targets_fitted.py")
        _write_fitted(fit, fitted_path, quiet=bool(args.quiet))

    if not args.quiet:
        print(f"  {len(fit.flagged_bands)} band(s) and {len(fit.flagged_scalars)} "
              f"scalar(s) disagree with the hand-set targets.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:  # noqa: BLE001 — one readable traceback, not a bare crash
        traceback.print_exc()
        sys.exit(1)
