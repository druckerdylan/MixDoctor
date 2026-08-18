"""Long-term spectrum, tonal balance, and resonance measurement.

This is the one DSP module allowed to import `targets`: a band's *level* is a
measurement, but a band's *verdict* only means something against a genre curve,
and the UI wants both in the same object.

Conventions used throughout, because mixing them up is how spectral analysis
goes quietly wrong:

* Frame averaging happens in the **power** domain, never in dB. Averaging dB
  is a weighted vote that under-counts the loud passages -- exactly the
  passages a listener judges the mix by.
* "Level" of a 1/3-octave band means the **summed** power in that band, which
  makes pink noise read flat across the 31 bands. That is the convention
  `targets.target_curve()` is written in, so `third_octave_db` drops straight
  onto it after normalizing the 1 kHz band to 0 dB.
* Ratios between two *unequal-width* regions (mud vs low bass, harsh vs its
  neighbors) are computed as power **density** -- mean power per FFT bin --
  so the answer doesn't change just because one region is wider than the other.
* Everything returned is finite. NaN and inf serialise to invalid JSON, so
  every scalar goes through `_finite()` and every list through `_round_list()`.

Known limit: `core.stft_power` runs an 8192-point FFT at 48 kHz, so the bins
are 5.9 Hz apart while the 20 Hz 1/3-octave band is only 4.6 Hz wide. The three
lowest bands (20/25/31.5 Hz) therefore carry ~2 dB of quantisation error. Bands
from 40 Hz up land within ~1 dB of a synthesised reference, and the `sub` macro
band averages five bands, which washes the error back out.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from .. import targets
from ..core import (
    EPS,
    HOP_SEC,
    MACRO_BANDS,
    THIRD_OCTAVE_CENTERS,
    AudioBuffer,
    band_edges,
    band_energy,
    band_matrix,
    clamp,
    merge_spans,
    stft_power,
)
from ..types import Band, Moment, Resonance, SpectralMeasurement

__all__ = ["measure_spectral"]


# --- resonance search -------------------------------------------------------
# The LTAS is resampled onto a log-frequency grid so "one octave" is a fixed
# number of samples everywhere. 24 points/octave is finer than any resonance
# we would report and coarse enough to stay cheap.
_GRID_PPO = 24
_GRID_LO_HZ = 25.0
_GRID_HI_HZ = 18_000.0

# Analysis bandwidth of the resonance curve. 1/6 octave is narrow enough to
# keep a real resonance intact and wide enough to average over the individual
# harmonics of a bass note, which would otherwise each look like a peak.
_CURVE_REL_BW = 1.0 / 6.0
_CURVE_MIN_BW_HZ = 30.0

# Below ~80 Hz a "peak" is the kick or the bass fundamental far more often
# than it is a resonance, and low_end.py owns that region anyway. Above 12 kHz
# there is rarely enough energy for the estimate to mean anything.
_SEARCH_LO_HZ = 80.0
_SEARCH_HI_HZ = 12_000.0

_MIN_PROMINENCE_DB = 3.0
_MAX_PROMINENCE_DB = 20.0  # past this the number stops carrying information
_MAX_RESONANCES = 6
_MAX_Q = 40.0

# Anything this far under the loudest part of the spectrum is inaudible in
# context; a "resonance" down there is a measurement artefact.
_AUDIBILITY_FLOOR_DB = 45.0

# Frames quieter than this relative to the loudest frame are excluded from the
# long-term average so lead-in silence and fade-outs don't tilt the LTAS.
_FRAME_GATE_DB = 50.0

_MACRO_ORDER: Tuple[str, ...] = tuple(MACRO_BANDS.keys())
_CENTERS = np.asarray(THIRD_OCTAVE_CENTERS, dtype=np.float64)

# The 0 dB reference for the whole curve. `targets.target_curve()` is written
# "relative to the 1 kHz band", but a single 1/3-octave band is a fragile
# anchor: a mix with a notch at 1 kHz, or a sparse arrangement with nothing
# playing there, would shift all 31 numbers by however deep the hole is. So the
# reference is the power mean of the octave around 1 kHz (800/1000/1250). Every
# target curve in targets.py is flat to within 0.15 dB across that octave, so
# the two definitions agree on real material and only the degenerate case
# differs -- and there, this one is right.
_REF_INDEX = np.flatnonzero((_CENTERS >= 800.0) & (_CENTERS <= 1250.0))


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _finite(x, default: float = 0.0, lo: float = -1e6, hi: float = 1e6) -> float:
    """Coerce anything to a finite, bounded python float."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return float(min(max(v, lo), hi))


def _round_list(x: np.ndarray, nd: int = 2, lo: float = -1e6, hi: float = 1e6) -> List[float]:
    arr = np.asarray(x, dtype=np.float64).ravel()
    arr = np.nan_to_num(arr, nan=lo, posinf=hi, neginf=lo)
    return [float(v) for v in np.round(np.clip(arr, lo, hi), nd)]


def _pdb(p) -> np.ndarray:
    """Power -> dB with a hard floor. Never -inf, never NaN."""
    return 10.0 * np.log10(np.maximum(np.asarray(p, dtype=np.float64), EPS))


def _logistic(x: float, x0: float, k: float) -> float:
    """Smooth 0-1 map. `x0` is the halfway point, `k` the softness in x units."""
    z = clamp((x - x0) / max(k, 1e-6), -40.0, 40.0)
    return float(1.0 / (1.0 + np.exp(-z)))


def _patch_empty(mat: np.ndarray, freqs: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Give every band at least one FFT bin.

    At 48 kHz with an 8192-point FFT the bins are 5.9 Hz apart, which is wider
    than the 20 Hz 1/3-octave band (17.8-22.4 Hz) and wider than any of the
    narrow log-grid bands down low. Those rows come back all-zero from
    `band_matrix`, which would read as digital silence and drag the sub band
    and the tilt regression down with it. Assign the nearest bin instead.
    """
    empty = np.flatnonzero(mat.sum(axis=1) == 0.0)
    if empty.size:
        nearest = np.argmin(np.abs(freqs[None, :] - centers[empty][:, None]), axis=1)
        mat[empty, nearest] = 1.0
    return mat


def _density_matrix(freqs: np.ndarray, ranges: Sequence[Tuple[float, float]]) -> np.ndarray:
    """(n_ranges, n_bins) matrix where `M @ power` is mean power per bin."""
    lo = np.asarray([r[0] for r in ranges], dtype=np.float64)
    hi = np.asarray([r[1] for r in ranges], dtype=np.float64)
    mat = ((freqs[None, :] >= lo[:, None]) & (freqs[None, :] < hi[:, None])).astype(np.float64)
    mat = _patch_empty(mat, freqs, np.sqrt(lo * hi))
    return mat / mat.sum(axis=1, keepdims=True)


def _log_center(lo: float, hi: float) -> float:
    return float(np.sqrt(max(lo, EPS) * max(hi, EPS)))


# ---------------------------------------------------------------------------
# Psychoacoustics
# ---------------------------------------------------------------------------


def _bark(f: np.ndarray | float) -> np.ndarray:
    """Traunmuller/Zwicker critical-band rate in Bark."""
    f = np.asarray(f, dtype=np.float64)
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def _sharpness_acum(band_power: np.ndarray, edges: np.ndarray) -> float:
    """Zwicker-style sharpness from 1/3-octave band powers.

    S = 0.11 * integral(N'(z) g(z) z dz) / integral(N'(z) dz), with the standard
    weighting g(z) = 1 below 16 Bark and 0.066 exp(0.171 z) above (the two
    branches meet at z = 16). Specific loudness is approximated as
    N' ~ (E - E_thr)^0.23 -- the 0.23 compression exponent of the Zwicker
    model, with a relative threshold subtracted so bands buried 60 dB under the
    loudest one stop contributing. Because both integrals scale with the same
    power of any gain applied to the signal, the result is level-independent,
    which is what we want from an uncalibrated digital file.
    """
    z_lo, z_hi = _bark(edges[:, 0]), _bark(edges[:, 1])
    z_c = 0.5 * (z_lo + z_hi)
    dz = np.maximum(z_hi - z_lo, 1e-4)

    peak = float(np.max(band_power)) if band_power.size else 0.0
    if peak <= EPS:
        return 0.0
    excitation = np.maximum(band_power - peak * 1e-6, 0.0)
    n_spec = np.power(excitation / peak, 0.23)

    denom = float(np.sum(n_spec * dz))
    if denom <= 1e-9:
        return 0.0
    g = np.where(z_c < 16.0, 1.0, 0.066 * np.exp(0.171 * z_c))
    numer = float(np.sum(n_spec * g * z_c * dz))
    return _finite(0.11 * numer / denom, 0.0, 0.0, 20.0)


# ---------------------------------------------------------------------------
# Resonance curve
# ---------------------------------------------------------------------------


def _log_grid() -> np.ndarray:
    n = int(round(np.log2(_GRID_HI_HZ / _GRID_LO_HZ) * _GRID_PPO)) + 1
    return _GRID_LO_HZ * 2.0 ** (np.arange(n) / _GRID_PPO)


def _curve_matrix(freqs: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Constant-relative-bandwidth analysis matrix on the log grid.

    Rows are normalized to mean-power-per-bin and then re-scaled by the band's
    width in Hz, so the resulting curve is band *energy* at constant relative
    bandwidth: flat for pink noise, which is the right baseline for spotting a
    bump. The `_CURVE_MIN_BW_HZ` floor keeps the low bands wide enough to span
    the harmonic spacing of a bass line, so individual partials don't read as
    resonances.
    """
    half = 2.0 ** (_CURVE_REL_BW / 2.0)
    lo = np.minimum(grid / half, grid - _CURVE_MIN_BW_HZ * 0.5)
    hi = np.maximum(grid * half, grid + _CURVE_MIN_BW_HZ * 0.5)
    mat = ((freqs[None, :] >= lo[:, None]) & (freqs[None, :] < hi[:, None])).astype(np.float64)
    mat = _patch_empty(mat, freqs, grid)
    mat /= mat.sum(axis=1, keepdims=True)
    return mat * (hi - lo)[:, None]


def _crossing(curve: np.ndarray, peak: int, target: float, step: int) -> float:
    """Fractional index where `curve` first falls to `target` walking outward."""
    n = curve.shape[0]
    idx = peak
    while 0 <= idx + step < n and curve[idx + step] >= target:
        idx += step
    edge = idx + step
    if edge < 0 or edge >= n:
        return float(idx)
    span = curve[idx] - curve[edge]
    if span <= 1e-9:
        return float(idx)
    frac = (curve[idx] - target) / span
    return float(idx + step * clamp(frac, 0.0, 1.0))


def _resonance_severity(prominence_db: float, q: float) -> str:
    """Narrow *and* loud is the worst case; broad-and-loud is tonal balance."""
    score = prominence_db * (0.55 + 0.45 * clamp(q / 6.0, 0.0, 1.0))
    if score >= 10.0:
        return "critical"
    if score >= 6.5:
        return "major"
    if score >= 3.5:
        return "minor"
    return "clean"


def _tracking_ranges(freq_hz: float) -> List[Tuple[float, float]]:
    """The narrow band a resonance lives in, and the octave it stands out from."""
    return [
        (
            min(freq_hz / 2.0 ** (1.0 / 12.0), freq_hz - _CURVE_MIN_BW_HZ * 0.5),
            max(freq_hz * 2.0 ** (1.0 / 12.0), freq_hz + _CURVE_MIN_BW_HZ * 0.5),
        ),
        (max(freq_hz * 0.5, 20.0), min(freq_hz * 2.0, 22_000.0)),
    ]


def _resonance_moments(
    freq_hz: float,
    prominence_db: float,
    ratio: np.ndarray,
    times: np.ndarray,
) -> List[Moment]:
    """When is this resonance worst?

    `ratio` is the peak's level against its own one-octave surroundings, frame
    by frame, so a resonance that only rings out on the sustained notes
    separates from one baked into the whole track. A resonance that never
    varies is not a resonance without moments -- it is one whose moment is the
    entire track, and the timeline has to be able to say so.
    """
    if times.size == 0 or ratio.size == 0:
        return []

    median = float(np.median(ratio))
    thresh = max(float(np.percentile(ratio, 75.0)), median + 0.5)
    mask = ratio >= thresh
    if not np.any(mask) or times.size < 3:
        # Dead flat over time (or too few frames to have a "when"): the whole
        # track is the moment.
        mask = ratio >= float(np.percentile(ratio, 90.0))
    if not np.any(mask):
        return []

    spans = merge_spans(times, mask, min_gap=0.75, min_len=HOP_SEC)
    spread = max(float(np.percentile(ratio, 98.0)) - median, 1.0)
    # A steady resonance still deserves a visible marker, so intensity mixes
    # how bad the peak is overall with how much worse it gets right here.
    base = clamp(prominence_db / _MAX_PROMINENCE_DB, 0.0, 1.0)

    moments: List[Moment] = []
    for t0, t1 in spans:
        window = (times >= t0 - 1e-6) & (times <= t1 + 1e-6)
        if not np.any(window):
            continue
        worst = float(np.max(ratio[window]))
        temporal = clamp((worst - median) / spread, 0.0, 1.0)
        moments.append(
            Moment(
                t_start=_finite(t0, 0.0, 0.0, 1e5),
                t_end=_finite(max(t1, t0 + HOP_SEC), 0.0, 0.0, 1e5),
                intensity=clamp(0.45 * base + 0.55 * temporal, 0.1, 1.0),
                value=_finite(round(worst, 2), 0.0, -200.0, 200.0),
                label=f"{freq_hz:.0f} Hz peak",
            )
        )
    moments.sort(key=lambda m: m.intensity, reverse=True)
    return moments[:4]


def _find_resonances(
    grid: np.ndarray,
    curve_db: np.ndarray,
    freqs: np.ndarray,
    power: np.ndarray,
    times: np.ndarray,
) -> Tuple[List[Resonance], np.ndarray]:
    """Peak-pick the log-frequency LTAS against an octave-smoothed baseline.

    The baseline is a one-octave running **median**, not a mean: a mean is
    inflated by the very peak it is supposed to be a reference for, which
    shrinks strong resonances toward the threshold. A median ignores them.

    Returns the resonances and the prominence curve (reused by the harshness
    and sibilance indices).
    """
    n = curve_db.shape[0]
    # Synthetic and heavily band-limited material can have 80 dB gaps between
    # partials. Flooring the curve keeps the baseline from collapsing into a
    # hole and reporting an absurd prominence.
    floored = np.maximum(curve_db, float(np.max(curve_db)) - 55.0)
    baseline = median_filter(floored, size=_GRID_PPO + 1, mode="nearest")
    prominence = np.clip(floored - baseline, 0.0, _MAX_PROMINENCE_DB)

    # 1/4 octave apart is the closest two peaks can be and still be two peaks.
    # Wider than this and a twin-peak buildup (say 250 Hz *and* 310 Hz) gets
    # reported as one, hiding half of what the producer has to fix.
    peaks, _ = find_peaks(prominence, height=_MIN_PROMINENCE_DB, distance=max(_GRID_PPO // 4, 1))
    if peaks.size == 0:
        return [], prominence

    audible = float(np.max(curve_db)) - _AUDIBILITY_FLOOR_DB
    in_range = (
        (grid[peaks] >= _SEARCH_LO_HZ)
        & (grid[peaks] <= _SEARCH_HI_HZ)
        & (curve_db[peaks] >= audible)
    )
    peaks = peaks[in_range]
    if peaks.size == 0:
        return [], prominence

    order = np.argsort(prominence[peaks])[::-1]
    peaks = peaks[order][: _MAX_RESONANCES * 2]

    # Q and severity first, moments second: locating a resonance in time costs
    # a pass over the spectrogram, and only the survivors are worth it.
    scored: List[Tuple[float, int, float, float]] = []
    for idx in peaks:                       # at most 12 iterations, not per-sample
        idx = int(idx)
        prom = float(prominence[idx])
        target = prom - 3.0
        left = _crossing(prominence, idx, target, -1)
        right = _crossing(prominence, idx, target, +1)
        width_oct = max((right - left) / _GRID_PPO, 1.0 / (2.0 * _GRID_PPO))
        bw_ratio = 2.0 ** (width_oct * 0.5) - 2.0 ** (-width_oct * 0.5)
        q = clamp(1.0 / max(bw_ratio, 1e-6), 0.3, _MAX_Q)
        scored.append((prom * (0.55 + 0.45 * clamp(q / 6.0, 0.0, 1.0)), idx, prom, q))

    scored.sort(key=lambda s: s[0], reverse=True)
    scored = scored[:_MAX_RESONANCES]

    # One pass over the spectrogram for every resonance at once. Six separate
    # (n_frames x n_bins) matmuls is six times the memory traffic for the same
    # answer, and on a 6-minute track that is the dominant cost in this module.
    ranges: List[Tuple[float, float]] = []
    for _, idx, _, _ in scored:
        ranges.extend(_tracking_ranges(float(grid[idx])))
    tracked = power @ _density_matrix(freqs, ranges).T      # (n_frames, 2k)
    ratios = _pdb(tracked[:, 0::2]) - _pdb(tracked[:, 1::2])
    ratios = np.nan_to_num(ratios, nan=0.0, posinf=0.0, neginf=0.0)

    found: List[Resonance] = []
    for n, (_, idx, prom, q) in enumerate(scored):
        freq = float(grid[idx])
        found.append(
            Resonance(
                freq_hz=_finite(round(freq, 1), 1000.0, 10.0, 24_000.0),
                q=_finite(round(q, 2), 1.0, 0.1, _MAX_Q),
                prominence_db=_finite(round(prom, 2), 0.0, 0.0, _MAX_PROMINENCE_DB),
                severity=_resonance_severity(prom, q),  # type: ignore[arg-type]
                moments=_resonance_moments(freq, prom, ratios[:, n], times),
            )
        )
    return found, prominence


# ---------------------------------------------------------------------------
# Empty / degenerate result
# ---------------------------------------------------------------------------


def _empty_measurement(genre: str) -> SpectralMeasurement:
    """A fully-populated, all-finite result for audio with nothing in it."""
    bands: List[Band] = []
    for name in _MACRO_ORDER:
        lo, hi = MACRO_BANDS[name]
        target = _finite(targets.macro_target(genre, name, lo, hi))
        bands.append(
            Band(
                name=name,
                low_hz=float(lo),
                high_hz=float(hi),
                center_hz=round(_log_center(lo, hi), 1),
                level_db=target,
                target_db=target,
                deviation_db=0.0,
                verdict="good",
            )
        )
    return SpectralMeasurement(
        third_octave_centers=[float(c) for c in _CENTERS],
        third_octave_db=[0.0] * _CENTERS.size,
        bands=bands,
        spectral_tilt_db_per_decade=0.0,
        spectral_centroid_hz=0.0,
        mud_ratio_db=0.0,
        mud_to_mid_db=0.0,
        boxiness_db=0.0,
        harshness_index=0.0,
        sibilance_index=0.0,
        sharpness_acum=0.0,
        resonances=[],
        band_series={name: [] for name in _MACRO_ORDER},
        times=[],
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def measure_spectral(buf: AudioBuffer, genre: str) -> SpectralMeasurement:
    """Measure the mix's tonal balance, coloration and resonances.

    `genre` only selects the target curve the macro bands are scored against;
    every raw number in the result is genre-independent.
    """
    x = np.asarray(buf.mono, dtype=np.float64)
    if x.size:
        x = x - float(np.mean(x))          # DC offset is not sub-bass
    sr = int(buf.sr)

    freqs, times, power = stft_power(x, sr)
    if times.size == 0 or power.size == 0:
        return _empty_measurement(genre)
    # Below this the normal path still works: every level floors at -120 dB,
    # every ratio comes out 0, and the caller gets aligned series instead of a
    # special case it has to know about.

    # -- long-term average spectrum ------------------------------------------
    # Average POWER over frames, then take dB. The other order silently
    # under-weights the loudest passages.
    frame_power = power.sum(axis=1)
    gate = frame_power >= float(np.max(frame_power)) * 10.0 ** (-_FRAME_GATE_DB / 10.0)
    if np.count_nonzero(gate) < 1:
        gate = np.ones_like(frame_power, dtype=bool)
    # `gate @ power` rather than `power[gate].mean(0)`: fancy-indexing a
    # 6-minute spectrogram copies ~50 MB to compute one 4097-point average.
    weights = gate.astype(np.float64)
    mean_power = (weights @ power) / float(weights.sum())

    edges = band_edges(THIRD_OCTAVE_CENTERS)
    tmat = _patch_empty(band_matrix(freqs, edges), freqs, _CENTERS)
    toct_frames = band_energy(power, tmat)                 # (n_frames, 31)
    toct_mean = np.asarray(band_energy(mean_power[None, :], tmat)[0], dtype=np.float64)

    # Normalize to the 1 kHz band so the curve is directly comparable with
    # targets.target_curve(). Guard against a mix with nothing at 1 kHz.
    peak_band = float(np.max(toct_mean))
    ref = max(float(np.mean(toct_mean[_REF_INDEX])), peak_band * 1e-9, EPS)
    ref_db = 10.0 * np.log10(ref)
    # Digital silence still produces frames and aligned series, but a 0-1
    # "how harsh is it" score on nothing at all is noise, so it reads 0.
    has_signal = 1.0 if peak_band > EPS else 0.0
    third_octave_db = np.clip(_pdb(toct_mean) - ref_db, -90.0, 40.0)

    # -- macro bands ---------------------------------------------------------
    # macro_target() averages the target curve in the power domain over the
    # 1/3-octave bands it spans, so the measured level must be the mean band
    # power over the same bands for the two to be comparable.
    bands: List[Band] = []
    for name in _MACRO_ORDER:
        lo, hi = MACRO_BANDS[name]
        mask = (_CENTERS >= lo) & (_CENTERS < hi)
        if np.any(mask):
            level = float(np.mean(toct_mean[mask]))
            level_db = clamp(10.0 * np.log10(max(level, EPS)) - ref_db, -90.0, 40.0)
        else:
            level_db = 0.0
        target_db = _finite(targets.macro_target(genre, name, lo, hi))
        deviation = level_db - target_db
        tol = max(_finite(targets.band_tolerance(genre, name), 3.0), 0.1)
        if abs(deviation) <= tol:
            verdict = "good"
        elif abs(deviation) <= 2.0 * tol:
            verdict = "watch"
        else:
            verdict = "problem"
        bands.append(
            Band(
                name=name,
                low_hz=float(lo),
                high_hz=float(hi),
                center_hz=round(_log_center(lo, hi), 1),
                level_db=round(_finite(level_db), 2),
                target_db=round(target_db, 2),
                deviation_db=round(_finite(deviation), 2),
                verdict=verdict,  # type: ignore[arg-type]
            )
        )

    # -- tilt ----------------------------------------------------------------
    # Fit only where there is signal: a brick-wall lowpass or a dead 20 kHz
    # band would otherwise dominate the regression and invent a 40 dB tilt.
    tilt = 0.0
    fit_mask = (
        (_CENTERS >= 40.0)
        & (_CENTERS <= 16_000.0)
        & (third_octave_db > float(np.max(third_octave_db)) - 45.0)
    )
    if np.count_nonzero(fit_mask) >= 5:
        slope = np.polyfit(np.log10(_CENTERS[fit_mask]), third_octave_db[fit_mask], 1)[0]
        tilt = _finite(slope, 0.0, -60.0, 40.0)

    # -- centroid ------------------------------------------------------------
    audible = (freqs >= 20.0) & (freqs <= 20_000.0)
    total = float(np.sum(mean_power[audible]))
    centroid = (
        float(np.sum(freqs[audible] * mean_power[audible]) / total) if total > EPS else 0.0
    )

    # -- region ratios (power density, so region width cancels) --------------
    regions: Dict[str, Tuple[float, float]] = {
        "mud": (150.0, 400.0),          # core.MUD_HZ
        "low_bass": (60.0, 120.0),
        "mid": (1000.0, 3000.0),
        "harsh": (2000.0, 5000.0),      # core.HARSH_HZ
        "harsh_lo": (1000.0, 2000.0),
        "harsh_hi": (5000.0, 8000.0),
        "sib": (5000.0, 9000.0),        # core.SIBILANCE_HZ
        "sib_lo": (3000.0, 5000.0),
        "sib_hi": (9000.0, 13000.0),
        "sib_ref": (1000.0, 5000.0),
    }
    names = list(regions.keys())
    dmat = _density_matrix(freqs, [regions[k] for k in names])
    dens_db = {k: float(v) for k, v in zip(names, _pdb(dmat @ mean_power))}

    mud_ratio_db = _finite(dens_db["mud"] - dens_db["low_bass"], 0.0, -80.0, 80.0)
    mud_to_mid_db = _finite(dens_db["mud"] - dens_db["mid"], 0.0, -80.0, 80.0)

    # Boxiness: 300-600 Hz against the median 1/3-octave level. The median is
    # the mix's own broadband reference and, unlike a mean, isn't dragged
    # around by a big sub or a rolled-off top.
    boxy_mask = (_CENTERS >= 300.0) & (_CENTERS < 600.0)     # core.BOXY_HZ
    content = third_octave_db[(_CENTERS >= 40.0) & (_CENTERS <= 16_000.0)]
    broadband_db = float(np.median(content)) if content.size else 0.0
    if np.any(boxy_mask):
        boxy_db = 10.0 * np.log10(max(float(np.mean(toct_mean[boxy_mask])), EPS)) - ref_db
    else:
        boxy_db = broadband_db
    boxiness_db = _finite(boxy_db - broadband_db, 0.0, -60.0, 60.0)

    # -- resonances ----------------------------------------------------------
    grid = _log_grid()
    curve_db = _pdb(_curve_matrix(freqs, grid) @ mean_power)
    resonances, prominence = _find_resonances(grid, curve_db, freqs, power, times)

    # -- harshness -----------------------------------------------------------
    # Not "is 2-5 kHz loud" but "does 2-5 kHz stick out of the curve its own
    # neighbors describe". A bright mix runs a smooth slope through here; a
    # harsh one has a shelf or a spike sitting on top of it.
    harsh_excess = _neighbour_excess(
        dens_db["harsh"], regions["harsh"],
        dens_db["harsh_lo"], regions["harsh_lo"],
        dens_db["harsh_hi"], regions["harsh_hi"],
    )
    harsh_peak = _peak_prominence(grid, prominence, 2000.0, 5000.0)
    harsh_raw = harsh_excess + min(0.5 * max(harsh_peak - 4.0, 0.0), 6.0)
    harshness_index = has_signal * clamp(_logistic(harsh_raw, 4.0, 2.5), 0.0, 1.0)

    # -- sibilance -----------------------------------------------------------
    # Steady 5-9 kHz is air. Sibilance is the same energy arriving in bursts,
    # so the average alone can't tell them apart -- the frame-to-frame spread
    # of the 5-9 kHz level is what separates an "ess" from a shimmer.
    sib_excess = _neighbour_excess(
        dens_db["sib"], regions["sib"],
        dens_db["sib_lo"], regions["sib_lo"],
        dens_db["sib_hi"], regions["sib_hi"],
    )
    sib_mat = _density_matrix(freqs, [regions["sib"], regions["sib_ref"]])
    sib_frames = (power @ sib_mat.T)[gate]     # project first, then select
    sib_series = _pdb(sib_frames[:, 0]) - _pdb(sib_frames[:, 1])
    sib_series = np.nan_to_num(sib_series, nan=0.0, posinf=0.0, neginf=0.0)
    if sib_series.size >= 4:
        burst = float(np.percentile(sib_series, 90.0) - np.percentile(sib_series, 50.0))
    else:
        burst = 0.0
    sib_raw = sib_excess + min(0.7 * max(burst - 2.0, 0.0), 6.0)
    sibilance_index = has_signal * clamp(_logistic(sib_raw, 4.0, 2.5), 0.0, 1.0)

    # -- sharpness -----------------------------------------------------------
    sharpness = _sharpness_acum(toct_mean, edges)

    # -- per-band time series ------------------------------------------------
    # Absolute band level (dBFS-referenced), one value per STFT frame, all
    # series the same length as `times` and on HOP_SEC spacing.
    band_series: Dict[str, List[float]] = {}
    for name in _MACRO_ORDER:
        lo, hi = MACRO_BANDS[name]
        mask = (_CENTERS >= lo) & (_CENTERS < hi)
        if np.any(mask):
            series = _pdb(np.mean(toct_frames[:, mask], axis=1))
        else:
            series = np.full(times.shape[0], -120.0)
        band_series[name] = _round_list(np.maximum(series, -120.0), 2, -120.0, 40.0)

    return SpectralMeasurement(
        third_octave_centers=[float(c) for c in _CENTERS],
        third_octave_db=_round_list(third_octave_db, 2, -90.0, 40.0),
        bands=bands,
        spectral_tilt_db_per_decade=round(tilt, 2),
        spectral_centroid_hz=round(_finite(centroid, 0.0, 0.0, 24_000.0), 1),
        mud_ratio_db=round(mud_ratio_db, 2),
        mud_to_mid_db=round(mud_to_mid_db, 2),
        boxiness_db=round(boxiness_db, 2),
        harshness_index=round(_finite(harshness_index, 0.0, 0.0, 1.0), 3),
        sibilance_index=round(_finite(sibilance_index, 0.0, 0.0, 1.0), 3),
        sharpness_acum=round(_finite(sharpness, 0.0, 0.0, 20.0), 3),
        resonances=resonances,
        band_series=band_series,
        times=_round_list(times, 3, 0.0, 1e5),
    )


def _neighbour_excess(
    level_db: float,
    region: Tuple[float, float],
    low_db: float,
    low_region: Tuple[float, float],
    high_db: float,
    high_region: Tuple[float, float],
) -> float:
    """How far a region sits above the line its two neighbors draw.

    Interpolating in log-frequency means a mix with an ordinary downward tilt
    scores ~0 here: only a genuine bump registers. The upper neighbor is
    clamped so a band-limited or lowpassed track can't turn a missing top end
    into a fake 30 dB peak.
    """
    high_db = max(high_db, low_db - 30.0)
    g_lo = np.log10(_log_center(*low_region))
    g_hi = np.log10(_log_center(*high_region))
    g_t = np.log10(_log_center(*region))
    span = g_hi - g_lo
    if abs(span) < 1e-9:
        expected = 0.5 * (low_db + high_db)
    else:
        w = clamp(float((g_t - g_lo) / span), 0.0, 1.0)
        expected = low_db + w * (high_db - low_db)
    return _finite(level_db - expected, 0.0, -60.0, 60.0)


def _peak_prominence(grid: np.ndarray, prominence: np.ndarray, lo: float, hi: float) -> float:
    """Largest prominence found inside a frequency window, in dB."""
    mask = (grid >= lo) & (grid <= hi)
    if not np.any(mask):
        return 0.0
    return _finite(float(np.max(prominence[mask])), 0.0, 0.0, _MAX_PROMINENCE_DB)
