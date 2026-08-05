"""Stereo-field measurement: correlation, width and mono fold-down survival.

This module answers "what *is* the stereo image", never "is it good". Genre
windows and verdicts belong to the detector layer.

Two numbers here matter more than the broadband headline:

  * **Per-band correlation.** A mix can read +0.7 broadband and still have its
    sub smeared across the field, because the sub carries a small share of the
    total energy and the broadband figure is energy-weighted. Only the band
    breakdown catches it.
  * **Per-band mono-sum loss.** Low-frequency cancellation is what gets a
    record rejected from a vinyl cut and what makes a mix collapse on a club
    system, where a lot of the rig is effectively mono.

Everything is second-order statistics, so every measurement reduces to three
dot products per band (E[L²], E[R²], E[LR]) plus five strided frame sums for
the time series. No Python loop ever touches a sample.

Sign conventions
----------------
``mono_sum_loss_db``   level of (L+R)/2 against the average channel level.
                       0 dB = perfectly correlated, -3 dB = fully decorrelated
                       (the normal, harmless cost of summing), very negative =
                       cancellation.
``balance_db``         +ve means right-heavy.
``low_end_side_energy_db``  side vs mid below 120 Hz. Very negative = mono bass.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
from scipy import signal as sps

from ..core import (
    EPS,
    FRAME_SEC,
    HOP_SEC,
    MACRO_BANDS,
    MACRO_BAND_ORDER,
    AudioBuffer,
    bandpass,
    frame_signal,
    frame_times,
)
from ..types import StereoMeasurement

__all__ = ["measure_stereo"]

# Ratios are reported in dB against a readable floor rather than -120: a
# "-58 dB" side channel and a "-inf" one mean the same thing to a producer,
# and the finite value keeps the JSON valid.
DB_FLOOR = -60.0

# Side/Mid can legitimately run away when the mid is nearly cancelled (an
# inverted channel gives ~12x). Cap it so the number stays plottable.
WIDTH_MAX = 20.0

# Mean-square below this is numerical residue from the filter, not signal
# (~-140 dBFS RMS). Bands under it report neutral rather than noise-driven
# correlations.
ENERGY_FLOOR = 1e-14

# Everything below here is expected to be effectively mono on a real system.
LOW_END_HZ = 120.0

# Band-appropriate decimation. Nine bands cost 18 zero-phase filter passes,
# and sosfiltfilt is a sequential recursion that neither vectorises across
# channels nor parallelises — measured, a 2-D (2, N) call is no faster than
# two 1-D calls. The only lever is giving the low bands less data: a 20-60 Hz
# band carried at 48 kHz is 400x oversampled for its content.
#
# Ratios are what make this safe. Everything reported here is a ratio of
# second moments (correlation, side/mid, mid/average), and mean-square is
# average *power*, independent of the rate it was measured at — so the working
# rate cancels out. Verified band-by-band against full-rate filtering.
#
# One tier, not a ladder. A 2/4/8 cascade held ~240 MB of intermediate copies
# for a 6-minute track, and its middle rungs each served a single band: the
# 24 kHz rung saved half a filter pass and cost a resample of the same size —
# a net loss before counting the memory. The 8x rung alone carries the three
# genuinely oversampled bands, is built in one step, and leaves peak memory
# unchanged. Measured on a 6-minute stereo track: 11.1 s -> 9.0 s.
_DECIMATION_STEPS = (8,)

# A band must end below this fraction of its working rate, so it is moved only
# where it is *heavily* oversampled. The limit is deliberately strict: at
# 48 kHz it admits sub, low_bass and upper_bass and nothing else.
#
# Butterworth edges are placed by the bilinear transform, which warps them by
# an amount depending on where they sit relative to Nyquist, so redesigning a
# band at a lower rate shifts its skirts slightly. That is invisible in an
# ordinary band, but where the channels nearly cancel the surviving mid is a
# small difference of large numbers and a skirt shift moves it a lot: at 0.30
# headroom mix_phase's upper_mid width drifted 2.7% (5.608 -> 5.458). At 0.06
# every moved band sits under 0.12 of its working Nyquist and the worst drift
# across all fixtures is 0.001 correlation / 0.012 width / 0.02 dB, below the
# precision this module reports.
_BAND_HEADROOM = 0.06

# Only decimate material long enough for the saving to matter and for the
# resampler's edge transient to be negligible. Short files are already fast
# (tens of ms), so there is nothing to buy and accuracy to lose.
_MIN_DECIMATE_SEC = 5.0


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _finite(value: float, default: float = 0.0) -> float:
    """NaN/inf -> a sensible finite value. These serialise to invalid JSON."""
    v = float(value)
    return v if math.isfinite(v) else float(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(min(max(value, lo), hi))


def _ratio_db(numer_ms: float, denom_ms: float) -> float:
    """Power ratio in dB: floored, finite, and *directional* when a side is empty.

    The empty-denominator case has to pin to the ceiling rather than collapse
    to 0 dB. A dead channel is one of the most important things this module can
    catch, and 0 dB on `balance_db` would read as "perfectly balanced" — the
    single most misleading answer available.
    """
    num_live = numer_ms > ENERGY_FLOOR
    den_live = denom_ms > ENERGY_FLOOR
    if not num_live and not den_live:
        return 0.0  # nothing on either side: there is no ratio to report
    if not den_live:
        return -DB_FLOOR  # numerator only
    if not num_live:
        return DB_FLOOR  # denominator only
    ratio = numer_ms / denom_ms
    return _clamp(_finite(10.0 * math.log10(ratio), DB_FLOOR), DB_FLOOR, -DB_FLOOR)


def _safe_bandpass(x: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    """core.bandpass, but a filter that refuses a short input yields silence.

    sosfiltfilt needs a few times the filter order in samples. A 1.2 s file is
    fine, but we never want a valid upload to raise out of the DSP layer.
    """
    try:
        return bandpass(x, sr, lo, hi)
    except Exception:
        return np.zeros_like(x)


def _decimation(hi_hz: float, sr: int) -> int:
    """Largest supported decimation that still holds a band ending at `hi_hz`."""
    for q in _DECIMATION_STEPS:
        if sr % q == 0 and hi_hz <= _BAND_HEADROOM * (sr / q):
            return q
    return 1


def _working_pair(
    cache: Dict[int, Tuple[np.ndarray, np.ndarray, int]], q: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    """L/R decimated by `q` in a single polyphase step, memoised.

    Decimating straight from full rate means no intermediate rate is ever
    allocated, which keeps the extra footprint to one 1/q-size copy per
    channel. Any failure falls back to full rate — a slower correct answer
    beats an exception escaping the DSP layer.
    """
    if q in cache:
        return cache[q]

    full_l, full_r, full_sr = cache[1]
    try:
        if q < 2 or full_sr % q != 0 or full_l.size < _MIN_DECIMATE_SEC * full_sr:
            raise ValueError("not worth decimating")
        pair = (
            np.ascontiguousarray(sps.resample_poly(full_l, 1, q)),
            np.ascontiguousarray(sps.resample_poly(full_r, 1, q)),
            full_sr // q,
        )
    except Exception:
        pair = cache[1]

    cache[q] = pair
    return pair


def _moments(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """(E[x²], E[y²], E[xy]) via BLAS dot products — no temporaries."""
    n = float(max(x.size, 1))
    return (
        float(np.dot(x, x)) / n,
        float(np.dot(y, y)) / n,
        float(np.dot(x, y)) / n,
    )


def _pair_stats(ll: float, rr: float, lr: float) -> Tuple[float, float, float]:
    """Correlation, width and mono-sum loss from the three second moments.

    Mid/side energies come straight out of the moments:
        E[M²] = (E[L²] + 2E[LR] + E[R²]) / 4
        E[S²] = (E[L²] - 2E[LR] + E[R²]) / 4
    which is why no separate M/S filtering is needed.
    """
    avg_ms = (ll + rr) * 0.5
    if avg_ms <= ENERGY_FLOOR:
        # Nothing measurable in this band. Report neutral, not noise.
        return 1.0, 0.0, 0.0

    mid_ms = max((ll + 2.0 * lr + rr) * 0.25, 0.0)
    side_ms = max((ll - 2.0 * lr + rr) * 0.25, 0.0)

    denom = math.sqrt(max(ll, 0.0) * max(rr, 0.0))
    corr = _clamp(_finite(lr / denom, 1.0), -1.0, 1.0) if denom > EPS else 1.0

    if mid_ms > ENERGY_FLOOR:
        width = _clamp(_finite(math.sqrt(side_ms / mid_ms), 0.0), 0.0, WIDTH_MAX)
    else:
        width = WIDTH_MAX if side_ms > ENERGY_FLOOR else 0.0

    mono_loss_db = _ratio_db(mid_ms, avg_ms)
    return corr, width, mono_loss_db


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------


def _series(
    left: np.ndarray, right: np.ndarray, sr: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame correlation and width on the core HOP_SEC grid.

    Five strided reductions carry every statistic we need. `einsum` reduces
    along the frame axis without materialising the (n_frames, frame_len)
    products, which for a 6-minute track would be a few hundred MB each.
    """
    frame_len = max(int(FRAME_SEC * sr), 16)
    hop = max(int(HOP_SEC * sr), 1)

    f_l = frame_signal(left, frame_len, hop)
    f_r = frame_signal(right, frame_len, hop)
    n_frames = f_l.shape[0]
    times = frame_times(n_frames, frame_len, hop, sr)
    if n_frames == 0:
        return times, np.zeros(0), np.zeros(0)

    n = float(frame_len)
    m_l = f_l.sum(axis=1) / n
    m_r = f_r.sum(axis=1) / n
    ll = np.einsum("ij,ij->i", f_l, f_l) / n
    rr = np.einsum("ij,ij->i", f_r, f_r) / n
    lr = np.einsum("ij,ij->i", f_l, f_r) / n

    # Mean-removed: a frame can carry real DC from a slow LF component, and an
    # uncentred correlation would read it as agreement between the channels.
    var_l = np.maximum(ll - m_l * m_l, 0.0)
    var_r = np.maximum(rr - m_r * m_r, 0.0)
    cov = lr - m_l * m_r

    denom = np.sqrt(var_l * var_r)
    quiet = denom <= ENERGY_FLOOR
    corr = np.divide(cov, denom, out=np.ones_like(cov), where=~quiet)
    # A silent frame has no phase relationship; 1.0 keeps it out of the
    # problem-moment mask instead of inventing a defect in a gap.
    corr = np.where(quiet, 1.0, corr)
    corr = np.clip(np.nan_to_num(corr, nan=1.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)

    mid_ms = np.maximum((var_l + 2.0 * cov + var_r) * 0.25, 0.0)
    side_ms = np.maximum((var_l - 2.0 * cov + var_r) * 0.25, 0.0)
    width = np.sqrt(np.divide(side_ms, mid_ms, out=np.zeros_like(mid_ms), where=mid_ms > ENERGY_FLOOR))
    width = np.where((mid_ms <= ENERGY_FLOOR) & (side_ms > ENERGY_FLOOR), WIDTH_MAX, width)
    width = np.clip(np.nan_to_num(width, nan=0.0, posinf=WIDTH_MAX, neginf=0.0), 0.0, WIDTH_MAX)

    return times, corr, width


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def measure_stereo(buf: AudioBuffer) -> StereoMeasurement:
    """Measure the stereo field of `buf`.

    A mono source is reported as mono with honest neutral values — never as a
    stereo defect. A file that never had a stereo field cannot have lost one.
    """
    sr = buf.sr
    bands = list(MACRO_BAND_ORDER)

    frame_len = max(int(FRAME_SEC * sr), 16)
    hop = max(int(HOP_SEC * sr), 1)

    if buf.is_mono:
        n_frames = frame_signal(buf.left, frame_len, hop).shape[0]
        times = frame_times(n_frames, frame_len, hop, sr)
        return StereoMeasurement(
            is_mono_source=True,
            correlation=1.0,
            width=0.0,
            band_correlation={b: 1.0 for b in bands},
            band_width={b: 0.0 for b in bands},
            mono_sum_loss_db=0.0,
            band_mono_loss_db={b: 0.0 for b in bands},
            low_end_side_energy_db=DB_FLOOR,
            balance_db=0.0,
            correlation_series=[1.0] * n_frames,
            width_series=[0.0] * n_frames,
            times=[round(float(t), 3) for t in times],
        )

    left, right = buf.left, buf.right

    # --- broadband -------------------------------------------------------
    # Mean-removed second moments, computed from sums so we never copy the
    # signal just to strip a DC offset.
    n = float(max(left.size, 1))
    sum_l, sum_r = float(left.sum()), float(right.sum())
    mean_l, mean_r = sum_l / n, sum_r / n
    ll = float(np.dot(left, left)) / n - mean_l * mean_l
    rr = float(np.dot(right, right)) / n - mean_r * mean_r
    lr = float(np.dot(left, right)) / n - mean_l * mean_r
    ll, rr = max(ll, 0.0), max(rr, 0.0)

    correlation, width, mono_sum_loss_db = _pair_stats(ll, rr, lr)
    # Directional by construction: a silent right channel pins this to the
    # floor (hard left-heavy) instead of reporting a balanced mix.
    balance_db = _ratio_db(rr, ll)

    # --- per macro band --------------------------------------------------
    band_correlation: Dict[str, float] = {}
    band_width: Dict[str, float] = {}
    band_mono_loss: Dict[str, float] = {}
    low_mid_ms = 0.0
    low_side_ms = 0.0

    rate_cache: Dict[int, Tuple[np.ndarray, np.ndarray, int]] = {1: (left, right, sr)}

    for name in bands:
        lo_hz, hi_hz = MACRO_BANDS[name]
        w_l, w_r, w_sr = _working_pair(rate_cache, _decimation(hi_hz, sr))
        l_b = _safe_bandpass(w_l, w_sr, lo_hz, hi_hz)
        r_b = _safe_bandpass(w_r, w_sr, lo_hz, hi_hz)
        b_ll, b_rr, b_lr = _moments(l_b, r_b)
        b_corr, b_width, b_loss = _pair_stats(b_ll, b_rr, b_lr)

        band_correlation[name] = round(b_corr, 3)
        band_width[name] = round(b_width, 3)
        band_mono_loss[name] = round(b_loss, 2)

        if hi_hz <= LOW_END_HZ:
            # Adjacent, effectively disjoint bands: adding in the power domain
            # reconstructs the 20-120 Hz region without two more filter passes.
            low_mid_ms += max((b_ll + 2.0 * b_lr + b_rr) * 0.25, 0.0)
            low_side_ms += max((b_ll - 2.0 * b_lr + b_rr) * 0.25, 0.0)

    if low_mid_ms <= ENERGY_FLOOR and low_side_ms <= ENERGY_FLOOR:
        low_end_side_energy_db = DB_FLOOR  # no low end at all: no side problem
    elif low_mid_ms <= ENERGY_FLOOR:
        low_end_side_energy_db = -DB_FLOOR  # side only: total LF cancellation
    else:
        low_end_side_energy_db = _ratio_db(low_side_ms, low_mid_ms)

    # --- time series -----------------------------------------------------
    times, corr_series, width_series = _series(left, right, sr)

    return StereoMeasurement(
        is_mono_source=False,
        correlation=round(_finite(correlation, 1.0), 4),
        width=round(_finite(width, 0.0), 4),
        band_correlation=band_correlation,
        band_width=band_width,
        mono_sum_loss_db=round(_finite(mono_sum_loss_db, 0.0), 2),
        band_mono_loss_db=band_mono_loss,
        low_end_side_energy_db=round(_finite(low_end_side_energy_db, DB_FLOOR), 2),
        balance_db=round(_finite(balance_db, 0.0), 2),
        correlation_series=[round(float(v), 3) for v in corr_series],
        width_series=[round(float(v), 3) for v in width_series],
        times=[round(float(t), 3) for t in times],
    )
