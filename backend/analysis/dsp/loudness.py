"""ITU-R BS.1770-4 / EBU R128 loudness, true peak and peak-to-loudness ratios.

The whole module is built on one observation: every quantity R128 asks for is a
*mean square of the K-weighted signal over some window*. So we filter once,
square once, and then every window — 400 ms gating blocks, 400 ms momentary,
3 s short-term — is a cheap reduction over that one array. Nothing here loops
over samples in Python, and pyloudnorm's per-window gating routine is never
called; the gate is applied once, to the whole block vector, as the spec
actually describes it.

Measurement only. No genre knowledge, no thresholds, no verdicts.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import List, Sequence, Tuple

import numpy as np
from scipy import signal as sps

from ..core import (
    EPS,
    HOP_SEC,
    AudioBuffer,
    amp_to_db,
    frame_signal,
    frame_times,
)
from ..types import LoudnessMeasurement

__all__ = ["measure_loudness", "k_weight", "true_peak_dbtp", "quantum_true_peaks"]


# --- BS.1770 constants -----------------------------------------------------

# The -0.691 offset calibrates the K-weighted mean square so a 1 kHz sine at
# -20 dBFS reads -20 LUFS.
_LUFS_OFFSET = -0.691

BLOCK_SEC = 0.400          # gating block length (BS.1770-4 §5)
BLOCK_OVERLAP = 0.75       # 75 % overlap -> 100 ms step
SHORT_TERM_SEC = 3.0       # EBU R128 short-term window
ABS_GATE_LUFS = -70.0      # absolute gate
REL_GATE_LU = -10.0        # relative gate, below the ungated loudness
LRA_REL_GATE_LU = -20.0    # EBU 3342 relative gate for loudness range
LRA_LOW_PCT = 10.0
LRA_HIGH_PCT = 95.0
OVERSAMPLE = 4             # BS.1770-4 Annex 2 true-peak oversampling factor

# Series floor. Meters bottom out around here; keeping -inf out of the JSON is
# not optional, and a floor also stops a silent intro from dominating a plot.
SERIES_FLOOR_LUFS = -70.0
DB_FLOOR = -120.0  # matches core.amp_to_db's floor so peak fields agree everywhere

# Analog prototype parameters for the two K-weighting stages. Expressing them
# this way rather than hard-coding the 48 kHz biquads means the filter is
# re-derived correctly for any sample rate, and at 48 kHz it reproduces the
# tabulated BS.1770-4 coefficients to 1e-12 (asserted in the module test).
#
# The bilinear/K formulation below is Brecht De Man's: the textbook RBJ shelf
# fed the same parameters lands ~0.5 dB low through the 1-2 kHz transition,
# which is exactly where a vocal-heavy mix has most of its energy.
_SHELF = dict(gain_db=3.999843853973347, q=0.7071752369554196, fc=1681.9744509555319)
_HPF = dict(gain_db=0.0, q=0.5003270373238773, fc=38.13547087602444)

# Channel weights G_i for L/R (BS.1770-4 Table 3). Surround channels would be
# +1.5 dB; core.py has already folded anything beyond a pair into L/R.
_G_LEFT = 1.0
_G_RIGHT = 1.0


# ---------------------------------------------------------------------------
# K-weighting
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _k_weight_sos(sr: int) -> np.ndarray:
    """Two-stage K-weighting as a (2, 6) SOS matrix for `sr`.

    Stage 1 is the ~+4 dB high shelf above 1.5 kHz (head/torso response),
    stage 2 the ~38 Hz high-pass (RLB curve).
    """
    sr = int(max(sr, 1))
    sos = np.zeros((2, 6), dtype=np.float64)

    # --- stage 1: high shelf, ~+4 dB above 1.5 kHz ---
    k = math.tan(math.pi * _SHELF["fc"] / sr)
    vh = 10.0 ** (_SHELF["gain_db"] / 20.0)
    vb = vh ** 0.499666774155
    q = _SHELF["q"]
    denom = 1.0 + k / q + k * k
    sos[0] = [
        (vh + vb * k / q + k * k) / denom,
        2.0 * (k * k - vh) / denom,
        (vh - vb * k / q + k * k) / denom,
        1.0,
        2.0 * (k * k - 1.0) / denom,
        (1.0 - k / q + k * k) / denom,
    ]

    # --- stage 2: high pass at ~38 Hz (the RLB curve) ---
    k = math.tan(math.pi * _HPF["fc"] / sr)
    q = _HPF["q"]
    denom = 1.0 + k / q + k * k
    sos[1] = [
        1.0,
        -2.0,
        1.0,
        1.0,
        2.0 * (k * k - 1.0) / denom,
        (1.0 - k / q + k * k) / denom,
    ]

    return sos


def k_weight(x: np.ndarray, sr: int) -> np.ndarray:
    """Apply BS.1770 K-weighting. Causal (sosfilt), as the standard specifies."""
    if x.size == 0:
        return x.astype(np.float64, copy=False)
    return sps.sosfilt(_k_weight_sos(sr), np.asarray(x, dtype=np.float64))


# ---------------------------------------------------------------------------
# True peak
# ---------------------------------------------------------------------------


def quantum_true_peaks(
    channels: Sequence[np.ndarray],
    sr: int,
    quantum: int,
    factor: int = OVERSAMPLE,
    chunk_quanta: int = 120,
) -> Tuple[np.ndarray, float]:
    """Oversampled peak envelope at `quantum`-sample resolution, plus the global peak.

    BS.1770-4 Annex 2 asks for a >=4x oversampled peak. Doing that on a whole
    10-minute track at once would allocate ~1 GB, so it runs in chunks that are
    padded on both sides and trimmed after resampling — the interior of every
    chunk is bit-identical to a single-shot resample.

    Returns (envelope[n_samples // quantum], global_peak_linear). The envelope
    is what PSR needs; the scalar is the reported dBTP.
    """
    # float32 through the polyphase filter: 24 bits of mantissa puts the error
    # near -144 dBFS, which cannot move a peak reading, and it runs ~1.7x
    # faster than float64 on the one operation that dominates this module.
    chans = [np.ascontiguousarray(c, dtype=np.float32) for c in channels if c is not None]
    chans = [c for c in chans if c.size]
    if not chans:
        return np.zeros(0, dtype=np.float64), 0.0

    n = min(c.shape[0] for c in chans)
    quantum = int(max(quantum, 1))
    n_q = n // quantum
    env = np.zeros(max(n_q, 0), dtype=np.float64)
    peak = 0.0

    # Padding covers the polyphase filter's transient at each chunk edge.
    pad = 256
    chunk_quanta = int(max(chunk_quanta, 1))

    for q0 in range(0, n_q, chunk_quanta):
        q1 = min(n_q, q0 + chunk_quanta)
        s, e = q0 * quantum, q1 * quantum
        a, b = max(0, s - pad), min(n, e + pad)
        block = None
        for c in chans:
            up = sps.resample_poly(c[a:b], factor, 1)
            lo = (s - a) * factor
            hi = lo + (e - s) * factor
            seg = np.abs(up[lo:hi])
            if seg.size != (e - s) * factor:  # defensive: ragged tail
                seg = np.resize(seg, (e - s) * factor)
            m = seg.reshape(q1 - q0, quantum * factor).max(axis=1)
            block = m if block is None else np.maximum(block, m)
        env[q0:q1] = block
        peak = max(peak, float(block.max()))

    # Tail shorter than one quantum still counts towards the global peak.
    tail_start = n_q * quantum
    if tail_start < n:
        a = max(0, tail_start - pad)
        for c in chans:
            up = sps.resample_poly(c[a:n], factor, 1)
            lo = (tail_start - a) * factor
            if up.size > lo:
                peak = max(peak, float(np.abs(up[lo:]).max()))

    # An oversampled peak can never legitimately sit below the sample peak.
    peak = max(peak, max(float(np.abs(c).max()) for c in chans))
    return env, peak


def true_peak_dbtp(channels: Sequence[np.ndarray], sr: int, factor: int = OVERSAMPLE) -> float:
    """Scalar true peak in dBTP. Convenience wrapper over `quantum_true_peaks`."""
    _, peak = quantum_true_peaks(channels, sr, quantum=int(max(sr // 4, 1)), factor=factor)
    return _finite(amp_to_db(peak), DB_FLOOR, lo=DB_FLOOR)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _finite(value, default: float = 0.0, lo: float | None = None, hi: float | None = None) -> float:
    """Coerce anything to a finite float. NaN/inf serialise to invalid JSON."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = default
    if not math.isfinite(out):
        out = default
    if lo is not None:
        out = max(out, lo)
    if hi is not None:
        out = min(out, hi)
    return out


def _lufs(mean_square: np.ndarray | float) -> np.ndarray | float:
    """Mean square of the K-weighted, channel-summed signal -> LUFS."""
    ms = np.asarray(mean_square, dtype=np.float64)
    out = _LUFS_OFFSET + 10.0 * np.log10(np.maximum(ms, EPS))
    return float(out) if out.ndim == 0 else out


def _series(values: np.ndarray, floor: float = SERIES_FLOOR_LUFS) -> List[float]:
    """Finite, floored, 2-dp list — small JSON and no NaN in the payload."""
    arr = np.asarray(values, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=floor, posinf=floor, neginf=floor)
    return np.round(np.maximum(arr, floor), 2).tolist()


def _gated_integrated(block_ms: np.ndarray) -> float:
    """BS.1770-4 two-stage gate applied to the 400 ms block mean squares.

    Absolute gate at -70 LUFS, then a relative gate 10 LU below the loudness of
    whatever survived it. Skipping the relative stage is what makes a naive
    implementation read a couple of LU low on anything with quiet passages.
    """
    if block_ms.size == 0:
        return ABS_GATE_LUFS

    loud = np.asarray(_lufs(block_ms), dtype=np.float64)

    above_abs = loud > ABS_GATE_LUFS
    if not np.any(above_abs):
        return ABS_GATE_LUFS

    relative_gate = float(_lufs(block_ms[above_abs].mean())) + REL_GATE_LU
    keep = above_abs & (loud > relative_gate)
    if not np.any(keep):
        keep = above_abs

    return float(_lufs(block_ms[keep].mean()))


def _loudness_range(short_term: np.ndarray) -> float:
    """EBU Tech 3342 loudness range, in LU."""
    st = short_term[np.isfinite(short_term)]
    st = st[st > ABS_GATE_LUFS]
    if st.size < 2:
        return 0.0

    # Relative threshold from the power mean of the absolute-gated values.
    mean_power = np.mean(10.0 ** ((st - _LUFS_OFFSET) / 10.0))
    threshold = _LUFS_OFFSET + 10.0 * math.log10(max(float(mean_power), EPS)) + LRA_REL_GATE_LU

    gated = st[st > threshold]
    if gated.size < 2:
        gated = st

    lo, hi = np.percentile(gated, [LRA_LOW_PCT, LRA_HIGH_PCT])
    return _finite(hi - lo, 0.0, lo=0.0)


def _trailing_max(env: np.ndarray, width: int) -> np.ndarray:
    """Sliding maximum over the previous `width` entries, same length as `env`.

    Vectorised with a strided view; the leading edge is padded with env[0] so
    no value is invented that the signal did not actually contain.
    """
    if env.size == 0:
        return env
    width = int(max(width, 1))
    if width == 1:
        return env.copy()
    padded = np.concatenate([np.full(width - 1, env[0], dtype=np.float64), env])
    return frame_signal(np.ascontiguousarray(padded), width, 1).max(axis=1)[: env.size]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def measure_loudness(buf: AudioBuffer) -> LoudnessMeasurement:
    """Full R128 measurement set for one buffer."""
    sr = int(buf.sr)
    n = int(buf.n_samples)

    # One filter pass, one square, one channel sum. Everything below reduces
    # over `power`; L and R both weight 1.0 so the sum *is* sum(G_i * y_i^2).
    y_left = k_weight(buf.left, sr)
    y_right = k_weight(buf.right, sr)
    power = np.ascontiguousarray(_G_LEFT * np.square(y_left) + _G_RIGHT * np.square(y_right))
    del y_left, y_right

    # --- integrated, from 400 ms blocks with 75 % overlap ---
    block_len = int(round(BLOCK_SEC * sr))
    block_hop = max(1, int(round(BLOCK_SEC * (1.0 - BLOCK_OVERLAP) * sr)))
    if n >= block_len:
        block_ms = frame_signal(power, block_len, block_hop).mean(axis=1)
    else:
        # Shorter than one gating block: the whole file is the only block.
        block_ms = np.array([float(power.mean())], dtype=np.float64)
    integrated = _finite(_gated_integrated(block_ms), ABS_GATE_LUFS, lo=DB_FLOOR)

    # --- time series on the shared HOP_SEC grid ---
    hop = max(1, int(round(HOP_SEC * sr)))
    mom_len = min(block_len, n)
    mom_frames = frame_signal(power, mom_len, hop)
    n_frames = int(mom_frames.shape[0])
    momentary = np.asarray(_lufs(mom_frames.mean(axis=1)), dtype=np.float64)
    times = frame_times(n_frames, mom_len, hop, sr)

    # Short-term uses a 3 s window ending where the momentary window ends, so
    # both series land on the same grid. A cumulative sum keeps this O(n)
    # instead of materialising a 12x-overlapped 3 s frame matrix.
    cumulative = np.concatenate(([0.0], np.cumsum(power)))
    ends = np.minimum(np.arange(n_frames) * hop + mom_len, n)
    st_len = min(int(round(SHORT_TERM_SEC * sr)), n)
    starts = np.maximum(ends - st_len, 0)
    spans = np.maximum(ends - starts, 1)
    short_term = np.asarray(
        _lufs((cumulative[ends] - cumulative[starts]) / spans), dtype=np.float64
    )
    full_window = (ends - starts) >= st_len

    # LRA wants genuine 3 s windows; partial ones at the head would compress it.
    lra_source = short_term[full_window] if int(full_window.sum()) >= 2 else short_term
    loudness_range = _loudness_range(lra_source)

    # --- true peak (BS.1770-4 Annex 2) + per-quantum envelope for PSR ---
    channels = (buf.left,) if buf.is_mono else (buf.left, buf.right)
    peak_env, peak_linear = quantum_true_peaks(channels, sr, quantum=hop)

    # `peak_sample` comes from core's original decode, before resampling. Rate
    # conversion can shave the sample peak, so the reported true peak is held at
    # or above it rather than appearing to undercut it.
    sample_peak_linear = max(float(buf.peak_sample), float(np.abs(buf.left).max()),
                             float(np.abs(buf.right).max()))
    sample_peak = _finite(amp_to_db(sample_peak_linear), DB_FLOOR, lo=DB_FLOOR)
    true_peak = max(_finite(amp_to_db(peak_linear), DB_FLOOR, lo=DB_FLOOR), sample_peak)

    # --- PSR: short-term peak minus short-term loudness, per 3 s window ---
    st_quanta = max(1, int(round(SHORT_TERM_SEC / HOP_SEC)))
    if peak_env.size:
        window_peak = _trailing_max(peak_env, st_quanta)
        # Frame k's momentary window ends inside quantum k+1 (hop == quantum,
        # mom_len == 1.6 quanta), so that is the quantum its 3 s window reaches.
        idx = np.minimum(np.arange(n_frames) + 1, window_peak.size - 1)
        st_peak_db = np.asarray(amp_to_db(window_peak[idx]), dtype=np.float64)
    else:
        st_peak_db = np.full(n_frames, sample_peak, dtype=np.float64)

    psr = st_peak_db - short_term
    usable = (
        np.isfinite(psr)
        & (short_term > ABS_GATE_LUFS)
        & (short_term > integrated + LRA_REL_GATE_LU)
        & (st_peak_db > DB_FLOOR + 1.0)
    )
    if not np.any(usable):
        usable = np.isfinite(psr) & (short_term > ABS_GATE_LUFS)
    if np.any(usable):
        psr_values = psr[usable]
        # K-weighting can lift a very bright signal above its own true peak, so
        # a slightly negative PSR is physically possible; don't clamp it to 0.
        psr_p10 = _finite(np.percentile(psr_values, 10.0), 0.0, lo=-20.0, hi=60.0)
        psr_median = _finite(np.median(psr_values), 0.0, lo=-20.0, hi=60.0)
    else:
        psr_p10 = psr_median = 0.0

    # Floored to match the series the UI plots them against, so a silent file
    # doesn't report a maximum that sits below its own curve.
    momentary_max = _finite(momentary.max() if momentary.size else SERIES_FLOOR_LUFS,
                            SERIES_FLOOR_LUFS, lo=SERIES_FLOOR_LUFS)
    short_term_max = _finite(short_term.max() if short_term.size else SERIES_FLOOR_LUFS,
                             SERIES_FLOOR_LUFS, lo=SERIES_FLOOR_LUFS)

    # Nothing cleared the absolute gate: there is no programme to take a ratio
    # against, and a ratio built from two floors would be pure noise.
    gated_silent = integrated <= ABS_GATE_LUFS
    plr = 0.0 if gated_silent else _finite(true_peak - integrated, 0.0, lo=-20.0, hi=80.0)
    if gated_silent:
        psr_p10 = psr_median = 0.0

    return LoudnessMeasurement(
        integrated_lufs=round(integrated, 2),
        momentary_max_lufs=round(momentary_max, 2),
        short_term_max_lufs=round(short_term_max, 2),
        loudness_range_lu=round(loudness_range, 2),
        true_peak_dbtp=round(true_peak, 2),
        sample_peak_dbfs=round(sample_peak, 2),
        plr_db=round(plr, 2),
        psr_p10_db=round(psr_p10, 2),
        psr_median_db=round(psr_median, 2),
        short_term_series=_series(short_term),
        momentary_series=_series(momentary),
        times=np.round(times, 3).tolist(),
    )
