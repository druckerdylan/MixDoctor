"""Dynamics measurement: crest, macro/micro dynamics, DR, pumping, gain reduction.

Four different numbers in this module all get called "dynamics" in casual use
and they mean genuinely different things:

    macro_dynamics_lu   does the chorus lift above the verse? (3 s time scale)
    micro_dynamics_db   is there still a transient inside each hit? (50 ms)
    dr_value            TT-DR style headroom over the loud sections (3 s blocks)
    crest_factor_db     one number for the whole file (whole-file time scale)

A brickwalled master can keep a respectable macro range while its micro
dynamics are gone — that is exactly the mix that measures "fine" and sounds
dead. Reporting them separately is the point of this module.

Measurement only. No thresholds, no genre knowledge, no verdicts: whether
7 dB of crest is good depends on the genre and that decision belongs to the
detector layer.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from scipy import signal as sps
from scipy.ndimage import uniform_filter1d

from ..core import (
    EPS,
    FRAME_SEC,
    HOP_SEC,
    AudioBuffer,
    amp_to_db,
    clamp,
    envelope,
    frame_signal,
    frame_times,
)
from ..types import DynamicsMeasurement

__all__ = ["measure_dynamics"]


# --- analysis windows ------------------------------------------------------

# 50 ms is short enough to sit inside a single drum hit, so the crest measured
# here is the hit's own attack rather than the arrangement's level changes.
MICRO_WIN_SEC = 0.050
MICRO_HOP_SEC = 0.025

# 3 s is the EBU short-term window and the TT-DR block length. Both time
# scales in this module that mean "section" use it.
BLOCK_SEC = 3.0

# Envelope modulation above ~20 Hz stops being "pumping" and starts being
# audible as tone, and below 0.3 Hz it is arrangement, not a compressor.
ENV_SR = 200.0
MOD_LO_HZ = 0.30
MOD_HI_HZ = 20.0

# Everything more than this far under the loudest window is treated as
# inter-song silence / fades and excluded from averages.
ACTIVE_RANGE_DB = 30.0
SILENCE_FLOOR_DB = -80.0


# ITU-R BS.1770-4 K-weighting, pre-solved at 48 kHz (= core.ANALYSIS_SR).
# Only used as a fallback when the loudness module handed us no series.
_K_B1 = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
_K_A1 = np.array([1.0, -1.69065929318241, 0.73248077421585])
_K_B2 = np.array([1.0, -2.0, 1.0])
_K_A2 = np.array([1.0, -1.99004745483398, 0.99007225036621])


# ---------------------------------------------------------------------------
# Sanitising helpers — nothing leaves this module as NaN, inf or None
# ---------------------------------------------------------------------------


def _f(value, default: float = 0.0, lo: Optional[float] = None,
       hi: Optional[float] = None) -> float:
    """Coerce to a finite float, clamped. NaN/inf serialise to invalid JSON."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    if lo is not None:
        out = max(out, lo)
    if hi is not None:
        out = min(out, hi)
    return out


def _round_series(values, nd: int = 2, lo: float = -160.0, hi: float = 160.0) -> List[float]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return []
    arr = np.nan_to_num(arr, nan=0.0, posinf=hi, neginf=lo)
    return [float(v) for v in np.round(np.clip(arr, lo, hi), nd)]


def _mean(x: np.ndarray, default: float = 0.0) -> float:
    if x.size == 0:
        return default
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return default
    return float(np.mean(finite))


def _pct(x: np.ndarray, q: float, default: float = 0.0) -> float:
    if x.size == 0:
        return default
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return default
    return float(np.percentile(finite, q))


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------


def _dc_removed(x: np.ndarray) -> np.ndarray:
    """Strip DC so an offset track doesn't read as a huge crest factor."""
    if x.size == 0:
        return x
    offset = float(np.mean(x))
    if not math.isfinite(offset) or abs(offset) < 1e-7:
        return x
    return np.ascontiguousarray(x - offset)


def _frame_peak_rms(
    left: np.ndarray, right: np.ndarray, win: int, hop: int
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Per-frame (peak, rms) across both channels, vectorised.

    Peak comes from min/max on the strided view and rms from an einsum
    contraction — neither materialises the (n_frames, win) block, which for a
    six-minute track would be hundreds of megabytes of temporary.
    """
    win = int(max(1, min(win, max(left.shape[0], 1))))
    hop = int(max(1, hop))

    fl = frame_signal(left, win, hop)
    fr = frame_signal(right, win, hop)
    n_frames = fl.shape[0]

    peak = np.maximum(
        np.maximum(np.abs(fl.max(axis=1)), np.abs(fl.min(axis=1))),
        np.maximum(np.abs(fr.max(axis=1)), np.abs(fr.min(axis=1))),
    )
    sumsq = np.einsum("ij,ij->i", fl, fl) + np.einsum("ij,ij->i", fr, fr)
    rms = np.sqrt(np.maximum(sumsq / (2.0 * win), 0.0))
    return peak, rms, n_frames, win


def _active_mask(rms_db: np.ndarray) -> np.ndarray:
    """Frames loud enough to say anything about. Never returns all-False."""
    if rms_db.size == 0:
        return np.zeros(0, dtype=bool)
    ceiling = float(np.max(rms_db))
    mask = (rms_db > ceiling - ACTIVE_RANGE_DB) & (rms_db > SILENCE_FLOOR_DB)
    if not np.any(mask):
        mask = rms_db >= ceiling
    return mask


# ---------------------------------------------------------------------------
# Macro dynamics (3 s short-term loudness spread)
# ---------------------------------------------------------------------------


def _k_weight(x: np.ndarray, sr: int) -> np.ndarray:
    if sr != 48_000 or x.size < 8:
        return x
    return sps.lfilter(_K_B2, _K_A2, sps.lfilter(_K_B1, _K_A1, x))


def _short_term_lufs(left: np.ndarray, right: np.ndarray, sr: int) -> np.ndarray:
    """3 s / 1 s-hop short-term loudness. Fallback only — the loudness module
    owns this measurement; we recompute solely when it handed us nothing."""
    n = left.shape[0]
    if n < 8:
        return np.zeros(0)
    win = min(int(BLOCK_SEC * sr), n)
    hop = max(1, min(int(1.0 * sr), win))

    kl, kr = _k_weight(left, sr), _k_weight(right, sr)
    fl = frame_signal(np.ascontiguousarray(kl), win, hop)
    fr = frame_signal(np.ascontiguousarray(kr), win, hop)
    ms = (np.einsum("ij,ij->i", fl, fl) + np.einsum("ij,ij->i", fr, fr)) / win
    return -0.691 + 10.0 * np.log10(np.maximum(ms, EPS))


def _loudness_spread(series: np.ndarray) -> float:
    """EBU-LRA-style spread: 95th minus 10th percentile of gated blocks."""
    if series.size < 3:
        return 0.0
    vals = series[np.isfinite(series)]
    vals = vals[vals > -70.0]
    if vals.size < 3:
        return 0.0
    # Relative gate: ignore anything 20 LU under the ungated mean level.
    ref = 10.0 * np.log10(max(float(np.mean(10.0 ** (vals / 10.0))), EPS))
    gated = vals[vals > ref - 20.0]
    if gated.size < 3:
        gated = vals
    return float(np.percentile(gated, 95.0) - np.percentile(gated, 10.0))


def _macro_dynamics(
    left: np.ndarray, right: np.ndarray, sr: int, loudness
) -> float:
    """Prefer the loudness module's own short-term series; recompute only if
    it is unusable, so the two modules can never disagree by construction."""
    series = np.asarray(getattr(loudness, "short_term_series", None) or [], dtype=np.float64)
    if series.size >= 4:
        spread = _loudness_spread(series)
        if spread > 0.0:
            return spread

    # Only trust the reported LRA if it is physically plausible. An ungated
    # range computed over a file that fades to digital silence runs to
    # hundreds of LU, and no music has a real range beyond ~40.
    reported = _f(getattr(loudness, "loudness_range_lu", 0.0), 0.0)
    if 0.0 < reported < 40.0:
        return reported

    return _loudness_spread(_short_term_lufs(left, right, sr))


# ---------------------------------------------------------------------------
# TT-DR
# ---------------------------------------------------------------------------


def _dr_value(left: np.ndarray, right: np.ndarray, sr: int) -> float:
    """TT-DR style: per-channel 3 s blocks, mean peak-minus-RMS over the
    loudest 20 % of them, averaged across channels.

    RMS carries the official sqrt(2) factor so a sine reads DR 0 and the
    numbers line up with what a TT DR meter or foobar shows.
    """
    n = left.shape[0]
    if n < 8:
        return 0.0
    block = min(int(BLOCK_SEC * sr), n)

    per_channel: List[float] = []
    for chan in (left, right):
        peak, rms, n_blocks, win = _frame_peak_rms(chan, chan, block, block)
        rms_db = np.asarray(amp_to_db(np.sqrt(2.0) * rms), dtype=np.float64)
        peak_db = np.asarray(amp_to_db(peak), dtype=np.float64)

        keep = (rms_db > SILENCE_FLOOR_DB) & np.isfinite(rms_db) & np.isfinite(peak_db)
        if not np.any(keep):
            continue
        rms_db, peak_db = rms_db[keep], peak_db[keep]

        n_top = max(1, int(round(0.2 * rms_db.size)))
        top = np.argsort(rms_db)[-n_top:]
        per_channel.append(float(np.mean(peak_db[top] - rms_db[top])))

    if not per_channel:
        return 0.0
    return float(np.mean(per_channel))


# ---------------------------------------------------------------------------
# Pumping
# ---------------------------------------------------------------------------


def _estimate_tempo(mono: np.ndarray, sr: int) -> float:
    try:
        import librosa

        if mono.size < sr // 2:
            return 0.0
        tempo, _ = librosa.beat.beat_track(y=np.asarray(mono, dtype=np.float32), sr=sr)
        return _f(np.atleast_1d(tempo)[0], 0.0, 0.0, 400.0)
    except Exception:
        return 0.0


def _pumping(
    left: np.ndarray, right: np.ndarray, sr: int, tempo_bpm: float
) -> Tuple[float, float]:
    """How much of the envelope's modulation energy is locked to the beat.

    A compressor with a musical release imprints a periodic gain ripple at the
    beat rate (or a simple subdivision of it) onto everything in the mix. We
    take the broadband envelope in dB — gain modulation is multiplicative on
    the signal and therefore additive in dB — detrend it, and ask what
    fraction of its 0.3-20 Hz modulation energy sits in narrow bins at the
    beat frequency and its harmonics.

    Honest caveat: strongly percussive material modulates its own envelope at
    the beat rate without any compressor involved. This measures *periodic,
    tempo-locked envelope modulation*; the detector layer is what decides
    whether that reads as groove or as breathing.
    """
    n = left.shape[0]
    if n < int(2.0 * sr):
        return 0.0, 0.0

    rect = np.maximum(np.abs(left), np.abs(right))
    env = envelope(rect, sr, smooth_ms=15.0)

    # Drop the one-pole's startup ramp. It is a step from digital silence up
    # to the signal level, and on a very quiet file that single artefact is
    # the largest "modulation" in the record — enough to read as 100 % pumping.
    skip = min(int(0.5 * sr), max(env.size // 4, 0))
    env = env[skip:]

    factor = max(1, int(round(sr / ENV_SR)))
    usable = (env.size // factor) * factor
    if usable < factor * 32:
        return 0.0, 0.0
    env_ds = env[:usable].reshape(-1, factor).mean(axis=1)
    env_sr = sr / factor

    # Nothing above roughly -100 dBFS is a mix; refuse rather than measure the
    # shape of a noise floor.
    if float(np.max(env_ds)) <= 1e-5:
        return 0.0, 0.0

    env_db = 20.0 * np.log10(np.maximum(env_ds, EPS))
    env_db = np.maximum(env_db, float(np.max(env_db)) - 40.0)  # ignore fade tails

    # Detrend: subtract a 4 s moving average. What is left is modulation, not
    # arrangement — a chorus arriving is a trend, a compressor is a ripple.
    trend_win = int(min(max(4.0 * env_sr, 5), env_db.size))
    trend_win += 1 - (trend_win % 2)
    mod = env_db - uniform_filter1d(env_db, size=trend_win, mode="nearest")
    if mod.size < 32 or not np.all(np.isfinite(mod)):
        return 0.0, 0.0

    spec = np.fft.rfft(mod * np.hanning(mod.size))
    psd = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(mod.size, 1.0 / env_sr)

    band = (freqs >= MOD_LO_HZ) & (freqs <= MOD_HI_HZ)
    total = float(psd[band].sum())
    n_band = int(np.count_nonzero(band))
    if total <= EPS or n_band < 8:
        return 0.0, 0.0

    band_freqs, band_psd = freqs[band], psd[band]
    dominant_hz = float(band_freqs[int(np.argmax(band_psd))])

    beat_hz = tempo_bpm / 60.0 if tempo_bpm > 0 else dominant_hz
    if not math.isfinite(beat_hz) or beat_hz <= 0:
        return 0.0, dominant_hz

    bin_hz = float(freqs[1] - freqs[0]) if freqs.size > 1 else env_sr
    picked = np.zeros(band_freqs.shape, dtype=bool)
    for mult in (0.25, 0.5, 1.0, 2.0, 3.0, 4.0):
        target = beat_hz * mult
        if not (MOD_LO_HZ <= target <= MOD_HI_HZ):
            continue
        tol = max(2.0 * bin_hz, 0.04 * target)
        picked |= np.abs(band_freqs - target) <= tol

    n_picked = int(np.count_nonzero(picked))
    if n_picked == 0:
        return 0.0, dominant_hz

    frac = float(band_psd[picked].sum()) / total
    # Selecting k of N bins captures k/N of a flat spectrum by chance; only
    # the excess over that is evidence of periodicity.
    chance = n_picked / n_band
    excess = clamp((frac - chance) / max(1.0 - chance, EPS), 0.0, 1.0)

    # Depth gate: a perfectly periodic ripple of 0.3 dB is not pumping.
    depth_db = 2.0 * float(np.std(mod))
    gate = clamp((depth_db - 0.75) / 4.0, 0.0, 1.0)

    rate_hz = dominant_hz
    if n_picked:
        local = np.where(picked, band_psd, 0.0)
        if local.max() > 0:
            rate_hz = float(band_freqs[int(np.argmax(local))])

    return clamp(excess * gate, 0.0, 1.0), rate_hz


# ---------------------------------------------------------------------------
# Gain reduction estimate
# ---------------------------------------------------------------------------


def _gain_reduction(
    peak_db: np.ndarray, rms_db: np.ndarray, active: np.ndarray
) -> Tuple[float, float, float]:
    """Estimate how far the peaks have been pulled toward the RMS.

    Two independent symptoms of a limiter, in 50 ms windows:

    1. *Crest collapse.* An untouched mix has roughly the same micro crest in
       its loud sections as in its moderate ones. A limiter only works when
       the signal is loud, so crest drops specifically where level rises.
    2. *Peak pinning.* Level keeps varying while peaks stop varying — the
       peak distribution flattens against the ceiling even though the RMS
       distribution has not.

    This is an estimate, not a readout of the limiter's meter: quiet sparse
    passages genuinely have more crest than dense loud ones, so the number
    reads slightly high on arrangement-driven material even with no processing.
    Returns (estimate_db, crest_collapse_db, peak_pinning_db).
    """
    if not np.any(active):
        return 0.0, 0.0, 0.0

    crest = peak_db[active] - rms_db[active]
    level = rms_db[active]
    peaks = peak_db[active]
    if crest.size < 8:
        return 0.0, 0.0, 0.0

    loud = level >= np.percentile(level, 80.0)
    mids = (level >= np.percentile(level, 35.0)) & (level <= np.percentile(level, 70.0))
    collapse = 0.0
    if np.any(loud) and np.any(mids):
        collapse = max(0.0, _mean(crest[mids]) - _mean(crest[loud]))

    level_spread = _pct(level, 98.0) - _pct(level, 50.0)
    peak_spread = _pct(peaks, 98.0) - _pct(peaks, 50.0)
    pinning = max(0.0, level_spread - peak_spread)

    estimate = clamp(0.6 * collapse + 0.4 * pinning, 0.0, 24.0)
    return estimate, float(collapse), float(pinning)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def measure_dynamics(
    buf: AudioBuffer, loudness, tempo_bpm: Optional[float] = None
) -> DynamicsMeasurement:
    """Measure every time scale of dynamics for `buf`.

    `loudness` is the LoudnessMeasurement for the same buffer; PLR is taken
    straight from it rather than recomputed. `tempo_bpm` is an optional hint —
    pass the value the transient module already estimated to skip a second
    beat-tracking pass; omit it and we estimate our own.
    """
    sr = int(buf.sr)
    left = _dc_removed(buf.left)
    right = _dc_removed(buf.right)
    n = left.shape[0]

    # --- broadband crest ---------------------------------------------------
    global_peak = float(max(np.max(np.abs(left)) if n else 0.0,
                            np.max(np.abs(right)) if n else 0.0))
    mean_sq = float(0.5 * (np.mean(np.square(left)) + np.mean(np.square(right)))) if n else 0.0
    rms_db_val = _f(amp_to_db(math.sqrt(max(mean_sq, 0.0))), -120.0)
    peak_db_val = _f(amp_to_db(global_peak), -120.0)
    crest_factor_db = _f(peak_db_val - rms_db_val, 0.0, 0.0, 60.0)

    # --- crest over time, on the shared HOP_SEC grid ------------------------
    win = max(1, min(int(FRAME_SEC * sr), max(n, 1)))
    hop = max(1, int(HOP_SEC * sr))
    f_peak, f_rms, n_frames, win = _frame_peak_rms(left, right, win, hop)
    f_peak_db = np.asarray(amp_to_db(f_peak), dtype=np.float64)
    f_rms_db = np.asarray(amp_to_db(f_rms), dtype=np.float64)
    crest_series = np.clip(f_peak_db - f_rms_db, 0.0, 60.0)
    times = frame_times(n_frames, win, hop, sr)

    # --- micro dynamics: crest inside a single drum hit ---------------------
    m_win = max(8, min(int(MICRO_WIN_SEC * sr), max(n, 8)))
    m_hop = max(1, int(MICRO_HOP_SEC * sr))
    m_peak, m_rms, _, m_win = _frame_peak_rms(left, right, m_win, m_hop)
    m_peak_db = np.asarray(amp_to_db(m_peak), dtype=np.float64)
    m_rms_db = np.asarray(amp_to_db(m_rms), dtype=np.float64)
    m_active = _active_mask(m_rms_db)
    micro_dynamics_db = _f(
        _mean((m_peak_db - m_rms_db)[m_active]), 0.0, 0.0, 60.0
    )

    # --- macro dynamics: does the chorus lift above the verse ---------------
    macro_dynamics_lu = _f(_macro_dynamics(left, right, sr, loudness), 0.0, 0.0, 60.0)

    # --- TT-DR --------------------------------------------------------------
    dr_value = _f(_dr_value(left, right, sr), 0.0, 0.0, 60.0)

    # --- pumping ------------------------------------------------------------
    tempo = _f(tempo_bpm, 0.0, 0.0, 400.0)
    if tempo <= 0:
        tempo = _estimate_tempo(0.5 * (left + right), sr)
    pumping_index, pumping_rate_hz = _pumping(left, right, sr, tempo)

    # --- compression estimate ----------------------------------------------
    gr_estimate, _, _ = _gain_reduction(m_peak_db, m_rms_db, m_active)

    # --- PLR comes from the loudness module, never recomputed ---------------
    plr = _f(getattr(loudness, "plr_db", None), float("nan"))
    if not math.isfinite(plr) or plr == 0.0:
        tp = _f(getattr(loudness, "true_peak_dbtp", None), float("nan"))
        li = _f(getattr(loudness, "integrated_lufs", None), float("nan"))
        plr = tp - li if math.isfinite(tp) and math.isfinite(li) else crest_factor_db
    peak_to_loudness_db = _f(plr, crest_factor_db, -20.0, 60.0)

    return DynamicsMeasurement(
        crest_factor_db=round(crest_factor_db, 2),
        peak_to_loudness_db=round(peak_to_loudness_db, 2),
        macro_dynamics_lu=round(macro_dynamics_lu, 2),
        micro_dynamics_db=round(micro_dynamics_db, 2),
        dr_value=round(dr_value, 2),
        rms_db=round(rms_db_val, 2),
        pumping_index=round(_f(pumping_index, 0.0, 0.0, 1.0), 3),
        pumping_rate_hz=round(_f(pumping_rate_hz, 0.0, 0.0, 100.0), 3),
        gain_reduction_estimate_db=round(_f(gr_estimate, 0.0, 0.0, 24.0), 2),
        crest_series=_round_series(crest_series, nd=2, lo=0.0, hi=60.0),
        times=_round_series(times, nd=3, lo=0.0, hi=1e6),
    )
