"""Transient measurement: attack, punch, smearing, and where the drums go flat.

Punch is not loudness. A kick that is loud and then *gone* hits harder than a
louder one that hangs around for 80 ms, because the ear reads the drop, not
the peak. Every number here is built on that one relationship — the level at
the transient versus the level shortly after it.

Per-band punch matters as much as the broadband figure: low-band punch is kick
weight and high-band punch is snare snap, and a mix can lose one while keeping
the other. Over-compression flattens the highs first; a muddy low end flattens
the lows first. One number cannot say which happened.

Measurement only. No thresholds, no genre knowledge, no verdicts.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sps
from scipy.ndimage import uniform_filter1d

from ..core import (
    EPS,
    FRAME_SEC,
    HOP_SEC,
    MACRO_BAND_ORDER,
    MACRO_BANDS,
    AudioBuffer,
    amp_to_db,
    bandpass,
    clamp,
    envelope,
    frame_signal,
    frame_times,
    merge_spans,
)
from ..types import Moment, TransientMeasurement

logger = logging.getLogger(__name__)

__all__ = ["measure_transients"]


# --- onset / envelope geometry ---------------------------------------------

ONSET_HOP = 512  # librosa frame hop; ~10.7 ms at 48 kHz

# Envelopes are analysed on a common 8 kHz grid: 0.125 ms resolution, far finer
# than any attack we care to report, and 6x cheaper than working at 48 kHz.
ENV_SR = 8_000.0

PRE_SEC = 0.030      # librosa marks the strength peak, which trails the attack
SEARCH_SEC = 0.060   # window the transient peak is located in
SUSTAIN_SEC = 0.050  # "the level 50 ms later"
TAIL_SEC = 0.010     # margin so the sustain probe never runs off the end
SUSTAIN_AVG_SEC = 0.003

# Onsets more than this far below the loudest one are noise-floor artefacts.
ONSET_GATE_DB = 30.0

# Maps transient-to-sustain dB onto 0-1 as r / (r + PUNCH_HALF_DB), so the
# scale saturates smoothly instead of clipping. A hard ceiling is unusable
# here: the air band routinely runs 40-55 dB of transient-to-sustain because
# there is simply nothing up there between hits, and any linear scale wide
# enough for it squashes the broadband figure into the bottom fifth.
# 9 dB is the half-scale point, which puts a well-articulated broadband mix
# (6-12 dB) at 0.40-0.57 and an over-limited one (< 5 dB) below 0.36.
PUNCH_HALF_DB = 9.0

# Attack at or above this reads as fully smeared for the smearing index.
SMEAR_ATTACK_MS = 30.0

MAX_ONSETS = 2_000
MAX_WEAK_MOMENTS = 8

# Seconds of onset envelope fed to the beat tracker. See `_beat_track`.
TEMPO_WINDOW_SEC = 90.0


# ---------------------------------------------------------------------------
# Sanitising helpers
# ---------------------------------------------------------------------------


def _f(value, default: float = 0.0, lo: Optional[float] = None,
       hi: Optional[float] = None) -> float:
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


def _mean(x: np.ndarray, default: float = 0.0) -> float:
    if x.size == 0:
        return default
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return default
    return float(np.mean(finite))


def _empty(tempo: float = 0.0) -> TransientMeasurement:
    """Everything zeroed but structurally complete — used when a file has no
    measurable onsets (a pad, a drone, 1 s of noise)."""
    return TransientMeasurement(
        onset_density=0.0,
        estimated_tempo=round(_f(tempo, 0.0, 0.0, 400.0), 2),
        attack_time_ms=0.0,
        punch_index=0.0,
        transient_to_sustain_db=0.0,
        band_punch={name: 0.0 for name in MACRO_BAND_ORDER},
        smearing_index=0.0,
        weak_moments=[],
    )


# ---------------------------------------------------------------------------
# Signal preparation
# ---------------------------------------------------------------------------


def _onset_strength_numpy(y: np.ndarray, sr: int) -> np.ndarray:
    """Spectral-flux onset envelope, numpy and scipy only.

    librosa's own onset functions route through numba, and numba is the part of
    this stack most likely to be missing or unable to JIT on a host we do not
    control — which is exactly what happened in production: the whole transient
    dimension silently read zero onsets on Railway while measuring correctly
    here. Half-wave-rectified spectral flux is the textbook method librosa
    implements anyway, so this is a faithful fallback rather than a lesser one.
    """
    n_fft = 2048
    frames = frame_signal(np.ascontiguousarray(y, dtype=np.float64), n_fft, ONSET_HOP)
    if frames.shape[0] < 2:
        return np.zeros(0)

    mag = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1))
    # Log-compress before differencing: onsets are a ratio change, not an
    # absolute one, so a quiet hi-hat should count as much as a loud kick.
    logmag = np.log1p(mag * 1000.0)
    flux = np.diff(logmag, axis=0)
    strength = np.maximum(flux, 0.0).sum(axis=1)

    # Subtract a local median so a rising arrangement doesn't read as onsets.
    win = 9
    if strength.size > win:
        pad = win // 2
        padded = np.pad(strength, pad, mode="edge")
        local = np.median(np.lib.stride_tricks.sliding_window_view(padded, win), axis=-1)
        strength = np.maximum(strength - local, 0.0)

    return np.concatenate([[0.0], strength])


def _onsets_numpy(strength: np.ndarray, sr: int) -> np.ndarray:
    """Peak-pick an onset envelope into onset times, in seconds."""
    if strength.size < 3:
        return np.zeros(0)

    # Adaptive threshold: mean plus a fraction of the spread, so the same code
    # works on a sparse ballad and a dense drum loop.
    thresh = strength.mean() + 0.6 * strength.std()
    if not np.isfinite(thresh) or thresh <= 0:
        return np.zeros(0)

    above = strength > thresh
    peaks = above & (strength >= np.roll(strength, 1)) & (strength > np.roll(strength, -1))
    peaks[0] = peaks[-1] = False
    idx = np.flatnonzero(peaks)
    if idx.size == 0:
        return np.zeros(0)

    # Debounce: two onsets closer than 50 ms are one event.
    min_gap = max(1, int(0.050 * sr / ONSET_HOP))
    keep = [idx[0]]
    for i in idx[1:]:
        if i - keep[-1] >= min_gap:
            keep.append(i)
    return np.asarray(keep, dtype=np.float64) * ONSET_HOP / sr


def _tempo_numpy(strength: np.ndarray, sr: int) -> float:
    """Tempo by autocorrelating the onset envelope over 60-200 BPM."""
    if strength.size < 16:
        return 0.0

    span = int(TEMPO_WINDOW_SEC * sr / ONSET_HOP)
    if strength.size > span:
        start = (strength.size - span) // 2
        strength = strength[start : start + span]

    x = strength - strength.mean()
    if not np.any(x):
        return 0.0

    ac = np.correlate(x, x, mode="full")[x.size - 1 :]
    frames_per_sec = sr / ONSET_HOP
    lo = int(frames_per_sec * 60.0 / 200.0)   # 200 BPM
    hi = int(frames_per_sec * 60.0 / 60.0)    # 60 BPM
    lo, hi = max(lo, 1), min(hi, ac.size - 1)
    if hi <= lo:
        return 0.0

    lag = lo + int(np.argmax(ac[lo:hi]))
    return float(60.0 * frames_per_sec / lag) if lag > 0 else 0.0


def _beat_track(strength: np.ndarray, sr: int) -> float:
    """Tempo, from a bounded slice of the onset-strength envelope.

    librosa's tempo estimation autocorrelates the whole envelope, which costs
    five seconds on a six-minute track and returns the same BPM as a 90 s
    window does — produced music does not change tempo underneath you, and if
    it did, one number could not describe it anyway. The window is taken from
    the middle so an ambient intro doesn't get a vote it hasn't earned.
    """
    import librosa

    span = int(TEMPO_WINDOW_SEC * sr / ONSET_HOP)
    if strength.size > span:
        start = (strength.size - span) // 2
        strength = strength[start:start + span]
    bpm, _ = librosa.beat.beat_track(onset_envelope=strength, sr=sr, hop_length=ONSET_HOP)
    return bpm


def _analysis_mono(buf: AudioBuffer) -> np.ndarray:
    """Mid channel, unless the sides are out of polarity and it cancels.

    On a polarity-flipped file the mid sum is near-silent, and onset detection
    on near-silence returns garbage. Falling back to the louder single channel
    keeps the transient report meaningful for exactly the file that most needs
    a phase warning attached to it.
    """
    left, right = buf.left, buf.right
    mid = 0.5 * (left + right)
    mid_rms = float(np.sqrt(np.mean(np.square(mid)))) if mid.size else 0.0
    l_rms = float(np.sqrt(np.mean(np.square(left)))) if left.size else 0.0
    r_rms = float(np.sqrt(np.mean(np.square(right)))) if right.size else 0.0
    best = max(l_rms, r_rms)
    if best > 0 and mid_rms < 0.3 * best:
        mid = left if l_rms >= r_rms else right
    out = np.ascontiguousarray(mid, dtype=np.float64)
    offset = float(np.mean(out)) if out.size else 0.0
    if math.isfinite(offset) and abs(offset) > 1e-7:
        out = np.ascontiguousarray(out - offset)
    return out


def _decimate_env(env: np.ndarray, sr: float, target_sr: float = ENV_SR) -> Tuple[np.ndarray, float]:
    """Block-mean an already-smoothed envelope down to the common grid."""
    factor = max(1, int(round(sr / target_sr)))
    if factor == 1:
        return env, float(sr)
    usable = (env.size // factor) * factor
    if usable < factor:
        return env, float(sr)
    return env[:usable].reshape(-1, factor).mean(axis=1), sr / factor


def _work_rate(hi_hz: float, sr: int) -> int:
    """Lowest standard rate that still resolves this band's top end."""
    for rate in (8_000, 16_000, 24_000):
        if hi_hz <= rate * 0.45 and sr % rate == 0 and rate < sr:
            return rate
    return sr


def _band_envelopes(mono: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    """Envelope of every macro band, all on the common ENV_SR grid.

    Bands that need no top end are filtered at a decimated rate — six of the
    nine macro bands live entirely below 2 kHz. The decimated signals are
    computed once per rate and shared, which is the whole trick: resampling
    per band instead of per rate costs more than the filtering does.
    """
    decimated: Dict[int, np.ndarray] = {sr: mono}

    out: Dict[str, np.ndarray] = {}
    for name in MACRO_BAND_ORDER:
        lo, hi = MACRO_BANDS[name]
        rate = _work_rate(hi, sr)
        if rate not in decimated:
            factor = sr // rate
            decimated[rate] = (
                np.ascontiguousarray(sps.resample_poly(mono, 1, factor))
                if mono.size > factor * 32
                else mono
            )
        x = decimated[rate]
        work_sr = rate if x is not mono else sr

        band = bandpass(x, work_sr, lo, min(hi, work_sr * 0.5 * 0.98))
        # A 30 Hz carrier ripples any short envelope at its own period, so the
        # smoothing tracks the band's lowest frequency instead of being fixed.
        smooth_ms = clamp(500.0 / max(lo, 1.0), 1.0, 15.0)
        env, _ = _decimate_env(envelope(band, work_sr, smooth_ms=smooth_ms), work_sr)
        out[name] = env
    return out


# ---------------------------------------------------------------------------
# The core transient measurement
# ---------------------------------------------------------------------------


def _onset_stats(
    env: np.ndarray, env_sr: float, onset_idx: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-onset (transient_to_sustain_db, attack_ms, peak_level, keep_mask).

    One gathered (n_onsets, window) matrix, then argmax/argmin reductions
    across it. No Python loop over onsets and none over samples.

    `keep_mask` is boolean over the *input* onsets, so the caller can recover
    which onset time each result belongs to. Onsets too close to either end of
    the file — no room for the 50 ms sustain probe — are dropped, which is why
    the caller cannot just assume a 1:1 mapping.
    """
    no_keep = np.zeros(onset_idx.shape, dtype=bool)
    empty = (np.zeros(0), np.zeros(0), np.zeros(0), no_keep)
    if env.size == 0 or onset_idx.size == 0:
        return empty

    pre = max(1, int(PRE_SEC * env_sr))
    search = max(1, int(SEARCH_SEC * env_sr))
    sus = max(1, int(SUSTAIN_SEC * env_sr))
    tail = max(1, int(TAIL_SEC * env_sr))
    head_len = pre + search
    win = head_len + sus + tail
    if env.size <= win:
        return empty

    starts = onset_idx - pre
    keep = (starts >= 0) & (starts + win <= env.size)
    starts = starts[keep]
    if starts.size == 0:
        return empty

    seg = env[starts[:, None] + np.arange(win)[None, :]]
    head = seg[:, :head_len]
    rows = np.arange(seg.shape[0])

    peak_i = np.argmax(head, axis=1)
    peak = head[rows, peak_i]
    good = peak > EPS
    if not np.any(good):
        return empty
    keep[np.flatnonzero(keep)[~good]] = False

    # --- level 50 ms after the transient peak ------------------------------
    avg = max(1, int(SUSTAIN_AVG_SEC * env_sr))
    offsets = np.arange(-avg, avg + 1)
    cols = np.clip(peak_i[:, None] + sus + offsets[None, :], 0, win - 1)
    sustain = seg[rows[:, None], cols].mean(axis=1)

    ratio_db = np.asarray(amp_to_db(peak), dtype=np.float64) - np.asarray(
        amp_to_db(np.maximum(sustain, EPS)), dtype=np.float64
    )

    # --- 10 % -> 90 % attack time ------------------------------------------
    # Measured on the rise out of the local valley in front of the peak, not
    # on 10 % of the absolute peak: in dense material the envelope never gets
    # near zero between hits, and an absolute threshold then reports the whole
    # search window as "attack" for every onset that overlaps its predecessor.
    col = np.arange(head_len)[None, :]
    valley_i = np.argmin(np.where(col <= peak_i[:, None], head, np.inf), axis=1)
    base = head[rows, valley_i]
    rise = np.maximum(peak - base, EPS)
    hi_thr = base + 0.9 * rise
    lo_thr = base + 0.1 * rise

    after_valley = col >= valley_i[:, None]
    # Both searches are bounded by construction: head[peak_i] == peak clears
    # hi_thr, and head[valley_i] == base sits under lo_thr.
    i90 = np.argmax((head >= hi_thr[:, None]) & after_valley, axis=1)
    mask_lo = (head <= lo_thr[:, None]) & after_valley & (col <= i90[:, None])
    i10 = head_len - 1 - np.argmax(mask_lo[:, ::-1], axis=1)

    attack_ms = np.clip((i90 - i10) / env_sr * 1000.0, 0.05, 200.0)

    return ratio_db[good], attack_ms[good], peak[good], keep


def _punch_from_ratio(ratio_db) -> np.ndarray | float:
    """Transient-to-sustain dB -> 0-1. Monotone, saturating, never exactly 1."""
    r = np.maximum(np.asarray(ratio_db, dtype=np.float64), 0.0)
    out = r / (r + PUNCH_HALF_DB)
    return float(out) if out.ndim == 0 else out


def _weak_moments(
    onset_times: np.ndarray,
    onset_punch: np.ndarray,
    duration: float,
    sr: int,
) -> List[Moment]:
    """Spans where punch drops well below what this track manages elsewhere.

    Judged against the track itself, never against a target — "your drums go
    flat at 1:42" is a statement about this mix, and stays measurement.

    The reference is the 75th percentile of per-onset punch, not the mean.
    On a track that is half punchy and half flat the mean sits between the two
    and the flat half no longer looks flat relative to it — exactly the case
    worth flagging, silently missed. A high percentile asks the more useful
    question: what does this track do when it is working, and where does it
    stop doing it?
    """
    if onset_times.size < 6 or duration <= 0:
        return []

    reference = float(np.percentile(onset_punch, 75.0))
    if reference <= 0.10:
        # Nothing here has punch anywhere. That is a whole-track observation,
        # not a moment, and the detector layer reads punch_index for it.
        return []

    win = max(1, min(int(FRAME_SEC * sr), max(int(duration * sr), 1)))
    hop = max(1, int(HOP_SEC * sr))
    n_frames = max(1, 1 + max(0, int(duration * sr) - win) // hop)
    times = frame_times(n_frames, win, hop, sr)
    if times.size < 4:
        return []

    order = np.argsort(onset_times)
    curve = np.interp(times, onset_times[order], onset_punch[order])
    curve = uniform_filter1d(curve, size=min(5, max(1, curve.size)), mode="nearest")

    threshold = 0.6 * reference
    mask = curve < threshold
    if not np.any(mask):
        return []

    moments: List[Moment] = []
    for t0, t1 in merge_spans(times, mask, min_gap=0.75, min_len=0.5):
        window = (times >= t0) & (times <= max(t1, t0))
        vals = curve[window]
        local = float(np.min(vals)) if vals.size else float(threshold)
        moments.append(
            Moment(
                t_start=round(_f(t0, 0.0, 0.0, duration), 2),
                t_end=round(_f(min(t1, duration), 0.0, 0.0, duration), 2),
                intensity=round(
                    clamp((reference - local) / max(reference, EPS), 0.0, 1.0), 3
                ),
                value=round(_f(local, 0.0, 0.0, 1.0), 3),
                label=f"punch {local:.2f} vs {reference:.2f} typical",
            )
        )

    moments.sort(key=lambda m: m.intensity, reverse=True)
    moments = moments[:MAX_WEAK_MOMENTS]
    moments.sort(key=lambda m: m.t_start)
    return moments


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def measure_transients(buf: AudioBuffer) -> TransientMeasurement:
    """Measure onset behaviour, attack, punch and smearing for `buf`."""
    sr = int(buf.sr)
    duration = _f(buf.duration, 0.0, 0.0, 1e6)
    mono = _analysis_mono(buf)
    if mono.size < 64 or duration <= 0:
        return _empty()

    y32 = np.asarray(mono, dtype=np.float32)

    # --- onsets and tempo ---------------------------------------------------
    #
    # librosa first, numpy spectral flux if it fails. The fallback is not
    # cosmetic: librosa's onset path goes through numba, numba could not JIT on
    # the production host, and the failure was being swallowed here — so the
    # whole transient dimension read "no onsets" and the detector reported it
    # as *clean*. Claiming someone's drums are fine because the measurement
    # crashed is the worst thing this file could do, so a total failure now
    # raises and lets the engine record it as unassessed.
    onset_times = np.zeros(0)
    tempo = 0.0
    librosa_failed: Optional[Exception] = None

    try:
        import librosa

        strength = librosa.onset.onset_strength(y=y32, sr=sr, hop_length=ONSET_HOP)
        onset_times = np.asarray(
            librosa.onset.onset_detect(
                onset_envelope=strength, sr=sr, hop_length=ONSET_HOP, units="time"
            ),
            dtype=np.float64,
        )
        try:
            tempo = _f(np.atleast_1d(_beat_track(strength, sr))[0], 0.0, 0.0, 400.0)
        except Exception:
            tempo = _f(_tempo_numpy(np.asarray(strength, dtype=np.float64), sr), 0.0, 0.0, 400.0)
    except Exception as exc:
        librosa_failed = exc
        logger.warning(
            "transients: librosa onset detection unavailable (%s: %s); "
            "falling back to numpy spectral flux",
            type(exc).__name__, exc,
        )
        try:
            strength = _onset_strength_numpy(mono, sr)
            onset_times = _onsets_numpy(strength, sr)
            tempo = _f(_tempo_numpy(strength, sr), 0.0, 0.0, 400.0)
        except Exception as fallback_exc:
            # Both paths gone. Raising is deliberate: the engine turns this
            # into a recorded warning and an "unassessed" dimension, which is
            # honest, where returning zeros reads as a clean bill of health.
            raise RuntimeError(
                f"onset detection failed (librosa: {librosa_failed}; numpy: {fallback_exc})"
            ) from fallback_exc

    onset_density = _f(onset_times.size / duration if duration > 0 else 0.0, 0.0, 0.0, 200.0)

    if onset_times.size == 0:
        return _empty(tempo)

    # Thin out pathological onset counts; a few hundred hits already pin the
    # statistics and the gathered segment matrix stays small.
    used = onset_times
    if used.size > MAX_ONSETS:
        used = used[np.linspace(0, used.size - 1, MAX_ONSETS).astype(int)]

    # --- broadband envelope on the common grid ------------------------------
    env, env_sr = _decimate_env(envelope(mono, sr, smooth_ms=2.0), sr)
    onset_idx = np.round(used * env_sr).astype(np.int64)

    ratio_db, attack_ms, peaks, keep = _onset_stats(env, env_sr, onset_idx)
    if ratio_db.size == 0:
        return _empty(tempo)

    # Gate out onsets riding on the noise floor before averaging anything.
    peak_db = np.asarray(amp_to_db(peaks), dtype=np.float64)
    gate = peak_db > float(np.max(peak_db)) - ONSET_GATE_DB
    if not np.any(gate):
        gate = np.ones(peak_db.shape, dtype=bool)
    ratio_db, attack_ms = ratio_db[gate], attack_ms[gate]
    measured_times = used[keep][gate]

    transient_to_sustain_db = _f(_mean(ratio_db), 0.0, -30.0, 60.0)
    punch_index = _f(_punch_from_ratio(transient_to_sustain_db), 0.0, 0.0, 1.0)
    attack_time_ms = _f(_mean(attack_ms), 0.0, 0.0, 200.0)

    # --- per-band punch: kick weight and snare snap are different stories ---
    band_punch: Dict[str, float] = {name: 0.0 for name in MACRO_BAND_ORDER}
    b_idx = np.round(used * ENV_SR).astype(np.int64)
    for name, b_env in _band_envelopes(mono, sr).items():
        b_ratio, _, b_peaks, _ = _onset_stats(b_env, ENV_SR, b_idx)
        if b_ratio.size == 0:
            continue
        b_peak_db = np.asarray(amp_to_db(b_peaks), dtype=np.float64)
        b_gate = b_peak_db > float(np.max(b_peak_db)) - ONSET_GATE_DB
        if not np.any(b_gate):
            b_gate = np.ones(b_peak_db.shape, dtype=bool)
        band_punch[name] = round(
            _f(_punch_from_ratio(_mean(b_ratio[b_gate])), 0.0, 0.0, 1.0), 3
        )

    # --- smearing: long attacks and no drop after the hit -------------------
    smearing_index = _f(
        0.5 * clamp(attack_time_ms / SMEAR_ATTACK_MS, 0.0, 1.0)
        + 0.5 * (1.0 - punch_index),
        0.0,
        0.0,
        1.0,
    )

    # --- where the drums go flat -------------------------------------------
    per_onset_punch = np.asarray(_punch_from_ratio(ratio_db), dtype=np.float64)
    weak_moments = _weak_moments(measured_times, per_onset_punch, duration, sr)

    return TransientMeasurement(
        onset_density=round(onset_density, 3),
        estimated_tempo=round(_f(tempo, 0.0, 0.0, 400.0), 2),
        attack_time_ms=round(attack_time_ms, 2),
        punch_index=round(punch_index, 3),
        transient_to_sustain_db=round(transient_to_sustain_db, 2),
        band_punch=band_punch,
        smearing_index=round(smearing_index, 3),
        weak_moments=weak_moments,
    )
