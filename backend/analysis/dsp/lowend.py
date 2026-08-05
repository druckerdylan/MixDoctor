"""Kick / bass (808) relationship.

The single most asked-about problem in modern production is "why is my low end
mush". It is almost always one of four things, and this module measures all
four rather than guessing between them:

1. The kick and the bass share a fundamental, so every kick hit re-triggers the
   same resonance instead of punching through it (`kick_bass_collision_db`).
2. The bass never gets out of the kick's way — no sidechain, no carving
   (`ducking_depth_db`, `has_sidechain`).
3. The kick has no transient left above the sustained low end
   (`kick_definition_db`).
4. There is inaudible energy below 25 Hz eating the limiter's headroom
   (`sub_rumble_db`), or the low end is not mono (`low_end_mono_ratio`).

Method, briefly. Kick hits come from spectral flux inside `core.KICK_HZ`,
refined against the band envelope so the timestamps are usable. The kick's
fundamental is measured from windows that *start* at those hits; the bass's
fundamental from windows sitting **between** them, so the kick cannot
contaminate the estimate. Collision is then the spectral overlap between the
kick's own contribution (at-onset spectrum minus between-onset spectrum) and
the bass spectrum — 0 dB means the two are the same shape in the same place.

Measurement only: no thresholds, no genre knowledge, no verdicts. Everything is
vectorised; the only Python loops iterate over *blocks of frames* or *chunks of
onsets*, never over samples.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy import signal as sps

from ..core import (
    EPS,
    HOP_SEC,
    KICK_HZ,
    SUB_HZ,
    AudioBuffer,
    bandpass,
    clamp,
    envelope,
    frame_signal,
    merge_spans,
)
from ..types import LowEndMeasurement, Moment

# --- analysis constants ----------------------------------------------------

_ONSET_NFFT = 2048          # 42.7 ms at 48 kHz: 23.4 Hz bins, ~3 usable in KICK_HZ
_ONSET_HOP = 256            # 5.3 ms: fine enough to place a kick transient
_MIN_KICK_GAP_S = 0.11      # ~545 hits/min ceiling; stops double-triggering
_REFINE_S = 0.060           # onset snap search radius against the band envelope

_SPEC_WIN = 8192            # 170 ms measurement window (9 cycles of a 55 Hz tone)
_SPEC_NFFT = 16384          # zero-padded to 2.93 Hz for sub-bin peak location
_MAX_SPEC_WINDOWS = 200     # bound the cost on long tracks; onsets are subsampled

_LOW_LO, _LOW_HI = 25.0, 250.0      # region the collision maths lives in
_KICK_SEARCH = (40.0, 110.0)        # == KICK_HZ, named for readability
_BASS_SEARCH = (30.0, 200.0)
_RUMBLE_HI = 25.0
_MONO_HI = 120.0

_DUCK_PRE = (-0.140, -0.020)   # "just before the kick": the recovered sub level
_DUCK_POST = (0.045, 0.250)    # "after the kick": past its transient, into the duck
_DEF_WIN = (-0.010, 0.070)     # kick transient search window
_MIN_DEFINITION_DB = 3.0       # a hit has to punch this far above the low-end floor
_MIN_KICK_BAND_DB = -35.0      # KICK_HZ must carry this much of the broadband level
_MIN_REGULARITY = 0.40         # share of gaps that land on the dominant spacing


# ---------------------------------------------------------------------------
# small private helpers (duplicated rather than shared: sibling dsp modules are
# being written concurrently and this package has no agreed-on internal utils)
# ---------------------------------------------------------------------------


def _finite(x: float, default: float = 0.0) -> float:
    """NaN/inf -> a sensible finite value. These serialise to invalid JSON."""
    v = float(x)
    return v if np.isfinite(v) else float(default)


def _power_spectrogram(
    x: np.ndarray, sr: int, n_fft: int, hop: int, block: int = 256
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blocked power spectrogram: (freqs, times, power[n_frames, n_bins]).

    Blocked so a 10-minute track never materialises a multi-GB complex array.
    The loop runs over blocks of frames, not samples.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    frames = frame_signal(x, n_fft, hop)
    n = int(frames.shape[0])
    window = np.hanning(n_fft)
    scale = 1.0 / (window.sum() + EPS)
    n_bins = n_fft // 2 + 1

    out = np.empty((n, n_bins), dtype=np.float32)
    for i in range(0, n, block):
        seg = np.asarray(frames[i : i + block]) * window
        spec = np.fft.rfft(seg, n=n_fft, axis=1)
        out[i : i + block] = ((np.abs(spec) * scale) ** 2).astype(np.float32)

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    times = (np.arange(n) * hop + n_fft * 0.5) / sr
    return freqs, times, out


def _welch(x: np.ndarray, sr: int, nperseg: int = 16384) -> Tuple[np.ndarray, np.ndarray]:
    """Blackman-Harris Welch spectrum.

    A -92 dB sidelobe window matters here: measuring "energy below 25 Hz" with a
    Hann window next to a 0 dB kick at 55 Hz mostly measures the window.
    `detrend="constant"` drops DC per segment, so a DC-offset file does not read
    as rumble.
    """
    n = int(x.shape[0])
    nps = int(min(nperseg, n))
    if nps < 256:
        nps = max(64, n)
    freqs, psd = sps.welch(
        np.ascontiguousarray(x, dtype=np.float64),
        fs=sr,
        window="blackmanharris",
        nperseg=nps,
        noverlap=nps // 2,
        detrend="constant",
        scaling="spectrum",
    )
    return freqs, np.asarray(psd, dtype=np.float64)


def _band_sum(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    m = (freqs >= lo) & (freqs < hi)
    return float(psd[m].sum()) if np.any(m) else 0.0


def _ratio_db(num: float, den: float, floor_db: float = -90.0) -> float:
    if den <= EPS:
        return floor_db
    return _finite(clamp(10.0 * np.log10(max(num, EPS) / den), floor_db, 60.0), floor_db)


# ---------------------------------------------------------------------------
# kick onset detection
# ---------------------------------------------------------------------------


def _detect_onsets(kick_band: np.ndarray, sr: int) -> np.ndarray:
    """Sample indices of kick hits, from spectral flux inside KICK_HZ."""
    n = int(kick_band.shape[0])
    if n < _ONSET_NFFT * 2:
        return np.zeros(0, dtype=np.int64)

    freqs, _, power = _power_spectrogram(kick_band, sr, _ONSET_NFFT, _ONSET_HOP)
    if power.shape[0] < 4:
        return np.zeros(0, dtype=np.int64)

    # Bins whose bandwidth overlaps KICK_HZ at all — at 23.4 Hz resolution that
    # is only three or four, which is exactly why the signal is pre-filtered.
    half = 0.5 * sr / _ONSET_NFFT
    sel = (freqs + half >= KICK_HZ[0]) & (freqs - half <= KICK_HZ[1])
    if not np.any(sel):
        return np.zeros(0, dtype=np.int64)

    mag = np.sqrt(power[:, sel].astype(np.float64))
    flux = np.concatenate([[0.0], np.maximum(np.diff(mag, axis=0), 0.0).sum(axis=1)])
    if not np.any(flux > 0):
        return np.zeros(0, dtype=np.int64)

    # Adaptive baseline: a moving mean over ~0.4 s, so a quiet intro and a loud
    # drop are peak-picked with the same sensitivity.
    win = max(3, int(round(0.40 * sr / _ONSET_HOP)))
    from scipy.ndimage import uniform_filter1d

    detect = np.maximum(flux - uniform_filter1d(flux, size=win, mode="nearest"), 0.0)
    positive = detect[detect > 0]
    if positive.size < 2:
        return np.zeros(0, dtype=np.int64)

    prom = 0.30 * float(np.percentile(positive, 90))
    if prom <= EPS:
        return np.zeros(0, dtype=np.int64)

    peaks, _ = sps.find_peaks(
        detect,
        distance=max(1, int(round(_MIN_KICK_GAP_S * sr / _ONSET_HOP))),
        prominence=prom,
        height=prom * 0.5,
    )
    if peaks.size == 0:
        return np.zeros(0, dtype=np.int64)

    # Frame i covers [i*hop, i*hop+n_fft); flux peaks as the transient reaches
    # the strong part of the window, i.e. roughly the frame centre.
    approx = peaks.astype(np.int64) * _ONSET_HOP + _ONSET_NFFT // 2
    return _refine_onsets(approx, kick_band, sr)


def _refine_onsets(approx: np.ndarray, kick_band: np.ndarray, sr: int) -> np.ndarray:
    """Snap flux-frame estimates onto the steepest rise of the band envelope."""
    n = int(kick_band.shape[0])
    env = envelope(kick_band, sr, smooth_ms=8.0)
    slope = np.diff(env, prepend=env[:1])

    radius = int(_REFINE_S * sr)
    offsets = np.arange(-radius, radius + 1, dtype=np.int64)

    refined = np.empty(approx.shape[0], dtype=np.int64)
    for i in range(0, approx.shape[0], 128):  # chunks of onsets, not samples
        chunk = approx[i : i + 128]
        idx = np.clip(chunk[:, None] + offsets[None, :], 0, n - 1)
        refined[i : i + 128] = idx[np.arange(chunk.size), np.argmax(slope[idx], axis=1)]

    refined = np.unique(refined)
    if refined.size < 2:
        return refined
    # Two flux peaks can refine onto the same transient; keep the first.
    keep = np.concatenate([[True], np.diff(refined) >= int(_MIN_KICK_GAP_S * sr)])
    return refined[keep]


# ---------------------------------------------------------------------------
# spectra at / between the hits
# ---------------------------------------------------------------------------


def _subsample(starts: np.ndarray, limit: int) -> np.ndarray:
    if starts.size <= limit:
        return starts
    idx = np.linspace(0, starts.size - 1, limit).round().astype(np.int64)
    return starts[np.unique(idx)]


def _mean_spectrum(
    x: np.ndarray, starts: np.ndarray, win: int, n_fft: int, block: int = 64
) -> Optional[np.ndarray]:
    """Mean power spectrum of `win`-long windows beginning at each start."""
    if starts.size == 0 or win < 64:
        return None
    window = np.hanning(win)
    scale = 1.0 / (window.sum() + EPS)
    offsets = np.arange(win, dtype=np.int64)

    acc = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    for i in range(0, starts.size, block):  # chunks of windows, not samples
        seg = x[starts[i : i + block][:, None] + offsets[None, :]] * window
        acc += ((np.abs(np.fft.rfft(seg, n=n_fft, axis=1)) * scale) ** 2).sum(axis=0)
    return acc / float(starts.size)


def _peak_freq(spec: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> Tuple[float, float]:
    """Parabolically interpolated spectral peak in [lo, hi]. Returns (hz, power)."""
    m = (freqs >= lo) & (freqs <= hi)
    if not np.any(m):
        return 0.0, 0.0
    idx = np.flatnonzero(m)
    k = int(idx[int(np.argmax(spec[idx]))])
    df = float(freqs[1] - freqs[0])
    delta = 0.0
    if 0 < k < spec.size - 1:
        a, b, c = float(spec[k - 1]), float(spec[k]), float(spec[k + 1])
        denom = a - 2.0 * b + c
        if abs(denom) > EPS:
            delta = clamp(0.5 * (a - c) / denom, -0.5, 0.5)
    return _finite(freqs[k] + delta * df), float(spec[k])


def _prefer_subharmonic(
    spec: np.ndarray, freqs: np.ndarray, f0: float, p0: float, lo: float
) -> float:
    """If half the detected peak also carries real energy, that is the fundamental.

    A bass note's second harmonic is often louder than its fundamental; without
    this check a 55 Hz bass reads as a 110 Hz bass.
    """
    half = f0 * 0.5
    if f0 <= 0.0 or half < lo:
        return f0
    m = (freqs >= half * 0.94) & (freqs <= half * 1.06)
    if not np.any(m):
        return f0
    if float(spec[m].max()) >= 0.30 * p0:
        return _peak_freq(spec, freqs, half * 0.94, half * 1.06)[0] or f0
    return f0


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def measure_low_end(buf: AudioBuffer) -> LowEndMeasurement:
    """Measure the kick / bass relationship. Never raises on valid audio."""
    sr = buf.sr
    mid = np.ascontiguousarray(buf.mid, dtype=np.float64)
    side = np.ascontiguousarray(buf.side, dtype=np.float64)
    n = int(mid.shape[0])

    # -- static band energies, Blackman-Harris so the <25 Hz figure is real ---
    freqs_w, psd_m = _welch(mid, sr)
    _, psd_s = _welch(side, sr)

    total = _band_sum(freqs_w, psd_m, 20.0, min(20000.0, sr * 0.5))
    sub_energy_db = _ratio_db(_band_sum(freqs_w, psd_m, *SUB_HZ), total)
    sub_rumble_db = _ratio_db(_band_sum(freqs_w, psd_m, 3.0, _RUMBLE_HI), total)

    low_mid_e = _band_sum(freqs_w, psd_m, 20.0, _MONO_HI)
    low_side_e = _band_sum(freqs_w, psd_s, 20.0, _MONO_HI)
    denom = low_mid_e + low_side_e
    low_end_mono_ratio = 1.0 if denom <= EPS else clamp(low_mid_e / denom, 0.0, 1.0)
    if buf.is_mono:
        low_end_mono_ratio = 1.0

    # -- kick hits -----------------------------------------------------------
    kick_band = np.ascontiguousarray(bandpass(mid, sr, KICK_HZ[0], KICK_HZ[1]))
    onsets = _detect_onsets(kick_band, sr)
    kick_count = int(onsets.size)

    kick_env = envelope(kick_band, sr, smooth_ms=10.0)
    floor = float(np.percentile(kick_env, 30.0))

    kick_definition_db = 0.0
    weakest_hit_db = 0.0
    if kick_count:
        lo_off, hi_off = int(_DEF_WIN[0] * sr), int(_DEF_WIN[1] * sr)
        offs = np.arange(lo_off, hi_off, dtype=np.int64)
        idx = np.clip(onsets[:, None] + offs[None, :], 0, n - 1)
        peaks = kick_env[idx].max(axis=1)
        per_hit = 20.0 * np.log10(np.maximum(peaks, EPS) / max(floor, EPS))
        kick_definition_db = _finite(clamp(float(np.median(per_hit)), 0.0, 60.0))
        weakest_hit_db = _finite(clamp(float(np.percentile(per_hit, 25)), -60.0, 60.0))

    # A "kick" has to repeat on a recognisable grid, punch above the low-end
    # floor on most hits, and live in a band that actually carries some of the
    # mix. Spectral flux will happily find "onsets" in the filter's noise floor
    # under a sustained pad, and in white noise it finds dozens; these gates are
    # what stop either becoming a drum machine in the report.
    kick_band_db = _finite(
        20.0
        * np.log10(
            max(float(np.sqrt(np.mean(kick_band ** 2))), EPS)
            / max(float(np.sqrt(np.mean(mid ** 2))), EPS)
        ),
        -120.0,
    )
    kick_detected = bool(
        kick_count >= 3
        and kick_definition_db >= _MIN_DEFINITION_DB
        and weakest_hit_db >= 2.0
        and kick_band_db >= _MIN_KICK_BAND_DB
        and _regularity(onsets) >= _MIN_REGULARITY
    )

    # -- at-onset vs between-onset spectra -----------------------------------
    win = int(min(_SPEC_WIN, max(1024, n // 4)))
    n_fft = int(max(_SPEC_NFFT, win))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    on_starts = onsets[onsets + win <= n] if kick_count else np.zeros(0, dtype=np.int64)
    off_starts = _between_starts(onsets, win, n, sr)

    spec_on = _mean_spectrum(mid, _subsample(on_starts, _MAX_SPEC_WINDOWS), win, n_fft)
    spec_off = _mean_spectrum(mid, _subsample(off_starts, _MAX_SPEC_WINDOWS), win, n_fft)

    if spec_off is None:
        # No usable gap (very short file, or no kick): fall back to a whole-file
        # spectrum so the bass estimate is still a measurement, not a guess.
        stride = max(win // 2, 1)
        fallback = np.arange(0, max(n - win, 0) + 1, stride, dtype=np.int64)
        spec_off = _mean_spectrum(mid, _subsample(fallback, _MAX_SPEC_WINDOWS), win, n_fft)
    if spec_on is None:
        spec_on = spec_off
    if spec_on is None or spec_off is None:  # pathological: nothing measurable
        return _empty(
            kick_count if kick_detected else 0,
            sub_energy_db,
            sub_rumble_db,
            low_end_mono_ratio,
            kick_definition_db if kick_detected else 0.0,
        )

    # The kick's own contribution: what the at-onset windows have that the
    # between-onset windows do not. Without this subtraction a 110 Hz bass wins
    # the argmax inside KICK_HZ and gets reported as the kick's fundamental.
    kick_spec = np.maximum(spec_on - spec_off, 0.0)
    in_kick = (freqs >= _KICK_SEARCH[0]) & (freqs <= _KICK_SEARCH[1])
    if float(kick_spec[in_kick].sum()) <= 1e-3 * float(spec_on[in_kick].sum()):
        kick_spec = spec_on  # nothing separable; fall back to the raw spectrum

    kick_f, kick_p = _peak_freq(kick_spec, freqs, *_KICK_SEARCH)
    bass_f, bass_p = _peak_freq(spec_off, freqs, *_BASS_SEARCH)
    bass_f = _prefer_subharmonic(spec_off, freqs, bass_f, bass_p, _BASS_SEARCH[0])

    collision_db, band_lo, band_hi = _collision(spec_on, spec_off, freqs, kick_f, bass_f)

    # -- sidechain -----------------------------------------------------------
    ducking_depth_db, has_sidechain = _ducking(mid, sr, onsets, kick_detected)

    # -- where the collision is worst ---------------------------------------
    moments = _collision_moments(
        mid, sr, band_lo, band_hi, kick_detected, collision_db, buf.duration
    )

    if not kick_detected:
        # No kick means nothing for the bass to collide with, nothing to duck
        # against, and no fundamental to report. The bass estimate survives; it
        # was measured from the whole file, not from the gaps between hits.
        kick_count = 0
        kick_f = 0.0
        kick_definition_db = 0.0
        collision_db = -40.0
        ducking_depth_db = 0.0
        has_sidechain = False
        moments = []

    return LowEndMeasurement(
        kick_detected=kick_detected,
        kick_fundamental_hz=_finite(round(kick_f, 2)),
        kick_count=kick_count,
        bass_fundamental_hz=_finite(round(bass_f, 2)),
        sub_energy_db=round(sub_energy_db, 2),
        kick_bass_collision_db=round(collision_db, 2),
        ducking_depth_db=round(ducking_depth_db, 2),
        has_sidechain=has_sidechain,
        kick_definition_db=round(kick_definition_db, 2),
        low_end_mono_ratio=round(clamp(low_end_mono_ratio, 0.0, 1.0), 4),
        sub_rumble_db=round(sub_rumble_db, 2),
        collision_moments=moments,
    )


# ---------------------------------------------------------------------------
# pieces of the above
# ---------------------------------------------------------------------------


def _regularity(onsets: np.ndarray) -> float:
    """Share of inter-onset gaps that land on the dominant spacing.

    A four-on-the-floor kick scores 1.0; a syncopated pattern mixing one and two
    beat gaps still scores well above half; peak-picked noise scores low,
    because its gaps are spread over everything the minimum spacing allows.
    """
    if onsets.size < 3:
        return 0.0
    gaps = np.diff(onsets).astype(np.float64)
    median = float(np.median(gaps))
    if median <= 0.0:
        return 0.0
    return float(np.mean(np.abs(gaps - median) <= 0.15 * median))


def _between_starts(onsets: np.ndarray, win: int, n: int, sr: int) -> np.ndarray:
    """Window starts that sit in the gaps between kicks.

    Placed 45 % of the way into each gap and never within 150 ms of the hit, so
    the kick's decay tail is not measured as "the bass".
    """
    if onsets.size < 2:
        return np.zeros(0, dtype=np.int64)
    gaps = np.diff(onsets)
    need = win + int(0.15 * sr)
    usable = gaps >= need
    if not np.any(usable):
        return np.zeros(0, dtype=np.int64)
    lead = np.maximum(int(0.15 * sr), (gaps[usable] * 0.45).astype(np.int64))
    starts = onsets[:-1][usable] + lead
    starts = starts[(starts + win) <= np.minimum(onsets[1:][usable], n)]
    return starts.astype(np.int64)


def _collision(
    spec_on: np.ndarray,
    spec_off: np.ndarray,
    freqs: np.ndarray,
    kick_f: float,
    bass_f: float,
) -> Tuple[float, float, float]:
    """Spectral overlap of the kick's own energy with the bass's, in dB.

    `spec_on - spec_off` isolates what the kick adds on top of whatever the bass
    was already doing. Both distributions are normalised over 25-250 Hz, so the
    result is a shape comparison, not a level comparison: 0 dB means the kick
    and the bass occupy the same frequencies in the same proportions.

    Returns (collision_db, overlap_lo_hz, overlap_hi_hz).
    """
    band = (freqs >= _LOW_LO) & (freqs <= _LOW_HI)
    if not np.any(band):
        return -40.0, 0.0, 0.0

    on = np.maximum(spec_on[band], 0.0)
    off = np.maximum(spec_off[band], 0.0)

    kick_only = np.maximum(on - off, 0.0)
    if kick_only.sum() <= EPS * on.size:
        kick_only = on  # kick indistinguishable from the bed; compare as-is
    if off.sum() <= EPS * off.size or on.sum() <= EPS * on.size:
        return -40.0, 0.0, 0.0

    k = kick_only / (kick_only.sum() + EPS)
    b = off / (off.sum() + EPS)
    overlap = float(np.minimum(k, b).sum())
    collision_db = _finite(clamp(10.0 * np.log10(max(overlap, 1e-4)), -40.0, 0.0), -40.0)

    lo = hi = 0.0
    if kick_f > 0.0 and bass_f > 0.0:
        lo = min(kick_f, bass_f) * 2.0 ** (-1.0 / 6.0)
        hi = max(kick_f, bass_f) * 2.0 ** (1.0 / 6.0)
    return collision_db, lo, hi


def _ducking(
    mid: np.ndarray, sr: int, onsets: np.ndarray, kick_detected: bool
) -> Tuple[float, bool]:
    """Sub level just before each kick vs the deepest point shortly after it.

    Positive = the sub drops when the kick lands, which is what a sidechain
    compressor does. "Before" is the 75th percentile of the pre-hit window (the
    recovered level) and "after" the 10th percentile of the post-hit window (the
    bottom of the duck), because what a producer means by "ducking 4 dB" is the
    depth of the dip, not its average. The post window starts 45 ms in so the
    kick's own transient is not measured as the bass failing to duck.
    """
    if onsets.size < 3:
        return 0.0, False

    n = int(mid.shape[0])
    sub = np.ascontiguousarray(bandpass(mid, sr, SUB_HZ[0], SUB_HZ[1]))
    env = envelope(sub, sr, smooth_ms=12.0)

    pre_off = np.arange(int(_DUCK_PRE[0] * sr), int(_DUCK_PRE[1] * sr), dtype=np.int64)
    post_off = np.arange(int(_DUCK_POST[0] * sr), int(_DUCK_POST[1] * sr), dtype=np.int64)
    if pre_off.size == 0 or post_off.size == 0:
        return 0.0, False

    valid = (onsets + pre_off[0] >= 0) & (onsets + post_off[-1] < n)
    hits = onsets[valid]
    if hits.size < 3:
        return 0.0, False

    pre = np.percentile(env[hits[:, None] + pre_off[None, :]], 75.0, axis=1)
    post = np.percentile(env[hits[:, None] + post_off[None, :]], 10.0, axis=1)

    live = pre > (float(np.percentile(env, 95)) * 0.02)  # ignore hits in silence
    if np.count_nonzero(live) < 3:
        return 0.0, False

    depth = 20.0 * np.log10(np.maximum(pre[live], EPS) / np.maximum(post[live], EPS))
    depth = depth[np.isfinite(depth)]
    if depth.size < 3:
        return 0.0, False

    median_depth = _finite(clamp(float(np.median(depth)), -30.0, 30.0))
    # A sidechain is consistent by construction — it fires on every hit. A bass
    # that merely happens to decay after some kicks is not one.
    consistent = float(np.mean(depth > 1.5))
    has_sidechain = bool(
        kick_detected
        and hits.size >= 4
        and median_depth >= 2.5
        and consistent >= 0.70
        and float(np.percentile(depth, 25)) >= 1.5
    )
    return median_depth, has_sidechain


def _collision_moments(
    mid: np.ndarray,
    sr: int,
    band_lo: float,
    band_hi: float,
    kick_detected: bool,
    collision_db: float,
    duration: float,
) -> List[Moment]:
    """Spans where the low end is most concentrated into the shared region."""
    if not kick_detected or band_hi <= band_lo or collision_db < -12.0:
        return []

    hop = int(round(HOP_SEC * sr))
    n_fft = 8192
    freqs, times, power = _power_spectrogram(mid, sr, n_fft, hop, block=128)
    if power.shape[0] == 0:
        return []

    low = (freqs >= _LOW_LO) & (freqs <= _LOW_HI)
    shared = (freqs >= band_lo) & (freqs <= band_hi)
    if not np.any(low) or not np.any(shared):
        return []

    e_low = power[:, low].sum(axis=1).astype(np.float64)
    e_shared = power[:, shared].sum(axis=1).astype(np.float64)
    active = e_low > (float(np.percentile(e_low, 90)) * 1e-3 + EPS)
    frac_db = 10.0 * np.log10(np.maximum(e_shared, EPS) / np.maximum(e_low, EPS))

    # -6 dB = a quarter of the low end inside the shared band; 0 dB = all of it.
    score = np.clip((frac_db + 6.0) / 6.0, 0.0, 1.0)
    score[~active] = 0.0
    if not np.any(score > 0):
        return []

    thresh = max(0.55, float(np.percentile(score[score > 0], 75)))
    mask = score >= thresh
    if not np.any(mask):
        return []

    spans = merge_spans(times, mask, min_gap=0.75, min_len=0.5)
    moments: List[Moment] = []
    for t0, t1 in spans:
        sel = (times >= t0) & (times <= t1)
        peak = float(score[sel].max()) if np.any(sel) else thresh
        moments.append(
            Moment(
                t_start=round(clamp(float(t0), 0.0, duration), 3),
                t_end=round(clamp(float(t1), 0.0, duration), 3),
                intensity=round(clamp(peak, 0.0, 1.0), 3),
                value=round(_finite(float(frac_db[sel].max()) if np.any(sel) else 0.0), 2),
                label="kick/bass sharing the same band",
            )
        )
    moments.sort(key=lambda m: m.intensity, reverse=True)
    return moments[:8]


def _empty(
    kick_count: int,
    sub_energy_db: float,
    sub_rumble_db: float,
    mono_ratio: float,
    definition_db: float,
) -> LowEndMeasurement:
    """Fully-populated, honest "could not separate kick from bass" result."""
    return LowEndMeasurement(
        kick_detected=False,
        kick_fundamental_hz=0.0,
        kick_count=int(kick_count),
        bass_fundamental_hz=0.0,
        sub_energy_db=round(sub_energy_db, 2),
        kick_bass_collision_db=-40.0,
        ducking_depth_db=0.0,
        has_sidechain=False,
        kick_definition_db=round(definition_db, 2),
        low_end_mono_ratio=round(clamp(mono_ratio, 0.0, 1.0), 4),
        sub_rumble_db=round(sub_rumble_db, 2),
        collision_moments=[],
    )


__all__ = ["measure_low_end"]
