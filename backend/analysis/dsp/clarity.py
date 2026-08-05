"""Mix clarity: masking, congestion and definition.

A mix can measure perfectly against a target curve and still sound like a
blanket. That happens when the energy is *arranged* badly rather than
distributed badly — three sources stacked inside one auditory filter, each one
hiding the others. This module measures the arrangement.

`masking_index` is the headline, and it is the most defensible number in the
product, so it is a real psychoacoustic model rather than an analogy:

1. **ERB filterbank.** The spectrum is projected onto 40 channels spaced one
   ERB apart from 20 Hz to 20 kHz, on the Glasberg & Moore ERB-rate scale with
   bandwidth ERB(f) = 24.7·(4.37·f/1000 + 1). One channel is one place on the
   basilar membrane, so a 40 Hz and a 55 Hz tone stop being two separate
   things down there — which is exactly the resolution at which masking
   questions are worth asking.

2. **Simultaneous masking, asymmetric and level-dependent.** Each band casts a
   masking skirt on its neighbours. Downward (a masker hiding something below
   it) the skirt is steep, 27 dB/Bark. Upward it is shallow — and gets
   shallower as the masker gets louder — at (24 + 0.23/f_kHz − 0.2·L) dB/Bark.
   That asymmetry is the whole reason a loud low end eats a mix's midrange and
   not the other way round. The slopes are specified per Bark and evaluated at
   the Bark distance between the ERB centres, so the ERB grid inherits the
   classic slopes without re-deriving them. Level dependence is handled by
   building one spreading matrix per 10 dB of masker level and mixing them per
   frame, so the whole thing stays a handful of matrix multiplies.

3. **Temporal masking.** A loud transient hides what follows it for 100-200 ms
   (post-masking) and, much more weakly, what precedes it by 5-20 ms
   (pre-masking). Both are applied to the excitation as decayed running
   maxima — causal for post, anti-causal for pre — computed with a cumulative
   maximum in the dB domain, so no per-frame Python loop.

4. **Threshold and index.** The spread excitation is normalised by the
   spreading gain (standard practice: convolving with a spreading function
   adds energy that was never there) and dropped by a tonality-dependent
   signal-to-mask offset, then floored at the absolute threshold of hearing.
   A band whose own excitation sits under that threshold is contributing
   energy nobody can hear as a separate thing. `masking_index` is the
   loudness-weighted fraction of the audible excitation in that state.

Level is normalised to a fixed presentation SPL first, so turning the file
down 6 dB does not change the answer — but the *internal* level asymmetries
that drive spread and audibility are preserved.

`band_congestion` is the same quantity restricted to one macro band: the share
of that band's audible energy that is masked.

What this deliberately does *not* claim: from a two-track you can see energy
that is buried, not sources that are burying each other. Two synths sharing
one ERB band look like one loud band here, and no spectrum-only model can say
otherwise. Source-against-source masking is what `StemAnalysis.masking_pairs`
is for; this module measures the part that is visible without separation, and
the two are complementary rather than redundant.

`definition_db` compares the level of transient frames against sustained
frames. Limiting, over-compression and reverb all pull that number down, and
all three make individual elements stop being individually audible.

Measurement only: no thresholds, no genre knowledge, no verdicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import get_window as sps_get_window

from ..core import (
    EPS,
    MACRO_BAND_ORDER,
    MACRO_BANDS,
    AudioBuffer,
    clamp,
    frame_signal,
    merge_spans,
    percentile_safe,
)
from ..types import ClarityMeasurement, Moment

# --- analysis resolution ---------------------------------------------------

# The masking model runs on its own grid, not the package's 0.25 s one:
# temporal masking lives on a 100-200 ms scale and cannot be measured at all
# with 250 ms hops. 4096 points is 85 ms — long enough that an 11.7 Hz bin
# spacing still resolves the low ERB filters (ERB(50 Hz) = 30 Hz), short enough
# that a 21 ms hop tracks a transient decay. Blackman-Harris because its -92 dB
# sidelobes keep leakage from being reported as buried content.
_MASK_NFFT = 4096
_MASK_HOP = 1024
_MASK_WINDOW = "blackmanharris"
# Frames per FFT batch. Big batches amortise the transform (4x faster than 128
# on a six-minute track); 1024 keeps the working set near 50 MB.
_EXC_BLOCK = 1024

_FLUX_NFFT = 2048             # 42.7 ms
_FLUX_HOP = 512               # 10.7 ms: fine enough to separate hit from tail
_LIBROSA_MAX_SEC = 120.0      # bound librosa's cost on long tracks

# --- auditory model constants ----------------------------------------------

_N_ERB = 40                   # ~1 ERB apart over 20 Hz .. 20 kHz
_ERB_LO_HZ = 20.0
_ERB_HI_HZ = 20000.0
_MIN_ANALYSIS_HZ = 15.0       # below this the filterbank ignores the bin (DC guard)

# Spreading-function slopes, dB per Bark (Terhardt / MPEG psychoacoustic
# model 1 form). Downward is steep and level-independent; upward is shallow
# and gets shallower as the masker gets louder.
_SPREAD_DOWN_DB_PER_BARK = 27.0
_SPREAD_UP_BASE = 24.0
_SPREAD_UP_PER_KHZ = 0.23
_SPREAD_UP_PER_DB = 0.20
_SPREAD_UP_MIN = 5.0
_SPREAD_UP_MAX = 30.0
_LEVEL_BINS_DB = np.arange(20.0, 105.0, 10.0)   # masker levels the bank is built for

# Temporal masking, dB of decay per second of separation.
_POST_DB_PER_SEC = 200.0      # ~30 dB gone after 150 ms
_PRE_DB_PER_SEC = 1800.0      # ~30 dB gone after 17 ms

# Signal-to-mask offset. A tonal masker masks far less effectively than a
# noise masker at the same level, so the offset is interpolated by the
# spectral flatness of the frame's excitation pattern.
#
# The noise offset is deliberately half of _MASK_SOFT_DB. Because the spread
# threshold is renormalised, a spectrum that is flat across the ERB scale
# spreads to exactly itself, and that pairing makes the criterion reduce to
# something you can say in one sentence: a band counts as masked in proportion
# to how many dB it sits *below* the level its neighbourhood spreads onto it,
# reaching fully masked 12 dB under. Pink noise, which is flat per ERB, sits
# exactly at zero — correctly, because you can hear a notch cut into pink
# noise; its bands do not bury each other.
_OFFSET_NOISE_DB = 6.0
_OFFSET_TONAL_DB = 14.0
_SFM_FULL_TONAL_DB = -60.0    # flatness that counts as fully tonal
_SFM_FLOOR_DB = 60.0          # dB under the frame peak that the flatness ignores

_MASK_SOFT_DB = 12.0          # margin over which "masked" fades from 0 to 1

# Absolute threshold of hearing (Terhardt), clamped: the quartic term runs away
# above 16 kHz and the low end of the formula is an average, not a young ear.
_ATH_MIN_DB = -5.0
_ATH_MAX_DB = 70.0

# Playback level the model assumes. Masking is level-dependent, so the model
# needs *a* level; pinning the programme's own RMS to a fixed SPL makes the
# measurement invariant to the file's gain, which is the property we want.
_PRESENTATION_SPL_DB = 80.0
_ACTIVE_FLOOR_DB = 40.0       # frames this far under the loudest are not programme

_LOUDNESS_EXP = 0.23          # Zwicker specific-loudness compression exponent

# `masking_index` is a share of the mix's loudness, and shares of loudness are
# small numbers: an on-target master buries ~2% of itself, a badly arranged one
# ~10-15%. `clarity_index` is not a share of anything — it is a 0-1 verdict
# blend — so it needs to know what counts as "as masked as it gets". 12% of the
# audible loudness sitting under its own masking threshold is that point.
_CLARITY_MASK_FULL = 0.12

_MOMENT_SMOOTH_SEC = 0.30
_MOMENT_MIN_FRACTION = 0.04   # below this a span is not worth a timeline marker

_librosa = None               # cached module handle; the import is the slow part
_BANK_CACHE: Dict[Tuple[int, int], Tuple[np.ndarray, ...]] = {}


# ---------------------------------------------------------------------------
# small private helpers (duplicated rather than shared: sibling dsp modules are
# being written concurrently and this package has no agreed-on internal utils)
# ---------------------------------------------------------------------------


def _finite(x: float, default: float = 0.0) -> float:
    v = float(x)
    return v if np.isfinite(v) else float(default)


def _power_spectrogram(
    x: np.ndarray, sr: int, n_fft: int, hop: int, block: int = 256, window_name: str = "hann"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blocked power spectrogram: (freqs, times, power[n_frames, n_bins]).

    Scaled by the window's coherent gain. Used by `_definition_db`, which only
    ever compares frames against each other; the masking model has its own
    path in `_erb_excitation` that never materialises the full spectrogram.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    frames = frame_signal(x, n_fft, hop)
    n = int(frames.shape[0])
    window = sps_get_window(window_name, n_fft)
    scale = 1.0 / (window.sum() + EPS)

    out = np.empty((n, n_fft // 2 + 1), dtype=np.float32)
    for i in range(0, n, block):  # blocks of frames, not samples
        seg = np.asarray(frames[i : i + block]) * window
        spec = np.fft.rfft(seg, n=n_fft, axis=1)
        out[i : i + block] = ((np.abs(spec) * scale) ** 2).astype(np.float32)

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    times = (np.arange(n) * hop + n_fft * 0.5) / sr
    return freqs, times, out


def _bark(f: np.ndarray) -> np.ndarray:
    """Traunmuller/Zwicker critical-band rate."""
    f = np.asarray(f, dtype=np.float64)
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def _db(x: np.ndarray, floor: float = -200.0) -> np.ndarray:
    return np.maximum(10.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float64), EPS)), floor)


# ---------------------------------------------------------------------------
# the auditory filterbank
# ---------------------------------------------------------------------------


def _erb_number(f: np.ndarray) -> np.ndarray:
    """Glasberg & Moore ERB-rate scale: number of ERBs below f."""
    return 21.4 * np.log10(1.0 + 0.00437 * np.asarray(f, dtype=np.float64))


def _erb_hz(f: np.ndarray) -> np.ndarray:
    """Glasberg & Moore ERB bandwidth in Hz: 24.7 * (4.37 * f/1000 + 1)."""
    return 24.7 * (4.37 * np.asarray(f, dtype=np.float64) / 1000.0 + 1.0)


def _erb_inverse(e: np.ndarray) -> np.ndarray:
    return (10.0 ** (np.asarray(e, dtype=np.float64) / 21.4) - 1.0) / 0.00437


def _erb_centers() -> Tuple[np.ndarray, np.ndarray]:
    """(centre_hz, bandwidth_hz) for _N_ERB filters, one ERB apart."""
    lo, hi = _erb_number(_ERB_LO_HZ), _erb_number(_ERB_HI_HZ)
    centers = _erb_inverse(np.linspace(lo, hi, _N_ERB))
    return centers, _erb_hz(centers)


def _erb_matrix(freqs: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """(n_erb, n_bins) analysis channel weights on the ERB-rate scale.

    Each channel is one ERB wide, one ERB apart, cos^2-shaped in ERB-rate — so
    neighbouring channels sum to exactly 1 and the bank neither loses nor
    double-counts energy. The shape is deliberately *selective* rather than
    roex(p): the ear's filter skirt belongs in the spreading function, and
    putting it in the analysis bank too would both double-count it and destroy
    the measurement. A roex bank smears a loud 1 kHz tone to -21 dB a third of
    an octave away, which is louder than the quiet neighbour we are trying to
    decide the audibility of — the bank would answer the question before the
    masking model got to it. Standard psychoacoustic models (MPEG model 1,
    PEAQ) take the same split: selective partitions in, spreading function
    after. The factor of 2 is the single-sided correction for using an rfft.
    """
    e_c = _erb_number(centers)
    spacing = float(e_c[-1] - e_c[0]) / max(len(e_c) - 1, 1)
    d = (_erb_number(np.maximum(freqs, _MIN_ANALYSIS_HZ))[None, :] - e_c[:, None]) / spacing
    w = np.where(np.abs(d) < 1.0, np.cos(0.5 * np.pi * d) ** 2, 0.0)
    w[:, freqs < _MIN_ANALYSIS_HZ] = 0.0          # DC / rumble bins are not audio
    return 2.0 * w


def _spread_bank(centers: np.ndarray) -> np.ndarray:
    """(n_levels, n_erb, n_erb) linear spreading gains; [level, masker, maskee].

    Slopes are the classic per-Bark ones, evaluated at the *Bark* distance
    between ERB centres — that is the conversion from Bark spacing to ERB
    spacing, and it is frequency-dependent (1 ERB is ~0.35 Bark at 100 Hz and
    ~0.85 Bark at 1 kHz), which a single scalar factor would get wrong at both
    ends. The diagonal is unity: a band is its own strongest masker, and the
    normalisation in `_masking_model` depends on it being included.
    """
    z = _bark(centers)
    dz = z[None, :] - z[:, None]                  # maskee bark - masker bark
    f_khz = np.maximum(centers, 20.0) / 1000.0

    banks = np.empty((_LEVEL_BINS_DB.size, centers.size, centers.size), dtype=np.float64)
    for k, level in enumerate(_LEVEL_BINS_DB):
        up = np.clip(
            _SPREAD_UP_BASE + _SPREAD_UP_PER_KHZ / f_khz - _SPREAD_UP_PER_DB * level,
            _SPREAD_UP_MIN,
            _SPREAD_UP_MAX,
        )
        att_db = np.where(dz >= 0.0, -up[:, None] * dz, _SPREAD_DOWN_DB_PER_BARK * dz)
        banks[k] = 10.0 ** (att_db / 10.0)
    return banks


def _absolute_threshold_db(f: np.ndarray) -> np.ndarray:
    """Threshold in quiet, dB SPL (Terhardt), clamped to a usable range."""
    k = np.maximum(np.asarray(f, dtype=np.float64), 20.0) / 1000.0
    ath = 3.64 * k ** -0.8 - 6.5 * np.exp(-0.6 * (k - 3.3) ** 2) + 1e-3 * k ** 4
    return np.clip(ath, _ATH_MIN_DB, _ATH_MAX_DB)


def _macro_matrix(centers: np.ndarray, widths: np.ndarray) -> Dict[str, np.ndarray]:
    """Per macro band, the share of each ERB filter that falls inside it."""
    lo_hz = centers - widths * 0.5
    hi_hz = centers + widths * 0.5
    out: Dict[str, np.ndarray] = {}
    for name, (lo, hi) in MACRO_BANDS.items():
        overlap = np.minimum(hi_hz, hi) - np.maximum(lo_hz, lo)
        out[name] = np.clip(overlap / np.maximum(widths, EPS), 0.0, 1.0)
    return out


def _erb_excitation(
    x: np.ndarray, sr: int, n_fft: int, hop: int, block: int = _EXC_BLOCK
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(times, excitation[n_frames, n_erb], frame_power[n_frames]).

    The projection onto the filterbank happens inside the block loop, so the
    full spectrogram is never held: six minutes of stereo would be 280 MB of
    float64 bins, against 5 MB of excitation.
    """
    _, _, erb_mat, _, _ = _bank(sr, n_fft)
    x = np.ascontiguousarray(x, dtype=np.float64)
    frames = frame_signal(x, n_fft, hop)
    n = int(frames.shape[0])
    window = sps_get_window(_MASK_WINDOW, n_fft)
    scale = 1.0 / (window.sum() + EPS)

    exc = np.empty((n, erb_mat.shape[1]), dtype=np.float64)
    total = np.empty(n, dtype=np.float64)
    for i in range(0, n, block):
        spec = np.fft.rfft(np.asarray(frames[i : i + block]) * window, n=n_fft, axis=1)
        p = (np.abs(spec) * scale) ** 2
        exc[i : i + block] = p @ erb_mat
        total[i : i + block] = p.sum(axis=1)

    times = (np.arange(n) * hop + n_fft * 0.5) / sr
    return times, np.maximum(exc, 0.0), total


def _bank(sr: int, n_fft: int) -> Tuple[np.ndarray, ...]:
    """Everything that depends only on (sr, n_fft), built once."""
    key = (int(sr), int(n_fft))
    cached = _BANK_CACHE.get(key)
    if cached is not None:
        return cached
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    centers, widths = _erb_centers()
    built = (
        centers,
        widths,
        _erb_matrix(freqs, centers).T,            # (n_bins, n_erb) for power @ mat
        _spread_bank(centers),
        _absolute_threshold_db(centers),
    )
    _BANK_CACHE[key] = built
    return built


# ---------------------------------------------------------------------------
# temporal masking
# ---------------------------------------------------------------------------


def _decayed_max(x_db: np.ndarray, db_per_frame: float, reverse: bool = False) -> np.ndarray:
    """Running max with an exponential (linear-in-dB) decay, vectorised.

    y[n] = max_k<=n ( x[k] - (n-k)*d ) is an IIR max, which reads as a serial
    recursion. Adding k*d before a cumulative maximum and taking it off again
    afterwards turns it into one `np.maximum.accumulate`, exact and with no
    loop over frames.
    """
    n = x_db.shape[0]
    if n == 0 or db_per_frame <= 0.0:
        return x_db
    ramp = (np.arange(n, dtype=np.float64) * db_per_frame)[:, None]
    if reverse:
        ramp = ramp[::-1]
        return np.maximum.accumulate((x_db + ramp)[::-1], axis=0)[::-1] - ramp
    return np.maximum.accumulate(x_db + ramp, axis=0) - ramp


def _temporal_spread(exc_db: np.ndarray, hop_sec: float) -> np.ndarray:
    """Excitation smeared forward (post-masking) and backward (pre-masking)."""
    post = _decayed_max(exc_db, _POST_DB_PER_SEC * hop_sec, reverse=False)
    pre = _decayed_max(exc_db, _PRE_DB_PER_SEC * hop_sec, reverse=True)
    return np.maximum(post, pre)


# ---------------------------------------------------------------------------
# the masking model
# ---------------------------------------------------------------------------


@dataclass
class _Masking:
    """Per-cell masking state for one track. Plain container, no behaviour."""

    times: np.ndarray            # (frames,) centre time of each frame, seconds
    centers: np.ndarray          # (n_erb,) filter centre frequencies, Hz
    widths: np.ndarray           # (n_erb,) filter bandwidths, Hz
    excitation_db: np.ndarray    # (frames, n_erb) dB SPL at the presentation level
    threshold_db: np.ndarray     # (frames, n_erb) masking threshold, dB SPL
    masked: np.ndarray           # (frames, n_erb) 0-1, how buried each cell is
    weight: np.ndarray           # (frames, n_erb) specific loudness, the perceptual weight
    per_frame: np.ndarray        # (frames,) weighted masked share of that frame
    index: float                 # weighted masked share of the whole track


def _presentation_offset_db(frame_power: np.ndarray) -> float:
    """dB to add so the programme sits at _PRESENTATION_SPL_DB, level-invariantly."""
    if frame_power.size == 0:
        return _PRESENTATION_SPL_DB
    peak = float(frame_power.max())
    if peak <= EPS:
        return _PRESENTATION_SPL_DB
    active = frame_power[frame_power > peak * 10.0 ** (-_ACTIVE_FLOOR_DB / 10.0)]
    ref = float(active.mean()) if active.size else peak
    return _PRESENTATION_SPL_DB - 10.0 * np.log10(max(ref, EPS))


def _spread_threshold(exc_db_spl: np.ndarray, banks: np.ndarray) -> np.ndarray:
    """Normalised, level-dependent spread of the excitation pattern, in dB.

    The masker's level picks the spreading matrix, so the bank is applied in
    10 dB slices with linear interpolation between neighbouring slices — nine
    small matmuls instead of one per frame. Dividing by the spread of the
    *weights* renormalises the convolution: a flat excitation pattern spreads
    to itself, which is what makes this a measure of arrangement rather than a
    measure of how much energy the track happens to contain.
    """
    n_lv = banks.shape[0]
    exc = 10.0 ** (exc_db_spl / 10.0)

    idx = np.clip((exc_db_spl - _LEVEL_BINS_DB[0]) / 10.0, 0.0, float(n_lv - 1))
    lo = np.floor(idx)
    frac = idx - lo
    lo_i = lo.astype(np.int32)
    hi_i = np.minimum(lo_i + 1, n_lv - 1)

    num = np.zeros_like(exc)
    den = np.zeros_like(exc)
    for k in range(n_lv):
        w = np.where(lo_i == k, 1.0 - frac, 0.0) + np.where(hi_i == k, frac, 0.0)
        if not w.any():
            continue
        num += (exc * w) @ banks[k]
        den += w @ banks[k]
    return _db(num) - _db(den)


def _masking_model(
    times: np.ndarray, exc: np.ndarray, frame_power: np.ndarray, sr: int, n_fft: int, hop: int
) -> _Masking:
    """Excitation -> spread + temporal masking -> threshold -> masked fraction."""
    centers, widths, _, banks, ath_db = _bank(sr, n_fft)

    exc_db = _db(exc) + _presentation_offset_db(frame_power)

    # Tonality per frame, from the flatness of the excitation pattern. Floored
    # 60 dB under the frame's own peak so that a brick-walled lowpass or a
    # digitally empty air band reads as "no content there" rather than as
    # infinite tonality.
    floored = np.maximum(exc_db, exc_db.max(axis=1, keepdims=True) - _SFM_FLOOR_DB)
    sfm_db = np.mean(floored, axis=1) - _db(np.mean(10.0 ** (floored / 10.0), axis=1))
    alpha = np.clip(sfm_db / _SFM_FULL_TONAL_DB, 0.0, 1.0)[:, None]
    offset_db = alpha * _OFFSET_TONAL_DB + (1.0 - alpha) * _OFFSET_NOISE_DB

    spread_db = _spread_threshold(_temporal_spread(exc_db, hop / float(sr)), banks)
    threshold_db = np.maximum(spread_db - offset_db, ath_db[None, :])

    margin_db = exc_db - threshold_db
    masked = np.clip(0.5 - margin_db / _MASK_SOFT_DB, 0.0, 1.0)

    # Perceptual weight: Zwicker-style specific loudness above the threshold in
    # quiet. Content below the threshold in quiet weighs nothing, so an empty
    # air band cannot be "congested" and a rumbling 20 Hz band only counts once
    # it is actually audible.
    over = np.maximum(exc_db - ath_db[None, :], 0.0)
    weight = np.maximum(10.0 ** (_LOUDNESS_EXP * over / 10.0) - 1.0, 0.0)

    wsum = weight.sum(axis=1)
    per_frame = np.divide(
        (weight * masked).sum(axis=1), wsum, out=np.zeros_like(wsum), where=wsum > EPS
    )
    total = float(weight.sum())
    index = float((weight * masked).sum() / total) if total > EPS else 0.0

    return _Masking(
        times=times,
        centers=centers,
        widths=widths,
        excitation_db=exc_db,
        threshold_db=threshold_db,
        masked=masked,
        weight=weight,
        per_frame=per_frame,
        index=clamp(_finite(index), 0.0, 1.0),
    )


def _congestion(mk: _Masking) -> Dict[str, float]:
    """Per macro band: the share of that band's audible energy that is masked."""
    shares = _macro_matrix(mk.centers, mk.widths)
    out: Dict[str, float] = {}
    for name in MACRO_BAND_ORDER:
        s = shares[name]
        if not np.any(s > 0.0):
            out[name] = 0.0
            continue
        w = mk.weight * s[None, :]
        total = float(w.sum())
        out[name] = round(
            clamp(_finite(float((w * mk.masked).sum() / total) if total > EPS else 0.0), 0.0, 1.0),
            3,
        )
    return out


# ---------------------------------------------------------------------------
# transient definition
# ---------------------------------------------------------------------------


def _definition_db(x: np.ndarray, sr: int) -> float:
    """Level of onset frames minus level of sustained frames.

    Not an onset *count* — a proxy for whether the loud moments in this mix are
    attacks or just more sustain. Limiting and reverb both collapse it.
    """
    n = int(x.shape[0])
    if n < _FLUX_NFFT * 2:
        return 0.0

    _, _, power = _power_spectrogram(x, sr, _FLUX_NFFT, _FLUX_HOP, block=512)
    if power.shape[0] < 8:
        return 0.0

    mag = np.sqrt(power.astype(np.float64))
    flux = np.concatenate([[0.0], np.maximum(np.diff(mag, axis=0), 0.0).sum(axis=1)])
    energy = power.sum(axis=1).astype(np.float64)

    live = energy > (float(np.percentile(energy, 95)) * 1e-4 + EPS)
    if np.count_nonzero(live) < 8:
        return 0.0

    hot = np.percentile(flux[live], 80.0)
    cold = np.percentile(flux[live], 40.0)
    transient = live & (flux >= hot)
    sustained = live & (flux <= cold)
    if not np.any(transient) or not np.any(sustained):
        return 0.0

    ratio = float(energy[transient].mean()) / max(float(energy[sustained].mean()), EPS)
    return _finite(clamp(10.0 * np.log10(max(ratio, EPS)), -20.0, 40.0))


def _librosa_features(x: np.ndarray, sr: int) -> Tuple[float, float]:
    """(spectral_flatness, spectral_contrast_db), averaged over a bounded excerpt."""
    global _librosa
    if _librosa is None:
        try:
            import librosa  # deferred: importing it costs seconds, measuring does not
        except Exception:  # pragma: no cover - librosa is a hard dep of this project
            return 0.0, 0.0
        _librosa = librosa
    librosa = _librosa

    span = int(min(x.shape[0], _LIBROSA_MAX_SEC * sr))
    start = (x.shape[0] - span) // 2
    y = np.ascontiguousarray(x[start : start + span], dtype=np.float32)
    if y.size < 4096:
        y = np.pad(y, (0, 4096 - y.size))

    try:
        flat = float(np.mean(librosa.feature.spectral_flatness(y=y, n_fft=2048, hop_length=512)))
    except Exception:
        flat = 0.0
    try:
        contrast = float(
            np.mean(
                librosa.feature.spectral_contrast(
                    y=y, sr=sr, n_fft=2048, hop_length=512, fmin=100.0, n_bands=6
                )
            )
        )
    except Exception:
        contrast = 0.0
    return clamp(_finite(flat), 0.0, 1.0), _finite(clamp(contrast, 0.0, 80.0))


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def measure_clarity(buf: AudioBuffer) -> ClarityMeasurement:
    """Measure masking, congestion and definition. Never raises on valid audio."""
    sr = buf.sr
    mono = np.ascontiguousarray(buf.mono, dtype=np.float64)

    times, exc, frame_power = _erb_excitation(mono, sr, _MASK_NFFT, _MASK_HOP)
    if times.size == 0 or not np.any(frame_power > 0.0):
        return _neutral()

    mk = _masking_model(times, exc, frame_power, sr, _MASK_NFFT, _MASK_HOP)
    masking_index = mk.index
    band_congestion = _congestion(mk)

    worst_congested_band: Optional[str] = None
    if band_congestion:
        worst_congested_band = max(band_congestion, key=lambda k: band_congestion[k])
        if band_congestion[worst_congested_band] <= 0.0:
            worst_congested_band = None

    definition_db = _definition_db(mono, sr)
    spectral_flatness, spectral_contrast = _librosa_features(mono, sr)

    mean_congestion = (
        float(np.mean(list(band_congestion.values()))) if band_congestion else 0.0
    )
    masked_severity = clamp(masking_index / _CLARITY_MASK_FULL, 0.0, 1.0)
    congested_severity = clamp(mean_congestion / _CLARITY_MASK_FULL, 0.0, 1.0)
    clarity_index = clamp(
        0.40 * (1.0 - masked_severity)
        + 0.25 * clamp(definition_db / 10.0, 0.0, 1.0)
        + 0.20 * (1.0 - congested_severity)
        + 0.15 * clamp((spectral_contrast - 8.0) / 16.0, 0.0, 1.0),
        0.0,
        1.0,
    )

    congested_moments = _moments(mk, buf.duration, _MASK_HOP / float(sr))

    return ClarityMeasurement(
        clarity_index=round(_finite(clarity_index), 3),
        spectral_flatness=round(_finite(spectral_flatness), 5),
        spectral_contrast=round(_finite(spectral_contrast), 2),
        masking_index=round(_finite(masking_index), 3),
        band_congestion=band_congestion,
        worst_congested_band=worst_congested_band,
        definition_db=round(_finite(definition_db), 2),
        congested_moments=congested_moments,
    )


def _moments(mk: _Masking, duration: float, hop_sec: float) -> List[Moment]:
    """The worst spans, by loudness-weighted masked fraction of the frame."""
    times = mk.times
    score = np.array(mk.per_frame, dtype=np.float64, copy=True)
    if score.size == 0:
        return []

    # 21 ms frames flicker; the report wants regions, not frames.
    span = max(int(round(_MOMENT_SMOOTH_SEC / max(hop_sec, 1e-6))), 1)
    if span > 1 and score.size > span:
        kernel = np.ones(span) / float(span)
        score = np.convolve(score, kernel, mode="same")

    loud = mk.weight.sum(axis=1)
    if loud.size:
        score[loud < float(np.percentile(loud, 95)) * 0.05] = 0.0
    if not np.any(score > 0):
        return []

    # A span earns a timeline marker when it is this track's worst *and*
    # masked enough in absolute terms to be worth scrubbing to — otherwise an
    # already-clear mix gets eight markers pointing at its clearest moments.
    threshold = max(_MOMENT_MIN_FRACTION, percentile_safe(score[score > 0], 85.0))
    mask = score >= threshold
    if not np.any(mask):
        return []

    out: List[Moment] = []
    for t0, t1 in merge_spans(times, mask, min_gap=0.75, min_len=0.5):
        sel = (times >= t0) & (times <= t1)
        if not np.any(sel):
            continue
        peak = float(score[sel].max())
        out.append(
            Moment(
                t_start=round(clamp(float(t0), 0.0, duration), 3),
                t_end=round(clamp(float(t1), 0.0, duration), 3),
                # `value` is the measured share of loudness masked in the span;
                # `intensity` is that share on the same 0-1 severity scale the
                # clarity index uses, because it drives marker size.
                intensity=round(clamp(_finite(peak) / _CLARITY_MASK_FULL, 0.0, 1.0), 3),
                value=round(_finite(peak), 4),
                label="masked / congested",
            )
        )
    out.sort(key=lambda m: m.intensity, reverse=True)
    return out[:8]


def _neutral() -> ClarityMeasurement:
    """Nothing measurable — fully populated, opinion-free."""
    return ClarityMeasurement(
        clarity_index=0.5,
        spectral_flatness=0.0,
        spectral_contrast=0.0,
        masking_index=0.0,
        band_congestion={name: 0.0 for name in MACRO_BAND_ORDER},
        worst_congested_band=None,
        definition_db=0.0,
        congested_moments=[],
    )


__all__ = ["measure_clarity"]
