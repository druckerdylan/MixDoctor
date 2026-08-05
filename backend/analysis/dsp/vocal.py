"""Vocal balance, estimated from the centre channel.

Be honest about what this is: there are no stems. Everything below is inferred
from a centre estimate, and a centre estimate is not a vocal — a centred synth
pad, a snare and a mono bass all live there too. Two things keep the numbers
defensible:

* **Centre extraction.** Per time-frequency cell, `1 - |S| / |M|` is the share
  of the mid channel that is not explained by the side channel. A perfectly
  centred source gives 1, a hard-panned source gives 0, a 70/30 pan gives 0.6.
  Multiplying the mid magnitude by that mask gives the centre; everything else
  (the rest of the mid, plus all of the side) is "instruments".

* **A voice test, not a level test.** `vocal_present` is decided by the
  temporal signature — sustained energy in 300 Hz-6 kHz whose envelope is
  modulated at the syllabic rate, 2-8 Hz — and, crucially, by whether that
  modulation is *specific to the centre*. A hard-limited mix pumps its whole
  spectrum at the tempo; comparing the centre's syllabic modulation against the
  non-centre's stops that pumping from being reported as a singer.

On a mono source there is no side channel, so centre extraction is meaningless.
Rather than invent a verdict, `vocal_present` is False and the separation-derived
figures are returned neutral; the purely spectral ones are still measured.

Levels use A-weighting, so "is my vocal too quiet" is answered against what the
ear hears rather than against the sub, which otherwise owns the energy budget
and would make every mix look vocal-starved.

Measurement only: no thresholds, no genre knowledge, no verdicts.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..core import (
    EPS,
    HOP_SEC,
    MACRO_BANDS,
    SIBILANCE_HZ,
    VOCAL_HZ,
    AudioBuffer,
    clamp,
    frame_signal,
    merge_spans,
    percentile_safe,
)
from ..types import Moment, VocalMeasurement

# --- analysis constants ----------------------------------------------------

_COARSE_NFFT = 8192                 # 170 ms / 5.9 Hz — the HOP_SEC-rate pass
_FINE_NFFT = 4096                   # 85 ms: resolves envelope modulation to ~11 Hz
_FINE_HOP = 600                     # 12.5 ms -> 80 Hz envelope rate
_FINE_MAX_SEC = 90.0                # bound the modulation pass on long tracks

_BODY_HZ = (300.0, 1000.0)
_CONSONANT_HZ = (1000.0, 4000.0)
_PRESENCE_HZ = (2000.0, 6000.0)
_SYLLABIC_HZ = (2.0, 8.0)
_MOD_TOTAL_HZ = (0.5, 16.0)
# Below this the envelope is effectively flat and its modulation *share* is
# computed from numerical noise, so the share is not evidence of anything.
_MIN_SYLLABIC_DEPTH = 0.05      # 5 % AM, i.e. about 0.4 dB of swing

# Macro bands that can actually mask a vocal. Sub and low_bass cannot.
_VOCAL_MACRO = ("low_mid", "mid", "upper_mid", "presence", "brilliance")

_MIN_CENTRE_RATIO = 0.03            # below this there is no centre to describe
_MASK_WITHIN_DB = 3.0               # "non-centre sits within ~3 dB of the centre"
_MOMENT_DB = 6.0                    # how far from its own median counts as an event


# ---------------------------------------------------------------------------
# small private helpers (duplicated rather than shared: sibling dsp modules are
# being written concurrently and this package has no agreed-on internal utils)
# ---------------------------------------------------------------------------


def _finite(x: float, default: float = 0.0) -> float:
    v = float(x)
    return v if np.isfinite(v) else float(default)


def _power_spectrogram(
    x: np.ndarray, sr: int, n_fft: int, hop: int, block: int = 256
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blocked power spectrogram: (freqs, times, power[n_frames, n_bins])."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    frames = frame_signal(x, n_fft, hop)
    n = int(frames.shape[0])
    window = np.hanning(n_fft)
    scale = 1.0 / (window.sum() + EPS)

    out = np.empty((n, n_fft // 2 + 1), dtype=np.float32)
    for i in range(0, n, block):  # blocks of frames, not samples
        seg = np.asarray(frames[i : i + block]) * window
        spec = np.fft.rfft(seg, n=n_fft, axis=1)
        out[i : i + block] = ((np.abs(spec) * scale) ** 2).astype(np.float32)

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    times = (np.arange(n) * hop + n_fft * 0.5) / sr
    return freqs, times, out


def _a_weight_power(freqs: np.ndarray) -> np.ndarray:
    """IEC 61672 A-weighting as a linear power gain per FFT bin.

    Used so the vocal-versus-instrument ratio reflects audibility: without it a
    55 Hz kick outweighs the entire vocal band and every mix reads as
    vocal-starved.
    """
    f = np.asarray(freqs, dtype=np.float64)
    f2 = f * f
    num = (12194.0 ** 2) * f2 * f2
    den = (
        (f2 + 20.6 ** 2)
        * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
        * (f2 + 12194.0 ** 2)
    )
    ra = num / np.maximum(den, EPS)
    a_db = 20.0 * np.log10(np.maximum(ra, 1e-12)) + 2.0
    gain = 10.0 ** (a_db / 10.0)          # power domain
    gain[f < 10.0] = 0.0                  # kill DC / infrasonic leakage outright
    return gain


def _band_mask(freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (freqs >= lo) & (freqs < hi)


def _db(x: float | np.ndarray, floor: float = -120.0):
    return np.maximum(10.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float64), EPS)), floor)


def _ratio_db(num: float, den: float, floor: float = -60.0, ceil: float = 60.0) -> float:
    if den <= EPS:
        return floor
    return _finite(clamp(10.0 * np.log10(max(num, EPS) / den), floor, ceil), floor)


# ---------------------------------------------------------------------------
# centre extraction
# ---------------------------------------------------------------------------


def _centre_mask(power_mid: np.ndarray, power_side: np.ndarray) -> np.ndarray:
    """Per-cell share of the mid channel not explained by the side channel."""
    mag_m = np.sqrt(power_mid)
    mag_s = np.sqrt(power_side)
    return np.clip(1.0 - mag_s / (mag_m + 1e-20), 0.0, 1.0).astype(np.float32)


def _syllabic(env: np.ndarray, rate: float) -> Tuple[float, float]:
    """(share, depth) of the envelope's modulation at the syllable rate.

    `share` is the fraction of 0.5-16 Hz modulation energy that falls in 2-8 Hz.
    `depth` is the absolute size of that modulation, as a fraction of the mean
    level. Both are needed: a dead-steady pad has a meaningless share computed
    from numerical noise, and only the depth tells you it is dead steady.
    """
    e = np.asarray(env, dtype=np.float64)
    if e.size < 32 or not np.isfinite(e).all():
        return 0.0, 0.0
    mean = float(e.mean())
    if mean <= EPS:
        return 0.0, 0.0
    e = e / mean - 1.0
    variance = float(np.mean(e * e))
    if variance <= 1e-12:
        return 0.0, 0.0

    spec = np.abs(np.fft.rfft(e * np.hanning(e.size))) ** 2
    freqs = np.fft.rfftfreq(e.size, 1.0 / rate)
    spec[0] = 0.0

    total = float(spec[(freqs >= _MOD_TOTAL_HZ[0]) & (freqs < _MOD_TOTAL_HZ[1])].sum())
    if total <= EPS:
        return 0.0, 0.0
    syl = float(spec[(freqs >= _SYLLABIC_HZ[0]) & (freqs < _SYLLABIC_HZ[1])].sum())
    share = clamp(syl / total, 0.0, 1.0)
    depth = float(np.sqrt(max(syl / max(float(spec.sum()), EPS), 0.0) * variance))
    return share, clamp(depth, 0.0, 4.0)


def _modulation(buf: AudioBuffer) -> Tuple[float, float]:
    """(centre syllabic ratio, side-channel syllabic ratio) over a bounded excerpt.

    The reference is the **side channel**, not "everything that isn't the
    centre". The side channel is mask-free — no centred source appears in it at
    all — so it is an independent witness to whether the whole arrangement is
    modulating at 2-8 Hz. `total - centre` is not independent: it moves in
    antiphase with the mask, so a real vocal makes its own reference light up.

    A hard-limited mix pumps every channel at the tempo, which lands inside the
    syllabic band; comparing against the side channel is what stops that
    pumping from being reported as a singer.
    """
    sr = buf.sr
    n = int(buf.n_samples)
    span = int(min(n, _FINE_MAX_SEC * sr))
    start = (n - span) // 2                      # the middle: intros are often bare
    mid = np.ascontiguousarray(buf.mid[start : start + span], dtype=np.float64)
    side = np.ascontiguousarray(buf.side[start : start + span], dtype=np.float64)
    if mid.shape[0] < _FINE_NFFT + _FINE_HOP:
        return 0.0, 0.0

    freqs, _, pm = _power_spectrogram(mid, sr, _FINE_NFFT, _FINE_HOP, block=512)
    _, _, ps = _power_spectrogram(side, sr, _FINE_NFFT, _FINE_HOP, block=512)
    if pm.shape[0] < 32:
        return 0.0, 0.0

    mask = _centre_mask(pm, ps)
    band = _band_mask(freqs, *VOCAL_HZ)
    centre = (mask * mask * pm)[:, band].sum(axis=1).astype(np.float64)
    sides = ps[:, band].sum(axis=1).astype(np.float64)

    rate = sr / float(_FINE_HOP)
    share_c, depth_c = _syllabic(np.sqrt(centre), rate)
    mod_centre = share_c if depth_c >= _MIN_SYLLABIC_DEPTH else 0.0

    # With almost no side energy, or a side channel that barely moves, there is
    # nothing to compare against; the centre gets judged on its own.
    if sides.sum() < 0.01 * float(pm[:, band].sum()):
        return mod_centre, 0.0
    share_s, depth_s = _syllabic(np.sqrt(sides), rate)
    return mod_centre, (share_s if depth_s >= _MIN_SYLLABIC_DEPTH else 0.0)


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def measure_vocal(buf: AudioBuffer) -> VocalMeasurement:
    """Measure vocal balance from a centre estimate. Never raises on valid audio."""
    sr = buf.sr
    hop = int(round(HOP_SEC * sr))
    mid = np.ascontiguousarray(buf.mid, dtype=np.float64)
    side = np.ascontiguousarray(buf.side, dtype=np.float64)

    freqs, times, pm = _power_spectrogram(mid, sr, _COARSE_NFFT, hop, block=128)
    if pm.shape[0] == 0:
        return _neutral(buf.is_mono)
    _, _, ps = _power_spectrogram(side, sr, _COARSE_NFFT, hop, block=128)

    mask = _centre_mask(pm, ps)
    centre_p = (mask * mask * pm).astype(np.float64)          # centre power per cell
    total_p = (pm.astype(np.float64) + ps.astype(np.float64))  # L^2 + R^2, halved
    other_p = np.maximum(total_p - centre_p, 0.0)             # sides + off-centre mid

    aw = _a_weight_power(freqs)
    centre_a = centre_p * aw
    total_a = total_p * aw

    voc = _band_mask(freqs, *VOCAL_HZ)

    # -- centre share --------------------------------------------------------
    c_voc = float(centre_p[:, voc].sum())
    t_voc = float(total_p[:, voc].sum())
    center_energy_ratio = 1.0 if t_voc <= EPS else clamp(c_voc / t_voc, 0.0, 1.0)

    # -- purely spectral figures: valid mono or stereo -----------------------
    c_body = float(centre_p[:, _band_mask(freqs, *_BODY_HZ)].sum())
    c_pres = float(centre_p[:, _band_mask(freqs, *_PRESENCE_HZ)].sum())
    presence_balance_db = _ratio_db(c_pres, c_body, -40.0, 40.0)

    sib_band = _band_mask(freqs, *SIBILANCE_HZ)
    sib_series = centre_p[:, sib_band].sum(axis=1)
    voc_series = centre_p[:, voc].sum(axis=1)
    sibilance_db = _ratio_db(
        percentile_safe(sib_series, 95.0), max(float(np.median(voc_series)), EPS), -60.0, 20.0
    )

    if buf.is_mono:
        # No side channel: the "centre" is just the mix. Report the spectral
        # facts, stay silent on everything separation-derived.
        return VocalMeasurement(
            vocal_present=False,
            center_energy_ratio=1.0,
            vocal_to_instrument_db=0.0,
            intelligibility_index=0.5,
            presence_balance_db=round(presence_balance_db, 2),
            sibilance_db=round(sibilance_db, 2),
            consistency_db=0.0,
            masked_bands=[],
            buried_moments=[],
            loud_moments=[],
        )

    # -- vocal vs instruments, A-weighted ------------------------------------
    c_voc_a = float(centre_a[:, voc].sum())
    rest_a = float(total_a.sum()) - c_voc_a
    vocal_to_instrument_db = _ratio_db(c_voc_a, max(rest_a, 0.0), -40.0, 40.0)

    if center_energy_ratio < _MIN_CENTRE_RATIO:
        # A polarity-inverted or fully-decorrelated mix has no centre at all.
        # The one figure still worth reporting is how far down it is; the
        # spectral shape of what's left is measuring cancellation residue.
        return VocalMeasurement(
            vocal_present=False,
            center_energy_ratio=round(center_energy_ratio, 4),
            vocal_to_instrument_db=round(vocal_to_instrument_db, 2),
            intelligibility_index=0.0,
            presence_balance_db=0.0,
            sibilance_db=-60.0,
            consistency_db=0.0,
            masked_bands=[],
            buried_moments=[],
            loud_moments=[],
        )

    # -- intelligibility -----------------------------------------------------
    cons = _band_mask(freqs, *_CONSONANT_HZ)
    c_cons = float(centre_p[:, cons].sum())
    o_cons = float(other_p[:, cons].sum())
    snr_db = _ratio_db(c_cons, o_cons, -40.0, 40.0)
    share_db = _ratio_db(c_cons, max(c_voc, EPS), -40.0, 0.0)
    # -18 dB of consonant SNR is unintelligible, +10 dB is in the clear. The
    # second term stops a centre made entirely of low-mid body from scoring on
    # SNR alone: no consonants means no words, however clean the background.
    intelligibility_index = clamp(
        (
            0.65 * clamp((snr_db + 18.0) / 28.0, 0.0, 1.0)
            + 0.35 * clamp((share_db + 24.0) / 20.0, 0.0, 1.0)
        )
        # Both terms are ratios *within* the centre, so a centre that barely
        # exists can still score well on them. Scale by how much centre there is.
        * clamp(center_energy_ratio / 0.15, 0.0, 1.0),
        0.0,
        1.0,
    )

    # -- level over time -----------------------------------------------------
    v_db = np.asarray(_db(voc_series), dtype=np.float64)
    active = v_db > (float(v_db.max()) - 30.0)
    if np.count_nonzero(active) < 2:
        active = np.ones_like(v_db, dtype=bool)
    consistency_db = _finite(
        clamp(
            percentile_safe(v_db[active], 90.0) - percentile_safe(v_db[active], 10.0),
            0.0,
            60.0,
        )
    )

    # -- which bands are crowding the centre ---------------------------------
    masked_bands = _masked_bands(freqs, centre_p, other_p)

    # -- where the vocal drops out or jumps ----------------------------------
    reference = percentile_safe(v_db[active], 50.0)
    buried_moments = _moments(times, reference - v_db, "vocal buried", -1.0, buf.duration)
    loud_moments = _moments(
        times, v_db - reference, "vocal jumps forward", 1.0, buf.duration
    )

    # -- is it actually a voice ---------------------------------------------
    mod_centre, mod_side = _modulation(buf)
    vocal_present = bool(
        center_energy_ratio >= 0.12
        and c_voc_a > EPS
        and vocal_to_instrument_db > -25.0
        and mod_centre >= 0.30
        and mod_centre >= mod_side * 1.15 + 0.05
    )

    return VocalMeasurement(
        vocal_present=vocal_present,
        center_energy_ratio=round(center_energy_ratio, 4),
        vocal_to_instrument_db=round(vocal_to_instrument_db, 2),
        intelligibility_index=round(intelligibility_index, 3),
        presence_balance_db=round(presence_balance_db, 2),
        sibilance_db=round(sibilance_db, 2),
        consistency_db=round(consistency_db, 2),
        masked_bands=masked_bands,
        buried_moments=buried_moments,
        loud_moments=loud_moments,
    )


# ---------------------------------------------------------------------------
# pieces of the above
# ---------------------------------------------------------------------------


def _masked_bands(
    freqs: np.ndarray, centre_p: np.ndarray, other_p: np.ndarray
) -> List[str]:
    """Macro bands where non-centre content sits within ~3 dB of the centre."""
    centre_by_band: Dict[str, float] = {}
    other_by_band: Dict[str, float] = {}
    for name in _VOCAL_MACRO:
        lo, hi = MACRO_BANDS[name]
        m = _band_mask(freqs, lo, hi)
        centre_by_band[name] = float(centre_p[:, m].sum())
        other_by_band[name] = float(other_p[:, m].sum())

    peak = max(centre_by_band.values(), default=0.0)
    if peak <= EPS:
        return []

    out: List[str] = []
    for name in _VOCAL_MACRO:
        c = centre_by_band[name]
        # Skip bands where the centre carries nothing worth defending.
        if c <= EPS or 10.0 * np.log10(c / peak) < -25.0:
            continue
        if 10.0 * np.log10(max(other_by_band[name], EPS) / c) >= -_MASK_WITHIN_DB:
            out.append(name)
    return out


def _moments(
    times: np.ndarray, excess_db: np.ndarray, label: str, sign: float, duration: float
) -> List[Moment]:
    """Spans where the vocal sits `_MOMENT_DB` below / above its own median."""
    if times.size == 0:
        return []
    mask = excess_db >= _MOMENT_DB
    if not np.any(mask):
        return []

    spans = merge_spans(times, mask, min_gap=0.75, min_len=0.5)
    out: List[Moment] = []
    for t0, t1 in spans:
        sel = (times >= t0) & (times <= t1)
        if not np.any(sel):
            continue
        worst = float(excess_db[sel].max())
        out.append(
            Moment(
                t_start=round(clamp(float(t0), 0.0, duration), 3),
                t_end=round(clamp(float(t1), 0.0, duration), 3),
                intensity=round(clamp((worst - _MOMENT_DB) / 12.0 + 0.25, 0.0, 1.0), 3),
                # A dropout against a floored level can read as -100 dB, which
                # is true and useless. 40 dB down is already "it disappeared".
                value=round(_finite(sign * clamp(worst, 0.0, 40.0)), 2),
                label=label,
            )
        )
    out.sort(key=lambda m: m.intensity, reverse=True)
    return out[:8]


def _neutral(is_mono: bool) -> VocalMeasurement:
    """Nothing measurable — return a fully populated, opinion-free result."""
    return VocalMeasurement(
        vocal_present=False,
        center_energy_ratio=1.0 if is_mono else 0.0,
        vocal_to_instrument_db=0.0,
        intelligibility_index=0.5,
        presence_balance_db=0.0,
        sibilance_db=-60.0,
        consistency_db=0.0,
        masked_bands=[],
        buried_moments=[],
        loud_moments=[],
    )


__all__ = ["measure_vocal"]
