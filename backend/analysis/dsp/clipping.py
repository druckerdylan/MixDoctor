"""Clipping, flat-topping and inter-sample overs.

Three different things get called "clipping" and only one of them shows up in a
level meter:

  * samples pinned at the ceiling — the count, and what share of the file;
  * *runs* of pinned samples — the actual signature of hard clipping, as
    opposed to a peak that merely touched the ceiling once;
  * inter-sample peaks — inaudible in the file, audible after a lossy encoder
    reconstructs the waveform between the samples.

Two things make this harder than `abs(x) >= 0.9995`:

1. Analysis runs at 48 kHz, and most uploads are 44.1 kHz. Sample-rate
   conversion puts ripple on what used to be a bit-exact flat top: a
   synthetic hard-clipped 220 Hz square measures 78 % of samples pinned at
   native rate and *zero* runs at 48 kHz if you insist on 0.9995 of the peak.
   So "at the ceiling" is a narrow band below the file's own ceiling, wide
   enough to survive conversion (0.1 dB).
2. That band, on its own, flags any sine sitting near full scale — the top of
   a 55 Hz cycle spends 30 samples inside it. What separates a plateau from a
   peak is *flatness*: a clipped run barely moves inside the band, while a
   sinusoid traverses the whole of it. Measured on this repo's fixtures, real
   clipping runs at 0.1-0.5 of the band and sine peaks at 0.6-0.97, so runs
   are only counted when they stay under half the band's width.

Everything is array work. Run lengths come from `np.diff` on the mask and
`np.flatnonzero` on the transitions; per-run extremes come from
`np.maximum.reduceat`. Nothing walks samples in Python.

Measurement only. No genre knowledge, no thresholds, no verdicts.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
from scipy import signal as sps

from ..core import (
    EPS,
    HOP_SEC,
    AudioBuffer,
    amp_to_db,
    clamp,
    frame_signal,
    frame_times,
    merge_spans,
)
from ..types import ClippingMeasurement, Moment

__all__ = ["measure_clipping"]


# The literal rule: within 0.00434 dB of full scale.
CEILING_FRACTION = 0.9995

# A master limited to -0.3 dBTP, or a 44.1 kHz clip resampled to 48 kHz, never
# reaches that. When a file is already sitting on a ceiling near full scale,
# that ceiling is what we measure against, inside a resampling-proof band.
NEAR_FULL_SCALE_DB = -2.0     # peak this close to FS => the file has a ceiling
CEILING_TOLERANCE_DB = -0.10  # band below the ceiling that counts as "at" it

# Fraction of the band a run may span and still be called flat. Calibrated
# against sine peaks (0.61-0.97 of the band) versus clipped plateaus (0.0-0.5).
FLATNESS_MAX = 0.50

FLAT_RUN_MIN_SAMPLES = 4  # shorter than this, flatness can't be told from luck
FLAT_RUN_REPORT_MIN = 3   # `flat_run_count` counts runs longer than this

OVERSAMPLE = 4            # BS.1770-4 Annex 2
HF_SPLIT_HZ = 10_000.0    # clipping products land above the program material
DISTORTION_FULL_SCALE_DB = 8.0  # HF excess mapping to distortion_index 1.0
MAX_EVENTS = 12
DB_FLOOR = -120.0


# ---------------------------------------------------------------------------
# Helpers
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


def _run_bounds(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Start/end indices of every True run in a boolean mask. Vectorised."""
    empty = np.zeros(0, dtype=np.int64)
    if mask.size == 0 or not mask.any():
        return empty, empty
    # Sentinel-padded diff: +1 opens a run, -1 closes it.
    padded = np.zeros(mask.size + 2, dtype=np.int8)
    padded[1:-1] = mask
    edges = np.diff(padded)
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _segment_span(values: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Peak-to-peak of `values` inside each [start, end) segment, without a loop.

    `reduceat` over interleaved [s0, e0, s1, e1, ...] reduces each segment in
    turn; the odd results are the gaps between runs and get discarded.
    """
    if starts.size == 0:
        return np.zeros(0, dtype=np.float64)
    bounds = np.empty(starts.size * 2, dtype=np.int64)
    bounds[0::2] = starts
    bounds[1::2] = ends
    # reduceat treats a trailing index as "to the end"; one pad element keeps
    # bounds[-1] == len(values) in range.
    padded = np.append(values, values[-1])
    highs = np.maximum.reduceat(padded, bounds)[0::2]
    lows = np.minimum.reduceat(padded, bounds)[0::2]
    return highs - lows


def _flat_runs(
    magnitude: np.ndarray, ceiling: float, band: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plateau runs at the ceiling: (starts, lengths, membership mask).

    A run qualifies when it is long enough to be more than sampling luck and
    flat enough that a sinusoid could not have produced it.
    """
    empty_i = np.zeros(0, dtype=np.int64)
    no_mask = np.zeros(magnitude.size, dtype=bool)
    if magnitude.size == 0 or band <= 0.0:
        return empty_i, empty_i, no_mask

    starts, ends = _run_bounds(magnitude >= ceiling)
    if starts.size == 0:
        return empty_i, empty_i, no_mask

    lengths = ends - starts
    keep = lengths >= FLAT_RUN_MIN_SAMPLES
    starts, ends, lengths = starts[keep], ends[keep], lengths[keep]
    if starts.size == 0:
        return empty_i, empty_i, no_mask

    keep = _segment_span(magnitude, starts, ends) <= FLATNESS_MAX * band
    starts, lengths = starts[keep], lengths[keep]
    if starts.size == 0:
        return empty_i, empty_i, no_mask

    # Rebuild membership from the surviving runs: +1 at each start, -1 one past
    # each end, cumulative sum gives the mask. Still no Python loop.
    marks = np.zeros(magnitude.size + 1, dtype=np.int32)
    np.add.at(marks, starts, 1)
    np.add.at(marks, starts + lengths, -1)
    return starts, lengths, np.cumsum(marks)[: magnitude.size] > 0


def _oversample_scan(
    channels: Sequence[np.ndarray],
    over_threshold: float,
    factor: int = OVERSAMPLE,
    chunk: int = 1 << 21,
) -> Tuple[float, int]:
    """One 4x-oversampling pass: (true peak linear, inter-sample over count).

    Inter-sample overs are oversampled samples above 0 dBFS that were *not*
    already over at base rate — base-rate overs are dilated by two oversampled
    taps each side first, so the reconstruction ripple around an already-clipped
    sample isn't counted as a new over.

    Chunked with padded, trimmed edges: a whole 10-minute track oversampled in
    one allocation would be about a gigabyte.
    """
    peak = 0.0
    overs = 0
    pad = 256
    count_overs = over_threshold > 0.0 and math.isfinite(over_threshold)

    for chan in channels:
        # float32 puts the resampler's error near -144 dBFS — far below any
        # peak decision — and roughly halves the cost of the dominant step.
        chan = np.ascontiguousarray(chan, dtype=np.float32)
        n = chan.shape[0]
        if n == 0:
            continue
        peak = max(peak, float(np.abs(chan).max()))
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            a, b = max(0, start - pad), min(n, end + pad)
            up = sps.resample_poly(chan[a:b], factor, 1)
            lo = (start - a) * factor
            hi = min(up.size, lo + (end - start) * factor)
            if hi <= lo:
                continue
            seg = np.abs(up[lo:hi])
            peak = max(peak, float(seg.max()))
            if not count_overs:
                continue

            base_over = np.abs(chan[start:end]) > over_threshold
            expanded = np.repeat(base_over, factor)
            if expanded.size < seg.size:
                expanded = np.concatenate(
                    [expanded, np.zeros(seg.size - expanded.size, dtype=bool)]
                )
            expanded = expanded[: seg.size]
            dilated = expanded.copy()
            for shift in (1, 2):
                dilated[shift:] |= expanded[:-shift]
                dilated[:-shift] |= expanded[shift:]
            overs += int(np.count_nonzero((seg > over_threshold) & ~dilated))

    return peak, overs


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def measure_clipping(buf: AudioBuffer) -> ClippingMeasurement:
    """Clipping evidence for one buffer."""
    sr = int(buf.sr)
    n = int(buf.n_samples)
    hop = max(1, int(round(HOP_SEC * sr)))

    channels = (buf.left,) if buf.is_mono else (buf.left, buf.right)
    magnitudes = [np.ascontiguousarray(np.abs(c)) for c in channels]
    n_channels = len(magnitudes)

    buffer_peak = float(max((m.max() for m in magnitudes), default=0.0))
    decoded_peak = max(float(buf.peak_sample), buffer_peak)

    # Float formats carry values above 1.0 legitimately. There is no digital
    # ceiling in that file, so nothing gets called clipping for being over
    # unity; only genuine plateaus at the material's own ceiling count.
    is_float_over_unity = bool(decoded_peak > 1.0 + 1e-6)

    # Disabling both absolute tests for such a file is the whole point: 0 dBFS
    # is not this file's ceiling, so neither "at 0.9995" nor "over 0 dBFS" means
    # anything, and only the plateau test below still applies.
    strict_threshold = math.inf if is_float_over_unity else CEILING_FRACTION
    over_threshold = math.inf if is_float_over_unity else 1.0
    # A float file's ceiling is its own peak; a PCM file only has one if it is
    # sitting close enough to 0 dBFS to have been limited there.
    near_full_scale = is_float_over_unity or (
        buffer_peak >= 10.0 ** (NEAR_FULL_SCALE_DB / 20.0)
    )
    if near_full_scale and buffer_peak > 0.0:
        ceiling = buffer_peak * 10.0 ** (CEILING_TOLERANCE_DB / 20.0)
    else:
        ceiling = CEILING_FRACTION
    band = max(buffer_peak - ceiling, 0.0)

    longest_flat_run = 0
    flat_run_count = 0
    clip_mask_any = np.zeros(n, dtype=bool)
    clipped_samples = 0
    per_channel_runs: List[Tuple[np.ndarray, np.ndarray]] = []

    for magnitude in magnitudes:
        starts, lengths, in_run = _flat_runs(magnitude, ceiling, band)
        strict = magnitude >= strict_threshold
        clipped = in_run | strict

        clipped_samples += int(np.count_nonzero(clipped))
        clip_mask_any |= clipped
        per_channel_runs.append((starts, lengths))

        if lengths.size:
            longest_flat_run = max(longest_flat_run, int(lengths.max()))
            flat_run_count += int(np.count_nonzero(lengths > FLAT_RUN_REPORT_MIN))
        elif strict.any():
            s_starts, s_ends = _run_bounds(strict)
            longest_flat_run = max(longest_flat_run, int((s_ends - s_starts).max()))
            flat_run_count += int(np.count_nonzero((s_ends - s_starts) > FLAT_RUN_REPORT_MIN))

    total_samples = max(n * n_channels, 1)
    clip_percentage = _finite(100.0 * clipped_samples / total_samples, 0.0, lo=0.0, hi=100.0)

    tp_linear, inter_sample_overs = _oversample_scan(channels, over_threshold)
    # `peak_sample` is core's pre-resampling decode peak; rate conversion can
    # shave it, so the true peak is held at or above it rather than undercutting.
    sample_peak = _finite(amp_to_db(decoded_peak), DB_FLOOR, lo=DB_FLOOR)
    true_peak = max(
        _finite(amp_to_db(max(tp_linear, buffer_peak)), DB_FLOOR, lo=DB_FLOOR), sample_peak
    )

    header = dict(
        sample_peak_dbfs=round(sample_peak, 2),
        true_peak_dbtp=round(true_peak, 2),
        clipped_samples=clipped_samples,
        clip_percentage=round(clip_percentage, 4),
        longest_flat_run=longest_flat_run,
        flat_run_count=flat_run_count,
        inter_sample_overs=inter_sample_overs,
        is_float_over_unity=is_float_over_unity,
    )

    n_frames = n // hop
    if n_frames == 0 or clipped_samples == 0:
        return ClippingMeasurement(distortion_index=0.0, events=[], **header)

    # --- per-frame view: 0.25 s non-overlapping quanta ---
    usable = n_frames * hop
    times = frame_times(n_frames, hop, hop, sr)

    clipped_per_frame = frame_signal(
        np.ascontiguousarray(clip_mask_any[:usable], dtype=np.float64), hop, hop
    ).sum(axis=1)
    frame_clipped = clipped_per_frame > 0

    distortion_index = _distortion_index(buf, sr, hop, n_frames, usable, frame_clipped)
    events = _events(times, clipped_per_frame, hop, n_channels, per_channel_runs, usable)

    return ClippingMeasurement(
        distortion_index=round(distortion_index, 3), events=events, **header
    )


# ---------------------------------------------------------------------------
# Distortion index
# ---------------------------------------------------------------------------


def _distortion_index(
    buf: AudioBuffer,
    sr: int,
    hop: int,
    n_frames: int,
    usable: int,
    frame_clipped: np.ndarray,
) -> float:
    """0-1 evidence that the flat tops are actually generating harmonics.

    Clipping folds odd harmonics of everything into the top octave, so the
    >10 kHz share of a clipped frame's energy runs above that of a clean frame
    *from the same track*. Using the track as its own reference is what keeps
    this from just re-measuring how bright the mix is.

    A pervasively limited master has few clean frames left to compare against,
    so the score also carries how much of the track is pinned to the ceiling.
    """
    n_clipped = int(np.count_nonzero(frame_clipped))
    if n_clipped == 0:
        return 0.0

    mono = np.ascontiguousarray(buf.mono[:usable])
    cutoff = min(HF_SPLIT_HZ / (sr * 0.5), 0.98)
    high = sps.sosfilt(sps.butter(6, cutoff, btype="high", output="sos"), mono)

    total_energy = frame_signal(np.square(mono), hop, hop).mean(axis=1)
    high_energy = frame_signal(np.ascontiguousarray(np.square(high)), hop, hop).mean(axis=1)
    hf_ratio = high_energy / (total_energy + EPS)

    clean = ~frame_clipped
    excess = 0.0
    if int(np.count_nonzero(clean)) >= 3:
        reference = float(np.median(hf_ratio[clean]))
        hit = float(np.median(hf_ratio[frame_clipped]))
        if reference > EPS and hit > EPS:
            excess = clamp(
                10.0 * math.log10(hit / reference) / DISTORTION_FULL_SCALE_DB, 0.0, 1.0
            )

    density = clamp(n_clipped / max(n_frames, 1), 0.0, 1.0)

    # Weighted towards the spectral evidence; density carries the case where the
    # limiter has been on for the whole track and there is no clean reference.
    return _finite(0.65 * excess + 0.35 * density, 0.0, lo=0.0, hi=1.0)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def _events(
    times: np.ndarray,
    clipped_per_frame: np.ndarray,
    hop: int,
    n_channels: int,
    per_channel_runs: Sequence[Tuple[np.ndarray, np.ndarray]],
    usable: int,
) -> List[Moment]:
    """Merged, ranked clipping regions — a readable timeline, not a picket fence."""
    flagged = clipped_per_frame > 0
    if not np.any(flagged):
        return []

    spans = merge_spans(times, flagged, min_gap=0.75, min_len=0.25)
    if not spans:
        return []

    density = clipped_per_frame / float(hop * max(n_channels, 1))

    # Longest at-ceiling run starting in each frame, for the intensity weighting.
    longest_in_frame = np.zeros(times.size, dtype=np.float64)
    for starts, lengths in per_channel_runs:
        if starts.size == 0:
            continue
        inside = starts < usable
        if not np.any(inside):
            continue
        frame_idx = np.minimum(starts[inside] // hop, times.size - 1)
        np.maximum.at(longest_in_frame, frame_idx, lengths[inside].astype(np.float64))

    moments: List[Moment] = []
    for t0, t1 in spans:
        sel = flagged & (times >= t0 - 1e-6) & (times <= t1 + 1e-6)
        if not np.any(sel):
            continue
        worst_run = float(longest_in_frame[sel].max())
        worst_density = float(density[sel].max())
        # 16 consecutive samples, or 0.5 % of a window pinned, reads as full
        # intensity — both are unambiguous hard-clipping territory.
        intensity = clamp(max(worst_run / 16.0, worst_density / 0.005), 0.0, 1.0)
        moments.append(
            Moment(
                t_start=round(_finite(t0, 0.0, lo=0.0), 3),
                t_end=round(_finite(max(t1, t0 + 0.25), 0.0, lo=0.0), 3),
                intensity=round(_finite(intensity, 0.0, lo=0.0, hi=1.0), 3),
                value=round(worst_run, 1),
                label=f"{worst_run:.0f}-sample flat run, {worst_density * 100:.2f}% pinned",
            )
        )

    moments.sort(key=lambda m: (-m.intensity, m.t_start))
    moments = moments[:MAX_EVENTS]
    moments.sort(key=lambda m: m.t_start)
    return moments
