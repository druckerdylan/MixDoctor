"""Source separation: the point where inference becomes measurement.

Everything the 2-track analysis says about vocal balance, kick-versus-808 and
per-element compression is *inferred*. `vocal.py` is explicit about it — a
center estimate is not a vocal, and a centered synth, a snare and a mono bass
all live in the same place. `lowend.py` has to reconstruct the kick by
subtracting a between-hits spectrum from an at-hits spectrum, because the two
objects are summed into one waveform and cannot be looked at separately.

With stems they can. This module runs Demucs (htdemucs) and then measures each
source on its own, which turns four previously-guessed quantities into direct
readings:

    level_ratio_db            "are my vocals too quiet" — against the actual mix
    band_occupancy            who owns each band, measured, not proxied
    gain_reduction_estimate   per-element compression, impossible on a 2-track
    kick_to_bass_db           two objects compared, not one object split

What this module is careful about
---------------------------------
* **It never fails the analysis.** Every path returns a fully-populated
  `StemAnalysis`; a missing model, a failed download, a dead GPU or a blown
  time budget all come back as `available=False` plus a warning that says
  what happened. Nothing here raises.
* **It never invents a source.** An instrumental has no vocal. Demucs still
  emits a `vocals` array for one — full of bleed and artefacts at -40 dB —
  and `present=False` is how we say so instead of reporting a level for it.
* **It says what it actually analyzed.** Over ~6 minutes we separate the
  loudest contiguous 3 minutes rather than the whole file, and the warning
  names the span. Same when the time budget runs out mid-way: the excerpt is
  truncated to what genuinely completed and the warning says where.
* **The model is loaded once.** An 80 MB checkpoint behind a lazy singleton
  and a lock. Reloading it per request is the obvious way to make this
  unusable.

Cost, honestly: this is the one heavyweight stage in the analysis. It is
opt-in, it is bounded by `timeout_s`, and every caller can ignore it.
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal as sps

from ..core import (
    ANALYSIS_SR,
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
    merge_spans,
)
from ..types import STEM_KINDS, MaskingPair, Moment, StemAnalysis, StemMeasurement

__all__ = [
    "separate",
    "separate_detailed",
    "measure_stems",
    "SeparationResult",
    "MODEL_NAME",
    "SEPARATION_TIMEOUT_S",
]


# ---------------------------------------------------------------------------
# Separation configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "htdemucs"
DEMUCS_SR = 44_100                 # what the model was trained at; ours is 48k
SEPARATION_TIMEOUT_S = 300.0

# Over this length we separate an excerpt instead of the whole file. Demucs
# segments internally, but the total cost still scales linearly with duration
# and a ten-minute track is not four times more informative than a three-
# minute one.
MAX_FULL_SEC = 360.0
EXCERPT_SEC = 180.0

# Inference is chunked at this granularity so the deadline can actually be
# enforced: torch offers no way to interrupt a running graph, so the only
# honest timeout is a cooperative one checked between chunks. 45 s is far
# longer than htdemucs's own 7.8 s segment, so chunking costs no quality —
# apply_model is already doing overlap-add inside each chunk.
CHUNK_SEC = 45.0
CHUNK_FADE_SEC = 1.0

# Below this there is not enough material for a separated measurement to mean
# anything: one bar of audio gives a masking verdict built on three frames.
MIN_SEPARATION_SEC = 3.0

# A load failure (no network on first run, corrupt cache) is remembered for
# this long so a broken environment costs one attempt per two minutes rather
# than one 30-second timeout per request.
_LOAD_RETRY_S = 120.0


# ---------------------------------------------------------------------------
# Measurement configuration
# ---------------------------------------------------------------------------

_NFFT = 8192                # 171 ms / 5.9 Hz — matches the sibling dsp modules
_ENV_SR = 8_000.0           # envelope grid for onsets: 0.125 ms is plenty

# "Essentially silent" for a separated stem. Demucs leaks a little of
# everything into everything; an instrumental's vocals stem typically lands
# 35-55 dB under the mix and is entirely artefact.
_PRESENT_RATIO_DB = -30.0
_PRESENT_PEAK_DBFS = -55.0
_PRESENT_MIN_ACTIVE = 0.02

# A frame counts as "this stem is sounding" when it is within 35 dB of the
# stem's own loudest frame *and* not more than 60 dB under the mix — the
# second term is what stops a silent stem reporting active_ratio 1.0 on the
# shape of its own artefact floor.
_ACTIVE_BELOW_STEM_DB = 35.0
_ACTIVE_BELOW_MIX_DB = 60.0

# Masking gates.
_MASK_ONSET_DB = 3.0        # masker must clear the maskee by this much
_MASK_MIN_CO = 0.15         # ... for at least this share of shared frames
_MASK_BAND_FLOOR_DB = -20.0 # band must be within 20 dB of the stem's best band
_MASK_MAX_PER_PAIR = 2
_MASK_MAX_TOTAL = 8

# Punch: transient-to-sustain dB -> 0-1, r / (r + 9). Same mapping and same
# half-scale point as transients.py, so a per-stem punch figure is directly
# comparable with the broadband one.
_PUNCH_HALF_DB = 9.0
_ONSET_GATE_DB = 30.0
_MIN_ONSET_GAP_S = 0.050

# Micro-dynamics / gain-reduction window. 50 ms sits inside a single drum hit.
_MICRO_WIN_SEC = 0.050
_MICRO_HOP_SEC = 0.025
_ACTIVE_RANGE_DB = 30.0
_SILENCE_FLOOR_DB = -80.0

_SERIES_FLOOR_DB = -100.0

# Kick / bass.
_KICK_SEARCH = (35.0, 130.0)
_BASS_SEARCH = (25.0, 200.0)
_LOW_BAND = (20.0, 250.0)
_KICK_WIN_SEC = 0.080       # the window a kick's level is measured over
_MIN_KICK_HITS = 4
_MIN_KICK_DEFINITION_DB = 3.0   # a hit must punch this far over the low-end floor
_MIN_KICK_REGULARITY = 0.40     # share of gaps landing on the dominant spacing
_MIN_BASS_PEAK_DB = 6.0         # bass peak must stand this far over the band's median

# BS.1770 gating constants, for per-stem integrated loudness.
_LUFS_OFFSET = -0.691
_ABS_GATE_LUFS = -70.0
_REL_GATE_LU = -10.0


# ---------------------------------------------------------------------------
# Module state: one model, one lock
# ---------------------------------------------------------------------------

_MODEL_LOCK = threading.Lock()   # guards the load
_APPLY_LOCK = threading.Lock()   # serialises inference on the single instance

_MODEL = None
_DEVICE = ""
_LOAD_ERROR = ""
_LOAD_ERROR_AT = 0.0
_MPS_BLOCKED = False             # set once MPS has proved it can't run this


# ---------------------------------------------------------------------------
# small private helpers (duplicated rather than shared: sibling dsp modules are
# being written concurrently and this package has no agreed-on internal utils)
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


def _db(x, floor: float = -120.0):
    """Power ratio -> dB, floored."""
    return np.maximum(10.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float64), EPS)), floor)


def _ratio_db(num: float, den: float, floor: float = -80.0, ceil: float = 40.0) -> float:
    if den <= EPS:
        return floor
    return _f(10.0 * np.log10(max(num, EPS) / den), floor, floor, ceil)


def _power_spectrogram(
    x: np.ndarray, sr: int, n_fft: int, hop: int, block: int = 128
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parseval-calibrated power spectrogram: (freqs, times, power).

    Scaled so that summing a frame's bins reproduces that frame's windowed
    mean square. Sibling modules normalize by `sum(window)` instead, which is
    right for peak-of-a-sinusoid work but leaves band levels off by a constant
    — and here `band_levels_db` is supposed to be a real dBFS figure that a
    producer can compare against a meter.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    frames = frame_signal(x, n_fft, hop)
    n = int(frames.shape[0])
    window = np.hanning(n_fft)
    # sum_k |X_k|^2 = N * sum_n (x*w)^2 (Parseval); the one-sided rfft holds
    # half of it, and sum_n (x*w)^2 ~= mean_square(x) * sum(w^2).
    scale = 2.0 / (float(n_fft) * float(np.sum(window * window)) + EPS)

    out = np.empty((n, n_fft // 2 + 1), dtype=np.float64)
    for i in range(0, n, block):  # blocks of frames, never samples
        seg = np.asarray(frames[i : i + block]) * window
        spec = np.fft.rfft(seg, n=n_fft, axis=1)
        out[i : i + block] = (np.abs(spec) ** 2) * scale

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    times = (np.arange(n) * hop + n_fft * 0.5) / sr
    return freqs, times, out


def _macro_band_matrix(freqs: np.ndarray) -> np.ndarray:
    """(n_bands, n_bins) 0/1 map of FFT bins onto MACRO_BANDS."""
    lo = np.asarray([MACRO_BANDS[b][0] for b in MACRO_BAND_ORDER])[:, None]
    hi = np.asarray([MACRO_BANDS[b][1] for b in MACRO_BAND_ORDER])[:, None]
    return ((freqs[None, :] >= lo) & (freqs[None, :] < hi)).astype(np.float64)


def _welch(x: np.ndarray, sr: int, nperseg: int = 16384) -> Tuple[np.ndarray, np.ndarray]:
    """Blackman-Harris Welch spectrum, for fundamental-frequency work."""
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


def _peak_freq(spec: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> Tuple[float, float]:
    """Parabolically interpolated spectral peak in [lo, hi]. Returns (hz, power)."""
    m = (freqs >= lo) & (freqs <= hi)
    if not np.any(m) or spec.size != freqs.size:
        return 0.0, 0.0
    idx = np.flatnonzero(m)
    k = int(idx[int(np.argmax(spec[idx]))])
    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 0.0
    delta = 0.0
    if 0 < k < spec.size - 1:
        a, b, c = float(spec[k - 1]), float(spec[k]), float(spec[k + 1])
        denom = a - 2.0 * b + c
        if abs(denom) > EPS:
            delta = clamp(0.5 * (a - c) / denom, -0.5, 0.5)
    nyquist = float(freqs[-1]) if freqs.size else 0.0
    return _f(freqs[k] + delta * df, 0.0, 0.0, nyquist), float(spec[k])


def _prefer_subharmonic(
    spec: np.ndarray, freqs: np.ndarray, f0: float, p0: float, lo: float
) -> float:
    """A bass note's 2nd harmonic often beats its fundamental; check half."""
    half = f0 * 0.5
    if f0 <= 0.0 or half < lo:
        return f0
    m = (freqs >= half * 0.94) & (freqs <= half * 1.06)
    if not np.any(m):
        return f0
    if float(spec[m].max()) >= 0.30 * max(p0, EPS):
        return _peak_freq(spec, freqs, half * 0.94, half * 1.06)[0] or f0
    return f0


def _mono(stereo: np.ndarray) -> np.ndarray:
    """(2, n) -> mid channel."""
    return np.ascontiguousarray(0.5 * (stereo[0] + stereo[1]), dtype=np.float64)


def _mmss(seconds: float) -> str:
    s = max(int(round(seconds)), 0)
    return f"{s // 60}:{s % 60:02d}"


# ---------------------------------------------------------------------------
# Per-stem integrated loudness (BS.1770-4, two-stage gate)
# ---------------------------------------------------------------------------


def _gated_integrated(block_ms: np.ndarray) -> float:
    if block_ms.size == 0:
        return _ABS_GATE_LUFS
    loud = _LUFS_OFFSET + 10.0 * np.log10(np.maximum(block_ms, EPS))
    above = loud > _ABS_GATE_LUFS
    if not np.any(above):
        return _ABS_GATE_LUFS
    rel = _LUFS_OFFSET + 10.0 * math.log10(max(float(block_ms[above].mean()), EPS)) + _REL_GATE_LU
    keep = above & (loud > rel)
    if not np.any(keep):
        keep = above
    return float(_LUFS_OFFSET + 10.0 * math.log10(max(float(block_ms[keep].mean()), EPS)))


def _integrated_lufs(left: np.ndarray, right: np.ndarray, sr: int) -> float:
    """Gated integrated loudness of one stem.

    Gating is the right behavior here and not an accident: a vocal that sings
    on a third of the track should report the level it sings *at*, not that
    level averaged with two thirds of silence. That is the number a producer
    means by "how loud is the vocal".
    """
    from .loudness import k_weight  # public in loudness.__all__; no cycle

    if left.size == 0:
        return _ABS_GATE_LUFS
    power = np.square(k_weight(left, sr)) + np.square(k_weight(right, sr))
    block = int(round(0.400 * sr))
    hop = max(1, int(round(0.100 * sr)))
    if power.shape[0] >= block:
        block_ms = frame_signal(np.ascontiguousarray(power), block, hop).mean(axis=1)
    else:
        block_ms = np.array([float(power.mean())], dtype=np.float64)
    return _f(_gated_integrated(block_ms), _ABS_GATE_LUFS, _ABS_GATE_LUFS, 10.0)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _pick_device(torch) -> str:
    if not _MPS_BLOCKED:
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
    return "cpu"


def _load_model() -> Tuple[object, str, str]:
    """Lazy singleton. Returns (model, device, error). Never raises.

    The first call downloads ~80 MB of weights into the torch hub cache. A
    download failure comes back as an error string, not an exception, and is
    remembered for `_LOAD_RETRY_S` so a machine with no network does not pay
    the connect timeout on every single request.
    """
    global _MODEL, _DEVICE, _LOAD_ERROR, _LOAD_ERROR_AT

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL, _DEVICE, ""
        if _LOAD_ERROR and (time.monotonic() - _LOAD_ERROR_AT) < _LOAD_RETRY_S:
            return None, "", _LOAD_ERROR

        try:
            import torch
            from demucs.pretrained import get_model

            # Leave a couple of cores for the rest of the request. torch's
            # default (every core) oversubscribes badly once FastAPI is
            # serving more than one analysis at a time.
            try:
                torch.set_num_threads(max(1, min(8, (os.cpu_count() or 4) - 2)))
            except Exception:
                pass

            model = get_model(MODEL_NAME)
            model.eval()
            device = _pick_device(torch)
            model.to(device)

            _MODEL, _DEVICE, _LOAD_ERROR = model, device, ""
            return _MODEL, _DEVICE, ""
        except Exception as exc:  # download failure, corrupt cache, no torch
            _LOAD_ERROR = f"{type(exc).__name__}: {exc}".strip()
            _LOAD_ERROR_AT = time.monotonic()
            return None, "", _LOAD_ERROR


def _to_cpu_forever(torch) -> str:
    """MPS refused the graph. Move the singleton to CPU and stop trying."""
    global _DEVICE, _MPS_BLOCKED
    _MPS_BLOCKED = True
    with _MODEL_LOCK:
        if _MODEL is not None:
            try:
                _MODEL.to("cpu")
            except Exception:
                pass
        _DEVICE = "cpu"
    return "cpu"


# ---------------------------------------------------------------------------
# Excerpt selection
# ---------------------------------------------------------------------------


def _loudest_window(buf: AudioBuffer, span_sec: float) -> Tuple[int, int]:
    """Start/end sample of the loudest contiguous `span_sec` by short-term loudness.

    One-second K-weighted blocks and a sliding sum via cumsum — no loop, and
    the result is the window a mastering engineer would have picked to audition.
    """
    sr = int(buf.sr)
    n = int(buf.n_samples)
    span = int(span_sec * sr)
    if n <= span:
        return 0, n

    try:
        from .loudness import k_weight

        y = k_weight(buf.mid, sr)
    except Exception:
        y = np.asarray(buf.mid, dtype=np.float64)

    block = sr
    usable = (y.shape[0] // block) * block
    if usable < block * 2:
        return 0, span
    energy = np.square(y[:usable]).reshape(-1, block).mean(axis=1)

    width = max(1, span // block)
    if energy.size <= width:
        return 0, span
    cumulative = np.concatenate([[0.0], np.cumsum(energy)])
    sums = cumulative[width:] - cumulative[:-width]
    start = int(np.argmax(sums)) * block
    start = int(min(max(start, 0), n - span))
    return start, start + span


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _chunk_weight(length: int, fade: int, first: bool, last: bool) -> np.ndarray:
    """Crossfade weight for one chunk. Overlaps are normalized by the sum, so
    the ramps only have to be complementary in shape, not exact."""
    w = np.ones(length, dtype=np.float32)
    fade = int(min(fade, length // 2))
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
        if not first:
            w[:fade] = ramp
        if not last:
            w[length - fade :] = ramp[::-1]
    return w


def _apply_chunked(
    model, device: str, mix: np.ndarray, deadline: float
) -> Tuple[Optional[np.ndarray], int, bool, str]:
    """Run the model over `mix` (2, n) float32 @ DEMUCS_SR in bounded chunks.

    Returns (sources[4, 2, m] float32, samples_done, timed_out, device_used).
    `samples_done` is how much of the input genuinely completed — the caller
    truncates the excerpt to that, so a blown deadline degrades to "we analyzed
    the first two minutes" instead of "we analyzed nothing".
    """
    import torch
    from demucs.apply import apply_model

    n = int(mix.shape[1])
    chunk = int(CHUNK_SEC * DEMUCS_SR)
    fade = int(CHUNK_FADE_SEC * DEMUCS_SR)
    stride = max(chunk - fade, chunk // 2, 1)

    # Normalize once over the whole excerpt, not per chunk: demucs's own CLI
    # normalizes before calling apply_model, and doing it per chunk would make
    # a quiet chunk come back at the same level as a loud one.
    ref = mix.mean(axis=0)
    mean = float(ref.mean())
    std = float(ref.std())
    if not math.isfinite(std) or std < 1e-8:
        std = 1.0
    norm = ((mix - mean) / std).astype(np.float32, copy=False)

    n_sources = len(getattr(model, "sources", STEM_KINDS))
    out = np.zeros((n_sources, 2, n), dtype=np.float32)
    wsum = np.zeros(n, dtype=np.float32)

    done = 0
    timed_out = False
    offset = 0
    while offset < n:
        # Cooperative, because it is the only kind available: torch offers no
        # way to interrupt a running graph, so the deadline is checked between
        # chunks. Checked before the first one too — a budget already spent by
        # the time we get here means we should not start at all.
        if time.monotonic() > deadline:
            timed_out = True
            break
        end = min(offset + chunk, n)
        seg = np.ascontiguousarray(norm[:, offset:end])

        try:
            with torch.inference_mode():
                tensor = torch.from_numpy(seg)[None].to(device)
                est = apply_model(
                    model,
                    tensor,
                    shifts=0,          # deterministic: the shift trick is random
                    split=True,
                    overlap=0.25,
                    progress=False,
                    device=device,
                )
            block = est[0].to("cpu").numpy()
        except Exception:
            if device == "mps":
                # MPS could not run the graph. Fall back once, permanently,
                # and redo this chunk on the CPU.
                device = _to_cpu_forever(torch)
                try:
                    model.to(device)
                    with torch.inference_mode():
                        tensor = torch.from_numpy(seg)[None].to(device)
                        est = apply_model(
                            model, tensor, shifts=0, split=True,
                            overlap=0.25, progress=False, device=device,
                        )
                    block = est[0].to("cpu").numpy()
                except Exception:
                    return None, 0, False, device
            else:
                return None, 0, False, device

        w = _chunk_weight(end - offset, fade, first=(offset == 0), last=(end >= n))
        out[:, :, offset:end] += block[:, :, : end - offset] * w
        wsum[offset:end] += w
        done = end
        if end >= n:
            break
        offset += stride

    if done <= 0:
        return None, 0, timed_out, device

    out = out[:, :, :done]
    out /= np.maximum(wsum[:done], 1e-6)
    # Denormalize. The DC term is deliberately not added back: adding the
    # mixture's mean to all four sources would quadruple any DC offset, and a
    # stem has no business carrying one.
    out *= std
    return out, done, timed_out, device


# ---------------------------------------------------------------------------
# Public separation API
# ---------------------------------------------------------------------------


@dataclass
class SeparationResult:
    """Stems plus the context needed to interpret them.

    `t_start`/`t_end` matter: on a long track, or one that ran out of time,
    the arrays do not cover the whole file and every timestamp derived from
    them has to be offset. `mix` is the exact excerpt that was separated, so
    ratios against it are internally consistent by construction.
    """

    stems: Dict[str, np.ndarray]      # kind -> (2, n) float64 @ ANALYSIS_SR
    mix: np.ndarray                   # (2, n) float64 @ ANALYSIS_SR
    t_start: float
    t_end: float
    model_name: str
    device: str
    separation_ms: int
    warnings: List[str] = field(default_factory=list)


def _resample(x: np.ndarray, up: int, down: int) -> np.ndarray:
    if up == down:
        return np.ascontiguousarray(x, dtype=np.float64)
    return sps.resample_poly(np.asarray(x, dtype=np.float64), up, down, axis=-1)


def _separate_impl(
    buf: AudioBuffer, timeout_s: float
) -> Tuple[Optional[SeparationResult], List[str]]:
    """The real separation path. Returns (result | None, warnings). Never raises."""
    warnings: List[str] = []
    started = time.monotonic()
    deadline = started + _f(timeout_s, SEPARATION_TIMEOUT_S, 0.0, 86_400.0)

    if buf.duration < MIN_SEPARATION_SEC:
        warnings.append(
            f"Stem separation skipped: {buf.duration:.1f}s of audio is below the "
            f"{MIN_SEPARATION_SEC:.0f}s minimum for a separated measurement to mean anything."
        )
        return None, warnings

    model, device, error = _load_model()
    if model is None:
        warnings.append(
            "Stem separation unavailable: the Demucs model could not be loaded "
            f"({error or 'unknown error'}). The mix was analyzed without stems."
        )
        return None, warnings

    # --- excerpt -----------------------------------------------------------
    s0, s1 = 0, int(buf.n_samples)
    excerpt_note = ""
    if buf.duration > MAX_FULL_SEC:
        s0, s1 = _loudest_window(buf, EXCERPT_SEC)
        # Held back until inference actually succeeds: "we analyzed the loudest
        # three minutes" is a lie when we analyzed nothing.
        excerpt_note = (
            f"Track runs {_mmss(buf.duration)}; separated the loudest "
            f"{_mmss(EXCERPT_SEC)} only ({_mmss(s0 / buf.sr)}-{_mmss(s1 / buf.sr)}). "
            "Every per-stem number below describes that excerpt, not the whole file."
        )

    left = np.ascontiguousarray(buf.left[s0:s1], dtype=np.float64)
    right = np.ascontiguousarray(buf.right[s0:s1], dtype=np.float64)

    # --- 48k -> 44.1k ------------------------------------------------------
    g = math.gcd(ANALYSIS_SR, DEMUCS_SR)
    up_in, down_in = DEMUCS_SR // g, ANALYSIS_SR // g          # 147 / 160
    try:
        mix44 = np.stack([_resample(left, up_in, down_in), _resample(right, up_in, down_in)])
    except Exception as exc:
        warnings.append(f"Stem separation failed while resampling ({type(exc).__name__}).")
        return None, warnings
    mix44 = np.ascontiguousarray(mix44, dtype=np.float32)

    # --- inference ---------------------------------------------------------
    with _APPLY_LOCK:
        try:
            sources, done44, timed_out, device_used = _apply_chunked(
                model, device, mix44, deadline
            )
        except Exception as exc:  # belt and braces: nothing escapes this module
            warnings.append(
                f"Stem separation failed during inference ({type(exc).__name__}: {exc})."
            )
            return None, warnings

    if sources is None:
        if timed_out:
            warnings.append(
                f"Stem separation was given {timeout_s:.3g}s, which was already gone before "
                "the first chunk could start. Nothing was separated."
            )
        else:
            warnings.append(
                "Stem separation produced no output; the mix was analyzed without stems."
            )
        return None, warnings

    if excerpt_note:
        warnings.append(excerpt_note)

    if timed_out:
        got = done44 / DEMUCS_SR
        warnings.append(
            f"Stem separation hit its {timeout_s:.0f}s budget after "
            f"{_mmss(got)} of audio. Per-stem numbers cover "
            f"{_mmss(s0 / buf.sr)}-{_mmss(s0 / buf.sr + got)} only."
        )

    if device_used != device:
        warnings.append(
            "Stem separation fell back to the CPU: this machine's GPU backend "
            "could not run the model."
        )

    # --- 44.1k -> 48k ------------------------------------------------------
    names: Sequence[str] = list(getattr(model, "sources", STEM_KINDS))
    n_out = int(round(done44 * ANALYSIS_SR / DEMUCS_SR))
    n_out = int(min(n_out, left.shape[0]))

    stems: Dict[str, np.ndarray] = {}
    try:
        for i, name in enumerate(names):
            back = _resample(sources[i].astype(np.float64, copy=False), down_in, up_in)
            if back.shape[1] < n_out:
                pad = np.zeros((2, n_out - back.shape[1]), dtype=np.float64)
                back = np.concatenate([back, pad], axis=1)
            stems[str(name)] = np.ascontiguousarray(back[:, :n_out])
    except Exception as exc:
        warnings.append(f"Stem separation failed while resampling back ({type(exc).__name__}).")
        return None, warnings

    mix_ref = np.ascontiguousarray(np.stack([left[:n_out], right[:n_out]]))
    elapsed_ms = int(round((time.monotonic() - started) * 1000.0))

    return (
        SeparationResult(
            stems=stems,
            mix=mix_ref,
            t_start=float(s0) / buf.sr,
            t_end=float(s0) / buf.sr + n_out / float(ANALYSIS_SR),
            model_name=MODEL_NAME,
            device=device_used,
            separation_ms=elapsed_ms,
            warnings=list(warnings),
        ),
        warnings,
    )


def separate(buf: AudioBuffer, *, timeout_s: float = SEPARATION_TIMEOUT_S):
    """Separate `buf` into `{'drums','bass','other','vocals'}`. None on failure.

    Each value is a `(2, n)` float64 array at `core.ANALYSIS_SR`, channels
    first. On a track over ~6 minutes, or one that exhausts `timeout_s`, the
    arrays cover an excerpt rather than the whole file — compare `n` against
    `buf.n_samples` to detect that, or call `separate_detailed` for the exact
    span and the warnings that go with it.

    Never raises: a missing model, a failed download or a dead GPU all return
    None.
    """
    result, _ = _separate_impl(buf, timeout_s)
    return result.stems if result is not None else None


def separate_detailed(
    buf: AudioBuffer, *, timeout_s: float = SEPARATION_TIMEOUT_S
) -> Optional[SeparationResult]:
    """`separate`, plus the excerpt bounds, device, timing and warnings."""
    result, _ = _separate_impl(buf, timeout_s)
    return result


# ---------------------------------------------------------------------------
# Per-stem measurement
# ---------------------------------------------------------------------------


def _frame_peak_rms(
    left: np.ndarray, right: np.ndarray, win: int, hop: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Per-frame (peak, rms) across both channels. No (n_frames, win) copy."""
    n = max(left.shape[0], 1)
    win = int(max(1, min(win, n)))
    hop = int(max(1, hop))
    fl = frame_signal(left, win, hop)
    fr = frame_signal(right, win, hop)
    peak = np.maximum(
        np.maximum(np.abs(fl.max(axis=1)), np.abs(fl.min(axis=1))),
        np.maximum(np.abs(fr.max(axis=1)), np.abs(fr.min(axis=1))),
    )
    sumsq = np.einsum("ij,ij->i", fl, fl) + np.einsum("ij,ij->i", fr, fr)
    rms = np.sqrt(np.maximum(sumsq / (2.0 * win), 0.0))
    return peak, rms, int(fl.shape[0])


def _gain_reduction(peak_db: np.ndarray, rms_db: np.ndarray, active: np.ndarray) -> float:
    """Per-element compression estimate — the thing a 2-track cannot do at all.

    Same two symptoms `dynamics.py` looks for on the master, applied to one
    source: *crest collapse* (the loud frames have less crest than the medium
    ones, because the compressor only works when the signal is loud) and *peak
    pinning* (level keeps moving while peaks stop moving). On a stem this
    finally distinguishes "the vocal is crushed" from "the master is crushed".
    """
    if not np.any(active):
        return 0.0
    crest = peak_db[active] - rms_db[active]
    level = rms_db[active]
    peaks = peak_db[active]
    if crest.size < 8:
        return 0.0

    loud = level >= np.percentile(level, 80.0)
    mids = (level >= np.percentile(level, 35.0)) & (level <= np.percentile(level, 70.0))
    collapse = 0.0
    if np.any(loud) and np.any(mids):
        collapse = max(0.0, float(np.mean(crest[mids])) - float(np.mean(crest[loud])))

    level_spread = float(np.percentile(level, 98.0) - np.percentile(level, 50.0))
    peak_spread = float(np.percentile(peaks, 98.0) - np.percentile(peaks, 50.0))
    pinning = max(0.0, level_spread - peak_spread)

    return _f(0.6 * collapse + 0.4 * pinning, 0.0, 0.0, 24.0)


def _onsets_and_punch(mono: np.ndarray, sr: int) -> Tuple[int, float]:
    """(onset count, punch 0-1) for one stem.

    Punch is the level at the transient against the level 50 ms later, which
    is what the ear reads as impact — a loud kick that hangs around hits less
    hard than a quieter one that gets out of the way.
    """
    if mono.shape[0] < int(0.25 * sr):
        return 0, 0.0

    env = envelope(mono, sr, smooth_ms=3.0)
    factor = max(1, int(round(sr / _ENV_SR)))
    usable = (env.shape[0] // factor) * factor
    if usable < factor * 16:
        return 0, 0.0
    e = env[:usable].reshape(-1, factor).max(axis=1)
    env_sr = sr / float(factor)
    if float(e.max()) <= 1e-7:
        return 0, 0.0

    e_db = 20.0 * np.log10(np.maximum(e, 1e-9))
    e_db = np.maximum(e_db, float(e_db.max()) - 60.0)
    flux = np.maximum(np.diff(e_db, prepend=e_db[:1]), 0.0)
    positive = flux[flux > 0]
    if positive.size < 4:
        return 0, 0.0
    prom = max(1.0, 0.30 * float(np.percentile(positive, 90.0)))

    peaks, _ = sps.find_peaks(
        flux, distance=max(1, int(_MIN_ONSET_GAP_S * env_sr)), prominence=prom
    )
    if peaks.size == 0:
        return 0, 0.0

    search = max(1, int(0.060 * env_sr))
    idx = np.clip(peaks[:, None] + np.arange(search)[None, :], 0, e.size - 1)
    local = e[idx]
    peak_pos = peaks + np.argmax(local, axis=1)
    peak_val = local.max(axis=1)

    # Drop "onsets" that are 30 dB under the loudest one: those are the
    # separator's artefact floor, not hits.
    keep = peak_val > float(peak_val.max()) * (10.0 ** (-_ONSET_GATE_DB / 20.0))
    peak_pos, peak_val = peak_pos[keep], peak_val[keep]
    if peak_pos.size == 0:
        return 0, 0.0

    sus_lo = int(0.050 * env_sr)
    sus_hi = max(sus_lo + 1, int(0.053 * env_sr))
    sus_idx = np.clip(
        peak_pos[:, None] + np.arange(sus_lo, sus_hi)[None, :], 0, e.size - 1
    )
    sustain = e[sus_idx].mean(axis=1)

    usable_hit = (peak_pos + sus_hi) < e.size
    if not np.any(usable_hit):
        return int(peak_pos.size), 0.0
    ratio_db = 20.0 * np.log10(
        np.maximum(peak_val[usable_hit], EPS) / np.maximum(sustain[usable_hit], EPS)
    )
    ratio_db = ratio_db[np.isfinite(ratio_db)]
    if ratio_db.size == 0:
        return int(peak_pos.size), 0.0
    r = max(0.0, float(np.median(ratio_db)))
    return int(peak_pos.size), _f(r / (r + _PUNCH_HALF_DB), 0.0, 0.0, 1.0)


def _measure_one(
    kind: str,
    x: np.ndarray,
    sr: int,
    mix_lufs: float,
    mix_frame_peak_db: float,
    band_mean: np.ndarray,
    centroid_hz: float,
) -> StemMeasurement:
    """Measure one separated source on its own.

    `band_mean` is this stem's mean per-macro-band power (from the shared
    spectrogram pass); `band_occupancy` is filled in by the caller, which is
    the only place that can see all four stems at once.
    """
    left = np.ascontiguousarray(x[0], dtype=np.float64)
    right = np.ascontiguousarray(x[1], dtype=np.float64)
    n = int(left.shape[0])

    win = max(1, min(int(FRAME_SEC * sr), max(n, 1)))
    hop = max(1, int(HOP_SEC * sr))
    _, f_rms, _ = _frame_peak_rms(left, right, win, hop)
    f_rms_db = np.asarray(amp_to_db(f_rms), dtype=np.float64)

    stem_frame_peak_db = float(f_rms_db.max()) if f_rms_db.size else -120.0
    floor_db = max(
        stem_frame_peak_db - _ACTIVE_BELOW_STEM_DB,
        mix_frame_peak_db - _ACTIVE_BELOW_MIX_DB,
    )
    active = f_rms_db > floor_db
    active_ratio = float(np.mean(active)) if active.size else 0.0

    # --- levels ------------------------------------------------------------
    level_lufs = _integrated_lufs(left, right, sr)
    level_ratio_db = _f(level_lufs - mix_lufs, -80.0, -80.0, 40.0)

    peak_linear = float(max(np.abs(left).max() if n else 0.0, np.abs(right).max() if n else 0.0))
    peak_dbfs = _f(amp_to_db(peak_linear), -120.0, -120.0, 24.0)

    # Crest is taken over the frames the stem is actually sounding on. A stem
    # that plays for a third of the track would otherwise show 30 dB of crest
    # purely because two thirds of it is silence.
    if np.any(active):
        rms_active = float(np.sqrt(np.mean(np.square(f_rms[active]))))
    else:
        rms_active = float(np.sqrt(np.mean(np.square(f_rms)))) if f_rms.size else 0.0
    crest_factor_db = _f(peak_dbfs - _f(amp_to_db(rms_active), -120.0), 0.0, 0.0, 60.0)

    # --- micro dynamics + per-element compression ---------------------------
    m_win = max(8, min(int(_MICRO_WIN_SEC * sr), max(n, 8)))
    m_hop = max(1, int(_MICRO_HOP_SEC * sr))
    m_peak, m_rms, _ = _frame_peak_rms(left, right, m_win, m_hop)
    m_peak_db = np.asarray(amp_to_db(m_peak), dtype=np.float64)
    m_rms_db = np.asarray(amp_to_db(m_rms), dtype=np.float64)
    if m_rms_db.size:
        ceiling = float(m_rms_db.max())
        m_active = (m_rms_db > ceiling - _ACTIVE_RANGE_DB) & (m_rms_db > _SILENCE_FLOOR_DB)
        if not np.any(m_active):
            m_active = m_rms_db >= ceiling
    else:
        m_active = np.zeros(0, dtype=bool)

    micro = (m_peak_db - m_rms_db)[m_active] if np.any(m_active) else np.zeros(0)
    micro_dynamics_db = _f(float(np.mean(micro)) if micro.size else 0.0, 0.0, 0.0, 60.0)
    gain_reduction_estimate_db = _gain_reduction(m_peak_db, m_rms_db, m_active)

    # --- stereo -------------------------------------------------------------
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    mid_rms = float(np.sqrt(np.mean(np.square(mid)))) if n else 0.0
    side_rms = float(np.sqrt(np.mean(np.square(side)))) if n else 0.0
    stereo_width = _f(side_rms / (mid_rms + EPS), 0.0, 0.0, 8.0)

    correlation = 1.0
    if n > 1:
        sl, sr_ = float(np.std(left)), float(np.std(right))
        if sl > 1e-9 and sr_ > 1e-9:
            correlation = _f(
                float(np.mean((left - left.mean()) * (right - right.mean())) / (sl * sr_)),
                1.0, -1.0, 1.0,
            )

    # --- spectral -----------------------------------------------------------
    band_levels_db: Dict[str, float] = {}
    for j, name in enumerate(MACRO_BAND_ORDER):
        band_levels_db[name] = round(_f(float(_db(band_mean[j])), -120.0, -120.0, 24.0), 2)
    total_band = float(band_mean.sum())

    # --- transients ---------------------------------------------------------
    onset_count, transient_punch = _onsets_and_punch(mid, sr)

    # --- series -------------------------------------------------------------
    level_series = np.maximum(np.nan_to_num(f_rms_db, nan=_SERIES_FLOOR_DB,
                                            posinf=0.0, neginf=_SERIES_FLOOR_DB),
                              _SERIES_FLOOR_DB)

    present = bool(
        level_ratio_db > _PRESENT_RATIO_DB
        and peak_dbfs > _PRESENT_PEAK_DBFS
        and active_ratio >= _PRESENT_MIN_ACTIVE
        and total_band > EPS
    )

    measurement = StemMeasurement(
        kind=kind,  # type: ignore[arg-type]
        present=present,
        level_lufs=round(level_lufs, 2),
        level_ratio_db=round(level_ratio_db, 2),
        peak_dbfs=round(peak_dbfs, 2),
        crest_factor_db=round(crest_factor_db, 2),
        micro_dynamics_db=round(micro_dynamics_db, 2),
        gain_reduction_estimate_db=round(gain_reduction_estimate_db, 2),
        spectral_centroid_hz=round(_f(centroid_hz, 0.0, 0.0, 24_000.0), 1),
        band_levels_db=band_levels_db,
        band_occupancy={},
        stereo_width=round(stereo_width, 4),
        correlation=round(correlation, 4),
        transient_punch=round(transient_punch, 3),
        onset_count=int(onset_count),
        active_ratio=round(_f(active_ratio, 0.0, 0.0, 1.0), 4),
        level_series=[round(float(v), 2) for v in level_series],
    )
    return measurement


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


def _severity(overlap_db: float, co_ratio: float) -> str:
    if overlap_db >= 12.0 and co_ratio >= 0.50:
        return "critical"
    if overlap_db >= 8.0 and co_ratio >= 0.35:
        return "major"
    return "minor"


def _masking_pairs(
    present: Sequence[str],
    band_db: Dict[str, np.ndarray],
    band_active: Dict[str, np.ndarray],
    band_rel_db: Dict[str, np.ndarray],
    times: np.ndarray,
    t_offset: float,
    t_limit: float,
) -> List[MaskingPair]:
    """Which source is burying which, in which band, and when.

    Simultaneous masking, measured within the critical band rather than
    modeled with a spreading function: for an ordered pair we ask how far the
    maskee sits under the masker *in a band the maskee actually needs*, and for
    what share of the frames where both are sounding. Both conditions matter —
    a masker 15 dB up in a band the maskee barely occupies is not masking
    anything, and 15 dB up for 4 % of the track is not either.

    Deliberately conservative: only downward-in-level, within-band overlap is
    reported. Upward spread of masking is real, but attributing it needs an
    excitation-pattern model this module does not have, and an over-claimed
    masking pair sends a producer carving a band that was never the problem.
    """
    out: List[Tuple[float, MaskingPair]] = []
    n_frames = int(times.shape[0])
    if n_frames < 4:
        return []

    for masker in present:
        for maskee in present:
            if masker == maskee:
                continue
            co = band_active[masker] & band_active[maskee]
            n_co = int(np.count_nonzero(co))
            if n_co < max(4, int(0.05 * n_frames)):
                continue

            per_pair: List[Tuple[float, MaskingPair]] = []
            for j, name in enumerate(MACRO_BAND_ORDER):
                if band_rel_db[maskee][j] < _MASK_BAND_FLOOR_DB:
                    continue  # the maskee does not live here
                if band_rel_db[masker][j] < _MASK_BAND_FLOOR_DB:
                    continue  # nor does the masker

                delta = band_db[masker][:, j] - band_db[maskee][:, j]
                shared = delta[co]
                overlap_db = float(np.median(shared))
                if overlap_db < _MASK_ONSET_DB:
                    continue
                hit = co & (delta >= _MASK_ONSET_DB)
                co_ratio = float(np.count_nonzero(hit)) / float(n_co)
                if co_ratio < _MASK_MIN_CO:
                    continue

                lo, hi = MACRO_BANDS[name]
                # Where it is worst: frames in the top quartile of the overlap.
                worst_gate = max(_MASK_ONSET_DB, float(np.percentile(shared, 75.0)))
                mask = co & (delta >= worst_gate)
                moments: List[Moment] = []
                for t0, t1 in merge_spans(times, mask, min_gap=0.75, min_len=0.5)[:4]:
                    sel = (times >= t0) & (times <= t1)
                    worst = float(delta[sel].max()) if np.any(sel) else overlap_db
                    moments.append(
                        Moment(
                            t_start=round(clamp(t_offset + t0, 0.0, t_limit), 3),
                            t_end=round(clamp(t_offset + t1, 0.0, t_limit), 3),
                            intensity=round(clamp(worst / 24.0, 0.0, 1.0), 3),
                            value=round(_f(worst, 0.0, 0.0, 60.0), 2),
                            label=f"{masker} over {maskee}",
                        )
                    )

                label = name.replace("_", " ")
                detail = (
                    f"{masker.capitalize()} sits {overlap_db:.1f} dB over {maskee} through the "
                    f"{label} band ({lo:.0f}-{hi:.0f} Hz) for {co_ratio * 100:.0f}% of the "
                    f"frames where both are playing."
                )
                pair = MaskingPair(
                    masker=masker,  # type: ignore[arg-type]
                    maskee=maskee,  # type: ignore[arg-type]
                    band=name,
                    low_hz=float(lo),
                    high_hz=float(hi),
                    overlap_db=round(_f(overlap_db, 0.0, 0.0, 60.0), 2),
                    severity=_severity(overlap_db, co_ratio),  # type: ignore[arg-type]
                    detail=detail,
                    moments=moments,
                )
                # Rank by how much the burial *costs the maskee*: the same
                # 20 dB matters far more in the band that source lives in than
                # in one holding 3 % of its energy.
                relevance = clamp(
                    (band_rel_db[maskee][j] - _MASK_BAND_FLOOR_DB) / -_MASK_BAND_FLOOR_DB, 0.0, 1.0
                )
                per_pair.append((overlap_db * co_ratio * relevance, pair))

            per_pair.sort(key=lambda kv: kv[0], reverse=True)
            out.extend(per_pair[:_MASK_MAX_PER_PAIR])

    out.sort(key=lambda kv: kv[0], reverse=True)
    return [p for _, p in out[:_MASK_MAX_TOTAL]]


# ---------------------------------------------------------------------------
# Kick vs bass — two objects, finally looked at separately
# ---------------------------------------------------------------------------


def _kick_regularity(onsets: np.ndarray) -> float:
    """Share of inter-onset gaps landing on the dominant spacing.

    Four-on-the-floor scores 1.0; a syncopated pattern still scores well over
    half; envelope peaks picked out of shaped noise score low, because their
    gaps spread over everything the minimum spacing allows. Without this a
    noise-based reference file reports a kick at 34 Hz, which is a number about
    the peak-picker rather than about the music.
    """
    if onsets.size < 3:
        return 0.0
    gaps = np.diff(onsets).astype(np.float64)
    median = float(np.median(gaps))
    if median <= 0.0:
        return 0.0
    return float(np.mean(np.abs(gaps - median) <= 0.15 * median))


def _kick_vs_bass(
    drums: Optional[np.ndarray], bass: Optional[np.ndarray], sr: int
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """(kick_to_bass_db, kick_fundamental_hz, bass_fundamental_hz).

    `lowend.py` has to reconstruct the kick by subtracting a between-hits
    spectrum from an at-hits spectrum, because in a 2-track the kick and the
    808 are one waveform. Here the drums are already separated from the bass,
    so the only work left is isolating the kick *inside* the drum stem — low
    band plus transient gating — and the comparison against the bass stem is a
    direct one.

    Everything here stays None unless a kick is genuinely there: enough hits,
    on a recognizable grid, punching above the stem's own low-end floor.
    """
    kick_f: Optional[float] = None
    bass_f: Optional[float] = None
    kick_to_bass: Optional[float] = None

    kick_rms = 0.0
    if drums is not None:
        dm = _mono(drums)
        low = np.ascontiguousarray(bandpass(dm, sr, _LOW_BAND[0], _LOW_BAND[1]))
        n = int(low.shape[0])
        env = envelope(low, sr, smooth_ms=6.0)
        # Kick hits: peaks of the low-band envelope, at least 110 ms apart.
        if n > int(0.5 * sr) and float(env.max()) > 1e-7:
            prom = 0.25 * float(np.percentile(env, 98.0))
            peaks, _ = sps.find_peaks(
                env, distance=max(1, int(0.110 * sr)), prominence=max(prom, 1e-6)
            )
        else:
            peaks = np.zeros(0, dtype=np.int64)

        # A hit has to punch above the surrounding low end, and the hits have
        # to form a grid. Shaped noise clears neither.
        if peaks.size >= _MIN_KICK_HITS:
            floor = max(float(np.percentile(env, 30.0)), EPS)
            definition_db = float(
                np.median(20.0 * np.log10(np.maximum(env[peaks], EPS) / floor))
            )
            if (
                definition_db < _MIN_KICK_DEFINITION_DB
                or _kick_regularity(peaks) < _MIN_KICK_REGULARITY
            ):
                peaks = np.zeros(0, dtype=np.int64)
        else:
            peaks = np.zeros(0, dtype=np.int64)

        win = int(min(_NFFT, max(1024, n // 2)))
        if peaks.size >= 2 and n >= win + 1:
            starts = peaks[peaks + win <= n].astype(np.int64)
            if starts.size > 200:
                starts = starts[np.linspace(0, starts.size - 1, 200).round().astype(np.int64)]
            if starts.size:
                nfft = max(16384, win)
                window = np.hanning(win)
                scale = 1.0 / (window.sum() + EPS)
                acc = np.zeros(nfft // 2 + 1, dtype=np.float64)
                offsets = np.arange(win, dtype=np.int64)
                for i in range(0, starts.size, 64):  # chunks of windows, not samples
                    seg = dm[starts[i : i + 64][:, None] + offsets[None, :]] * window
                    acc += ((np.abs(np.fft.rfft(seg, n=nfft, axis=1)) * scale) ** 2).sum(axis=0)
                acc /= float(starts.size)
                freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
                f0, _ = _peak_freq(acc, freqs, *_KICK_SEARCH)
                kick_f = _f(f0, 0.0, 0.0, 500.0) or None

                # The kick's level: RMS of the low band over its own hit windows.
                hit = int(_KICK_WIN_SEC * sr)
                hit_starts = peaks[peaks + hit <= n].astype(np.int64)
                if hit_starts.size:
                    idx = hit_starts[:, None] + np.arange(hit)[None, :]
                    kick_rms = float(np.sqrt(np.mean(np.square(low[idx]))))

    bass_rms = 0.0
    if bass is not None:
        bm = _mono(bass)
        if float(np.abs(bm).max() if bm.size else 0.0) > 1e-7:
            freqs_w, psd = _welch(bm, sr)
            f0, p0 = _peak_freq(psd, freqs_w, *_BASS_SEARCH)
            # A note has a peak; broadband low end does not. Without this a
            # noise-shaped reference reports a "bass fundamental" that is just
            # wherever the noise happened to be loudest.
            in_range = (freqs_w >= _BASS_SEARCH[0]) & (freqs_w <= _BASS_SEARCH[1])
            median = float(np.median(psd[in_range])) if np.any(in_range) else 0.0
            if p0 > EPS and median > EPS and 10.0 * math.log10(p0 / median) >= _MIN_BASS_PEAK_DB:
                f0 = _prefer_subharmonic(psd, freqs_w, f0, p0, _BASS_SEARCH[0])
                bass_f = _f(f0, 0.0, 0.0, 500.0) or None

            blow = bandpass(bm, sr, _LOW_BAND[0], _LOW_BAND[1])
            win = max(1, int(FRAME_SEC * sr))
            hop = max(1, int(HOP_SEC * sr))
            frames = frame_signal(np.ascontiguousarray(blow), min(win, max(blow.shape[0], 1)), hop)
            rms = np.sqrt(np.maximum(np.einsum("ij,ij->i", frames, frames) / frames.shape[1], 0.0))
            if rms.size:
                db = np.asarray(amp_to_db(rms), dtype=np.float64)
                act = db > (float(db.max()) - _ACTIVE_BELOW_STEM_DB)
                bass_rms = float(np.sqrt(np.mean(np.square(rms[act])))) if np.any(act) else 0.0

    if kick_rms > EPS and bass_rms > EPS:
        kick_to_bass = _f(20.0 * math.log10(kick_rms / bass_rms), 0.0, -60.0, 60.0)

    return kick_to_bass, kick_f, bass_f


# ---------------------------------------------------------------------------
# Public measurement entry point
# ---------------------------------------------------------------------------


def _unavailable(warnings: List[str]) -> StemAnalysis:
    """Fully populated "we could not do this" result. Never None fields."""
    return StemAnalysis(
        available=False,
        model_name=MODEL_NAME,
        separation_ms=0,
        stems=[],
        masking_pairs=[],
        vocal_to_instrument_db=None,
        kick_to_bass_db=None,
        kick_fundamental_hz=None,
        bass_fundamental_hz=None,
        warnings=list(warnings),
    )


def _vocal_expected(genre: str) -> Optional[Tuple[bool, str]]:
    """(expected, label) for a genre, or None if we cannot say.

    The only genre-aware line in this module, and it changes no measurement —
    it decides whether "no vocal stem" is worth remarking on. `targets` is
    imported lazily and defensively so this module still works stand-alone.
    """
    if not genre:
        return None
    try:
        from ..targets import get_profile

        profile = get_profile(genre)
        return bool(profile.vocal_expected), str(profile.label)
    except Exception:
        return None


def measure_stems(
    buf: AudioBuffer, genre: str = "", *, timeout_s: float = SEPARATION_TIMEOUT_S
) -> StemAnalysis:
    """Separate `buf` and measure every source on its own.

    Returns a fully-populated `StemAnalysis` in every case. `available=False`
    plus a warning is how failure is reported; nothing here raises, and no
    field ever comes back NaN, inf or None where the model forbids it.

    `genre` is used for exactly one thing — deciding whether a missing vocal
    stem is worth a warning — and never touches a measurement.
    """
    warnings: List[str] = []

    if buf.duration < MIN_SEPARATION_SEC:
        return _unavailable(
            [
                f"Stem separation skipped: {buf.duration:.1f}s of audio is below the "
                f"{MIN_SEPARATION_SEC:.0f}s minimum for a separated measurement to mean "
                "anything."
            ]
        )

    # DC removed first: a constant-offset file has a healthy-looking RMS and no
    # audio in it at all, and the separator will happily hallucinate four stems
    # out of the filter transient at its leading edge.
    ac_left = buf.left - float(np.mean(buf.left))
    ac_right = buf.right - float(np.mean(buf.right))
    mix_rms = float(np.sqrt(np.mean(np.square(ac_left) + np.square(ac_right)) * 0.5))
    if not math.isfinite(mix_rms) or mix_rms < 1e-5:
        return _unavailable(
            ["Stem separation skipped: this file is at or below the noise floor "
             f"({_f(amp_to_db(mix_rms), -120.0):.0f} dBFS RMS once DC is removed)."]
        )

    # Every per-stem figure is a ratio against the mix. If the mix itself never
    # clears the -70 LUFS absolute gate — DC, infrasonic rumble, dither noise —
    # those ratios are two floors divided by each other, and separating first
    # would only produce four stems' worth of confident nonsense.
    mix_lufs = _integrated_lufs(buf.left, buf.right, int(buf.sr))
    if mix_lufs <= _ABS_GATE_LUFS + 0.5:
        return _unavailable(
            [
                "Stem separation skipped: the mix never clears the -70 LUFS gate "
                f"({mix_lufs:.1f} LUFS), so every per-stem level would be one floor "
                "measured against another."
            ]
        )

    result, sep_warnings = _separate_impl(buf, timeout_s)
    warnings.extend(sep_warnings)
    if result is None:
        return _unavailable(warnings)

    try:
        return _measure(result, buf, genre, warnings)
    except Exception as exc:  # a measurement bug must not kill the analysis
        warnings.append(
            f"Stems were separated but could not be measured ({type(exc).__name__}: {exc})."
        )
        return _unavailable(warnings)


def _measure(
    result: SeparationResult, buf: AudioBuffer, genre: str, warnings: List[str]
) -> StemAnalysis:
    sr = ANALYSIS_SR
    hop = max(1, int(HOP_SEC * sr))
    mix = result.mix
    n = int(mix.shape[1])

    # --- reference figures from the exact excerpt that was separated --------
    mix_lufs = _integrated_lufs(mix[0], mix[1], sr)
    _, mix_rms_frames, _ = _frame_peak_rms(
        mix[0], mix[1], max(1, min(int(FRAME_SEC * sr), n)), hop
    )
    mix_frame_peak_db = (
        float(np.asarray(amp_to_db(mix_rms_frames)).max()) if mix_rms_frames.size else -120.0
    )

    kinds = [k for k in STEM_KINDS if k in result.stems]
    if not kinds:
        warnings.append("The separator returned no recognizable sources.")
        return _unavailable(warnings)

    # --- one spectrogram pass per stem, folded straight into macro bands ----
    band_energy: Dict[str, np.ndarray] = {}   # (n_frames, 9) linear power
    centroid: Dict[str, float] = {}
    times = np.zeros(0)
    bmat = None
    audible = None
    for kind in kinds:
        x = result.stems[kind]
        freqs, t, pl = _power_spectrogram(x[0], sr, _NFFT, hop)
        _, _, pr = _power_spectrogram(x[1], sr, _NFFT, hop)
        if bmat is None:
            bmat = _macro_band_matrix(freqs)
            audible = (freqs >= 20.0) & (freqs <= 20_000.0)
            times = t
        # Average of the two channels: a per-channel mean square, so the dB
        # figures read like a meter rather than like "twice a meter".
        power = 0.5 * (pl + pr)
        band_energy[kind] = power @ bmat.T
        # Power-weighted spectral centroid over the audible range — the same
        # definition spectral.py uses for the mix, so the two are comparable.
        mean_spec = power.mean(axis=0)
        total = float(mean_spec[audible].sum())
        centroid[kind] = (
            float((freqs[audible] * mean_spec[audible]).sum() / total) if total > EPS else 0.0
        )
        del pl, pr, power, mean_spec

    n_frames = int(times.shape[0])
    if n_frames == 0 or bmat is None:
        warnings.append("The separated stems were too short to frame.")
        return _unavailable(warnings)

    # --- per-stem measurement ----------------------------------------------
    band_mean = {k: band_energy[k].mean(axis=0) for k in kinds}
    measurements: List[StemMeasurement] = [
        _measure_one(
            kind, result.stems[kind], sr, mix_lufs, mix_frame_peak_db,
            band_mean[kind], centroid[kind],
        )
        for kind in kinds
    ]

    # --- band occupancy: only computable with every stem in hand ------------
    total_band = np.zeros(len(MACRO_BAND_ORDER), dtype=np.float64)
    for kind in kinds:
        total_band += band_mean[kind]
    for m in measurements:
        share = band_mean[m.kind] / np.maximum(total_band, EPS)
        m.band_occupancy = {
            name: round(_f(float(share[j]), 0.0, 0.0, 1.0), 4)
            for j, name in enumerate(MACRO_BAND_ORDER)
        }

    present = [m.kind for m in measurements if m.present]

    # --- masking ------------------------------------------------------------
    band_db: Dict[str, np.ndarray] = {}
    band_active: Dict[str, np.ndarray] = {}
    band_rel_db: Dict[str, np.ndarray] = {}
    for kind in present:
        e = band_energy[kind]
        band_db[kind] = np.asarray(_db(e), dtype=np.float64)
        total = e.sum(axis=1)
        total_db = np.asarray(_db(total), dtype=np.float64)
        ceiling = float(total_db.max()) if total_db.size else -120.0
        band_active[kind] = total_db > (ceiling - _ACTIVE_BELOW_STEM_DB)
        peak_band = float(band_mean[kind].max())
        band_rel_db[kind] = np.asarray(
            _db(band_mean[kind] / max(peak_band, EPS)), dtype=np.float64
        )

    masking_pairs = _masking_pairs(
        present, band_db, band_active, band_rel_db, times, result.t_start, buf.duration
    )

    # --- vocal vs the rest of the band --------------------------------------
    vocal_to_instrument_db: Optional[float] = None
    if "vocals" in present:
        others = [k for k in kinds if k != "vocals"]
        if others:
            inst = np.zeros_like(result.stems[others[0]])
            for k in others:
                inst = inst + result.stems[k]
            inst_lufs = _integrated_lufs(inst[0], inst[1], sr)
            voc = next(m for m in measurements if m.kind == "vocals")
            vocal_to_instrument_db = round(
                _f(voc.level_lufs - inst_lufs, 0.0, -60.0, 60.0), 2
            )

    # --- kick vs bass --------------------------------------------------------
    drums = result.stems.get("drums") if "drums" in present else None
    bass = result.stems.get("bass") if "bass" in present else None
    kick_to_bass_db, kick_hz, bass_hz = _kick_vs_bass(drums, bass, sr)

    # --- honest notes about what was and wasn't there ------------------------
    absent = [m for m in measurements if not m.present]
    for m in absent:
        if m.kind != "vocals":
            continue
        note = (
            "No vocal stem: the separator found nothing above "
            f"{_PRESENT_RATIO_DB:.0f} dB relative to the mix "
            f"(vocals came back {m.level_ratio_db:.1f} dB down, peak "
            f"{m.peak_dbfs:.1f} dBFS). Treating this as an instrumental."
        )
        expected = _vocal_expected(genre)
        if expected is not None and expected[0]:
            note += f" That is unusual for {expected[1]}."
        warnings.append(note)
    others = [m for m in absent if m.kind != "vocals"]
    if others:
        listed = ", ".join(f"{m.kind} ({m.level_ratio_db:.0f} dB down)" for m in others)
        one = len(others) == 1
        warnings.append(
            f"No {'source' if one else 'sources'} for {listed} — what came back is "
            f"separation bleed, not a part, so nothing is reported for "
            f"{'it' if one else 'them'}."
        )

    if result.device == "cpu":
        warnings.append(
            f"Separation ran on the CPU and took {result.separation_ms / 1000.0:.1f}s."
        )

    return StemAnalysis(
        available=True,
        model_name=result.model_name,
        separation_ms=int(result.separation_ms),
        stems=measurements,
        masking_pairs=masking_pairs,
        vocal_to_instrument_db=vocal_to_instrument_db,
        kick_to_bass_db=None if kick_to_bass_db is None else round(kick_to_bass_db, 2),
        kick_fundamental_hz=None if kick_hz is None else round(kick_hz, 2),
        bass_fundamental_hz=None if bass_hz is None else round(bass_hz, 2),
        warnings=warnings,
    )
