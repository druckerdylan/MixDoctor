"""Shared audio loading, framing, and band math.

Everything downstream measures against the structures defined here, so the
band edges and hop size are fixed in one place. Two rules for this module:

1. No genre knowledge and no thresholds. Those live in `targets.py`.
2. Nothing loops over samples in Python. A 6-minute stereo track is ~32M
   samples per channel; a per-sample loop turns a 2-second job into a minute.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy import signal as sps

# Analysis runs at a fixed rate so band edges, window lengths and hop counts
# are identical for every upload regardless of the source file's rate.
ANALYSIS_SR = 48_000

# Time-series resolution. 0.25 s hops let us place an issue on the timeline
# tightly enough to scrub to it, without generating thousands of frames.
FRAME_SEC = 0.400
HOP_SEC = 0.250

EPS = 1e-12

# ISO 1/3-octave centres, 20 Hz .. 20 kHz.
THIRD_OCTAVE_CENTERS: Tuple[float, ...] = (
    20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0,
    200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0,
    2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0, 12500.0,
    16000.0, 20000.0,
)

# Macro bands used for stereo, phase and per-band reporting. The low end is
# split three ways on purpose: sub / kick-fundamental / bass-body behave very
# differently and collapsing them hides exactly the problems producers care
# about most.
MACRO_BANDS: Dict[str, Tuple[float, float]] = {
    "sub": (20.0, 60.0),
    "low_bass": (60.0, 120.0),
    "upper_bass": (120.0, 250.0),
    "low_mid": (250.0, 500.0),
    "mid": (500.0, 1000.0),
    "upper_mid": (1000.0, 2000.0),
    "presence": (2000.0, 5000.0),
    "brilliance": (5000.0, 10000.0),
    "air": (10000.0, 20000.0),
}
MACRO_BAND_ORDER: Tuple[str, ...] = tuple(MACRO_BANDS.keys())

# Named regions that specific detectors care about.
MUD_HZ = (150.0, 400.0)
BOXY_HZ = (300.0, 600.0)
HARSH_HZ = (2000.0, 5000.0)
SIBILANCE_HZ = (5000.0, 9000.0)
KICK_HZ = (40.0, 110.0)
SUB_HZ = (20.0, 60.0)
VOCAL_HZ = (300.0, 6000.0)
PRESENCE_HZ = (2000.0, 6000.0)

SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".aiff", ".aif", ".ogg", ".opus"}


class AudioTooShortError(ValueError):
    """Raised when a file is too short for the analysis windows to mean anything."""


class SilentAudioError(ValueError):
    """Raised when a file carries no measurable signal."""


@dataclass
class AudioBuffer:
    """A decoded, analysis-rate stereo buffer plus the derived views everyone needs.

    `left`/`right` are always populated — a mono source is duplicated, and
    `is_mono` records that so detectors don't report a stereo verdict on a file
    that never had a stereo field to begin with.
    """

    left: np.ndarray
    right: np.ndarray
    sr: int
    duration: float
    original_sr: int
    is_mono: bool
    peak_sample: float
    source_bit_depth: Optional[int] = None

    @property
    def mid(self) -> np.ndarray:
        return (self.left + self.right) * 0.5

    @property
    def side(self) -> np.ndarray:
        return (self.left - self.right) * 0.5

    @property
    def mono(self) -> np.ndarray:
        return self.mid

    @property
    def stereo(self) -> np.ndarray:
        """(n, 2) interleaved view, for libraries that want a 2-D array."""
        return np.column_stack([self.left, self.right])

    @property
    def n_samples(self) -> int:
        return self.left.shape[0]


def _decode(path: str) -> Tuple[np.ndarray, int, Optional[int]]:
    """Decode any supported container to float64 samples.

    soundfile handles wav/flac/aiff/ogg natively and is exact about sample
    values, which matters for clip detection. Anything it refuses goes through
    ffmpeg via pydub.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported audio format: {ext or '(none)'}")

    bit_depth = None
    try:
        info = sf.info(path)
        subtype = (info.subtype or "").upper()
        for depth in (8, 16, 24, 32):
            if str(depth) in subtype:
                bit_depth = depth
                break
        data, sr = sf.read(path, always_2d=True, dtype="float64")
        return data, sr, bit_depth
    except Exception:
        pass

    from pydub import AudioSegment  # imported lazily: pulls in ffmpeg

    seg = AudioSegment.from_file(path)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        seg.export(tmp.name, format="wav")
        data, sr = sf.read(tmp.name, always_2d=True, dtype="float64")
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
    return data, sr, seg.sample_width * 8


def load_audio(path: str, max_seconds: float = 600.0) -> AudioBuffer:
    """Decode, downmix to stereo, resample to ANALYSIS_SR, and validate.

    `peak_sample` is captured from the *original* decode, before resampling —
    resampling can overshoot, and we don't want a resampler artefact to be
    reported as the user's clipping.
    """
    data, sr, bit_depth = _decode(path)

    if data.ndim == 1:
        data = data[:, None]
    n_ch = data.shape[1]
    is_mono = n_ch == 1

    if n_ch == 1:
        left = right = data[:, 0]
    elif n_ch == 2:
        left, right = data[:, 0], data[:, 1]
    else:
        # >2 channels: fold everything past the first pair into both sides so
        # a surround stem doesn't silently lose its centre channel.
        extra = data[:, 2:].mean(axis=1)
        left = data[:, 0] + extra * 0.7071
        right = data[:, 1] + extra * 0.7071

    peak_sample = float(np.max(np.abs(data))) if data.size else 0.0

    if sr != ANALYSIS_SR:
        g = np.gcd(int(sr), ANALYSIS_SR)
        up, down = ANALYSIS_SR // g, sr // g
        left = sps.resample_poly(left, up, down)
        right = sps.resample_poly(right, up, down)

    if max_seconds:
        limit = int(max_seconds * ANALYSIS_SR)
        if left.shape[0] > limit:
            left, right = left[:limit], right[:limit]

    duration = left.shape[0] / ANALYSIS_SR
    if duration < 1.0:
        raise AudioTooShortError(
            f"Track is {duration:.2f}s. At least 1 second of audio is needed."
        )

    rms = float(np.sqrt(np.mean(left**2 + right**2) * 0.5))
    if rms < 1e-6:
        raise SilentAudioError("This file is silent (or near-silent) — nothing to measure.")

    return AudioBuffer(
        left=np.ascontiguousarray(left, dtype=np.float64),
        right=np.ascontiguousarray(right, dtype=np.float64),
        sr=ANALYSIS_SR,
        duration=duration,
        original_sr=int(sr),
        is_mono=bool(is_mono),
        peak_sample=peak_sample,
        source_bit_depth=bit_depth,
    )


# ---------------------------------------------------------------------------
# Framing / spectra
# ---------------------------------------------------------------------------


def frame_signal(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Strided (n_frames, frame_len) view. No copy, no Python loop."""
    if x.shape[0] < frame_len:
        pad = np.zeros(frame_len, dtype=x.dtype)
        pad[: x.shape[0]] = x
        return pad[None, :]
    n_frames = 1 + (x.shape[0] - frame_len) // hop
    return np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, frame_len),
        strides=(x.strides[0] * hop, x.strides[0]),
        writeable=False,
    )


def frame_times(n_frames: int, frame_len: int, hop: int, sr: int) -> np.ndarray:
    """Centre time, in seconds, of each frame."""
    return (np.arange(n_frames) * hop + frame_len * 0.5) / sr


def stft_power(
    x: np.ndarray, sr: int, n_fft: int = 8192, hop: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Power spectrogram.

    Returns (freqs[n_bins], times[n_frames], power[n_frames, n_bins]).
    8192 points at 48 kHz is ~5.9 Hz resolution — enough to separate a kick
    fundamental from a bass note a semitone away, which coarser windows can't.
    """
    hop = hop or int(HOP_SEC * sr)
    win_len = min(n_fft, max(256, int(FRAME_SEC * sr)))
    window = np.hanning(win_len)

    frames = frame_signal(x, win_len, hop)
    if frames.shape[0] == 0:
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        return freqs, np.zeros(0), np.zeros((0, freqs.shape[0]))

    windowed = frames * window
    spec = np.fft.rfft(windowed, n=n_fft, axis=1)
    # Normalise so the result is independent of window length/gain.
    scale = 1.0 / (np.sum(window) + EPS)
    power = (np.abs(spec) * scale) ** 2

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    times = frame_times(frames.shape[0], win_len, hop, sr)
    return freqs, times, power


def band_edges(centers: Tuple[float, ...] = THIRD_OCTAVE_CENTERS) -> np.ndarray:
    """Lower/upper edges for 1/3-octave bands: centre * 2^(±1/6)."""
    c = np.asarray(centers, dtype=np.float64)
    return np.stack([c * 2 ** (-1 / 6), c * 2 ** (1 / 6)], axis=1)


def band_matrix(freqs: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """(n_bands, n_bins) 0/1 matrix mapping FFT bins onto bands.

    Precomputing this turns per-band energy into one matmul instead of a
    Python loop per band per frame.
    """
    lo, hi = edges[:, 0][:, None], edges[:, 1][:, None]
    return ((freqs[None, :] >= lo) & (freqs[None, :] < hi)).astype(np.float64)


def band_energy(power: np.ndarray, bmat: np.ndarray) -> np.ndarray:
    """Per-band power. power is (n_frames, n_bins) -> (n_frames, n_bands)."""
    return power @ bmat.T


def to_db(x: np.ndarray | float, floor_db: float = -120.0) -> np.ndarray | float:
    """Power ratio -> dB, floored so silence doesn't produce -inf."""
    arr = np.asarray(x, dtype=np.float64)
    out = 10.0 * np.log10(np.maximum(arr, EPS))
    out = np.maximum(out, floor_db)
    return float(out) if np.isscalar(x) or out.ndim == 0 else out


def amp_to_db(x: np.ndarray | float, floor_db: float = -120.0) -> np.ndarray | float:
    """Amplitude ratio -> dB."""
    arr = np.asarray(x, dtype=np.float64)
    out = 20.0 * np.log10(np.maximum(np.abs(arr), EPS))
    out = np.maximum(out, floor_db)
    return float(out) if np.isscalar(x) or out.ndim == 0 else out


def rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    return float(amp_to_db(np.sqrt(np.mean(np.square(x)))))


def bandpass(x: np.ndarray, sr: int, lo: float, hi: float, order: int = 4) -> np.ndarray:
    """Zero-phase band-pass. filtfilt so transient timing isn't shifted."""
    nyq = sr * 0.5
    lo_n = max(lo / nyq, 1e-5)
    hi_n = min(hi / nyq, 0.999)
    if lo_n >= hi_n:
        return np.zeros_like(x)
    if hi_n >= 0.998:
        sos = sps.butter(order, lo_n, btype="high", output="sos")
    elif lo_n <= 1e-4:
        sos = sps.butter(order, hi_n, btype="low", output="sos")
    else:
        sos = sps.butter(order, [lo_n, hi_n], btype="band", output="sos")
    return sps.sosfiltfilt(sos, x)


def band_rms_db(x: np.ndarray, sr: int, lo: float, hi: float) -> float:
    return rms_db(bandpass(x, sr, lo, hi))


def envelope(x: np.ndarray, sr: int, smooth_ms: float = 10.0) -> np.ndarray:
    """Smoothed amplitude envelope via rectify + one-pole low-pass."""
    rect = np.abs(x)
    alpha = float(np.exp(-1.0 / (max(smooth_ms, 0.1) * 1e-3 * sr)))
    return sps.lfilter([1 - alpha], [1, -alpha], rect)


def percentile_safe(x: np.ndarray, q: float, default: float = 0.0) -> float:
    if x.size == 0:
        return default
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return default
    return float(np.percentile(finite, q))


def clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def merge_spans(
    times: np.ndarray, mask: np.ndarray, min_gap: float = 0.5, min_len: float = 0.0
) -> List[Tuple[float, float]]:
    """Turn a per-frame boolean mask into merged (start, end) time spans.

    Frames flagged within `min_gap` seconds of each other become one span, so
    a flickering detector produces "0:42–0:51" rather than forty markers.
    """
    if times.size == 0 or not np.any(mask):
        return []
    idx = np.flatnonzero(mask)
    breaks = np.flatnonzero(np.diff(times[idx]) > min_gap)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [idx.size - 1]])

    spans: List[Tuple[float, float]] = []
    for s, e in zip(starts, ends):
        t0, t1 = float(times[idx[s]]), float(times[idx[e]])
        if t1 - t0 < min_len:
            t1 = t0 + min_len
        spans.append((t0, t1))
    return spans
