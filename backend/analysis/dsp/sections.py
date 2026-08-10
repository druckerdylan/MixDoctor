"""Structural segmentation: what the record does over its own length.

One set of numbers for a whole track hides the things producers actually get
wrong. The chorus that fails to lift. The drop where the low end collapses. The
verse sitting 4 dB under everything around it. None of those move the
whole-file integrated loudness enough to notice, and all of them are audible
the moment somebody plays the record end to end.

So this module measures the record *in pieces*:

1. **Segment.** A feature matrix — MFCC, chroma and per-macro-band energy on
   the shared 0.25 s grid — becomes a self-similarity matrix, and a Foote
   checkerboard kernel run down its diagonal becomes a novelty curve. Peaks in
   that curve are candidate boundaries.
2. **Gate.** A novelty peak is only accepted where the audio on either side
   *measurably* differs: a level change, or a change in tonal balance, in dB.
   Self-normalising similarity will find "structure" in a 4-minute loop if you
   let it, and a report that invents an arrangement is worse than one that says
   the track is uniform.
3. **Cluster and label.** Segments are clustered by feature similarity, then
   given human labels from what was *measured* — loudness, onset density,
   position — not from a guess at song form. Segments that cluster together
   share a label. This is a heuristic and the docstrings say so; the numbers
   attached to each section are not.
4. **Measure.** Per section: integrated LUFS (BS.1770 gating over the K-weighted
   power, from `loudness.k_weight` — the weighting is not reimplemented here),
   crest factor, stereo width, and all nine macro-band levels.

Measurement only. No thresholds, no verdicts, no genre knowledge — the optional
`genre` argument is a *hint* for one cosmetic choice (whether the loudest
recurring cluster is called "chorus" or "drop") and the module measures its own
answer when it is not supplied.

Degrades rather than fails: librosa provides the MFCC/chroma/onset front end
when it imports, numpy equivalents cover it when it does not, and any internal
failure falls back to a single section covering the file. It never returns zero
sections and it never raises.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal as sps
from scipy.cluster import hierarchy as sch
from scipy.fft import dct
from scipy.spatial.distance import pdist, squareform

from ..core import (
    EPS,
    FRAME_SEC,
    HOP_SEC,
    MACRO_BAND_ORDER,
    MACRO_BANDS,
    AudioBuffer,
    amp_to_db,
    band_energy,
    band_matrix,
    clamp,
    frame_signal,
    stft_power,
)
from ..types import Section, SectionAnalysis

__all__ = ["measure_sections"]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

# Below this a track has no structure to find: a 15 s idea is one section, and
# saying so is more useful than cutting it into thirds.
MIN_SEGMENTABLE_SEC = 20.0

# Section length bounds. The divisor is what keeps a 4-minute song in the 4-12
# range the UI is built for instead of producing forty micro-segments: at 240 s
# it puts the floor at 17 s, so twelve sections is the hard maximum and eight
# is typical.
MIN_SECTION_DIVISOR = 14.0
MIN_SECTION_FLOOR_SEC = 6.0
MIN_SECTION_CEIL_SEC = 25.0
MAX_SECTIONS = 12

# Foote kernel half-width. 8 s either side is long enough to average over a
# phrase and short enough to place a boundary within a bar or two.
KERNEL_HALF_SEC = 8.0

# Novelty is normalised so it reads as (mean similarity within a side) minus
# (mean similarity across the boundary): 0 on uniform material, ~0.3-0.8 at a
# real section change. The floor is deliberately low because the contrast gate
# below, not this, is what rejects imaginary boundaries.
NOVELTY_FLOOR = 0.035

# A boundary must correspond to an audible change. `CONTRAST_DB` is the RMS of
# the per-macro-band level change across it, in dB, combined with the broadband
# level change — roughly "the two sides differ by 1.6 dB somewhere that matters".
CONTRAST_DB = 1.6
RELAXED_CONTRAST_DB = 1.0

# Similarity bandwidth floor, in feature units. Without a floor the affinity
# self-normalises and a dead-uniform loop's numerical noise blows up into
# structure; with it, uniform material stays uniform.
SIGMA_FLOOR = 0.30

# Cut height for the segment dendrogram, in the same feature units. Because the
# feature vector is scaled by fixed constants rather than z-scored, this is a
# real distance and not a per-track guess: measured across the test material,
# two passes of the same part land at 0.0-0.4, a build against the drop it feeds
# at ~1.7, and a chorus against a verse at 5-8.
CLUSTER_T_MIN = 0.8
CLUSTER_T_MAX = 1.2
MAX_CLUSTERS = 5

# How far under the median a cluster has to sit before it stops being one of the
# record's main parts and starts being a breakdown.
QUIET_CLUSTER_LU = 4.5

# Sections quieter than this are treated as silence for the summary statistics —
# a silent tail must not become "the quietest section" and inflate the spread.
AUDIBLE_FLOOR_LUFS = -60.0

# Low-end swing is measured across the record's *core* — every section within
# this much of the loudest one. An intro with no bass in it is an arrangement,
# not a fault; a chorus with no bass in it is a fault, and mixing the two into
# one number turns the useful signal into "the intro is quiet", which we
# already report as loudness_spread_lu.
LOW_CORE_LU = 9.0

# Floor for the low-end share, so a section with literally nothing under 120 Hz
# cannot produce a three-figure swing.
LOW_SHARE_FLOOR_DB = -40.0

# BS.1770 gating, mirrored from the loudness module so a section's LUFS is the
# same measurement the whole-file figure uses.
BLOCK_SEC = 0.400
BLOCK_STEP_SEC = 0.100
ABS_GATE_LUFS = -70.0
_LUFS_OFFSET = -0.691

WIDTH_MAX = 20.0          # matches stereo.WIDTH_MAX
ENERGY_FLOOR = 1e-14      # matches stereo.ENERGY_FLOOR

# Feature front end.
N_MFCC = 13               # coefficients kept, c0 (level) excluded
N_CHROMA = 12
CEPSTRAL_BANDS = 48       # log-spaced analysis bands behind the numpy fallback
FEATURE_LO_HZ = 30.0
FEATURE_HI_HZ = 16_000.0

# Group scaling for the feature vector. Fixed constants, not per-track z-scores:
# z-scoring makes a uniform track's noise look like structure, which is exactly
# the failure this module has to avoid.
_W_MFCC = 1.0
_W_CHROMA = 0.7
_W_SHAPE = 1.0
_W_LEVEL = 1.2

_ONSET_SR = 24_000        # onset front end runs decimated; structure is slow
_ONSET_HOP = 512          # ~21 ms at 24 kHz

# Genres whose loudest recurring section is a "drop" rather than a "chorus".
# The only genre knowledge in the module, it changes one word, and it is
# optional — `_reads_electronic` measures the same call when no hint arrives.
_DROP_GENRES = frozenset({
    "edm", "house", "techno", "dnb", "dubstep", "trap", "future_bass",
    "electronic", "dance", "drum_and_bass",
})
_DROP_CREST_DB = 8.5
_DROP_LOW_DOMINANCE_DB = 14.0

_librosa = None
_librosa_tried = False


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _finite(value, default: float = 0.0, lo: float = -1e6, hi: float = 1e6) -> float:
    """Coerce anything to a finite, bounded float. NaN/inf break JSON."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        out = float(default)
    return float(min(max(out, lo), hi))


def _clock(t: float) -> str:
    """Seconds -> m:ss (or h:mm:ss), for notes a human reads."""
    t = max(0.0, _finite(t, 0.0, 0.0, 86_400.0))
    total = int(round(t))
    if total >= 3600:
        return f"{total // 3600}:{(total % 3600) // 60:02d}:{total % 60:02d}"
    return f"{total // 60}:{total % 60:02d}"


def _num(value: float, nd: int = 1) -> str:
    v = _finite(value, 0.0)
    if abs(v) < 5e-3:
        v = 0.0
    text = f"{v:.{nd}f}"
    return "0" if text in ("-0", "-0.0", "-0.00") else text


def _get_librosa():
    """Import librosa once, and never let its absence take the module down."""
    global _librosa, _librosa_tried
    if not _librosa_tried:
        _librosa_tried = True
        try:
            import librosa  # deferred: the import is the slow part, not the use
            _librosa = librosa
        except Exception:  # pragma: no cover - librosa is a project dependency
            _librosa = None
    return _librosa


# ---------------------------------------------------------------------------
# Frame features
# ---------------------------------------------------------------------------


def _log_bank(freqs: np.ndarray, n_bands: int, lo: float, hi: float) -> np.ndarray:
    """(n_bands, n_bins) log-spaced band matrix for the cepstral fallback."""
    edges = np.geomspace(max(lo, 1.0), min(hi, float(freqs[-1]) if freqs.size else hi), n_bands + 1)
    mat = np.zeros((n_bands, freqs.size), dtype=np.float64)
    for i in range(n_bands):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        if not np.any(mask):
            # Band narrower than one bin: take the nearest bin so no row is empty.
            mask = np.zeros_like(freqs, dtype=bool)
            mask[int(np.argmin(np.abs(freqs - math.sqrt(edges[i] * edges[i + 1]))))] = True
        mat[i, mask] = 1.0 / float(np.count_nonzero(mask))
    return mat


def _chroma_bank(freqs: np.ndarray, lo: float = 55.0, hi: float = 5000.0) -> np.ndarray:
    """(12, n_bins) pitch-class fold, for when librosa is unavailable."""
    mat = np.zeros((12, freqs.size), dtype=np.float64)
    usable = (freqs >= lo) & (freqs <= hi)
    if not np.any(usable):
        return mat
    idx = np.flatnonzero(usable)
    pitch = np.round(12.0 * np.log2(freqs[idx] / 440.0)).astype(int) % 12
    mat[pitch, idx] = 1.0
    return mat


def _timbre(
    freqs: np.ndarray, power: np.ndarray, sr: int
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """(mfcc[n_frames, N_MFCC], chroma[n_frames, 12], used_librosa).

    librosa's MFCC and chroma are computed from the spectrogram this module
    already has — no second transform, no resample. The numpy path behind them
    is the same idea done plainly: a DCT of log band energies, and a pitch-class
    fold of the spectrum.
    """
    n_fft = 2 * (freqs.size - 1) if freqs.size > 1 else 2048
    spec = np.ascontiguousarray(power.T)  # librosa wants (n_bins, n_frames)

    librosa = _get_librosa()
    if librosa is not None:
        try:
            mel_basis = librosa.filters.mel(
                sr=sr, n_fft=n_fft, n_mels=64, fmin=FEATURE_LO_HZ,
                fmax=min(FEATURE_HI_HZ, sr * 0.5),
            )
            mel_db = librosa.power_to_db(mel_basis @ spec, ref=1.0, top_db=None)
            mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=N_MFCC + 1)[1:]
            chroma = librosa.feature.chroma_stft(S=spec, sr=sr, n_fft=n_fft)
            mfcc = np.nan_to_num(np.asarray(mfcc, dtype=np.float64).T)
            chroma = np.nan_to_num(np.asarray(chroma, dtype=np.float64).T)
            if mfcc.shape[0] == power.shape[0] and chroma.shape[0] == power.shape[0]:
                return mfcc, chroma, True
        except Exception:
            pass

    bank = _log_bank(freqs, CEPSTRAL_BANDS, FEATURE_LO_HZ, min(FEATURE_HI_HZ, sr * 0.5))
    log_bands = 10.0 * np.log10(np.maximum(power @ bank.T, EPS))
    cep = dct(log_bands, type=2, axis=1, norm="ortho")[:, 1 : N_MFCC + 1]
    if cep.shape[1] < N_MFCC:
        cep = np.pad(cep, ((0, 0), (0, N_MFCC - cep.shape[1])))

    chroma_raw = power @ _chroma_bank(freqs).T
    total = np.maximum(chroma_raw.sum(axis=1, keepdims=True), EPS)
    return np.nan_to_num(cep), np.nan_to_num(chroma_raw / total), False


def _power_scale(freqs: np.ndarray, sr: int) -> float:
    """Factor turning `core.stft_power` output into mean-square (i.e. dBFS).

    Two corrections. The rfft is one-sided, so the total needs doubling. And
    `stft_power` divides by the window's *coherent* gain (sum of the window),
    which is the right normalisation for a sinusoid and 1.76 dB hot for
    broadband content through a Hann window — band levels are broadband, so the
    noise-power bandwidth is the one to correct for. Measured against
    `core.band_rms_db` on pink noise this lands inside half a decibel.

    Derived from the window `stft_power` actually builds rather than assumed, so
    a change to `FRAME_SEC` or the FFT length stays consistent here.
    """
    n_fft = 2 * (int(freqs.size) - 1) if freqs.size > 1 else 2048
    win_len = min(n_fft, max(256, int(FRAME_SEC * sr)))
    window = np.hanning(max(win_len, 2))
    coherent = float(window.sum()) ** 2
    incoherent = float(n_fft) * float(np.square(window).sum())
    if incoherent <= EPS or coherent <= EPS:
        return 2.0
    return 2.0 * (coherent / incoherent)


def _band_levels(
    freqs: np.ndarray, power: np.ndarray, sr: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame (band_db[n, 9], level_db[n]) in honest dBFS.

    The same quantity `core.band_rms_db` returns for the same band, so a
    section's band levels can be read against anything else in the report
    instead of living on a private scale.
    """
    scale = _power_scale(freqs, sr)
    edges = np.array([MACRO_BANDS[name] for name in MACRO_BAND_ORDER], dtype=np.float64)
    bmat = band_matrix(freqs, edges)
    band_power = scale * band_energy(power, bmat)            # (n, 9)
    total_power = scale * power.sum(axis=1)                  # (n,)
    return (
        10.0 * np.log10(np.maximum(band_power, EPS)),
        10.0 * np.log10(np.maximum(total_power, EPS)),
    )


def _frame_matrix(
    freqs: np.ndarray, power: np.ndarray, sr: int,
    band_db: np.ndarray, level_db: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """The feature matrix the segmentation runs on: (features[n, d], used_librosa)."""
    mfcc, chroma, used_librosa = _timbre(freqs, power, sr)

    # Tonal *shape*: band level minus the frame's own level, so a fade does not
    # read as a timbre change.
    shape = band_db - level_db[:, None]
    # Loudness contour, relative to the track's own loudest frame.
    contour = (level_db - float(np.max(level_db))) / 12.0

    groups = [
        (mfcc / 10.0, _W_MFCC),
        ((chroma - chroma.mean(axis=1, keepdims=True)) * 4.0, _W_CHROMA),
        (shape / 8.0, _W_SHAPE),
        (contour[:, None], _W_LEVEL),
    ]
    parts = [
        np.nan_to_num(block) * (weight / math.sqrt(max(block.shape[1], 1)))
        for block, weight in groups
    ]
    features = np.ascontiguousarray(np.concatenate(parts, axis=1), dtype=np.float64)
    return features, used_librosa


# ---------------------------------------------------------------------------
# Novelty
# ---------------------------------------------------------------------------


def _checkerboard(half: int) -> np.ndarray:
    """Gaussian-tapered Foote kernel, normalised to read as a similarity gap.

    Positive weights sum to +1 and negative weights to -1, so the convolution
    result is (mean similarity either side of the centre) - (mean similarity
    across it): 0 on uniform material, up to ~1 at a hard cut.
    """
    idx = np.arange(-half, half, dtype=np.float64) + 0.5
    taper = np.exp(-0.5 * (idx / (half * 0.5)) ** 2)
    kernel = np.outer(taper, taper) * np.sign(np.outer(idx, idx))
    positive = kernel[kernel > 0].sum()
    negative = -kernel[kernel < 0].sum()
    if positive <= EPS or negative <= EPS:
        return np.zeros_like(kernel)
    kernel[kernel > 0] /= positive
    kernel[kernel < 0] /= negative
    return kernel


def _self_similarity(features: np.ndarray) -> np.ndarray:
    """Gaussian affinity from the feature distance matrix, float32.

    The bandwidth adapts to the track but is floored: an adaptive sigma alone
    turns the numerical noise of a dead-uniform loop into a full-contrast
    similarity matrix, which is how self-similarity segmentation invents
    arrangements that are not there.

    That floor is also why this is a dense affinity rather than
    `librosa.segment.recurrence_matrix`. A k-nearest-neighbour recurrence plot
    is self-normalising by construction — every frame gets its k neighbours
    whether or not anything in the track resembles it — so on a four-minute
    loop it produces exactly the confident, structured-looking matrix this
    module must not produce. Foote novelty wants a dense similarity anyway.
    """
    n = features.shape[0]
    if n < 2:
        return np.ones((n, n), dtype=np.float32)
    distances = pdist(features, metric="euclidean")
    sigma = max(float(np.median(distances)), SIGMA_FLOOR)
    affinity = np.exp(-0.5 * np.square(squareform(distances) / sigma))
    return np.nan_to_num(affinity, nan=0.0).astype(np.float32)


def _foote_novelty(similarity: np.ndarray, half: int) -> np.ndarray:
    """Checkerboard novelty down the diagonal, without a Python loop.

    The diagonal blocks are taken as one strided view of the edge-padded matrix
    and reduced in a single tensordot. Padding with the edge value makes the
    first and last frames maximally self-similar, so the curve does not spike
    at the ends of the file.
    """
    n = similarity.shape[0]
    kernel = _checkerboard(half).astype(np.float32)
    if n < 2 * half + 1 or not np.any(kernel):
        return np.zeros(n, dtype=np.float64)

    padded = np.ascontiguousarray(np.pad(similarity, half, mode="edge"))
    row, col = padded.strides
    blocks = np.lib.stride_tricks.as_strided(
        padded, shape=(n, 2 * half, 2 * half), strides=(row + col, row, col), writeable=False
    )
    novelty = np.tensordot(blocks, kernel, axes=([1, 2], [0, 1]))
    return np.nan_to_num(np.asarray(novelty, dtype=np.float64), nan=0.0)


def _pick_peaks(novelty: np.ndarray, half: int, min_gap: int) -> np.ndarray:
    """Candidate boundary frames: librosa's picker, or a local-max fallback."""
    if novelty.size == 0:
        return np.zeros(0, dtype=int)
    window = max(2, half // 2)

    librosa = _get_librosa()
    if librosa is not None:
        try:
            peaks = librosa.util.peak_pick(
                np.asarray(novelty, dtype=np.float64),
                pre_max=window, post_max=window,
                pre_avg=window, post_avg=window,
                delta=NOVELTY_FLOOR * 0.5, wait=max(1, min_gap),
            )
            return np.asarray(peaks, dtype=int)
        except Exception:
            pass

    peaks, _ = sps.find_peaks(novelty, distance=max(1, min_gap), height=NOVELTY_FLOOR * 0.5)
    return np.asarray(peaks, dtype=int)


def _contrast_db(band_db: np.ndarray, level_db: np.ndarray, frame: int, half: int) -> float:
    """How much the audio actually changes across a candidate boundary, in dB.

    Two terms, both in units anybody can argue with: the broadband level step,
    and the RMS change in tonal balance once the level step is taken out. A
    novelty peak that cannot show one of these is a peak in the noise.
    """
    n = level_db.shape[0]
    lo, hi = max(0, frame - half), min(n, frame + half)
    if frame - lo < 2 or hi - frame < 2:
        return 0.0

    before_level = float(np.mean(level_db[lo:frame]))
    after_level = float(np.mean(level_db[frame:hi]))
    level_step = abs(after_level - before_level)

    before_shape = np.mean(band_db[lo:frame], axis=0) - before_level
    after_shape = np.mean(band_db[frame:hi], axis=0) - after_level
    shape_step = float(np.sqrt(np.mean(np.square(after_shape - before_shape))))

    return _finite(math.hypot(level_step, shape_step), 0.0, 0.0, 120.0)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def _select_boundaries(
    novelty: np.ndarray,
    band_db: np.ndarray,
    level_db: np.ndarray,
    half: int,
    min_gap: int,
    max_boundaries: int,
    want_boundaries: int,
) -> List[int]:
    """Greedy, spacing-constrained boundary selection.

    Candidates are ranked by novelty weighted by measured contrast, so a strong
    peak that corresponds to nothing audible loses to a weaker one that does.
    Two passes: everything clearing the strict gate first, then — only if the
    track is long enough to expect more structure than that — the best of the
    near misses, down to `want_boundaries`.
    """
    n = level_db.shape[0]
    candidates = _pick_peaks(novelty, half, min_gap)
    if candidates.size == 0:
        return []

    # Never cut inside the first or last section's minimum length.
    candidates = candidates[(candidates >= min_gap) & (candidates <= n - min_gap)]
    if candidates.size == 0:
        return []

    scored: List[Tuple[float, int, float, float]] = []
    for frame in candidates.tolist():
        strength = float(novelty[frame])
        if strength < NOVELTY_FLOOR * 0.5:
            continue
        contrast = _contrast_db(band_db, level_db, frame, half)
        scored.append((strength * max(contrast, 0.05), frame, strength, contrast))
    scored.sort(key=lambda item: -item[0])

    chosen: List[int] = []

    def _take(strict: bool) -> None:
        floor = NOVELTY_FLOOR if strict else NOVELTY_FLOOR * 0.6
        contrast_floor = CONTRAST_DB if strict else RELAXED_CONTRAST_DB
        for _, frame, strength, contrast in scored:
            if len(chosen) >= max_boundaries:
                return
            if not strict and len(chosen) >= want_boundaries:
                return
            if strength < floor or contrast < contrast_floor:
                continue
            if any(abs(frame - other) < min_gap for other in chosen):
                continue
            chosen.append(frame)

    _take(strict=True)
    if len(chosen) < want_boundaries:
        _take(strict=False)

    return sorted(chosen)


def _cluster_segments(segment_features: np.ndarray) -> np.ndarray:
    """Group segments that sound like each other. Returns a label per segment.

    Average linkage on the same feature vectors the segmentation used, cut at an
    *absolute* distance. Cutting a dendrogram at a fraction of its own height is
    the usual heuristic and it fails exactly where it matters: on a track whose
    quiet break sits eight units away from its drop, half the tree height is
    ~3.8 and the build gets merged into the drop it was leading up to. The
    feature scale here is fixed by construction, so a real distance is available
    and is used. Capped at five groups, because a labeller with more categories
    than a song has parts is guessing.
    """
    n = segment_features.shape[0]
    if n <= 1:
        return np.zeros(max(n, 0), dtype=int)
    if n == 2:
        distance = float(np.linalg.norm(segment_features[0] - segment_features[1]))
        return np.array([0, 0], dtype=int) if distance < CLUSTER_T_MIN else np.array([0, 1])

    try:
        linkage = sch.linkage(segment_features, method="average", metric="euclidean")
        heights = np.asarray(linkage[:, 2], dtype=np.float64)
        threshold = clamp(float(np.max(heights)) * 0.5, CLUSTER_T_MIN, CLUSTER_T_MAX)
        labels = sch.fcluster(linkage, t=threshold, criterion="distance")
        if int(np.max(labels)) > MAX_CLUSTERS:
            labels = sch.fcluster(linkage, t=MAX_CLUSTERS, criterion="maxclust")
    except Exception:
        return np.arange(n, dtype=int)

    # Renumber to 0..k-1 in order of first appearance, so cluster ids read in
    # the order a listener meets them.
    order: Dict[int, int] = {}
    out = np.zeros(n, dtype=int)
    for i, raw in enumerate(labels.tolist()):
        if raw not in order:
            order[raw] = len(order)
        out[i] = order[raw]
    return out


# ---------------------------------------------------------------------------
# Per-section measurement
# ---------------------------------------------------------------------------


def _gated_lufs(power_slice: np.ndarray, sr: int) -> float:
    """BS.1770-4 gated loudness of one slice of the K-weighted power signal."""
    if power_slice.size == 0:
        return ABS_GATE_LUFS

    block_len = int(round(BLOCK_SEC * sr))
    block_hop = max(1, int(round(BLOCK_STEP_SEC * sr)))
    if power_slice.size >= block_len:
        blocks = frame_signal(np.ascontiguousarray(power_slice), block_len, block_hop).mean(axis=1)
    else:
        blocks = np.array([float(power_slice.mean())], dtype=np.float64)

    try:
        from .loudness import _gated_integrated  # same gate as the whole-file figure
        return _finite(_gated_integrated(blocks), ABS_GATE_LUFS, ABS_GATE_LUFS, 20.0)
    except Exception:
        pass

    # Fallback: the two-stage gate, written out. Only reached if the loudness
    # module's internals move.
    loud = _LUFS_OFFSET + 10.0 * np.log10(np.maximum(blocks, EPS))
    above = loud > ABS_GATE_LUFS
    if not np.any(above):
        return ABS_GATE_LUFS
    relative = _LUFS_OFFSET + 10.0 * math.log10(max(float(blocks[above].mean()), EPS)) - 10.0
    keep = above & (loud > relative)
    if not np.any(keep):
        keep = above
    return _finite(
        _LUFS_OFFSET + 10.0 * math.log10(max(float(blocks[keep].mean()), EPS)),
        ABS_GATE_LUFS, ABS_GATE_LUFS, 20.0,
    )


def _slice_moments(left: np.ndarray, right: np.ndarray) -> Tuple[float, float, float]:
    """DC-corrected (E[L²], E[R²], E[LR]) for one span, via BLAS dots.

    DC is removed arithmetically rather than by building a corrected copy: a
    six-minute section is 17 M samples per channel and the copy is pure waste.
    """
    n = float(max(left.size, 1))
    mu_l = float(left.mean()) if left.size else 0.0
    mu_r = float(right.mean()) if right.size else 0.0
    ll = float(np.dot(left, left)) / n - mu_l * mu_l
    rr = float(np.dot(right, right)) / n - mu_r * mu_r
    lr = float(np.dot(left, right)) / n - mu_l * mu_r
    return max(ll, 0.0), max(rr, 0.0), lr


def _slice_crest(left: np.ndarray, right: np.ndarray, ll: float, rr: float) -> float:
    """Peak-to-RMS in dB for one span, matching `dynamics.crest_factor_db`."""
    if left.size == 0:
        return 0.0
    mu_l = float(left.mean())
    mu_r = float(right.mean()) if right.size else 0.0
    peak = max(
        abs(float(left.max()) - mu_l), abs(float(left.min()) - mu_l),
        abs(float(right.max()) - mu_r) if right.size else 0.0,
        abs(float(right.min()) - mu_r) if right.size else 0.0,
    )
    mean_sq = 0.5 * (ll + rr)
    if mean_sq <= EPS or peak <= 0.0:
        return 0.0
    return _finite(amp_to_db(peak) - amp_to_db(math.sqrt(mean_sq)), 0.0, 0.0, 60.0)


def _slice_width(ll: float, rr: float, lr: float, is_mono: bool) -> float:
    """Side/Mid RMS ratio for one span, matching `stereo._pair_stats`."""
    if is_mono:
        return 0.0
    if (ll + rr) * 0.5 <= ENERGY_FLOOR:
        return 0.0
    mid_ms = max((ll + 2.0 * lr + rr) * 0.25, 0.0)
    side_ms = max((ll - 2.0 * lr + rr) * 0.25, 0.0)
    if mid_ms <= ENERGY_FLOOR:
        return WIDTH_MAX if side_ms > ENERGY_FLOOR else 0.0
    return clamp(_finite(math.sqrt(side_ms / mid_ms), 0.0), 0.0, WIDTH_MAX)


# ---------------------------------------------------------------------------
# Onsets (label evidence only)
# ---------------------------------------------------------------------------


def _onset_times(mono: np.ndarray, sr: int) -> np.ndarray:
    """Onset times in seconds. Best effort — labels use it, measurements do not."""
    if mono.size < sr // 4:
        return np.zeros(0, dtype=np.float64)

    step = max(1, sr // _ONSET_SR)
    try:
        y = sps.resample_poly(mono, 1, step) if step > 1 else mono
    except Exception:
        y, step = mono, 1
    work_sr = sr // step

    librosa = _get_librosa()
    if librosa is not None:
        try:
            strength = librosa.onset.onset_strength(
                y=np.asarray(y, dtype=np.float32), sr=work_sr, hop_length=_ONSET_HOP
            )
            frames = librosa.onset.onset_detect(
                onset_envelope=strength, sr=work_sr, hop_length=_ONSET_HOP, units="frames"
            )
            return np.asarray(
                librosa.frames_to_time(frames, sr=work_sr, hop_length=_ONSET_HOP),
                dtype=np.float64,
            )
        except Exception:
            pass

    # Fallback: peaks in the rectified, smoothed envelope's rise.
    try:
        win = max(8, work_sr // 200)
        n_blocks = y.size // win
        if n_blocks < 4:
            return np.zeros(0, dtype=np.float64)
        env = np.abs(y[: n_blocks * win]).reshape(n_blocks, win).max(axis=1)
        env_db = np.asarray(amp_to_db(env), dtype=np.float64)
        flux = np.maximum(np.diff(env_db, prepend=env_db[0]), 0.0)
        peaks, _ = sps.find_peaks(flux, height=max(3.0, float(np.percentile(flux, 90))),
                                  distance=max(1, int(0.05 * work_sr / win)))
        return peaks.astype(np.float64) * (win / float(work_sr))
    except Exception:
        return np.zeros(0, dtype=np.float64)


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------


def _reads_electronic(
    density: np.ndarray, band_db: Sequence[Dict[str, float]],
    crest: np.ndarray, peak_idx: int,
) -> bool:
    """Does the loudest section behave like a drop rather than a chorus?

    A crest factor under `_DROP_CREST_DB` in the loudest section is mandatory:
    "drop" is a word for a section that arrives as a wall, and a hook with 12 dB
    of crest left in it is a chorus whatever the genre. On top of that it needs
    either a low end that buries the presence band or a genuinely dense onset
    pattern.

    Onset density cannot carry this alone, which is worth writing down: measured
    on limited electronic material the detector finds *fewer* onsets in the drop
    (1.8/s) than in the sparse intro (2.7/s), because limiting is what removes
    the flux it keys on. Deliberately conservative in any case — "chorus" is the
    safer word to be wrong with, and a genre hint overrides all of this.
    """
    if peak_idx < 0 or peak_idx >= len(band_db):
        return False
    if float(crest[peak_idx]) > _DROP_CREST_DB:
        return False
    bands = band_db[peak_idx]
    low = 10.0 * math.log10(max(
        10.0 ** (bands.get("sub", -90.0) / 10.0)
        + 10.0 ** (bands.get("low_bass", -90.0) / 10.0), EPS
    ))
    presence = bands.get("presence", -90.0)
    dense = float(density[peak_idx]) >= 3.0 if density.size > peak_idx else False
    return bool((low - presence) >= _DROP_LOW_DOMINANCE_DB or dense)


def _assign_labels(
    clusters: np.ndarray,
    lufs: np.ndarray,
    peak_word: str,
) -> List[str]:
    """Human labels from measured character. A heuristic, and honest about it.

    Labels are assigned per *cluster*, so two passes of the same part always get
    the same name. The loudest recurring cluster is the hook. Clusters near the
    median level are the record's main body — verses. Clusters well under the
    median are breakdowns, which read as "bridge" anywhere but the ends of the
    file, where position supplies intro and outro instead. Anything the rules
    cannot justify stays "section N" rather than taking a name it has not
    earned.
    """
    n = int(lufs.size)
    if n <= 0:
        return []
    if n == 1:
        return ["section 1"]
    if n == 2:
        # Two spans is not a song form. Numbering them is the honest answer.
        return ["section 1", "section 2"]

    median = float(np.median(lufs))
    peak_lift = float(np.max(lufs)) - median
    n_clusters = int(np.max(clusters)) + 1 if clusters.size else 1

    # Uniform material: one cluster and no lift means there is no hook to name.
    if n_clusters <= 1 and peak_lift < 1.0:
        return [f"section {i + 1}" for i in range(n)]

    members: Dict[int, List[int]] = {}
    for i, c in enumerate(clusters.tolist()):
        members.setdefault(int(c), []).append(i)
    cluster_mean = {c: float(np.mean(lufs[idx])) for c, idx in members.items()}

    recurring = [c for c, idx in members.items() if len(idx) >= 2]
    if recurring:
        hook = max(recurring, key=lambda c: cluster_mean[c])
        if cluster_mean[hook] < median - 0.5:
            hook = int(clusters[int(np.argmax(lufs))])
    else:
        hook = int(clusters[int(np.argmax(lufs))])

    labels: Dict[int, str] = {}
    for i in members.get(hook, []):
        labels[i] = peak_word

    for c, idx in members.items():
        if c == hook:
            continue
        if cluster_mean[c] >= median - QUIET_CLUSTER_LU:
            # Sitting at the record's working level: this is a main part.
            for i in idx:
                labels[i] = "verse"
        else:
            # Well under it: a breakdown. At the ends of the file the positional
            # rules below will call the same thing an intro or an outro.
            for i in idx:
                if 0 < i < n - 1:
                    labels[i] = "bridge"

    # Position names the ends, but it must not break a cluster that genuinely
    # repeats. In a plain verse/chorus song the verses are always the quietest
    # cluster, and renaming the first one "intro" purely because it comes first
    # is the obvious way to get this wrong — so a section only takes a
    # positional name when it stands alone or belongs to a cluster that sits
    # well under the record's working level.
    def _positional(i: int) -> bool:
        c = int(clusters[i])
        return bool(
            labels.get(i) != peak_word
            and lufs[i] <= median - 0.5
            and (len(members.get(c, [])) == 1
                 or cluster_mean.get(c, median) < median - QUIET_CLUSTER_LU)
        )

    if _positional(0):
        labels[0] = "intro"
    if _positional(n - 1):
        labels[n - 1] = "outro"

    return [labels.get(i, f"section {i + 1}") for i in range(n)]


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def _build_notes(
    sections: List[Section],
    lufs: np.ndarray,
    audible: np.ndarray,
    peak_lift: float,
    spread: float,
    low_rel: np.ndarray,
    low_swing: float,
    core: np.ndarray,
    extra: List[str],
) -> List[str]:
    """Written observations. Every one of them names the number behind it."""
    notes: List[str] = list(extra)
    n = len(sections)
    if n <= 1:
        return notes

    loudest = sections[int(np.argmax(np.where(audible, lufs, -np.inf)))] if np.any(audible) \
        else sections[int(np.argmax(lufs))]
    quietest = sections[int(np.argmin(np.where(audible, lufs, np.inf)))] if np.any(audible) \
        else sections[int(np.argmin(lufs))]

    if n >= 3:
        if peak_lift < 1.0:
            notes.append(
                f"The loudest section is only {_num(peak_lift)} LU above the median — the "
                f"chorus is not lifting. Every part of this arrangement arrives at the same "
                f"level, so nothing reads as the payoff."
            )
        elif peak_lift < 2.0:
            notes.append(
                f"The loudest section sits {_num(peak_lift)} LU above the median. That is a "
                f"lift you can measure but barely hear; most records put 2-4 LU between the "
                f"verse and the hook."
            )
        elif peak_lift > 7.0:
            notes.append(
                f"The loudest section is {_num(peak_lift)} LU above the median — a big swing. "
                f"Check the quiet parts still read on a phone speaker."
            )
        else:
            notes.append(
                f"The loudest section ({loudest.label}, {_clock(loudest.t_start)}) sits "
                f"{_num(peak_lift)} LU above the median section: the arrangement has real lift."
            )

    if spread >= 1.0:
        notes.append(
            f"Loudest to quietest is {_num(spread)} LU — {loudest.label} at "
            f"{_clock(loudest.t_start)} ({_num(loudest.integrated_lufs)} LUFS) against "
            f"{quietest.label} at {_clock(quietest.t_start)} "
            f"({_num(quietest.integrated_lufs)} LUFS)."
        )

    # Sections that cluster together are the same part of the record played
    # twice. If they do not come back at the same level, that is a fader move
    # nobody intended, and no whole-file number will ever show it. Grouped by
    # the label rather than the raw cluster so the sentence names one part.
    by_label: Dict[str, List[int]] = {}
    for i, section in enumerate(sections):
        if audible[i] and not section.label.startswith("section "):
            by_label.setdefault(section.label, []).append(i)
    for label, idx in by_label.items():
        if len(idx) < 2:
            continue
        gap = float(np.max(lufs[idx]) - np.min(lufs[idx]))
        if gap < 1.5:
            continue
        high = int(idx[int(np.argmax(lufs[idx]))])
        low = int(idx[int(np.argmin(lufs[idx]))])
        notes.append(
            f"The same part comes back at a different level: the {label} at "
            f"{_clock(sections[high].t_start)} is {_num(gap)} LU louder than the one at "
            f"{_clock(sections[low].t_start)}. If that is not deliberate, it is a fader "
            f"that moved between them."
        )
        break

    core_idx = np.flatnonzero(core)
    if low_swing >= 3.0 and core_idx.size >= 2:
        weakest = int(core_idx[int(np.argmin(low_rel[core_idx]))])
        strongest = int(core_idx[int(np.argmax(low_rel[core_idx]))])
        # The label matters to how this reads. A thin intro or outro is what
        # arrangements do, and a note that does not say so invites whoever
        # reads it next to call an ordinary arrangement a fault.
        bookend = sections[weakest].label in ("intro", "outro")
        notes.append(
            f"Low end swings {_num(low_swing)} dB across the sections carrying this record: "
            f"sub and low bass are strongest in {sections[strongest].label} at "
            f"{_clock(sections[strongest].t_start)} and weakest in {sections[weakest].label} "
            f"at {_clock(sections[weakest].t_start)}, measured against each section's own "
            f"level."
            + (f" An {sections[weakest].label} carrying less underneath it is ordinary "
               f"arrangement rather than a fault." if bookend else
               " If that was not deliberate, the bottom is falling out of one part of "
               "the arrangement.")
        )
    elif n >= 3 and core_idx.size >= 2 and low_swing < 1.0:
        notes.append(
            f"Low-end balance holds to {_num(low_swing)} dB across every section that carries "
            f"the record — the bottom end is consistent from one part to the next."
        )

    widths = np.array([s.stereo_width for s in sections], dtype=np.float64)
    if n >= 3 and widths.size and float(np.max(widths)) - float(np.min(widths)) >= 0.25:
        widest = sections[int(np.argmax(widths))]
        narrowest = sections[int(np.argmin(widths))]
        notes.append(
            f"Stereo width moves from {_num(narrowest.stereo_width, 2)} in "
            f"{narrowest.label} to {_num(widest.stereo_width, 2)} in {widest.label} — the "
            f"image opens and closes with the arrangement."
        )

    return notes


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------


def _one_section(buf: AudioBuffer, notes: List[str]) -> SectionAnalysis:
    """The whole file as a single section. Always available, never zero sections."""
    try:
        section = _measure_span(buf, 0.0, float(buf.duration), 0, "section 1", None, None)
    except Exception:
        section = Section(
            index=0, label="section 1", t_start=0.0,
            t_end=round(_finite(buf.duration, 0.0, 0.0, 1e5), 3),
            integrated_lufs=ABS_GATE_LUFS, crest_factor_db=0.0, stereo_width=0.0,
            band_levels_db={name: -90.0 for name in MACRO_BAND_ORDER},
        )
    section.is_loudest = True
    section.is_quietest = True
    return SectionAnalysis(
        available=True, sections=[section], loudness_spread_lu=0.0,
        peak_lift_db=0.0, low_end_swing_db=0.0, notes=notes,
    )


def _measure_span(
    buf: AudioBuffer,
    t_start: float,
    t_end: float,
    index: int,
    label: str,
    kpower: Optional[np.ndarray],
    band_db: Optional[np.ndarray],
) -> Section:
    """Measure one span. `kpower`/`band_db` are precomputed where available."""
    sr = int(buf.sr)
    n = int(buf.n_samples)
    s0 = int(max(0, min(n, round(t_start * sr))))
    s1 = int(max(s0 + 1, min(n, round(t_end * sr))))

    left = buf.left[s0:s1]
    right = buf.right[s0:s1]
    ll, rr, lr = _slice_moments(left, right)

    if kpower is not None:
        lufs = _gated_lufs(kpower[s0:s1], sr)
    else:
        from .loudness import k_weight
        power = np.square(k_weight(left, sr)) + np.square(k_weight(right, sr))
        lufs = _gated_lufs(power, sr)

    if band_db is not None:
        levels = {
            name: round(_finite(band_db[i], -120.0, -120.0, 40.0), 2)
            for i, name in enumerate(MACRO_BAND_ORDER)
        }
    else:
        levels = {name: -120.0 for name in MACRO_BAND_ORDER}

    return Section(
        index=int(index),
        label=str(label),
        t_start=round(_finite(t_start, 0.0, 0.0, 1e5), 3),
        t_end=round(_finite(t_end, 0.0, 0.0, 1e5), 3),
        integrated_lufs=round(_finite(lufs, ABS_GATE_LUFS, ABS_GATE_LUFS, 20.0), 2),
        crest_factor_db=round(_slice_crest(left, right, ll, rr), 2),
        stereo_width=round(_slice_width(ll, rr, lr, buf.is_mono), 3),
        band_levels_db=levels,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def measure_sections(buf: AudioBuffer, genre: str = "") -> SectionAnalysis:
    """Segment the track structurally and measure every section on its own.

    `genre` is an optional hint and changes exactly one thing: whether the
    loudest recurring cluster is called a "drop" or a "chorus". Omit it and the
    module decides from onset density, low-end dominance and crest factor.

    Never raises, never returns zero sections. A file under 20 s, or one the
    segmenter cannot make sense of, comes back as a single section covering the
    whole thing with a note saying why.
    """
    try:
        return _measure_sections(buf, genre)
    except Exception as exc:  # pragma: no cover - the point is that it never escapes
        return _one_section(buf, [
            f"Structural segmentation could not be completed on this file "
            f"({type(exc).__name__}); the whole track is reported as one section."
        ])


def _measure_sections(buf: AudioBuffer, genre: str = "") -> SectionAnalysis:
    sr = int(buf.sr)
    duration = float(_finite(buf.duration, 0.0, 0.0, 1e5))
    notes: List[str] = []

    if int(buf.n_samples) <= 0 or duration <= 0.0:
        return SectionAnalysis(
            available=False,
            notes=["There is no audio in this file to segment."],
        )

    # --- K-weighted power, once, for every section's LUFS -------------------
    from .loudness import k_weight  # the K-weighting is not reimplemented here

    kpower = np.square(k_weight(buf.left, sr))
    kpower += np.square(k_weight(buf.right, sr))
    kpower = np.ascontiguousarray(kpower)

    if duration < MIN_SEGMENTABLE_SEC:
        # Band levels only — there is nothing to segment, so the MFCC/chroma
        # front end (and librosa's import cost) is skipped entirely.
        freqs, times, power = stft_power(np.ascontiguousarray(buf.mono), sr)
        band_db = _band_levels(freqs, power, sr)[0].mean(axis=0) if times.size else None
        section = _measure_span(buf, 0.0, duration, 0, "section 1", kpower, band_db)
        section.is_loudest = True
        section.is_quietest = True
        return SectionAnalysis(
            available=True, sections=[section],
            loudness_spread_lu=0.0, peak_lift_db=0.0, low_end_swing_db=0.0,
            notes=[
                f"This file is {_num(duration)} s — too short to segment, so the whole "
                f"track is measured as one section."
            ],
        )

    # --- frame grid ---------------------------------------------------------
    mono = np.ascontiguousarray(buf.mono)
    mono = mono - float(np.mean(mono))          # DC is not sub-bass
    freqs, times, power = stft_power(mono, sr)
    if times.size < 8:
        return _one_section(buf, [
            "There were not enough analysis frames in this file to look for structure; "
            "the whole track is reported as one section."
        ])

    band_db, level_db = _band_levels(freqs, power, sr)
    features, used_librosa = _frame_matrix(freqs, power, sr, band_db, level_db)
    del power
    if not used_librosa:
        notes.append(
            "librosa was unavailable, so segmentation used the built-in cepstral and "
            "chroma front end. Boundaries may be placed a little less precisely."
        )

    frame_hop = float(times[1] - times[0]) if times.size > 1 else HOP_SEC
    frame_hop = max(frame_hop, 1e-3)
    min_section_sec = clamp(duration / MIN_SECTION_DIVISOR, MIN_SECTION_FLOOR_SEC,
                            MIN_SECTION_CEIL_SEC)
    min_gap = max(2, int(round(min_section_sec / frame_hop)))
    half = max(3, int(round(KERNEL_HALF_SEC / frame_hop)))
    half = min(half, max(3, times.size // 4))

    max_sections = int(clamp(math.floor(duration / min_section_sec), 1.0, float(MAX_SECTIONS)))
    # 4 is the low end of "a song has parts"; below ~100 s there is no room for
    # four sections that are each long enough to measure.
    want_sections = 4 if duration >= 100.0 else (2 if duration >= 45.0 else 1)
    want_sections = min(want_sections, max_sections)

    # --- novelty -> boundaries ---------------------------------------------
    similarity = _self_similarity(features)
    novelty = _foote_novelty(similarity, half)
    del similarity

    boundaries: List[int] = []
    if max_sections > 1:
        boundaries = _select_boundaries(
            novelty, band_db, level_db, half, min_gap,
            max_boundaries=max_sections - 1,
            want_boundaries=max(0, want_sections - 1),
        )

    edges = [0] + [int(b) for b in boundaries] + [int(times.size)]
    spans: List[Tuple[float, float, int, int]] = []
    for i in range(len(edges) - 1):
        f0, f1 = edges[i], edges[i + 1]
        t0 = 0.0 if i == 0 else float(times[f0])
        t1 = duration if i == len(edges) - 2 else float(times[f1])
        spans.append((t0, t1, f0, f1))

    n = len(spans)

    # --- per-section measurement -------------------------------------------
    # Onsets are evidence for one cosmetic decision (drop vs chorus) and cost
    # ~0.5 s on a five-minute track, so they are only measured when no genre
    # hint arrived to make that decision for us.
    hint = str(genre or "").strip().lower().replace(" ", "_").replace("-", "_")
    onsets = np.zeros(0, dtype=np.float64) if hint else _onset_times(mono, sr)
    del mono

    raw: List[Section] = []
    seg_features = np.zeros((n, features.shape[1]), dtype=np.float64)
    density = np.zeros(n, dtype=np.float64)
    for i, (t0, t1, f0, f1) in enumerate(spans):
        f1 = max(f1, f0 + 1)
        seg_features[i] = features[f0:f1].mean(axis=0)
        raw.append(_measure_span(buf, t0, t1, i, f"section {i + 1}", kpower,
                                 band_db[f0:f1].mean(axis=0)))
        span = max(t1 - t0, 1e-3)
        if onsets.size:
            density[i] = float(np.count_nonzero((onsets >= t0) & (onsets < t1))) / span

    lufs = np.array([s.integrated_lufs for s in raw], dtype=np.float64)
    crest = np.array([s.crest_factor_db for s in raw], dtype=np.float64)

    # --- labels -------------------------------------------------------------
    clusters = _cluster_segments(seg_features)
    peak_idx = int(np.argmax(lufs))
    use_drop = hint in _DROP_GENRES or (
        not hint and _reads_electronic(density, [s.band_levels_db for s in raw],
                                       crest, peak_idx)
    )
    labels = _assign_labels(clusters, lufs, "drop" if use_drop else "chorus")
    for section, label in zip(raw, labels):
        section.label = label

    # --- summary statistics -------------------------------------------------
    # The flags describe every section, including a silent tail — it really is
    # the quietest part of the file. The *statistics* exclude silence, because
    # "loudest to quietest is 53 LU" measured against a digital-black outro is
    # arithmetic, not information.
    raw[int(np.argmax(lufs))].is_loudest = True
    raw[int(np.argmin(lufs))].is_quietest = True

    audible = lufs > AUDIBLE_FLOOR_LUFS
    if not np.any(audible):
        audible = np.ones(n, dtype=bool)
    silent = int(n - np.count_nonzero(audible))

    usable = lufs[audible]
    loudest_lufs = float(np.max(usable))
    quietest_lufs = float(np.min(usable))
    spread = _finite(loudest_lufs - quietest_lufs, 0.0, 0.0, 200.0)
    peak_lift = _finite(loudest_lufs - float(np.median(usable)), 0.0, 0.0, 200.0)

    # Low-end swing is the sub+low-bass *share* of each section, so a quiet
    # intro does not read as a collapsing bottom end — what it catches is the
    # balance change, the drop whose sub disappears while everything else stays
    # loud. It is taken over the record's core (everything within LOW_CORE_LU of
    # the loudest section) for the same reason: a bass-free intro is an
    # arrangement, and reporting it here would just restate loudness_spread_lu.
    low_rel = np.zeros(n, dtype=np.float64)
    for i, section in enumerate(raw):
        sub = 10.0 ** (section.band_levels_db.get("sub", -120.0) / 10.0)
        low_bass = 10.0 ** (section.band_levels_db.get("low_bass", -120.0) / 10.0)
        total = sum(10.0 ** (v / 10.0) for v in section.band_levels_db.values())
        low_rel[i] = clamp(
            10.0 * math.log10(max(sub + low_bass, EPS) / max(total, EPS)),
            LOW_SHARE_FLOOR_DB, 0.0,
        )
    core = audible & (lufs >= loudest_lufs - LOW_CORE_LU)
    low_swing = 0.0
    if int(np.count_nonzero(core)) >= 2:
        pool = low_rel[core]
        low_swing = _finite(float(np.max(pool) - np.min(pool)), 0.0, 0.0, 120.0)

    if n == 1:
        notes.append(
            "No structural boundary in this track survived the contrast test: its level "
            "and tonal balance hold steady from start to finish, so it is reported as a "
            "single section rather than being cut into invented parts."
        )
    elif n == 2:
        notes.append(
            "Only one structural boundary was found, so the two spans are numbered rather "
            "than named — two parts is not enough to infer a song form from."
        )
    if silent:
        quiet_spans = ", ".join(
            f"{_clock(raw[i].t_start)}-{_clock(raw[i].t_end)}"
            for i in np.flatnonzero(~audible).tolist()
        )
        notes.append(
            f"{'One section' if silent == 1 else f'{silent} sections'} measured below "
            f"{_num(AUDIBLE_FLOOR_LUFS, 0)} LUFS ({quiet_spans}) — that is silence, and it "
            f"is left out of the spread and lift figures."
        )

    notes = _build_notes(raw, lufs, audible, peak_lift, spread,
                         low_rel, low_swing, core, notes)

    return SectionAnalysis(
        available=True,
        sections=raw,
        loudness_spread_lu=round(spread, 2),
        peak_lift_db=round(peak_lift, 2),
        low_end_swing_db=round(low_swing, 2),
        notes=notes,
    )
