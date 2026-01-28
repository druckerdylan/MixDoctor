"""Audio analysis module for extracting mix metrics."""

import os
import tempfile
from typing import Tuple
import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
from pydub import AudioSegment
from scipy import signal

from typing import Dict, List
from schemas import AudioMetrics


# Frequency bands for spectrum analysis (Hz)
FREQUENCY_BANDS = {
    "sub": (20, 60),
    "low": (60, 250),
    "low_mid": (250, 500),
    "mid": (500, 2000),
    "presence": (2000, 6000),
    "air": (6000, 20000),
}


def convert_to_wav(input_path: str) -> str:
    """Convert audio file to WAV format if needed."""
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".wav":
        return input_path

    # Use pydub to convert
    if ext == ".mp3":
        audio = AudioSegment.from_mp3(input_path)
    elif ext == ".m4a":
        audio = AudioSegment.from_file(input_path, format="m4a")
    elif ext == ".flac":
        audio = AudioSegment.from_file(input_path, format="flac")
    elif ext == ".aiff" or ext == ".aif":
        audio = AudioSegment.from_file(input_path, format="aiff")
    else:
        raise ValueError(f"Unsupported audio format: {ext}")

    # Export to temp WAV file
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(temp_wav.name, format="wav")
    return temp_wav.name


def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """Load audio file and return samples and sample rate."""
    wav_path = convert_to_wav(file_path)

    # Load with soundfile for accurate sample values
    audio, sr = sf.read(wav_path)

    # Clean up temp file if we created one
    if wav_path != file_path and os.path.exists(wav_path):
        os.unlink(wav_path)

    return audio, sr


def analyze_loudness(audio: np.ndarray, sr: int) -> dict:
    """Analyze loudness metrics using pyloudnorm."""
    meter = pyln.Meter(sr)

    # Ensure stereo for consistent analysis
    if audio.ndim == 1:
        audio_stereo = np.column_stack([audio, audio])
    else:
        audio_stereo = audio

    # Integrated loudness
    integrated_lufs = meter.integrated_loudness(audio_stereo)

    # Short-term loudness (3-second windows)
    block_size = int(3 * sr)
    short_term_values = []
    for i in range(0, len(audio_stereo) - block_size, int(0.1 * sr)):
        block = audio_stereo[i:i + block_size]
        try:
            st_lufs = meter.integrated_loudness(block)
            if not np.isinf(st_lufs):
                short_term_values.append(st_lufs)
        except Exception:
            pass

    short_term_max = max(short_term_values) if short_term_values else integrated_lufs

    # True peak
    if audio.ndim == 1:
        true_peak = 20 * np.log10(np.max(np.abs(audio)) + 1e-10)
    else:
        true_peak = 20 * np.log10(np.max(np.abs(audio)) + 1e-10)

    # Loudness range (simplified)
    if short_term_values:
        lra = np.percentile(short_term_values, 95) - np.percentile(short_term_values, 10)
    else:
        lra = 0.0

    return {
        "integrated_lufs": round(integrated_lufs, 1),
        "short_term_lufs_max": round(short_term_max, 1),
        "true_peak_dbfs": round(true_peak, 1),
        "loudness_range": round(lra, 1),
    }


def analyze_spectrum(audio: np.ndarray, sr: int) -> dict:
    """Analyze frequency spectrum energy distribution."""
    # Convert to mono for spectrum analysis
    if audio.ndim > 1:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio

    # Compute power spectrum
    n_fft = 4096
    freqs = np.fft.rfftfreq(n_fft, 1/sr)

    # Compute STFT and average power
    hop_length = n_fft // 4
    stft = librosa.stft(mono, n_fft=n_fft, hop_length=hop_length)
    power = np.mean(np.abs(stft) ** 2, axis=1)

    # Calculate energy per band
    spectrum = {}
    total_power = np.sum(power)

    for band_name, (low_freq, high_freq) in FREQUENCY_BANDS.items():
        band_mask = (freqs >= low_freq) & (freqs < high_freq)
        band_power = np.sum(power[band_mask])
        # Convert to dB relative to total
        if total_power > 0 and band_power > 0:
            spectrum[band_name] = round(10 * np.log10(band_power / total_power), 1)
        else:
            spectrum[band_name] = -60.0

    return spectrum


def analyze_stereo(audio: np.ndarray) -> dict:
    """Analyze stereo field characteristics."""
    if audio.ndim == 1:
        return {"stereo_correlation": 1.0, "stereo_width": 0.0}

    left = audio[:, 0]
    right = audio[:, 1]

    # Stereo correlation (-1 to 1)
    correlation = np.corrcoef(left, right)[0, 1]

    # Stereo width (based on side/mid ratio)
    mid = (left + right) / 2
    side = (left - right) / 2

    mid_rms = np.sqrt(np.mean(mid ** 2))
    side_rms = np.sqrt(np.mean(side ** 2))

    if mid_rms > 0:
        width = side_rms / mid_rms
    else:
        width = 0.0

    return {
        "stereo_correlation": round(correlation, 3),
        "stereo_width": round(min(width, 2.0), 3),  # Cap at 2.0
    }


def analyze_dynamics(audio: np.ndarray) -> dict:
    """Analyze dynamic range and crest factor."""
    # Convert to mono if stereo
    if audio.ndim > 1:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio

    # RMS level
    rms = np.sqrt(np.mean(mono ** 2))
    rms_db = 20 * np.log10(rms + 1e-10)

    # Peak level
    peak = np.max(np.abs(mono))

    # Crest factor (peak to RMS ratio)
    crest_factor_db = 20 * np.log10(peak / (rms + 1e-10))

    # Dynamic range (simplified - based on windowed RMS)
    window_size = int(0.1 * 44100)  # 100ms windows
    if len(mono) > window_size:
        n_windows = len(mono) // window_size
        windowed_rms = []
        for i in range(n_windows):
            window = mono[i * window_size:(i + 1) * window_size]
            w_rms = np.sqrt(np.mean(window ** 2))
            if w_rms > 1e-10:
                windowed_rms.append(20 * np.log10(w_rms))

        if windowed_rms:
            dynamic_range = np.percentile(windowed_rms, 95) - np.percentile(windowed_rms, 5)
        else:
            dynamic_range = 0.0
    else:
        dynamic_range = 0.0

    return {
        "crest_factor_db": round(crest_factor_db, 1),
        "rms_db": round(rms_db, 1),
        "dynamic_range_db": round(dynamic_range, 1),
    }


def analyze_clipping(audio: np.ndarray) -> dict:
    """Analyze clipping in audio samples.

    Detects samples at or near max value and identifies consecutive clipped samples
    which indicate hard clipping vs occasional peaks.
    """
    # Convert to mono if stereo for total sample count
    if audio.ndim > 1:
        # Analyze both channels separately for clipping
        total_samples = audio.shape[0] * audio.shape[1]
        flat_audio = audio.flatten()
    else:
        total_samples = len(audio)
        flat_audio = audio

    # Threshold for considering a sample as clipped (0.99 of max value)
    clip_threshold = 0.99

    # Count clipped samples
    clipped_mask = np.abs(flat_audio) >= clip_threshold
    clipped_samples = int(np.sum(clipped_mask))
    clip_percentage = (clipped_samples / total_samples) * 100 if total_samples > 0 else 0.0

    # Find consecutive clipped samples (indicates hard clipping)
    max_consecutive = 0
    current_consecutive = 0

    for is_clipped in clipped_mask:
        if is_clipped:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0

    # Determine severity
    has_clipping = clipped_samples > 0

    if clipped_samples == 0:
        severity = "good"
    elif clip_percentage < 0.01 and max_consecutive < 10:
        # Very minor clipping, occasional peaks
        severity = "good"
    elif clip_percentage < 0.1 and max_consecutive < 50:
        # Some clipping but not severe
        severity = "warning"
    else:
        # Significant clipping
        severity = "critical"

    return {
        "clipped_samples": clipped_samples,
        "clip_percentage": round(clip_percentage, 4),
        "max_consecutive_clips": max_consecutive,
        "has_clipping": has_clipping,
        "clipping_severity": severity,
    }


def analyze_phase(audio: np.ndarray) -> dict:
    """Analyze phase and mono compatibility.

    Calculates stereo correlation and mono energy loss to determine
    phase issues that could cause problems on mono playback systems.
    """
    if audio.ndim == 1:
        # Mono audio - perfect mono compatibility
        return {
            "mono_compatible": True,
            "phase_status": "good",
            "mono_energy_loss_db": 0.0,
        }

    left = audio[:, 0]
    right = audio[:, 1]

    # Stereo correlation (-1 to 1)
    correlation = np.corrcoef(left, right)[0, 1]

    # Calculate mono energy loss
    # Compare energy of stereo sum vs original
    mid = (left + right) / 2

    # RMS of original stereo (average of both channels)
    stereo_rms = np.sqrt((np.mean(left ** 2) + np.mean(right ** 2)) / 2)
    mono_rms = np.sqrt(np.mean(mid ** 2))

    # Calculate energy loss in dB
    if stereo_rms > 1e-10 and mono_rms > 1e-10:
        mono_energy_loss_db = 20 * np.log10(mono_rms / stereo_rms)
    else:
        mono_energy_loss_db = 0.0

    # Determine phase status based on correlation thresholds
    if correlation > 0.8:
        phase_status = "good"
        mono_compatible = True
    elif correlation > 0.5:
        phase_status = "acceptable"
        mono_compatible = True
    elif correlation > 0.3:
        phase_status = "warning"
        mono_compatible = False
    else:
        phase_status = "critical"
        mono_compatible = False

    return {
        "mono_compatible": mono_compatible,
        "phase_status": phase_status,
        "mono_energy_loss_db": round(mono_energy_loss_db, 1),
    }


def analyze_transients(audio: np.ndarray, sr: int) -> dict:
    """Analyze transients and rhythm characteristics."""
    # Convert to mono
    if audio.ndim > 1:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio

    # Onset detection
    onset_env = librosa.onset.onset_strength(y=mono, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)

    duration = len(mono) / sr
    onset_density = len(onsets) / duration if duration > 0 else 0

    # Tempo estimation
    tempo, _ = librosa.beat.beat_track(y=mono, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0])

    return {
        "onset_density": round(onset_density, 2),
        "estimated_tempo": round(tempo, 1),
    }


def analyze_audio(file_path: str) -> AudioMetrics:
    """Perform complete audio analysis and return metrics."""
    # Load audio
    audio, sr = load_audio(file_path)

    # Run all analyses
    loudness = analyze_loudness(audio, sr)
    spectrum = analyze_spectrum(audio, sr)
    stereo = analyze_stereo(audio)
    dynamics = analyze_dynamics(audio)
    transients = analyze_transients(audio, sr)
    clipping = analyze_clipping(audio)
    phase = analyze_phase(audio)

    # Calculate duration
    duration = len(audio) / sr

    return AudioMetrics(
        integrated_lufs=loudness["integrated_lufs"],
        short_term_lufs_max=loudness["short_term_lufs_max"],
        true_peak_dbfs=loudness["true_peak_dbfs"],
        loudness_range=loudness["loudness_range"],
        spectrum=spectrum,
        stereo_correlation=stereo["stereo_correlation"],
        stereo_width=stereo["stereo_width"],
        crest_factor_db=dynamics["crest_factor_db"],
        rms_db=dynamics["rms_db"],
        dynamic_range_db=dynamics["dynamic_range_db"],
        onset_density=transients["onset_density"],
        estimated_tempo=transients["estimated_tempo"],
        duration_seconds=round(duration, 2),
        sample_rate=sr,
        # Clipping metrics
        clipped_samples=clipping["clipped_samples"],
        clip_percentage=clipping["clip_percentage"],
        max_consecutive_clips=clipping["max_consecutive_clips"],
        has_clipping=clipping["has_clipping"],
        clipping_severity=clipping["clipping_severity"],
        # Phase metrics
        mono_compatible=phase["mono_compatible"],
        phase_status=phase["phase_status"],
        mono_energy_loss_db=phase["mono_energy_loss_db"],
    )


def analyze_stem_interactions(
    stems: Dict[str, AudioMetrics]
) -> Dict[str, any]:
    """
    Analyze interactions between stems.
    Returns data about frequency masking, phase issues, and balance.
    """
    interactions = []
    stem_names = list(stems.keys())

    # Frequency masking analysis - compare spectrum overlap between stem pairs
    for i, stem_a_name in enumerate(stem_names):
        for stem_b_name in stem_names[i + 1:]:
            stem_a = stems[stem_a_name]
            stem_b = stems[stem_b_name]

            # Check for frequency overlap/masking in each band
            masking_bands = []
            for band in ['sub', 'low', 'low_mid', 'mid', 'presence', 'air']:
                a_energy = stem_a.spectrum.get(band, -60)
                b_energy = stem_b.spectrum.get(band, -60)

                # Both stems have significant energy in this band (potential masking)
                if a_energy > -20 and b_energy > -20:
                    # Closer values = more masking potential
                    diff = abs(a_energy - b_energy)
                    if diff < 3:  # Similar levels = high masking potential
                        masking_bands.append(band)

            if masking_bands:
                severity = 'critical' if len(masking_bands) >= 3 else 'warning' if len(masking_bands) >= 2 else 'info'
                interactions.append({
                    'stem_a': stem_a_name,
                    'stem_b': stem_b_name,
                    'interaction_type': 'frequency_masking',
                    'severity': severity,
                    'bands': masking_bands,
                })

    # Phase analysis - check stereo correlation between stems
    phase_issues = []
    for stem_name, metrics in stems.items():
        if metrics.stereo_correlation < 0.5:
            phase_issues.append({
                'stem': stem_name,
                'correlation': metrics.stereo_correlation,
            })

    # Balance analysis - check relative loudness
    lufs_values = {name: m.integrated_lufs for name, m in stems.items()}
    avg_lufs = sum(lufs_values.values()) / len(lufs_values) if lufs_values else -20
    balance_issues = []
    for name, lufs in lufs_values.items():
        delta = lufs - avg_lufs
        if abs(delta) > 6:
            balance_issues.append({
                'stem': name,
                'lufs': lufs,
                'delta_from_avg': round(delta, 1),
            })

    return {
        'masking_interactions': interactions,
        'phase_issues': phase_issues,
        'balance_issues': balance_issues,
    }
