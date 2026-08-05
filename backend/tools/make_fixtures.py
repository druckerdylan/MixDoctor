"""Regenerate the test fixtures in `backend/testdata/`.

The fixtures are ~58 MB of WAV, which does not belong in git — it would sit in
every clone forever and be pulled on every deploy build. They are fully
synthetic, so the generator is the artefact worth committing instead.

    python tools/make_fixtures.py

Two families come out of this:

**Defect fixtures** (`mix_*.wav`) carry known, deliberate faults so a detector
can be checked against ground truth rather than against a vibe. `mix_problem`
in particular stacks a 250/310 Hz mud pair, a 3.2 kHz tone, and a kick and bass
sharing a ~55 Hz fundamental, then drives the lot into tanh saturation.

**Reference fixtures** (`reference_*.wav`) are the calibration ground truth.
Each is noise shaped to its genre's target curve and then run through a
synthetic mastering chain in the order a real master is built:

    spectrum -> arrangement dynamics -> limit until crest hits the genre figure
    -> peak set to -1 dBFS

Loudness is an *output* of that chain, not a knob, so it lands inside the genre
window on its own. That is what makes these usable as ground truth: they are
correct on spectrum, dynamics, loudness and peak at the same time. An earlier
cut fitted spectrum only, rendered everything near -15 LUFS, and the trap
reference scored D+ for being 6 LU too quiet — a fixture defect that looked
exactly like an analyser defect.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy import signal as sps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.core import ANALYSIS_SR, THIRD_OCTAVE_CENTERS  # noqa: E402
from analysis import targets as T  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "testdata")
CEILING = 10 ** (-1.0 / 20)  # -1 dBFS


# ---------------------------------------------------------------------------
# Defect fixtures
# ---------------------------------------------------------------------------


def _defect_stereo(sr: int = 44_100, dur: float = 20.0):
    """A deliberately broken 120 bpm loop, returned pre-limiting."""
    n = int(sr * dur)
    t = np.arange(n) / sr
    rng = np.random.default_rng(7)
    spb = 60.0 / 120.0
    beats = np.arange(0, dur, spb)
    eighths = np.arange(0, dur, spb / 2)

    def hits(times, decay):
        e = np.zeros(n)
        for tt in times:
            i = int(tt * sr)
            length = min(int(decay * sr), n - i)
            if length > 0:
                e[i : i + length] += np.exp(-np.arange(length) / (0.004 * sr))
        return np.clip(e, 0, 4)

    # Kick: 120 -> 50 Hz sweep.
    kick = np.zeros(n)
    for b in beats:
        i = int(b * sr)
        length = min(int(0.35 * sr), n - i)
        lt = np.arange(length) / sr
        f = 50 + 90 * np.exp(-lt / 0.03)
        kick[i : i + length] += np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-lt / 0.10)

    # Bass deliberately sharing the kick's fundamental — the collision under test.
    bass = np.zeros(n)
    for k, b in enumerate(beats):
        f = [55, 55, 73.4, 65.4][k % 4]
        i = int(b * sr)
        length = min(int(spb * sr), n - i)
        lt = np.arange(length) / sr
        bass[i : i + length] += (
            np.sin(2 * np.pi * f * lt) * 0.8 + 0.3 * np.sin(2 * np.pi * 2 * f * lt)
        ) * np.exp(-lt / 0.5)

    snare = sps.lfilter(*sps.butter(2, [180 / (sr / 2), 7000 / (sr / 2)], "band"),
                        hits(beats[1::2], 0.2) * rng.normal(0, 1, n))
    hats = sps.lfilter(*sps.butter(4, 7000 / (sr / 2), "high"),
                       hits(eighths, 0.06) * rng.normal(0, 1, n))

    pad = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
              for f in (130.8, 164.8, 196.0, 261.6, 329.6))
    mud = np.sin(2 * np.pi * 250 * t) * 0.9 + np.sin(2 * np.pi * 310 * t) * 0.7
    harsh = np.sin(2 * np.pi * 3200 * t) * 0.35 * (1 + 0.5 * np.sin(2 * np.pi * 0.3 * t))

    voc = sum(a * np.sin(2 * np.pi * f * t + 0.4 * np.sin(2 * np.pi * 4.5 * t))
              for f, a in ((220, 1.0), (440, 0.5), (880, 0.35), (2600, 0.22), (5500, 0.16)))
    voc *= 0.5 + 0.5 * np.sin(2 * np.pi * 0.8 * t)

    def at(x, db):
        return x / (np.max(np.abs(x)) + 1e-9) * 10 ** (db / 20)

    left = (at(kick, -6) + at(bass, -8) + at(snare, -11) + at(hats, -20)
            + at(pad, -14) + at(mud, -13) + at(harsh, -24) + at(voc, -19))
    right = (at(kick, -6) + at(bass, -8) + at(snare, -11) + at(np.roll(hats, 700), -20)
             + at(np.roll(pad, 1100), -14) + at(mud, -13) + at(np.roll(harsh, 300), -24)
             + at(voc, -19))

    st = np.column_stack([left, right])
    return st / (np.max(np.abs(st)) + 1e-9) * 0.98, sr


def write_defect_fixtures() -> None:
    st, sr = _defect_stereo()

    # Hot-limited and clipped: the "what is wrong with my master" case.
    sf.write(f"{OUT_DIR}/mix_problem.wav", np.tanh(st * 3.2) * 0.999, sr, subtype="PCM_24")
    # Same material with headroom intact.
    sf.write(f"{OUT_DIR}/mix_clean.wav", st * 0.45, sr, subtype="PCM_24")
    # Right channel inverted — must trip the polarity check.
    sf.write(f"{OUT_DIR}/mix_phase.wav",
             np.column_stack([st[:, 0], -0.85 * st[:, 1] + 0.15 * st[:, 0]]) * 0.5,
             sr, subtype="PCM_24")
    sf.write(f"{OUT_DIR}/mix_mono.wav", st.mean(axis=1) * 0.5, sr, subtype="PCM_24")
    # Edge cases: too short to frame, and no signal at all.
    sf.write(f"{OUT_DIR}/mix_short.wav", st[: int(sr * 1.2)] * 0.5, sr, subtype="PCM_24")
    sf.write(f"{OUT_DIR}/mix_silent.wav", np.zeros((sr * 2, 2)) + 1e-7, sr, subtype="PCM_24")

    for name in ("mix_problem", "mix_clean", "mix_phase", "mix_mono", "mix_short", "mix_silent"):
        path = f"{OUT_DIR}/{name}.wav"
        print(f"  {name:16s} {os.path.getsize(path) / 1e6:5.1f} MB")


# ---------------------------------------------------------------------------
# Reference fixtures
# ---------------------------------------------------------------------------


def _shaped_noise(genre: str, seed: int, n: int, sr: int) -> np.ndarray:
    """Noise whose 1/3-octave spectrum matches the genre target curve."""
    rng = np.random.default_rng(seed)
    spec = np.fft.rfft(rng.normal(0, 1, n))
    freqs = np.fft.rfftfreq(n, 1 / sr)

    curve = T.target_curve(genre, THIRD_OCTAVE_CENTERS)
    gain = np.interp(np.log10(np.maximum(freqs, 1.0)), np.log10(THIRD_OCTAVE_CENTERS), curve)
    # 1/3-octave bands are constant-Q, so per-band targets need a -3 dB/oct
    # correction to become per-bin amplitudes.
    gain -= 10 * np.log10(np.maximum(freqs, 20.0) / 1000.0)
    gain[freqs < 18] = -90
    gain[freqs > 20_500] = -90

    y = np.fft.irfft(spec * 10 ** (gain / 20), n)
    return y / (np.std(y) + 1e-12)


def _crest_db(st: np.ndarray) -> float:
    mono = np.mean(st, axis=1)
    return float(20 * np.log10(np.max(np.abs(mono)) / (np.sqrt(np.mean(mono**2)) + 1e-12)))


def _build_reference(genre: str, sr: int, dur: float) -> np.ndarray:
    profile = T.get_profile(genre)
    crest_want = float(np.mean(profile.crest_factor_db))
    width = float(np.mean(profile.stereo_width))
    lra_want = float(np.mean(profile.loudness_range_lu))

    n = int(sr * dur)
    mid = _shaped_noise(genre, 1, n, sr)
    side = sps.sosfilt(sps.butter(4, 200 / (sr / 2), "high", output="sos"),
                       _shaped_noise(genre, 2, n, sr)) * width

    t = np.arange(n) / sr
    # Section movement sized to the genre's own LRA, plus a 120 bpm pulse so
    # there is something for the limiter below to actually catch.
    sections = 10 ** ((lra_want * 0.40 * np.sign(np.sin(2 * np.pi * t / 11.0))) / 20)
    pulse = 0.45 + 0.55 * np.abs(np.sin(2 * np.pi * t * 2.0)) ** 1.5
    env = sections * pulse

    st = np.column_stack([(mid + side) * env, (mid - side) * env])
    st /= np.max(np.abs(st)) + 1e-9

    # Drive until crest reaches the genre figure. Trap gets hit hard, folk
    # barely at all — which is the real difference between those masters.
    drive = 1.0
    for _ in range(200):
        candidate = np.tanh(st * drive) / np.tanh(drive)
        if _crest_db(candidate) <= crest_want:
            st = candidate
            break
        drive *= 1.06
    else:
        st = np.tanh(st * drive) / np.tanh(drive)

    return st / (np.max(np.abs(st)) + 1e-9) * CEILING


def write_reference_fixtures(dur: float = 45.0) -> None:
    meter = pyln.Meter(ANALYSIS_SR)
    for genre, name in (("trap", "reference_trap"),
                        ("pop", "reference_pop"),
                        ("folk", "reference_folk")):
        st = _build_reference(genre, ANALYSIS_SR, dur)
        sf.write(f"{OUT_DIR}/{name}.wav", st, ANALYSIS_SR, subtype="PCM_24")

        profile = T.get_profile(genre)
        lufs = meter.integrated_loudness(st)
        crest = _crest_db(st)
        lo, hi = profile.integrated_lufs
        clo, chi = profile.crest_factor_db
        print(f"  {name:16s} LUFS {lufs:6.1f} [{lo}, {hi}] {'OK ' if lo <= lufs <= hi else 'OUT'}"
              f"   crest {crest:5.1f} [{clo}, {chi}] {'OK ' if clo <= crest <= chi else 'OUT'}")


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Writing fixtures to {OUT_DIR}\n\nDefect fixtures:")
    write_defect_fixtures()
    print("\nReference fixtures (must all read OK):")
    write_reference_fixtures()
    print("\nDone. These are gitignored — regenerate after a fresh clone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
