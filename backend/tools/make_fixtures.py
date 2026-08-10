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
from typing import Dict, Tuple

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



def write_beat_fixture(sr: int = 48_000, dur: float = 30.0) -> None:
    """A trap beat with bright hats and a hook tucked well down.

    This is the regression case for a reported bug: the 5-9 kHz detector fired
    on hi-hat bursts and called them vocal sibilance, because burstiness in that
    band is what a consonant and a closed hat have in common. The hook sits 26 dB
    under the drums on purpose — a beat is supposed to leave that room for
    someone to rap over — so nothing here should read as a missing or buried
    vocal either.
    """
    n = int(sr * dur)
    t = np.arange(n) / sr
    rng = np.random.default_rng(3)
    spb = 60.0 / 140.0                      # 140 bpm
    beats = np.arange(0, dur, spb)

    def hits(times, decay):
        e = np.zeros(n)
        for tt in times:
            i = int(tt * sr)
            length = min(int(decay * sr), n - i)
            if length > 0:
                e[i : i + length] += np.exp(-np.arange(length) / (0.0015 * sr))
        return e

    sub = np.zeros(n)
    for k, b in enumerate(beats):
        f = [41.2, 41.2, 55.0, 49.0][k % 4]
        i = int(b * sr)
        length = min(int(spb * 1.6 * sr), n - i)
        lt = np.arange(length) / sr
        sub[i : i + length] += np.sin(2 * np.pi * f * lt) * np.exp(-lt / 0.55)

    # Triplet hat rolls: the bursty top end that was being misread.
    hat_times = np.sort(np.concatenate([beats, beats + spb / 3, beats + 2 * spb / 3,
                                        beats + spb / 6]))
    hats = sps.sosfilt(sps.butter(4, 6500 / (sr / 2), "high", output="sos"),
                       hits(hat_times, 0.035) * rng.normal(0, 1, n))
    snare = sps.sosfilt(sps.butter(2, [190 / (sr / 2), 8000 / (sr / 2)], "band", output="sos"),
                        hits(beats[1::2], 0.18) * rng.normal(0, 1, n))
    mel = sum(np.sin(2 * np.pi * f * t) * np.exp(-((t % spb) / 0.4))
              for f in (329.6, 392.0, 493.9))
    hook = sum(a * np.sin(2 * np.pi * f * t + 0.5 * np.sin(2 * np.pi * 5.0 * t))
               for f, a in ((196, 1.0), (392, 0.5), (784, 0.3), (2400, 0.18), (6200, 0.12)))
    hook *= 0.5 + 0.5 * np.sin(2 * np.pi * 0.55 * t)

    def at(x, db):
        return x / (np.max(np.abs(x)) + 1e-9) * 10 ** (db / 20)

    left = at(sub, -4) + at(hats, -13) + at(snare, -10) + at(mel, -15) + at(hook, -26)
    right = (at(sub, -4) + at(np.roll(hats, 410), -13) + at(snare, -10)
             + at(np.roll(mel, 900), -15) + at(hook, -26))

    st = np.column_stack([left, right])
    st /= np.max(np.abs(st)) + 1e-9
    st = np.tanh(st * 1.6) / np.tanh(1.6)
    st = st / (np.max(np.abs(st)) + 1e-9) * CEILING
    sf.write(f"{OUT_DIR}/beat_tucked_hook.wav", st, sr, subtype="PCM_24")
    print(f"  {'beat_tucked_hook':16s} hats -13 dB, hook 26 dB down")


# ---------------------------------------------------------------------------
# Arrangement fixtures
#
# The regression pair for the reported bug: `low_end.section_collapse` fired on
# a track whose intro deliberately has little under it, and called that "the
# dominant issue". Both files here are the same song, arranged the same way, and
# both land on the *same measured low-end swing* — around 4.6 dB, which is past
# the 3.0 dB a trap arrangement is held to. The only difference is which section
# the bottom end was taken out of:
#
#   arrangement_sparse_intro.wav   pulled back in the intro   -> silent
#   arrangement_hollow_chorus.wav  pulled out of the chorus   -> fires
#
# Identical numbers, opposite verdicts, which is the whole claim: an intro that
# sheds its bottom end is an arrangement and a chorus that sheds it is a fault,
# and no amount of dB tells the two apart.
#
# Long enough to have a form (the arrangement detector needs 100 s and three
# sections before it says anything at all), and shaped so the segmenter can name
# the parts: a sparse intro, two verses at the working level, two choruses above
# them, and a sparse outro.
# ---------------------------------------------------------------------------

# (label, start, end, section level in dB relative to the chorus). The intro
# sits 6 dB down — thin and quiet, but well inside the 9 LU of the loudest
# section that decides whether the analyser looks at its low end at all. A quiet
# intro is already excluded; this one is not quiet enough to be, which is
# exactly the case that produced the bug.
_SONG_FORM: Tuple[Tuple[str, float, float, float], ...] = (
    ("intro",   0.0,  22.0, -6.0),
    ("verse",  22.0,  52.0, -2.5),
    ("chorus", 52.0,  82.0,  0.0),
    ("verse",  82.0, 108.0, -2.5),
    ("chorus", 108.0, 136.0, 0.0),
    ("outro",  136.0, 150.0, -8.5),
)

# Element gains per part, so the segmenter has something to cluster on beyond
# level: the intro and outro are sparse, the verses are the working body, the
# choruses add the lead and the full kit. The intro deliberately carries plenty
# of mid content — a real thin intro is *arranged* thin, not simply turned
# down, and it has to stay inside 9 LU of the loudest section or the analyser
# treats it as a quiet part and never looks at its low end at all.
_SONG_MIX = {
    "intro":  {"sub": 1.0, "kick": 0.0, "snare": 0.0, "hats": 0.0,
               "pad": 1.8, "lead": 0.6},
    "verse":  {"sub": 1.0, "kick": 1.0, "snare": 1.0, "hats": 1.0,
               "pad": 0.8, "lead": 0.3},
    "chorus": {"sub": 1.0, "kick": 1.0, "snare": 1.0, "hats": 1.0,
               "pad": 1.0, "lead": 1.0},
    "outro":  {"sub": 1.0, "kick": 0.0, "snare": 0.0, "hats": 0.0,
               "pad": 1.0, "lead": 0.5},
}

# How far the bottom end is pulled back, per section under test. Deliberately
# not "all of it": the complaint is about a section with *little* low end, which
# is what an arrangement actually does, and a section with literally nothing
# under 120 Hz would clear any ceiling in the file and prove nothing.
#
# The two figures differ because the two sections are not the same size — the
# intro is thin to begin with, the chorus is the densest part of the record — and
# what has to match between the fixtures is the *measured swing*, not the fader
# move. Both land near 4.6 dB: past the 3.0 dB ceiling a trap arrangement is
# held to (so the old code fired on both) and under the 6.0 dB a bookend has to
# clear (so the new code fires on one).
_LOW_PULL_DB: Dict[str, float] = {"intro": 5.5, "chorus": 10.5}


def _song_stereo(hollow: str, sr: int = 44_100) -> np.ndarray:
    """The song, with the low end pulled back in whichever part is named."""
    dur = _SONG_FORM[-1][2]
    n = int(sr * dur)
    t = np.arange(n) / sr
    rng = np.random.default_rng(11)
    spb = 60.0 / 96.0                       # 96 bpm
    beats = np.arange(0, dur, spb)

    def hits(times, decay):
        e = np.zeros(n)
        for tt in times:
            i = int(tt * sr)
            length = min(int(decay * sr), n - i)
            if length > 0:
                e[i : i + length] += np.exp(-np.arange(length) / (0.002 * sr))
        return e

    # -- elements ----------------------------------------------------------
    sub = np.zeros(n)                        # the bottom end under test
    for k, b in enumerate(beats):
        f = [49.0, 49.0, 65.4, 55.0][k % 4]
        i = int(b * sr)
        length = min(int(spb * 1.5 * sr), n - i)
        lt = np.arange(length) / sr
        sub[i : i + length] += np.sin(2 * np.pi * f * lt) * np.exp(-lt / 0.6)

    kick = np.zeros(n)
    for b in beats:
        i = int(b * sr)
        length = min(int(0.3 * sr), n - i)
        lt = np.arange(length) / sr
        f = 55 + 110 * np.exp(-lt / 0.025)
        kick[i : i + length] += np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-lt / 0.09)

    snare = sps.sosfilt(sps.butter(2, [190 / (sr / 2), 7500 / (sr / 2)], "band", output="sos"),
                        hits(beats[1::2], 0.16) * rng.normal(0, 1, n))
    hats = sps.sosfilt(sps.butter(4, 7000 / (sr / 2), "high", output="sos"),
                       hits(np.arange(0, dur, spb / 2), 0.04) * rng.normal(0, 1, n))
    pad = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
              for f in (196.0, 246.9, 293.7, 392.0, 493.9))
    lead = sum(a * np.sin(2 * np.pi * f * t + 0.3 * np.sin(2 * np.pi * 4.0 * t))
               for f, a in ((587.3, 1.0), (1174.6, 0.4), (2349.2, 0.2)))
    lead *= 0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * t)

    def at(x, db):
        return x / (np.max(np.abs(x)) + 1e-9) * 10 ** (db / 20)

    sub, kick = at(sub, -5), at(kick, -7)
    snare, hats = at(snare, -12), at(hats, -20)
    pad, lead = at(pad, -15), at(lead, -18)

    # -- per-section arrangement -------------------------------------------
    mix = _SONG_MIX
    parts = {"sub": sub, "kick": kick, "snare": snare, "hats": hats,
             "pad": pad, "lead": lead}
    fade = int(0.15 * sr)                   # crossfade, so no boundary clicks
    ramp = np.linspace(0.0, 1.0, fade)

    def render(label: str, i0: int, i1: int, gains: Dict[str, float]) -> np.ndarray:
        return sum(parts[name][i0:i1] * g for name, g in gains.items() if g > 0.0)

    out = np.zeros(n)
    for label, t0, t1, level_db in _SONG_FORM:
        i0, i1 = int(t0 * sr), min(int(t1 * sr), n)
        block = render(label, i0, i1, mix[label])
        if label == hollow:
            # The section under test: same arrangement, bottom end pulled back,
            # then made back up to the level it had. A record that thins its
            # intro still hits its intro level — the whole point of measuring
            # the low end as a *share* of the section's own energy is that it
            # separates "there is less down there" from "this part is quieter",
            # and a fixture that conflated the two would not test anything.
            pull = 10 ** (-_LOW_PULL_DB.get(label, 0.0) / 20)
            pulled = dict(mix[label])
            pulled["sub"] *= pull
            pulled["kick"] *= pull
            thin = render(label, i0, i1, pulled)
            makeup = float(np.sqrt(np.mean(block**2)) / (np.sqrt(np.mean(thin**2)) + 1e-12))
            block = thin * makeup
        block = block * 10 ** (level_db / 20)
        if fade < len(block):
            block[:fade] *= ramp
            block[-fade:] *= ramp[::-1]
        out[i0:i1] += block

    # A little stereo on everything above the bass, and the low end left mono.
    side = sps.sosfilt(sps.butter(4, 220 / (sr / 2), "high", output="sos"),
                       np.roll(out, 300)) * 0.22
    st = np.column_stack([out + side, out - side])
    st /= np.max(np.abs(st)) + 1e-9
    st = np.tanh(st * 1.35) / np.tanh(1.35)
    return st / (np.max(np.abs(st)) + 1e-9) * CEILING


def write_arrangement_fixtures(sr: int = 44_100) -> None:
    for hollow, name in (("intro", "arrangement_sparse_intro"),
                         ("chorus", "arrangement_hollow_chorus")):
        st = _song_stereo(hollow, sr)
        sf.write(f"{OUT_DIR}/{name}.wav", st, sr, subtype="PCM_24")
        print(f"  {name:26s} {_LOW_PULL_DB[hollow]:4.1f} dB of low end out of the {hollow}, "
              f"level made back up")


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Writing fixtures to {OUT_DIR}\n\nDefect fixtures:")
    write_defect_fixtures()
    print("\nIntent fixture:")
    write_beat_fixture()
    print("\nArrangement fixtures:")
    write_arrangement_fixtures()

    print("\nReference fixtures (must all read OK):")
    write_reference_fixtures()
    print("\nDone. These are gitignored — regenerate after a fresh clone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
