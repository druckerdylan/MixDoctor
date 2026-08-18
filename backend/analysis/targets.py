"""Genre reference data — the encoded listening experience.

This is where the opinions live. Everything else in the analysis layer is
measurement; this file is the part that says "for trap, 8 dB of sub over the
1 kHz band is correct, and for folk it is a mistake."

Target curves are anchor points in (Hz, dB relative to the 1 kHz band),
log-interpolated onto the 1/3-octave grid. Anchors are easier to reason about
and to argue with than 31 hand-typed numbers per genre.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .core import THIRD_OCTAVE_CENTERS

Anchors = Sequence[Tuple[float, float]]


# ---------------------------------------------------------------------------
# Target curves
# ---------------------------------------------------------------------------

_CURVES: Dict[str, Anchors] = {
    # Balanced modern pop master: firm low shelf, flat mids, gentle air lift.
    "pop": (
        (20, -3.0), (40, 4.0), (63, 6.0), (100, 5.5), (160, 3.0), (250, 1.0),
        (400, -0.5), (630, -0.5), (1000, 0.0), (2000, 0.5), (3150, 0.5),
        (5000, 0.0), (8000, -1.0), (12500, -3.0), (20000, -9.0),
    ),
    # Sub is the instrument. Low-mids stay clear so the 808 reads as pitch.
    "hip_hop": (
        (20, 2.0), (40, 8.0), (63, 9.0), (100, 7.0), (160, 3.5), (250, 0.5),
        (400, -1.5), (630, -1.0), (1000, 0.0), (2000, 0.5), (3150, 0.0),
        (5000, -0.5), (8000, -2.0), (12500, -4.5), (20000, -11.0),
    ),
    # Club-oriented: big sub *and* big air, scooped low-mids for headroom.
    "edm": (
        (20, 0.0), (40, 7.0), (63, 8.0), (100, 6.0), (160, 3.0), (250, 0.5),
        (400, -1.0), (630, -0.5), (1000, 0.0), (2000, 1.0), (3150, 1.0),
        (5000, 0.5), (8000, 0.0), (12500, -2.0), (20000, -8.0),
    ),
    # Guitars own the mids; sub is mostly absent by design.
    "rock": (
        (20, -8.0), (40, 0.0), (63, 3.0), (100, 4.0), (160, 3.0), (250, 1.5),
        (400, 0.0), (630, 0.0), (1000, 0.0), (2000, 1.0), (3150, 1.5),
        (5000, 0.5), (8000, -1.0), (12500, -4.0), (20000, -11.0),
    ),
    "metal": (
        (20, -9.0), (40, -1.0), (63, 3.0), (100, 4.0), (160, 2.0), (250, 0.0),
        (400, -1.5), (630, -1.0), (1000, 0.0), (2000, 1.5), (3150, 2.0),
        (5000, 1.0), (8000, -0.5), (12500, -3.5), (20000, -11.0),
    ),
    # Warm and round; presence pulled back so vocals stay silky.
    "rnb": (
        (20, -3.0), (40, 5.0), (63, 7.0), (100, 6.0), (160, 3.5), (250, 1.5),
        (400, 0.0), (630, -0.5), (1000, 0.0), (2000, 0.0), (3150, -0.5),
        (5000, -1.0), (8000, -2.0), (12500, -4.0), (20000, -10.0),
    ),
    "acoustic": (
        (20, -10.0), (40, -1.0), (63, 2.0), (100, 3.0), (160, 2.5), (250, 1.5),
        (400, 0.5), (630, 0.0), (1000, 0.0), (2000, 0.5), (3150, 0.5),
        (5000, 0.0), (8000, -1.0), (12500, -3.0), (20000, -9.0),
    ),
    "jazz": (
        (20, -11.0), (40, -2.0), (63, 1.5), (100, 3.0), (160, 3.0), (250, 2.0),
        (400, 1.0), (630, 0.5), (1000, 0.0), (2000, 0.0), (3150, 0.0),
        (5000, -0.5), (8000, -1.5), (12500, -3.5), (20000, -10.0),
    ),
    "classical": (
        (20, -12.0), (40, -3.0), (63, 0.0), (100, 2.0), (160, 2.0), (250, 1.5),
        (400, 1.0), (630, 0.5), (1000, 0.0), (2000, 0.0), (3150, -0.5),
        (5000, -1.0), (8000, -2.5), (12500, -5.0), (20000, -12.0),
    ),
    "country": (
        (20, -9.0), (40, 0.5), (63, 3.5), (100, 4.0), (160, 3.0), (250, 1.5),
        (400, 0.0), (630, -0.5), (1000, 0.0), (2000, 0.5), (3150, 1.0),
        (5000, 0.5), (8000, -1.0), (12500, -3.5), (20000, -10.0),
    ),
    "indie": (
        (20, -7.0), (40, 1.0), (63, 4.0), (100, 4.5), (160, 3.0), (250, 1.5),
        (400, 0.0), (630, -0.5), (1000, 0.0), (2000, 0.5), (3150, 0.5),
        (5000, 0.0), (8000, -1.5), (12500, -4.0), (20000, -10.0),
    ),
    "ambient": (
        (20, -6.0), (40, 2.0), (63, 4.0), (100, 4.0), (160, 3.0), (250, 2.0),
        (400, 1.0), (630, 0.5), (1000, 0.0), (2000, -0.5), (3150, -1.0),
        (5000, -1.0), (8000, -1.5), (12500, -3.0), (20000, -8.0),
    ),
    # Deliberately dark and boxy — the "flaws" are the genre.
    "lofi": (
        (20, -8.0), (40, 2.0), (63, 5.0), (100, 5.0), (160, 4.0), (250, 2.5),
        (400, 1.0), (630, 0.0), (1000, 0.0), (2000, -1.0), (3150, -2.0),
        (5000, -4.0), (8000, -7.0), (12500, -12.0), (20000, -20.0),
    ),
}

# Sub-genres that share a parent curve but differ in the scalar targets below.
_CURVE_ALIASES: Dict[str, str] = {
    "house": "edm",
    "techno": "edm",
    "dnb": "edm",
    "trap": "hip_hop",
    "soul": "rnb",
    "folk": "acoustic",
    "orchestral": "classical",
    "cinematic": "classical",
    "alternative": "indie",
    "punk": "rock",
    "other": "pop",
}

# Free-text genre strings from the UI mapped onto internal keys.
_GENRE_NORMALISE: Dict[str, str] = {
    "pop": "pop",
    "rock": "rock",
    "metal": "metal",
    "hip hop": "hip_hop", "hip-hop": "hip_hop", "hiphop": "hip_hop",
    "hip hop / trap": "hip_hop", "rap": "hip_hop", "trap": "trap",
    "r&b": "rnb", "r&b / soul": "rnb", "rnb": "rnb", "soul": "soul",
    "edm": "edm", "edm / electronic": "edm", "electronic": "edm",
    "dance": "edm", "dubstep": "edm", "future bass": "edm",
    "house": "house", "deep house": "house", "tech house": "house",
    "techno": "techno",
    "drum & bass": "dnb", "drum and bass": "dnb", "dnb": "dnb", "d&b": "dnb",
    "jazz": "jazz",
    "classical": "classical",
    "orchestral": "orchestral", "orchestral / cinematic": "orchestral",
    "cinematic": "cinematic", "score": "cinematic",
    "indie": "indie", "indie / alternative": "indie", "alternative": "alternative",
    "acoustic": "acoustic", "folk": "folk", "singer-songwriter": "acoustic",
    "ambient": "ambient", "downtempo": "ambient",
    "country": "country", "americana": "country",
    "lo-fi": "lofi", "lofi": "lofi", "lo fi": "lofi",
    "punk": "punk",
    "other": "other",
}


@dataclass(frozen=True)
class GenreProfile:
    """Everything genre-dependent, in one object.

    Ranges are (low, high) inclusive-ish "this is fine" windows. A detector
    only fires outside the window, and severity scales with how far outside.
    """

    key: str
    label: str
    curve_key: str

    integrated_lufs: Tuple[float, float]
    loudness_range_lu: Tuple[float, float]
    crest_factor_db: Tuple[float, float]
    micro_dynamics_db: Tuple[float, float]
    psr_p10_db: Tuple[float, float]

    stereo_width: Tuple[float, float]
    correlation_min: float
    low_end_mono_min: float = 0.80

    mud_ratio_db: Tuple[float, float] = (-14.0, -5.0)
    harshness_max: float = 0.42
    sibilance_max: float = 0.40
    sharpness_max_acum: float = 2.6

    punch_min: float = 0.35
    clarity_min: float = 0.45
    masking_max: float = 0.55

    vocal_to_instrument_db: Tuple[float, float] = (-4.0, 4.0)
    vocal_expected: bool = True

    kick_bass_collision_max_db: float = -6.0
    sub_energy_db: Tuple[float, float] = (-24.0, -8.0)

    band_tolerance_db: Dict[str, float] = field(default_factory=dict)
    notes: str = ""


# Tolerance per macro band: how far from target before we call it a problem.
# Tightest through the mids because that is where the ear is least forgiving
# and where a 2 dB error is audible on any playback system.
_DEFAULT_TOLERANCE: Dict[str, float] = {
    "sub": 4.5,
    "low_bass": 3.0,
    "upper_bass": 2.5,
    "low_mid": 2.2,
    "mid": 2.0,
    "upper_mid": 2.0,
    "presence": 2.2,
    "brilliance": 3.0,
    "air": 4.0,
}


_PROFILES: Dict[str, GenreProfile] = {}


def _register(
    key: str,
    label: str,
    curve_key: str,
    *,
    lufs: Tuple[float, float],
    lra: Tuple[float, float],
    crest: Tuple[float, float],
    micro: Tuple[float, float],
    psr: Tuple[float, float],
    width: Tuple[float, float],
    corr_min: float,
    **kwargs,
) -> None:
    tol = dict(_DEFAULT_TOLERANCE)
    tol.update(kwargs.pop("tolerance", {}) or {})
    _PROFILES[key] = GenreProfile(
        key=key,
        label=label,
        curve_key=curve_key,
        integrated_lufs=lufs,
        loudness_range_lu=lra,
        crest_factor_db=crest,
        micro_dynamics_db=micro,
        psr_p10_db=psr,
        stereo_width=width,
        correlation_min=corr_min,
        band_tolerance_db=tol,
        **kwargs,
    )


# Loudness windows are "where commercial releases in this genre actually sit",
# not "where streaming normalization would prefer they sat".
_register(
    "pop", "Pop", "pop",
    lufs=(-11.0, -7.5), lra=(3.0, 8.0), crest=(8.0, 13.0), micro=(5.0, 11.0),
    psr=(6.0, 12.0), width=(0.28, 0.60), corr_min=0.30,
    mud_ratio_db=(-14.0, -5.5), harshness_max=0.40, punch_min=0.38,
    notes="Vocal-forward. Low-mids stay clean so the lead sits without EQ carving.",
)
_register(
    "hip_hop", "Hip-Hop", "hip_hop",
    lufs=(-10.0, -6.5), lra=(2.5, 6.5), crest=(6.0, 11.0), micro=(4.0, 9.0),
    psr=(4.5, 10.0), width=(0.20, 0.48), corr_min=0.40,
    mud_ratio_db=(-16.0, -7.0), harshness_max=0.38, punch_min=0.42,
    low_end_mono_min=0.90, sub_energy_db=(-18.0, -5.0),
    kick_bass_collision_max_db=-4.0,
    tolerance={"sub": 5.5, "low_bass": 3.5},
    notes="Sub is the hook. Kick and 808 must not occupy the same fundamental.",
)
_register(
    "trap", "Trap", "hip_hop",
    lufs=(-9.5, -6.0), lra=(2.0, 6.0), crest=(5.5, 10.0), micro=(3.5, 8.5),
    psr=(4.0, 9.0), width=(0.18, 0.45), corr_min=0.45,
    mud_ratio_db=(-17.0, -7.5), harshness_max=0.36, punch_min=0.45,
    low_end_mono_min=0.92, sub_energy_db=(-16.0, -4.0),
    kick_bass_collision_max_db=-4.0,
    tolerance={"sub": 6.0, "low_bass": 3.5},
    notes="808 glides need pitch definition; hats carry the top with no harsh 3-5k.",
)
_register(
    "edm", "EDM / Electronic", "edm",
    lufs=(-9.5, -6.0), lra=(2.5, 6.5), crest=(5.5, 10.0), micro=(4.0, 9.0),
    psr=(4.0, 9.5), width=(0.35, 0.72), corr_min=0.25,
    mud_ratio_db=(-16.0, -6.5), harshness_max=0.44, punch_min=0.45,
    low_end_mono_min=0.90,
    tolerance={"sub": 5.0, "air": 3.5},
    notes="Wide above 200 Hz, mono below. Drop needs headroom the intro doesn't.",
)
_register(
    "house", "House", "edm",
    lufs=(-9.5, -6.5), lra=(2.5, 6.0), crest=(6.0, 10.5), micro=(4.5, 9.0),
    psr=(4.5, 9.5), width=(0.32, 0.65), corr_min=0.30,
    mud_ratio_db=(-15.0, -6.0), punch_min=0.45, low_end_mono_min=0.90,
    notes="Four-on-the-floor: kick consistency matters more than peak loudness.",
)
_register(
    "techno", "Techno", "edm",
    lufs=(-9.0, -6.0), lra=(2.0, 5.5), crest=(5.5, 10.0), micro=(4.0, 8.5),
    psr=(4.0, 9.0), width=(0.28, 0.60), corr_min=0.30,
    mud_ratio_db=(-15.0, -6.0), punch_min=0.48, low_end_mono_min=0.92,
    notes="Relentless by design; groove lives in the kick's attack, not its length.",
)
_register(
    "dnb", "Drum & Bass", "edm",
    lufs=(-9.0, -5.5), lra=(2.5, 7.0), crest=(6.0, 11.0), micro=(4.5, 10.0),
    psr=(4.0, 10.0), width=(0.30, 0.65), corr_min=0.28,
    mud_ratio_db=(-16.0, -7.0), punch_min=0.55, low_end_mono_min=0.92,
    notes="Breaks need transient survival; over-limiting kills the whole genre.",
)
_register(
    "rock", "Rock", "rock",
    lufs=(-12.0, -8.0), lra=(4.0, 10.0), crest=(9.0, 15.0), micro=(6.0, 13.0),
    psr=(7.0, 14.0), width=(0.30, 0.60), corr_min=0.35,
    mud_ratio_db=(-12.0, -4.0), harshness_max=0.46, punch_min=0.40,
    notes="Guitars and vocals collide at 2-4 kHz; that fight is the whole mix.",
)
_register(
    "punk", "Punk", "rock",
    lufs=(-11.0, -7.0), lra=(3.0, 8.0), crest=(8.0, 13.0), micro=(5.0, 11.0),
    psr=(6.0, 12.0), width=(0.25, 0.55), corr_min=0.40,
    mud_ratio_db=(-12.0, -4.0), harshness_max=0.50, punch_min=0.42,
)
_register(
    "metal", "Metal", "metal",
    lufs=(-11.0, -7.0), lra=(3.0, 8.0), crest=(7.0, 12.0), micro=(5.0, 10.0),
    psr=(5.0, 11.0), width=(0.28, 0.58), corr_min=0.38,
    mud_ratio_db=(-15.0, -6.0), harshness_max=0.48, punch_min=0.45,
    notes="Scooped low-mids are stylistic; scooped *clarity* is not.",
)
_register(
    "rnb", "R&B / Soul", "rnb",
    lufs=(-12.0, -8.0), lra=(3.5, 9.0), crest=(8.0, 14.0), micro=(5.5, 12.0),
    psr=(6.5, 13.0), width=(0.25, 0.55), corr_min=0.35,
    mud_ratio_db=(-13.0, -5.0), harshness_max=0.34, sibilance_max=0.34,
    notes="Warmth without mud. Sibilance is the single most common defect here.",
)
_register(
    "soul", "Soul", "rnb",
    lufs=(-13.0, -8.5), lra=(4.0, 10.0), crest=(9.0, 15.0), micro=(6.0, 13.0),
    psr=(7.0, 14.0), width=(0.25, 0.55), corr_min=0.35,
    mud_ratio_db=(-13.0, -5.0), harshness_max=0.34, sibilance_max=0.34,
)
_register(
    "acoustic", "Acoustic", "acoustic",
    lufs=(-16.0, -11.0), lra=(6.0, 14.0), crest=(11.0, 18.0), micro=(8.0, 16.0),
    psr=(9.0, 17.0), width=(0.25, 0.55), corr_min=0.40,
    mud_ratio_db=(-13.0, -5.0), harshness_max=0.34, punch_min=0.30,
    notes="Dynamics are the performance. Compressing this to -9 LUFS destroys it.",
)
_register(
    "folk", "Folk", "acoustic",
    lufs=(-16.0, -11.0), lra=(6.0, 14.0), crest=(11.0, 18.0), micro=(8.0, 16.0),
    psr=(9.0, 17.0), width=(0.25, 0.55), corr_min=0.40,
    mud_ratio_db=(-13.0, -5.0), harshness_max=0.34, punch_min=0.30,
)
_register(
    "jazz", "Jazz", "jazz",
    lufs=(-18.0, -12.0), lra=(7.0, 16.0), crest=(12.0, 20.0), micro=(9.0, 18.0),
    psr=(10.0, 18.0), width=(0.25, 0.60), corr_min=0.35,
    mud_ratio_db=(-12.0, -4.0), harshness_max=0.32, punch_min=0.28,
    notes="Room and depth over level. Loudness targets barely apply.",
)
_register(
    "classical", "Classical", "classical",
    lufs=(-23.0, -16.0), lra=(10.0, 24.0), crest=(14.0, 24.0), micro=(11.0, 22.0),
    psr=(12.0, 22.0), width=(0.30, 0.70), corr_min=0.30,
    mud_ratio_db=(-12.0, -4.0), harshness_max=0.30, punch_min=0.20,
    vocal_expected=False,
    notes="Any limiting at all is usually a mistake.",
)
_register(
    "orchestral", "Orchestral", "classical",
    lufs=(-22.0, -14.0), lra=(8.0, 22.0), crest=(13.0, 22.0), micro=(10.0, 20.0),
    psr=(11.0, 20.0), width=(0.30, 0.70), corr_min=0.30,
    mud_ratio_db=(-12.0, -4.0), harshness_max=0.32, punch_min=0.22,
    vocal_expected=False,
)
_register(
    "cinematic", "Cinematic", "classical",
    lufs=(-20.0, -12.0), lra=(7.0, 20.0), crest=(12.0, 20.0), micro=(9.0, 18.0),
    psr=(10.0, 19.0), width=(0.35, 0.80), corr_min=0.20,
    mud_ratio_db=(-14.0, -5.0), harshness_max=0.36, punch_min=0.30,
    vocal_expected=False,
    tolerance={"sub": 5.5, "air": 4.5},
)
_register(
    "country", "Country", "country",
    lufs=(-12.0, -8.5), lra=(4.0, 10.0), crest=(9.0, 15.0), micro=(6.0, 13.0),
    psr=(7.0, 14.0), width=(0.28, 0.58), corr_min=0.38,
    mud_ratio_db=(-13.0, -5.0), harshness_max=0.40, punch_min=0.36,
)
_register(
    "indie", "Indie / Alternative", "indie",
    lufs=(-13.0, -8.5), lra=(4.0, 10.0), crest=(9.0, 15.0), micro=(6.0, 13.0),
    psr=(7.0, 14.0), width=(0.30, 0.62), corr_min=0.32,
    mud_ratio_db=(-13.0, -5.0), harshness_max=0.44, punch_min=0.35,
)
_register(
    "alternative", "Alternative", "indie",
    lufs=(-12.5, -8.0), lra=(4.0, 10.0), crest=(9.0, 15.0), micro=(6.0, 13.0),
    psr=(7.0, 14.0), width=(0.30, 0.62), corr_min=0.32,
    mud_ratio_db=(-13.0, -5.0), harshness_max=0.44, punch_min=0.35,
)
_register(
    "ambient", "Ambient", "ambient",
    lufs=(-20.0, -13.0), lra=(5.0, 16.0), crest=(10.0, 20.0), micro=(7.0, 16.0),
    psr=(9.0, 18.0), width=(0.35, 0.85), corr_min=0.10,
    mud_ratio_db=(-12.0, -3.0), harshness_max=0.30, punch_min=0.10,
    vocal_expected=False,
    notes="Width is the point; low correlation here is intent, not a fault.",
)
_register(
    "lofi", "Lo-Fi", "lofi",
    lufs=(-16.0, -10.0), lra=(3.0, 9.0), crest=(8.0, 15.0), micro=(5.0, 12.0),
    psr=(6.0, 13.0), width=(0.20, 0.55), corr_min=0.35,
    mud_ratio_db=(-10.0, -2.0), harshness_max=0.28, punch_min=0.20,
    vocal_expected=False,
    tolerance={"brilliance": 6.0, "air": 8.0, "low_mid": 4.0},
    notes="Dark and boxy on purpose. Do not 'fix' the rolled-off top.",
)
_register(
    "other", "Other", "pop",
    lufs=(-14.0, -8.0), lra=(3.0, 11.0), crest=(8.0, 15.0), micro=(5.0, 13.0),
    psr=(6.0, 14.0), width=(0.25, 0.65), corr_min=0.30,
    mud_ratio_db=(-14.0, -4.5),
)


@dataclass(frozen=True)
class Platform:
    name: str
    target_lufs: float
    max_true_peak_dbtp: float
    note: str


PLATFORMS: Tuple[Platform, ...] = (
    Platform("Spotify", -14.0, -1.0, "Normalizes loud tracks down; quiet ones are left alone."),
    Platform("Apple Music", -16.0, -1.0, "Sound Check targets -16; masters above it lose punch."),
    Platform("YouTube", -14.0, -1.0, "Turns down anything louder; no upward normalization."),
    Platform("Tidal", -14.0, -1.0, "Album-mode normalization preserves relative track levels."),
    Platform("Amazon Music", -14.0, -2.0, "Stricter peak ceiling than most."),
    Platform("SoundCloud", -14.0, -1.0, "Transcodes to lossy; leave true-peak headroom."),
    Platform("Club / DJ", -8.0, -0.3, "Played loud on a big system; sub control matters most."),
)


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def normalise_genre(genre: str) -> str:
    """Map a UI genre string onto an internal profile key."""
    if not genre:
        return "other"
    key = genre.strip().lower()
    if key in _PROFILES:
        return key
    if key in _GENRE_NORMALISE:
        return _GENRE_NORMALISE[key]
    # Fall back to a substring hit so "Melodic Dubstep" still lands on EDM.
    for phrase, mapped in _GENRE_NORMALISE.items():
        if phrase in key or key in phrase:
            return mapped
    return "other"


def get_profile(genre: str) -> GenreProfile:
    return _PROFILES[normalise_genre(genre)]


def target_curve(genre: str, centers: Sequence[float] = THIRD_OCTAVE_CENTERS) -> np.ndarray:
    """Log-interpolated 1/3-octave target curve, in dB relative to 1 kHz."""
    profile = get_profile(genre)
    curve_key = _CURVE_ALIASES.get(profile.curve_key, profile.curve_key)
    anchors = _CURVES[curve_key]

    a_hz = np.log10([a[0] for a in anchors])
    a_db = np.asarray([a[1] for a in anchors], dtype=np.float64)
    return np.interp(np.log10(np.asarray(centers, dtype=np.float64)), a_hz, a_db)


def macro_target(genre: str, band_name: str, low_hz: float, high_hz: float) -> float:
    """Energy-weighted target level for a macro band, in dB relative to 1 kHz."""
    centers = np.asarray(THIRD_OCTAVE_CENTERS, dtype=np.float64)
    curve = target_curve(genre, THIRD_OCTAVE_CENTERS)
    mask = (centers >= low_hz) & (centers < high_hz)
    if not np.any(mask):
        return 0.0
    # Sum in the power domain: two bands at 0 dB carry more total energy than one.
    lin = 10 ** (curve[mask] / 10.0)
    return float(10.0 * np.log10(lin.sum() / mask.sum()))


def band_tolerance(genre: str, band_name: str) -> float:
    return get_profile(genre).band_tolerance_db.get(band_name, 3.0)


def all_genres() -> List[Tuple[str, str]]:
    """(key, label) for every profile, for populating the UI selector."""
    seen: Dict[str, str] = {}
    for key, profile in _PROFILES.items():
        seen.setdefault(profile.label, key)
    return sorted(((k, lbl) for lbl, k in seen.items()), key=lambda kv: kv[1])


def in_range(value: float, window: Tuple[float, float]) -> bool:
    return window[0] <= value <= window[1]


def range_miss(value: float, window: Tuple[float, float]) -> float:
    """Signed distance outside a window. 0 inside, negative below, positive above."""
    lo, hi = window
    if value < lo:
        return value - lo
    if value > hi:
        return value - hi
    return 0.0
