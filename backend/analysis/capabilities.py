"""What a plugin can *do*, as opposed to what it is called.

The point of this file is that a prescription should change shape based on the
tools available, not just swap a brand name in. "Cut the 318 Hz resonance"
becomes three different instructions depending on the box:

    resonance_suppressor  -> let Soothe track it; set depth, not frequency
    eq_dynamic            -> dynamic band at 318 Hz, threshold -24, range -6
    eq_static only        -> static notch, and automate it off in the sparse bars

Only the last one needs the producer to do the work by hand, and telling
someone with Soothe to hand-notch is the kind of advice that makes a tool feel
generic. Names go stale; capabilities do not.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

# Capability slug -> (short label, what it unlocks in a prescription).
# The second string is written for the model: it says what becomes possible,
# not what the plugin is.
CAPABILITIES: Dict[str, Tuple[str, str]] = {
    # --- EQ -----------------------------------------------------------------
    "eq_static": ("Static EQ", "Fixed bells, shelves and filters."),
    "eq_dynamic": (
        "Dynamic EQ",
        "A band that only acts when the region crosses a threshold — fixes a problem that "
        "comes and goes without thinning the parts where it does not.",
    ),
    "eq_linear_phase": (
        "Linear-phase EQ",
        "No phase smear across the crossover. Worth it on a mix bus or on parallel paths that "
        "get summed; costs pre-ringing on transients, so not for drums.",
    ),
    "eq_mid_side": (
        "Mid/Side EQ",
        "EQ the centre and the sides independently — the correct tool for widening the top "
        "without touching the lead, or mono-ing the low end.",
    ),
    "eq_match": (
        "Match EQ",
        "Fit the spectrum to a captured reference curve rather than dialling bands by ear.",
    ),
    # --- Dynamics -----------------------------------------------------------
    "comp_single": ("Compressor", "Standard downward compression."),
    "comp_multiband": (
        "Multiband compression",
        "Compress one frequency region without the rest ducking with it.",
    ),
    "comp_sidechain_ext": (
        "External sidechain",
        "Duck one source from another's signal — the honest way to do kick/808 separation.",
    ),
    "comp_parallel": (
        "Parallel / blend",
        "Built-in dry-wet, so density can be added without losing the transient.",
    ),
    "comp_bus": ("Bus compressor", "Glue-style bus compression, slow attack, programme-dependent."),
    "comp_opto": ("Opto compressor", "Slow, level-dependent release. Forgiving on vocals."),
    "comp_fet": ("FET compressor", "Very fast attack. Aggressive; good on drums and rock vocals."),
    "comp_vca": ("VCA compressor", "Precise, punchy, tight control of attack and release."),
    "expander": ("Expander / gate", "Push down what sits below a threshold."),
    # --- Loudness -----------------------------------------------------------
    "limiter": ("Limiter", "A hard ceiling."),
    "limiter_truepeak": (
        "True-peak limiter",
        "Oversampled ceiling detection — the only kind that actually holds -1.0 dBTP through "
        "a lossy encoder.",
    ),
    "clipper": (
        "Clipper",
        "Shave peaks before the limiter so the limiter stops working so hard. The standard "
        "modern loudness move on drums.",
    ),
    # --- Spectral repair ----------------------------------------------------
    "deesser": ("De-esser", "Frequency-selective ducking aimed at sibilance."),
    "resonance_suppressor": (
        "Resonance suppressor",
        "Tracks and ducks narrow resonances dynamically across the spectrum — replaces hunting "
        "for individual peaks with a notch.",
    ),
    "spectral_repair": ("Spectral repair", "Remove a sound from the spectrogram directly."),
    # --- Colour -------------------------------------------------------------
    "saturation": ("Saturation", "Harmonic generation for density and perceived loudness."),
    "tape": ("Tape", "Compression, HF softening and wow/flutter as one character."),
    "exciter": (
        "Exciter",
        "Synthesise harmonics above what is there — the only way to get top end back from a "
        "file that has none.",
    ),
    # --- Space --------------------------------------------------------------
    "imager": ("Stereo imager", "Widen or narrow, usually per band."),
    "mono_maker": (
        "Mono maker",
        "Force everything below a frequency to mono — the direct fix for a smeared low end.",
    ),
    "reverb": ("Reverb", "Space and depth."),
    "delay": ("Delay", "Echo, and width via short offsets."),
    "transient_shaper": (
        "Transient shaper",
        "Attack and sustain independent of level — restores punch that compression took, "
        "without touching the balance.",
    ),
    # --- Measurement --------------------------------------------------------
    "meter_loudness": ("Loudness meter", "LUFS, LRA and true peak readout."),
    "meter_spectrum": ("Spectrum analyzer", "See the balance while working."),
    "meter_correlation": ("Correlation meter", "Watch phase and mono compatibility."),
    "reference_matching": ("Reference tool", "A/B against a commercial record, level-matched."),
    # --- Restoration / pitch -------------------------------------------------
    "pitch_correction": ("Pitch correction", "Tuning."),
    "noise_reduction": ("Noise reduction", "Broadband noise, hum and clicks."),
}

# Everyone has these, whatever DAW they use. Advice may always assume them, so
# a producer who has configured nothing still gets an executable plan.
STOCK_CAPABILITIES: Tuple[str, ...] = (
    "eq_static",
    "comp_single",
    "limiter",
    "expander",
    "reverb",
    "delay",
    "saturation",
    "meter_spectrum",
)

# Which capability actually solves which finding. The detector layer names the
# problem; this says what class of tool resolves it, so the AI can be told
# "they own one of these" or "they own none of these, route around it".
CAPABILITY_FOR_DIMENSION: Dict[str, Tuple[str, ...]] = {
    "clipping": ("limiter_truepeak", "clipper", "limiter"),
    "limiter": ("limiter_truepeak", "clipper", "comp_multiband"),
    "loudness": ("limiter_truepeak", "clipper", "comp_bus"),
    "dynamic_range": ("comp_bus", "comp_parallel", "transient_shaper", "clipper"),
    "compression": ("comp_parallel", "transient_shaper", "comp_multiband", "comp_opto"),
    "frequency_balance": ("eq_static", "eq_match", "eq_linear_phase"),
    "mud": ("eq_dynamic", "resonance_suppressor", "comp_multiband", "eq_static"),
    "harshness": ("resonance_suppressor", "eq_dynamic", "deesser", "eq_static"),
    "phase": ("mono_maker", "eq_mid_side", "meter_correlation"),
    "stereo_width": ("imager", "eq_mid_side", "mono_maker", "delay"),
    "low_end": ("comp_sidechain_ext", "eq_dynamic", "mono_maker", "comp_multiband"),
    "vocal_balance": ("comp_opto", "eq_dynamic", "deesser", "comp_parallel"),
    "transients": ("transient_shaper", "clipper", "comp_parallel"),
    "clarity": ("resonance_suppressor", "eq_dynamic", "comp_multiband", "eq_mid_side"),
}


def normalise(caps: Sequence[str]) -> List[str]:
    """Drop unknown slugs and de-duplicate, preserving order.

    Capability tags arrive from the client, so they are untrusted input; an
    unrecognised slug means a stale build, not a new feature.
    """
    seen: Dict[str, None] = {}
    for cap in caps:
        slug = str(cap).strip().lower()
        if slug in CAPABILITIES and slug not in seen:
            seen[slug] = None
    return list(seen)


def owned_capabilities(plugins: Sequence["OwnedPluginLike"]) -> List[str]:
    """Every capability the producer can reach, stock included."""
    out: Dict[str, None] = {c: None for c in STOCK_CAPABILITIES}
    for plugin in plugins:
        for cap in normalise(getattr(plugin, "capabilities", []) or []):
            out[cap] = None
    return list(out)


def missing_for(dimension: str, owned: Sequence[str]) -> List[str]:
    """Capabilities that would help this dimension but the producer lacks.

    Used to tell the model when to route around a gap — and, in the UI, to
    suggest what would actually be worth buying.
    """
    wanted = CAPABILITY_FOR_DIMENSION.get(dimension, ())
    have = set(owned)
    return [c for c in wanted if c not in have]


def describe(caps: Sequence[str]) -> str:
    """Render capabilities as a brief for the model."""
    lines = []
    for cap in normalise(caps):
        label, unlocks = CAPABILITIES[cap]
        lines.append(f"- **{label}** (`{cap}`) — {unlocks}")
    return "\n".join(lines)


class OwnedPluginLike:
    """Structural stand-in for `types.OwnedPlugin`.

    Declared here only so this module stays importable from the DSP layer
    without pulling in the Pydantic models and creating an import cycle.
    """

    name: str
    capabilities: List[str]
