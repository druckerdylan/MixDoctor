"""The analysis contract.

Every DSP module returns one of the `*Measurement` models; every detector
returns `Finding`s; the engine composes both into `MixAnalysis`, which is what
the API serialises and the UI renders. Changing a field here is a breaking
change on both sides — add, don't rename.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

Severity = Literal["critical", "major", "minor", "clean"]
Verdict = Literal["good", "watch", "problem"]

# What the file *is*, which decides which questions are even worth asking of it.
# A tucked lead is a fault in a finished song and the entire point of a beat; a
# bass stem is supposed to be all low end; nobody needs to be told to fix a
# released record. Defaults to `full_mix`, so every existing caller keeps the
# behaviour it has today.
TrackIntent = Literal["full_mix", "beat", "instrumental", "stem", "reference", "demo"]

TRACK_INTENTS: Tuple[str, ...] = (
    "full_mix", "beat", "instrumental", "stem", "reference", "demo",
)

TRACK_INTENT_LABELS: Dict[str, str] = {
    "full_mix": "Full mix",
    "beat": "Beat (instrumental for a topline)",
    "instrumental": "Instrumental",
    "stem": "Single stem",
    "reference": "Reference track",
    "demo": "Demo / rough",
}

# The line the whole report turns on.
#
# A **defect** is wrong no matter the genre, the intent, the artist or the year:
# a squared-off waveform, an inverted channel, a mix that disappears in mono.
# Nobody chose it and nobody would.
#
# A **deviation** is a measured difference from a reference — the genre profile,
# or a target curve derived from it. It may be deliberate, it may be the best
# decision on the record, and it is reported as "this differs from the reference,
# here is what that costs and what it buys". It is never reported as broken, and
# it never earns a critical.
FindingKind = Literal["defect", "deviation"]

Dimension = Literal[
    "frequency_balance",
    "mud",
    "harshness",
    "stereo_width",
    "dynamic_range",
    "clipping",
    "phase",
    "low_end",
    "vocal_balance",
    "compression",
    "loudness",
    "limiter",
    "transients",
    "clarity",
]

# Display order is diagnostic order: things that make a mix sound *broken*
# come before things that make it sound *unfinished*.
DIMENSIONS: Tuple[Dimension, ...] = (
    "clipping",
    "phase",
    "loudness",
    "limiter",
    "dynamic_range",
    "compression",
    "frequency_balance",
    "mud",
    "harshness",
    "low_end",
    "vocal_balance",
    "stereo_width",
    "transients",
    "clarity",
)

DIMENSION_LABELS: Dict[str, str] = {
    "frequency_balance": "Frequency Balance",
    "mud": "Mud & Low-Mid Buildup",
    "harshness": "Harshness & Sibilance",
    "stereo_width": "Stereo Width",
    "dynamic_range": "Dynamic Range",
    "clipping": "Clipping & True Peak",
    "phase": "Phase & Mono Compatibility",
    "low_end": "Kick / 808 Relationship",
    "vocal_balance": "Vocal Balance",
    "compression": "Compression",
    "loudness": "Loudness",
    "limiter": "Limiter Behaviour",
    "transients": "Transient Impact",
    "clarity": "Mix Clarity",
}

SEVERITY_RANK: Dict[str, int] = {"critical": 0, "major": 1, "minor": 2, "clean": 3}


class Band(BaseModel):
    """One frequency band's measured level against its genre target."""

    name: str
    low_hz: float
    high_hz: float
    center_hz: float
    level_db: float = Field(description="Level relative to the mix's own broadband level")
    target_db: float
    deviation_db: float = Field(description="level_db - target_db; + means hot")
    verdict: Verdict


class Moment(BaseModel):
    """A timestamped occurrence of an issue — what makes the timeline clickable."""

    t_start: float
    t_end: float
    intensity: float = Field(ge=0.0, le=1.0, description="0-1, drives marker size/glow")
    value: Optional[float] = None
    label: str = ""


class Evidence(BaseModel):
    """One measured number backing a finding. Never LLM-generated."""

    label: str
    value: float
    unit: str = ""
    target: Optional[float] = None
    target_range: Optional[Tuple[float, float]] = None
    verdict: Verdict = "good"
    detail: str = ""


class Finding(BaseModel):
    """A detected, measured problem. Produced by rules, explained later by AI.

    `detail` is a deterministic sentence built from the numbers — it is what
    guarantees the report says something true even if the AI layer is
    unavailable.
    """

    id: str = Field(description="Stable slug, e.g. 'mud.low_mid_buildup'")
    dimension: Dimension
    title: str
    kind: FindingKind = Field(
        default="deviation",
        description=(
            "'defect' = objectively wrong regardless of genre, intent or taste. "
            "'deviation' = a measured difference from the genre reference, which "
            "may well be deliberate. Only defects may be critical."
        ),
    )
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    detail: str
    evidence: List[Evidence] = Field(default_factory=list)
    band_hz: Optional[Tuple[float, float]] = None
    moments: List[Moment] = Field(default_factory=list)
    impact: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Estimated health-score points recoverable"
    )
    miss_ratio: float = Field(
        default=0.0, ge=0.0,
        description=(
            "How far outside its window the measurement sits, in tolerance units. "
            "0 = inside. 1.5 = a defect's 'major'. 3.0 = a defect's 'critical'. "
            "This is the magnitude the severity label buckets away, and the scorer "
            "reads it directly so that two findings sharing a label but not a "
            "distance do not score the same."
        ),
    )


class DimensionScore(BaseModel):
    """Per-dimension roll-up. This is what the radar / mix map renders."""

    dimension: Dimension
    label: str
    score: float = Field(ge=0.0, le=100.0)
    severity: Severity
    headline: str
    finding_ids: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Measurements — one model per DSP module
# ---------------------------------------------------------------------------


class LoudnessMeasurement(BaseModel):
    integrated_lufs: float
    momentary_max_lufs: float
    short_term_max_lufs: float
    loudness_range_lu: float
    true_peak_dbtp: float
    sample_peak_dbfs: float
    plr_db: float = Field(description="Peak to Loudness Ratio: true peak - integrated")
    psr_p10_db: float = Field(description="10th-percentile short-term PSR; low = squashed")
    psr_median_db: float
    short_term_series: List[float] = Field(default_factory=list)
    momentary_series: List[float] = Field(default_factory=list)
    times: List[float] = Field(default_factory=list)


class ClippingMeasurement(BaseModel):
    sample_peak_dbfs: float
    true_peak_dbtp: float
    clipped_samples: int
    clip_percentage: float
    longest_flat_run: int = Field(description="Longest consecutive at-ceiling run, in samples")
    flat_run_count: int
    inter_sample_overs: int
    is_float_over_unity: bool
    distortion_index: float = Field(
        ge=0.0, le=1.0, description="High-frequency residual energy consistent with clipping"
    )
    events: List[Moment] = Field(default_factory=list)


class SpectralMeasurement(BaseModel):
    third_octave_centers: List[float]
    third_octave_db: List[float]
    bands: List[Band]
    spectral_tilt_db_per_decade: float
    spectral_centroid_hz: float
    mud_ratio_db: float = Field(description="150-400 Hz relative to 60-120 Hz")
    mud_to_mid_db: float = Field(description="150-400 Hz relative to 1-3 kHz")
    boxiness_db: float
    harshness_index: float = Field(ge=0.0, le=1.0)
    sibilance_index: float = Field(ge=0.0, le=1.0)
    sharpness_acum: float = Field(description="Zwicker-style psychoacoustic sharpness")
    resonances: List["Resonance"] = Field(default_factory=list)
    band_series: Dict[str, List[float]] = Field(
        default_factory=dict, description="Per-macro-band level over time, dB"
    )
    times: List[float] = Field(default_factory=list)


class Resonance(BaseModel):
    freq_hz: float
    q: float
    prominence_db: float
    severity: Severity
    moments: List[Moment] = Field(default_factory=list)


class StereoMeasurement(BaseModel):
    is_mono_source: bool
    correlation: float
    width: float = Field(description="Side/Mid RMS ratio, broadband")
    band_correlation: Dict[str, float] = Field(default_factory=dict)
    band_width: Dict[str, float] = Field(default_factory=dict)
    mono_sum_loss_db: float
    band_mono_loss_db: Dict[str, float] = Field(default_factory=dict)
    low_end_side_energy_db: float = Field(description="Side energy below 120 Hz, relative to mid")
    balance_db: float = Field(description="L/R level asymmetry; + means right-heavy")
    correlation_series: List[float] = Field(default_factory=list)
    width_series: List[float] = Field(default_factory=list)
    times: List[float] = Field(default_factory=list)


class PhaseMeasurement(BaseModel):
    correlation: float
    polarity_inverted: bool
    mono_compatible: bool
    mono_sum_loss_db: float
    worst_band: Optional[str] = None
    worst_band_correlation: float = 1.0
    band_mono_loss_db: Dict[str, float] = Field(default_factory=dict)
    problem_moments: List[Moment] = Field(default_factory=list)


class DynamicsMeasurement(BaseModel):
    crest_factor_db: float
    peak_to_loudness_db: float
    macro_dynamics_lu: float = Field(description="Section-to-section variation (LRA-like)")
    micro_dynamics_db: float = Field(description="Short-window crest; survives limiting or not")
    dr_value: float = Field(description="TT-DR style value")
    rms_db: float
    pumping_index: float = Field(ge=0.0, le=1.0, description="Envelope modulation locked to tempo")
    pumping_rate_hz: float
    gain_reduction_estimate_db: float
    crest_series: List[float] = Field(default_factory=list)
    times: List[float] = Field(default_factory=list)


class TransientMeasurement(BaseModel):
    onset_density: float
    estimated_tempo: float
    attack_time_ms: float
    punch_index: float = Field(ge=0.0, le=1.0, description="Transient peak vs 50 ms sustain")
    transient_to_sustain_db: float
    band_punch: Dict[str, float] = Field(default_factory=dict)
    smearing_index: float = Field(ge=0.0, le=1.0)
    weak_moments: List[Moment] = Field(default_factory=list)


class LowEndMeasurement(BaseModel):
    kick_detected: bool
    kick_fundamental_hz: float
    kick_count: int
    bass_fundamental_hz: float
    sub_energy_db: float
    kick_bass_collision_db: float = Field(description="Shared energy in the overlap region")
    ducking_depth_db: float = Field(description="Sub level drop at kick hits; sidechain evidence")
    has_sidechain: bool
    kick_definition_db: float = Field(description="Kick transient above surrounding low-end")
    low_end_mono_ratio: float = Field(ge=0.0, le=1.0)
    sub_rumble_db: float = Field(description="Energy below 25 Hz")
    collision_moments: List[Moment] = Field(default_factory=list)


VocalProminence = Literal["absent", "tucked", "balanced", "forward"]


class VocalMeasurement(BaseModel):
    vocal_present: bool
    #: How sure the voice test is, 0-1. `vocal_present` is this crossing a
    #: threshold, so a detector that wants to be gentle about a borderline call
    #: reads the float rather than the boolean.
    vocal_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Graded confidence that a real lead voice is present, 0-1",
    )
    #: Where the lead sits against the bed. A vocal tucked under the
    #: instruments is a production decision on a beat, not a balance fault, and
    #: this is what lets the detector layer tell the two apart.
    vocal_prominence: VocalProminence = Field(
        default="absent",
        description="absent / tucked / balanced / forward, from the centre-to-instrument ratio",
    )
    center_energy_ratio: float = Field(ge=0.0, le=1.0)
    vocal_to_instrument_db: float = Field(description="Centre 300-6k vs sides + non-vocal bands")
    intelligibility_index: float = Field(ge=0.0, le=1.0)
    presence_balance_db: float
    sibilance_db: float
    consistency_db: float = Field(description="Spread of vocal level over time; high = uneven")
    masked_bands: List[str] = Field(default_factory=list)
    buried_moments: List[Moment] = Field(default_factory=list)
    loud_moments: List[Moment] = Field(default_factory=list)


class ClarityMeasurement(BaseModel):
    clarity_index: float = Field(ge=0.0, le=1.0)
    spectral_flatness: float
    spectral_contrast: float
    masking_index: float = Field(ge=0.0, le=1.0)
    band_congestion: Dict[str, float] = Field(default_factory=dict)
    worst_congested_band: Optional[str] = None
    definition_db: float
    congested_moments: List[Moment] = Field(default_factory=list)


StemKind = Literal["vocals", "drums", "bass", "other"]

STEM_KINDS: Tuple[StemKind, ...] = ("vocals", "drums", "bass", "other")


class StemMeasurement(BaseModel):
    """One separated source, measured on its own.

    Everything here is inferred from the 2-track without separation — badly.
    A centre-channel proxy cannot tell a lead vocal from a centred synth, and
    it cannot measure the kick and the 808 as separate objects at all. With
    stems these become direct measurements instead of guesses, which is why
    `confidence` on the findings they feed rises when this is populated.
    """

    kind: StemKind
    present: bool
    level_lufs: float
    level_ratio_db: float = Field(description="Stem loudness relative to the full mix")
    peak_dbfs: float
    crest_factor_db: float
    micro_dynamics_db: float
    gain_reduction_estimate_db: float
    spectral_centroid_hz: float
    band_levels_db: Dict[str, float] = Field(default_factory=dict)
    band_occupancy: Dict[str, float] = Field(
        default_factory=dict, description="Share of each macro band this stem owns, 0-1"
    )
    stereo_width: float
    correlation: float
    transient_punch: float
    onset_count: int = 0
    active_ratio: float = Field(default=0.0, description="Fraction of the track this stem sounds on")
    level_series: List[float] = Field(default_factory=list)


class MaskingPair(BaseModel):
    """One source burying another in a specific band, measured not guessed."""

    masker: StemKind
    maskee: StemKind
    band: str
    low_hz: float
    high_hz: float
    overlap_db: float = Field(description="How far the maskee sits under the masker's skirt")
    severity: Severity
    detail: str = ""
    moments: List[Moment] = Field(default_factory=list)


class StemAnalysis(BaseModel):
    """Per-source analysis. Absent unless separation was requested and succeeded."""

    # `model_name` collides with Pydantic's protected `model_` namespace, which
    # emits a UserWarning on every import. The field name is part of the wire
    # contract, so the namespace guard is what gives way.
    model_config = {"protected_namespaces": ()}

    available: bool = False
    model_name: str = ""
    separation_ms: int = 0
    stems: List[StemMeasurement] = Field(default_factory=list)
    masking_pairs: List[MaskingPair] = Field(default_factory=list)
    vocal_to_instrument_db: Optional[float] = None
    kick_to_bass_db: Optional[float] = None
    kick_fundamental_hz: Optional[float] = None
    bass_fundamental_hz: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)


class Section(BaseModel):
    """One structural span of the track, measured on its own.

    A single set of numbers for a whole record hides the thing producers
    actually get wrong: the chorus that fails to lift, the drop where the low
    end collapses, the verse that is 4 dB quieter than everything around it.
    """

    index: int
    label: str = Field(description="intro / verse / chorus / drop / bridge / outro / section N")
    t_start: float
    t_end: float
    integrated_lufs: float
    crest_factor_db: float
    stereo_width: float
    band_levels_db: Dict[str, float] = Field(default_factory=dict)
    is_loudest: bool = False
    is_quietest: bool = False


class SectionAnalysis(BaseModel):
    available: bool = False
    sections: List[Section] = Field(default_factory=list)
    loudness_spread_lu: float = 0.0
    peak_lift_db: float = Field(
        default=0.0, description="Loudest section vs the median — does the chorus actually lift?"
    )
    low_end_swing_db: float = Field(
        default=0.0, description="Sub/low-bass variation across sections"
    )
    notes: List[str] = Field(default_factory=list)


class OwnedPlugin(BaseModel):
    """A plugin the producer actually has, with what it can do.

    The capability list is the point. Knowing the user owns "Pro-Q 3" is worth
    little on its own; knowing they own something with `eq_dynamic` changes the
    prescription from "automate a static cut" to "dynamic band, threshold here,
    range there". Names go stale, capabilities do not.
    """

    name: str
    manufacturer: Optional[str] = None
    category: str = ""
    capabilities: List[str] = Field(default_factory=list)


class Measurements(BaseModel):
    """Everything the DSP layer measured. Deterministic, genre-independent."""

    duration_seconds: float
    sample_rate: int
    original_sample_rate: int
    is_mono: bool
    bit_depth: Optional[int] = None

    loudness: LoudnessMeasurement
    clipping: ClippingMeasurement
    spectral: SpectralMeasurement
    stereo: StereoMeasurement
    phase: PhaseMeasurement
    dynamics: DynamicsMeasurement
    transients: TransientMeasurement
    low_end: LowEndMeasurement
    vocal: VocalMeasurement
    clarity: ClarityMeasurement

    # Optional depth. Both default to an "unavailable" object rather than None
    # so consumers can read `.available` without a null check on every access.
    stems: StemAnalysis = Field(default_factory=StemAnalysis)
    sections: SectionAnalysis = Field(default_factory=SectionAnalysis)


class PlatformTarget(BaseModel):
    """Where the mix lands against one streaming/playback target."""

    platform: str
    target_lufs: float
    max_true_peak_dbtp: float
    delta_lufs: float
    will_be_turned_down: bool
    peak_ok: bool
    verdict: Verdict
    note: str


class ReferenceDelta(BaseModel):
    """Measured difference against an uploaded reference track."""

    integrated_lufs: float
    dynamic_range_db: float
    stereo_width: float
    true_peak_dbtp: float
    band_deltas: Dict[str, float] = Field(default_factory=dict)
    third_octave_delta_db: List[float] = Field(default_factory=list)
    similarity: float = Field(ge=0.0, le=100.0)
    biggest_gaps: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# AI layer output
# ---------------------------------------------------------------------------


class Move(BaseModel):
    """One concrete, executable action in a producer's DAW."""

    order: int
    target: str = Field(description="What to apply it to: 'lead vocal', 'bass bus', 'master'")
    action: str = Field(description="Imperative one-liner: 'Cut 3.5 dB at 280 Hz, Q 1.4'")
    tool: str = Field(default="", description="Plugin or stock-device suggestion")
    settings: Dict[str, str] = Field(default_factory=dict)
    why: str = ""


class Prescription(BaseModel):
    """The engineer's fix for one finding: what's wrong, what to do, when to stop."""

    finding_id: str
    dimension: Dimension
    headline: str
    diagnosis: str = Field(description="What this sounds like and why it's happening")
    root_cause: str
    moves: List[Move] = Field(default_factory=list)
    alternative: str = Field(default="", description="A different route to the same result")
    do_not: str = Field(default="", description="The mistake people make fixing this")
    done_when: str = Field(default="", description="The audible/measurable stop condition")
    expected_gain: float = Field(default=0.0, description="Health-score points if fixed")
    minutes: int = Field(default=5)
    difficulty: Literal["easy", "moderate", "advanced"] = "moderate"


class EngineerReport(BaseModel):
    """The AI layer's contribution. Language and judgement only — never numbers
    it invented; every figure it cites comes from Measurements."""

    verdict: str = Field(description="Two or three sentences, spoken like a real engineer")
    the_one_thing: str = Field(description="The single highest-leverage fix")
    strengths: List[str] = Field(default_factory=list)
    prescriptions: List[Prescription] = Field(default_factory=list)
    session_plan: List[str] = Field(default_factory=list)
    mastering_verdict: str = ""
    reference_notes: List[str] = Field(default_factory=list)


class MixAnalysis(BaseModel):
    """The complete response. Everything the UI needs, in one payload."""

    schema_version: int = 2
    filename: str = ""
    genre: str = ""
    intent: TrackIntent = Field(
        default="full_mix",
        description="What the file is. Decides which findings are even asked for.",
    )

    health_score: float = Field(ge=0.0, le=100.0)
    grade: str = Field(description="A+ .. F")
    ceiling_score: float = Field(
        ge=0.0, le=100.0, description="Score reachable by applying every prescription"
    )
    mastering_ready: bool
    mastering_blockers: List[str] = Field(default_factory=list)

    dimensions: List[DimensionScore] = Field(default_factory=list)
    findings: List[Finding] = Field(default_factory=list)
    measurements: Measurements
    platform_targets: List[PlatformTarget] = Field(default_factory=list)
    reference: Optional[ReferenceDelta] = None
    engineer: Optional[EngineerReport] = None

    waveform_peaks: List[float] = Field(
        default_factory=list, description="Downsampled 0-1 peak envelope for the timeline"
    )
    waveform_rms: List[float] = Field(default_factory=list)
    analysis_ms: int = 0
    warnings: List[str] = Field(default_factory=list)


SpectralMeasurement.model_rebuild()
