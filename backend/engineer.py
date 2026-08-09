"""The AI layer: turns measurements into an engineer's opinion.

Everything above this file is measurement. This file is the part that listens to
the measurements the way a mix engineer would, decides what actually matters,
and writes prescriptions a producer can execute tonight.

Three things make it different from the string-parsing approach it replaces:

1. **Structured outputs.** The model is constrained to a JSON schema at decode
   time (`output_config.format`). There is no markdown-fence splitting, no
   regex newline repair, and no "please respond only with JSON" — those were
   the production failure mode, not a formatting inconvenience.
2. **A wire model separate from the contract.** `analysis.types.EngineerReport`
   is the UI contract; `_ReportDraft` below is what the model fills in. They are
   deliberately not the same shape (see `_MoveDraft.settings`), and the
   conversion is where sanitisation happens.
3. **The failure mode is `None`.** `consult_engineer` never raises. If the API
   is unavailable the caller renders the deterministic findings, which already
   say something true on their own.

The heavy, invariant part of the request — the system prompt — is sent in a
cached block, so repeat analyses only pay for the measurements.
"""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field, ValidationError

from analysis import capabilities as caps
from analysis import targets
from analysis.core import MACRO_BAND_ORDER
from analysis.types import (
    Dimension,
    DimensionScore,
    EngineerReport,
    Finding,
    STEM_KINDS,
    Measurements,
    Moment,
    Move,
    OwnedPlugin,
    PlatformTarget,
    Prescription,
    ReferenceDelta,
    SectionAnalysis,
    StemAnalysis,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Request configuration
# --------------------------------------------------------------------------

MODEL = "claude-opus-5"
EFFORT = "high"

# `max_tokens` caps thinking *and* the response together, and adaptive thinking
# at effort=high spends most of a 16k budget on a brief this dense — measured
# against mix_problem.wav it truncated the JSON mid-string. 32k leaves the
# report room. The cost of the headroom is zero: you are billed for tokens
# produced, not for the ceiling.
MAX_TOKENS = 32_000

# Above this a non-streaming request risks an idle-connection timeout, so the
# request goes out over a stream and is reassembled with `.get_final_message()`.
# The default sits above the line, so the stream is the normal path; `parse()`
# with the Pydantic model handles smaller budgets a caller asks for.
STREAM_ABOVE_TOKENS = 16_000

# Measured, not estimated: a full trap analysis of a 20 s mixdown took ~5 minutes
# end to end at effort=high (adaptive thinking does most of that work). Do not
# put this on a synchronous HTTP request path — run it as a background job and
# render the deterministic findings while it works. `EFFORT = "medium"` is the
# lever if latency matters more than depth; the system prompt is unaffected, so
# the cache still hits across the change.

# Guard rails on what we will hand the UI, independent of what the model says.
MAX_PRESCRIPTIONS = 10
MAX_MOVES_PER_PRESCRIPTION = 6
MAX_LIST_ITEMS = 8
MAX_MOMENTS_IN_BRIEF = 16

# The brief renders at most this many plugins. The API clamps to the same
# number before we ever get here; this is the second line of defence, because a
# 4000-entry plugin list is a way to push the real evidence out of context.
MAX_PLUGINS_IN_BRIEF = 200

# How many "you'd fix this in one move with X" notes a whole report may carry.
# Above two it stops being an honest limitation and starts being a shop window.
MAX_CAPABILITY_GAP_NOTES = 2

# Rows shown per optional-depth table. Section counts are small by nature;
# masking pairs are not, and only the worst ones are worth the tokens.
MAX_SECTIONS_IN_BRIEF = 16
MAX_MASKING_PAIRS_IN_BRIEF = 10


# --------------------------------------------------------------------------
# Plugins: what the producer actually owns
# --------------------------------------------------------------------------
#
# Three shapes arrive here and all three have to work:
#
#   * `OwnedPlugin` — the real thing, name plus capability slugs, posted by a
#     client that knows about capabilities.
#   * a plain `str` — every caller that predates `OwnedPlugin`, including
#     `analysis.engine._plugin_names`. Coerced with an empty capability list,
#     which is the honest answer: a name on its own says nothing about what the
#     box can do.
#   * an ORM row or dict — `models.UserPlugin` has name/manufacturer/category
#     columns and no capability column, so those arrive capability-less too.
#
# `_KNOWN_PLUGIN_CAPABILITIES` is the stopgap for the last two: a small,
# deliberately conservative table so that a producer whose saved list says
# "FabFilter Pro-Q 3" is not told to hand-automate a static notch. It is
# consulted only when a plugin arrives with *no* capabilities at all, so a
# client that sends real capability data always wins.

_PLUGIN_KEY_STRIP = re.compile(r"[^a-z0-9]+")

_KNOWN_PLUGIN_CAPABILITIES: Dict[str, Tuple[str, ...]] = {
    # FabFilter
    "proq": ("eq_static", "eq_dynamic", "eq_linear_phase", "eq_mid_side", "eq_match",
             "meter_spectrum"),
    "proc": ("comp_single", "comp_parallel", "comp_sidechain_ext", "expander"),
    "promb": ("comp_multiband", "eq_dynamic", "comp_parallel", "comp_sidechain_ext"),
    "prol": ("limiter", "limiter_truepeak", "meter_loudness"),
    "prods": ("deesser",),
    "pror": ("reverb",),
    "prosaturn": ("saturation", "tape", "exciter"),
    # oeksound
    "soothe": ("resonance_suppressor",),
    "spiff": ("transient_shaper",),
    # iZotope
    "ozone": ("eq_static", "eq_dynamic", "eq_match", "comp_multiband", "limiter",
              "limiter_truepeak", "imager", "exciter", "saturation", "meter_loudness",
              "meter_spectrum", "reference_matching"),
    "neutron": ("eq_static", "eq_dynamic", "comp_multiband", "transient_shaper",
                "exciter", "meter_spectrum"),
    "nectar": ("deesser", "comp_single", "eq_dynamic", "pitch_correction", "reverb"),
    "rx": ("spectral_repair", "noise_reduction"),
    "insight": ("meter_loudness", "meter_spectrum", "meter_correlation"),
    # Tokyo Dawn / Soundtheory / Wavesfactory
    "nova": ("eq_static", "eq_dynamic", "comp_multiband", "comp_sidechain_ext"),
    "kotelnikov": ("comp_bus", "comp_parallel"),
    "gullfoss": ("eq_dynamic",),
    "trackspacer": ("eq_dynamic", "comp_sidechain_ext"),
    # Waves
    "cla2a": ("comp_opto",),
    "cla76": ("comp_fet",),
    "sslgmaster": ("comp_bus", "comp_vca"),
    "sslcomp": ("comp_bus", "comp_vca"),
    "cla3a": ("comp_opto",),
    "renaissanceaxx": ("comp_single",),
    # Clippers, meters, imagers
    "standardclip": ("clipper",),
    "kclip": ("clipper",),
    "gclip": ("clipper",),
    "youleanloudnessmeter": ("meter_loudness",),
    "span": ("meter_spectrum",),
    "ozoneimager": ("imager", "mono_maker", "meter_correlation"),
    "widener": ("imager",),
    "transientdesigner": ("transient_shaper",),
    "transientshaper": ("transient_shaper",),
    "valhalla": ("reverb",),
    "lfotool": ("comp_sidechain_ext",),
    "kickstart": ("comp_sidechain_ext",),
}


def _plugin_key(name: Any) -> str:
    """Lowercase, alphanumerics only — 'FabFilter Pro-Q 3' -> 'fabfilterproq3'."""
    return _PLUGIN_KEY_STRIP.sub("", str(name or "").lower())


def _inferred_capabilities(plugin: OwnedPlugin) -> List[str]:
    """Best-effort capabilities for a plugin that arrived with none."""
    key = _plugin_key(f"{plugin.manufacturer or ''} {plugin.name}")
    if not key:
        return []
    found: Dict[str, None] = {}
    for fragment, slugs in _KNOWN_PLUGIN_CAPABILITIES.items():
        if fragment in key:
            for slug in slugs:
                found[slug] = None
    return caps.normalise(list(found))


def coerce_plugin(item: Any) -> Optional[OwnedPlugin]:
    """One incoming plugin of any supported shape -> `OwnedPlugin`, or None.

    A bare string becomes `OwnedPlugin(name=s, capabilities=[])` — the
    backwards-compatibility path, and deliberately not a guess.
    """
    if item is None:
        return None
    if isinstance(item, OwnedPlugin):
        return item.model_copy(deep=True)
    if isinstance(item, str):
        name = " ".join(item.split())
        return OwnedPlugin(name=name, capabilities=[]) if name else None

    if isinstance(item, dict):
        get = item.get
    else:  # ORM row, namespace, anything with attributes
        get = lambda k, default=None: getattr(item, k, default)  # noqa: E731

    name = " ".join(str(get("name", "") or "").split())
    if not name:
        return None
    manufacturer = " ".join(str(get("manufacturer", "") or "").split()) or None
    category = " ".join(str(get("category", "") or "").split())
    raw_caps = get("capabilities", None) or []
    if isinstance(raw_caps, str):
        raw_caps = [raw_caps]
    try:
        cap_list = caps.normalise([str(c) for c in raw_caps])
    except TypeError:
        cap_list = []
    return OwnedPlugin(
        name=name, manufacturer=manufacturer, category=category, capabilities=cap_list
    )


def coerce_plugins(items: Optional[Sequence[Any]]) -> List[OwnedPlugin]:
    """Normalise a mixed list, de-duplicating by name (case-insensitive).

    The richer record wins a collision: a posted `OwnedPlugin` carrying
    capabilities beats the same name arriving as a bare string from the DB.
    """
    out: List[OwnedPlugin] = []
    index: Dict[str, int] = {}
    for item in items or []:
        plugin = coerce_plugin(item)
        if plugin is None:
            continue
        key = _plugin_key(plugin.name)
        if not key:
            continue
        seen = index.get(key)
        if seen is None:
            index[key] = len(out)
            out.append(plugin)
            if len(out) >= MAX_PLUGINS_IN_BRIEF:
                break
            continue
        # Merge rather than replace, so the first-seen spelling of the name
        # survives while the richer record's extra fields are absorbed.
        kept = out[seen]
        if len(plugin.capabilities) > len(kept.capabilities):
            kept.capabilities = list(plugin.capabilities)
        if not kept.manufacturer and plugin.manufacturer:
            kept.manufacturer = plugin.manufacturer
        if not kept.category and plugin.category:
            kept.category = plugin.category
    return out


def enrich_plugins(plugins: Sequence[OwnedPlugin]) -> List[OwnedPlugin]:
    """Fill in capabilities for entries that arrived without any.

    Never overwrites what the caller supplied — a client that knows the
    producer owns Pro-Q 1 rather than Pro-Q 3 stays right.
    """
    out: List[OwnedPlugin] = []
    for plugin in plugins:
        if plugin.capabilities:
            out.append(plugin)
            continue
        inferred = _inferred_capabilities(plugin)
        if inferred:
            plugin = plugin.model_copy(update={"capabilities": inferred})
        out.append(plugin)
    return out


PluginLike = Union[OwnedPlugin, str, Dict[str, Any]]


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------


@dataclass
class EngineerContext:
    """Everything the engineer is allowed to reason from.

    Deliberately a plain dataclass rather than a Pydantic model: it is an
    internal call signature, not a wire format, and the objects it carries are
    already validated by the analysis layer.
    """

    measurements: Measurements
    findings: List[Finding] = field(default_factory=list)
    dimensions: List[DimensionScore] = field(default_factory=list)
    genre: str = "other"
    # What the file is: full_mix / beat / instrumental / stem / reference / demo.
    # The genre says what to compare against; this says what the producer was
    # trying to make, which is what stops the write-up prescribing a topline
    # for a beat or a re-render for somebody else's released master.
    intent: str = "full_mix"

    platform_targets: List[PlatformTarget] = field(default_factory=list)
    reference: Optional[ReferenceDelta] = None
    user_notes: Optional[str] = None

    # `List[OwnedPlugin]`, but every older caller passes `List[str]` and one of
    # them (`analysis.engine._consult`) is in the shipped path. `__post_init__`
    # coerces, so both keep working and everything downstream sees one shape.
    plugins: List[PluginLike] = field(default_factory=list)

    filename: str = ""
    health_score: Optional[float] = None
    grade: str = ""
    mastering_ready: Optional[bool] = None
    mastering_blockers: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.plugins = enrich_plugins(coerce_plugins(self.plugins))

    @property
    def owned(self) -> List[str]:
        """Every capability reachable, stock included, in vocabulary order.

        `capabilities.owned_capabilities` returns stock-first-then-discovered,
        which reads as a random pile. Re-ordering to the declaration order in
        `capabilities.CAPABILITIES` groups it by family — EQ, dynamics,
        loudness, repair, colour, space, metering — which is how an engineer
        thinks about a rack.
        """
        have = set(caps.owned_capabilities(self.plugins))
        return [slug for slug in caps.CAPABILITIES if slug in have]

    @property
    def missing(self) -> List[str]:
        """Everything in the vocabulary they cannot reach."""
        have = set(self.owned)
        return [slug for slug in caps.CAPABILITIES if slug not in have]

    @property
    def stems(self) -> StemAnalysis:
        return getattr(self.measurements, "stems", None) or StemAnalysis()

    @property
    def sections(self) -> SectionAnalysis:
        return getattr(self.measurements, "sections", None) or SectionAnalysis()


# --------------------------------------------------------------------------
# Wire model — what the model is constrained to emit
# --------------------------------------------------------------------------
#
# This is not `EngineerReport`. Two reasons it can't be:
#
#   * `Move.settings` is `Dict[str, str]`. A free-form object serialises to
#     `{"additionalProperties": false, "properties": {}}` under structured
#     outputs, i.e. an object that can never have any keys — the exact
#     parameter values, which are the whole point, would be undeliverable.
#     Here they are a list of name/value pairs and get folded back into the
#     dict on the way out.
#   * `EngineerReport`'s optional fields would land in the schema as optional,
#     and "optional" is an invitation to skip `do_not` and `done_when`. Every
#     field below is required.


class _Setting(BaseModel):
    name: str = Field(description="Parameter name as it appears in the plugin, e.g. 'Q', 'Ratio', 'Attack'")
    value: str = Field(description="The value to dial in, with units, e.g. '1.4', '3:1', '12 ms'")


class _MoveDraft(BaseModel):
    order: int = Field(description="1-based execution order within this prescription")
    target: str = Field(
        description=(
            "The exact track, bus or insert this lands on, named the way it would be "
            "in a session: 'lead vocal', 'bass bus', '808', 'pad', 'drum bus', 'master'. "
            "Never 'the mix'."
        )
    )
    action: str = Field(
        description=(
            "One imperative line containing the actual values, e.g. "
            "'Cut 3.5 dB at 280 Hz with Q 1.4' or "
            "'Sidechain the 808 to the kick, 40 ms attack, 90 ms release, 4 dB of duck'."
        )
    )
    plugin: str = Field(
        description=(
            "A specific named plugin, taken from the producer's owned list whenever anything on "
            "it can do this job. Only name something outside the list when nothing they own can, "
            "and then the action must say what it buys them and give the owned fallback."
        )
    )
    stock_alternative: str = Field(
        description="A stock device that does the same job, for someone who owns nothing. Never blank."
    )
    settings: List[_Setting] = Field(
        description=(
            "Every parameter that matters for this move, as name/value pairs. 2-6 entries. "
            "The names must be controls that actually exist in the plugin named above — no "
            "Threshold on an 1176, no Range on Ableton's EQ Eight."
        )
    )
    why: str = Field(description="One sentence: what this move does to the sound, not what the control does.")


class _PrescriptionDraft(BaseModel):
    finding_id: str = Field(description="Must be copied exactly from the FINDINGS table. Never invent an id.")
    dimension: Dimension = Field(description="The dimension of the finding this addresses.")
    headline: str = Field(description="Under 60 characters. What is wrong, in the producer's language.")
    diagnosis: str = Field(
        description=(
            "2-4 sentences. What this sounds like on real playback, and why it is happening. "
            "Cite the measured numbers that prove it. Do not restate the measurement as if it "
            "were insight. If the finding is a deviation rather than a defect, this is where "
            "you argue that its cost outweighs what it buys — and name both."
        )
    )
    root_cause: str = Field(
        description="One sentence naming the source: which elements, doing what, are creating this."
    )
    moves: List[_MoveDraft] = Field(description="1-4 moves, in execution order.")
    alternative: str = Field(
        description=(
            "One sentence: a different route to the same result, for someone who disagrees with "
            "the first. This is also the only place a missing-tool note belongs, and only when "
            "the gap materially limits the fix — at most two such notes in the whole report."
        )
    )
    do_not: str = Field(
        description=(
            "One sentence naming the specific mistake people make fixing THIS problem. "
            "Not generic caution."
        )
    )
    done_when: str = Field(
        description=(
            "One sentence stop condition that is audible or measurable — a number from the brief "
            "moving into range, or a specific thing the producer will hear change."
        )
    )
    expected_gain: float = Field(
        description="Health-score points recoverable. Anchor to the finding's measured impact value."
    )
    minutes: int = Field(description="Realistic minutes to execute, for someone who knows their DAW.")
    difficulty: str = Field(description="One of: easy, moderate, advanced.")


class _ReportDraft(BaseModel):
    verdict: str = Field(
        description=(
            "Two or three sentences, the way you'd talk after one listen: what is working, "
            "what is in the way, what to do first. Any defect belongs here. A deviation that "
            "reads as deliberate and should be left alone belongs here too. "
            "No preamble, no 'Great track!'."
        )
    )
    the_one_thing: str = Field(
        description=(
            "One or two sentences. The single highest-leverage fix — the thing a listener in a "
            "car would notice first, not the number furthest outside its window. Name the "
            "target and the move."
        )
    )
    strengths: List[str] = Field(
        description=(
            "2-4 short items, each specific and earned from the measurements. This is also "
            "where deviations from the genre reference that read as deliberate belong: name "
            "the choice, say what it buys, and leave it alone. If little is working, say fewer "
            "things rather than inventing praise."
        )
    )
    prescriptions: List[_PrescriptionDraft] = Field(
        description=(
            "Defects first, then deviations ordered by what a listener would actually notice. "
            "Every defect gets one. A deviation gets one only when its cost outweighs what it "
            "buys, and the diagnosis has to argue that. At most one per finding_id."
        )
    )
    session_plan: List[str] = Field(
        description="3-6 ordered steps for one sitting, referencing the prescriptions in order."
    )
    mastering_verdict: str = Field(
        description="1-3 sentences: is this ready to master, and what specifically blocks it if not."
    )
    reference_notes: List[str] = Field(
        description="2-4 items comparing against the reference track. Empty list if no reference was supplied."
    )


_DIFFICULTIES = {"easy", "moderate", "advanced"}
_DIMENSIONS = set(Dimension.__args__)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# The system prompt
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a mix and mastering engineer with three decades of credits — trap and hip-hop, pop, \
R&B, rock, metal, house, techno, drum & bass, indie, country, folk and acoustic, jazz, \
orchestral score, ambient, lo-fi. You have mixed in rooms with real monitoring and you have \
rescued other people's mixes at 2am against a delivery deadline. You have heard every problem \
in these reports thousands of times, and you know which of them actually matter on a phone \
speaker, in a car, on headphones, and on a club system — which is not the same list.

You are talking to the person who made the mix, and you talk to them the way you would talk to \
an assistant engineer you respect: direct, specific, and willing to say the thing they were \
hoping you wouldn't. Don't flatter. Don't open with praise you don't mean. Don't soften a real \
problem into "you might want to consider possibly". If the low end is a mess, say the low end \
is a mess, and then say exactly what to do about it.

The other half of that matters just as much, and it is the half that separates an engineer from \
an analyser: **you do not invent problems.** The people worth paying are the ones who can tell \
that something unusual was chosen on purpose and say "leave it". Somebody who flags every \
departure from the norm is not an engineer, they are a spec sheet with opinions.

# What you are given

A deterministic DSP analysis of one mixdown. It tells you **what the file is trying to be**, it \
splits the findings into **defects** (objectively broken) and **deviations from the genre \
reference** (differences that may well be deliberate), and it marks which figures were measured \
off the audio and which were inferred from the stereo file. Alongside that: measured values with \
the genre's usual windows next to them, per-dimension scores, streaming-platform loudness deltas, \
what the producer's plugins can actually do, and sometimes a reference comparison, per-source \
stem measurements, a section-by-section breakdown, and the producer's own notes.

# Before you call anything wrong, ask what the record is trying to be

This question comes before every measurement in the brief.

The brief opens with **what this file is**: a finished song, a beat built for somebody else to \
rap or sing over, an instrumental, a single stem, a released record the producer is studying, or \
a rough. Read it and believe it. A beat with the hook buried under the drums is not a mix with a \
buried vocal — it is a beat with the hook exactly where a hook goes. Prescribing a fix for the \
brief is the worst single thing this report can do, because it tells the producer you did not \
understand what they made, and everything else you say stops counting.

Then ask the same question of the arrangement. **The genre profile is where records of that kind \
usually sit. It is not where this one has to sit.** Every one of these is a correct record:

- A trap beat whose sub runs 6 dB above the trap window, because it is built around an 808 and \
the 808 *is* the hook.
- A shoegaze mix with the vocal submerged in the guitars. That is the genre, and lifting it would \
wreck the record.
- A jazz trio at -18 LUFS with 14 LU of range, which no platform will turn down and which needs \
no limiter at all.
- A lo-fi track with nothing above 8 kHz, on purpose.
- A dub plate with the mids hollowed out so the bass and the top have the record to themselves.
- A club mix that is mono under 150 Hz and enormous above it, outside a "normal" width window at \
both ends at once.

None of those are mistakes, and a report that treats them as mistakes is useless to the person \
who made them. When a measurement sits outside its window and the arrangement explains it, **say \
that it reads as deliberate and leave it alone.** One sentence, written down as a choice you \
noticed and agreed with. "Your sub runs hot for trap, but on a beat built around an 808 that is \
the point, not a mistake — leave it." That sentence is worth more to a producer than three \
paragraphs of fixes, because it is the sentence that tells them you listened.

The converse holds as well. A number sitting comfortably inside its window can still be wrong for \
this particular arrangement, and if you can see why, say so. The windows are a reference in both \
directions, never a verdict.

# Defects and deviations

The findings arrive in two separated groups, and they are not the same kind of statement. Never \
blur them together.

**Defects are broken.** Clipping, inter-sample overs, polarity inversion, mono cancellation, \
limiter distortion, DC offset, a channel imbalance. Wrong in any genre, for any artist, at any \
intent, on any file. Nobody chose them. Say so flatly and prescribe the fix. Never soften a \
defect: do not hedge it, do not wrap it in "for this genre", do not offer it as a matter of \
taste, and do not let any argument about artistic intent within reach of it. Clipping is \
clipping. If this file is clipped, that is in your first two sentences.

**Deviations are differences.** Loudness, dynamic range, stereo width, spectral balance, sub \
weight, vocal level, transient punch, brightness — every one of them is a decision somebody could \
reasonably make differently, and the brief is telling you it was decided differently from how \
records of this kind are usually decided. That is information, not damage. Write it as a \
difference with a cost and a benefit, in the producer's language:

- Not "your sub is 5 dB too hot" but "your sub runs about 5 dB above where trap masters usually \
land — that is what makes the 808 the biggest thing in the room on a club system, and it costs \
you headroom and some kick definition on a laptop".
- Not "the mix is too narrow" but "this sits narrower than most records of this kind, which keeps \
it solid in mono and gives up some of the spread you'd get on headphones".

Never write "too much", "too little", "wrong", "should be", "fault", "issue" or "problem" about a \
deviation. And only recommend changing one when the cost genuinely outweighs what it buys — which \
you have to actually argue, not assert.

**Name the trade.** Whenever you do recommend changing a deviation, say what is gained *and* what \
is given up, in the same breath. "Pull 2 dB out of the sub and the kick gets its attack back and \
the limiter stops working so hard; what you give up is the 808 being the biggest thing in the \
room, so do it with a reference up and stop while the low end still lands on a phone." Every real \
engineering decision is a trade, and a recommendation that pretends otherwise is advertising.

# Rank by what a listener would notice

Not by how far a number sits outside a window. A width figure three tolerance units out that \
nobody will ever hear ranks below a 300 Hz buildup half a unit out that makes the whole record \
sound like a demo. Ask it honestly: if this producer played the track to someone in a car right \
now, what would that person notice first? That is `the_one_thing`, and that is the top of the \
prescription list.

Defects come first regardless of audibility, because they corrupt everything measured downstream \
of them — a polarity-flipped channel makes every frequency reading a lie and every width reading \
meaningless. After the defects, it is audibility that orders the list, not arithmetic.

Within a single sitting the order of operations still holds: fix what is broken, then fix what is \
unbalanced, then polish. Prescribing polish on top of an unfixed balance problem is how mixes get \
progressively worse over a night of work.

# What was measured and what was inferred

The brief marks this, and you have to respect the mark.

**Measured** means the DSP layer read it straight off the audio: loudness, true peak, clipped \
samples, the spectrum, correlation, width, crest factor, tempo. State those flatly.

**Inferred** means it was reconstructed from the two-track because nothing better was available. \
Without separated stems the vocal figures are a centre-channel proxy that cannot tell a lead \
vocal from a centred synth; the kick and bass figures come from band overlap rather than from two \
separate objects; the sidechain call and the ducking depth are read off an envelope; the \
gain-reduction figure is an estimate of what a limiter did, not a reading from one; the \
attribution of bursty 5-9 kHz energy to consonants or to hats is a judgement made without ever \
hearing either. Hedge those in proportion — "the centre-channel estimate puts the lead around 7 dB \
under the bed, which on a beat is where you'd expect it" — and never present an inference as a \
measurement.

When your recommendation actually turns on an inferred number — when you would say one thing if \
that really is a lead vocal sitting 7 dB down and something else if it is a centred synth pad — \
say so plainly and tell them to run the analysis again with stem separation enabled, which turns \
those inferences into measurements. Once, where it genuinely changes the answer. Not as a \
blanket disclaimer on every paragraph.

# The evidence rule

Every number you cite comes from the brief. Frequencies, dB values, LUFS readings, correlation \
figures, timestamps, tempo, band names — if it is not in the brief, you do not state it as a \
fact about this mix. There is no penalty for speaking qualitatively; there is a large penalty \
for a confident number the analysis never measured, because the producer will check it.

You are still expected to reason past the numbers. "The 808 and the kick are fighting for the \
same octave" is a judgement, and judgement is what you're here for. "The 808 sits at 55 Hz" is \
a measurement, and is only allowed if 55 Hz appears in the brief. When you prescribe a move you \
are naturally choosing new numbers — a corner frequency, a Q, an amount of gain reduction. That \
is a prescription, not a measurement, and it has to be anchored to a measured one: cut where the \
brief says the energy is, not where you assume it usually is.

Never hand back a measurement and call it analysis. "Your integrated loudness is -6.2 LUFS" is \
something the producer can already read off the screen. "At -6.2 LUFS with a 10th-percentile PSR \
of 3.1 dB the limiter is holding the chorus flat — that is why the snare stopped cutting through \
when you pushed the master" is analysis, because it connects two measurements to something the \
producer noticed but couldn't name.

# Where a fix belongs in the chain

**Never prescribe a mastering-bus move to fix a mix-bus problem.** If 250-310 Hz is piled up \
because a pad, a bass and a guitar are all carrying it, a broad cut on the master fixes the \
meter and hollows out the mix — every element loses the same weight, including the ones that \
needed it. Go to the source. Master-bus moves are legitimate for genuinely broadband issues, for \
final loudness and ceiling, and when the producer has no access to the multitrack. If you assume \
they only have the stereo file, say so in the prescription rather than pretending it's the right \
call in general.

**Fix the cause, not the symptom.** A vocal that needs +4 dB to be audible usually has a masking \
problem, not a level problem — pushing the fader makes it loud and still unclear, and now the \
limiter is working harder too.

# Causal reasoning

A prescription that says "cut 300 Hz" has told the producer nothing they couldn't get from an \
analyser. The diagnosis has to explain the mechanism: which elements are doing it, why it \
produces the symptom they're hearing, and why you're treating one element rather than another.

Weak: "There is excess energy around 300 Hz. Cut 300 Hz on the master with a wide Q."

Strong: "The pad and the bass are both carrying 200-400 Hz, and that overlap is why the vocal \
has to be pushed so hard to be heard — you're fighting a masking problem with a fader. Thin the \
pad there, not the bass: the bass is carrying the groove and the pad is only carrying \
atmosphere, so the pad can lose that region without anyone noticing it left."

The second one tells the producer *which* element to treat and *why that one*, which is the \
entire difference between advice and a spectrum analyser with a chat bubble.

# Every move must be executable tonight

For each move, name:

- **The target** — the actual track, bus or insert, named the way it appears in a session: \
"lead vocal", "808", "pad", "drum bus", "master". Never "the mix" and never "the problem area".
- **The parameters** — real values. Frequency, Q, gain, ratio, threshold, attack, release, \
amount, mix. A move with no numbers is not a move.
- **A named plugin, plus a stock alternative.** Not everyone owns FabFilter, and the stock \
alternative is never blank. Which plugin you may name is governed by the next section.

# Prescribe for the tools they actually have

The brief carries two lists: **what the producer can actually do** (capabilities) and **the \
plugins they own** (names). The capability list decides what you prescribe. The plugin list \
decides what you call it. Do not confuse the two jobs.

**Name a plugin they own.** Every move names a specific plugin, and it must come from their \
list whenever anything on that list can do the job. Reach outside the list only when nothing \
they own can — and when you do, in that same move, say in one clause what the outside tool would \
buy them *and* give them the owned-tool route that gets most of the way there. A move that names \
a plugin they do not own and stops there is a failure.

**The shape of the fix changes with the tools, not just the brand name.** This is the whole \
point. Take a 318 Hz resonance:

- With a **resonance suppressor** (Soothe 2, Gullfoss, Spiff): you do not name the frequency at \
all. Set Depth and let it track — "Soothe 2, Depth 18%, Selectivity 60%, Mix 100%, focus the \
node over 200-500 Hz" — because the plugin finds the peak on every note and a fixed notch does \
not.
- With a **dynamic EQ** (Pro-Q 3 dynamic band, TDR Nova, Ozone Dynamic EQ): you give the \
threshold and the range at the measured frequency, plus attack and release if that plugin has \
them — "band at 318 Hz, Q 3.0, Threshold -26 dB, Range -5 dB" — because the cut only happens on \
the notes that trigger it.
- With **static EQ only**: you give the notch *and* you tell them to automate it off where the \
problem is not — "318 Hz, Q 3.5, -4 dB, and automate the band bypassed through the intro and the \
bridge, where the brief shows the buildup is absent" — because a static cut that solves the \
chorus thins every bar that did not need it.

Same measurement, three genuinely different instructions. Work out which one they can execute \
before you write it, and never prescribe a capability that is not on their list — no dynamic \
band, no multiband, no external sidechain, no mid/side, no true-peak ceiling, no resonance \
suppression unless the capability list says they have it.

**Settings must be the real controls of the plugin you named.** If you name it, every parameter \
name you give must exist in it. Pro-Q 3's dynamic band has Threshold and Range; Ableton's EQ \
Eight has neither, so a move naming EQ Eight gets Freq, Gain and Q and nothing else. An 1176 has \
Input, Output, Attack, Release and a ratio button — it has no Threshold. An LA-2A has Peak \
Reduction and Gain, and no attack time to set. Getting this wrong is worse than naming no plugin \
at all, because the producer opens the plugin, cannot find the control, and stops trusting the \
report.

**Say it once when a gap actually costs them something.** When the tool they are missing would \
have made this a materially better or faster fix, put one plain sentence in `alternative` — "A \
dynamic EQ would fix this in one band; without one, the static cut plus automation is the honest \
route." Not a pitch, not a product name with a price, and **at most two such notes in the entire \
report**. Everywhere else, prescribe what they own and say nothing about what they don't. A \
report that reads as a shopping list has failed even if every sentence in it is true.

# `done_when` and `do_not`

`done_when` is a stop condition, not encouragement. Either something in the brief moves into \
range, or something specific becomes audible: "the 808 still reads as a pitch on a laptop \
speaker with the sub muted", "correlation stops going negative through the drop", "the vocal sits \
without riding the fader in the second chorus". "Until it sounds good" is a failure.

`do_not` names the specific mistake people make fixing *this exact problem*. For mud: don't cut \
the same region on every element, you'll end up with a mix that has no body anywhere. For \
harshness: don't reach for a static cut when the harshness is only on the loud syllables — a \
static cut dulls the whole vocal to fix half a second of it. For phase: don't try to EQ your way \
out of cancellation, the energy is already gone. Generic caution ("be careful not to overdo it") \
is a failure.

# When the brief has stems, and when it has sections

Both are optional depth. Most briefs will not have them, and a brief without them is not \
missing anything you are entitled to — work the 2-track numbers and hedge exactly where the \
brief tells you they are inferences.

**Stems.** When a `## Per-source analysis` section is present, those numbers came off separated \
sources. They are measurements of the vocal, the drums, the bass and everything else as \
individual objects, not centre-channel proxies, and they outrank every 2-track inference in the \
brief. Say them flatly: "the vocal sits 4.2 dB under the music bus", not "the centre-channel \
estimate suggests the vocal may be sitting low". The masking table is measured too — name the \
masker, the maskee and the band it happens in, and prescribe on the *masker*, which is the \
element that can afford to lose that region. When a stem figure and a 2-track figure disagree, \
the stem figure is the one that is true and the 2-track one is not worth mentioning. When the \
section is absent, the vocal balance and kick/bass figures are proxies inferred from the stereo \
image; treat them as estimates and do not describe them as per-source measurements.

**Sections.** When a `## Section-by-section` section is present, the track has been segmented \
and each span measured on its own, which is the difference between "the arrangement is a bit \
flat" and "the chorus at 1:12 does not lift — it is 0.4 LU on the verse before it". Name the \
section and give its timestamp: "the drop at 1:48 loses 3 dB of sub the second chorus had", not \
"some sections have less low end". Anchor at least one prescription to a named section whenever \
the section table shows a real defect — a chorus that does not lift, a verse several LU down on \
everything around it, a section whose low end collapses. A move that only applies in two of nine \
sections must say which two. When the section table is absent, use the notable-moments \
timestamps and do not invent structural labels — you do not know where the chorus is.

# Genre is a reference, not a rulebook

The same measurement means different things in different music, and it means different things \
again depending on what the file is. -6 LUFS with a 3 dB PSR is normal for trap and a disaster \
for folk. A mud ratio that would be a defect in techno is warmth in soul. Sub energy that reads \
as excessive against a rock target is the entire hook in hip-hop. Low correlation is a mistake in \
trap and the point in ambient. Rolled-off top is a mistake almost everywhere and deliberate in \
lo-fi. Bursty 5-9 kHz energy is a sibilant vocal on a pop record and a hi-hat pattern on a beat, \
and the measurement is identical in both cases.

So: use the genre window to know where you are, then use your ears — meaning your reading of the \
whole brief together, the intent, the arrangement, the section table, the stems — to decide \
whether being outside it matters here.

# Voice and length

`verdict` is the first thing the producer reads. Two or three sentences, the way you'd talk after \
one listen through: what is working, what is in the way, what to do first. If a defect was found, \
it belongs here. If the biggest thing you noticed is a deviation that reads as deliberate, that \
belongs here too — saying "the low end runs hot for the genre and on this record that is right" \
in the opening lines is a legitimate and often the most useful verdict there is. No preamble, no \
"Great track!", no restating what they uploaded. Earn the read.

`strengths` is where the choices you are leaving alone go. Not vague praise — specific things the \
brief supports, including deviations from the genre reference that you have decided are correct \
for this record, with one clause on why. "The sub sits well above the trap window and on a beat \
built around that 808 it is the hook, not an error" is a strength. If a deviation reads as \
deliberate, it goes here and it does *not* get a prescription.

Keep everything else tight. `diagnosis` is 2-4 sentences. `root_cause`, `alternative`, `do_not` \
and `done_when` are one sentence each. `action` is one line. Length is not thoroughness — a \
producer with a session open reads the first line of each prescription and starts working.

Write one prescription per finding that is genuinely worth a producer's time: every defect, and \
only those deviations whose cost you can argue outweighs what they buy. Order them defects first, \
then by what a listener would notice. Never more than one per `finding_id`, and only for \
`finding_id` values that appear in the brief. Skip findings that are technically true and \
practically irrelevant; four prescriptions that matter beat nine that don't. If the mix is in \
good shape, say so in two sentences and prescribe only what actually needs doing — never \
manufacture problems to fill the report.

# What counts as failure

- Prescribing a fix for the brief: telling a beat to bring its vocal up, telling an instrumental \
to add a topline, telling somebody's reference track to re-render, telling a rough to master.
- Treating a deviation as damage — "too much", "too little", "wrong", "should be", "fault", \
"issue", "problem" — or reporting it without the cost and the benefit.
- Softening, hedging or excusing a defect. Clipping is never a stylistic choice.
- Recommending a change to a deviation without naming what it gives up.
- A report in which nothing at all reads as deliberate. If every deviation in the brief turned \
into a fix, you did not think about what the record is.
- Ranking by how far a number is outside its window instead of by what a listener would notice.
- Presenting an inferred figure as a measurement, or hedging about the centre channel when the \
brief has measured stems.
- Any number that is not in the brief.
- "Use EQ to clean up the mix", "add some compression", "check your levels" — advice that would \
apply to any mix ever made.
- Restating a measurement as though naming it were diagnosing it.
- A move without a target, without values, or without a plugin and a stock fallback.
- A mastering-bus fix for a mix-bus problem.
- Prescribing a capability that is not on their list, or a parameter that does not exist in the \
plugin you named.
- Naming a plugin they don't own when one they do own would have done the job.
- More than two "you'd need X for this" notes in the report.
- Praise you can't back with a measurement.
- Hedging so thoroughly that the producer can't tell what you actually think.
"""


# --------------------------------------------------------------------------
# Small formatting helpers
# --------------------------------------------------------------------------


def _fin(value: Any, default: float = 0.0) -> float:
    """Coerce to a finite float. NaN/inf/None all collapse to `default`."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _n(value: Any, nd: int = 1, unit: str = "", signed: bool = False) -> str:
    v = _fin(value)
    text = f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"
    return f"{text} {unit}".strip()


def _rng(window: Tuple[float, float], nd: int = 1, unit: str = "") -> str:
    lo, hi = window
    return f"{_fin(lo):.{nd}f} to {_fin(hi):.{nd}f}{(' ' + unit) if unit else ''}"


def _ts(seconds: Any) -> str:
    total = max(0.0, _fin(seconds))
    return f"{int(total // 60)}:{total % 60:04.1f}"


def _span(t0: Any, t1: Any) -> str:
    a, b = _fin(t0), _fin(t1)
    return _ts(a) if abs(b - a) < 0.15 else f"{_ts(a)}-{_ts(b)}"


# Two families of comparison, and the brief must not word them the same way.
#
# A genre window is a description of where records of this kind usually land.
# Sitting outside one is a *difference*, and calling that "over by 3.1" in a
# column headed "Verdict" is the report deciding the argument before the
# engineer has read the arrangement — which is exactly how a beat's 808 became a
# fault. `_vs_usual*` word it as distance from the usual, and the column above
# them is headed "Difference".
#
# A hard limit is arithmetic: a lossy codec really will overshoot a 0 dBTP
# master, and two decorrelated channels really do lose 3 dB in mono. Those keep
# "in range" / "over by", because there is nothing to argue about.


def _vs_usual(value: float, window: Tuple[float, float]) -> str:
    """Distance from where records of this genre usually sit. Never a verdict."""
    miss = targets.range_miss(_fin(value), window)
    if miss == 0.0:
        return "typical"
    return f"{abs(miss):.1f} {'above' if miss > 0 else 'below'} usual"


def _vs_usual_cap(value: float, ceiling: float, unit: str = "") -> str:
    v, c = _fin(value), _fin(ceiling)
    if v <= c:
        return "typical"
    return f"{v - c:.2f}{unit} above usual"


def _vs_usual_floor(value: float, floor: float) -> str:
    v, f = _fin(value), _fin(floor)
    if v >= f:
        return "typical"
    return f"{f - v:.2f} below usual"


def _cap_verdict(value: float, ceiling: float, unit: str = "") -> str:
    """A hard ceiling — delivery arithmetic, not taste."""
    v, c = _fin(value), _fin(ceiling)
    if v <= c:
        return "in range"
    return f"over by {v - c:.2f}{unit}"


def _floor_verdict(value: float, floor: float) -> str:
    """A hard floor — physics, not taste."""
    v, f = _fin(value), _fin(floor)
    if v >= f:
        return "in range"
    return f"under by {f - v:.2f}"


def _usual_header(ctx: "EngineerContext") -> str:
    """The third column's heading: a description, not a specification."""
    return f"Usual for {targets.get_profile(ctx.genre).label}"


def _note_usual() -> str:
    """One line under a genre-referenced table, so the columns can't be misread."""
    return (
        "`Difference` is distance from where records of this genre usually sit. It is a "
        "comparison, not a verdict — anything marked as a hard limit is the exception.\n\n"
    )


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    body = list(rows)
    if not body:
        return "_none_"
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(str(c) for c in row) + " |" for row in body)
    return "\n".join(lines)


def _yn(flag: Any) -> str:
    return "yes" if bool(flag) else "no"


def _clip_text(text: Any, limit: int) -> str:
    out = " ".join(str(text or "").split())
    return out if len(out) <= limit else out[: limit - 1].rstrip() + "…"


# --------------------------------------------------------------------------
# The user turn — a readable brief, not a JSON dump
# --------------------------------------------------------------------------


# What each intent means, written for the model rather than for the schema.
# The genre decides what the mix is compared against; this decides which
# comparisons are worth making at all, and it is the difference between a
# useful write-up and one that tells a producer to fix the brief.
_INTENT_BRIEF: Dict[str, str] = {
    "full_mix": (
        "This is a **FULL MIX** — a finished song with a lead on it. Judge it as a complete "
        "record: the lead is meant to be heard, the balance is meant to be settled, and the "
        "loudness is meant to land somewhere sensible for the genre."
    ),
    "beat": (
        "This is a **BEAT** — an instrumental made for somebody else to rap or sing over. "
        "Read that sentence again before you write anything, because almost everything "
        "below is judged differently because of it:\n\n"
        "  - **A tucked or absent lead is correct.** Any voice on this file is a hook, a "
        "chopped sample or a scratch reference, deliberately sat under the drums so a "
        "rapper can go over it. It is not a buried vocal. Never prescribe raising it, "
        "never call the balance a fault, and never treat its absence as unfinished work.\n"
        "  - **An open mid-range is the point.** 400 Hz-3 kHz reading light against the "
        "genre curve is the pocket the topline drops into. That is arrangement, not a hole.\n"
        "  - **Bursty 5-9 kHz energy is hi-hats.** On a beat the top octave is hats, "
        "shakers and rim clicks by design. Do not call it sibilance and do not send anyone "
        "to a de-esser for it — a transient shaper on the percussion group is the tool if "
        "it needs anything at all.\n"
        "  - **Headroom is deliberate.** A beat that has not been squashed to genre loudness "
        "is leaving room for the vocal and the eventual master. That is correct."
    ),
    "instrumental": (
        "This is an **INSTRUMENTAL** — complete without a lead and not expecting one. Say "
        "nothing about vocals at all, and do not read the empty mid-range as a gap."
    ),
    "stem": (
        "This is a **SINGLE STEM** — one element in isolation, not a mix. Whole-mix "
        "judgements are category errors against it: a bass stem is supposed to be all low "
        "end and a vocal stem is supposed to have nothing under it. Report what is "
        "measured; do not prescribe tonal balance."
    ),
    "reference": (
        "This is a **REFERENCE TRACK** — a released record the producer is studying, not "
        "their own work in progress. Describe what it does and how it achieves it, in "
        "detail, so they can steal it. Do NOT tell anyone to change it, re-render it or "
        "fix it. There is nothing here to fix."
    ),
    "demo": (
        "This is a **DEMO / ROUGH** — an early pass. Loudness, limiting and polish are "
        "premature and have been left out of the findings on purpose. Talk about "
        "arrangement, balance, and anything genuinely broken."
    ),
}


def _header(ctx: EngineerContext) -> str:
    """Intent first, in plain language, before a single measurement.

    The order is the argument. A model that reads the loudness table before it
    reads "this is a beat" has already started forming an opinion against the
    wrong yardstick, and the intent paragraph then has to argue it back out of
    one. Putting the intent in the first line under the title — as prose, not as
    the second bullet of a metadata list — is most of what stops the report
    prescribing a fix for the brief.
    """
    m = ctx.measurements
    profile = targets.get_profile(ctx.genre)
    mins, secs = divmod(_fin(m.duration_seconds), 60.0)

    intent_text = _INTENT_BRIEF.get(str(ctx.intent), _INTENT_BRIEF["full_mix"])

    bits = [
        f"**Duration:** {int(mins)}:{secs:04.1f}",
        f"**Source:** {'mono' if m.is_mono else 'stereo'}, {int(m.original_sample_rate)} Hz"
        + (f", {int(m.bit_depth)}-bit" if m.bit_depth else ""),
        f"**Analysed at:** {int(m.sample_rate)} Hz",
    ]
    if ctx.health_score is not None:
        grade = f" (grade {ctx.grade})" if ctx.grade else ""
        bits.append(f"**Health score:** {_fin(ctx.health_score):.0f}/100{grade}")
    if ctx.mastering_ready is not None:
        bits.append(f"**Mastering-ready:** {_yn(ctx.mastering_ready)}")

    head = (
        f"# MIX BRIEF — {ctx.filename or 'untitled mixdown'}\n\n"
        f"## What this file is\n\n{intent_text}\n\n"
        f"## What it is being compared against\n\n"
        f"**Reference genre:** {profile.label} (profile `{profile.key}`).\n\n"
        f"Every \"genre target\" and \"usual range\" in the tables below is where {profile.label} "
        f"records **usually** sit. It is a reference, not a specification, and this record does "
        f"not owe it anything. A number outside one of those windows is a *difference* that has "
        f"to be judged against what this file is and what the arrangement is doing — it is not, "
        f"on its own, a fault. The things that are faults regardless are listed separately, under "
        f"Defects.\n\n"
        + "\n".join(f"- {b}" for b in bits)
    )
    if profile.notes:
        head += f"\n\n**How {profile.label} records usually behave:** {profile.notes}"
    if ctx.mastering_blockers:
        blockers = "; ".join(_clip_text(b, 120) for b in ctx.mastering_blockers[:6])
        head += f"\n\n**Mastering blockers (deterministic):** {blockers}"
    return head


def _provenance_section(ctx: EngineerContext) -> str:
    """Which figures came off the audio and which were reconstructed from it.

    Every number in the brief is real, but they are not all the same kind of
    real. A true-peak reading is a measurement. "The vocal sits 7 dB under the
    instruments" — with no stems — is a centre-channel proxy that cannot tell a
    lead vocal from a centred synth, and a report that states the two in the same
    voice is overclaiming on the second one. This table is what lets the write-up
    hedge in proportion instead of hedging everywhere or nowhere.
    """
    has_stems = bool(ctx.stems.available and any(s.present for s in ctx.stems.stems))
    has_sections = bool(ctx.sections.available and ctx.sections.sections)

    measured = (
        "Integrated / short-term / momentary loudness, loudness range, true peak, sample peak, "
        "PLR and PSR; clipped samples, flat-topped runs and inter-sample overs; the third-octave "
        "spectrum, the macro bands and the resonant peaks; correlation, width, mono-sum loss and "
        "L/R balance per band; crest factor, micro- and macro-dynamics; onset density and tempo. "
        "State any of these flatly — they were read off the audio."
    )

    rows: List[Tuple[str, str, str]] = []

    if has_stems:
        rows.append((
            "Vocal level, prominence, presence and sibilance",
            "**measured**",
            "Separation ran. These came off a separated vocal source, so state them as facts "
            "and do not hedge about the centre channel.",
        ))
        rows.append((
            "Kick and bass fundamentals, kick/bass collision",
            "**measured**",
            "Two separated objects compared against each other, not band overlap.",
        ))
    else:
        rows.append((
            "Vocal present, prominence, vocal-to-instrument, intelligibility, centre sibilance",
            "_inferred_",
            "A centre-channel proxy. It cannot tell a lead vocal from a centred synth, a "
            "chopped sample or a mono-summed pad, and a tucked lead is exactly the case it is "
            "least sure about. Say \"the centre-channel estimate puts...\", never \"the vocal "
            "is...\".",
        ))
        rows.append((
            "Kick fundamental, bass fundamental, kick/bass collision, kick definition",
            "_inferred_",
            "Reconstructed from band overlap and subtraction on the two-track, not from two "
            "separated objects. Directionally right, not precise.",
        ))
        rows.append((
            "Which source owns the bursty 5-9 kHz energy",
            "_inferred_",
            "Consonants and closed hi-hats are the same shape of burst in the same band. With "
            "no stems, the attribution in the findings table is a judgement from the voice test "
            "and the intent, not a measurement.",
        ))

    rows.append((
        "Sidechain detected, ducking depth at the kick",
        "_inferred_",
        "Read off the sub-band envelope at kick hits. A pumping bassline or a tight release on "
        "a bus compressor reads the same way. Never state as fact that a sidechain is or is not "
        "on the session.",
    ))
    rows.append((
        "Estimated gain reduction (mix and per stem)",
        "_inferred_",
        "An estimate of what a limiter or compressor did, from crest-factor loss. Not a reading "
        "off anybody's meter. Say \"consistent with about N dB\", not \"you have N dB of GR\".",
    ))
    rows.append((
        "Masking index, band congestion, clarity index",
        "_modelled_",
        "Psychoacoustic models over the measured spectrum, not direct observations. Good for "
        "ranking bands against each other; weak as absolute values.",
    ))
    if has_sections:
        rows.append((
            "Section labels (verse / chorus / drop)",
            "_inferred_",
            "The boundaries are measured from the audio; the names attached to them are a "
            "guess from loudness and density. Trust the timestamps, hedge the labels.",
        ))

    return (
        "## What was measured and what was inferred\n\n"
        "Both kinds of number appear in the tables below and they do not carry the same weight.\n\n"
        f"**Measured off the audio.** {measured}\n\n"
        "**Reconstructed rather than observed.** Hedge these in proportion — not with a blanket "
        "disclaimer, but by saying where the figure came from at the point you lean on it:\n\n"
        + _table(("Figure", "Status", "What you may say about it"), rows)
        + (
            ""
            if has_stems else
            "\n\nIf a recommendation genuinely turns on one of the inferred rows above — you "
            "would say one thing if that is a lead vocal and something else if it is a centred "
            "synth — say so once, in the `alternative` of that prescription, and tell the "
            "producer to re-run the analysis with **stem separation enabled**, which converts "
            "those rows into measurements. Once, where it changes the answer. Not everywhere."
        )
    )


def _loudness_section(ctx: EngineerContext) -> str:
    L = ctx.measurements.loudness
    p = targets.get_profile(ctx.genre)
    rows = [
        ("Integrated", _n(L.integrated_lufs, 1, "LUFS"), _rng(p.integrated_lufs, 1, "LUFS"),
         _vs_usual(L.integrated_lufs, p.integrated_lufs)),
        ("Short-term max", _n(L.short_term_max_lufs, 1, "LUFS"), "—", "—"),
        ("Momentary max", _n(L.momentary_max_lufs, 1, "LUFS"), "—", "—"),
        ("Loudness range", _n(L.loudness_range_lu, 1, "LU"), _rng(p.loudness_range_lu, 1, "LU"),
         _vs_usual(L.loudness_range_lu, p.loudness_range_lu)),
        ("True peak", _n(L.true_peak_dbtp, 2, "dBTP"),
         "at or below -1.0 dBTP — **hard delivery limit, not a genre norm**",
         _cap_verdict(L.true_peak_dbtp, -1.0, " dB")),
        ("Sample peak", _n(L.sample_peak_dbfs, 2, "dBFS"), "—", "—"),
        ("PLR (peak to loudness)", _n(L.plr_db, 1, "dB"), "—", "—"),
        ("PSR 10th percentile", _n(L.psr_p10_db, 1, "dB"), _rng(p.psr_p10_db, 1, "dB"),
         _vs_usual(L.psr_p10_db, p.psr_p10_db)),
        ("PSR median", _n(L.psr_median_db, 1, "dB"), "—", "—"),
    ]
    return "## Loudness & headroom\n\n" + _note_usual() + _table(("Measure", "This mix", _usual_header(ctx), "Difference"), rows)


def _dynamics_section(ctx: EngineerContext) -> str:
    D = ctx.measurements.dynamics
    p = targets.get_profile(ctx.genre)
    rows = [
        ("Crest factor", _n(D.crest_factor_db, 1, "dB"), _rng(p.crest_factor_db, 1, "dB"),
         _vs_usual(D.crest_factor_db, p.crest_factor_db)),
        ("Micro-dynamics", _n(D.micro_dynamics_db, 1, "dB"), _rng(p.micro_dynamics_db, 1, "dB"),
         _vs_usual(D.micro_dynamics_db, p.micro_dynamics_db)),
        ("Macro-dynamics", _n(D.macro_dynamics_lu, 1, "LU"), "—", "—"),
        ("DR value (TT style)", _n(D.dr_value, 1), "—", "—"),
        ("Peak to loudness", _n(D.peak_to_loudness_db, 1, "dB"), "—", "—"),
        ("Programme RMS", _n(D.rms_db, 1, "dBFS"), "—", "—"),
        ("Estimated gain reduction", _n(D.gain_reduction_estimate_db, 1, "dB"), "—", "—"),
        ("Pumping index", _n(D.pumping_index, 2), "0 = none, 1 = severe", "—"),
        ("Pumping rate", _n(D.pumping_rate_hz, 2, "Hz"), "—", "—"),
    ]
    return "## Dynamics & compression\n\n" + _note_usual() + _table(("Measure", "This mix", _usual_header(ctx), "Difference"), rows)


def _clipping_section(ctx: EngineerContext) -> str:
    C = ctx.measurements.clipping
    rows = [
        ("Clipped samples", f"{int(C.clipped_samples)}", f"{_fin(C.clip_percentage):.4f}% of the file"),
        ("Flat-topped runs", f"{int(C.flat_run_count)}", f"longest {int(C.longest_flat_run)} samples"),
        ("Inter-sample overs", f"{int(C.inter_sample_overs)}", "peaks above 0 dBTP between samples"),
        ("Distortion index", _n(C.distortion_index, 3), "0 = clean, 1 = heavily clipped"),
        ("Float over unity", _yn(C.is_float_over_unity), "float file exceeding 1.0"),
        ("Sample peak", _n(C.sample_peak_dbfs, 2, "dBFS"), "—"),
        ("True peak", _n(C.true_peak_dbtp, 2, "dBTP"), "—"),
    ]
    return "## Clipping & distortion\n\n" + _table(("Measure", "Measured", "Note"), rows)


def _stereo_phase_section(ctx: EngineerContext) -> str:
    S = ctx.measurements.stereo
    P = ctx.measurements.phase
    p = targets.get_profile(ctx.genre)

    rows = [
        ("Correlation", _n(S.correlation, 2), f"at or above {p.correlation_min:.2f}",
         _vs_usual_floor(S.correlation, p.correlation_min)),
        ("Width (side/mid)", _n(S.width, 2), _rng(p.stereo_width, 2),
         _vs_usual(S.width, p.stereo_width)),
        ("Mono-sum loss", _n(S.mono_sum_loss_db, 2, "dB"),
         "at or above -1.0 dB — **arithmetic, not a genre norm**",
         _floor_verdict(S.mono_sum_loss_db, -1.0)),
        ("Low-end side energy", _n(S.low_end_side_energy_db, 1, "dB"), "sub should be near-mono", "—"),
        ("L/R balance", _n(S.balance_db, 2, "dB", signed=True), "near 0 (+ = right-heavy)", "—"),
        ("Polarity inverted", _yn(P.polarity_inverted), "—", "—"),
        ("Mono compatible", _yn(P.mono_compatible), "—", "—"),
        ("Mono source", _yn(S.is_mono_source), "—", "—"),
    ]
    out = ["## Stereo & phase\n\n" + _note_usual(), _table(("Measure", "This mix", _usual_header(ctx), "Difference"), rows)]

    band_rows = []
    for band in sorted(set(S.band_correlation) | set(S.band_width) | set(S.band_mono_loss_db)):
        band_rows.append((
            band,
            _n(S.band_correlation.get(band, 1.0), 2),
            _n(S.band_width.get(band, 0.0), 2),
            _n(S.band_mono_loss_db.get(band, 0.0), 2, "dB"),
        ))
    if band_rows:
        out.append("\n\nPer band:\n\n")
        out.append(_table(("Band", "Correlation", "Width", "Mono-sum loss dB"), band_rows))
    if P.worst_band:
        out.append(
            f"\n\nWorst band for phase: **{P.worst_band}** "
            f"(correlation {_fin(P.worst_band_correlation):.2f})."
        )
    return "".join(out)


def _spectral_section(ctx: EngineerContext) -> str:
    S = ctx.measurements.spectral
    p = targets.get_profile(ctx.genre)

    band_rows = []
    for band in S.bands:
        tol = targets.band_tolerance(ctx.genre, band.name)
        band_rows.append((
            band.name,
            f"{_fin(band.low_hz):.0f}-{_fin(band.high_hz):.0f}",
            _n(band.level_db, 1),
            _n(band.target_db, 1),
            _n(band.deviation_db, 1, signed=True),
            f"+/-{tol:.1f}",
            band.verdict,
        ))

    out = [
        "## Frequency balance\n\n"
        "Levels are relative to the mix's own broadband level. The `Usual` column is the "
        f"{p.label} curve — where records of that kind sit on average, not a shape this one "
        "owes anybody. Spectral balance is the single most arrangement-dependent thing in this "
        "brief: a hollow mid-range, a hot sub or a rolled-off top can each be the whole point "
        "of a record. Read the differences against what the arrangement is doing before you "
        "call any of them a problem. The `Flag` column is only the deterministic "
        "inside-or-outside test.\n\n",
        _table(
            ("Band", "Hz", "This mix dB", f"Usual for {p.label} dB", "Difference",
             "Usual spread", "Flag"),
            band_rows,
        ),
    ]

    scalar_rows = [
        ("Mud ratio (150-400 vs 60-120 Hz)", _n(S.mud_ratio_db, 1, "dB"), _rng(p.mud_ratio_db, 1, "dB"),
         _vs_usual(S.mud_ratio_db, p.mud_ratio_db)),
        ("Mud to mid (150-400 vs 1-3 kHz)", _n(S.mud_to_mid_db, 1, "dB"), "—", "—"),
        ("Boxiness (300-600 Hz)", _n(S.boxiness_db, 1, "dB"), "—", "—"),
        ("Harshness index (2-5 kHz)", _n(S.harshness_index, 2), f"at or below {p.harshness_max:.2f}",
         _vs_usual_cap(S.harshness_index, p.harshness_max)),
        ("Sibilance index (5-9 kHz)", _n(S.sibilance_index, 2), f"at or below {p.sibilance_max:.2f}",
         _vs_usual_cap(S.sibilance_index, p.sibilance_max)),
        ("Sharpness", _n(S.sharpness_acum, 2, "acum"), f"at or below {p.sharpness_max_acum:.2f}",
         _vs_usual_cap(S.sharpness_acum, p.sharpness_max_acum)),
        ("Spectral tilt", _n(S.spectral_tilt_db_per_decade, 1, "dB/decade"), "—", "—"),
        ("Spectral centroid", _n(S.spectral_centroid_hz, 0, "Hz"), "—", "—"),
    ]
    out.append("\n\n" + _table(("Measure", "This mix", _usual_header(ctx), "Difference"), scalar_rows))

    if S.resonances:
        res_rows = [
            (_n(r.freq_hz, 0, "Hz"), _n(r.q, 1), _n(r.prominence_db, 1, "dB"), r.severity,
             ", ".join(_span(m.t_start, m.t_end) for m in r.moments[:3]) or "—")
            for r in S.resonances[:8]
        ]
        out.append("\n\nResonant peaks:\n\n" + _table(("Freq", "Q", "Prominence", "Severity", "When"), res_rows))
    return "".join(out)


def _low_end_section(ctx: EngineerContext) -> str:
    L = ctx.measurements.low_end
    p = targets.get_profile(ctx.genre)
    rows = [
        ("Kick detected", _yn(L.kick_detected), "—", "—"),
        ("Kick fundamental", _n(L.kick_fundamental_hz, 1, "Hz"), "—", "—"),
        ("Kick hits", f"{int(L.kick_count)}", "—", "—"),
        ("Bass fundamental", _n(L.bass_fundamental_hz, 1, "Hz"), "—", "—"),
        ("Kick/bass collision", _n(L.kick_bass_collision_db, 1, "dB"),
         f"at or below {p.kick_bass_collision_max_db:.1f} dB",
         _vs_usual_cap(L.kick_bass_collision_db, p.kick_bass_collision_max_db, " dB")),
        ("Sub energy (20-60 Hz)", _n(L.sub_energy_db, 1, "dB"), _rng(p.sub_energy_db, 1, "dB"),
         _vs_usual(L.sub_energy_db, p.sub_energy_db)),
        ("Sidechain detected", _yn(L.has_sidechain), "—", "—"),
        ("Ducking depth at kick", _n(L.ducking_depth_db, 1, "dB"), "—", "—"),
        ("Kick definition", _n(L.kick_definition_db, 1, "dB"), "transient above surrounding low end", "—"),
        ("Low-end mono ratio", _n(L.low_end_mono_ratio, 2), f"at or above {p.low_end_mono_min:.2f}",
         _vs_usual_floor(L.low_end_mono_ratio, p.low_end_mono_min)),
        ("Sub-rumble (below 25 Hz)", _n(L.sub_rumble_db, 1, "dB"), "—", "—"),
    ]
    return "## Low end (kick / 808 relationship)\n\n" + _table(
        ("Measure", "This mix", _usual_header(ctx), "Difference"), rows
    )


def _vocal_section(ctx: EngineerContext) -> str:
    V = ctx.measurements.vocal
    p = targets.get_profile(ctx.genre)
    has_vocal_stem = bool(
        ctx.stems.available and any(s.present and str(s.kind) == "vocals" for s in ctx.stems.stems)
    )
    no_lead_expected = str(ctx.intent) in ("beat", "instrumental", "stem", "reference")
    rows = [
        ("Vocal present", _yn(V.vocal_present),
         ("no lead expected on this file — see \"What this file is\""
          if no_lead_expected else
          f"a lead is {'expected' if p.vocal_expected else 'not expected'} on a "
          f"{p.label} record"),
         "—"),
        ("Voice-test confidence", _n(getattr(V, "vocal_confidence", 0.0), 2),
         "0 = certainly not a voice, 1 = certainly a voice", "—"),
        # The row that stops the AI layer calling a deliberate choice a mistake.
        ("Lead prominence", str(getattr(V, "vocal_prominence", "absent")),
         "tucked = deliberately under the bed (correct for a beat), "
         "balanced = sitting in it, forward = out in front", "—"),
        ("Vocal to instrument", _n(V.vocal_to_instrument_db, 1, "dB", signed=True),
         _rng(p.vocal_to_instrument_db, 1, "dB"),
         _vs_usual(V.vocal_to_instrument_db, p.vocal_to_instrument_db)),
        ("Centre energy ratio", _n(V.center_energy_ratio, 2), "—", "—"),
        ("Intelligibility index", _n(V.intelligibility_index, 2), "0 = buried, 1 = crystal", "—"),
        ("Presence balance", _n(V.presence_balance_db, 1, "dB", signed=True), "—", "—"),
        ("Sibilance", _n(V.sibilance_db, 1, "dB"), "—", "—"),
        ("Level consistency", _n(V.consistency_db, 1, "dB"), "high = uneven, needs riding", "—"),
    ]
    provenance = (
        "Measured on a separated vocal source — state these as facts.\n\n"
        if has_vocal_stem else
        "**Every figure in this table is inferred from the centre channel, not measured on a "
        "vocal.** The test cannot tell a lead vocal from a centred synth, a mono-summed pad or "
        "a chopped sample, and it is least certain exactly when the voice is quiet. Word these "
        "as estimates.\n\n"
    )
    out = [
        "## Vocal\n\n" + provenance,
        _table(("Measure", "This mix", _usual_header(ctx), "Difference"), rows),
    ]

    if no_lead_expected:
        out.append(
            "\n\n**No lead is expected on this file** — see \"What this file is\" at the top. "
            "Whatever the centre channel is showing, there is no vocal balance here to fix. Do "
            "not prescribe raising, de-essing, compressing or riding a lead, and do not read a "
            "low figure in this table as an unfinished mix."
        )
    elif str(getattr(V, "vocal_prominence", "absent")) == "tucked":
        out.append(
            "\n\n**The lead is tucked under the instrument bed.** On a record built for someone "
            "to rap over — and on shoegaze, lo-fi, dream pop and much of indie — that is the "
            "correct decision and not a balance fault. Treat it as a deviation from the "
            f"{p.label} reference with a cost and a benefit, and leave it alone unless something "
            "else in this brief says the lead is meant to be the focus. If your recommendation "
            "would change depending on whether that really is a lead vocal, say so and point "
            "them at stem separation rather than guessing."
        )

    if V.masked_bands:
        out.append(f"\n\nBands masking the centre: {', '.join(V.masked_bands)}.")
    return "".join(out)


def _clarity_section(ctx: EngineerContext) -> str:
    C = ctx.measurements.clarity
    T = ctx.measurements.transients
    p = targets.get_profile(ctx.genre)

    rows = [
        ("Clarity index", _n(C.clarity_index, 2), f"at or above {p.clarity_min:.2f}",
         _vs_usual_floor(C.clarity_index, p.clarity_min)),
        ("Masking index", _n(C.masking_index, 2), f"at or below {p.masking_max:.2f}",
         _vs_usual_cap(C.masking_index, p.masking_max)),
        ("Definition", _n(C.definition_db, 1, "dB"), "—", "—"),
        ("Spectral flatness", _n(C.spectral_flatness, 3), "—", "—"),
        ("Spectral contrast", _n(C.spectral_contrast, 2), "—", "—"),
        ("Punch index", _n(T.punch_index, 2), f"at or above {p.punch_min:.2f}",
         _vs_usual_floor(T.punch_index, p.punch_min)),
        ("Transient to sustain", _n(T.transient_to_sustain_db, 1, "dB"), "—", "—"),
        ("Attack time", _n(T.attack_time_ms, 1, "ms"), "—", "—"),
        ("Smearing index", _n(T.smearing_index, 2), "0 = tight, 1 = smeared", "—"),
        ("Onset density", _n(T.onset_density, 2, "/s"), "—", "—"),
        ("Estimated tempo", _n(T.estimated_tempo, 1, "BPM"), "—", "—"),
    ]
    out = ["## Clarity & transients\n\n", _table(("Measure", "This mix", _usual_header(ctx), "Difference"), rows)]

    if C.band_congestion:
        worst = sorted(C.band_congestion.items(), key=lambda kv: -_fin(kv[1]))[:6]
        out.append(
            "\n\nBand congestion (higher = more crowded): "
            + ", ".join(f"{k} {_fin(v):.2f}" for k, v in worst)
            + "."
        )
    if C.worst_congested_band:
        out.append(f" Worst: **{C.worst_congested_band}**.")
    if T.band_punch:
        out.append(
            "\n\nPer-band punch: "
            + ", ".join(f"{k} {_fin(v):.2f}" for k, v in sorted(T.band_punch.items()))
            + "."
        )
    return "".join(out)


_FINDING_HEADERS = ("finding_id", "dimension", "severity", "confidence", "impact", "band", "detail")


def _finding_rows(findings: Sequence[Finding]) -> List[Tuple[str, ...]]:
    rows: List[Tuple[str, ...]] = []
    for f in findings:
        band = f"{_fin(f.band_hz[0]):.0f}-{_fin(f.band_hz[1]):.0f} Hz" if f.band_hz else "—"
        rows.append((
            f"`{f.id}`",
            f.dimension,
            f.severity,
            f"{_fin(f.confidence):.2f}",
            f"{_fin(f.impact):.1f}",
            band,
            _clip_text(f.detail, 260),
        ))
    return rows


def _evidence_lines(findings: Sequence[Finding], limit: int = 40) -> List[str]:
    """Evidence rows, worded by what the comparison actually is.

    A defect's comparison is a limit — zero clipped samples is not "what trap
    records usually do", it is what a file that fits in its container does. A
    deviation's is a description of where similar records land. Same field on
    `Evidence`, two different claims, and the brief has to say which.
    """
    lines: List[str] = []
    for f in findings:
        word = "limit" if str(getattr(f, "kind", "deviation")) == "defect" else "usual"
        for e in f.evidence[:4]:
            target = ""
            if e.target is not None:
                target = f" ({word} {_fin(e.target):.2f}{(' ' + e.unit) if e.unit else ''})"
            elif e.target_range is not None:
                target = (
                    f" ({word} {_fin(e.target_range[0]):.2f} to "
                    f"{_fin(e.target_range[1]):.2f})"
                )
            detail = f" — {_clip_text(e.detail, 140)}" if e.detail else ""
            lines.append(
                f"- `{f.id}` · {e.label}: {_fin(e.value):.2f}{(' ' + e.unit) if e.unit else ''}"
                f"{target} [{e.verdict}]{detail}"
            )
            if len(lines) >= limit:
                return lines
    return lines


def _findings_section(ctx: EngineerContext) -> str:
    """The findings, in two physically separated groups.

    The split is the whole point and a `kind` column was not enough to carry it.
    A single table with "defect" and "deviation" in the third column reads as one
    list of problems with a taxonomy attached, and the write-up treats it as one
    list of problems. Two headings, two tables, and a paragraph of framing above
    each is what makes the distinction survive into the prose.
    """
    if not ctx.findings:
        return (
            "## Findings\n\n_No findings were raised by the detectors — nothing is broken and "
            "nothing sits far enough from the reference to be worth naming. Do not invent "
            "`finding_id` values. Return an empty `prescriptions` list and say so in the "
            "verdict._"
        )

    label = targets.get_profile(ctx.genre).label
    defects = [f for f in ctx.findings if str(getattr(f, "kind", "deviation")) == "defect"]
    deviations = [f for f in ctx.findings if str(getattr(f, "kind", "deviation")) != "defect"]

    out = [
        "## Findings\n\nThese are the detectors' conclusions, in two groups that mean "
        "different things. `finding_id` values in your prescriptions must be copied exactly "
        "from these tables.\n\n"
    ]

    # -- defects -------------------------------------------------------------
    out.append(
        "### 1. DEFECTS — wrong regardless of genre, intent or taste\n\n"
        "Nobody chose these and no argument about style excuses them: clipping, inter-sample "
        "overs, polarity inversion, mono cancellation, limiter distortion, channel imbalance. "
        "The threshold is arithmetic or physics, not a reference curve. Say plainly that they "
        "are broken, prescribe the fix for every one of them, and never soften them.\n\n"
    )
    if defects:
        out.append(_table(_FINDING_HEADERS, _finding_rows(defects)))
    else:
        out.append(
            "_No defects. Nothing on this file is broken — no clipping, no inter-sample overs, "
            "no polarity inversion, no mono collapse. Say so; it is worth the producer knowing._"
        )

    # -- deviations ----------------------------------------------------------
    out.append(
        f"\n\n### 2. DEVIATIONS FROM THE {label.upper()} REFERENCE — differences, not damage\n\n"
        f"Each of these is a measured difference between this record and where {label} records "
        f"usually sit. **Every one of them may be deliberate, and on this file some of them "
        f"probably are.** Loudness, dynamic range, width, spectral balance, sub weight, vocal "
        f"level, punch and brightness are decisions, not defects.\n\n"
        f"For each one, decide which it is before you write about it:\n\n"
        f"- **Deliberate and right for this record** — say that it reads as a choice, say in "
        f"one clause what it buys, and leave it alone. Put it in `strengths`. Do not write a "
        f"prescription for it. It is actively good for this report to tell a producer that "
        f"something unusual is correct.\n"
        f"- **Costing more than it buys** — then write the prescription, and name both sides of "
        f"the trade in the diagnosis: what changing it gains, and what it gives up.\n\n"
        f"Language rule for this whole group: write \"sits above / below where {label} records "
        f"usually land\", never \"too much\", \"too little\", \"wrong\", \"fault\", \"issue\" or "
        f"\"problem\". The severity column here is only how far from the reference the number "
        f"sits — it is not how much it matters. Rank these by what a listener would notice.\n\n"
    )
    if deviations:
        out.append(_table(_FINDING_HEADERS, _finding_rows(deviations)))
    else:
        out.append(f"_This record sits inside the {label} reference on everything measured._")

    ev_lines = _evidence_lines(ctx.findings)
    if ev_lines:
        out.append("\n\nEvidence behind the findings:\n\n" + "\n".join(ev_lines))
    return "".join(out)


def _dimensions_section(ctx: EngineerContext) -> str:
    if not ctx.dimensions:
        return ""
    rows = [
        (d.label, d.dimension, f"{_fin(d.score):.0f}", d.severity, _clip_text(d.headline, 110),
         ", ".join(f"`{i}`" for i in d.finding_ids) or "—")
        for d in ctx.dimensions
    ]
    return "## Dimension scores\n\n" + _table(
        ("Dimension", "key", "Score", "Severity", "Headline", "Findings"), rows
    )


def _moments_section(ctx: EngineerContext) -> str:
    m = ctx.measurements
    pool: List[Tuple[float, str, Moment]] = []

    def add(source: str, moments: Sequence[Moment]) -> None:
        for mo in moments:
            pool.append((_fin(mo.t_start), source, mo))

    for f in ctx.findings:
        add(f.id, f.moments)
    add("clipping", m.clipping.events)
    add("phase", m.phase.problem_moments)
    add("kick/bass collision", m.low_end.collision_moments)
    add("vocal buried", m.vocal.buried_moments)
    add("vocal loud", m.vocal.loud_moments)
    add("congestion", m.clarity.congested_moments)
    add("weak transient", m.transients.weak_moments)
    for r in m.spectral.resonances:
        add(f"{_fin(r.freq_hz):.0f} Hz resonance", r.moments)

    if not pool:
        return ""

    # A finding and the measurement it was built from carry the same Moment
    # objects, so the same event arrives two or three times. Collapse on
    # (time, label) or the table fills up with one resonance repeated.
    deduped: Dict[Tuple[float, float, str], Tuple[float, str, Moment]] = {}
    for t0, source, mo in pool:
        key = (round(t0, 1), round(_fin(mo.t_end), 1), str(mo.label or "").strip().lower())
        if key not in deduped:
            deduped[key] = (t0, source, mo)
    pool = list(deduped.values())

    # Strongest first so the cap keeps what matters, then back into time order.
    pool.sort(key=lambda item: -_fin(item[2].intensity))
    kept = pool[:MAX_MOMENTS_IN_BRIEF]
    kept.sort(key=lambda item: item[0])

    rows = []
    for _, source, mo in kept:
        value = f"{_fin(mo.value):.2f}" if mo.value is not None else "—"
        rows.append((
            _span(mo.t_start, mo.t_end),
            source,
            f"{_fin(mo.intensity):.2f}",
            value,
            _clip_text(mo.label, 90) or "—",
        ))
    return (
        "## Notable moments\n\nTimestamps you may reference. Do not invent others.\n\n"
        + _table(("Time", "Source", "Intensity", "Value", "Label"), rows)
    )


def _stems_section(ctx: EngineerContext) -> str:
    """Per-source measurements, when separation ran. Otherwise, say it didn't.

    The "didn't" branch matters as much as the other one: without it the model
    reads `vocal_to_instrument_db` as a per-source measurement, and it is a
    centre-channel proxy that cannot tell a lead vocal from a centred synth.
    """
    S = ctx.stems
    if not S.available:
        reason = ""
        if S.warnings:
            reason = " Reason: " + _clip_text(S.warnings[0], 200)
        return (
            "## Per-source analysis\n\n_Source separation was not run on this mix._"
            + reason
            + " Everything above is measured from the 2-track, which means the vocal figures "
            "are centre-channel proxies and the kick/bass figures come from band overlap rather "
            "than from two separated objects. Treat them as estimates, say so when you lean on "
            "them, and do not describe them as per-source measurements."
        )

    present = [s for s in S.stems if s.present]
    order = {kind: i for i, kind in enumerate(STEM_KINDS)}
    present.sort(key=lambda s: order.get(str(s.kind), 99))
    absent = [str(s.kind) for s in S.stems if not s.present]

    out = [
        "## Per-source analysis\n\nThese are **measurements of separated sources**, not "
        f"inferences from the stereo image (`{_clip_text(S.model_name, 40) or 'separator'}`, "
        f"{int(_fin(S.separation_ms))} ms). Where these disagree with anything above, these "
        "win.\n\n",
    ]

    rows = [
        (
            s.kind,
            _n(s.level_lufs, 1, "LUFS"),
            _n(s.level_ratio_db, 1, "dB", signed=True),
            _n(s.peak_dbfs, 1, "dBFS"),
            _n(s.crest_factor_db, 1, "dB"),
            _n(s.micro_dynamics_db, 1, "dB"),
            _n(s.gain_reduction_estimate_db, 1, "dB"),
            _n(s.spectral_centroid_hz, 0, "Hz"),
            _n(s.stereo_width, 2),
            _n(s.correlation, 2),
            _n(s.transient_punch, 2),
            f"{int(s.onset_count)}",
            _n(s.active_ratio, 2),
        )
        for s in present
    ]
    out.append(_table(
        ("Stem", "Level", "vs mix", "Peak", "Crest", "Micro-dyn", "Est. GR", "Centroid",
         "Width", "Corr", "Punch", "Onsets", "Active"),
        rows,
    ))

    band_names = [b for b in MACRO_BAND_ORDER
                  if any(b in (s.band_levels_db or {}) for s in present)]
    if band_names and present:
        band_rows = []
        for band in band_names:
            cells = []
            for s in present:
                level = (s.band_levels_db or {}).get(band)
                share = (s.band_occupancy or {}).get(band)
                if level is None:
                    cells.append("—")
                else:
                    suffix = f" ({_fin(share):.2f})" if share is not None else ""
                    cells.append(f"{_fin(level):.1f}{suffix}")
            band_rows.append((band, *cells))
        out.append(
            "\n\nPer-stem band level in dB, with that stem's share of the band in brackets "
            "(1.00 = it owns the band outright):\n\n"
            + _table(("Band", *[s.kind for s in present]), band_rows)
        )

    derived = []
    if S.vocal_to_instrument_db is not None:
        derived.append(
            f"- **Vocal against the rest of the music:** "
            f"{_n(S.vocal_to_instrument_db, 1, 'dB', signed=True)} — measured between the "
            f"separated vocal and the sum of the other stems, so state it as a fact."
        )
    if S.kick_to_bass_db is not None:
        derived.append(f"- **Kick against bass:** {_n(S.kick_to_bass_db, 1, 'dB', signed=True)}")
    if S.kick_fundamental_hz is not None:
        derived.append(f"- **Kick fundamental (separated):** {_n(S.kick_fundamental_hz, 1, 'Hz')}")
    if S.bass_fundamental_hz is not None:
        derived.append(f"- **Bass fundamental (separated):** {_n(S.bass_fundamental_hz, 1, 'Hz')}")
    if absent:
        derived.append(f"- **No source found for:** {', '.join(absent)}")
    if derived:
        out.append("\n\n" + "\n".join(derived))

    if S.masking_pairs:
        pairs = sorted(S.masking_pairs, key=lambda p: -_fin(p.overlap_db))[:MAX_MASKING_PAIRS_IN_BRIEF]
        pair_rows = [
            (
                f"{p.masker} over {p.maskee}",
                p.band,
                f"{_fin(p.low_hz):.0f}-{_fin(p.high_hz):.0f}",
                _n(p.overlap_db, 1, "dB"),
                p.severity,
                ", ".join(_span(m.t_start, m.t_end) for m in p.moments[:3]) or "—",
                _clip_text(p.detail, 160) or "—",
            )
            for p in pairs
        ]
        out.append(
            "\n\nMeasured masking — how far the second source sits under the first inside that "
            "band:\n\n"
            + _table(("Masker over maskee", "Band", "Hz", "Overlap", "Severity", "When", "Detail"),
                     pair_rows)
        )

    if S.warnings:
        out.append(
            "\n\nSeparation warnings: "
            + "; ".join(_clip_text(w, 180) for w in S.warnings[:4])
            + "."
        )
    return "".join(out)


def _sections_section(ctx: EngineerContext) -> str:
    """The structural breakdown, when segmentation ran."""
    A = ctx.sections
    if not A.available or not A.sections:
        note = "## Section-by-section\n\n_The track was not segmented into sections._"
        if A.notes:
            note += " " + _clip_text(A.notes[0], 200)
        return (
            note
            + " Every figure above is for the whole track, so you cannot say which section a "
            "problem lives in beyond the timestamps in the notable-moments table. Do not invent "
            "structural labels — you do not know where the chorus is."
        )

    sections = list(A.sections)[:MAX_SECTIONS_IN_BRIEF]
    levels = sorted(_fin(s.integrated_lufs, -70.0) for s in sections)
    mid = len(levels) // 2
    median = levels[mid] if len(levels) % 2 else 0.5 * (levels[mid - 1] + levels[mid])

    rows = []
    for s in sections:
        flags = []
        if s.is_loudest:
            flags.append("loudest")
        if s.is_quietest:
            flags.append("quietest")
        rows.append((
            f"{int(s.index)}",
            _clip_text(s.label, 24) or f"section {int(s.index)}",
            f"{_ts(s.t_start)}-{_ts(s.t_end)}",
            f"{max(0.0, _fin(s.t_end) - _fin(s.t_start)):.1f}s",
            _n(s.integrated_lufs, 1, "LUFS"),
            _n(_fin(s.integrated_lufs, -70.0) - median, 1, "LU", signed=True),
            _n(s.crest_factor_db, 1, "dB"),
            _n(s.stereo_width, 2),
            ", ".join(flags) or "—",
        ))

    out = [
        "## Section-by-section\n\nThe track segmented and measured span by span. `vs median` is "
        "that section's loudness against the median section — it is what tells you whether the "
        "chorus actually lifts. Reference these timestamps by name in the diagnosis.\n\n",
        _table(
            ("#", "Label", "Span", "Length", "Loudness", "vs median", "Crest", "Width", ""),
            rows,
        ),
    ]

    band_names = [b for b in MACRO_BAND_ORDER
                  if any(b in (s.band_levels_db or {}) for s in sections)]
    if band_names:
        band_rows = []
        for s in sections:
            cells = [
                (f"{_fin((s.band_levels_db or {}).get(band)):.1f}"
                 if (s.band_levels_db or {}).get(band) is not None else "—")
                for band in band_names
            ]
            band_rows.append((
                f"{int(s.index)} {_clip_text(s.label, 18)}",
                *cells,
            ))
        out.append(
            "\n\nPer-section band levels (dB):\n\n"
            + _table(("Section", *band_names), band_rows)
        )

    summary = [
        f"- **Loudness spread across sections:** {_n(A.loudness_spread_lu, 1, 'LU')}",
        f"- **Peak lift (loudest section vs the median):** {_n(A.peak_lift_db, 1, 'dB')} — "
        f"under about 1.5 dB and the loud sections are not arriving.",
        f"- **Low-end swing across sections:** {_n(A.low_end_swing_db, 1, 'dB')}",
    ]
    out.append("\n\n" + "\n".join(summary))
    if A.notes:
        out.append(
            "\n\nSegmentation notes: "
            + "; ".join(_clip_text(n, 200) for n in A.notes[:4])
            + "."
        )
    return "".join(out)


def _platform_section(ctx: EngineerContext) -> str:
    if not ctx.platform_targets:
        return ""
    rows = [
        (
            p.platform,
            _n(p.target_lufs, 1, "LUFS"),
            _n(p.delta_lufs, 1, "LU", signed=True),
            "yes" if p.will_be_turned_down else "no",
            _n(p.max_true_peak_dbtp, 1, "dBTP"),
            "ok" if p.peak_ok else "over",
            p.verdict,
            _clip_text(p.note, 90),
        )
        for p in ctx.platform_targets
    ]
    return (
        "## Streaming platforms\n\n`delta` is measured integrated loudness minus the platform "
        "target: positive means it gets turned down.\n\n"
        + _table(
            ("Platform", "Target", "Delta", "Turned down", "Peak ceiling", "Peak", "Verdict", "Note"),
            rows,
        )
    )


def _reference_section(ctx: EngineerContext) -> str:
    r = ctx.reference
    if r is None:
        return (
            "## Reference comparison\n\n_No reference track was uploaded. Leave `reference_notes` "
            "as an empty list._"
        )
    rows = [
        ("Integrated loudness", _n(r.integrated_lufs, 1, "LU", signed=True)),
        ("Dynamic range", _n(r.dynamic_range_db, 1, "dB", signed=True)),
        ("Stereo width", _n(r.stereo_width, 2, signed=True)),
        ("True peak", _n(r.true_peak_dbtp, 2, "dB", signed=True)),
        ("Similarity", _n(r.similarity, 0, "/100")),
    ]
    out = [
        "## Reference comparison\n\nAll deltas are **this mix minus the reference**: positive "
        "means this mix has more.\n\n",
        _table(("Measure", "Delta"), rows),
    ]
    if r.band_deltas:
        out.append(
            "\n\nPer-band delta (dB): "
            + ", ".join(f"{k} {_fin(v):+.1f}" for k, v in r.band_deltas.items())
            + "."
        )
    if r.biggest_gaps:
        out.append("\n\nBiggest gaps: " + "; ".join(_clip_text(g, 120) for g in r.biggest_gaps[:6]) + ".")
    return "".join(out)


def _notes_section(ctx: EngineerContext) -> str:
    if not (ctx.user_notes and ctx.user_notes.strip()):
        return ""
    return (
        "## What the producer told you\n\n"
        + _clip_text(ctx.user_notes, 1500)
        + "\n\nTake this seriously. If the measurements contradict it, say so plainly."
    )


# Fallback grouping for a plugin that arrives without a category. Keyed by the
# first capability it declares, so "Soothe 2" with `resonance_suppressor` files
# under Spectral repair rather than under Other.
_CAPABILITY_FAMILY: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("eq_",), "EQ"),
    (("comp_", "expander"), "Dynamics"),
    (("limiter", "clipper"), "Loudness"),
    (("deesser", "resonance_suppressor", "spectral_repair", "noise_reduction"), "Spectral repair"),
    (("saturation", "tape", "exciter"), "Colour"),
    (("imager", "mono_maker", "reverb", "delay", "transient_shaper"), "Space & shaping"),
    (("meter_", "reference_matching"), "Metering"),
    (("pitch_correction",), "Pitch"),
)


def _plugin_category(plugin: OwnedPlugin) -> str:
    category = " ".join(str(plugin.category or "").split())
    if category:
        return category[:40]
    for slug in plugin.capabilities:
        for prefixes, label in _CAPABILITY_FAMILY:
            if any(slug.startswith(p) for p in prefixes):
                return label
    return "Other"


def _capabilities_section(ctx: EngineerContext) -> str:
    """What is POSSIBLE for this producer, in capability terms rather than brands."""
    owned = ctx.owned
    missing = ctx.missing

    out = [
        "## What the producer can actually do\n\n"
        "This is the vocabulary every move has to fit inside. It already includes the stock "
        "devices any DAW ships, so it is never empty and there is always an executable plan. "
        "Decide the *shape* of each fix from this list, then name a plugin from the list below "
        "it.\n\n",
        caps.describe(owned),
    ]
    if missing:
        out.append(
            "\n\n**Not available to them:** "
            + ", ".join(f"{caps.CAPABILITIES[c][0]} (`{c}`)" for c in missing)
            + ". Do not prescribe any of these, and do not write a move whose settings only "
            "make sense on one of them."
        )
    return "".join(out)


def _plugins_section(ctx: EngineerContext) -> str:
    """Names only — the capability list above is what decides the move."""
    if not ctx.plugins:
        return (
            "## Plugins they own\n\n_None on file._ Assume stock DAW devices only: name the "
            "stock device that does the job (Ableton EQ Eight, Logic Channel EQ, FL Parametric "
            "EQ 2, Pro Tools EQ3 — whichever fits the move) and keep every setting inside what "
            "a stock device actually has."
        )

    grouped: Dict[str, List[str]] = {}
    for plugin in ctx.plugins[:MAX_PLUGINS_IN_BRIEF]:
        label = _clip_text(plugin.name, 60)
        if plugin.manufacturer and _plugin_key(plugin.manufacturer) not in _plugin_key(label):
            label = _clip_text(f"{plugin.manufacturer} {plugin.name}", 60)
        grouped.setdefault(_plugin_category(plugin), []).append(label)

    lines = [f"- **{category}:** " + ", ".join(names) for category, names in sorted(grouped.items())]
    return (
        "## Plugins they own\n\n"
        "Use these names in the `plugin` field. Owning the name is not owning the capability — "
        "the list above is the one that decides what the move can be.\n\n" + "\n".join(lines)
    )


def _gaps_section(ctx: EngineerContext) -> str:
    """Per finding dimension: what would help, and what they have instead."""
    if not ctx.findings:
        return ""

    owned = set(ctx.owned)
    seen: List[str] = []
    for finding in ctx.findings:
        dimension = str(finding.dimension)
        if dimension not in seen:
            seen.append(dimension)

    rows = []
    for dimension in seen:
        wanted = caps.CAPABILITY_FOR_DIMENSION.get(dimension, ())
        if not wanted:
            continue
        have = [c for c in wanted if c in owned]
        gap = caps.missing_for(dimension, owned)
        rows.append((
            dimension,
            ", ".join(caps.CAPABILITIES[c][0] for c in have) or "**nothing purpose-built**",
            ", ".join(caps.CAPABILITIES[c][0] for c in gap) or "—",
        ))
    if not rows:
        return ""

    return (
        "## Capability gaps on the problems this mix actually has\n\n"
        "One row per dimension a finding was raised in. The middle column is what they can reach "
        "for; the right column is what would have helped and is not there. A gap is a reason to "
        "change the shape of the fix, not a reason to prescribe the missing tool — and at most "
        "two prescriptions in the whole report may mention a gap at all.\n\n"
        + _table(("Dimension", "What they have for it", "What is missing"), rows)
    )


def _closing_section(ctx: EngineerContext) -> str:
    """The last thing in the brief, which is the last thing read before writing.

    Repeats the two rules the whole rewrite exists to enforce, in the position
    that gets the most weight, and names the intent one final time so it is the
    freshest fact in context when the first sentence of the verdict is written.
    """
    label = targets.get_profile(ctx.genre).label
    intent_line = {
        "beat": (
            "**This is a beat.** A tucked or absent lead is correct, an open mid-range is the "
            "pocket for the topline, and bursty 5-9 kHz energy is hi-hats. Do not prescribe a "
            "vocal fix, a de-esser for the hats, or a master to genre loudness."
        ),
        "instrumental": "**This is an instrumental.** Say nothing about vocals.",
        "stem": "**This is a single stem.** Whole-mix balance judgements do not apply to it.",
        "reference": (
            "**This is somebody else's finished record.** Describe what it does. Prescribe "
            "nothing."
        ),
        "demo": "**This is a rough.** Loudness and limiting are premature; arrangement is not.",
    }.get(str(ctx.intent), "**This is a finished mix with a lead.** Judge it as a whole record.")

    return (
        "## Your job\n\nWrite the report.\n\n"
        f"- {intent_line}\n"
        "- **Fix every defect and say plainly that it is broken.** No hedging, no genre excuse.\n"
        f"- **Treat every deviation as a difference from the {label} reference, not as damage.** "
        "Decide for each one whether it reads as deliberate. Say so and leave the deliberate "
        "ones alone — that judgement is the most valuable thing in this report. Prescribe a "
        "change only where you can argue the cost beats what it buys, and name both sides.\n"
        "- **Rank by what a listener would notice**, not by how far a number sits outside a "
        "window.\n"
        "- **Say where each figure came from.** Measured is stated; inferred is hedged.\n"
        "- Every number from this brief and nowhere else. Every move executable tonight. Every "
        "`finding_id` copied from the findings tables. Every move shaped for the capabilities "
        "listed above and named with a plugin they own."
    )


def build_user_brief(ctx: EngineerContext) -> str:
    """Serialise the evidence as a readable brief. No raw time-series arrays."""
    sections = [
        _header(ctx),
        _provenance_section(ctx),
        _findings_section(ctx),
        _dimensions_section(ctx),
        _clipping_section(ctx),
        _stereo_phase_section(ctx),
        _loudness_section(ctx),
        _dynamics_section(ctx),
        _spectral_section(ctx),
        _low_end_section(ctx),
        _vocal_section(ctx),
        _clarity_section(ctx),
        _stems_section(ctx),
        _sections_section(ctx),
        _moments_section(ctx),
        _platform_section(ctx),
        _reference_section(ctx),
        _notes_section(ctx),
        _capabilities_section(ctx),
        _plugins_section(ctx),
        _gaps_section(ctx),
        _closing_section(ctx),
    ]
    return "\n\n---\n\n".join(s for s in sections if s)


# --------------------------------------------------------------------------
# Request construction
# --------------------------------------------------------------------------


def build_request(ctx: EngineerContext, max_tokens: int = MAX_TOKENS) -> Dict[str, Any]:
    """The exact kwargs handed to the SDK. Exposed so it can be inspected in tests."""
    return {
        "model": MODEL,
        "max_tokens": max_tokens,
        # Cached prefix: identical on every request, so repeat analyses read it
        # back at ~10% of input cost. Everything volatile lives in the user turn
        # *after* this block, or the cache never hits.
        "system": [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": build_user_brief(ctx)}],
        # Adaptive thinking is on by default on Opus 5; `effort` is the depth
        # control. temperature/top_p/top_k and `budget_tokens` are 400s here.
        "output_config": {"effort": EFFORT},
    }


def draft_schema() -> Dict[str, Any]:
    """The JSON schema the model is constrained to, as the API will see it."""
    from anthropic.resources.messages.messages import transform_schema
    from pydantic import TypeAdapter

    return transform_schema(TypeAdapter(_ReportDraft).json_schema())


# --------------------------------------------------------------------------
# Draft -> contract
# --------------------------------------------------------------------------


def _tool_label(move: _MoveDraft) -> str:
    plugin = " ".join(str(move.plugin or "").split())
    stock = " ".join(str(move.stock_alternative or "").split())
    if plugin and stock and stock.lower() not in plugin.lower():
        return f"{plugin} (stock: {stock})"
    return plugin or stock


def _to_move(draft: _MoveDraft, order: int) -> Move:
    settings: Dict[str, str] = {}
    for s in draft.settings:
        name = " ".join(str(s.name or "").split())
        if name:
            settings[name] = " ".join(str(s.value or "").split())
    return Move(
        order=order,
        target=_clip_text(draft.target, 120) or "master",
        action=_clip_text(draft.action, 400),
        tool=_clip_text(_tool_label(draft), 160),
        settings=settings,
        why=_clip_text(draft.why, 400),
    )


def _to_prescription(draft: _PrescriptionDraft, dimension: Dimension) -> Prescription:
    moves = [
        _to_move(m, i)
        for i, m in enumerate(draft.moves[:MAX_MOVES_PER_PRESCRIPTION], start=1)
        if str(m.action or "").strip()
    ]
    difficulty = str(draft.difficulty or "").strip().lower()
    if difficulty not in _DIFFICULTIES:
        difficulty = "moderate"
    return Prescription(
        finding_id=draft.finding_id.strip(),
        dimension=dimension,
        headline=_clip_text(draft.headline, 120),
        diagnosis=_clip_text(draft.diagnosis, 1200),
        root_cause=_clip_text(draft.root_cause, 600),
        moves=moves,
        alternative=_clip_text(draft.alternative, 600),
        do_not=_clip_text(draft.do_not, 600),
        done_when=_clip_text(draft.done_when, 600),
        expected_gain=round(max(0.0, min(100.0, _fin(draft.expected_gain))), 1),
        minutes=int(max(1, min(600, _fin(draft.minutes, 5)))),
        difficulty=difficulty,  # type: ignore[arg-type]
    )


def _clean_list(items: Sequence[str], limit: int, item_chars: int) -> List[str]:
    out: List[str] = []
    for item in items:
        text = _clip_text(item, item_chars)
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _log_tool_fit(prescriptions: Sequence[Prescription], ctx: EngineerContext) -> None:
    """Count how many moves stayed inside the producer's own rack.

    Whether the advice is genuinely shaped by what they own is not something a
    server can assert on — it is a judgement about prose. What it can do is
    count, and a run where most moves name something outside the list is the
    signal that either the prompt has drifted or the plugin list never made it
    through the API. Observability only; nothing is dropped on this basis,
    because naming a stock device is legitimate and stock devices are not in
    the owned list.
    """
    if not ctx.plugins:
        return
    keys = [k for k in (_plugin_key(p.name) for p in ctx.plugins) if k]
    inside = 0
    outside: List[str] = []
    for prescription in prescriptions:
        for move in prescription.moves:
            tool = _plugin_key(move.tool)
            if any(key in tool for key in keys):
                inside += 1
            else:
                outside.append(move.tool or "(unnamed)")
    logger.info(
        "engineer: %d/%d move(s) named a plugin the producer owns%s",
        inside,
        inside + len(outside),
        f"; outside the list: {', '.join(outside[:6])}" if outside else "",
    )


def _to_report(draft: _ReportDraft, ctx: EngineerContext) -> EngineerReport:
    """Convert, sanitise, and drop anything that references a finding we never raised.

    Dropping is deliberate. A prescription pinned to an id the UI has never
    heard of renders as an orphan card, and an id the model invented is by
    definition not backed by a measurement.
    """
    known: Dict[str, Finding] = {f.id: f for f in ctx.findings}
    prescriptions: List[Prescription] = []
    seen: set[str] = set()

    for item in draft.prescriptions:
        fid = str(item.finding_id or "").strip()
        finding = known.get(fid)
        if finding is None:
            logger.warning(
                "engineer: dropping prescription for unknown finding_id %r (headline=%r)",
                fid,
                _clip_text(item.headline, 80),
            )
            continue
        if fid in seen:
            logger.info("engineer: dropping duplicate prescription for %r", fid)
            continue
        seen.add(fid)

        # The finding's own dimension wins: it came from a measurement.
        dimension = finding.dimension
        if item.dimension != dimension and item.dimension in _DIMENSIONS:
            logger.debug(
                "engineer: prescription %r said dimension %r, finding says %r; using the finding",
                fid,
                item.dimension,
                dimension,
            )
        prescriptions.append(_to_prescription(item, dimension))
        if len(prescriptions) >= MAX_PRESCRIPTIONS:
            break

    reference_notes = (
        _clean_list(draft.reference_notes, MAX_LIST_ITEMS, 400) if ctx.reference is not None else []
    )
    _log_tool_fit(prescriptions, ctx)

    return EngineerReport(
        verdict=_clip_text(draft.verdict, 900),
        the_one_thing=_clip_text(draft.the_one_thing, 500),
        strengths=_clean_list(draft.strengths, MAX_LIST_ITEMS, 240),
        prescriptions=prescriptions,
        session_plan=_clean_list(draft.session_plan, MAX_LIST_ITEMS, 300),
        mastering_verdict=_clip_text(draft.mastering_verdict, 700),
        reference_notes=reference_notes,
    )


# --------------------------------------------------------------------------
# The call
# --------------------------------------------------------------------------


def _client(api_key: Optional[str] = None):
    """Build a client, or None if there is no usable credential.

    Imported lazily so that importing this module — which `main.py` does at
    startup — never depends on the SDK being installed or a key being present.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key or not key.strip():
        logger.warning("engineer: ANTHROPIC_API_KEY is not set; skipping the AI layer")
        return None
    try:
        from anthropic import Anthropic
    except ImportError:
        logger.warning("engineer: the `anthropic` package is not installed; skipping the AI layer")
        return None
    return Anthropic(api_key=key.strip())


def _parse_draft(client, request: Dict[str, Any]) -> Optional[_ReportDraft]:
    """One API round trip. Returns the validated draft, or None with a logged reason."""
    max_tokens = int(request["max_tokens"])

    if max_tokens > STREAM_ABOVE_TOKENS:
        # Large ceilings need the stream so a long generation can't trip an
        # idle-connection timeout. Same schema, applied by hand.
        streamed = dict(request)
        streamed["output_config"] = {
            **request["output_config"],
            "format": {"type": "json_schema", "schema": draft_schema()},
        }
        with client.messages.stream(**streamed) as stream:
            message = stream.get_final_message()

        # stop_reason is checked before anything reads content.
        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            logger.warning(
                "engineer: the model declined the request (category=%s)",
                getattr(details, "category", "unknown"),
            )
            return None
        if message.stop_reason == "max_tokens":
            logger.warning(
                "engineer: response hit max_tokens (%d) — thinking plus output exceeded the "
                "budget, so the JSON is truncated. Raise max_tokens.",
                max_tokens,
            )
            return None

        text = next((b.text for b in message.content if b.type == "text"), "")
        if not text:
            logger.warning("engineer: empty response (stop_reason=%s)", message.stop_reason)
            return None
        logger.info(
            "engineer: %d output tokens, %d cached / %d fresh input tokens",
            message.usage.output_tokens,
            message.usage.cache_read_input_tokens or 0,
            message.usage.input_tokens,
        )
        return _ReportDraft.model_validate_json(text)

    # `parse()` validates inside the SDK, so a truncated body surfaces as a
    # ValidationError rather than a readable stop_reason — say so in the log.
    try:
        message = client.messages.parse(output_format=_ReportDraft, **request)
    except ValidationError:
        logger.warning(
            "engineer: could not parse the response at max_tokens=%d; the most likely cause is "
            "truncation (thinking plus output exceeded the budget). Raise max_tokens above %d "
            "to take the streaming path.",
            max_tokens,
            STREAM_ABOVE_TOKENS,
        )
        return None

    if message.stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        logger.warning(
            "engineer: the model declined the request (category=%s)",
            getattr(details, "category", "unknown"),
        )
        return None
    draft = message.parsed_output
    if draft is None:
        logger.warning("engineer: no parsed output in the response (stop_reason=%s)", message.stop_reason)
    return draft


def consult_engineer(
    analysis_ctx: EngineerContext,
    *,
    api_key: Optional[str] = None,
    max_tokens: int = MAX_TOKENS,
) -> Optional[EngineerReport]:
    """Ask the engineer for a report.

    Returns `None` — never raises — when the API is unavailable, declines, or
    returns something that will not validate. The caller renders the
    deterministic findings in that case, which is a worse report but never a
    wrong one. Every `None` path logs why.
    """
    import anthropic

    client = _client(api_key)
    if client is None:
        return None

    request = build_request(analysis_ctx, max_tokens=max_tokens)

    try:
        draft = _parse_draft(client, request)
    except anthropic.RateLimitError as exc:
        retry_after = getattr(getattr(exc, "response", None), "headers", {}) or {}
        logger.warning(
            "engineer: rate limited by the API (retry-after=%s); falling back to findings",
            retry_after.get("retry-after", "unknown"),
        )
        return None
    except anthropic.APIStatusError as exc:
        logger.warning(
            "engineer: API returned %s (%s): %s",
            getattr(exc, "status_code", "?"),
            getattr(exc, "type", "?"),
            getattr(exc, "message", exc),
        )
        return None
    except anthropic.APIConnectionError as exc:
        logger.warning("engineer: could not reach the API: %s", exc)
        return None
    except ValidationError as exc:
        logger.warning("engineer: model output did not match the schema: %s", exc)
        return None
    except Exception:  # never take the whole analysis down with us
        logger.exception("engineer: unexpected failure in the AI layer")
        return None

    if draft is None:
        return None

    try:
        report = _to_report(draft, analysis_ctx)
    except ValidationError:
        logger.exception("engineer: could not build an EngineerReport from the draft")
        return None

    if not report.verdict.strip():
        logger.warning("engineer: model returned an empty verdict; discarding the report")
        return None

    logger.info(
        "engineer: report ready — %d prescription(s), %d move(s)",
        len(report.prescriptions),
        sum(len(p.moves) for p in report.prescriptions),
    )
    return report


__all__ = [
    "EngineerContext",
    "EngineerReport",
    "consult_engineer",
    "build_user_brief",
    "build_request",
    "draft_schema",
    "coerce_plugin",
    "coerce_plugins",
    "enrich_plugins",
    "MAX_PLUGINS_IN_BRIEF",
    "SYSTEM_PROMPT",
    "MODEL",
]
