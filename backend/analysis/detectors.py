"""Measurements -> evidence-backed Findings, and the per-dimension roll-up.

This is the layer that has opinions. Everything under it (`dsp/`) measures and
refuses to judge; everything over it (the AI layer) writes prose and is not
allowed to invent a number. What happens here is the join: a measured value is
compared against the window `targets.py` says this genre lives in, and if it
falls outside, a `Finding` is emitted carrying the figure, the window, and a
sentence that stands on its own without any model in the loop.

Five rules hold the whole file together.

1. **No finding without a number.** Every detector calls `targets.range_miss`
   (or an equivalent explicit comparison) against a window before it emits
   anything. There is no path that produces a Finding from a hunch.

2. **Severity is not chosen, it is computed.** Each metric declares a `scale` —
   "one tolerance unit in this metric's own units" — and severity comes from
   `miss / scale` through one shared mapping (`_severity`). A detector cannot
   decide it is important; it can only say how far out of range its number is.

3. **Confidence tracks how the number was obtained.** True peak, LUFS and
   inter-channel correlation are direct: 0.9+. Gain-reduction estimates,
   sidechain detection and anything derived from center extraction are
   inferences from a mixed-down file with no stems, and are scored accordingly
   (0.4-0.6). Short files lower confidence further, because most of these
   measurements are statistics over time.

4. **Genre decides the window, not the physics.** -9 LUFS with 4 LU of range is
   a correct trap master and a ruined folk record; 0.75 of width is the point of
   an ambient mix and wrong in hip-hop; lo-fi's missing top end is the genre.
   Every stylistic threshold is read from `targets.get_profile()`. The handful
   of thresholds that are *arithmetic* rather than taste — a mono fold-down
   losing 8 dB, a limiter emitting flat tops — are genre-independent on purpose
   and are marked as such where they appear.

5. **Say it once.** Mud and a hot low-mid band are the same energy described
   twice, so the frequency-balance detector stands down where a more specific
   detector has already spoken. The list is ranked worst-first and capped, and
   every dimension keeps at least its own worst finding.

6. **Where the number cannot tell a decision from a mistake, ask — but first
   look at where it happened.** An empty intro, a lead under the beat and a
   mono-leaning image all measure exactly like the accident that produces the
   same reading, and the question that resolves them is written in `clarify.py`
   rather than here. What belongs here is the context that makes the question
   unnecessary: an intro is *expected* to shed its bottom end, so the detector
   that noticed reads the section label before deciding whether there is
   anything to ask about at all. See `_section_role`.

Optional depth follows the same five rules, and rules 3 and 5 are what shape
it. `Measurements.stems` and `Measurements.sections` do not add a parallel set
of findings; they make the existing ones better:

* **Stems replace inferences with measurements, and the confidence says so.**
  Vocal balance goes from a center estimate (0.55 — a center estimate cannot
  tell a singer from a centered synth) to the vocal stem's own loudness against
  the actual instrumental (0.90). Kick versus bass goes from a spectrum
  reconstructed by subtraction (0.70) to two separated objects compared
  directly (0.93). The two-track path stays exactly as it was and is used
  whenever separation was not run or did not find the source.
* **A masking pair is claimed exactly once.** `ctx.take_masking` hands each
  measured masker/maskee pair to the one detector that describes the problem
  best — vocal balance if it is burying the vocal, mud if it is in the low
  mids, clarity for whatever is left — so "the guitars are on top of the
  vocal" is one finding, not three.
* **Sections beat whole-file statistics where they overlap.** A chorus that
  fails to lift and a low end that collapses in one part of the arrangement are
  both invisible to any number averaged over the whole file, and the first of
  them is the same observation as a low loudness range with the timestamps
  attached — so the arrangement detector speaks and the loudness-range arm of
  `dynamic_range` stands down.
* **Genre still decides.** Nothing here is a new global threshold. Whether a
  genre expects a chorus to lift is derived from `vocal_expected`, `punch_min`
  and the loudness-range window it already has (`_expects_arrangement_lift`);
  how far its low end may swing between sections comes from
  `low_end_mono_min`; a drum stem's crest floor is the genre's own crest
  window. Ambient and classical are never told their chorus should be louder.

Two derivations are worth calling out because they are what let this file stay
genre-aware without inventing a second copy of `targets.py`:

* **Region targets are read off the genre's own curve.** `mud_to_mid_db` and
  `boxiness_db` have no window in `targets.py`, so instead of hard-coding one,
  the same ratio is computed *from `targets.target_curve(genre)`* and used as
  the target. Because the DSP layer reports those two as power **densities**
  while the curve is in the 1/3-octave summed convention, the conversion
  applies the -10log10(f) density correction (`_curve_density_db`). Sanity
  check: reconstructing `mud_ratio_db` this way for pop gives -8.7 dB, which
  lands inside pop's hand-written (-14.0, -5.5) window — the two agree.

* **Spectral tilt is compared against the tilt of the genre's target curve**,
  fitted over exactly the 1/3-octave bands the DSP layer used (recoverable from
  `third_octave_db`). Lo-fi's steep roll-off is therefore expected rather than
  penalised, with no lo-fi special case anywhere in this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import clarify, targets
from .core import THIRD_OCTAVE_CENTERS
from .types import (
    DIMENSION_LABELS,
    DIMENSIONS,
    DimensionScore,
    Evidence,
    Finding,
    FindingKind,
    MaskingPair,
    Measurements,
    Moment,
    Section,
    SectionAnalysis,
    Severity,
    StemAnalysis,
    StemMeasurement,
    TRACK_INTENTS,
    TrackIntent,
    Verdict,
)

__all__ = ["detect_all", "score_dimensions", "finding_kind"]


# ---------------------------------------------------------------------------
# Severity, impact and confidence scales
# ---------------------------------------------------------------------------

# `miss / scale` -> severity. One shared mapping for every detector, so
# "how bad is this" is never a per-detector opinion. A ratio of 1.0 means the
# value sits one full tolerance unit outside the genre's window.
_MAJOR_AT = 1.5
_CRITICAL_AT = 3.0


# ---------------------------------------------------------------------------
# Defects and deviations
#
# The distinction the whole report now turns on, and the reason a beat stopped
# being told its hi-hats are a vocal problem.
#
# A **defect** is wrong independent of genre, intent, artist and decade. Nobody
# chose a squared-off waveform, an inverted channel or a mix that vanishes in
# mono, and no reference curve is involved in noticing it — the threshold is
# arithmetic or physics. Defects keep the full severity range and the full pull
# on the health score.
#
# Everything else is a **deviation**: a measured difference from a reference,
# where the reference is the genre profile in `targets.py` or a curve derived
# from it. Loudness, dynamic range, width, spectral balance, sub weight, vocal
# level, punch — every one of these is a decision somebody could reasonably make
# differently, and on some records the departure *is* the record. A deviation is
# reported as a difference with a cost and a benefit, never as damage.
# ---------------------------------------------------------------------------

_DEFECT_IDS: frozenset = frozenset({
    "phase.polarity_inverted",
    "phase.mono_incompatible",
    "phase.band_cancellation",
    "limiter.over_driven",
    "low_end.not_mono",
    "stereo_width.channel_imbalance",
})

# `clipping.*` in full: hard clipping and inter-sample overs are both the same
# kind of fact — the waveform, or the decoder's reconstruction of it, does not
# fit in the container. No genre wants that.
_DEFECT_PREFIXES: Tuple[str, ...] = ("clipping.",)


def finding_kind(finding_id: str) -> FindingKind:
    """Defect or deviation, from the finding id alone.

    Kept as a pure function of the id so the classification is auditable in one
    place and a new detector cannot quietly invent a third category.
    """
    fid = str(finding_id or "")
    if fid in _DEFECT_IDS or fid.startswith(_DEFECT_PREFIXES):
        return "defect"
    return "deviation"


# Where a deviation is allowed to reach "major": the distance at which a *defect*
# would be critical. The rule is "one severity step gentler at the same distance"
# — a deviation 3 tolerance units out is a major, where a defect that far out is
# a critical. "critical" belongs to defects and a deviation never reaches it.
#
# This was 2x `_CRITICAL_AT` (6.0) and that number is unreachable: across every
# fixture and every genre the largest deviation measured is ~4.2 tolerance units,
# so *every* deviation bucketed to "minor". Combined with a penalty table keyed
# only on the label, that made a trap master judged against ambient score
# identically to one judged against rock — the genre signal was not softened, it
# was discarded. See `deviation_penalty` for the other half of the fix.
_DEVIATION_MAJOR_AT = _CRITICAL_AT


def _severity_for(kind: str, ratio: float) -> Severity:
    """Severity under the defect/deviation rule.

    Defects use the shared mapping unchanged. Deviations cap at major and only
    reach it a long way outside the window — a departure from a reference is
    never a five-alarm, however far it goes.
    """
    if kind == "defect":
        return _severity(ratio)
    r = _fin(ratio, 0.0)
    if r >= _DEVIATION_MAJOR_AT:
        return "major"
    if r > 0.0:
        return "minor"
    return "clean"


# ---------------------------------------------------------------------------
# Track intent
#
# The second half of the same idea. The genre says what a finished record of
# this kind usually measures; the intent says whether this file is trying to be
# one. A beat is an instrumental built for somebody else to rap over: the lead
# is deliberately absent or tucked, the mid-range is deliberately left open, and
# the 5-9 kHz burstiness that a full mix would owe to consonants is hi-hats.
# Judging it against a finished-song checklist produces exactly the complaint
# this module exists to stop producing.
# ---------------------------------------------------------------------------

# No lead vocal is expected on the record, so nothing about a lead is reported.
_NO_LEAD_INTENTS: Tuple[str, ...] = ("beat", "instrumental", "stem", "reference")

# How much wider the 5-9 kHz window is when the burstiness in it is percussion
# rather than consonants.
#
# `sibilance_max` is written for a record with a singer on it, where a bursty
# 5-9 kHz band means one thing and the ear is unforgiving about it. With no lead
# on the file the same measurement is hi-hats, shakers and rim clicks, and every
# genre that has a hat pattern expects those to cut — trap's ceiling of 0.36
# exists to protect a vocal that a trap *beat* does not have yet. 1.6x puts
# trap's percussion ceiling at 0.58 and pop's at 0.64, which on the fixtures is
# the difference between "your hats are bright" firing on a normal beat and
# firing only on one that will genuinely fatigue.
_PERCUSSION_SIB_FACTOR = 1.6

# One element in isolation. Almost every whole-mix balance judgment is a
# category error against it: a bass stem is *supposed* to be all low end, a
# vocal stem is supposed to have nothing under 100 Hz, and neither has a
# "mix" to be muddy or congested.
_STEM_SUPPRESSED_DIMENSIONS: Tuple[str, ...] = (
    "mud", "clarity", "vocal_balance", "frequency_balance",
)
# The two arrangement findings are suppressed for the same reason. A record's
# lift and its low end are properties of the *whole* arrangement: a bass stem that
# sits out the intro, or a pad that plays at one level under a chorus that
# lifts around it, is doing its job. "The chorus does not lift" and "the low
# end steps back in the intro" are statements about a mix, and a single element
# is not one.
_STEM_SUPPRESSED_IDS: Tuple[str, ...] = (
    "low_end.kick_bass_collision",
    "dynamic_range.no_section_lift",
    "low_end.section_collapse",
)

# A rough. Where it lands on the loudness ladder and how its limiter behaves are
# questions about a master that has not been attempted yet. Structure, balance
# and anything actually broken still matter — a demo with an inverted channel is
# still an inverted channel.
_DEMO_SUPPRESSED_IDS: Tuple[str, ...] = (
    "loudness.cannot_reach_level",
    "dynamic_range.unmastered",
)
_DEMO_SUPPRESSED_DIMENSIONS: Tuple[str, ...] = ("limiter",)

# A beat is unfinished on purpose: the headroom it is carrying is the room the
# topline needs. Telling a producer their beat "has not been mastered yet" is
# describing the brief back to them.
_BEAT_SUPPRESSED_IDS: Tuple[str, ...] = ("dynamic_range.unmastered",)

# The pocket a topline drops into. On a beat these bands reading light against
# the genre curve is the arrangement leaving room, not a hole in the mix, so the
# "thin" side of the frequency-balance test stands down there. The "hot" side
# still fires: mids that are *full* are the thing that leaves a rapper nowhere
# to sit.
_TOPLINE_BANDS: Tuple[str, ...] = ("mid", "upper_mid")

# How sure the center-channel voice test has to be, and how far up in the bed
# the lead has to sit, before the word "sibilance" is used for bursty 5-9 kHz
# energy. Below either bar the energy is still reported — it is measured, and it
# is fatiguing whatever makes it — but it is attributed to percussion, which is
# what it usually is when the voice test is unsure or the voice is tucked.
_LEAD_UP_FRONT_CONFIDENCE = 0.65

# Share of the 5-10 kHz band a separated source has to own before it is called
# the owner. Four stems splitting a band evenly sit at 0.25 each, so 0.40 is a
# clear plurality without demanding a majority — a real lead shares the top with
# the cymbals on every record ever made, and requiring it to beat them outright
# would attribute every sibilant vocal to the drums.
_TOP_OWNER_SHARE = 0.40

# `band_occupancy` is keyed by macro band, and the macro band covering 5-9 kHz
# is `brilliance` at 5-10 kHz. The extra octave-tenth on top is cymbal wash
# rather than consonant, so this over-credits percussion very slightly — which
# is the safe direction: it can only ever make us less willing to say
# "sibilance", never more.
_TOP_BAND = "brilliance"

# Above this the voice test is sure enough that a lead sitting under the bed is
# worth reporting at full weight. Below it, "tucked" is treated as a decision:
# the finding is still emitted, because the producer may want to know, but it is
# held to an observation.
_TUCKED_SURE_AT = 0.80

# The ceiling that holds it there. `_severity` returns "minor" for anything
# under `_MAJOR_AT`, so capping the ratio just below that is the whole mechanism
# — no second severity table, no special case in the scorer.
_TUCKED_MAX_RATIO = _MAJOR_AT - 0.01


def _normalise_intent(intent: Optional[str]) -> TrackIntent:
    key = str(intent or "full_mix").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "mix": "full_mix", "full": "full_mix", "song": "full_mix", "master": "full_mix",
        "instrumental_beat": "beat", "type_beat": "beat",
        "rough": "demo", "sketch": "demo", "wip": "demo",
        "ref": "reference",
        "stems": "stem", "single_stem": "stem",
    }
    key = aliases.get(key, key)
    # Read off the contract rather than a second copy of it: a value added to
    # `TrackIntent` cannot be silently rejected here.
    if key not in TRACK_INTENTS:
        return "full_mix"
    return key  # type: ignore[return-value]

# Health-score points recoverable if a dimension's worst problem is fixed.
# These are ceiling values, reached only by a fully critical finding; the actual
# impact scales with the miss. Ordered by how much a listener notices.
_IMPACT_WEIGHT: Dict[str, float] = {
    "clipping": 12.0,
    "phase": 12.0,
    "loudness": 6.0,
    "limiter": 9.0,
    "dynamic_range": 7.0,
    "compression": 6.0,
    "frequency_balance": 9.0,
    "mud": 9.0,
    "harshness": 8.0,
    "low_end": 9.0,
    "vocal_balance": 8.0,
    "stereo_width": 5.0,
    "transients": 6.0,
    "clarity": 6.0,
}

# A second finding in the same dimension is mostly the same fix, so it recovers
# less; a third, less again.
_REPEAT_DISCOUNT = (1.0, 0.45, 0.25)

# `impact` carries no deviation discount either, for the same reason
# `engine.compute_health` does not: a deviation now pulls the score by its
# measured distance, so closing one returns what closing one actually returns.
# Discounting here while the score charges full price made `ceiling_score`
# quietly disagree with `health_score` — the report would show a mix at 65 with
# 8 points of recoverable work when the work was worth 20.
#
# Defects still read *first*. That is an ordering preference and it lives in
# `_rank_key`, where changing it cannot corrupt a number the ceiling is built
# from.

# Total recoverable points across the whole report. `ceiling_score` is built
# from these, so they must never be able to sum past 100 — 60 leaves the
# ceiling honest (fixing everything does not make an average mix a 100).
_IMPACT_TOTAL_CAP = 60.0

MAX_FINDINGS = 12

# A miss smaller than a tenth of one tolerance unit is a rounding artefact, not
# a problem: reporting "0.00 narrower than the genre" is worse than silence.
# This is a significance floor on the miss, not a second window.
MIN_REPORTABLE_RATIO = 0.10

# Delivery ceiling. Lossy encoders reconstruct between the samples and overshoot
# by up to ~1 dB; every major platform asks for -1.0 dBTP or lower. This is
# codec arithmetic, not taste, so it does not vary by genre.
TRUE_PEAK_CEILING_DBTP = -1.0
# Slack on the ceiling comparison. See the note in the clipping detector.
TRUE_PEAK_TOLERANCE_DB = 0.1

# Two fully decorrelated channels lose 3.01 dB when summed. That is arithmetic.
# Anything past this is cancellation, in any genre.
MONO_LOSS_FLOOR_DB = -3.5

# Below 25 Hz, at this analysis resolution, a strong 40-60 Hz fundamental leaks
# into the measurement window. Under -20 dB the rumble figure cannot be
# separated from that leakage, so nothing is claimed there. This is the *floor*
# of the ceiling — see `_rumble_ceiling_db`, which raises it for the genres
# whose own target curve asks for energy down there.
RUMBLE_CEILING_DB = -20.0

# How far above the level its own target curve implies a genre is allowed to sit
# before the sub-25 Hz energy is called rumble. The measurement is a ratio of two
# wide-band sums, so a couple of dB of it is the curve interpolation and the
# 5.86 Hz analysis grid rather than content.
RUMBLE_TOLERANCE_DB = 3.0


def _rumble_ceiling_db(genre: str) -> float:
    """The sub-25 Hz ceiling for one genre, derived from that genre's own curve.

    A flat -20 dB ceiling contradicts `targets.py`. The trap/hip-hop anchor curve
    carries +2.0 dB at 20 Hz and +3.9 dB at 25 Hz, which works out to a sub-25 Hz
    share of -16.0 dB for a mix sitting *exactly* on the curve — 4 dB past the
    flat ceiling. The consequence was a major "inaudible energy below 25 Hz"
    finding against mixes whose only crime was matching the target the same
    module handed them, systematically, on every bass-forward genre (trap and
    hip-hop at -16.0 dB implied, EDM and D&B at -17.5).

    So the ceiling is the genre's own implied share plus a tolerance, and never
    below the flat floor: this only ever relaxes the test for a genre that asks
    for sub energy, and leaves folk, rock and lo-fi exactly where they were.
    """
    try:
        centers = np.asarray(targets.THIRD_OCTAVE_CENTERS, dtype=np.float64)
        curve = targets.target_curve(genre, targets.THIRD_OCTAVE_CENTERS)
        power = np.power(10.0, np.asarray(curve, dtype=np.float64) / 10.0)
        total = float(power.sum())
        below = float(power[centers < 25.0].sum())
        if total <= 0.0 or below <= 0.0:
            return RUMBLE_CEILING_DB
        implied = 10.0 * math.log10(below / total)
        if not math.isfinite(implied):
            return RUMBLE_CEILING_DB
        return max(RUMBLE_CEILING_DB, implied + RUMBLE_TOLERANCE_DB)
    except Exception:
        return RUMBLE_CEILING_DB

# Minimum duration, in seconds, before a measurement means anything. Most of
# this module is statistics over time; on a 1.2 s file a loudness range or a
# pumping index is measuring the file's edges.
_MIN_SEC: Dict[str, float] = {
    "loudness": 3.0,
    "limiter": 5.0,
    "dynamics": 5.0,
    "arrangement": 30.0,   # loudness range needs an arrangement to range over
    "transients": 5.0,
    "vocal": 8.0,
    "clarity": 3.0,
    "kick": 5.0,
    # A separated source needs far less material to be trustworthy than a
    # center estimate does: the 8 s under "vocal" buys enough syllabic
    # modulation to decide whether a voice is *there*, and a stem has already
    # answered that. What is left is the ordinary "is this statistic settled"
    # question, which four seconds covers.
    "stem": 4.0,
}

# Under this integrated loudness there is no program to have an opinion about.
NO_PROGRAMME_LUFS = -50.0

_CENTERS = np.asarray(THIRD_OCTAVE_CENTERS, dtype=np.float64)


# ---------------------------------------------------------------------------
# Stem-derived thresholds
#
# None of these have a home in `targets.py` — it predates separation and is
# shared with concurrent work — so they live here and are derived from figures
# that file *does* carry wherever a genre has an opinion.
# ---------------------------------------------------------------------------

# Two sources this close in pitch beat rather than stack. At 3 semitones a
# 55 Hz 808 and the kick above it differ by 10 Hz, which the ear still resolves
# as two pitches; inside that, the difference tone lands in the same critical
# band as both of them and the low end reads as one thick object. This is the
# comparison a two-track cannot make honestly — `lowend.py` reconstructs the
# kick's spectrum by subtraction — and it is the whole reason separation earns
# its runtime here.
_SAME_NOTE_SEMITONES = 3.0
_SEMITONE_SCALE = 1.5

# A separated source's level over the frames it is actually sounding on. Same
# figure `separation._ACTIVE_BELOW_STEM_DB` gates its own statistics with, so a
# spread computed here describes the same frames the stem's crest does.
_STEM_ACTIVE_BELOW_PEAK_DB = 35.0
_STEM_MIN_SERIES = 8

# Per-element compression needs the gain-reduction estimate to agree before it
# is claimed on anything but the drums: a vocal with little crest may simply be
# a quiet, even performance.
_STEM_GR_CORROBORATION_DB = 3.0

# Source-against-source masking. `clarity.py` fades a band from audible to
# fully masked over 12 dB under the level its neighborhood spreads onto it;
# the same 12 dB is what "buried" means between two sources, so the ceiling
# sits below it and one tolerance unit is half of it.
_MASK_PAIR_CEILING_DB = 9.0
_MASK_PAIR_SCALE_DB = 6.0
_MASK_PAIR_MIN_OVERLAP_DB = 6.0
# Never critical on its own: this is a model's output about a model's output,
# and a masking pair alone should not be able to pin the health score.
_MASK_PAIR_MAX_RATIO = 2.4
# The bands the mud detector already owns. A masking pair in here is the same
# problem as the low-mid buildup, described from the other side, so mud
# absorbs it instead of the report saying it twice.
_MASK_MUD_BANDS: Tuple[str, ...] = ("upper_bass", "low_mid", "mid")

# `clarity.masking_index` is the loudness-weighted share of the mix's audible
# loudness sitting under its own masking threshold. Shares of loudness are
# small numbers: measured on the calibration fixtures, the three on-target
# reference tracks land at 1.5-1.8% and the deliberately congested mixes at
# 6.5-7.2%, which matches the DSP module's own documented range (~2% for an
# on-target master, 10-15% for a badly arranged one) and its 12% "as masked as
# it gets".
#
# `profile.masking_max` is 0.55 for every genre, written on the 0-1 "share of
# the 1/3-octave grid" scale the coarser model used before the ERB/spreading
# rewrite. Comparing a number that tops out near 0.15 against a 0.55 ceiling is
# a window that can never be missed, so the ceiling is converted here rather
# than in `targets.py`, which is shared with concurrent work. The conversion is
# written to be idempotent: a `masking_max` already on the new scale (anything
# at or under `_MASKING_INDEX_FULL`) is taken as-is, so whoever owns
# `targets.py` can move it without this silently squaring the correction.
_MASKING_INDEX_FULL = 0.12

# What a master with nothing to say about its masking buries. One tolerance
# unit is the distance from there to the genre's ceiling — "how much room is
# there between a clean mix and a congested one" — rather than an arbitrary
# fraction of the ceiling itself.
_MASKING_ON_TARGET = 0.02


# ---------------------------------------------------------------------------
# Section-derived thresholds
# ---------------------------------------------------------------------------

# Below three sections there is no form to have an opinion about, and under
# ~100 s there is no room for three sections long enough to measure.
_MIN_SECTIONS_FOR_FORM = 3
_MIN_FORM_SEC = 100.0

# What "the chorus lifts" means as a number. Most records put 2-4 LU between
# the verse and the hook; under 1.5 LU the difference is measurable and not
# audible, which is the specific complaint this catches.
_CHORUS_LIFT_MIN_LU = 1.5
_CHORUS_LIFT_SCALE = 1.0

# How much the sub+low-bass *share* of a section may move across the sections
# carrying the record before the bottom is falling out of one of them. Scaled
# by genre below: a house record whose drop loses its sub has lost the record.
_LOW_SWING_BASS_LED_DB = 3.0
_LOW_SWING_SONG_DB = 4.5
_LOW_SWING_FREE_DB = 7.0
_LOW_SWING_SCALE_DB = 3.0
_LOW_CORE_LU = 9.0          # matches sections.LOW_CORE_LU
_AUDIBLE_FLOOR_LUFS = -60.0  # matches sections.AUDIBLE_FLOOR_LUFS


# ---------------------------------------------------------------------------
# Arrangement roles
#
# `sections.py` names every span from what it measured — intro, verse, chorus,
# drop, bridge, outro, or "section N" where it could not justify a name. That
# label was the one piece of context the section detectors ignored, and
# ignoring it is what produced "the dominant issue is low end steps back in
# intro" against a record whose intro is deliberately empty. The swing was
# real; the verdict was about the wrong section.
#
# Three roles, because three different questions are being asked of them:
#
# * **bookend** — intro, outro. Pulling the bottom out at the top or tail of a
#   record is how arrangements work. Nothing in the audio separates that from a
#   mistake, and the prior is overwhelmingly "deliberate", so this stays quiet
#   unless the departure is extreme *and* the section is long enough for a
#   listener to sit in it. When it does speak it says so as an aside.
# * **payoff** — chorus, drop. The section the record is built to arrive at. A
#   hook with nothing under it is the case worth flagging and keeps the full
#   strength it has today.
# * **body** — verse, bridge, "section N". Genuinely ambiguous: reported, with
#   a little more room than the payoff gets, and always with the question
#   attached.
# ---------------------------------------------------------------------------

_BOOKEND_LABELS: frozenset = frozenset({"intro", "outro"})
_PAYOFF_LABELS: frozenset = frozenset({"chorus", "drop"})

# How far past the genre's own swing ceiling each role has to go before the
# bottom falling out of it is worth a sentence. 1.0 is today's behavior, kept
# for the section that matters; the bookend factor is "roughly double", which
# on trap moves the ceiling from 3.0 dB to 6.0 dB — past anything an
# arrangement does casually.
_ROLE_SWING_FACTOR: Dict[str, float] = {"payoff": 1.0, "body": 1.35, "bookend": 2.0}

# And how much of the record a bookend has to occupy before its missing bottom
# end can matter to a listener. A 6 s intro on a 3-minute record is 3% of it:
# gone before the ear has decided anything is absent.
_BOOKEND_MIN_SHARE = 0.12

# The ceiling that holds an arrangement aside to "minor". Same mechanism as
# `_TUCKED_MAX_RATIO`: `_severity` returns "minor" under `_MAJOR_AT`, so
# capping the ratio just below it caps the label with no second severity table
# — and because `deviation_penalty` reads the ratio rather than the label, it
# caps what the finding can cost the score at the same time.
_ARRANGEMENT_ASIDE_MAX_RATIO = _MAJOR_AT - 0.01


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _fin(value, default: float = 0.0) -> float:
    """Coerce to a finite float. NaN/inf serialise to invalid JSON."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(min(max(_fin(value, lo), lo), hi))


def _severity(ratio: float) -> Severity:
    """Shared miss-ratio -> severity mapping. See rule 2 in the module docstring."""
    r = _fin(ratio, 0.0)
    if r >= _CRITICAL_AT:
        return "critical"
    if r >= _MAJOR_AT:
        return "major"
    if r > 0.0:
        return "minor"
    return "clean"


def _ratio(miss: float, scale: float) -> float:
    """|miss| in units of one tolerance step."""
    return abs(_fin(miss, 0.0)) / max(abs(_fin(scale, 1.0)), 1e-9)


def _verdict(value: float, window: Tuple[float, float], scale: float) -> Verdict:
    miss = targets.range_miss(_fin(value), window)
    if miss == 0.0:
        return "good"
    return "problem" if _ratio(miss, scale) >= _MAJOR_AT else "watch"


def _num(value: float, nd: int = 1) -> str:
    """Format a figure for a sentence: no trailing '.0' noise, no -0.0."""
    v = _fin(value, 0.0)
    if abs(v) < 5e-3:
        v = 0.0
    text = f"{v:.{nd}f}"
    return "0" if text in ("-0", "-0.0", "-0.00") else text


def _win(window: Tuple[float, float], nd: int = 1, unit: str = "") -> str:
    return f"{_num(window[0], nd)} to {_num(window[1], nd)}{unit}"


def _pm(value: float) -> str:
    """'+/-2.2' — the tolerance band around a target, for a sentence."""
    return f"+/-{_num(value, 1)}"


def _ev(
    label: str,
    value: float,
    unit: str = "",
    target: Optional[float] = None,
    target_range: Optional[Tuple[float, float]] = None,
    verdict: Verdict = "good",
    detail: str = "",
) -> Evidence:
    return Evidence(
        label=label,
        value=round(_fin(value), 4),
        unit=unit,
        target=None if target is None else round(_fin(target), 4),
        target_range=(
            None
            if target_range is None
            else (round(_fin(target_range[0]), 4), round(_fin(target_range[1]), 4))
        ),
        verdict=verdict,
        detail=detail,
    )


def _moments(source: Optional[Sequence[Moment]], limit: int = 6) -> List[Moment]:
    """Copy the DSP layer's moments onto a finding, worst first."""
    if not source:
        return []
    ranked = sorted(source, key=lambda mo: -_fin(mo.intensity, 0.0))[:limit]
    return sorted(ranked, key=lambda mo: _fin(mo.t_start, 0.0))


def _clock(seconds: float) -> str:
    s = max(_fin(seconds, 0.0), 0.0)
    return f"{int(s // 60)}:{s % 60:04.1f}"


def _moment_span(moments: Sequence[Moment]) -> str:
    """'worst at 0:12.1' / 'worst at 0:12.1 and 3 other spans'."""
    if not moments:
        return ""
    worst = max(moments, key=lambda mo: _fin(mo.intensity, 0.0))
    if len(moments) == 1:
        return f" Worst at {_clock(worst.t_start)}."
    return f" Worst at {_clock(worst.t_start)}, across {len(moments)} flagged spans."


# ---------------------------------------------------------------------------
# Genre-curve derivations
#
# `targets.py` windows the metrics it names. For the two region ratios it does
# not name (`mud_to_mid_db`, `boxiness_db`) the target is derived from the
# genre's own curve rather than invented here, so a new genre added to
# targets.py automatically gets correct windows for both.
# ---------------------------------------------------------------------------


def _curve_region_db(curve: np.ndarray, lo: float, hi: float) -> float:
    """Power-mean of a target curve over a region, 1/3-octave summed convention.

    Identical maths to `targets.macro_target`, but taking the already-computed
    curve so a detector can ask about arbitrary regions.
    """
    mask = (_CENTERS >= lo) & (_CENTERS < hi)
    if not np.any(mask):
        return 0.0
    return float(10.0 * np.log10(np.mean(10.0 ** (curve[mask] / 10.0))))


def _log_center(lo: float, hi: float) -> float:
    return float(math.sqrt(max(lo, 1e-9) * max(hi, 1e-9)))


def _curve_density_db(curve: np.ndarray, region: Tuple[float, float],
                      reference: Tuple[float, float]) -> float:
    """Target for a power-*density* ratio between two regions of the curve.

    The DSP layer reports `mud_ratio_db`, `mud_to_mid_db` and friends as mean
    power **per FFT bin**, so region width cancels. A target curve is written in
    the 1/3-octave **summed** convention, where band power grows with center
    frequency for flat spectral density. Converting one to the other is a single
    -10log10(f) term:

        density(A)/density(B) = level(A)/level(B) * f_B/f_A

    Validated against the one density ratio `targets.py` does window: for pop
    this reproduces -8.7 dB for `mud_ratio_db`, inside pop's (-14.0, -5.5).
    """
    level = _curve_region_db(curve, *region) - _curve_region_db(curve, *reference)
    width = 10.0 * math.log10(_log_center(*reference) / max(_log_center(*region), 1e-9))
    return _fin(level + width, 0.0)


def _curve_tilt(curve: np.ndarray, third_octave_db: Sequence[float]) -> float:
    """Tilt of the genre curve, fitted over the bands the DSP layer actually used.

    `spectral.py` fits its tilt over 40 Hz-16 kHz, excluding bands more than
    45 dB under the loudest one, so a brick-walled top end does not invent a
    slope. Reconstructing that mask here from `third_octave_db` makes the two
    numbers directly comparable.
    """
    measured = np.asarray(third_octave_db, dtype=np.float64)
    if measured.size != _CENTERS.size or not np.any(np.isfinite(measured)):
        return 0.0
    mask = (
        (_CENTERS >= 40.0)
        & (_CENTERS <= 16_000.0)
        & (measured > float(np.max(measured)) - 45.0)
    )
    if int(np.count_nonzero(mask)) < 5:
        return 0.0
    return _fin(np.polyfit(np.log10(_CENTERS[mask]), curve[mask], 1)[0], 0.0)


@dataclass
class _BandView:
    """One macro band re-scored against the genre asked for now.

    `SpectralMeasurement.bands` already carries a target, but it was computed at
    whatever genre the measurement pass ran under. Recomputing here makes the
    detector correct regardless, and is the only place band verdicts come from.
    """

    name: str
    low_hz: float
    high_hz: float
    center_hz: float
    level_db: float
    target_db: float
    tolerance_db: float
    deviation_db: float
    miss_db: float

    @property
    def ratio(self) -> float:
        return _ratio(self.miss_db, self.tolerance_db)

    @property
    def label(self) -> str:
        return _BAND_LABELS.get(self.name, self.name.replace("_", " "))

    @property
    def span(self) -> str:
        return f"{_num(self.low_hz, 0)}-{_num(self.high_hz, 0)} Hz"

    @property
    def is_plural(self) -> bool:
        """"the mids run", "the low bass runs".

        Listed rather than inferred: "low bass" and "brilliance" both end in an
        s and neither takes a plural verb.
        """
        return self.name in _PLURAL_BANDS

    def verb(self, singular: str, plural: str) -> str:
        return plural if self.is_plural else singular


# The macro bands whose label takes a plural verb.
_PLURAL_BANDS: frozenset = frozenset({"low_mid", "mid", "upper_mid"})

_BAND_LABELS: Dict[str, str] = {
    "sub": "sub",
    "low_bass": "low bass",
    "upper_bass": "upper bass",
    "low_mid": "low mids",
    "mid": "mids",
    "upper_mid": "upper mids",
    "presence": "presence",
    "brilliance": "brilliance",
    "air": "air",
}


# ---------------------------------------------------------------------------
# Detection context
# ---------------------------------------------------------------------------


@dataclass
class _Hit:
    """A finding plus the miss ratio that produced it.

    The ratio drives impact, and impact has to be assigned globally (it is
    capped across the whole report), so it cannot be baked in at the point the
    detector runs.

    `observation` marks a finding that is not a problem at all — a confirmed
    virtue, or a reading taken off a record nobody is being asked to change. It
    carries no severity, no impact and no significance floor: it exists to be
    read, so the ratio gate that drops trivial misses must not drop it.
    """

    finding: Finding
    ratio: float
    observation: bool = False


def _expects_arrangement_lift(profile: targets.GenreProfile) -> bool:
    """Does this genre's form put a measurable lift into its loudest section?

    A chorus that arrives at the same level as the verse is a real complaint in
    pop, hip-hop, rock and everything shaped like them, and meaningless advice
    for ambient, classical, jazz or a lo-fi loop — those records are not built
    out of a hook that has to land. `targets.py` has no field saying which is
    which and cannot be edited from here, so the answer is derived from three
    that are already there and that already encode it:

    * `vocal_expected` — a song with a lead has a form for the lead to sit in.
      Classical, orchestral, cinematic, ambient and lo-fi are all False.
    * `punch_min` — beat-driven production. Ambient's 0.10 and jazz's 0.28 say
      the material is not built on hits arriving on a grid.
    * `loudness_range_lu` upper bound — where a genre is *allowed* 14-24 LU of
      range, the level movement in it is performance dynamics, not arrangement,
      and a "the chorus does not lift" verdict has nothing to attach to.

    Result: pop, hip-hop, trap, EDM, house, techno, D&B, rock, punk, metal,
    R&B, soul, country, indie and alternative expect a lift. Acoustic, folk,
    jazz, classical, orchestral, cinematic, ambient, lo-fi and the unknown-genre
    "other" fall-back do not.
    """
    return bool(
        getattr(profile, "vocal_expected", False)
        and _fin(profile.punch_min, 0.0) >= 0.30
        and _fin(profile.loudness_range_lu[1], 99.0) <= 10.5
    )


def _low_swing_ceiling(profile: targets.GenreProfile) -> float:
    """How far a section's low-end share may drift from its neighbors'.

    Bass-led genres are identified by the one figure in `targets.py` that says
    the bottom octave is load-bearing rather than decorative: hip-hop, trap,
    EDM, house, techno and D&B all raise `low_end_mono_min` to 0.90 or above
    because the sub is the instrument. A drop that loses its sub has lost the
    record there, so they get the tightest window.
    """
    if _fin(profile.low_end_mono_min, 0.80) >= 0.88:
        return _LOW_SWING_BASS_LED_DB
    return _LOW_SWING_SONG_DB if _expects_arrangement_lift(profile) else _LOW_SWING_FREE_DB


def _stem_level_spread(stem: StemMeasurement) -> Optional[float]:
    """90th minus 10th percentile of a stem's level, over the frames it sounds on.

    The two-track equivalent (`VocalMeasurement.consistency_db`) is the spread
    of a center estimate, which moves when anything centered moves. This is the
    spread of the source itself, so a vocal riding under a centered synth no
    longer reads as an uneven vocal.
    """
    series = np.asarray(
        [_fin(x, -120.0) for x in (stem.level_series or [])], dtype=np.float64
    )
    if series.size < _STEM_MIN_SERIES:
        return None
    active = series[series > float(series.max()) - _STEM_ACTIVE_BELOW_PEAK_DB]
    if active.size < _STEM_MIN_SERIES:
        return None
    return _fin(float(np.percentile(active, 90.0) - np.percentile(active, 10.0)), 0.0)


def _section_low_shares(
    sections: Sequence[Section],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(low share dB, integrated LUFS, core mask) per section.

    "Low share" is sub+low-bass against the section's own total, which is what
    makes this independent of the section simply being quieter — a bass-free
    intro is an arrangement, a bass-free chorus is a fault. Same derivation
    `sections.py` uses for `low_end_swing_db`, recomputed here because the
    per-section figures are what a finding has to name.
    """
    n = len(sections)
    low = np.zeros(n, dtype=np.float64)
    lufs = np.zeros(n, dtype=np.float64)
    for i, section in enumerate(sections):
        levels = section.band_levels_db or {}
        sub = 10.0 ** (_fin(levels.get("sub", -120.0), -120.0) / 10.0)
        low_bass = 10.0 ** (_fin(levels.get("low_bass", -120.0), -120.0) / 10.0)
        total = sum(10.0 ** (_fin(v, -120.0) / 10.0) for v in levels.values())
        low[i] = _clamp(
            10.0 * math.log10(max(sub + low_bass, 1e-12) / max(total, 1e-12)), -40.0, 0.0
        )
        lufs[i] = _fin(section.integrated_lufs, -70.0)

    audible = lufs > _AUDIBLE_FLOOR_LUFS
    if not np.any(audible):
        audible = np.ones(n, dtype=bool)
    core = audible & (lufs >= float(np.max(lufs[audible])) - _LOW_CORE_LU)
    return low, lufs, core


def _section_role(label: str) -> str:
    """bookend / payoff / body, from the name `sections.py` gave the span.

    Deliberately total: an unrecognized or numbered label falls to "body", the
    middle-ground treatment, so a new label added to the segmenter can never
    silently acquire either the suppression or the full strength.
    """
    key = str(label or "").strip().lower()
    if key in _BOOKEND_LABELS:
        return "bookend"
    if key in _PAYOFF_LABELS:
        return "payoff"
    return "body"


def _section_share(section: Section, duration: float) -> float:
    """How much of the record one section occupies, 0-1."""
    span = _fin(section.t_end, 0.0) - _fin(section.t_start, 0.0)
    return _clamp(span / max(_fin(duration, 0.0), 1e-6), 0.0, 1.0)


def _low_swing_case(
    sections: Sequence[Section],
    low_rel: np.ndarray,
    idx: Sequence[int],
    swing_max: float,
    duration: float,
) -> Optional[Tuple[int, int, float, float, str]]:
    """The worst low-end swing inside one group of sections, if it clears its bar.

    Returns `(weakest, strongest, swing_db, ceiling_db, role)`, where the bar
    the swing had to clear is set by the *role of the weakest section* — which
    is the whole point. The same 5 dB of swing is an ordinary arrangement when
    it is the intro that is empty and a broken drop when it is the drop.
    """
    if len(idx) < 2:
        return None
    weak_i = min(idx, key=lambda i: _fin(low_rel[i], 0.0))
    strong_i = max(idx, key=lambda i: _fin(low_rel[i], 0.0))
    role = _section_role(sections[weak_i].label)
    ceiling = max(_fin(swing_max, _LOW_SWING_SONG_DB), 0.1) * _ROLE_SWING_FACTOR.get(role, 1.0)
    swing = _fin(low_rel[strong_i], 0.0) - _fin(low_rel[weak_i], 0.0)
    if targets.range_miss(swing, (0.0, ceiling)) <= 0.0:
        return None
    # A bookend also has to be long enough to be sat in. Below that the
    # listener is out of it before the missing bottom registers as anything.
    if role == "bookend" and _section_share(sections[weak_i], duration) < _BOOKEND_MIN_SHARE:
        return None
    return weak_i, strong_i, swing, ceiling, role


def _masking_window(profile: targets.GenreProfile) -> Tuple[Tuple[float, float], float]:
    """(window, tolerance unit) for `clarity.masking_index` in this genre.

    See `_MASKING_INDEX_FULL`. `profile.masking_max` still decides how much
    this genre absorbs relative to every other one; only the scale moves.
    """
    raw = _fin(profile.masking_max, 0.55)
    ceiling = _clamp(raw * _MASKING_INDEX_FULL if raw > _MASKING_INDEX_FULL else raw,
                     0.005, 1.0)
    return (0.0, ceiling), max(ceiling - _MASKING_ON_TARGET, 0.01)


def _mask_ratio(pair: MaskingPair) -> float:
    """Miss ratio for one masking pair, from its measured overlap."""
    miss = targets.range_miss(_fin(pair.overlap_db, 0.0), (0.0, _MASK_PAIR_CEILING_DB))
    return min(_ratio(miss, _MASK_PAIR_SCALE_DB), _MASK_PAIR_MAX_RATIO)


def _mask_sentence(
    pairs: Sequence[MaskingPair], lead: str = "Separating the sources names who is doing it:"
) -> str:
    """One clause per claimed masking pair, in the DSP layer's own words."""
    if not pairs:
        return ""
    text = f" {lead.strip()} " + pairs[0].detail.strip()
    if len(pairs) > 1:
        text += " " + " ".join(p.detail.strip() for p in pairs[1:])
    return text


def _mask_evidence(pairs: Sequence[MaskingPair]) -> List[Evidence]:
    return [
        _ev(
            f"{p.masker.capitalize()} over {p.maskee} ({p.band.replace('_', ' ')})",
            _fin(p.overlap_db, 0.0), "dB",
            target_range=(0.0, _MASK_PAIR_CEILING_DB),
            verdict=_verdict(_fin(p.overlap_db, 0.0), (0.0, _MASK_PAIR_CEILING_DB),
                             _MASK_PAIR_SCALE_DB),
            detail=f"Measured between separated sources over "
                   f"{_num(p.low_hz, 0)}-{_num(p.high_hz, 0)} Hz, not inferred "
                   f"from the two-track.",
        )
        for p in pairs
    ]


@dataclass
class _Ctx:
    m: Measurements
    genre_key: str
    profile: targets.GenreProfile
    curve: np.ndarray
    bands: Dict[str, _BandView]
    duration: float
    is_mono: bool
    no_programme: bool
    intent: TrackIntent = "full_mix"
    # Optional depth. Both are always present as objects (never None), so a
    # detector reads `.available` rather than null-checking every access.
    stems: StemAnalysis = field(default_factory=StemAnalysis)
    sections: SectionAnalysis = field(default_factory=SectionAnalysis)
    stems_by_kind: Dict[str, StemMeasurement] = field(default_factory=dict)
    masking_pairs: List[MaskingPair] = field(default_factory=list)
    # Masking pairs already folded into a finding. A pair is claimed exactly
    # once, by whichever detector describes the problem best (rule 5).
    masking_used: set = field(default_factory=set)
    # Tags set by detectors that have already run, so later ones can stand down
    # rather than describe the same energy twice.
    tags: set = field(default_factory=set)

    def has(self, key: str) -> bool:
        return self.duration >= _MIN_SEC.get(key, 0.0)

    # -- intent ------------------------------------------------------------

    @property
    def expects_lead(self) -> bool:
        """Could there be a lead vocal on this file to have an opinion about?

        Intent decides this on its own, deliberately. The genre profile's
        `vocal_expected` says whether records of this kind *usually* carry a
        lead, which is the right input for a clean-report sentence and the wrong
        one for a gate: a lo-fi track with a vocal on it still has a vocal
        balance, and `targets.py` marks lo-fi as vocal_expected=False. Intent is
        a statement about this file rather than about the genre, so it is what
        gets to say there is no lead — and on `full_mix` it never does, which is
        what keeps today's behavior unchanged.
        """
        return self.intent not in _NO_LEAD_INTENTS

    @property
    def lead_confidence(self) -> float:
        """How sure the DSP layer is that a real lead voice is there, 0-1.

        `vocal_present` is this crossing a threshold. Reading the float instead
        is what lets a detector be gentle about a borderline call rather than
        treating a 0.56 and a 0.98 as the same fact.
        """
        return _clamp(_fin(getattr(self.m.vocal, "vocal_confidence", 0.0), 0.0), 0.0, 1.0)

    @property
    def lead_prominence(self) -> str:
        """absent / tucked / balanced / forward — where the lead sits in the bed.

        Forced to "absent" when the intent says no lead belongs on this file, so
        one accessor answers the question and no caller has to remember to check
        both.
        """
        if not self.expects_lead:
            return "absent"
        return str(getattr(self.m.vocal, "vocal_prominence", "absent"))

    @property
    def lead_on_record(self) -> bool:
        """Is a lead vocal actually present *and* expected?

        The gate for anything that describes a voice. The DSP layer's voice test
        is a syllabic-modulation and articulation test on the center channel; on
        a beat it can still fire on a chopped sample or a tucked hook that is not
        the lead at all, which is why intent has a vote.
        """
        return (
            self.expects_lead
            and bool(self.m.vocal.vocal_present)
            and self.lead_prominence != "absent"
        )

    @property
    def lead_is_up_front(self) -> bool:
        """A lead sitting in or above the bed, and confidently enough to own the top.

        The gate for blaming a lead vocal for something. A tucked hook is a real
        voice and still not what is making the top of the mix spit, and a
        borderline voice call is not enough to put the word "sibilance" on a
        finding — that word sends the producer to a de-esser, and a de-esser is
        the wrong tool for a hi-hat.
        """
        return (
            self.lead_on_record
            and self.lead_prominence in ("balanced", "forward")
            and self.lead_confidence >= _LEAD_UP_FRONT_CONFIDENCE
        )

    @property
    def is_reference(self) -> bool:
        return self.intent == "reference"

    def advice(self, text: str) -> str:
        """A prescriptive clause, dropped when the file is somebody else's record.

        On a reference the report's job is to say what the record does. "Back
        the input drive off" is not an observation about a finished master, it
        is an instruction to change it, so it is not written.
        """
        return "" if self.is_reference else text

    @property
    def ref(self) -> str:
        """'the Trap reference' — how a deviation names what it differs from."""
        return f"the {self.profile.label} reference"

    @property
    def has_stems(self) -> bool:
        """True only when separation ran *and* produced at least one real source."""
        return bool(self.stems.available and self.stems_by_kind)

    def stem(self, kind: str) -> Optional[StemMeasurement]:
        """The named source, or None if it was absent or never separated."""
        return self.stems_by_kind.get(kind)

    def take_masking(
        self,
        bands: Optional[Sequence[str]] = None,
        maskee: Optional[str] = None,
        limit: int = 2,
    ) -> List[MaskingPair]:
        """Claim the strongest unclaimed masking pairs, worst first.

        Claiming is what keeps one problem to one finding: vocal balance takes
        the pairs burying the vocal, mud takes the ones in its own bands, and
        whatever is left over goes to clarity.
        """
        out: List[MaskingPair] = []
        for pair in self.masking_pairs:
            key = (str(pair.masker), str(pair.maskee), str(pair.band))
            if key in self.masking_used:
                continue
            if bands is not None and str(pair.band) not in bands:
                continue
            if maskee is not None and str(pair.maskee) != maskee:
                continue
            self.masking_used.add(key)
            out.append(pair)
            if len(out) >= limit:
                break
        return out

    def trust(self, base: float, key: Optional[str] = None) -> float:
        """Confidence, reduced for files too short for the statistic to settle.

        A 6 s window of a 20 s file is a third of the record; the same window of
        a 6-minute track is a sample. Anything time-averaged loses confidence on
        short material even when it clears the hard minimum.
        """
        need = _MIN_SEC.get(key or "", 0.0)
        factor = 1.0
        if need > 0.0:
            factor = _clamp(self.duration / (need * 4.0), 0.55, 1.0)
        return round(_clamp(base * factor, 0.05, 0.99), 2)


def _build_ctx(m: Measurements, genre: str, intent: str = "full_mix") -> _Ctx:
    key = targets.normalise_genre(genre)
    profile = targets.get_profile(key)
    curve = np.asarray(targets.target_curve(key, THIRD_OCTAVE_CENTERS), dtype=np.float64)

    bands: Dict[str, _BandView] = {}
    for band in m.spectral.bands:
        target = _fin(targets.macro_target(key, band.name, band.low_hz, band.high_hz))
        tol = max(_fin(targets.band_tolerance(key, band.name), 3.0), 0.1)
        level = _fin(band.level_db)
        bands[band.name] = _BandView(
            name=band.name,
            low_hz=_fin(band.low_hz),
            high_hz=_fin(band.high_hz),
            center_hz=_fin(band.center_hz),
            level_db=level,
            target_db=target,
            tolerance_db=tol,
            deviation_db=level - target,
            miss_db=targets.range_miss(level, (target - tol, target + tol)),
        )

    stems = m.stems or StemAnalysis()
    # Only sources the separator actually found. An instrumental's `vocals`
    # array comes back full of bleed at -50 dB, and `present=False` is how the
    # DSP layer says so — a detector that read it anyway would report a level
    # for a singer who is not on the record.
    stems_by_kind: Dict[str, StemMeasurement] = {}
    masking_pairs: List[MaskingPair] = []
    if stems.available:
        stems_by_kind = {str(s.kind): s for s in (stems.stems or []) if s.present}
        masking_pairs = sorted(
            (
                p for p in (stems.masking_pairs or [])
                if _fin(p.overlap_db, 0.0) >= _MASK_PAIR_MIN_OVERLAP_DB
                and str(p.masker) in stems_by_kind
                and str(p.maskee) in stems_by_kind
            ),
            key=lambda p: -_mask_ratio(p),
        )

    return _Ctx(
        m=m,
        genre_key=key,
        profile=profile,
        curve=curve,
        bands=bands,
        duration=max(_fin(m.duration_seconds, 0.0), 0.0),
        is_mono=bool(m.is_mono or m.stereo.is_mono_source),
        intent=_normalise_intent(intent),
        no_programme=(
            _fin(m.loudness.integrated_lufs, -70.0) <= NO_PROGRAMME_LUFS
            or _fin(m.loudness.true_peak_dbtp, -120.0) <= NO_PROGRAMME_LUFS
        ),
        stems=stems,
        sections=m.sections or SectionAnalysis(),
        stems_by_kind=stems_by_kind,
        masking_pairs=masking_pairs,
    )


# ---------------------------------------------------------------------------
# 1. Clipping & true peak
# ---------------------------------------------------------------------------


def _detect_clipping(ctx: _Ctx) -> List[_Hit]:
    """Pinned samples and delivery headroom.

    Genre-independent by design: 0 dBFS is arithmetic, and a lossy encoder
    overshoots a trap master exactly as far as it overshoots a string quartet.
    """
    c = ctx.m.clipping
    hits: List[_Hit] = []

    clip_pct = _fin(c.clip_percentage, 0.0)
    longest = int(_fin(c.longest_flat_run, 0))
    runs = int(_fin(c.flat_run_count, 0))
    distortion = _fin(c.distortion_index, 0.0)
    true_peak = _fin(c.true_peak_dbtp, -120.0)
    overs = int(_fin(c.inter_sample_overs, 0))

    # A tolerance, because the ceiling is a delivery spec and not a knife edge.
    #
    # True peak is estimated by 4x oversampling, so the reading carries its own
    # small error, and every limiter's own meter disagrees with ours by about
    # this much. Without the slack a file at -0.99 dBTP is one hundredth of a
    # decibel over, and the report calls it a defect and blocks mastering —
    # which reads as pedantry and costs the score its credibility on exactly
    # the tracks that got it right. 0.1 dB is inaudible and well inside what
    # any encoder cares about; a master that is genuinely hot fails by whole
    # decibels, not by rounding.
    tp_window = (-120.0, TRUE_PEAK_CEILING_DBTP + TRUE_PEAK_TOLERANCE_DB)
    tp_miss = targets.range_miss(true_peak, tp_window)

    # -- flat tops: the waveform itself is damaged ---------------------------
    # Window is (0, 0): a mix that has been clipped has samples pinned at its
    # own ceiling, and the correct number of those is zero.
    if clip_pct > 0.0 and runs >= 1:
        ratio = max(
            _ratio(clip_pct, 0.03),          # 0.03 % pinned is already audible
            _ratio(longest, 12.0),           # 12 consecutive samples is a plateau
            _ratio(distortion, 0.12),        # HF products the clean frames lack
        )
        ratio = min(ratio, 6.0)
        evidence = [
            _ev("Clipped samples", c.clipped_samples, "samples",
                target=0.0, verdict="problem",
                detail=f"{clip_pct:.4f}% of all samples sit at the file's ceiling."),
            _ev("Longest flat run", longest, "samples", target=0.0,
                verdict="problem" if longest >= 12 else "watch",
                detail="Consecutive samples with no movement — a plateau, not a peak."),
            _ev("Flat-topped runs", runs, "runs", target=0.0, verdict="problem"),
            _ev("Distortion index", distortion, "",
                target_range=(0.0, _distortion_ceiling(ctx.profile)),
                verdict=_verdict(distortion, (0.0, _distortion_ceiling(ctx.profile)), 0.06),
                detail="High-frequency energy in clipped frames vs clean frames "
                       "from this same track."),
            _ev("True peak", true_peak, "dBTP", target=TRUE_PEAK_CEILING_DBTP,
                target_range=tp_window, verdict=_verdict(true_peak, tp_window, 1.0)),
            _ev("Sample peak", _fin(c.sample_peak_dbfs), "dBFS",
                target=TRUE_PEAK_CEILING_DBTP, verdict="problem"),
        ]
        moments = _moments(c.events)
        detail = (
            f"{c.clipped_samples:,} samples ({clip_pct:.4f}% of the file) are pinned at the "
            f"ceiling in {runs} flat-topped runs, the longest {longest} samples long, and the "
            f"clipped frames carry {distortion:.3f} on the distortion index, measured against "
            f"this track's own unclipped frames — the waveform tops are squared off, so that is "
            f"harmonic content the material never had, baked into the render rather than gain "
            f"that can be turned back down."
            + _moment_span(moments)
        )
        if bool(c.is_float_over_unity):
            detail += (
                " The file is float and peaks above unity, so the ceiling measured against "
                "is the material's own, not 0 dBFS."
            )
        hits.append(_Hit(
            Finding(
                id="clipping.hard_clipping",
                dimension="clipping",
                title="Hard clipping: the waveform tops are squared off",
                severity=_severity(ratio),
                confidence=0.97,   # counted samples, not an inference
                detail=detail,
                evidence=evidence,
                moments=moments,
            ),
            ratio,
        ))
        ctx.tags.add("clipping")

    # -- delivery headroom ---------------------------------------------------
    # Folded into the clipping finding above when both are true: a mix pinned at
    # its ceiling and a mix over -1 dBTP are the same fact stated twice.
    elif tp_miss > 0.0:
        ratio = _ratio(tp_miss, 1.0)
        detail = (
            f"True peak reaches {_num(true_peak, 2)} dBTP, {_num(tp_miss, 2)} dB above the "
            f"{_num(TRUE_PEAK_CEILING_DBTP, 1)} dBTP ceiling every major platform asks for "
            f"(Amazon Music wants -2.0). Nothing is clipped in the file itself — sample peak "
            f"is {_num(c.sample_peak_dbfs, 2)} dBFS — but MP3 and AAC reconstruct the waveform "
            f"between the samples, and the decoder will clip what the file does not."
        )
        if overs > 0:
            detail += (
                f" 4x oversampling already finds {overs:,} inter-sample overs that are not "
                f"over at base rate."
            )
        hits.append(_Hit(
            Finding(
                id="clipping.true_peak_over",
                dimension="clipping",
                title="True peak above the delivery ceiling",
                severity=_severity(ratio),
                confidence=0.96,
                detail=detail,
                evidence=[
                    _ev("True peak", true_peak, "dBTP", target=TRUE_PEAK_CEILING_DBTP,
                        target_range=tp_window, verdict="problem"),
                    _ev("Sample peak", _fin(c.sample_peak_dbfs), "dBFS",
                        target=TRUE_PEAK_CEILING_DBTP,
                        verdict=_verdict(_fin(c.sample_peak_dbfs), tp_window, 1.0)),
                    _ev("Inter-sample overs", overs, "samples", target=0.0,
                        verdict="problem" if overs else "good"),
                ],
                moments=_moments(c.events),
            ),
            ratio,
        ))
        ctx.tags.add("clipping")

    return hits


# ---------------------------------------------------------------------------
# 2. Phase & mono compatibility
# ---------------------------------------------------------------------------


def _detect_phase(ctx: _Ctx) -> List[_Hit]:
    """Polarity, correlation and what survives a fold-down.

    Two different kinds of threshold meet here and it matters which is which.
    *How wide a field the genre wants* is taste, and comes from
    `profile.correlation_min` — ambient's 0.10 against hip-hop's 0.40 is why a
    deliberately decorrelated ambient bed is never called a fault. *How much a
    mono fold-down may lose* is arithmetic: two fully decorrelated channels lose
    3.01 dB when summed and nothing legitimately loses more, in any genre.

    A mono source has no inter-channel relationship to break, so nothing is
    reported at all.
    """
    if ctx.is_mono:
        return []

    p = ctx.m.phase
    hits: List[_Hit] = []

    corr = _clamp(_fin(p.correlation, 1.0), -1.0, 1.0)
    loss = _fin(p.mono_sum_loss_db, 0.0)
    corr_min = _fin(ctx.profile.correlation_min, 0.30)
    corr_window = (corr_min, 1.0)
    loss_window = (MONO_LOSS_FLOOR_DB, 60.0)
    corr_miss = targets.range_miss(corr, corr_window)
    loss_miss = targets.range_miss(loss, loss_window)
    moments = _moments(p.problem_moments)

    worst_band = p.worst_band or ""
    worst_band_corr = _fin(p.worst_band_correlation, 1.0)
    band_losses = {k: _fin(v, 0.0) for k, v in (p.band_mono_loss_db or {}).items()}
    worst_loss_band, worst_loss = ("", 0.0)
    if band_losses:
        worst_loss_band = min(band_losses, key=lambda k: band_losses[k])
        worst_loss = band_losses[worst_loss_band]

    def _shared_evidence() -> List[Evidence]:
        return [
            _ev("Correlation", corr, "", target_range=corr_window,
                verdict=_verdict(corr, corr_window, 0.35),
                detail=f"{ctx.profile.label} sits above {_num(corr_min, 2)}."),
            _ev("Mono sum loss", loss, "dB", target_range=loss_window,
                verdict=_verdict(loss, loss_window, 2.0),
                detail="Level of (L+R)/2 against the average channel level; "
                       "-3.0 dB is the arithmetic cost of full decorrelation."),
            _ev("Worst band correlation", worst_band_corr, "",
                target_range=corr_window,
                verdict=_verdict(worst_band_corr, corr_window, 0.35),
                detail=f"Macro band: {worst_band or 'n/a'}."),
            _ev(f"Mono loss in {worst_loss_band or 'worst band'}", worst_loss, "dB",
                target_range=loss_window,
                verdict=_verdict(worst_loss, loss_window, 2.0)),
        ]

    # -- polarity inversion --------------------------------------------------
    # Severity comes from the fold-down loss rather than from the correlation
    # miss, precisely so it does not soften for a genre with a wide window:
    # ambient is allowed a decorrelated field, not a canceling one.
    if bool(p.polarity_inverted):
        ratio = _ratio(loss_miss if loss_miss else (corr - (-1.0)) - 1.0, 2.0)
        ratio = max(ratio, _ratio(corr_miss, 0.35) * 0.5, 1.6)
        evidence = _shared_evidence()
        low_mono = _fin(ctx.m.low_end.low_end_mono_ratio, 1.0)
        evidence.append(
            _ev("Low-end mono ratio", low_mono, "",
                target_range=(_fin(ctx.profile.low_end_mono_min, 0.8), 1.0),
                verdict=_verdict(low_mono,
                                 (_fin(ctx.profile.low_end_mono_min, 0.8), 1.0), 0.06),
                detail="Share of sub-120 Hz energy that is mono.")
        )
        detail = (
            f"The channels are in opposite polarity: correlation reads {_num(corr, 2)} and "
            f"summing to mono costs {_num(loss, 1)} dB, against the {_num(MONO_LOSS_FLOOR_DB, 1)} dB "
            f"that full decorrelation alone would cost. {worst_loss_band.replace('_', ' ') or 'The worst band'} "
            f"loses {_num(worst_loss, 1)} dB and only {low_mono * 100:.0f}% of the energy below "
            f"120 Hz survives the fold-down. This is a flipped cable or an inverted plugin, not "
            f"a width choice — on any mono playback (club sub, phone speaker, most Bluetooth "
            f"speakers) the mix largely disappears."
            + _moment_span(moments)
        )
        hits.append(_Hit(
            Finding(
                id="phase.polarity_inverted",
                dimension="phase",
                title="Right channel is polarity-inverted",
                severity=_severity(ratio),
                confidence=0.96,   # a correlation coefficient, directly computed
                detail=detail,
                evidence=evidence,
                moments=moments,
            ),
            ratio,
        ))
        ctx.tags.add("polarity")
        return hits

    # -- correlation below what the genre wants ------------------------------
    if corr_miss < 0.0 or loss_miss < 0.0:
        ratio = max(_ratio(corr_miss, 0.35), _ratio(loss_miss, 2.0))
        driver = "correlation" if _ratio(corr_miss, 0.35) >= _ratio(loss_miss, 2.0) else "fold-down"
        detail = (
            f"Inter-channel correlation is {_num(corr, 2)}, under the {_num(corr_min, 2)} floor "
            f"{ctx.profile.label} mixes hold, and folding to mono costs {_num(loss, 1)} dB "
            f"(decorrelation alone costs 3.0). The weakest band is "
            f"{worst_band.replace('_', ' ') or 'n/a'} at {_num(worst_band_corr, 2)}. "
            f"Anything played back in mono loses that much of this mix."
            + _moment_span(moments)
        )
        hits.append(_Hit(
            Finding(
                id="phase.mono_incompatible",
                dimension="phase",
                title=f"Mono fold-down loses level ({driver})",
                severity=_severity(ratio),
                confidence=0.93,
                detail=detail,
                evidence=_shared_evidence(),
                moments=moments,
            ),
            ratio,
        ))
        ctx.tags.add("phase")
        return hits

    # -- one band canceling under an otherwise healthy broadband figure -----
    # This is the case the headline number cannot catch: the sub is a small
    # share of the energy, so it can be smeared across the field while the
    # broadband correlation still reads +0.8.
    if worst_loss_band and targets.range_miss(worst_loss, (-6.0, 60.0)) < 0.0:
        ratio = _ratio(targets.range_miss(worst_loss, (-6.0, 60.0)), 2.0)
        detail = (
            f"Broadband correlation is fine at {_num(corr, 2)}, but the "
            f"{worst_loss_band.replace('_', ' ')} band loses {_num(worst_loss, 1)} dB when summed "
            f"to mono and correlates at {_num(_fin((ctx.m.stereo.band_correlation or {}).get(worst_loss_band, 1.0)), 2)}. "
            f"A single band canceling under a healthy headline figure is what makes a mix "
            f"sound complete on headphones and hollow on a mono system."
        )
        hits.append(_Hit(
            Finding(
                id="phase.band_cancellation",
                dimension="phase",
                title=f"{worst_loss_band.replace('_', ' ').title()} cancels on fold-down",
                severity=_severity(ratio),
                confidence=0.88,
                detail=detail,
                evidence=_shared_evidence(),
                moments=moments,
            ),
            ratio,
        ))
        ctx.tags.add("phase")

    return hits


# ---------------------------------------------------------------------------
# 3. Loudness
# ---------------------------------------------------------------------------


def _detect_loudness(ctx: _Ctx) -> List[_Hit]:
    """Integrated level against where this genre's releases actually sit.

    The quiet side deliberately does not mirror the loud side. A master that is
    under the window but still has headroom is not a defective mix — it is a
    fader move away from the window, and every platform leaves quiet tracks
    alone. So "too quiet" only fires when the track *cannot* reach the window
    with clean gain: `integrated + (ceiling - true_peak)` still under it. That
    is the case where getting to level costs limiting, which is a real problem.
    """
    if ctx.no_programme or not ctx.has("loudness"):
        return []

    lo = ctx.m.loudness
    window = ctx.profile.integrated_lufs
    integrated = _fin(lo.integrated_lufs, -70.0)
    true_peak = _fin(lo.true_peak_dbtp, -120.0)
    miss = targets.range_miss(integrated, window)
    if miss == 0.0:
        return []

    spotify_delta = integrated - (-14.0)
    headroom = TRUE_PEAK_CEILING_DBTP - true_peak
    attainable = integrated + headroom

    evidence = [
        _ev("Integrated loudness", integrated, "LUFS", target_range=window,
            verdict=_verdict(integrated, window, 1.5),
            detail=f"{ctx.profile.label} releases sit at {_win(window, 1, ' LUFS')}."),
        _ev("True peak", true_peak, "dBTP", target=TRUE_PEAK_CEILING_DBTP,
            verdict=_verdict(true_peak, (-120.0, TRUE_PEAK_CEILING_DBTP), 1.0)),
        _ev("Clean gain available", headroom, "dB", target=None, verdict="good",
            detail="Level that can be added before hitting -1.0 dBTP, with no limiting."),
        _ev("Loudness reachable with clean gain", attainable, "LUFS",
            target_range=window, verdict=_verdict(attainable, window, 1.5)),
        _ev("Delta vs Spotify reference", spotify_delta, "LU", target=0.0,
            verdict="problem" if spotify_delta > 1.0 else "good",
            detail="Spotify, YouTube and Tidal normalize to -14 LUFS."),
        _ev("Peak to loudness ratio", _fin(lo.plr_db), "dB",
            target_range=ctx.profile.psr_p10_db,
            verdict=_verdict(_fin(lo.plr_db), ctx.profile.psr_p10_db, 1.5)),
    ]

    if miss > 0.0:
        ratio = _ratio(miss, 1.5)
        detail = (
            f"This runs {_num(miss, 2)} LU louder than {ctx.ref}: integrated loudness is "
            f"{_num(integrated, 2)} LUFS against the {_win(window, 1, ' LUFS')} window "
            f"{ctx.profile.label} masters sit in. What that costs is specific — Spotify, "
            f"YouTube and Tidal all normalize to -14 LUFS, so the extra "
            f"{_num(spotify_delta, 1)} LU is turned straight back down on playback while "
            f"whatever was done to reach it (PLR is {_num(lo.plr_db, 1)} dB, loudness range "
            f"{_num(lo.loudness_range_lu, 1)} LU) stays in the file. What it buys is the "
            f"level on anything that does not normalize: a club rig, a car, a download."
        )
        return [_Hit(
            Finding(
                id="loudness.too_loud",
                dimension="loudness",
                title=f"Runs louder than {ctx.ref}",
                severity=_severity(ratio),
                confidence=ctx.trust(0.96, "loudness"),
                detail=detail,
                evidence=evidence,
            ),
            ratio,
        )]

    # Quiet side: only a problem if clean gain cannot get there.
    attain_miss = targets.range_miss(attainable, window)
    if attain_miss >= 0.0:
        return []

    ratio = _ratio(attain_miss, 2.0)
    detail = (
        f"This sits under {ctx.ref} and cannot gain up to it cleanly. Integrated loudness is "
        f"{_num(integrated, 2)} LUFS against a {_win(window, 1, ' LUFS')} window for "
        f"{ctx.profile.label}, and there is only {_num(headroom, 1)} dB of clean gain left "
        f"before -1.0 dBTP — turning it up as far as the peaks allow still lands at "
        f"{_num(attainable, 2)} LUFS, {_num(abs(attain_miss), 2)} LU short. Being quiet is "
        f"free (every platform leaves quiet tracks alone); what this costs is the choice: "
        f"reaching the window from here means limiting, so the gain staging in the mix is "
        f"what decides how much of it you pay."
    )
    return [_Hit(
        Finding(
            id="loudness.cannot_reach_level",
            dimension="loudness",
            title=f"Sits under {ctx.ref} with no clean gain left",
            severity=_severity(ratio),
            confidence=ctx.trust(0.92, "loudness"),
            detail=detail,
            evidence=evidence,
        ),
        ratio,
    )]


# ---------------------------------------------------------------------------
# 4. Limiter behavior
# ---------------------------------------------------------------------------


def _distortion_ceiling(profile: targets.GenreProfile) -> float:
    """How much limiter-generated harmonic content this genre absorbs.

    `targets.py` has no window for `distortion_index`, and rather than invent a
    second table this is derived from the one figure that already encodes how
    hard the genre drives its ceiling: the low edge of the PSR window. A genre
    that accepts 4 dB of short-term peak-to-loudness is a genre mastered into
    the limiter; one that expects 12 dB is not, and a limiter throwing harmonics
    there is a mistake. The mapping puts trap and EDM at 0.16, pop at 0.13, rock
    at 0.12, folk at 0.09 and classical at 0.05.
    """
    psr_low = _fin(profile.psr_p10_db[0], 6.0)
    return _clamp(0.22 - 0.015 * psr_low, 0.05, 0.18)


def _detect_limiter(ctx: _Ctx) -> List[_Hit]:
    """How the mix reaches its ceiling, as distinct from whether it goes over it.

    `clipping` owns the digital fact (samples pinned, true peak over spec).
    This owns the consequence: short-term peak-to-loudness crushed flat, and a
    limiter generating harmonics instead of holding a ceiling. Both can be true
    of the same file and they have different fixes — one is the master fader,
    the other is the limiter's input drive.
    """
    if ctx.no_programme or not ctx.has("limiter"):
        return []

    lo = ctx.m.loudness
    c = ctx.m.clipping
    d = ctx.m.dynamics

    psr = _fin(lo.psr_p10_db, 0.0)
    psr_window = ctx.profile.psr_p10_db
    psr_miss = min(targets.range_miss(psr, psr_window), 0.0)   # only the low side

    distortion = _fin(c.distortion_index, 0.0)
    dist_max = _distortion_ceiling(ctx.profile)
    dist_window = (0.0, dist_max)
    dist_miss = targets.range_miss(distortion, dist_window)

    clip_pct = _fin(c.clip_percentage, 0.0)
    runs = int(_fin(c.flat_run_count, 0))
    flat_drive = _ratio(clip_pct, 0.03) if runs >= 3 else 0.0

    ratio = min(max(_ratio(psr_miss, 1.5), _ratio(dist_miss, 0.06), flat_drive), 6.0)
    if ratio <= 0.0:
        return []

    gr = _fin(d.gain_reduction_estimate_db, 0.0)
    evidence = [
        _ev("Short-term PSR (10th pct)", psr, "dB", target_range=psr_window,
            verdict=_verdict(psr, psr_window, 1.5),
            detail=f"{ctx.profile.label} holds {_win(psr_window, 1, ' dB')} of peak over "
                   f"short-term loudness."),
        _ev("Median PSR", _fin(lo.psr_median_db), "dB", target_range=psr_window,
            verdict=_verdict(_fin(lo.psr_median_db), psr_window, 1.5)),
        _ev("Distortion index", distortion, "", target_range=dist_window,
            verdict=_verdict(distortion, dist_window, 0.06),
            detail="High-frequency products in ceiling-pinned frames, referenced to "
                   "this track's own clean frames."),
        _ev("Flat-topped runs at the ceiling", runs, "runs", target=0.0,
            verdict="problem" if runs >= 3 else "good",
            detail=f"{clip_pct:.4f}% of samples pinned; longest run "
                   f"{int(_fin(c.longest_flat_run))} samples."),
        _ev("Estimated gain reduction", gr, "dB", target=None,
            verdict="watch" if gr >= 3.0 else "good",
            detail="Inferred from crest collapse and peak pinning in 50 ms windows — "
                   "an estimate, not a reading off the limiter's meter."),
        _ev("Peak to loudness ratio", _fin(lo.plr_db), "dB", target_range=psr_window,
            verdict=_verdict(_fin(lo.plr_db), psr_window, 1.5)),
    ]

    parts: List[str] = []
    if psr_miss < 0.0:
        parts.append(
            f"short-term peak-to-loudness bottoms out at {_num(psr, 2)} dB against a "
            f"{_win(psr_window, 1, ' dB')} window for {ctx.profile.label} "
            f"({_num(abs(psr_miss), 2)} dB under)"
        )
    if dist_miss > 0.0:
        parts.append(
            f"the distortion index is {distortion:.3f} against {dist_max:.2f} for this genre"
        )
    if flat_drive > 0.0:
        parts.append(
            f"{runs} runs of samples are pinned flat at the ceiling ({clip_pct:.4f}% of the "
            f"file, longest {int(_fin(c.longest_flat_run))} samples)"
        )
    if not parts:
        parts.append(f"short-term peak-to-loudness sits at {_num(psr, 2)} dB")

    detail = (
        "The limiter is being driven past holding a ceiling and into generating one: "
        + "; ".join(parts)
        + f". Estimated gain reduction is {_num(gr, 1)} dB and micro-dynamics are down to "
        f"{_num(d.micro_dynamics_db, 1)} dB."
        + ctx.advice(
            " Back the input drive off and let the ceiling do less work — the level lost "
            "is recoverable, the harmonics are not."
        )
    )

    # Confidence follows the weakest link in the case being made. PSR is
    # measured directly; the distortion index and the gain-reduction figure are
    # inferences from the mixed file.
    confidence = 0.88 if psr_miss < 0.0 else 0.72

    ctx.tags.add("limiter")
    return [_Hit(
        Finding(
            id="limiter.over_driven",
            dimension="limiter",
            title="Limiter driven past its ceiling",
            severity=_severity(ratio),
            confidence=ctx.trust(confidence, "limiter"),
            detail=detail,
            evidence=evidence,
            moments=_moments(c.events, 4),
        ),
        ratio,
    )]


# ---------------------------------------------------------------------------
# 5. Arrangement: what the record does across its own length
#
# Runs before dynamic range on purpose. Both can be looking at the same
# flatness, and this one is the more specific description of it — it names the
# section, the timestamp and the lift in LU instead of reporting a loudness
# range — so it speaks first and `_detect_dynamic_range` stands down on its LRA
# arm afterwards.
# ---------------------------------------------------------------------------


def _detect_arrangement(ctx: _Ctx) -> List[_Hit]:
    """The chorus that does not lift, and the section whose bottom drops out.

    Neither of these moves a whole-file statistic enough to see. A hook that
    arrives at verse level leaves integrated loudness untouched; a drop that
    loses its sub while everything else stays loud leaves the long-term
    spectrum untouched. Both are obvious the moment somebody plays the record
    end to end, which is what `sections.py` measures and this reads.

    Both are genre-gated, and the gates are derived rather than invented — see
    `_expects_arrangement_lift` and `_low_swing_ceiling`. Ambient and classical
    are not told their chorus should be 6 dB louder.

    Both also read the section *labels*, not just the numbers, because which
    part of the record a measurement lands on decides whether it is a fault at
    all. An intro with the bottom pulled back and a chorus with the bottom
    pulled back produce the same swing in dB and are not the same event: the
    first is how arrangements work and the second is the record failing to
    arrive. See `_section_role`.
    """
    sa = ctx.sections
    if ctx.no_programme or not sa.available:
        return []

    sections = list(sa.sections or [])
    n = len(sections)
    if n < _MIN_SECTIONS_FOR_FORM or ctx.duration < _MIN_FORM_SEC:
        # One span, or a track too short to have parts. `sections.py` already
        # says so in its own notes; inventing a verdict on top would be worse
        # than the silence.
        return []

    hits: List[_Hit] = []
    lift = _fin(sa.peak_lift_db, 0.0)
    spread = _fin(sa.loudness_spread_lu, 0.0)
    loudest = next((s for s in sections if s.is_loudest), sections[0])
    quietest = next((s for s in sections if s.is_quietest), sections[-1])
    lra = _fin(ctx.m.loudness.loudness_range_lu, 0.0)

    # -- the loudest section does not arrive ---------------------------------
    lift_window = (_CHORUS_LIFT_MIN_LU, 60.0)
    lift_miss = targets.range_miss(lift, lift_window)
    if _expects_arrangement_lift(ctx.profile) and lift_miss < 0.0:
        ratio = _ratio(lift_miss, _CHORUS_LIFT_SCALE)
        part = loudest.label if not loudest.label.startswith("section ") else "loudest section"
        # Which part of the record turned out to be the loudest changes what
        # this finding is. A chorus that arrives at verse level is the complaint
        # in its pure form. A record whose peak is its *intro* is a stranger
        # observation and worth stating as one. And where the segmenter never
        # found anything hook-shaped, the flatness is measured just as directly
        # but the word "chorus" has nothing to attach to.
        loudest_role = _section_role(loudest.label)
        has_payoff = any(_section_role(s.label) == "payoff" for s in sections)
        named = (f"the {loudest.label}" if not loudest.label.startswith("section ")
                 else f"the section at {_clock(loudest.t_start)}")
        if loudest_role == "payoff":
            opening = (
                f"The {loudest.label} is there and it arrives at the same size as everything "
                f"around it. "
            )
        elif loudest_role == "bookend":
            opening = (
                f"The loudest thing on this record is its {loudest.label}, and nothing after "
                f"it goes above that — which is unusual enough to be worth saying on its own. "
            )
        else:
            opening = (
                f"The sections arrive at more even levels than {ctx.ref} does"
                + (", and nothing in this arrangement reads as a chorus or a drop, so this is "
                   "a measurement of the levels rather than a claim about a hook. "
                   if not has_payoff else f", and the loudest of them is {named}. ")
            )
        detail = (
            opening
            + f"Across {n} measured sections the loudest one ({part} at "
            f"{_clock(loudest.t_start)}) sits "
            f"{_num(lift, 1)} LU above the median section, against the "
            f"{_num(_CHORUS_LIFT_MIN_LU, 1)} LU a {ctx.profile.label} arrangement usually "
            f"puts there — most records run 2-4 LU. Loudest to quietest across the whole "
            f"record is {_num(spread, 1)} LU and EBU loudness range is {_num(lra, 1)} LU. "
            f"An even record buys relentlessness, which is the right call on plenty of "
            f"material; it costs the moment where the hook reads as the payoff."
            + ctx.advice(
                " If that moment is wanted, it comes from faders and arrangement — no "
                "amount of limiting on the master puts a lift back."
            )
        )
        hits.append(_Hit(
            Finding(
                id="dynamic_range.no_section_lift",
                dimension="dynamic_range",
                title=f"Sections arrive at one size against {ctx.ref}",
                # Every LUFS figure here is a direct BS.1770 measurement; the
                # uncertainty is in where the boundaries were placed, which is
                # a segmenter's judgment.
                confidence=ctx.trust(0.82, "arrangement"),
                severity=_severity(ratio),
                detail=detail,
                evidence=[
                    _ev("Loudest section vs median", lift, "LU", target_range=lift_window,
                        verdict=_verdict(lift, lift_window, _CHORUS_LIFT_SCALE),
                        detail=f"{part} at {_clock(loudest.t_start)}, "
                               f"{_num(loudest.integrated_lufs, 1)} LUFS."),
                    _ev("Loudest to quietest section", spread, "LU", target=None,
                        verdict="watch" if spread < 2.0 else "good",
                        detail=f"{loudest.label} at {_clock(loudest.t_start)} against "
                               f"{quietest.label} at {_clock(quietest.t_start)}."),
                    _ev("Loudness range", lra, "LU",
                        target_range=ctx.profile.loudness_range_lu,
                        verdict=_verdict(lra, ctx.profile.loudness_range_lu, 1.5),
                        detail=f"EBU 3342. {ctx.profile.label} runs "
                               f"{_win(ctx.profile.loudness_range_lu, 1, ' LU')}."),
                    _ev("Sections measured", float(n), "sections", verdict="good"),
                ],
                moments=[
                    Moment(
                        t_start=round(_fin(loudest.t_start), 3),
                        t_end=round(_fin(loudest.t_end), 3),
                        intensity=round(_clamp(1.0 - lift / _CHORUS_LIFT_MIN_LU, 0.0, 1.0), 3),
                        value=round(lift, 2),
                        label=f"{part}: +{_num(lift, 1)} LU",
                    )
                ],
            ),
            ratio,
        ))
        ctx.tags.add("section_lift")

    # -- one section's bottom end falls out ----------------------------------
    #
    # The swing is measured twice, and the order matters. First across the
    # record's working body with the intro and outro set aside, because that is
    # where a missing bottom end is a fault; only if the body holds together is
    # the whole core looked at, which is the pass a bookend can reach — and it
    # has to be twice as far out to get there. That ordering is what stops a
    # deliberately empty intro either inventing the finding or masking a chorus
    # with the same reading for a worse reason.
    low_rel, lufs, core = _section_low_shares(sections)
    swing_max = _low_swing_ceiling(ctx.profile)
    core_idx = [int(i) for i in np.flatnonzero(core)]
    body_idx = [i for i in core_idx if _section_role(sections[i].label) != "bookend"]

    case = _low_swing_case(sections, low_rel, body_idx, swing_max, ctx.duration)
    if case is None:
        case = _low_swing_case(sections, low_rel, core_idx, swing_max, ctx.duration)

    if case is not None:
        weak_i, strong_i, swing, ceiling, role = case
        swing_window = (0.0, ceiling)
        swing_miss = targets.range_miss(swing, swing_window)
        ratio = min(_ratio(swing_miss, _LOW_SWING_SCALE_DB), 6.0)
        weakest, strongest = sections[weak_i], sections[strong_i]
        weak_share = _fin(low_rel[weak_i], 0.0)
        strong_share = _fin(low_rel[strong_i], 0.0)
        weak_pct = _section_share(weakest, ctx.duration) * 100.0
        measured = (
            f"Sub and low bass carry {_num(weak_share, 1)} dB of {weakest.label}'s own energy "
            f"at {_clock(weakest.t_start)} against {_num(strong_share, 1)} dB in "
            f"{strongest.label} at {_clock(strongest.t_start)} — a {_num(swing, 1)} dB swing. "
            f"It is measured as each section's *share* of its own level, so this is not the "
            f"quiet parts being quiet: {weakest.label} is within {_num(_LOW_CORE_LU, 0)} LU of "
            f"the loudest section and still has little under it."
        )

        if role == "bookend":
            # The reported bug, and the shape of the fix: say the ordinary
            # explanation first, because it is almost always the true one.
            ratio = min(ratio, _ARRANGEMENT_ASIDE_MAX_RATIO)
            title = f"{weakest.label.capitalize()} runs emptier than most {weakest.label}s"
            detail = (
                f"An {weakest.label} with the bottom pulled back is how arrangements work, and "
                f"this is only raised because of how far it goes and how long it lasts. "
                + measured
                + f" The bar for an intro or an outro is {_num(ceiling, 1)} dB — already "
                f"double the {_num(swing_max, 1)} dB the rest of the record is held to, "
                f"precisely because a thin {weakest.label} is normal — and {weakest.label} "
                f"runs {_num(weak_pct, 0)}% of the track, long enough to sit in. Held to a "
                f"note rather than a fault: if the empty {weakest.label} is the intention, it "
                f"is working."
            )
        elif role == "payoff":
            title = f"Low end steps back in {weakest.label}"
            detail = (
                "The section this record is built to arrive at is the one carrying the least "
                "bottom end. "
                + measured
                + f" A {ctx.profile.label} arrangement holds this to {_num(swing_max, 1)} dB "
                f"across the {len(core_idx)} sections carrying the record. Taking the bottom "
                f"out of the section everything else leads to buys contrast on the way in; it "
                f"costs the moment the record exists for, and on a full-range system the "
                f"{weakest.label} reads smaller than the part before it."
            )
        else:
            title = f"Low end steps back in {weakest.label}"
            detail = (
                f"The bottom end moves between sections further than {ctx.ref} does. "
                + measured
                + f" {ctx.profile.label} typically holds {_num(swing_max, 1)} dB across the "
                f"sections carrying a record; the bar here is {_num(ceiling, 1)} dB, because "
                f"{weakest.label} is neither the intro — where a thin bottom is ordinary — nor "
                f"the hook, where it is a fault, and that is the case the measurement cannot "
                f"read on its own. Dropping the bottom out buys contrast and makes the return "
                f"hit harder; it costs a section that reads thin on a full-range system."
            )

        hits.append(_Hit(
            Finding(
                id="low_end.section_collapse",
                dimension="low_end",
                title=title,
                severity=_severity(ratio),
                confidence=ctx.trust(0.84, "arrangement"),
                detail=detail,
                evidence=[
                    _ev("Low-end swing across sections", swing, "dB",
                        target_range=swing_window,
                        verdict=_verdict(swing, swing_window, _LOW_SWING_SCALE_DB),
                        detail=f"Sub+low-bass share of each section's own total, over the "
                               f"sections within {_num(_LOW_CORE_LU, 0)} LU of the loudest. The "
                               f"ceiling is {ctx.profile.label}'s {_num(swing_max, 1)} dB scaled "
                               f"for a weakest section that is {'an' if role == 'bookend' else 'a'} "
                               f"{role} ({weakest.label})."),
                    _ev(f"Low-end share in {weakest.label}", weak_share, "dB",
                        target=strong_share, verdict="problem",
                        detail=f"{_clock(weakest.t_start)}-{_clock(weakest.t_end)}, "
                               f"{_num(weakest.integrated_lufs, 1)} LUFS, "
                               f"{_num(weak_pct, 0)}% of the track."),
                    _ev(f"Low-end share in {strongest.label}", strong_share, "dB",
                        target=None, verdict="good",
                        detail=f"{_clock(strongest.t_start)}-{_clock(strongest.t_end)}, "
                               f"{_num(strongest.integrated_lufs, 1)} LUFS."),
                    _ev("Sections carrying the record", float(len(core_idx)), "sections",
                        verdict="good",
                        detail=f"{len(body_idx)} of them outside the intro and outro."),
                ],
                band_hz=(20.0, 120.0),
                moments=[
                    Moment(
                        t_start=round(_fin(weakest.t_start), 3),
                        t_end=round(_fin(weakest.t_end), 3),
                        intensity=round(_clamp(swing / 12.0, 0.0, 1.0), 3),
                        value=round(weak_share, 2),
                        label=f"{weakest.label}: no low end",
                    )
                ],
            ),
            ratio,
        ))
        ctx.tags.add("section_low_end")

    return hits


# ---------------------------------------------------------------------------
# 6. Dynamic range (macro: whole-file and section-to-section)
# ---------------------------------------------------------------------------


def _detect_dynamic_range(ctx: _Ctx) -> List[_Hit]:
    """Crest, TT-DR and loudness range against the genre's windows.

    Loudness range is only consulted on material long enough to have an
    arrangement — on a 20 s loop it measures the loop, not the record.
    """
    if ctx.no_programme or not ctx.has("dynamics"):
        return []

    d = ctx.m.dynamics
    lo = ctx.m.loudness

    crest = _fin(d.crest_factor_db, 0.0)
    crest_window = ctx.profile.crest_factor_db
    crest_miss = targets.range_miss(crest, crest_window)

    lra = _fin(lo.loudness_range_lu, 0.0)
    lra_window = ctx.profile.loudness_range_lu
    lra_miss = targets.range_miss(lra, lra_window) if ctx.has("arrangement") else 0.0

    # The arrangement detector has already reported this flatness, and it did it
    # better: it names the section that fails to lift, when it happens, and by
    # how much. A loudness range is the same observation with the timestamps
    # thrown away, so it is dropped rather than said twice (rule 5). Crest is a
    # different measurement and keeps its arm.
    lra_owned = "section_lift" in ctx.tags
    if lra_owned:
        lra_miss = 0.0

    evidence = [
        _ev("Crest factor", crest, "dB", target_range=crest_window,
            verdict=_verdict(crest, crest_window, 1.5),
            detail=f"{ctx.profile.label} runs {_win(crest_window, 1, ' dB')}."),
        _ev("TT-DR value", _fin(d.dr_value), "dB", target_range=crest_window,
            verdict=_verdict(_fin(d.dr_value), crest_window, 2.0),
            detail="Peak minus RMS over the loudest 20% of 3 s blocks."),
        _ev("Loudness range", lra, "LU", target_range=lra_window,
            verdict=_verdict(lra, lra_window, 1.5) if ctx.has("arrangement") else "good",
            detail=(
                "EBU 3342 range, reported under 'The loudest section does not lift', "
                "which measures the same flatness section by section."
                if lra_owned else
                "EBU 3342 range." if ctx.has("arrangement")
                else f"Not scored: {_num(ctx.duration, 0)} s of audio is too short for a "
                     f"section-to-section range to mean anything.")),
        _ev("Macro dynamics", _fin(d.macro_dynamics_lu), "LU", target_range=lra_window,
            verdict="good"),
        _ev("Integrated loudness", _fin(lo.integrated_lufs), "LUFS",
            target_range=ctx.profile.integrated_lufs,
            verdict=_verdict(_fin(lo.integrated_lufs), ctx.profile.integrated_lufs, 1.5)),
    ]

    # -- squashed ------------------------------------------------------------
    if crest_miss < 0.0 or lra_miss < 0.0:
        ratio = max(_ratio(crest_miss, 1.5), _ratio(lra_miss, 1.5))
        parts = []
        if crest_miss < 0.0:
            parts.append(
                f"crest factor is {_num(crest, 1)} dB against {_win(crest_window, 1, ' dB')} "
                f"for {ctx.profile.label} and TT-DR reads {_num(d.dr_value, 1)}"
            )
        if lra_miss < 0.0:
            parts.append(
                f"loudness range is {_num(lra, 1)} LU against {_win(lra_window, 1, ' LU')}"
            )
        detail = (
            f"This moves less than {ctx.ref} does: "
            + "; ".join(parts)
            + f". At {_num(lo.integrated_lufs, 1)} LUFS integrated, the gap between the "
            f"quietest and loudest moment is narrow enough that the record reads as one "
            f"continuous level. That buys density and a mix that never drops away on a "
            f"phone; it costs the contrast that makes a section feel like it arrived."
        )
        return [_Hit(
            Finding(
                id="dynamic_range.squashed",
                dimension="dynamic_range",
                title=f"Flatter than {ctx.ref}",
                severity=_severity(ratio),
                confidence=ctx.trust(0.90, "dynamics"),
                detail=detail,
                evidence=evidence,
            ),
            ratio,
        )]

    # -- more dynamic than the genre's masters -------------------------------
    # Only reported when the level corroborates it: a wide crest at genre level
    # is a well-made record, a wide crest 6 LU under it is an unmastered mix.
    # A 1.5 dB grace keeps borderline cases quiet.
    level_miss = targets.range_miss(_fin(lo.integrated_lufs, -70.0), ctx.profile.integrated_lufs)
    if crest_miss > 1.5 and level_miss < 0.0:
        ratio = min(_ratio(crest_miss - 1.5, 2.0), 2.4)   # never critical: not damage
        detail = (
            f"This carries more dynamic range than {ctx.ref} does, at a lower level. Crest "
            f"factor is {_num(crest, 1)} dB and TT-DR {_num(d.dr_value, 1)}, above the "
            f"{_win(crest_window, 1, ' dB')} {ctx.profile.label} masters hold, and integrated "
            f"loudness is {_num(lo.integrated_lufs, 1)} LUFS — {_num(abs(level_miss), 1)} LU "
            f"under the {_win(ctx.profile.integrated_lufs, 1, ' LUFS')} window. Nothing here "
            f"is damaged: that combination is what an unmastered mix measures like, and the "
            f"headroom is worth something to whoever masters it. What it costs today is "
            f"comparison — played next to anything else in the genre it will read quiet and "
            f"soft."
        )
        return [_Hit(
            Finding(
                id="dynamic_range.unmastered",
                dimension="dynamic_range",
                title=f"Wider dynamics and lower level than {ctx.ref}",
                severity=_severity(ratio),
                confidence=ctx.trust(0.85, "dynamics"),
                detail=detail,
                evidence=evidence,
            ),
            ratio,
        )]

    return []


# ---------------------------------------------------------------------------
# 7. Compression (micro: inside the hit)
# ---------------------------------------------------------------------------


def _detect_compression(ctx: _Ctx) -> List[_Hit]:
    """What is left of the transient inside each hit, and whether the mix breathes.

    Pumping is the least trustworthy number in the whole analysis: percussive
    material modulates its own envelope at the beat rate with no compressor
    anywhere near it, which is why the DSP layer's own docstring warns about it.
    So it is never reported on its own — it needs the gain-reduction estimate
    and the micro-dynamics figure to corroborate, and even then its confidence
    says plainly that it is an inference.
    """
    if ctx.no_programme or not ctx.has("dynamics"):
        return []

    d = ctx.m.dynamics
    hits: List[_Hit] = []

    micro = _fin(d.micro_dynamics_db, 0.0)
    micro_window = ctx.profile.micro_dynamics_db
    micro_miss = min(targets.range_miss(micro, micro_window), 0.0)
    gr = _fin(d.gain_reduction_estimate_db, 0.0)
    pumping = _fin(d.pumping_index, 0.0)
    rate = _fin(d.pumping_rate_hz, 0.0)

    evidence = [
        _ev("Micro dynamics", micro, "dB", target_range=micro_window,
            verdict=_verdict(micro, micro_window, 1.5),
            detail=f"Crest inside a 50 ms window. {ctx.profile.label} keeps "
                   f"{_win(micro_window, 1, ' dB')}."),
        _ev("Estimated gain reduction", gr, "dB", target=None,
            verdict="watch" if gr >= 3.0 else "good",
            detail="Inferred from crest collapse in loud sections vs moderate ones."),
        _ev("Pumping index", pumping, "", target_range=(0.0, 0.55),
            verdict="watch" if pumping > 0.55 else "good",
            detail=f"Tempo-locked envelope modulation at {_num(rate, 2)} Hz. Percussive "
                   f"material modulates its own envelope at the beat rate, so this alone "
                   f"is not evidence of a compressor."),
        _ev("Crest factor", _fin(d.crest_factor_db), "dB",
            target_range=ctx.profile.crest_factor_db,
            verdict=_verdict(_fin(d.crest_factor_db), ctx.profile.crest_factor_db, 1.5)),
        _ev("Transient to sustain", _fin(ctx.m.transients.transient_to_sustain_db), "dB",
            target=None, verdict="good"),
    ]

    if micro_miss < 0.0:
        ratio = _ratio(micro_miss, 1.5)
        detail = (
            f"Inside each hit this is flatter than {ctx.ref}: there is {_num(micro, 1)} dB "
            f"between peak and RMS in a 50 ms window against the "
            f"{_win(micro_window, 1, ' dB')} {ctx.profile.label} keeps, and the crest collapse "
            f"between loud and moderate sections puts estimated gain reduction at "
            f"{_num(gr, 1)} dB. Macro level movement can survive that — crest factor is still "
            f"{_num(d.crest_factor_db, 1)} dB — while every individual hit has had its attack "
            f"taken off. That buys glue and consistency; it costs the impact that makes the "
            f"same arrangement feel like it is being played rather than played back."
        )
        hits.append(_Hit(
            Finding(
                id="compression.micro_dynamics_lost",
                dimension="compression",
                title=f"Hits sit flatter inside than {ctx.ref}",
                severity=_severity(ratio),
                confidence=ctx.trust(0.82, "dynamics"),
                detail=detail,
                evidence=evidence,
                moments=_moments(ctx.m.transients.weak_moments, 4),
            ),
            ratio,
        ))

    # Pumping: three independent signals must agree before this is claimed.
    pump_window = (0.0, 0.60)
    pump_miss = targets.range_miss(pumping, pump_window)
    if pump_miss > 0.0 and gr >= 2.5 and micro <= micro_window[0] + 1.0:
        ratio = min(_ratio(pump_miss, 0.15), 3.0)
        detail = (
            f"The whole mix breathes on the beat more than {ctx.ref} does. The broadband "
            f"envelope modulates at {_num(rate, 2)} Hz ({_num(rate * 60.0, 0)} BPM, against a "
            f"detected tempo of {_num(ctx.m.transients.estimated_tempo, 0)}) with a pumping "
            f"index of {pumping:.2f}, and two other numbers agree: estimated gain reduction "
            f"is {_num(gr, 1)} dB and micro-dynamics are {_num(micro, 1)} dB against "
            f"{_win(micro_window, 1, ' dB')}. Audible pumping is a production choice in "
            f"plenty of dance and hip-hop — it buys groove and forward motion. It costs "
            f"independence: every element moves together instead of separately."
        )
        hits.append(_Hit(
            Finding(
                id="compression.pumping",
                dimension="compression",
                title=f"Breathes on the beat harder than {ctx.ref}",
                severity=_severity(ratio),
                # Inferred from envelope periodicity, and the DSP layer cannot
                # separate a compressor from a groove. Said plainly.
                confidence=ctx.trust(0.45, "dynamics"),
                detail=detail,
                evidence=evidence,
            ),
            ratio,
        ))

    return hits


def _detect_stem_compression(ctx: _Ctx) -> List[_Hit]:
    """Compression on one source, which a two-track cannot see at all.

    Every compression figure above this is measured on the sum. That answers
    "has the master been crushed" and cannot answer "which element has been
    crushed", because a flattened drum bus under an untouched vocal and a
    flattened vocal over untouched drums produce the same crest factor on the
    master. With stems the question separates.

    Two cases, and the second is deliberately harder to trigger:

    * **Drums.** Crest factor against the genre's window for the *whole
      master*. That is a strong test on purpose: the drums are the most
      transient thing on the record, so a drum stem carrying less peak-to-RMS
      than the finished master is supposed to carry has been flattened. Trap's
      floor is 5.5 dB, pop's 8.0, classical's 14.0 — the genre does the work.
    * **Vocals.** Micro-dynamics against the genre's window, and only with the
      gain-reduction estimate agreeing. A vocal with little crest may just be
      an even performance, and calling that over-compression would be a taste
      judgment dressed up as a measurement.

    One finding, worst first: "the drums and the vocal are both squashed" is
    one session's work and one entry in the list.
    """
    if ctx.no_programme or not ctx.has_stems or not ctx.has("stem"):
        return []

    crest_window = ctx.profile.crest_factor_db
    micro_window = ctx.profile.micro_dynamics_db

    candidates: List[Tuple[float, StemMeasurement, str, float, Tuple[float, float], float]] = []
    for kind in ("drums", "vocals"):
        stem = ctx.stem(kind)
        if stem is None:
            continue
        gr = _fin(stem.gain_reduction_estimate_db, 0.0)
        if kind == "drums":
            value, window, scale, metric = (
                _fin(stem.crest_factor_db, 0.0), crest_window, 1.5, "crest factor"
            )
        else:
            if gr < _STEM_GR_CORROBORATION_DB:
                continue
            value, window, scale, metric = (
                _fin(stem.micro_dynamics_db, 0.0), micro_window, 1.5, "micro-dynamics"
            )
        miss = min(targets.range_miss(value, window), 0.0)   # only the flat side
        if miss >= 0.0:
            continue
        candidates.append((_ratio(miss, scale), stem, metric, value, window, gr))

    if not candidates:
        return []

    ratio, stem, metric, value, window, gr = max(candidates, key=lambda c: c[0])
    ratio = min(ratio, 6.0)
    if ratio < MIN_REPORTABLE_RATIO:
        return []

    kind = str(stem.kind)
    label = {"drums": "drum", "vocals": "vocal"}.get(kind, kind)
    consequence = (
        "The hits arrive without landing, and nothing on the master bus puts an attack "
        "back once it has been compressed off the source."
        if kind == "drums" else
        "The lead sits at one level with no push behind the loud lines, which reads as "
        "distant no matter how far the fader comes up."
    )
    detail = (
        f"The separated {label} stem is flatter than {ctx.ref}: it carries "
        f"{_num(value, 1)} dB of {metric} against the {_win(window, 1, ' dB')} window "
        f"{ctx.profile.label} holds on the finished master, and its own crest collapse and "
        f"peak pinning put an estimated {_num(gr, 1)} dB of gain reduction on it. This is a "
        f"per-element reading, which is the point: the two-track's crest factor is "
        f"{_num(ctx.m.dynamics.crest_factor_db, 1)} dB and its micro-dynamics "
        f"{_num(ctx.m.dynamics.micro_dynamics_db, 1)} dB, and those cannot tell a flattened "
        f"{label} under an untouched mix from a flattened master — separating them can. "
        f"{consequence}"
        + ctx.advice(f" If that is not the sound, it is the {label} bus that decides it, "
                     f"not the master.")
    )

    return [_Hit(
        Finding(
            id=f"compression.stem_{kind}_flat",
            dimension="compression",
            title=f"The {label} stem is flatter than {ctx.ref}",
            severity=_severity(ratio),
            # Directly measured on the separated source; the uncertainty that
            # is left belongs to the separation, not to the statistic.
            confidence=ctx.trust(0.86, "stem"),
            detail=detail,
            evidence=[
                _ev(f"{label.title()} stem {metric}", value, "dB", target_range=window,
                    verdict=_verdict(value, window, 1.5),
                    detail=f"Measured on the separated {label}, over the frames it is "
                           f"sounding on."),
                _ev(f"{label.title()} stem gain reduction", gr, "dB", target=None,
                    verdict="problem" if gr >= _STEM_GR_CORROBORATION_DB else "watch",
                    detail="Crest collapse between loud and moderate 50 ms frames, plus "
                           "peak pinning, on this source alone."),
                _ev(f"{label.title()} stem crest factor",
                    _fin(stem.crest_factor_db), "dB", target_range=crest_window,
                    verdict=_verdict(_fin(stem.crest_factor_db), crest_window, 1.5)),
                _ev(f"{label.title()} stem transient punch",
                    _fin(stem.transient_punch), "",
                    target_range=(_fin(ctx.profile.punch_min, 0.35), 1.0),
                    verdict=_verdict(_fin(stem.transient_punch),
                                     (_fin(ctx.profile.punch_min, 0.35), 1.0), 0.07)),
                _ev("Whole-mix crest factor", _fin(ctx.m.dynamics.crest_factor_db), "dB",
                    target_range=crest_window,
                    verdict=_verdict(_fin(ctx.m.dynamics.crest_factor_db), crest_window, 1.5),
                    detail="For comparison: the figure the two-track can produce."),
            ],
            moments=_moments(ctx.m.transients.weak_moments, 4),
        ),
        ratio,
    )]


# ---------------------------------------------------------------------------
# 8. Low end (kick / 808 relationship)
# ---------------------------------------------------------------------------


def _detect_low_end(ctx: _Ctx) -> List[_Hit]:
    """Kick against bass, sub weight, and whether the bottom is mono.

    The kick-versus-bass finding has two sources and prefers the better one.
    Without stems everything kick-related is gated on `kick_detected`, because
    the DSP layer applies regularity, definition and band-energy gates
    precisely so a pad's filter noise does not become a drum machine, and it
    reconstructs the kick's spectrum by subtracting a between-hits spectrum
    from an at-hits one — a good estimate of one object inside a waveform that
    contains two.

    With stems there is no reconstruction: the kick is isolated inside the drum
    stem and the bass is a separate array, so the two fundamentals and the
    level between them are direct readings. Same finding, real numbers, and the
    confidence moves from 0.70 to 0.93 to say so.
    """
    if ctx.no_programme:
        return []

    le = ctx.m.low_end
    hits: List[_Hit] = []

    # -- kick and bass on the same fundamental -------------------------------
    collision = _fin(le.kick_bass_collision_db, -40.0)
    collision_window = (-40.0, _fin(ctx.profile.kick_bass_collision_max_db, -6.0))
    collision_miss = targets.range_miss(collision, collision_window)

    sk = ctx.stems
    stem_kick_hz = _fin(sk.kick_fundamental_hz, 0.0) if sk.kick_fundamental_hz else 0.0
    stem_bass_hz = _fin(sk.bass_fundamental_hz, 0.0) if sk.bass_fundamental_hz else 0.0
    kick_to_bass = None if sk.kick_to_bass_db is None else _fin(sk.kick_to_bass_db, 0.0)
    from_stems = bool(
        ctx.has_stems and ctx.has("stem") and stem_kick_hz > 0.0 and stem_bass_hz > 0.0
    )

    kick_hz = stem_kick_hz if from_stems else _fin(le.kick_fundamental_hz)
    bass_hz = stem_bass_hz if from_stems else _fin(le.bass_fundamental_hz)
    semitones = (
        abs(12.0 * math.log2(max(kick_hz, 1e-6) / max(bass_hz, 1e-6)))
        if kick_hz > 0 and bass_hz > 0 else 0.0
    )

    # The pitch distance is only a *trigger* when it was measured on two
    # separated objects. On a two-track both figures come out of the same
    # waveform, and firing on them would be firing on the reconstruction's
    # own error.
    note_window = (_SAME_NOTE_SEMITONES, 120.0)
    note_miss = targets.range_miss(semitones, note_window) if from_stems else 0.0
    gate = (bool(le.kick_detected) and ctx.has("kick")) or from_stems

    if gate and (collision_miss > 0.0 or note_miss < 0.0):
        ratio = max(_ratio(collision_miss, 2.0), _ratio(note_miss, _SEMITONE_SCALE))
        moments = _moments(le.collision_moments)
        duck = _fin(le.ducking_depth_db, 0.0)
        # The overlap figure compares spectral *shapes* over 25-250 Hz, which can
        # be high even when the two fundamentals are far apart. Only claim a
        # shared fundamental when the pitches actually say so.
        same_note = 0.0 < semitones <= 4.0 and kick_hz > 0 and bass_hz > 0
        consequence = (
            "Every kick re-triggers the note the bass is already holding, so the low end "
            "reads as one thick sound instead of a hit and a pitch."
            if same_note else
            "The two fundamentals are far enough apart, but the energy either side of them "
            "is spread across the same shape, so the kick lands inside the bass's body "
            "rather than underneath it."
        )

        if from_stems:
            level = (
                f"the kick sits {_num(abs(kick_to_bass), 1)} dB "
                f"{'over' if kick_to_bass >= 0 else 'under'} the bass over its own hit windows"
                if kick_to_bass is not None else
                "their levels could not be compared"
            )
            detail = (
                f"Separated, the kick inside the drum stem lands on {_num(kick_hz, 1)} Hz and "
                f"the bass stem's fundamental is {_num(bass_hz, 1)} Hz — {_num(semitones, 1)} "
                f"semitones apart, against the {_num(_SAME_NOTE_SEMITONES, 0)} semitones two "
                f"low-frequency sources need before the ear reads them as separate pitches — "
                f"and {level}. Their 25-250 Hz distributions overlap at {_num(collision, 1)} dB "
                f"against a {_num(collision_window[1], 1)} dB ceiling for {ctx.profile.label}. "
                f"These are two measured objects, not one waveform split by an onset gate: the "
                f"kick and the bass were separated before either was measured. Across "
                f"{le.kick_count} detected hits the sub moves {_num(duck, 1)} dB when the kick "
                f"lands"
                + (", so there is no sidechain getting the bass out of the way"
                   if not bool(le.has_sidechain) else ", which is a sidechain working")
                + f". {consequence}"
                + _moment_span(moments)
            )
        else:
            detail = (
                f"The kick's fundamental is {_num(kick_hz, 1)} Hz and the bass sits at "
                f"{_num(bass_hz, 1)} Hz — {_num(semitones, 1)} semitones apart — and their "
                f"25-250 Hz energy distributions overlap at {_num(collision, 1)} dB against a "
                f"{_num(collision_window[1], 1)} dB ceiling for {ctx.profile.label}. Across "
                f"{le.kick_count} detected hits the sub moves {_num(duck, 1)} dB when the kick "
                f"lands"
                + (", so there is no sidechain getting the bass out of the way"
                   if not bool(le.has_sidechain) else ", which is a sidechain working")
                + f". {consequence}"
                + _moment_span(moments)
            )

        evidence = [
            _ev("Kick/bass spectral overlap", collision, "dB",
                target_range=collision_window,
                verdict=_verdict(collision, collision_window, 2.0),
                detail="0 dB means identical shape in the same place."),
            _ev("Kick fundamental", kick_hz, "Hz", verdict="watch",
                detail=("Measured inside the separated drum stem."
                        if from_stems else
                        "At-onset minus between-onset spectrum of the two-track.")),
            _ev("Bass fundamental", bass_hz, "Hz", verdict="watch",
                detail=f"{_num(semitones, 1)} semitones from the kick"
                       + (", both measured on separated sources." if from_stems else ".")),
        ]
        if from_stems:
            evidence.append(
                _ev("Pitch distance", semitones, "semitones", target_range=note_window,
                    verdict=_verdict(semitones, note_window, _SEMITONE_SCALE),
                    detail=f"Under {_num(_SAME_NOTE_SEMITONES, 0)} semitones the difference "
                           f"tone lands in the same critical band as both sources.")
            )
            if kick_to_bass is not None:
                evidence.append(
                    _ev("Kick vs bass level", kick_to_bass, "dB", target=None,
                        verdict="watch",
                        detail="Drum stem's low band over its hit windows against the "
                               "bass stem — two sources, directly compared.")
                )
        evidence.extend([
            _ev("Sidechain ducking depth", duck, "dB", target=None,
                verdict="problem" if duck < 1.5 else "good",
                detail="Sub level drop at each kick hit. Inferred, not read "
                       "off a plugin."),
            _ev("Kick definition", _fin(le.kick_definition_db), "dB",
                target=None, verdict="good",
                detail="Kick transient above the surrounding low-end floor."),
            _ev("Detected kick hits", le.kick_count, "hits", verdict="good"),
        ])

        hits.append(_Hit(
            Finding(
                id="low_end.kick_bass_collision",
                dimension="low_end",
                title=("Kick and bass share a fundamental" if same_note
                       else f"Kick and bass overlap more than {ctx.ref}"),
                severity=_severity(ratio),
                # With stems these are two separated objects compared directly.
                # Without them, the kick's spectrum is at-onset minus
                # between-onset — a good estimate, and not a stem.
                confidence=(ctx.trust(0.93, "stem") if from_stems
                            else ctx.trust(0.70, "kick")),
                detail=detail,
                evidence=evidence,
                band_hz=(
                    round(min(kick_hz, bass_hz) * 0.8, 1),
                    round(max(kick_hz, bass_hz) * 1.25, 1),
                ) if kick_hz > 0 and bass_hz > 0 else None,
                moments=moments,
            ),
            ratio,
        ))
        ctx.tags.add("low_end_collision")

    # -- sub weight ----------------------------------------------------------
    sub = _fin(le.sub_energy_db, -90.0)
    sub_window = ctx.profile.sub_energy_db
    sub_miss = targets.range_miss(sub, sub_window)
    if sub_miss != 0.0 and sub > -60.0:
        ratio = _ratio(sub_miss, 3.0)
        hot = sub_miss > 0.0
        band = ctx.bands.get("sub")
        detail = (
            f"Sub energy runs {_num(abs(sub_miss), 1)} dB "
            f"{'above' if hot else 'below'} where {ctx.profile.label} masters typically sit: "
            f"energy under 60 Hz measures {_num(sub, 1)} dB relative to the whole band, "
            f"against a {_win(sub_window, 1, ' dB')} window"
            + (f", and the sub macro band reads {_num(band.deviation_db, 1)} dB "
               f"{'over' if band.deviation_db > 0 else 'under'} its target curve"
               if band else "")
            + (". Weight down there buys the size and the physical push the genre is built "
               "on; it costs limiter headroom, and none of it reproduces on anything smaller "
               "than a full-range system, so it is level the small speakers never hear."
               if hot else
               ". A light bottom octave buys headroom and translation on small speakers; it "
               "costs the size the genre usually carries, so the mix will read smaller on a "
               "system that can actually play it.")
        )
        hits.append(_Hit(
            Finding(
                id="low_end.sub_energy_hot" if hot else "low_end.sub_energy_thin",
                dimension="low_end",
                title=(f"Sub energy runs hot against {ctx.ref}" if hot
                       else f"Sub octave sits light against {ctx.ref}"),
                severity=_severity(ratio),
                confidence=0.90,   # a band energy ratio, directly measured
                detail=detail,
                evidence=[
                    _ev("Sub energy (20-60 Hz)", sub, "dB", target_range=sub_window,
                        verdict=_verdict(sub, sub_window, 3.0)),
                    _ev("Sub band vs target curve",
                        band.deviation_db if band else 0.0, "dB",
                        target=0.0,
                        target_range=(-(band.tolerance_db if band else 4.5),
                                      (band.tolerance_db if band else 4.5)),
                        verdict="problem" if band and band.miss_db else "good"),
                    _ev("Sub-25 Hz rumble", _fin(le.sub_rumble_db), "dB",
                        target_range=(-90.0, _rumble_ceiling_db(ctx.genre_key)),
                        verdict=_verdict(_fin(le.sub_rumble_db),
                                         (-90.0, _rumble_ceiling_db(ctx.genre_key)),
                                         4.0)),
                ],
                band_hz=(20.0, 60.0),
            ),
            ratio,
        ))
        ctx.tags.add("sub_energy")

    # -- the bottom is not mono ----------------------------------------------
    # Suppressed under a polarity inversion: it is the same fault, and the phase
    # finding already carries the low-end mono ratio as evidence.
    mono_ratio = _fin(le.low_end_mono_ratio, 1.0)
    mono_window = (_fin(ctx.profile.low_end_mono_min, 0.80), 1.0)
    mono_miss = targets.range_miss(mono_ratio, mono_window)
    if mono_miss < 0.0 and not ctx.is_mono and "polarity" not in ctx.tags:
        ratio = _ratio(mono_miss, 0.06)
        side_db = _fin(ctx.m.stereo.low_end_side_energy_db, -60.0)
        detail = (
            f"Only {mono_ratio * 100:.0f}% of the energy below 120 Hz is mono, against the "
            f"{mono_window[0] * 100:.0f}% {ctx.profile.label} needs, and side energy down there "
            f"measures {_num(side_db, 1)} dB relative to mid. Stereo bass cancels unpredictably "
            f"on fold-down, wanders on a club rig where much of the low end is summed, and is "
            f"the single most common reason a cut gets rejected for vinyl."
        )
        hits.append(_Hit(
            Finding(
                id="low_end.not_mono",
                dimension="low_end",
                title="Low end is not mono",
                severity=_severity(ratio),
                confidence=0.92,
                detail=detail,
                evidence=[
                    _ev("Low-end mono ratio", mono_ratio, "", target_range=mono_window,
                        verdict=_verdict(mono_ratio, mono_window, 0.06)),
                    _ev("Side energy below 120 Hz", side_db, "dB", target=-12.0,
                        verdict="problem" if side_db > -12.0 else "good"),
                    _ev("Sub band width",
                        _fin((ctx.m.stereo.band_width or {}).get("sub", 0.0)), "",
                        target=0.0, verdict="watch"),
                ],
                band_hz=(20.0, 120.0),
            ),
            ratio,
        ))

    # -- inaudible energy under 25 Hz ---------------------------------------
    rumble = _fin(le.sub_rumble_db, -90.0)
    rumble_ceiling = _rumble_ceiling_db(ctx.genre_key)
    rumble_window = (-90.0, rumble_ceiling)
    rumble_miss = targets.range_miss(rumble, rumble_window)
    if rumble_miss > 0.0:
        ratio = _ratio(rumble_miss, 4.0)
        detail = (
            f"There is more sub-25 Hz energy here than {ctx.ref} carries: {_num(rumble, 1)} dB "
            f"of the total sits below 25 Hz, over the {_num(rumble_ceiling, 1)} dB point where "
            f"the figure stops being spectral leakage from the bass fundamental and starts "
            f"being real content — a ceiling set from {ctx.profile.label}'s own target curve, "
            f"which is why it is not the same number for every genre. It buys nothing audible: "
            f"no playback system reproduces it. It costs limiter headroom that the rest of the "
            f"record could be using."
            + ctx.advice(" A 24 dB/octave high-pass at 25 Hz hands that back for free.")
        )
        hits.append(_Hit(
            Finding(
                id="low_end.subsonic_rumble",
                dimension="low_end",
                title=f"More sub-25 Hz energy than {ctx.ref} carries",
                severity=_severity(ratio),
                confidence=0.80,
                detail=detail,
                evidence=[
                    _ev("Sub-25 Hz energy", rumble, "dB", target_range=rumble_window,
                        verdict="problem"),
                    _ev("Sub energy (20-60 Hz)", sub, "dB", target_range=sub_window,
                        verdict=_verdict(sub, sub_window, 3.0)),
                ],
                band_hz=(3.0, 25.0),
            ),
            ratio,
        ))

    return hits


# ---------------------------------------------------------------------------
# 9. Mud & low-mid buildup
# ---------------------------------------------------------------------------


def _detect_mud(ctx: _Ctx) -> List[_Hit]:
    """150-400 Hz against the low bass, against the mids, and 300-600 Hz boxiness.

    `mud_ratio_db` is the primary number because `targets.py` windows it per
    genre. The other two have their targets derived from the genre's own curve
    (see `_curve_density_db`), and because they are derived rather than
    hand-set, the detector only fires on them when *both* agree — one derived
    metric on its own is not enough to call a mix muddy.
    """
    if ctx.no_programme:
        return []

    s = ctx.m.spectral

    mud_ratio = _fin(s.mud_ratio_db, 0.0)
    mud_window = ctx.profile.mud_ratio_db
    mud_miss = targets.range_miss(mud_ratio, mud_window)

    tol_low_mid = _fin(targets.band_tolerance(ctx.genre_key, "low_mid"), 2.2)
    tol_mid = _fin(targets.band_tolerance(ctx.genre_key, "mid"), 2.0)
    tol_upper_mid = _fin(targets.band_tolerance(ctx.genre_key, "upper_mid"), 2.0)

    # A ratio between two bands can be wrong by either band's tolerance, so the
    # tolerance on the ratio is the sum of the two.
    m2m = _fin(s.mud_to_mid_db, 0.0)
    m2m_target = _curve_density_db(ctx.curve, (150.0, 400.0), (1000.0, 3000.0))
    m2m_tol = tol_low_mid + tol_upper_mid
    m2m_window = (m2m_target - m2m_tol, m2m_target + m2m_tol)
    m2m_miss = targets.range_miss(m2m, m2m_window)

    boxy = _fin(s.boxiness_db, 0.0)
    curve_median = float(np.median(ctx.curve[(_CENTERS >= 40.0) & (_CENTERS <= 16_000.0)]))
    boxy_target = _curve_region_db(ctx.curve, 300.0, 600.0) - curve_median
    boxy_tol = tol_low_mid + tol_mid
    boxy_window = (boxy_target - boxy_tol, boxy_target + boxy_tol)
    boxy_miss = targets.range_miss(boxy, boxy_window)

    primary = mud_miss > 0.0
    corroborated = m2m_miss > 0.0 and boxy_miss > 0.0
    if not (primary or corroborated):
        return []

    ratio = min(
        max(_ratio(mud_miss, 2.5), _ratio(m2m_miss, m2m_tol), _ratio(boxy_miss, boxy_tol)),
        6.0,
    )

    # A masking pair in the low mids is this same energy described from the
    # other side — the spectrum says the region is heavy, the stems say which
    # source is putting it there and which source it is burying. One problem,
    # one finding: mud claims those pairs so the clarity detector does not
    # report them again, and the pairs make this finding *specific* rather than
    # making it a second finding.
    mask_pairs = ctx.take_masking(_MASK_MUD_BANDS, limit=2)

    resonances = [r for r in (s.resonances or []) if 150.0 <= _fin(r.freq_hz) <= 500.0]
    resonances.sort(key=lambda r: -_fin(r.prominence_db))
    moments: List[Moment] = []
    for r in resonances[:2]:
        moments.extend(r.moments or [])
    moments = _moments(moments)

    res_text = ""
    if resonances:
        r = resonances[0]
        res_text = (
            f" The narrowest offender is a Q {_num(r.q, 1)} peak at {_num(r.freq_hz, 0)} Hz "
            f"standing {_num(r.prominence_db, 1)} dB above the octave around it"
        )
        if len(resonances) > 1:
            res_text += f", with a second at {_num(resonances[1].freq_hz, 0)} Hz"
        res_text += "."

    masking_text = ""
    # `lead_on_record` rather than the raw boolean: on a beat, or where the
    # voice test placed nothing, "the center vocal is being crowded" is a
    # sentence about a singer who is not on the file.
    if ctx.lead_on_record and ctx.m.vocal.masked_bands:
        masking_text = (
            f" The center vocal is being crowded in "
            f"{', '.join(b.replace('_', ' ') for b in ctx.m.vocal.masked_bands)}"
            + (" — though it is tucked under the bed here, so some of that is where it was "
               "placed rather than the low mids covering it."
               if ctx.lead_prominence == "tucked" else ".")
        )

    detail = (
        f"The low mids run heavier here than {ctx.ref} does. 150-400 Hz carries "
        f"{_num(mud_ratio, 1)} dB relative to 60-120 Hz against a "
        f"{_win(mud_window, 1, ' dB')} window for {ctx.profile.label}, and it sits "
        f"{_num(m2m, 1)} dB over 1-3 kHz where this genre's target curve puts it at "
        f"{_num(m2m_target, 1)} dB. 300-600 Hz reads {_num(boxy, 1)} dB against the mix's own "
        f"broadband median, {_num(boxy - boxy_target, 1)} dB more than the curve carries."
        f"{res_text} Weight in that region buys warmth and body; what it costs is everything "
        f"above the bass, which the region directly under it is covering."
        f"{masking_text}"
        f"{_mask_sentence(mask_pairs)}"
        + _moment_span(moments)
    )

    ctx.tags.add("mud")
    return [_Hit(
        Finding(
            id="mud.low_mid_buildup",
            dimension="mud",
            title=f"Low mids run heavier than {ctx.ref}",
            severity=_severity(ratio),
            # A long-term spectrum, directly measured. With a separated source
            # named as the masker it stops being "this region is heavy" and
            # becomes "this source is heavy here", which is a stronger claim
            # about the same numbers.
            confidence=0.94 if mask_pairs else 0.88,
            detail=detail,
            evidence=_mask_evidence(mask_pairs) + [
                _ev("150-400 Hz vs 60-120 Hz", mud_ratio, "dB", target_range=mud_window,
                    verdict=_verdict(mud_ratio, mud_window, 2.5),
                    detail=f"Windowed per genre in targets.py for {ctx.profile.label}."),
                _ev("150-400 Hz vs 1-3 kHz", m2m, "dB", target=m2m_target,
                    target_range=m2m_window, verdict=_verdict(m2m, m2m_window, m2m_tol),
                    detail="Target derived from this genre's own curve, density-corrected."),
                _ev("Boxiness (300-600 Hz)", boxy, "dB", target=boxy_target,
                    target_range=boxy_window, verdict=_verdict(boxy, boxy_window, boxy_tol)),
                _ev("Low-mid band vs target",
                    ctx.bands["low_mid"].deviation_db if "low_mid" in ctx.bands else 0.0,
                    "dB", target=0.0,
                    target_range=(-tol_low_mid, tol_low_mid),
                    verdict="problem" if ctx.bands.get("low_mid")
                    and ctx.bands["low_mid"].miss_db else "good"),
                _ev("Worst 150-500 Hz resonance",
                    _fin(resonances[0].prominence_db) if resonances else 0.0, "dB",
                    target_range=(0.0, 3.0),
                    verdict="problem" if resonances else "good",
                    detail=(f"{_num(resonances[0].freq_hz, 0)} Hz, Q {_num(resonances[0].q, 1)}"
                            if resonances else "No narrow peak found in this region.")),
            ],
            band_hz=(150.0, 400.0),
            moments=moments,
        ),
        ratio,
    )]


# ---------------------------------------------------------------------------
# 10. Harshness & sibilance
#
# Who owns the bursty top, and how we know.
#
# `sibilance_index` measures exactly one thing: how much louder the loudest
# 5-9 kHz frames are than the typical ones. A consonant does that. So does a
# closed hi-hat, a shaker, a tambourine and a rim click — a hat is a 40 ms noise
# burst in the same band, and on a beat there are two of them a beat. The index
# cannot tell them apart and was never asked to. Naming the source is this
# layer's job, and it was doing it by assumption.
#
# There are three states of knowledge here and the report should sound different
# in each. With stems we know, because the sources are separated and we can read
# the band off each one. Without stems but with a lead sitting up in the bed, it
# is a fair inference that the bursts are consonants. Without stems and without a
# lead up front, the only honest reading is percussion — and the confidence has
# to say that we inferred it.
# ---------------------------------------------------------------------------


@dataclass
class _TopBand:
    """The attribution of bursty 5-9 kHz energy to a source."""

    #: True when the bursts are consonants, i.e. the finding is sibilance.
    vocal: bool
    #: True when separation answered this, rather than us inferring it.
    measured: bool
    confidence: float
    #: The sentence naming the source and saying how we know.
    attribution: str
    #: What to call the culprit in the rest of the prose.
    culprit: str
    #: The evidence row carrying whichever figure did the attributing.
    evidence: Evidence


def _percussion_label(occ: Dict[str, float]) -> str:
    """Name the bright non-vocal sources, loudest first, in a producer's words.

    A second source is only named when it is genuinely comparable to the first.
    "The drums and the synths, guitars and everything else" is technically true
    of an 88/12 split and reads like a machine wrote it.
    """
    names = {
        "drums": "the drums",
        "other": "the synths, guitars and everything else",
        "bass": "the bass",
    }
    ranked = [k for k in sorted(occ, key=lambda k: -occ[k]) if occ[k] > 0.05 and k in names]
    if not ranked:
        return "the percussion"
    if len(ranked) > 1 and occ[ranked[1]] >= 0.5 * occ[ranked[0]]:
        return f"{names[ranked[0]]} and {names[ranked[1]]}"
    return names[ranked[0]]


def _attribute_top_band(ctx: _Ctx) -> _TopBand:
    """Decide what is making 5-9 kHz burst before the report names it.

    Ordered by how good the evidence is, not by convenience.

    1. **Stems.** `band_occupancy` is the share of a macro band each separated
       source owns, measured on the sources themselves. If the vocal stem owns
       the top, the bursts are consonants and we can say so outright; if the
       drums own it, they are hats and we can say that outright too. This is the
       only branch that is entitled to a high confidence, and the only one whose
       prose is allowed to say "measured".

    2. **A lead sitting up in the bed.** No stems, but the center-channel voice
       test is confident and the lead is balanced or forward. Consonants are the
       likeliest explanation, so the finding is sibilance at the confidence a
       center estimate has always earned.

    3. **Everything else.** No stems, and either no lead at all or a lead tucked
       under the bed. This is the case the whole rewrite exists for: a beat with
       a hook mixed low so someone can rap over it has a voice on it *and* its
       top octave is still hi-hats. Attributing that to the singer sends the
       producer to a de-esser to fix a hat pattern.
    """
    voc_sib = _fin(ctx.m.vocal.sibilance_db, -60.0)

    # -- 1. measured ---------------------------------------------------------
    if ctx.has_stems:
        occ = {
            kind: _fin((st.band_occupancy or {}).get(_TOP_BAND, 0.0), 0.0)
            for kind, st in ctx.stems_by_kind.items()
        }
        voc_occ = occ.get("vocals", 0.0)
        rest = {k: v for k, v in occ.items() if k != "vocals"}
        rest_occ = sum(rest.values())

        if voc_occ >= _TOP_OWNER_SHARE and voc_occ >= rest_occ:
            return _TopBand(
                vocal=True,
                measured=True,
                confidence=ctx.trust(0.90, "stem"),
                culprit="the lead vocal",
                attribution=(
                    f"Separation settles it rather than inferring it: the vocal stem owns "
                    f"{voc_occ * 100:.0f}% of the 5-10 kHz energy against "
                    f"{rest_occ * 100:.0f}% for everything else, so the bursts really are "
                    f"consonants."
                ),
                evidence=_ev(
                    "Vocal stem share of 5-10 kHz", voc_occ, "",
                    target_range=(0.0, _TOP_OWNER_SHARE),
                    verdict="problem" if voc_occ >= _TOP_OWNER_SHARE else "good",
                    detail="Measured on the separated sources. This is the one case where "
                           "the source of the burstiness is known rather than inferred.",
                ),
            )

        culprit = _percussion_label(rest)
        return _TopBand(
            vocal=False,
            measured=True,
            confidence=ctx.trust(0.88, "stem"),
            culprit=culprit,
            attribution=(
                f"Separation settles it rather than inferring it: {culprit} own "
                f"{rest_occ * 100:.0f}% of the 5-10 kHz energy against "
                f"{voc_occ * 100:.0f}% for the vocal stem"
                + ("" if "vocals" in ctx.stems_by_kind
                   else ", and the separator found no vocal source on this file at all")
                # Deliberately does not re-name the culprit: the detector's next
                # sentence does that, and `culprit` is not always percussion.
                + ". Whatever is bursting up there, it is not a singer's consonants."
            ),
            evidence=_ev(
                "Share of 5-10 kHz outside the vocal", rest_occ, "",
                target_range=(0.0, 1.0), verdict="good",
                detail=f"Measured on the separated sources: {culprit} against "
                       f"{voc_occ * 100:.0f}% for the vocal stem. A measurement, not a "
                       f"center-channel guess.",
            ),
        )

    # -- 2. inferred, lead up front -----------------------------------------
    if ctx.lead_is_up_front:
        return _TopBand(
            vocal=True,
            measured=False,
            confidence=0.72,
            culprit="the lead vocal",
            attribution=(
                f"There is a lead vocal on this record for the bursts to belong to — the "
                f"voice test scores {ctx.lead_confidence:.2f} and puts the lead "
                f"{ctx.lead_prominence} in the bed — and in the center channel the "
                f"95th-percentile 5-9 kHz level sits {_num(voc_sib, 1)} dB against the median "
                f"vocal-band level. That is an inference from a center estimate, not a stem."
            ),
            evidence=_ev(
                "Center 5-9 kHz vs vocal band", voc_sib, "dB", target=-12.0,
                verdict="watch" if voc_sib > -12.0 else "good",
                detail="Derived from center extraction — an inference, not a stem.",
            ),
        )

    # -- 3. inferred, no lead up front --------------------------------------
    if ctx.lead_prominence == "tucked":
        why = (
            f"There is a voice here, but the voice test puts it tucked under the bed at "
            f"{_num(_fin(ctx.m.vocal.vocal_to_instrument_db), 1)} dB against the "
            f"instruments, which is where a hook goes when it is meant to be rapped over. "
            f"A lead that far down is not what is making the top of the mix burst"
        )
    elif not ctx.expects_lead:
        why = {
            "beat": "A beat has no topline recorded yet, so nothing up here can be a consonant",
            "instrumental": "This is an instrumental, so nothing up here can be a consonant",
            "stem": "This is a single stem, so nothing up here can be a consonant",
        }.get(str(ctx.intent), "No lead vocal belongs on this file, so nothing up here "
                               "can be a consonant")
    elif ctx.lead_confidence > 0.0:
        why = (
            f"The voice test scores only {ctx.lead_confidence:.2f}, which is not enough to "
            f"put a singer behind these bursts"
        )
    else:
        why = (
            "No lead vocal was detected, so there is no voice here for these bursts to be "
            "the consonants of"
        )

    return _TopBand(
        vocal=False,
        measured=False,
        confidence=0.55,
        culprit="the hats, shakers and rim clicks",
        attribution=(
            f"{why}. Without separated stems this is an inference rather than a measurement, "
            f"which is what the confidence on this finding reflects — but a burst in this band "
            f"is a burst whatever makes it, and percussion is the overwhelmingly likelier "
            f"source."
        ),
        evidence=_ev(
            "Voice-test confidence", ctx.lead_confidence, "",
            target_range=(_LEAD_UP_FRONT_CONFIDENCE, 1.0),
            verdict="good",
            detail=f"Below {_num(_LEAD_UP_FRONT_CONFIDENCE, 2)}, or a lead sitting under the "
                   f"bed, means the 5-9 kHz bursts are attributed to percussion instead of "
                   f"consonants. Separate the stems to turn this inference into a measurement.",
        ),
    )


def _detect_harshness(ctx: _Ctx) -> List[_Hit]:
    """2-5 kHz edge and 5-9 kHz top-end burstiness, against the genre's ceilings.

    Both indices already answer "does this region stick out of the curve its own
    neighbors draw" rather than "is this region loud", so a legitimately bright
    mix scores near zero on them. The genre ceilings come straight from
    `targets.py`: rock tolerates 0.46, R&B 0.34, lo-fi 0.28.

    **The 5-9 kHz arm does not assume a singer.** `sibilance_index` measures
    frame-to-frame burstiness in 5-9 kHz, and that is the signature of a
    consonant *and* the signature of a closed hi-hat, a shaker, a tambourine and
    a rim click, which are not distinguishable from it. This detector used to
    fire on the index alone and title the result "Sibilance" whether or not
    there was a voice on the record — so a trap beat, whose whole top octave is
    hats by design, was told it had a vocal problem it could not possibly have.

    So the arm asks `_attribute_top_band` what is making the band burst before it
    names it, and emits a different finding depending on the answer:

    * **`harshness.sibilance`** when the bursts are consonants — either measured
      on a separated vocal stem, or inferred from a lead the voice test is
      confident about that is sitting in or above the bed. The window is the
      genre's own `sibilance_max`, which is written for exactly that record.
    * **`harshness.bright_transients`** otherwise. Not "no problem found": the
      measurement is real and a peaky, bursty top octave is fatiguing whatever
      makes it. But it is hats, shakers and rim clicks, the prose says so, and
      the fixes point at a transient shaper and the percussion bus rather than
      at a de-esser. The window widens by `_PERCUSSION_SIB_FACTOR`, because on
      that material the burstiness is the arrangement and hats are supposed to
      cut.

    A tucked lead lands in the second branch on purpose. A hook mixed low so a
    rapper can go over it is a real voice and is still not what is making the
    top of the mix spit, and sending that producer to a de-esser would have them
    dulling a vocal that is already too quiet to hear.

    Both are deviations either way: how much air and how much hat is taste.
    """
    if ctx.no_programme:
        return []

    s = ctx.m.spectral
    hits: List[_Hit] = []

    def _res_in(lo: float, hi: float):
        found = [r for r in (s.resonances or []) if lo <= _fin(r.freq_hz) <= hi]
        found.sort(key=lambda r: -_fin(r.prominence_db))
        return found

    # -- 2-5 kHz -------------------------------------------------------------
    harsh = _fin(s.harshness_index, 0.0)
    harsh_window = (0.0, _fin(ctx.profile.harshness_max, 0.42))
    harsh_miss = targets.range_miss(harsh, harsh_window)
    sharp = _fin(s.sharpness_acum, 0.0)
    sharp_window = (0.0, _fin(ctx.profile.sharpness_max_acum, 2.6))
    sharp_miss = targets.range_miss(sharp, sharp_window)

    if harsh_miss > 0.0 or sharp_miss > 0.0:
        ratio = min(max(_ratio(harsh_miss, 0.15), _ratio(sharp_miss, 0.5)), 6.0)
        res = _res_in(1800.0, 5200.0)
        moments = _moments([mo for r in res[:2] for mo in (r.moments or [])])
        band = ctx.bands.get("presence")
        res_text = ""
        if res:
            r = res[0]
            res_text = (
                f" The peak driving it is a Q {_num(r.q, 1)} resonance at {_num(r.freq_hz, 0)} Hz "
                f"standing {_num(r.prominence_db, 1)} dB above the octave around it"
                + (f", with another at {_num(res[1].freq_hz, 0)} Hz." if len(res) > 1 else ".")
            )
        detail = (
            f"2-5 kHz pushes harder here than {ctx.ref} does: the harshness index scores "
            f"{harsh:.3f} against a {_num(harsh_window[1], 2)} ceiling for "
            f"{ctx.profile.label}, and psychoacoustic sharpness reads {_num(sharp, 2)} acum "
            f"against {_num(sharp_window[1], 1)}. This is not the same as brightness — the "
            f"index measures how far the region stands above the line its own neighbors "
            f"draw, so a mix with an ordinary downward tilt scores near zero.{res_text}"
            + (f" The presence band sits {_num(band.deviation_db, 1)} dB "
               f"{'over' if band.deviation_db > 0 else 'under'} the {ctx.profile.label} curve."
               if band else "")
            + " Energy here buys cut and presence on small speakers; it costs listening "
              "fatigue at volume and is the first thing that turns unpleasant on earbuds."
            + _moment_span(moments)
        )
        hits.append(_Hit(
            Finding(
                id="harshness.upper_mid_edge",
                dimension="harshness",
                title=f"2-5 kHz pushes harder than {ctx.ref}",
                severity=_severity(ratio),
                # A modeled 0-1 index over a directly measured spectrum.
                confidence=0.78,
                detail=detail,
                evidence=[
                    _ev("Harshness index", harsh, "", target_range=harsh_window,
                        verdict=_verdict(harsh, harsh_window, 0.15),
                        detail=f"{ctx.profile.label} ceiling from targets.py."),
                    _ev("Sharpness", sharp, "acum", target_range=sharp_window,
                        verdict=_verdict(sharp, sharp_window, 0.5),
                        detail="Zwicker-style, level-independent."),
                    _ev("Presence band vs target",
                        band.deviation_db if band else 0.0, "dB", target=0.0,
                        target_range=(-(band.tolerance_db if band else 2.2),
                                      (band.tolerance_db if band else 2.2)),
                        verdict="problem" if band and band.miss_db else "good"),
                    _ev("Worst 2-5 kHz resonance",
                        _fin(res[0].prominence_db) if res else 0.0, "dB",
                        target_range=(0.0, 3.0), verdict="problem" if res else "good",
                        detail=(f"{_num(res[0].freq_hz, 0)} Hz, Q {_num(res[0].q, 1)}"
                                if res else "No narrow peak in this region.")),
                    _ev("Spectral centroid", _fin(s.spectral_centroid_hz), "Hz",
                        verdict="good"),
                ],
                band_hz=(2000.0, 5000.0),
                moments=moments,
            ),
            ratio,
        ))
        ctx.tags.add("harsh")

    # -- 5-9 kHz -------------------------------------------------------------
    # What is making this band burst decides what it gets called, which finding
    # is emitted, what window it is held to, and how confident the report is
    # allowed to be about it. `_attribute_top_band` is the whole decision.
    top = _attribute_top_band(ctx)
    sib = _fin(s.sibilance_index, 0.0)
    genre_sib_max = _fin(ctx.profile.sibilance_max, 0.40)
    sib_ceiling = genre_sib_max if top.vocal else genre_sib_max * _PERCUSSION_SIB_FACTOR
    sib_window = (0.0, sib_ceiling)
    sib_miss = targets.range_miss(sib, sib_window)
    if sib_miss > 0.0:
        ratio = min(_ratio(sib_miss, 0.15), 6.0)
        res = _res_in(5000.0, 9000.0)
        moments = _moments([mo for r in res[:2] for mo in (r.moments or [])])
        brilliance = ctx.bands.get("brilliance")
        peak_text = (
            f" The narrowest peak is {_num(res[0].freq_hz, 0)} Hz at "
            f"{_num(res[0].prominence_db, 1)} dB prominence." if res else ""
        )
        index_text = (
            f"The 5-9 kHz band is burstier than {ctx.ref} runs: it scores {sib:.3f} on the "
            f"burstiness index against a {_num(sib_ceiling, 2)} ceiling"
            + ("" if top.vocal else
               f" ({_num(genre_sib_max, 2)} for {ctx.profile.label}, widened because there is "
               f"no lead vocal up front for the band to be protecting)")
            + f". The index reads how far the loudest frames in the band stand above the "
              f"typical ones rather than how loud the band is, so a steady shimmer of cymbals "
              f"and room does not register at all — but a consonant and a closed hi-hat are "
              f"both short, bright noise bursts and the index cannot tell them apart. "
        )

        if top.vocal:
            finding_id = "harshness.sibilance"
            title = f"5-9 kHz sibilance runs above {ctx.ref}"
            detail = (
                index_text
                + top.attribution
                + peak_text
                + ctx.advice(
                    " De-essing buys smoother consonants and costs air on everything else "
                    "sharing the band, so it is worth doing on the vocal rather than the bus."
                )
                + _moment_span(moments)
            )
        else:
            finding_id = "harshness.bright_transients"
            title = f"Bright percussive transients above {ctx.ref}"
            detail = (
                index_text
                + top.attribution
                + f" So this is {top.culprit}: transients landing hard in the same octave an "
                  f"'ess' would, which means a de-esser is the wrong tool and a transient "
                  f"shaper on the percussion group is the right one."
                + peak_text
                + (" Hats cutting this hard is a choice, not a fault: it buys a top end that "
                   "reads on a phone and through a rapper sitting on top of it, and it costs "
                   "listening fatigue at volume plus headroom the topline is going to want."
                   if str(ctx.intent) == "beat" else
                   " It buys presence on small speakers and costs fatigue at volume — worth "
                   "keeping if the brightness is the point, worth controlling if it is not.")
                + _moment_span(moments)
            )

        hits.append(_Hit(
            Finding(
                id=finding_id,
                dimension="harshness",
                title=title,
                severity=_severity(ratio),
                confidence=top.confidence,
                detail=detail,
                evidence=[
                    _ev("5-9 kHz burstiness index", sib, "", target_range=sib_window,
                        verdict=_verdict(sib, sib_window, 0.15),
                        detail=(
                            f"Attributed to {top.culprit}, "
                            + ("measured on the separated sources."
                               if top.measured else
                               "inferred without stems — separate them to know for certain.")
                        )),
                    top.evidence,
                    _ev("Brilliance band vs target",
                        brilliance.deviation_db if brilliance else 0.0, "dB", target=0.0,
                        target_range=(-(brilliance.tolerance_db if brilliance else 3.0),
                                      (brilliance.tolerance_db if brilliance else 3.0)),
                        verdict="watch"),
                ],
                band_hz=(5000.0, 9000.0),
                moments=moments,
            ),
            ratio,
        ))
        # Either way the brilliance band has now been spoken for, so the
        # frequency-balance detector stands down on it (rule 5).
        ctx.tags.add("sibilance" if top.vocal else "bright_transients")

    return hits


# ---------------------------------------------------------------------------
# 11. Frequency balance
# ---------------------------------------------------------------------------

# Which macro bands a more specific detector has already spoken for. The
# frequency-balance detector stands down on those rather than describing the
# same energy a second time (rule 5).
_COVERED_BY: Dict[str, Tuple[str, ...]] = {
    "mud": ("upper_bass", "low_mid"),
    "harsh": ("presence",),
    "sibilance": ("brilliance",),
    "bright_transients": ("brilliance",),
    "sub_energy": ("sub", "low_bass"),
}


def _detect_frequency_balance(ctx: _Ctx) -> List[_Hit]:
    """Macro bands against the genre curve — at most one hot and one thin.

    Nine bands times two directions is a wall, and a wall is not a report. Only
    the worst deviation in each direction is emitted, measured in units of that
    band's own tolerance so a 3 dB miss in the mids outranks a 5 dB miss in the
    air band (which `targets.py` allows twice as much room).
    """
    if ctx.no_programme or not ctx.bands:
        return []

    covered: set = set()
    for tag, names in _COVERED_BY.items():
        if tag in ctx.tags:
            covered.update(names)

    candidates = [b for b in ctx.bands.values() if b.miss_db != 0.0 and b.name not in covered]

    # A beat leaves the mid-range open on purpose: that is where the topline
    # goes. Reading light there against a curve fitted to finished songs — which
    # have a voice filling exactly those bands — is the arrangement doing its
    # job, so the thin side stands down for the two bands a lead occupies. The
    # hot side is untouched, because mids that are *full* are the actual problem
    # a rapper runs into.
    if ctx.intent == "beat":
        candidates = [
            b for b in candidates
            if not (b.name in _TOPLINE_BANDS and b.miss_db < 0.0)
        ]

    if not candidates:
        return []

    measured_tilt = _fin(ctx.m.spectral.spectral_tilt_db_per_decade, 0.0)
    target_tilt = _curve_tilt(ctx.curve, ctx.m.spectral.third_octave_db)
    tilt_window = (target_tilt - 4.0, target_tilt + 4.0)
    tilt_miss = targets.range_miss(measured_tilt, tilt_window)

    hits: List[_Hit] = []
    for direction, sign in (("hot", 1.0), ("thin", -1.0)):
        pool = [b for b in candidates if b.miss_db * sign > 0.0]
        if not pool:
            continue
        band = max(pool, key=lambda b: b.ratio)
        ratio = min(band.ratio, 6.0)

        neighbours = [
            ctx.bands[n] for n in ctx.bands
            if n != band.name and abs(ctx.bands[n].center_hz / max(band.center_hz, 1e-9) - 1.0) < 3.0
        ]
        neighbours.sort(key=lambda b: abs(math.log2(max(b.center_hz, 1e-9) /
                                                    max(band.center_hz, 1e-9))))
        near = neighbours[0] if neighbours else None

        detail = (
            f"The {band.label} band ({band.span}) sits {_num(abs(band.deviation_db), 1)} dB "
            f"{'above' if sign > 0 else 'below'} where {ctx.ref} puts it: the band measures "
            f"{_num(band.level_db, 1)} dB against a {_num(band.target_db, 1)} dB target for "
            f"{ctx.profile.label}, which is {_num(abs(band.miss_db), 1)} dB outside the "
            f"{_pm(band.tolerance_db)} dB of room the reference leaves there"
            + (f", while {near.label} next to it sits {_num(near.deviation_db, 1)} dB "
               f"from its own target" if near else "")
            + f". Overall tilt is {_num(measured_tilt, 1)} dB/decade against "
            f"{_num(target_tilt, 1)} dB/decade for the {ctx.profile.label} curve"
            + (f", so the whole balance reads {'darker' if tilt_miss < 0 else 'brighter'} than "
               f"the reference and not just this one band." if tilt_miss != 0.0
               else ", so the overall slope matches and this is one band, not a tonality.")
            + (f" That buys {'weight and size' if sign > 0 else 'clarity and headroom'} and "
               f"costs {'clarity and headroom' if sign > 0 else 'weight and size'} — worth "
               f"keeping if it is the sound you want."
               if ctx.intent != "reference" else "")
        )

        hits.append(_Hit(
            Finding(
                id=f"frequency_balance.{band.name}_{direction}",
                dimension="frequency_balance",
                title=f"{band.label[:1].upper()}{band.label[1:]} "
                      + (band.verb("runs hot", "run hot") if sign > 0
                         else band.verb("sits light", "sit light"))
                      + f" against {ctx.ref}",
                severity=_severity(ratio),
                confidence=0.88,
                detail=detail,
                evidence=[
                    _ev(f"{band.label.title()} level", band.level_db, "dB",
                        target=band.target_db,
                        target_range=(band.target_db - band.tolerance_db,
                                      band.target_db + band.tolerance_db),
                        verdict="problem" if band.ratio >= _MAJOR_AT else "watch",
                        detail=f"{band.span}, relative to the mix's own 1 kHz octave."),
                    _ev("Deviation from target", band.deviation_db, "dB", target=0.0,
                        target_range=(-band.tolerance_db, band.tolerance_db),
                        verdict="problem"),
                    _ev("Spectral tilt", measured_tilt, "dB/decade", target=target_tilt,
                        target_range=tilt_window,
                        verdict=_verdict(measured_tilt, tilt_window, 4.0),
                        detail="Target tilt fitted from this genre's own curve over the "
                               "same bands the measurement used."),
                ] + ([
                    _ev(f"{near.label.title()} level", near.level_db, "dB",
                        target=near.target_db,
                        target_range=(near.target_db - near.tolerance_db,
                                      near.target_db + near.tolerance_db),
                        verdict="problem" if near.ratio >= _MAJOR_AT else "good")
                ] if near else []),
                band_hz=(band.low_hz, band.high_hz),
            ),
            ratio,
        ))

    return hits



# ---------------------------------------------------------------------------
# 12. Vocal balance
# ---------------------------------------------------------------------------


def _topline_headroom(ctx: _Ctx) -> List[_Hit]:
    """The one vocal-dimension finding a beat can produce, and it is good news.

    A beat is sold on whether somebody can rap over it. That needs two things
    at once and they are separately measurable: nothing hogging the center at
    lead level, and nothing filling the 400 Hz-6 kHz pocket a voice lives in.
    Both are already measured for other reasons, so confirming them costs
    nothing and answers the question the producer actually has.

    This is an observation, not a problem: severity "clean", no impact, and it
    is only emitted when the space is genuinely there. Silence is the answer
    when it is not — telling a producer their beat is *too full* for a topline
    is a real finding, but it comes from the mids running hot against the genre
    curve, which `_detect_frequency_balance` already reports.
    """
    v = ctx.m.vocal
    if ctx.is_mono or not ctx.has("vocal"):
        # Center extraction needs a side channel, and the syllabic test needs
        # material. Without both there is nothing measured to confirm.
        return []

    v2i = _fin(v.vocal_to_instrument_db, 0.0)
    v2i_window = ctx.profile.vocal_to_instrument_db
    centre_open = v2i <= _fin(v2i_window[0], -4.0)

    pocket = [ctx.bands[n] for n in _TOPLINE_BANDS if n in ctx.bands]
    pocket_open = bool(pocket) and all(b.miss_db <= 0.0 for b in pocket)
    if not (centre_open and pocket_open):
        return []

    hottest = max(pocket, key=lambda b: b.deviation_db)
    return [_Hit(
        Finding(
            id="vocal_balance.topline_headroom",
            dimension="vocal_balance",
            title="Center is open for a topline",
            kind="deviation",
            severity="clean",
            confidence=ctx.trust(0.60, "vocal"),
            detail=(
                f"There is room for a rapper or a singer to sit in this, and both halves of "
                f"that measure clean. The center 300 Hz-6 kHz runs {_num(v2i, 1)} dB against "
                f"everything else, at or under the {_win(v2i_window, 1, ' dB')} band "
                f"{ctx.profile.label} puts a lead in — so a topline can come up to level "
                f"without anything having to move out of its way. The pocket a voice occupies "
                f"is clear too: the mids and upper mids are inside the {ctx.profile.label} "
                f"curve's tolerance, the fullest of them "
                f"({hottest.label}, {hottest.span}) sitting {_num(hottest.deviation_db, 1)} dB "
                f"from target against {_pm(hottest.tolerance_db)} dB of allowance. Nothing "
                f"about the missing lead is being scored as a fault on this file."
            ),
            evidence=[
                _ev("Center vs everything else", v2i, "dB", target_range=v2i_window,
                    verdict="good",
                    detail="A-weighted center 300 Hz-6 kHz. Under the lead window means "
                           "space, not a buried vocal — there is no vocal here yet."),
                _ev("Center energy ratio", _fin(v.center_energy_ratio), "",
                    target_range=(0.0, 1.0), verdict="good"),
            ] + [
                _ev(f"{b.label.title()} vs target", b.deviation_db, "dB", target=0.0,
                    target_range=(-b.tolerance_db, b.tolerance_db), verdict="good",
                    detail=f"{b.span}, against the {ctx.profile.label} curve.")
                for b in pocket
            ],
            band_hz=(300.0, 6000.0),
        ),
        0.0,
        True,
    )]


def _detect_vocal(ctx: _Ctx) -> List[_Hit]:
    """Lead vocal level, consistency and intelligibility.

    Two sources, and this detector prefers the better one.

    *Without stems* everything here comes from a center estimate, and the DSP
    layer's own `vocal_present` test (syllabic modulation specific to the
    center) is the gate: if it says there is no voice, this reports nothing
    rather than describing a centered synth pad as a badly balanced singer. A
    center estimate cannot tell a lead vocal from a centered synth, a snare or a
    mono bass, so confidences top out around 0.55 and the sentences say the
    figure describes "everything centered".

    *With stems* the same two questions are answered by measurement.
    `vocal_to_instrument_db` is the vocal stem's gated loudness against the
    other three summed, and consistency is the spread of the vocal's own level
    over the frames it is actually singing on. Neither moves when a synth moves.
    Confidence goes to 0.90/0.84, the sentence cites the stem ratio, and the
    detector stops needing a stereo field at all — a mono file has no center
    channel and still has a vocal stem.

    **A tucked lead is a decision until proven otherwise.** This detector reads
    `vocal_prominence` and `vocal_confidence`, not the boolean. A lead the center
    estimate places under the bed, at a confidence that only just cleared the
    line, gets an observation and never a verdict: the miss ratio is held under
    `_MAJOR_AT` so the severity cannot exceed "minor", the sentence says what
    being down there costs and what it buys, and the health score barely moves.
    That is the correct answer for a beat with a hook mixed low for someone to
    rap over, and a plausible one for shoegaze, lo-fi and half of modern indie.
    With stems in hand the cap comes off, because then the level is measured on
    the vocal itself rather than on everything that happens to be centered.
    """
    v = ctx.m.vocal
    if ctx.no_programme:
        return []

    # Intent gate, before anything is measured. On a beat, an instrumental, a
    # stem or somebody else's record there is no lead vocal to balance, so no
    # vocal finding of any kind is emitted — a tucked hook under a beat is the
    # brief, not a buried vocal. The beat case gets one thing back: a positive
    # confirmation that the space a topline needs is actually there.
    if not ctx.expects_lead:
        if ctx.intent == "beat":
            return _topline_headroom(ctx)
        return []

    stem = ctx.stem("vocals")
    from_stems = bool(
        stem is not None
        and ctx.has_stems
        and ctx.has("stem")
        and ctx.stems.vocal_to_instrument_db is not None
    )
    if not from_stems:
        if ctx.is_mono or not ctx.has("vocal") or not ctx.lead_on_record:
            return []

    hits: List[_Hit] = []

    # The center measurement stays usable as *supporting* evidence whenever it
    # was meaningful — it is what carries the timeline moments and the
    # intelligibility index, neither of which the stem pass produces.
    centre_valid = ctx.lead_on_record and not ctx.is_mono

    # How the two graded figures from the DSP layer change what may be said.
    # `tucked` is the whole point of this: a lead the center estimate places
    # under the bed is a production decision far more often than it is a
    # mistake, and nothing below is allowed to call it broken. Confidence does
    # not gate that — how sure we are that a voice is there does not make being
    # under the bed any more of a fault — it only decides how hard the sentence
    # hedges. What does gate it is stems: once the level is measured on the
    # vocal itself rather than on everything that happens to be centered, the
    # figure is about the singer and stands on its own.
    lead_conf = ctx.lead_confidence
    prominence = ctx.lead_prominence
    tucked = (not from_stems) and prominence == "tucked"
    hedge = (
        "and on this file that may well be the point"
        if lead_conf < _TUCKED_SURE_AT
        else "and on this file that is very likely deliberate"
    )

    def _centre_trust(base: float) -> float:
        """Confidence for a center-derived vocal finding, scaled by the voice test.

        A 0.58 voice call and a 0.96 one are not the same evidence, and until
        `vocal_confidence` existed they produced identical numbers on the report.
        """
        return _clamp(ctx.trust(base, "vocal") * (0.45 + 0.55 * lead_conf), 0.05, 0.99)

    v2i_window = ctx.profile.vocal_to_instrument_db
    if from_stems:
        v2i = _fin(ctx.stems.vocal_to_instrument_db, 0.0)
        spread = _stem_level_spread(stem) if stem is not None else None
        consistency = spread if spread is not None else _fin(v.consistency_db, 0.0)
        consistency_source = "stem" if spread is not None else "centre"
    else:
        v2i = _fin(v.vocal_to_instrument_db, 0.0)
        consistency = _fin(v.consistency_db, 0.0)
        consistency_source = "centre"
    v2i_miss = targets.range_miss(v2i, v2i_window)

    # How uneven a vocal may be follows the genre's own tolerance for level
    # movement: a folk record with 14 LU of range earns a wider vocal spread
    # than a trap record with 6.
    consistency_max = 6.0 + 0.4 * _fin(ctx.profile.loudness_range_lu[1], 8.0)
    consistency_window = (0.0, consistency_max)
    consistency_miss = targets.range_miss(consistency, consistency_window)

    intel = _fin(v.intelligibility_index, 0.5)
    intel_window = (max(_fin(ctx.profile.clarity_min, 0.45) - 0.05, 0.1), 1.0)
    intel_miss = targets.range_miss(intel, intel_window) if centre_valid else 0.0

    # Only cited when the center measurement was meaningful: a masked-band list
    # from a file with no detected center voice describes whatever else is
    # centered, not the singer.
    masked = list(v.masked_bands or []) if centre_valid else []
    # Sources the separator says are burying the vocal. Claimed here so the
    # clarity detector does not report the same burial a second time — "the
    # guitars are on top of the vocal" is a vocal-balance problem, and that is
    # where a producer will look for it.
    vocal_masking = ctx.take_masking(maskee="vocals", limit=2) if from_stems else []

    if from_stems:
        v2i_evidence = _ev(
            "Vocal vs instruments", v2i, "dB", target_range=v2i_window,
            verdict=_verdict(v2i, v2i_window, 1.5),
            detail="Gated loudness of the separated vocal against drums, bass and "
                   "everything else summed. A measurement, not a center estimate.",
        )
        second = _ev(
            "Vocal stem level", _fin(stem.level_ratio_db if stem else 0.0), "dB",
            target=None, verdict="good",
            detail=f"The vocal stem against the full mix, and it sounds on "
                   f"{_fin(stem.active_ratio if stem else 0.0) * 100:.0f}% of the track.",
        )
        spread_detail = (
            f"90th minus 10th percentile of the vocal stem's own level, over the frames "
            f"it is singing on. Allowance scales with {ctx.profile.label}'s "
            f"{_win(ctx.profile.loudness_range_lu, 1, ' LU')} loudness range."
            if consistency_source == "stem" else
            f"90th minus 10th percentile over time. Allowance scales with "
            f"{ctx.profile.label}'s {_win(ctx.profile.loudness_range_lu, 1, ' LU')} "
            f"loudness range."
        )
    else:
        v2i_evidence = _ev(
            "Vocal vs instruments", v2i, "dB", target_range=v2i_window,
            verdict=_verdict(v2i, v2i_window, 1.5),
            detail="A-weighted center 300 Hz-6 kHz against everything else. Center "
                   "extraction, not a stem.",
        )
        second = _ev("Center energy ratio", _fin(v.center_energy_ratio), "",
                     target_range=(0.12, 1.0), verdict="good")
        spread_detail = (
            f"90th minus 10th percentile over time. Allowance scales with "
            f"{ctx.profile.label}'s {_win(ctx.profile.loudness_range_lu, 1, ' LU')} "
            f"loudness range."
        )

    evidence = [
        v2i_evidence,
        second,
        _ev("Vocal level spread", consistency, "dB", target_range=consistency_window,
            verdict=_verdict(consistency, consistency_window, 2.0),
            detail=spread_detail),
        _ev("Intelligibility index", intel, "", target_range=intel_window,
            verdict=_verdict(intel, intel_window, 0.12) if centre_valid else "good",
            detail="" if centre_valid else
                   "Not scored: it is derived from center extraction, which needs a "
                   "stereo field and a detected center voice."),
        _ev("Presence balance", _fin(v.presence_balance_db), "dB", verdict="good",
            detail="2-6 kHz against 300 Hz-1 kHz within the center."),
    ] + _mask_evidence(vocal_masking)

    if v2i_miss != 0.0:
        ratio = _ratio(v2i_miss, 1.5)
        buried = v2i_miss < 0.0
        # The cap. A tucked lead under the genre window is the one case where
        # the measurement is right and the verdict would be wrong.
        gentle = tucked and buried
        if gentle:
            ratio = min(ratio, _TUCKED_MAX_RATIO)
        moments = _moments(v.buried_moments if buried else v.loud_moments) if centre_valid else []
        if not moments and vocal_masking:
            moments = _moments([mo for p in vocal_masking for mo in (p.moments or [])])
        if from_stems:
            detail = (
                f"The separated vocal sits {_num(v2i, 1)} dB against the drums, bass and "
                f"everything else summed, {_num(abs(v2i_miss), 1)} dB "
                f"{'below' if buried else 'above'} the {_win(v2i_window, 1, ' dB')} window "
                f"{ctx.profile.label} places a lead vocal in. That is the vocal's own gated "
                f"loudness against the actual instrumental — not the center channel, so a "
                f"centered synth or a mono bass is on the other side of the ratio where it "
                f"belongs"
                + (f", and it sings on {_fin(stem.active_ratio if stem else 0.0) * 100:.0f}% "
                   f"of the track" if stem is not None else "")
                + "."
                + (f" Intelligibility scores {intel:.2f}." if centre_valid else "")
                # The center estimate agreeing that the lead is under the bed is
                # a second, independent source saying the same thing — which
                # makes it likelier to be a decision, not an accident.
                + (" The center estimate places it tucked under the bed too, so treat this as "
                   "a difference from the reference rather than a fault: what it costs is "
                   "intelligibility, what it buys is room for whatever goes on top."
                   if buried and prominence == "tucked" else "")
                + _mask_sentence(vocal_masking)
                + _moment_span(moments)
            )
        elif gentle:
            detail = (
                f"The lead is sitting under {ctx.ref}, {hedge}. The center 300 Hz-6 kHz "
                f"measures {_num(v2i, 1)} dB against everything "
                f"else (A-weighted), {_num(abs(v2i_miss), 1)} dB below the "
                f"{_win(v2i_window, 1, ' dB')} window {ctx.profile.label} usually places a "
                f"lead in, at a voice-test confidence of {lead_conf:.2f}. That figure says a "
                f"voice is there; nothing in it says the voice was meant to be louder. A hook "
                f"tucked this far down is what you do when somebody is going to rap over the "
                f"beat, and it is the sound of shoegaze, lo-fi and a good deal of indie. "
                f"What it costs is intelligibility, currently {intel:.2f}"
                + (f", with {', '.join(b.replace('_', ' ') for b in masked)} carrying "
                   f"non-center content within 3 dB of it" if masked else "")
                + ". What it buys is space — the arrangement reads as the record rather than "
                  "as a backing track, and anything added on top later has somewhere to sit. "
                  "Reported as a difference from the reference, not as a fault"
                + ctx.advice(
                    "; if the lead is meant to be the focus, 2-3 dB is the whole fix"
                )
                + "."
                + _moment_span(moments)
            )
        else:
            detail = (
                f"The center 300 Hz-6 kHz sits {_num(v2i, 1)} dB against everything else "
                f"(A-weighted), {_num(abs(v2i_miss), 1)} dB "
                f"{'below' if buried else 'above'} the {_win(v2i_window, 1, ' dB')} window "
                f"{ctx.profile.label} places a lead vocal in"
                + (f", and {', '.join(b.replace('_', ' ') for b in masked)} carry non-center "
                   f"content within 3 dB of the vocal there" if masked and buried else "")
                + f". The voice test scores {lead_conf:.2f} and places the lead "
                f"{prominence} in the bed. Intelligibility scores {intel:.2f}. This is "
                f"measured from a center estimate rather than a stem, so read it as the "
                f"balance of everything centered, not of the vocal alone."
                + _moment_span(moments)
            )
        hits.append(_Hit(
            Finding(
                id="vocal_balance.buried" if buried else "vocal_balance.too_loud",
                dimension="vocal_balance",
                title=(f"Lead sits under {ctx.ref} — deliberate or not" if gentle
                       else f"Vocal sits under {ctx.ref}" if buried
                       else f"Vocal sits over {ctx.ref}"),
                severity=_severity(ratio),
                # 0.90 against a separated source; against a center estimate that
                # cannot tell a singer from a centered synth, 0.55 scaled by how
                # sure the voice test actually was.
                confidence=(ctx.trust(0.90, "stem") if from_stems
                            else _centre_trust(0.55)),
                detail=detail,
                evidence=evidence,
                band_hz=(300.0, 6000.0),
                moments=moments,
            ),
            ratio,
        ))

    if consistency_miss > 0.0 or intel_miss < 0.0:
        ratio = max(_ratio(consistency_miss, 2.0), _ratio(intel_miss, 0.12))
        # Same cap, same reason. Low intelligibility on a deliberately tucked
        # lead is not a wandering vocal — it is the direct consequence of where
        # the vocal was put, and it is already described by the finding above.
        if tucked:
            ratio = min(ratio, _TUCKED_MAX_RATIO)
        moments = (
            _moments(list(v.buried_moments or []) + list(v.loud_moments or []))
            if centre_valid else []
        )
        parts = []
        if consistency_miss > 0.0:
            source = (
                "the separated vocal's level swings" if consistency_source == "stem"
                else "the vocal level swings"
            )
            parts.append(
                f"{source} {_num(consistency, 1)} dB between its 10th and 90th "
                f"percentile, against the {_num(consistency_max, 1)} dB "
                f"{ctx.profile.label} absorbs"
            )
        if intel_miss < 0.0:
            parts.append(
                f"intelligibility scores {intel:.2f} against a {_num(intel_window[0], 2)} floor"
            )
        detail = (
            f"The lead moves around more than {ctx.ref} holds it: "
            + "; ".join(parts)
            + (f". 1-4 kHz consonant energy is competing with non-center content in "
               f"{', '.join(b.replace('_', ' ') for b in masked)}." if masked else ".")
            + (" Measured on the separated vocal, over the frames it is singing on."
               if consistency_source == "stem" else
               f" Inferred from center extraction at a voice-test confidence of "
               f"{lead_conf:.2f}, so treat the figure as directional.")
            + (" The lead is tucked under the bed here, so some of this is simply where it "
               "was placed rather than the lead failing to hold a level — which is why this "
               "is reported as an observation."
               if tucked else "")
            + _moment_span(moments)
        )
        hits.append(_Hit(
            Finding(
                id="vocal_balance.inconsistent",
                dimension="vocal_balance",
                title=f"Vocal level wanders wider than {ctx.ref}",
                severity=_severity(ratio),
                confidence=(ctx.trust(0.84, "stem") if consistency_source == "stem"
                            else _centre_trust(0.50)),
                detail=detail,
                evidence=evidence,
                band_hz=(300.0, 6000.0),
                moments=moments,
            ),
            ratio,
        ))

    return hits


# ---------------------------------------------------------------------------
# 13. Stereo width
# ---------------------------------------------------------------------------


def _detect_stereo_width(ctx: _Ctx) -> List[_Hit]:
    """Width against the genre's window, and L/R balance.

    A mono source is never reported as a stereo defect — a file that never had a
    stereo field cannot have lost one. Under a polarity inversion the width
    reading is a symptom of that fault, not an independent one, so it is
    suppressed and the dimension cross-references the phase finding instead.
    """
    if ctx.no_programme or ctx.is_mono:
        return []

    st = ctx.m.stereo
    hits: List[_Hit] = []

    width = _fin(st.width, 0.0)
    width_window = ctx.profile.stereo_width
    width_miss = targets.range_miss(width, width_window)

    balance = _fin(st.balance_db, 0.0)
    balance_window = (-1.5, 1.5)
    balance_miss = targets.range_miss(balance, balance_window)

    band_width = {k: _fin(vv) for k, vv in (st.band_width or {}).items()}
    widest = max(band_width, key=lambda k: band_width[k]) if band_width else ""
    narrowest = min(band_width, key=lambda k: band_width[k]) if band_width else ""

    evidence = [
        _ev("Stereo width", width, "", target_range=width_window,
            verdict=_verdict(width, width_window, 0.08),
            detail=f"Side/Mid RMS ratio. {ctx.profile.label} sits at {_win(width_window, 2)}."),
        _ev("Correlation", _fin(st.correlation), "",
            target_range=(_fin(ctx.profile.correlation_min), 1.0),
            verdict=_verdict(_fin(st.correlation),
                             (_fin(ctx.profile.correlation_min), 1.0), 0.35)),
        _ev("L/R balance", balance, "dB", target_range=balance_window,
            verdict=_verdict(balance, balance_window, 1.0),
            detail="Positive is right-heavy."),
        _ev(f"Widest band ({widest.replace('_', ' ') or 'n/a'})",
            band_width.get(widest, 0.0), "", target_range=width_window, verdict="good"),
        _ev("Mono sum loss", _fin(st.mono_sum_loss_db), "dB",
            target_range=(MONO_LOSS_FLOOR_DB, 60.0),
            verdict=_verdict(_fin(st.mono_sum_loss_db), (MONO_LOSS_FLOOR_DB, 60.0), 2.0)),
    ]

    if width_miss != 0.0 and "polarity" not in ctx.tags:
        ratio = _ratio(width_miss, 0.08)
        wide = width_miss > 0.0
        detail = (
            f"This sits {_num(abs(width_miss), 2)} {'wider' if wide else 'narrower'} than "
            f"{ctx.ref}: Side/Mid measures {_num(width, 2)} against the "
            f"{_win(width_window, 2)} window {ctx.profile.label} works in. The widest band is "
            f"{widest.replace('_', ' ') or 'n/a'} at {_num(band_width.get(widest, 0.0), 2)} and "
            f"the narrowest {narrowest.replace('_', ' ') or 'n/a'} at "
            f"{_num(band_width.get(narrowest, 0.0), 2)}, with correlation at "
            f"{_num(st.correlation, 2)} and a {_num(st.mono_sum_loss_db, 1)} dB mono fold-down "
            f"cost."
            + (" A wide field buys scale on headphones; past this window it costs center "
               "density and mono translation without buying any more space."
               if wide else
               " A narrow field buys center weight and translates anywhere; this far in it "
               "costs the separation that gives each element somewhere to sit.")
        )
        hits.append(_Hit(
            Finding(
                id="stereo_width.too_wide" if wide else "stereo_width.too_narrow",
                dimension="stereo_width",
                title=f"{'Wider' if wide else 'Narrower'} than {ctx.ref}",
                severity=_severity(ratio),
                confidence=0.92,   # second-order statistics, directly computed
                detail=detail,
                evidence=evidence,
            ),
            ratio,
        ))

    if balance_miss != 0.0:
        ratio = _ratio(balance_miss, 1.0)
        detail = (
            f"The right channel runs {_num(balance, 2)} dB "
            f"{'hotter' if balance > 0 else 'quieter'} than the left across the whole file, "
            f"outside the {_pm(1.5)} dB a centered image tolerates. A constant offset "
            f"like this is a gain-staging or pan-law problem rather than an arrangement "
            f"choice — it pulls the phantom center off axis for every listener."
        )
        hits.append(_Hit(
            Finding(
                id="stereo_width.channel_imbalance",
                dimension="stereo_width",
                title="Left and right are not level-matched",
                severity=_severity(ratio),
                confidence=0.94,
                detail=detail,
                evidence=evidence,
            ),
            ratio,
        ))

    return hits


# ---------------------------------------------------------------------------
# 14. Transient impact
# ---------------------------------------------------------------------------


def _detect_transients(ctx: _Ctx) -> List[_Hit]:
    """Punch against the genre floor, with attack time and smearing as support."""
    if ctx.no_programme or not ctx.has("transients"):
        return []

    t = ctx.m.transients
    if _fin(t.onset_density, 0.0) <= 0.0:
        # Nothing struck anything. A drone has no transients to lose.
        return []

    punch = _fin(t.punch_index, 0.0)
    punch_window = (_fin(ctx.profile.punch_min, 0.35), 1.0)
    punch_miss = targets.range_miss(punch, punch_window)
    if punch_miss >= 0.0:
        return []

    ratio = _ratio(punch_miss, 0.07)
    band_punch = {k: _fin(v) for k, v in (t.band_punch or {}).items()}
    weakest = min(band_punch, key=lambda k: band_punch[k]) if band_punch else ""
    moments = _moments(t.weak_moments)

    detail = (
        f"The hits land softer than {ctx.ref}: punch scores {punch:.2f} against a "
        f"{_num(punch_window[0], 2)} floor for {ctx.profile.label}. Across "
        f"{int(_fin(t.onset_density) * ctx.duration)} detected onsets the average hit sits "
        f"{_num(t.transient_to_sustain_db, 1)} dB above its own level 50 ms later, with a "
        f"{_num(t.attack_time_ms, 1)} ms 10-90% attack and a smearing index of "
        f"{t.smearing_index:.2f}. The weakest band is "
        f"{weakest.replace('_', ' ') or 'n/a'} at {_num(band_punch.get(weakest, 0.0), 2)}. "
        f"The ear reads the drop after a hit rather than the peak, so a small fall-off buys "
        f"a smoother, more continuous feel and costs the sense that the drums arrived."
        + _moment_span(moments)
    )

    return [_Hit(
        Finding(
            id="transients.no_punch",
            dimension="transients",
            title=f"Hits land softer than {ctx.ref}",
            severity=_severity(ratio),
            confidence=ctx.trust(0.80, "transients"),
            detail=detail,
            evidence=[
                _ev("Punch index", punch, "", target_range=punch_window,
                    verdict=_verdict(punch, punch_window, 0.07),
                    detail="Transient peak against the level 50 ms later."),
                _ev("Transient to sustain", _fin(t.transient_to_sustain_db), "dB",
                    target=None, verdict="problem"),
                _ev("Attack time", _fin(t.attack_time_ms), "ms", target=None,
                    verdict="watch" if _fin(t.attack_time_ms) > 15.0 else "good"),
                _ev("Smearing index", _fin(t.smearing_index), "",
                    target_range=(0.0, 0.5),
                    verdict=_verdict(_fin(t.smearing_index), (0.0, 0.5), 0.15)),
                _ev(f"Weakest band punch ({weakest.replace('_', ' ') or 'n/a'})",
                    band_punch.get(weakest, 0.0), "", target_range=punch_window,
                    verdict="problem"),
                _ev("Onset density", _fin(t.onset_density), "onsets/s", verdict="good"),
            ],
            moments=moments,
        ),
        ratio,
    )]


# ---------------------------------------------------------------------------
# 15. Clarity
# ---------------------------------------------------------------------------


def _detect_clarity(ctx: _Ctx) -> List[_Hit]:
    """Masking and congestion — is the energy arranged badly rather than distributed badly.

    Runs last of the fourteen so it inherits whatever masking pairs the more
    specific detectors did not claim. A source burying the vocal is a vocal
    problem and a source burying the mids is the low-mid buildup; what is left
    over — bass under drums, a pad under a lead — has no better home, and this
    is it.

    The two-track model and the separated one are complementary rather than
    redundant, which is exactly why they are folded into one finding. From a
    spectrum you can see energy that is buried; you cannot see what buried it.
    From stems you can name the masker; you cannot see the bands where two
    parts of the *same* source stack inside one auditory filter. Reporting both
    as two findings would be reporting one problem twice.
    """
    if ctx.no_programme or not ctx.has("clarity"):
        return []

    c = ctx.m.clarity

    clarity = _fin(c.clarity_index, 0.5)
    clarity_window = (_fin(ctx.profile.clarity_min, 0.45), 1.0)
    clarity_miss = targets.range_miss(clarity, clarity_window)

    masking = _fin(c.masking_index, 0.0)
    masking_window, masking_scale = _masking_window(ctx.profile)
    masking_ceiling = masking_window[1]
    masking_miss = targets.range_miss(masking, masking_window)

    pairs = ctx.take_masking(limit=3)
    pair_ratio = max((_mask_ratio(p) for p in pairs), default=0.0)

    if clarity_miss >= 0.0 and masking_miss <= 0.0 and pair_ratio <= 0.0:
        return []

    ratio = min(
        max(_ratio(clarity_miss, 0.07), _ratio(masking_miss, masking_scale), pair_ratio),
        6.0,
    )
    congestion = {k: _fin(v) for k, v in (c.band_congestion or {}).items()}
    worst = c.worst_congested_band or (
        max(congestion, key=lambda k: congestion[k]) if congestion else ""
    )
    moments = _moments(c.congested_moments)
    if not moments and pairs:
        moments = _moments([mo for p in pairs for mo in (p.moments or [])])

    spectrum_says_nothing = clarity_miss >= 0.0 and masking_miss <= 0.0
    if spectrum_says_nothing:
        # The two-track is happy and the stems are not. That is the case the
        # spectrum structurally cannot see: two sources stacked inside one
        # band look like one well-behaved band from outside.
        detail = (
            f"Nothing in the two-track's own spectrum is congested — clarity scores "
            f"{clarity:.2f} against a {_num(clarity_window[0], 2)} floor for "
            f"{ctx.profile.label} and {masking * 100:.1f}% of its audible loudness sits under "
            f"its own masking threshold, inside the {masking_ceiling * 100:.1f}% ceiling. "
            f"Separating the sources shows why that is not the whole story — two of them "
            f"stacked inside one auditory filter read as one well-behaved band from "
            f"outside."
            + _mask_sentence(pairs, "Measured against each other:")
            + " Both elements are present and neither is separately audible."
            + _moment_span(moments)
        )
        title = "One source is burying another"
        finding_id = "clarity.stem_masking"
        confidence = ctx.trust(0.80, "stem")
    else:
        detail = (
            f"Elements overlap more here than {ctx.ref} does. Clarity scores {clarity:.2f} "
            f"against a {_num(clarity_window[0], 2)} floor for {ctx.profile.label}, with "
            f"{masking * 100:.1f}% of the mix's audible loudness sitting under its own "
            f"masking threshold against a {masking_ceiling * 100:.1f}% ceiling. The worst "
            f"band is {worst.replace('_', ' ') or 'n/a'} at "
            f"{congestion.get(worst, 0.0):.2f} congestion, and transient frames sit "
            f"{_num(c.definition_db, 1)} dB above sustained ones. Stacking energy buys "
            f"density and a wall of sound; it costs separation, so elements are present "
            f"without being separately audible."
            + _mask_sentence(pairs)
            + _moment_span(moments)
        )
        title = f"Elements overlap more than {ctx.ref}"
        finding_id = "clarity.congested"
        # A psychoacoustic spreading model over a measured spectrum, and with
        # separated sources naming the masker it is no longer only a model.
        confidence = ctx.trust(0.86 if pairs else 0.72, "clarity")

    return [_Hit(
        Finding(
            id=finding_id,
            dimension="clarity",
            title=title,
            severity=_severity(ratio),
            confidence=confidence,
            detail=detail,
            evidence=_mask_evidence(pairs) + [
                _ev("Clarity index", clarity, "", target_range=clarity_window,
                    verdict=_verdict(clarity, clarity_window, 0.07)),
                _ev("Masking index", masking, "", target_range=masking_window,
                    verdict=_verdict(masking, masking_window, masking_scale),
                    detail="Loudness-weighted share of the mix sitting under its own "
                           "masking threshold, from an ERB/spreading-function model."),
                _ev(f"Worst band congestion ({worst.replace('_', ' ') or 'n/a'})",
                    congestion.get(worst, 0.0), "", target_range=(0.0, 0.5),
                    verdict=_verdict(congestion.get(worst, 0.0), (0.0, 0.5), 0.15)),
                _ev("Definition", _fin(c.definition_db), "dB", target=None,
                    verdict="watch" if _fin(c.definition_db) < 4.0 else "good",
                    detail="Transient frames against sustained frames."),
                _ev("Spectral contrast", _fin(c.spectral_contrast), "dB", verdict="good"),
            ],
            moments=moments,
        ),
        ratio,
    )]


# ---------------------------------------------------------------------------
# Public API: detect_all
# ---------------------------------------------------------------------------

# Order matters. Specific detectors run before general ones so the general ones
# can stand down (`ctx.tags`) or find their evidence already claimed
# (`ctx.take_masking`):
#
#   phase        before stereo width and the low-end mono check
#   arrangement  before dynamic range — it describes the same flatness with the
#                section, the timestamp and the lift in LU attached
#   vocal        before mud and clarity — a source burying the vocal is a vocal
#                problem, and that is where a producer will look for it
#   mud          before clarity — a masking pair in the low mids is the low-mid
#                buildup seen from the other side
#   mud / harshness / low end   before frequency balance
#   clarity      last, so it inherits whatever masking pairs are left over
_DETECTORS = (
    _detect_clipping,
    _detect_phase,
    _detect_loudness,
    _detect_limiter,
    _detect_arrangement,
    _detect_dynamic_range,
    _detect_compression,
    _detect_stem_compression,
    _detect_low_end,
    _detect_vocal,
    _detect_mud,
    _detect_harshness,
    _detect_frequency_balance,
    _detect_stereo_width,
    _detect_transients,
    _detect_clarity,
)

_SEVERITY_RANK: Dict[str, int] = {"critical": 0, "major": 1, "minor": 2, "clean": 3}


def _rank_key(hit: _Hit) -> Tuple[int, int, float]:
    """Severity first, then defects before deviations, then recoverable points.

    Ranking the last key by `impact` rather than by the raw miss ratio is
    deliberate. The ratio says how many tolerance units a number is out by,
    which is not the same question as how much the listener gets back — a
    30 dB hole in the air band is a larger ratio than a clipped master and a
    smaller problem. `_IMPACT_WEIGHT` is where "how much this matters" lives,
    so the ordering the user sees is driven by it.

    Defects sort above deviations of equal severity because that is the order
    to work in: fix what is broken before reconsidering what was chosen. This
    is the *only* place that preference is expressed — it used to be baked into
    `impact` itself, which quietly made the recoverable-points figure wrong.
    """
    return (
        _SEVERITY_RANK.get(hit.finding.severity, 3),
        0 if hit.finding.kind == "defect" else 1,
        -_fin(hit.finding.impact),
    )


def _cap(hits: List[_Hit], limit: int = MAX_FINDINGS) -> List[_Hit]:
    """Worst first, capped, with every dimension keeping its own worst finding.

    A naive top-N drops a dimension's only finding to make room for a second
    finding of a louder dimension, which leaves that dimension scored as clean
    when it is not. So the first pass takes one per dimension and the second
    fills whatever room is left.

    Observations sort last (severity "clean", no impact) and so are the first
    thing squeezed out when there are real problems to report — which is the
    right order: nobody needs to be congratulated on their mid-range while
    twelve other things are outstanding.
    """
    ordered = sorted(hits, key=_rank_key)
    primary: List[_Hit] = []
    extra: List[_Hit] = []
    seen: set = set()
    for hit in ordered:
        if hit.finding.dimension in seen:
            extra.append(hit)
        else:
            seen.add(hit.finding.dimension)
            primary.append(hit)

    kept = primary[:limit]
    for hit in extra:
        if len(kept) >= limit:
            break
        kept.append(hit)
    return sorted(kept, key=_rank_key)


def _assign_impact(hits: List[_Hit], rescale: bool = True) -> None:
    """Health-score points recoverable, capped so the set can never sum past 100.

    Per finding: the dimension's ceiling weight, scaled by how far outside the
    window the value sits (a fully critical miss earns the whole weight), then
    discounted for repeats within a dimension because the second finding there
    is mostly the same fix.

    Called twice. The first pass is unscaled and exists only to rank and select
    (`_cap` needs an impact to sort by); the second runs on the surviving set
    and applies `_IMPACT_TOTAL_CAP`, so dropped findings cannot deflate the
    points attributed to the ones the user actually sees.

    A deviation's points are scaled by `_DEVIATION_IMPACT_WEIGHT`, matching the
    pull it has on the health score. `impact` is documented as "health-score
    points recoverable", so the two have to agree: if closing a stylistic gap
    only moves the score by 40% of what closing a defect does, then that is the
    number of points on offer, and `ceiling_score` must not promise more. It
    also puts defects above deviations of equal severity in the ranking, which
    is the order a producer should read them in.
    """
    per_dimension: Dict[str, int] = {}
    raw: List[float] = []
    for hit in sorted(hits, key=lambda h: -h.ratio):
        dim = hit.finding.dimension
        if hit.observation:
            # Nothing to recover: this is not a problem.
            hit.finding.impact = 0.0
            continue
        n = per_dimension.get(dim, 0)
        per_dimension[dim] = n + 1
        discount = _REPEAT_DISCOUNT[min(n, len(_REPEAT_DISCOUNT) - 1)]
        weight = _IMPACT_WEIGHT.get(dim, 5.0)
        hit.finding.impact = round(
            _clamp(
                weight * _clamp(hit.ratio / _CRITICAL_AT, 0.18, 1.0) * discount,
                0.0, 100.0,
            ),
            2,
        )
        raw.append(hit.finding.impact)

    total = sum(raw)
    if rescale and total > _IMPACT_TOTAL_CAP:
        scale = _IMPACT_TOTAL_CAP / total
        for hit in hits:
            hit.finding.impact = round(_clamp(hit.finding.impact * scale, 0.0, 100.0), 2)


def _suppressed_by_intent(ctx: _Ctx, finding: Finding) -> bool:
    """Is this finding a category error against what the file is?

    The gates that cannot be expressed inside a single detector, because they
    are about whole dimensions rather than one measurement. Everything here
    removes a finding that would be *wrong* on this material, not one that is
    merely unwelcome.
    """
    fid = str(finding.id)
    dim = str(finding.dimension)
    intent = ctx.intent

    # No lead on the record: nothing about a lead's balance can be true. The
    # positive topline observation a beat produces is exempt — it is the one
    # vocal-dimension statement that is *about* the absence.
    if dim == "vocal_balance" and intent in _NO_LEAD_INTENTS:
        return fid != "vocal_balance.topline_headroom"

    if intent == "stem":
        # One element in isolation. Whole-mix balance is meaningless against it:
        # a bass stem is all low end, a vocal stem has no drums to collide with,
        # and neither has a "mix" to be muddy or congested. The measurements
        # still appear — under Details, and in every dimension's evidence — the
        # verdicts do not.
        if dim in _STEM_SUPPRESSED_DIMENSIONS or fid in _STEM_SUPPRESSED_IDS:
            return True

    if intent == "demo":
        # A rough. Where it lands on the loudness ladder and how its limiter
        # behaves are questions about a master nobody has attempted.
        if fid in _DEMO_SUPPRESSED_IDS or dim in _DEMO_SUPPRESSED_DIMENSIONS:
            return True

    if intent == "beat" and fid in _BEAT_SUPPRESSED_IDS:
        # The headroom a beat is carrying is the room the topline needs.
        return True

    return False


def _as_observation(finding: Finding) -> None:
    """Strip a finding of its verdict, leaving the measurement and the sentence.

    What `intent="reference"` does to everything. Somebody measuring a record
    they admire is asking what it does, and a released master is not a work
    item — so it keeps every number and loses the severity, the recoverable
    points and any clause telling anyone to change it (those are already absent:
    see `_Ctx.advice`).
    """
    finding.severity = "clean"
    finding.impact = 0.0


def detect_all(
    m: Measurements, genre: str, intent: str = "full_mix"
) -> List[Finding]:
    """Turn a `Measurements` into evidence-backed `Finding`s for one genre.

    Every finding carries the figure that produced it, the window it missed,
    and a `detail` sentence that is complete without any AI layer. Nothing is
    emitted for a file with no measurable program, and nothing is emitted for
    a measurement whose value sits inside this genre's window.

    Two things happen to every finding on the way out, and they are the reason
    a stylistic choice can no longer be reported as damage:

    1. **It is classified.** `finding_kind` splits the list into defects — wrong
       in any genre, for any artist, at any intent — and deviations, which are
       measured differences from a reference somebody may well have chosen.
    2. **A deviation's severity is re-derived under the deviation ceiling.** It
       caps at major and only reaches it a long way outside the window.
       "critical" is reserved for things that are actually broken.

    `intent` then removes what is a category error against this file: no vocal
    findings on a beat, no mix balance on a single stem, no mastering verdicts
    on a rough, and no instructions at all on somebody else's record.
    """
    ctx = _build_ctx(m, genre, intent)
    if ctx.no_programme:
        return []

    hits: List[_Hit] = []
    for detector in _DETECTORS:
        try:
            hits.extend(detector(ctx))
        except Exception:
            # One detector failing must not cost the user the other thirteen.
            # The dimension simply scores as unassessed.
            continue

    # Classify, then re-derive severity under the rule for that class. Done here
    # rather than in each detector so the two can never drift apart: a detector
    # decides how far outside the window its number is, and nothing else.
    for hit in hits:
        hit.finding.kind = finding_kind(hit.finding.id)
        # Carry the magnitude out with the finding. The severity label buckets it
        # away, and the deviation scorer needs it back.
        hit.finding.miss_ratio = round(max(_fin(hit.ratio, 0.0), 0.0), 4)
        if hit.observation:
            _as_observation(hit.finding)
        elif hit.finding.kind == "deviation":
            hit.finding.severity = _severity_for("deviation", hit.ratio)

    hits = [h for h in hits if not _suppressed_by_intent(ctx, h.finding)]

    # On a reference nothing is a work item. The findings stay — they are what
    # the record does, which is the whole reason somebody uploaded it — and the
    # verdicts come off.
    if ctx.is_reference:
        for hit in hits:
            _as_observation(hit.finding)
            hit.observation = True

    # Drop misses too small to state honestly at the precision we report in.
    # Observations are exempt: they are not a miss at all.
    hits = [h for h in hits if h.observation or h.ratio >= MIN_REPORTABLE_RATIO]

    _assign_impact(hits, rescale=False)   # gives _cap something meaningful to rank by
    hits = _cap(hits)
    _assign_impact(hits)                  # final points, capped across the kept set

    findings = [hit.finding for hit in sorted(hits, key=_rank_key)]

    # Last, because it needs the finished article: the classification (a defect
    # may never be asked about), the final impact (which decides which four
    # questions are worth a user's time) and the evidence rows the questions
    # quote their figures from. Detectors that wrote their own question keep it;
    # `clarify` fills the rest and refuses to leave one on a defect.
    return clarify.attach(findings, ctx.profile.label, ask=not ctx.is_reference)


# ---------------------------------------------------------------------------
# Public API: score_dimensions
# ---------------------------------------------------------------------------

# Points removed from a dimension by its worst finding, plus a smaller amount
# for each additional one (a second problem in the same dimension is usually
# the same session's work).
_PENALTY: Dict[str, float] = {"critical": 58.0, "major": 34.0, "minor": 15.0}
_REPEAT_PENALTY = 8.0
_MAX_REPEAT_PENALTY = 16.0

# A deviation's penalty is a *curve*, not a table lookup — and the curve is the
# reason the severity cap does not cost the score its genre signal.
#
# Two separate things get called "how bad is this", and conflating them is what
# went wrong:
#
#   * The **label** is what a producer reads. Calling a stylistic choice
#     "critical" tells them their record is broken when it is simply not the
#     genre average, and that is the complaint this whole split exists to stop.
#     So a deviation's label caps at "major", always. See `_severity_for`.
#
#   * The **penalty** is what the health score is made of, and the health score
#     answers a different question: how far is this file from a finished record
#     of this kind? Stylistic distance belongs in that answer. A trap master
#     measured against an ambient reference *is* a long way from ambient, and a
#     number that refuses to say so is not being kind, it is being useless.
#
# So the label is capped and the penalty is not. What the deviation split
# actually buys is the wording, the absent imperatives, and the fact that
# nothing stylistic is ever stamped "critical" — not a flattened score.
#
# The curve carries the same three values the defect table steps through, so a
# deviation and a defect at comparable distance cost comparable points, and a
# deviation at 4 tolerance units can no longer cost the same as one at 0.16.
#
# On where the anchors sit. A step table applies one value across a whole band;
# a curve through the *band entry* would therefore over-charge everything above
# it, because the table stays flat where the curve keeps climbing. So each
# anchor places the table's value at a representative ratio *inside* its own
# band — 1.0 within minor's [0, 1.5), 1.6 within major's [1.5, 3.0), 3.75 within
# critical's [3.0, ∞).
#
# The exact positions are calibration, not derivation: they were fitted to
# reproduce the catalog behavior the step table already had, against
# reference_trap at trap/pop/rock/folk/ambient, reference_pop, reference_folk
# and mix_problem. That fit holds every one of those to within 2.8 points while
# making the score continuous in the miss. `tools/check_regressions.py` is the
# executable version of that claim; re-run it if you touch these.
_PENALTY_ANCHORS: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.0),
    (1.0, _PENALTY["minor"]),
    (1.6, _PENALTY["major"]),
    (3.75, _PENALTY["critical"]),
)


def deviation_penalty(ratio: float) -> float:
    """Points off a dimension for a deviation `ratio` tolerance units out.

    Piecewise-linear through `_PENALTY_ANCHORS` and flat above the last one, so
    it agrees with the defect table at the band boundaries but stays continuous
    between them. Continuity is the point: two mixes differing only in how far
    they sit from the reference must not score identically.
    """
    r = max(_fin(ratio, 0.0), 0.0)
    lo_r, lo_p = _PENALTY_ANCHORS[0]
    for hi_r, hi_p in _PENALTY_ANCHORS[1:]:
        if r <= hi_r:
            span = hi_r - lo_r
            if span <= 0.0:
                return hi_p
            return lo_p + (hi_p - lo_p) * (r - lo_r) / span
        lo_r, lo_p = hi_r, hi_p
    return lo_p


def _finding_penalty(finding: Finding) -> float:
    """Points off a dimension for one finding, by kind.

    Defects keep the severity table: the bands are a fair summary of damage and
    they are what the fixture baselines were calibrated against. Deviations read
    their measured distance instead. A deviation with no recorded distance (an
    older payload, or a detector that never set one) falls back to its band, so
    it can never score as harmless by accident.
    """
    severity = str(finding.severity)
    if severity == "clean":
        return 0.0
    if str(getattr(finding, "kind", "deviation")) == "defect":
        return _PENALTY.get(severity, 15.0)
    ratio = _fin(getattr(finding, "miss_ratio", 0.0), 0.0)
    if ratio <= 0.0:
        return _PENALTY.get(severity, 15.0)
    return deviation_penalty(ratio)


def scoring_grade(finding: Finding) -> Severity:
    """The band a finding's *cost* puts it in, ignoring the label it displays.

    A deviation's displayed severity caps at "major" so nothing stylistic is
    ever stamped "critical" at a producer. But the compound penalty and the
    score ceiling in `engine.compute_health` are scoring machinery, not wording,
    and keying them on the displayed label quietly switched off the critical
    ceiling for every deviation — which is most of how the cross-genre spread
    was produced in the first place.

    So the display reads `Finding.severity` and the scorer reads this. They
    agree for defects and for anything under the major threshold; they differ
    exactly where the cap bites, which is the one place they should.
    """
    penalty = _finding_penalty(finding)
    if penalty >= _PENALTY["critical"]:
        return "critical"
    if penalty >= _PENALTY["major"]:
        return "major"
    if penalty > 0.0:
        return "minor"
    return "clean"


# Score for a dimension with nothing to report but nothing to praise either —
# a measurement that could not be made rather than one that came back healthy.
_UNASSESSED_SCORE = 90.0

# Score for a dimension whose only entry is an observation: a confirmed virtue
# rather than a silence. Above the unassessed figure because something *was*
# measured and it came back the way it should, and under 100 because one
# confirmed thing is not the same as a whole dimension with nothing to say.
_OBSERVED_SCORE = 95.0


def _comfort(value: float, window: Tuple[float, float]) -> float:
    """0-1: how centrally a value sits inside its window. Edge = 0, middle = 1."""
    lo, hi = float(window[0]), float(window[1])
    if hi <= lo:
        return 0.5
    mid = 0.5 * (lo + hi)
    return _clamp(1.0 - 2.0 * abs(_fin(value, mid) - mid) / (hi - lo), 0.0, 1.0)


def _headroom_comfort(value: float, edge: float, scale: float) -> float:
    """0-1 for a one-sided window: how far inside the edge the value sits."""
    return _clamp(abs(_fin(value, edge) - edge) / max(scale, 1e-9), 0.0, 1.0)


def _elsewhere(
    what: str, value: float, unit: str, window: Tuple[float, float],
    genre_label: str, owner: str, nd: int = 1,
) -> Tuple[str, float, bool]:
    """Headline for a dimension with no findings whose figure is still out of window.

    A dimension can end up empty because a more specific detector claimed the
    problem (a polarity inversion owns the low end's mono ratio) or because the
    miss sits under the significance floor. Either way the clean headline must
    not turn that silence into praise, so it states the figure, states the
    window it is outside, and names where the problem is actually reported.
    """
    return (
        f"{what} measures {_num(value, nd)}{unit}, outside {genre_label}'s "
        f"{_win(window, nd, unit)} window — reported under {owner}, not here.",
        0.0,
        False,
    )


def _intent_silence(dim: str, ctx: _Ctx) -> Optional[str]:
    """The headline for a dimension this file's intent took off the table.

    Returns None when the dimension is in play as normal. When it is not, the
    sentence says which question was not asked and why — never "clean", because
    a question that was never put has no healthy answer.
    """
    intent = ctx.intent
    label = ctx.profile.label

    if dim == "vocal_balance" and intent in _NO_LEAD_INTENTS:
        if intent == "beat":
            return (
                "No lead to balance: this is a beat, so the topline is somebody else's and "
                "the space where it goes is the point. Nothing about a missing or tucked "
                "vocal is scored against this file."
            )
        if intent == "instrumental":
            return ("No lead to balance: this is an instrumental and is complete without "
                    "one. Vocal balance is not a question here.")
        if intent == "stem":
            return ("No lead to balance: a single stem has no mix for a vocal to sit in.")
        return (
            "Vocal balance is reported as a measurement rather than a verdict on a "
            "reference track — see Details for the center-channel figures."
        )

    if intent == "stem" and dim in _STEM_SUPPRESSED_DIMENSIONS:
        return (
            f"Not judged on a single stem: {DIMENSION_LABELS.get(dim, dim).lower()} is a "
            f"property of a mix, and one element in isolation has no mix to have it. A bass "
            f"stem is supposed to be all low end and a vocal stem is supposed to have nothing "
            f"under it — the {label} curve has nothing to say about either. The measurements "
            f"are still in Details."
        )

    if intent == "demo" and dim in _DEMO_SUPPRESSED_DIMENSIONS:
        return (
            "Not judged on a rough: how the ceiling is being reached is a question about a "
            "master, and there isn't one yet. The measurements are still in Details."
        )

    return None


def _clean_report(dim: str, ctx: _Ctx) -> Tuple[str, float, bool]:
    """(headline, comfort 0-1, assessed) for a dimension with no findings.

    The headline says what is *good* and cites the figure that makes it good,
    because "no issues found" tells a producer nothing they can act on or trust.
    Where a measurement genuinely could not be made — a mono file's stereo
    field, a vocal that is not there, a 1.2 s file's loudness range — it says
    that instead of claiming health it cannot support.

    A dimension that was *deliberately not asked about* is a third case, and it
    says so: a beat has no lead to balance, a bass stem has no mix to be muddy,
    a rough has no master to judge. Reporting those as clean would be praising
    a question nobody put.
    """
    m = ctx.m
    p = ctx.profile
    label = p.label

    if ctx.no_programme:
        return ("No measurable program in this file — nothing assessed.",
                0.0, False)

    silent = _intent_silence(dim, ctx)
    if silent is not None:
        return (silent, 0.0, False)

    if dim == "clipping":
        tp = _fin(m.clipping.true_peak_dbtp, -120.0)
        head = TRUE_PEAK_CEILING_DBTP - tp
        return (
            f"No clipped samples and no inter-sample overs; true peak sits at {_num(tp, 2)} dBTP, "
            f"{_num(head, 1)} dB under the -1.0 dBTP delivery ceiling.",
            _headroom_comfort(tp, TRUE_PEAK_CEILING_DBTP, 6.0), True,
        )

    if dim == "phase":
        if ctx.is_mono:
            return ("Mono source: it folds down perfectly because there is nothing to fold. "
                    "No phase relationship exists to break.", 1.0, False)
        corr = _fin(m.phase.correlation, 1.0)
        loss = _fin(m.phase.mono_sum_loss_db, 0.0)
        return (
            f"Mono-compatible: correlation {corr:+.2f} against a {_num(p.correlation_min, 2)} "
            f"floor for {label}, and summing to mono costs only {_num(loss, 2)} dB.",
            _headroom_comfort(corr, _fin(p.correlation_min), 0.5), True,
        )

    if dim == "loudness":
        if not ctx.has("loudness"):
            return (f"Not assessed: {_num(ctx.duration, 1)} s is shorter than the gating "
                    f"window integrated loudness needs.", 0.0, False)
        lufs = _fin(m.loudness.integrated_lufs, -70.0)
        window = p.integrated_lufs
        if targets.in_range(lufs, window):
            return (
                f"{_num(lufs, 1)} LUFS lands inside the {_win(window, 1, ' LUFS')} window "
                f"{label} releases sit in, with {_num(m.loudness.plr_db, 1)} dB of "
                f"peak-to-loudness in hand.",
                _comfort(lufs, window), True,
            )
        head = TRUE_PEAK_CEILING_DBTP - _fin(m.loudness.true_peak_dbtp, -120.0)
        return (
            f"Quiet at {_num(lufs, 1)} LUFS but not stuck: {_num(head, 1)} dB of clean gain "
            f"reaches {_num(lufs + head, 1)} LUFS, inside {label}'s "
            f"{_win(window, 1, ' LUFS')} window, with no limiting needed.",
            0.6, True,
        )

    if dim == "limiter":
        if not ctx.has("limiter"):
            return (f"Not assessed: {_num(ctx.duration, 1)} s is too short to judge how the "
                    f"ceiling is being reached.", 0.0, False)
        psr = _fin(m.loudness.psr_p10_db, 0.0)
        runs = int(_fin(m.clipping.flat_run_count))
        clean_ceiling = (
            f"{runs} flat-topped runs and a distortion index of "
            f"{_fin(m.clipping.distortion_index):.3f}"
        )
        if psr > p.psr_p10_db[1]:
            # More peak headroom than the genre's masters keep. That is not a
            # limiter fault — there is barely a limiter — and dynamic_range
            # owns the observation.
            return (
                f"Barely limited: short-term PSR holds {_num(psr, 1)} dB, above "
                f"{label}'s {_win(p.psr_p10_db, 1, ' dB')} window, with {clean_ceiling}. "
                f"Nothing is being crushed; how far under genre level that leaves the mix "
                f"is reported under Dynamic Range.",
                0.5, True,
            )
        return (
            f"The ceiling is being held rather than generated: short-term PSR holds "
            f"{_num(psr, 1)} dB against {_win(p.psr_p10_db, 1, ' dB')} for {label}, with "
            f"{clean_ceiling}.",
            _comfort(psr, p.psr_p10_db), True,
        )

    if dim == "dynamic_range":
        if not ctx.has("dynamics"):
            return (f"Not assessed: {_num(ctx.duration, 1)} s is too short for a dynamic-range "
                    f"statistic to settle.", 0.0, False)
        crest = _fin(m.dynamics.crest_factor_db, 0.0)
        if not targets.in_range(crest, p.crest_factor_db):
            return (
                f"Crest factor is {_num(crest, 1)} dB and TT-DR {_num(m.dynamics.dr_value, 1)}, "
                f"outside {label}'s {_win(p.crest_factor_db, 1, ' dB')} window but by too "
                f"little, or at too healthy a level, to be worth a finding.",
                0.2, True,
            )
        return (
            f"Level moves the way {label} needs it to: {_num(crest, 1)} dB crest factor and "
            f"TT-DR {_num(m.dynamics.dr_value, 1)}, inside the "
            f"{_win(p.crest_factor_db, 1, ' dB')} window.",
            _comfort(crest, p.crest_factor_db), True,
        )

    if dim == "compression":
        if not ctx.has("dynamics"):
            return (f"Not assessed: {_num(ctx.duration, 1)} s is too short to measure crest "
                    f"inside the hits.", 0.0, False)
        micro = _fin(m.dynamics.micro_dynamics_db, 0.0)
        inside = targets.in_range(micro, p.micro_dynamics_db)
        return (
            f"The transient survives the processing: {_num(micro, 1)} dB of crest in each "
            f"50 ms frame, "
            + (f"inside {label}'s " if inside else f"above {label}'s ")
            + f"{_win(p.micro_dynamics_db, 1, ' dB')} window, with an estimated "
            f"{_num(m.dynamics.gain_reduction_estimate_db, 1)} dB of gain reduction.",
            _comfort(micro, p.micro_dynamics_db) if inside else 0.6, True,
        )

    if dim == "frequency_balance":
        inside = [b for b in ctx.bands.values() if b.miss_db == 0.0]
        outside = sorted((b for b in ctx.bands.values() if b.miss_db != 0.0),
                         key=lambda b: -b.ratio)
        if outside:
            names = ", ".join(b.label for b in outside[:3])
            return (
                f"{len(inside)} of {len(ctx.bands)} macro bands land inside their "
                f"{label} tolerance; the deviations in {names} are reported under the "
                f"dimension that owns them.",
                0.35, True,
            )
        worst = max(ctx.bands.values(), key=lambda b: abs(b.deviation_db)) if ctx.bands else None
        return (
            f"All {len(ctx.bands)} macro bands sit inside their {label} tolerance"
            + (f"; the largest deviation is {worst.label} at {_num(worst.deviation_db, 1)} dB "
               f"against a {_pm(worst.tolerance_db)} dB allowance." if worst else "."),
            _clamp(1.0 - (abs(worst.deviation_db) / max(worst.tolerance_db, 1e-9)
                          if worst else 0.0), 0.0, 1.0),
            True,
        )

    if dim == "mud":
        mud = _fin(m.spectral.mud_ratio_db, 0.0)
        if mud < p.mud_ratio_db[0]:
            return (
                f"Nothing muddy here — if anything the opposite: 150-400 Hz sits "
                f"{_num(mud, 1)} dB against 60-120 Hz, under {label}'s "
                f"{_win(p.mud_ratio_db, 1, ' dB')} window, so the mix may be missing body "
                f"rather than carrying too much. Band levels are reported under "
                f"Frequency Balance.",
                0.5, True,
            )
        return (
            f"The low mids stay out of the way: 150-400 Hz sits {_num(mud, 1)} dB against "
            f"60-120 Hz, inside {label}'s {_win(p.mud_ratio_db, 1, ' dB')} window, with "
            f"{_num(m.spectral.boxiness_db, 1)} dB of 300-600 Hz boxiness.",
            _comfort(mud, p.mud_ratio_db), True,
        )

    if dim == "harshness":
        harsh = _fin(m.spectral.harshness_index, 0.0)
        # The 5-9 kHz figure is named for whatever is actually making it, on the
        # same rule the finding uses. Calling a beat's hat pattern "sibilance"
        # in a clean headline is the same error as calling it that in a finding.
        vocal_top = ctx.lead_is_up_front
        top_ceiling = (_fin(p.sibilance_max) if vocal_top
                       else _fin(p.sibilance_max) * _PERCUSSION_SIB_FACTOR)
        return (
            f"No edge in the ear's most sensitive region: harshness scores {harsh:.2f} against "
            f"a {_num(p.harshness_max, 2)} ceiling for {label}, and 5-9 kHz burstiness "
            f"{_fin(m.spectral.sibilance_index):.2f} against {_num(top_ceiling, 2)}"
            + (" — sibilance, with a lead vocal up front to own it." if vocal_top else
               " — hats and percussion, with no lead vocal up front, which is why that "
               "ceiling is the wider one."),
            _headroom_comfort(harsh, _fin(p.harshness_max), _fin(p.harshness_max)), True,
        )

    if dim == "low_end":
        le = m.low_end
        mono = _fin(le.low_end_mono_ratio, 1.0)
        mono_window = (_fin(p.low_end_mono_min, 0.80), 1.0)
        if not ctx.is_mono and targets.range_miss(mono, mono_window) < 0.0:
            return (
                f"Only {mono * 100:.0f}% of the energy below 120 Hz is mono, against a "
                f"{mono_window[0] * 100:.0f}% floor for {label}. That is the channels being "
                f"out of polarity rather than an independent low-end fault — it is reported "
                f"under Phase & Mono Compatibility, and fixing that fixes this.",
                0.0, False,
            )
        base = (
            f"{mono * 100:.0f}% of the energy below 120 Hz is mono against a "
            f"{_fin(p.low_end_mono_min) * 100:.0f}% floor, and sub energy is "
            f"{_num(le.sub_energy_db, 1)} dB inside {label}'s "
            f"{_win(p.sub_energy_db, 1, ' dB')} window"
        )
        comfort = _comfort(_fin(le.sub_energy_db), p.sub_energy_db)

        if bool(le.kick_detected) and ctx.has("kick"):
            return (
                f"Low end is tight and mono below 120 Hz: {base}, and the kick at "
                f"{_num(le.kick_fundamental_hz, 0)} Hz clears the bass at "
                f"{_num(le.bass_fundamental_hz, 0)} Hz with "
                f"{_num(le.kick_bass_collision_db, 1)} dB of overlap against a "
                f"{_num(p.kick_bass_collision_max_db, 1)} dB ceiling.",
                comfort, True,
            )
        if bool(le.kick_detected):
            # A kick was found but the file is too short to judge how it sits
            # against the bass. Saying "tight" here would be praising a
            # measurement that was never scored.
            return (
                f"The bottom is clean where it could be measured: {base}. Kick-versus-bass "
                f"separation is not assessed — {_num(ctx.duration, 1)} s is too few hits to "
                f"compare the at-onset and between-onset spectra.",
                comfort * 0.5, True,
            )
        return (
            f"No kick pattern to separate from the bass, and the bottom is clean: {base}.",
            comfort, True,
        )

    if dim == "vocal_balance":
        v = m.vocal
        stem = ctx.stem("vocals")
        if ctx.has_stems and stem is not None and ctx.stems.vocal_to_instrument_db is not None:
            v2i = _fin(ctx.stems.vocal_to_instrument_db)
            spread = _stem_level_spread(stem)
            return (
                f"The lead holds its place, measured on the separated vocal rather than the "
                f"center channel: it sits {_num(v2i, 1)} dB against drums, bass and everything "
                f"else summed, inside {label}'s {_win(p.vocal_to_instrument_db, 1, ' dB')} "
                f"window, and sings on {_fin(stem.active_ratio) * 100:.0f}% of the track"
                + (f" with {_num(spread, 1)} dB of level spread while it does."
                   if spread is not None else "."),
                _comfort(v2i, p.vocal_to_instrument_db), True,
            )
        if ctx.stems.available and stem is None:
            return (
                "Separation found no vocal source in this file, so there is no lead to "
                "balance. That is a measurement rather than an inference — the vocals stem "
                "came back at the separator's artefact floor"
                + (f", which is unusual for {label}." if p.vocal_expected else "."),
                # Absence is not health: there is nothing here to score well.
                1.0, False,
            )
        if ctx.is_mono:
            return ("Mono source: center extraction is meaningless without a side channel, so "
                    "vocal balance is not assessed.", 0.0, False)
        if not ctx.has("vocal"):
            return (f"Not assessed: {_num(ctx.duration, 1)} s is too short to test for the "
                    f"syllabic modulation that identifies a voice.", 0.0, False)
        if not ctx.lead_on_record:
            if not p.vocal_expected:
                return (
                    f"No lead vocal, which is how {label} usually works — "
                    f"{_fin(v.center_energy_ratio) * 100:.0f}% of the 300 Hz-6 kHz energy is "
                    f"centered but it does not carry the syllabic modulation and the "
                    f"consonant-against-vowel swing a voice does.",
                    1.0, False,
                )
            return (
                f"No sustained lead vocal detected: {_fin(v.center_energy_ratio) * 100:.0f}% of "
                f"300 Hz-6 kHz is centered, but the voice test scores only "
                f"{ctx.lead_confidence:.2f} against a {_num(0.55, 2)} bar — the center does "
                f"not modulate at 2-8 Hz any more than the sides do, or it holds one fixed "
                f"harmonic shape the way a synth does and speech does not. Vocal balance is "
                f"not assessed rather than guessed.",
                0.0, False,
            )
        if ctx.lead_prominence == "tucked":
            return (
                f"There is a lead here and it is deliberately under the bed: the center sits "
                f"{_num(v.vocal_to_instrument_db, 1)} dB against everything else, below the "
                f"{_win(p.vocal_to_instrument_db, 1, ' dB')} window {label} usually places a "
                f"lead in, at a voice-test confidence of {ctx.lead_confidence:.2f}. That is "
                f"the correct answer for a beat somebody is going to rap over and a normal "
                f"one for shoegaze or lo-fi, so it is reported as a difference from the "
                f"reference rather than scored as a fault.",
                # Not health, and not damage either: a decision we decline to grade.
                0.5, False,
            )
        return (
            f"The lead holds its place: center sits {_num(v.vocal_to_instrument_db, 1)} dB "
            f"against everything else, inside {label}'s "
            f"{_win(p.vocal_to_instrument_db, 1, ' dB')} window, with "
            f"{_num(v.consistency_db, 1)} dB of level spread and intelligibility at "
            f"{_fin(v.intelligibility_index):.2f}.",
            _comfort(_fin(v.vocal_to_instrument_db), p.vocal_to_instrument_db), True,
        )

    if dim == "stereo_width":
        if ctx.is_mono:
            return ("Mono source: there is no stereo field to be too wide or too narrow, and "
                    "nothing here is a defect.", 1.0, False)
        width = _fin(m.stereo.width, 0.0)
        inside = targets.in_range(width, p.stereo_width)
        return (
            (f"Width is right for the genre: Side/Mid measures {_num(width, 2)} inside "
             if inside else
             f"Width is within rounding of the genre: Side/Mid measures {_num(width, 2)} "
             f"against ")
            + f"{label}'s {_win(p.stereo_width, 2)} window, with L/R matched to "
            f"{_num(m.stereo.balance_db, 2)} dB and a {_num(m.stereo.mono_sum_loss_db, 2)} dB "
            f"fold-down cost.",
            _comfort(width, p.stereo_width) if inside else 0.5, True,
        )

    if dim == "transients":
        t = m.transients
        if not ctx.has("transients"):
            return (f"Not assessed: {_num(ctx.duration, 1)} s is too short to average punch "
                    f"over enough onsets.", 0.0, False)
        if _fin(t.onset_density, 0.0) <= 0.0:
            return ("No onsets detected — this is sustained material with no transients to "
                    "lose.", 0.0, False)
        return (
            f"Hits land: punch scores {_fin(t.punch_index):.2f} against a "
            f"{_num(p.punch_min, 2)} floor for {label}, with "
            f"{_num(t.transient_to_sustain_db, 1)} dB between the transient and its level "
            f"50 ms later and a {_num(t.attack_time_ms, 1)} ms attack.",
            _headroom_comfort(_fin(t.punch_index), _fin(p.punch_min), 0.25), True,
        )

    if dim == "clarity":
        if not ctx.has("clarity"):
            return (f"Not assessed: {_num(ctx.duration, 1)} s is too short to measure masking "
                    f"across an arrangement.", 0.0, False)
        clarity = _fin(m.clarity.clarity_index, 0.5)
        mask_window, _ = _masking_window(p)
        stem_note = ""
        if ctx.has_stems:
            stem_note = (
                " Separating the sources finds nothing burying anything else either."
                if not ctx.masking_pairs else
                f" Separation does find {len(ctx.masking_pairs)} source-against-source "
                f"overlap{'s' if len(ctx.masking_pairs) > 1 else ''}, reported under the "
                f"dimension that owns each one."
            )
        return (
            f"Elements stay separately audible: clarity scores {clarity:.2f} against a "
            f"{_num(p.clarity_min, 2)} floor for {label}, with "
            f"{_fin(m.clarity.masking_index) * 100:.1f}% of the mix's audible loudness "
            f"sitting under its own masking threshold against a "
            f"{mask_window[1] * 100:.1f}% ceiling." + stem_note,
            _headroom_comfort(clarity, _fin(p.clarity_min), 0.25), True,
        )

    return ("Measured and inside the window for this genre.", 0.5, True)


def score_dimensions(
    findings: List[Finding], m: Measurements, genre: str, intent: str = "full_mix"
) -> List[DimensionScore]:
    """Roll findings up into one score per dimension — all fourteen, always.

    A dimension with no findings is not silent: it gets 90-100 and a headline
    naming the figure that makes it healthy. A dimension whose measurement could
    not be made (mono file's stereo field, a vocal that is not there, a file too
    short for the statistic) says so plainly and scores neutral rather than
    claiming health it cannot support.

    A dimension whose only entries are *observations* — a beat's open center, or
    anything at all on a reference — is not penalised for them. They are read
    out as the headline and the dimension scores as healthy, because that is
    what they say.
    """
    ctx = _build_ctx(m, genre, intent)
    by_dimension: Dict[str, List[Finding]] = {}
    for finding in findings or []:
        by_dimension.setdefault(str(finding.dimension), []).append(finding)

    # A polarity inversion drives the width reading; the stereo dimension is not
    # independently healthy just because the width finding was suppressed.
    polarity = next(
        (f for f in (findings or []) if f.id == "phase.polarity_inverted"), None
    )

    scores: List[DimensionScore] = []
    for dim in DIMENSIONS:
        entries = sorted(
            by_dimension.get(dim, []),
            key=lambda f: (_SEVERITY_RANK.get(f.severity, 3), -_fin(f.impact)),
        )
        # Severity "clean" on a finding means it was emitted as an observation:
        # a confirmed virtue, or a reading off a record nobody is being asked to
        # change. Neither is a problem, so neither carries a penalty.
        hits = [f for f in entries if str(f.severity) != "clean"]
        observations = [f for f in entries if str(f.severity) == "clean"]

        if not hits and observations:
            note = observations[0]
            scores.append(DimensionScore(
                dimension=dim,  # type: ignore[arg-type]
                label=DIMENSION_LABELS.get(dim, dim),
                score=_OBSERVED_SCORE,
                severity="clean",
                headline=note.title,
                finding_ids=[f.id for f in observations],
            ))
            continue

        if hits:
            # Rank by what each finding actually costs, not by its label: a
            # deviation 4 tolerance units out outranks one a whisker outside the
            # window even though both are stamped "minor".
            hits.sort(key=lambda f: (-_finding_penalty(f), -_fin(f.impact)))
            worst = hits[0]
            penalty = _finding_penalty(worst)
            penalty += min(_REPEAT_PENALTY * (len(hits) - 1), _MAX_REPEAT_PENALTY)
            score = _clamp(100.0 - penalty, 2.0, 100.0)
            headline = worst.title
            if len(hits) > 1:
                headline += f" (+{len(hits) - 1} more in this dimension)"
            scores.append(DimensionScore(
                dimension=dim,  # type: ignore[arg-type]
                label=DIMENSION_LABELS.get(dim, dim),
                score=round(score, 1),
                severity=worst.severity,
                headline=headline,
                finding_ids=[f.id for f in hits],
            ))
            continue

        if dim == "stereo_width" and polarity is not None and not ctx.is_mono:
            width = _fin(m.stereo.width, 0.0)
            scores.append(DimensionScore(
                dimension=dim,  # type: ignore[arg-type]
                label=DIMENSION_LABELS.get(dim, dim),
                score=45.0,
                severity="major",
                headline=(
                    f"Width reads {_num(width, 2)} only because the channels are in opposite "
                    f"polarity — fix the phase problem and re-measure, do not narrow the mix."
                ),
                # Cross-reference, not a second charge: the impact stays on the
                # phase finding, so nothing is double-counted in the health score.
                finding_ids=[polarity.id],
            ))
            continue

        headline, comfort, assessed = _clean_report(dim, ctx)
        score = 90.0 + 10.0 * comfort if assessed else _UNASSESSED_SCORE
        scores.append(DimensionScore(
            dimension=dim,  # type: ignore[arg-type]
            label=DIMENSION_LABELS.get(dim, dim),
            score=round(_clamp(score, 0.0, 100.0), 1),
            severity="clean",
            headline=headline,
            finding_ids=[],
        ))

    return scores
