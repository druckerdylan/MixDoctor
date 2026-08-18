"""What each finding actually means, and what to do about it.

The measurement layer says *"true peak is -0.03 dBTP against a -1.0 ceiling"*.
That is correct and completely useless to someone who does not already know
what true peak is. This module is the other half: what the problem is, what it
sounds like, why it matters, what causes it, how to fix it with the tools you
own, and how to know when you are done.

Three rules, because they are what make this worth reading:

1. **Deterministic.** No API call, no credits, no waiting. Every finding has an
   explainer the moment the analysis finishes. The AI layer adds track-specific
   detail on top; it is not the only thing standing between the user and an
   explanation.
2. **Teach, don't restate.** If a sentence only makes sense to someone who
   already understands the problem, it has failed. Name the perceptual symptom
   before the number.
3. **Tool-tiered.** A fix step can require a capability. Someone with a
   resonance suppressor gets a different instruction from someone with stock
   EQ, and neither should be shown the other's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence

from .capabilities import CAPABILITIES, STOCK_CAPABILITIES


@dataclass(frozen=True)
class FixStep:
    """One thing to do, and why it works."""

    action: str
    detail: str = ""
    #: Capability slug this step needs. Empty means anyone can do it.
    needs: str = ""
    #: Shown instead when the producer lacks `needs`. Empty means hide the step.
    without: str = ""



@dataclass(frozen=True)
class Resource:
    """Somewhere to go and learn more about this specific problem.

    **The rule that shapes this whole feature: never invent a URL.** A
    plausible-looking link to a video that does not exist is worse than no link
    at all — it is the one thing on the page a user can check in a second, and
    a 404 tells them the analysis is guessing too. So there are exactly two
    permitted kinds, and neither can be fabricated:

    `search` — a pre-filled search on a site that always exists. Structurally
    incapable of 404ing, and it improves over time: the query keeps landing on
    whatever the best tutorial is this year rather than freezing a link to a
    2019 video that has since been deleted. This is the workhorse for "show me
    a video about this".

    `reference` — a specific URL, and only from `ALLOWED_RESOURCE_HOSTS`, every
    one of which is checked by `tools/check_links.py` against the live web. If
    it is not in the allowlist and passing the checker, it does not ship.

    **On copyright.** Everything here is a plain hyperlink to a page its
    publisher put on the open web, which is not reproduction and does not need
    a licence. What *would* need one — and what this app therefore never does —
    is copying their text, embedding their video in our player, or hotlinking
    their images. We send the reader to the source; we do not host, mirror or
    excerpt it. Every teaching sentence in these explainers is original writing,
    not a paraphrase of a linked page.

    `source` is required on a `reference` for that reason: the publisher is
    named wherever the link appears, so a reader always knows whose work they
    are about to read and can judge it accordingly. Attribution is a field
    rather than a convention because a convention is something you forget.
    """

    kind: Literal["search", "reference"]
    label: str
    url: str
    #: Why this is worth the click, in one line. Never "click here to learn more".
    note: str = ""
    #: Publisher, shown next to the link. Required for `reference`; a search
    #: names its own site, so it is optional there.
    source: str = ""


#: Hosts a `reference` resource may point at. Everything here was fetched and
#: returned 200 on 18 August 2026. Adding a host means verifying it first and
#: accepting that `tools/check_links.py` will police it from then on.
ALLOWED_RESOURCE_HOSTS: frozenset = frozenset({
    "en.wikipedia.org",       # the concepts, and stable URLs
    "www.itu.int",            # BS.1770, the loudness spec itself
    "www.izotope.com",        # vendor-neutral enough, genuinely good explainers
    "www.soundonsound.com",   # decades of technique writing
    "www.masteringthemix.com",
    "www.youtube.com",        # search only, enforced below
})


def youtube_search(query: str, label: str, note: str = "") -> Resource:
    """A YouTube search that cannot break.

    Deliberately a search rather than a video id. A curated query is a link
    that gets *better* with time; a specific video is one that gets deleted,
    goes private, or turns out to be from someone who was wrong.
    """
    from urllib.parse import quote_plus

    return Resource(
        kind="search",
        label=label,
        url=f"https://www.youtube.com/results?search_query={quote_plus(query)}",
        note=note,
        source="YouTube search",
    )


@dataclass(frozen=True)
class Explainer:
    """The teaching content behind one finding."""

    #: One line the user reads first. Plain language, no units.
    headline: str
    #: What the thing being measured actually is.
    what_it_is: str
    #: The perceptual symptom — what it sounds like, or why you cannot hear it.
    what_you_hear: str
    #: The consequence. On other systems, on streaming, to a listener.
    why_it_matters: str
    common_causes: Sequence[str] = ()
    how_to_fix: Sequence[FixStep] = ()
    #: The stop condition. Measurable or audible, never "until it sounds good".
    how_to_verify: str = ""
    #: The underlying concept, for someone who wants to actually learn it.
    learn_more: str = ""
    #: Roughly how long the fix takes, in minutes.
    minutes: int = 10
    #: Where to go to learn this properly. See `Resource` for the no-fabrication rule.
    resources: Sequence[Resource] = ()


EXPLAINERS: Dict[str, Explainer] = {}


def _add(finding_id: str, explainer: Explainer) -> None:
    EXPLAINERS[finding_id] = explainer


# ---------------------------------------------------------------------------
# Clipping and true peak
# ---------------------------------------------------------------------------

_add(
    "clipping.true_peak_over",
    Explainer(
        headline="Your master will distort on streaming, even though it sounds clean here.",
        what_it_is=(
            "True peak is the level your track actually reaches once it is turned back into "
            "sound — which is higher than the number your DAW's peak meter shows. A digital "
            "file only stores samples: individual points. The real waveform curves between "
            "them, and it can curve above the highest sample. Your meter reads the points; "
            "a converter, a phone, or an MP3 decoder reconstructs the curve. A track metering "
            "-0.1 dB can genuinely hit +0.4 dB on the way out.\n\n"
            "An overshoot that happens in the gap between two stored samples is called an "
            "inter-sample peak, and 'dBTP' — dB true peak — is simply the peak scale that "
            "counts them. Both terms turn up on limiter switches and in every platform's "
            "delivery spec, and they both mean this one thing."
        ),
        what_you_hear=(
            "On your system, nothing. That is exactly the trap — the damage is not in the "
            "file you are listening to, it is created later. On a listener's setup it shows "
            "up as brief crackles and spitting on snare hits, vocal consonants and hi-hats, "
            "plus a general grittiness that was not in your master."
        ),
        why_it_matters=(
            "Every streaming platform re-encodes your master to a lossy format, and that "
            "encoding can push inter-sample peaks another 1-2 dB higher. Anything already "
            "sitting at the ceiling clips inside the listener's player. You never hear it, "
            "and you cannot fix it after the fact — which is why every platform asks for "
            "-1.0 dBTP of headroom rather than 0."
        ),
        common_causes=(
            "The limiter's output ceiling is set to 0.0 or -0.1 dB instead of -1.0 dBTP.",
            "True-peak detection (or oversampling) is switched off in the limiter — it is "
            "off by default in a lot of stock plugins.",
            "Something adds gain after the limiter: a mix-bus trim, a metering plugin left "
            "in, or the export dialogue's own normalisation.",
            "A clipper or saturator ahead of the limiter. Squaring off a waveform adds "
            "harmonics high up in the spectrum, and high-frequency content is exactly what "
            "overshoots hardest between samples — so the limiter holds its ceiling on the "
            "sample values in front of it while the reconstructed curve goes over anyway.",
        ),
        how_to_fix=(
            FixStep(
                action="Set the limiter's output ceiling to -1.0 dBTP.",
                detail=(
                    "Not -0.1, and not 0. The extra dB is the encoder's working room, not "
                    "wasted loudness — streaming normalisation turns everything down to a "
                    "target loudness anyway, so that dB costs you nothing audible."
                ),
                needs="limiter",
            ),
            FixStep(
                action="Turn on true-peak limiting and set oversampling to 4x or higher.",
                detail=(
                    "Without it the limiter is watching the same samples your meter is, so "
                    "it cannot catch what happens between them."
                ),
                needs="limiter_truepeak",
                without=(
                    "Your limiter cannot detect inter-sample peaks, so give it more room "
                    "instead: set the ceiling to -1.5 dB. Cruder, but it works."
                ),
            ),
            FixStep(
                action="Check the signal path after the limiter.",
                detail=(
                    "Anything adding gain downstream undoes the ceiling — a trim, an "
                    "analyser with output gain, or normalisation in the bounce dialogue."
                ),
            ),
            FixStep(
                action="Re-render and re-analyse rather than trusting the meter in the session.",
                detail=(
                    "The in-session meter is reading the same samples again. Only a fresh "
                    "measurement of the rendered file tells you what actually shipped."
                ),
            ),
        ),
        how_to_verify=(
            "Re-analyse the bounced file: true peak should read -1.0 dBTP or below. If your "
            "limiter has a true-peak meter, it should never light an inter-sample over "
            "across a full playthrough."
        ),
        learn_more=(
            "This follows from how digital audio is reconstructed. Sampling stores points; "
            "playback fits a smooth curve back through them. That curve is free to overshoot "
            "the points it passes through, so peak level measured in samples is a lower bound "
            "on the real analogue level, never the true one. Measuring it properly means "
            "upsampling first — which is what the '4x oversampling' switch in a limiter does, "
            "and why a true-peak reading is always equal to or higher than sample peak."
        ),
        minutes=10,
        resources=(
            youtube_search(
                query="inter-sample peaks explained true peak limiter",
                label="Why the peak meter in your DAW is not the peak",
                note=(
                    "Start here if the idea that a file can be louder than its own samples "
                    "still sounds like a trick. These demonstrate the overshoot on a real "
                    "waveform, which is faster to believe once you have watched it happen."
                ),
            ),
            youtube_search(
                query="how to set limiter ceiling for streaming mastering",
                label="Setting the ceiling, in a limiter you actually have",
                note=(
                    "The fix is two switches — output ceiling and true-peak detection — and "
                    "where they live differs per limiter. These walk through the common ones "
                    "and why the extra dB costs you no audible loudness after normalisation."
                ),
            ),
            Resource(
                kind="reference",
                label="ITU-R BS.1770 — where the true-peak measurement is defined",
                url="https://www.itu.int/rec/R-REC-BS.1770/en",
                note=(
                    "The spec rather than a tutorial, and worth knowing exists: the "
                    "oversample-then-measure procedure described here is what every platform's "
                    "dBTP number and every 'true peak' switch is implementing."
                ),
                source="International Telecommunication Union",
            ),
        ),
    ),
)

_add(
    "clipping.hard_clipping",
    Explainer(
        headline="The loudest moments are squared off — that distortion is printed into the file.",
        what_it_is=(
            "Clipping is what happens when a waveform is asked to go higher than the format "
            "can represent. It cannot, so the top of the wave is sliced flat. Those flat tops "
            "are not silence or a gentle squash — a flattened peak is mathematically a burst "
            "of new harmonics that were never in the performance."
        ),
        what_you_hear=(
            "A crunch or fizz on the loudest hits, usually the kick and snare. On a mix that "
            "is heavily clipped it reads as fatigue rather than obvious distortion: the track "
            "feels tiring after thirty seconds and you cannot say why."
        ),
        why_it_matters=(
            "It is already in the rendered audio, so nobody downstream can undo it — not a "
            "mastering engineer, not a plugin. Turning the file down just gives you quieter "
            "clipping. It also eats the transient: the peak that made the kick hit is the "
            "exact part that got sliced off."
        ),
        common_causes=(
            "Too much gain into the limiter or clipper, chasing loudness.",
            "Several stages each adding a little gain, with only the last one metered.",
            "A channel or bus already clipping before the master, so the master limiter "
            "never sees it.",
            "Rendering to a 16-bit integer format from a session that peaks above 0 dBFS.",
        ),
        how_to_fix=(
            FixStep(
                action="Find where it starts. Bypass the master chain and check each bus meter.",
                detail=(
                    "Clipping upstream of the master is invisible on the master meter. Work "
                    "backwards until you find the first stage showing red."
                ),
            ),
            FixStep(
                action="Pull gain out ahead of the limiter rather than pulling the master fader down.",
                detail=(
                    "The fader is after the damage. Reducing input to the limiter is what "
                    "actually stops the flat tops being generated."
                ),
            ),
            FixStep(
                action="If you want the loudness, use a clipper deliberately instead of accidentally.",
                detail=(
                    "A dedicated clipper on the drum bus, shaving 1-2 dB off peaks, is a "
                    "standard modern move and sounds far better than an overdriven limiter "
                    "doing it by accident."
                ),
                needs="clipper",
                without=(
                    "Without a clipper, aim for 1.5-2 dB of limiter gain reduction on peaks "
                    "rather than 4-6, and get the rest of the loudness from balance."
                ),
            ),
            FixStep(
                action="Re-render. Clipping cannot be fixed in place.",
                detail="The flat tops are printed samples; only a new bounce removes them.",
            ),
        ),
        how_to_verify=(
            "A fresh analysis should report zero clipped samples and no flat-topped runs. "
            "By ear, the kick should regain its snap before anything else."
        ),
        learn_more=(
            "A sine wave clipped flat becomes something closer to a square wave, and a square "
            "wave is the sine plus every odd harmonic above it. Those harmonics are spread all "
            "the way up the spectrum, which is why heavy clipping on a bass-heavy mix makes "
            "the *mids* sound harsh — the distortion products land far above the note that "
            "caused them."
        ),
        minutes=20,
        resources=(
            youtube_search(
                query="what is clipping in audio distortion explained",
                label="What a squared-off waveform actually sounds like",
                note=(
                    "Hearing clipping next to the clean signal is the quickest way to learn to "
                    "recognise it in your own mix, which is the skill this finding is standing "
                    "in for until you have it."
                ),
            ),
            youtube_search(
                query="clipper vs limiter for loudness mixing tutorial",
                label="Clipping on purpose instead of by accident",
                note=(
                    "The fix list says to reach for a clipper deliberately rather than let the "
                    "limiter do it unintentionally. These cover where a clipper beats a limiter "
                    "and how much you can take before it stops being transparent."
                ),
            ),
            Resource(
                kind="reference",
                label="Clipping (audio)",
                url="https://en.wikipedia.org/wiki/Clipping_(audio)",
                note=(
                    "The mechanism in one page: why flattening a peak is the same thing as "
                    "adding harmonics that were never played, and why hard clipping generates "
                    "far more of them than soft."
                ),
                source="Wikipedia",
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def explain(finding_id: str) -> Optional[Explainer]:
    """Explainer for a finding, or None when we have nothing worth saying.

    Falls back from a specific id to its dimension, so a per-band finding like
    `frequency_balance.air_thin` can inherit the general frequency-balance
    teaching rather than showing nothing.
    """
    direct = EXPLAINERS.get(finding_id)
    if direct is not None:
        return direct

    dimension = finding_id.split(".", 1)[0]
    for suffix in ("_hot", "_thin"):
        if finding_id.endswith(suffix):
            generic = EXPLAINERS.get(f"{dimension}{suffix}")
            if generic is not None:
                return generic
    return EXPLAINERS.get(dimension)


def applicable_steps(
    explainer: Explainer, owned: Sequence[str] = STOCK_CAPABILITIES
) -> List[Dict[str, str]]:
    """Resolve fix steps against the tools the producer actually has.

    A step whose capability they own renders as written. A step they cannot do
    renders its `without` variant, and is dropped entirely when there isn't
    one — showing someone an instruction for a plugin they do not own is how a
    tool starts feeling generic.
    """
    have = set(owned)
    out: List[Dict[str, str]] = []

    for step in explainer.how_to_fix:
        if not step.needs or step.needs in have:
            out.append({"action": step.action, "detail": step.detail, "requires": step.needs})
        elif step.without:
            out.append(
                {
                    "action": step.without,
                    "detail": "",
                    "requires": "",
                    "substituted_for": CAPABILITIES.get(step.needs, (step.needs, ""))[0],
                }
            )
    return out


def coverage() -> Dict[str, int]:
    """How many findings have teaching content. Used by the tests."""
    return {"explainers": len(EXPLAINERS)}


# ---------------------------------------------------------------------------
# Registration of the per-dimension modules
#
# This block is at the bottom of the file on purpose, and the dependency only
# ever runs one way: the modules below do `from .knowledge import Explainer,
# FixStep`, and we import *them* only after those dataclasses — and the two
# explainers above — already exist. Registering from here rather than having
# each module mutate `EXPLAINERS` directly keeps the merge order explicit and
# visible in one place, and every registration is contained so one bad module
# costs its own findings and nothing else.
# ---------------------------------------------------------------------------

import importlib
import logging as _logging

_log = _logging.getLogger(__name__)

#: Registered in this order. Later modules win a key collision with earlier
#: ones, but never with the explainers defined in this file — see below.
_MODULES: Sequence[str] = (
    "knowledge_loudness",
    "knowledge_tonal",
    "knowledge_space",
    "knowledge_clarity",
)

#: The explainers this module owns, snapshotted before anything merges into
#: `EXPLAINERS`. They are the written standard the other modules are held to, so
#: a submodule cannot quietly replace one by reusing its key.
_CORE_EXPLAINERS: Dict[str, Explainer] = dict(EXPLAINERS)

#: Modules that have not contributed yet. A name leaves this set when it
#: registers, and when it fails in a way that retrying cannot fix.
_PENDING: set = set(_MODULES)


def _register_module(name: str) -> bool:
    """Merge one explainer module in. True when it will never need retrying.

    Never raises. An explainer module is content, not machinery — a stray comma
    turning a string into a tuple is the realistic failure — and the right
    response is an analysis that ships every other explainer plus a loud log
    line, not a 500 on `/analyze`.
    """
    try:
        module = importlib.import_module(f".{name}", __package__)
    except ImportError:
        _log.warning("knowledge: %s is unavailable; its findings ship without teaching "
                     "content", name, exc_info=True)
        return True
    except Exception:
        _log.exception("knowledge: %s raised while importing; its findings ship without "
                       "teaching content", name)
        return True

    register = getattr(module, "register", None)
    if register is None:
        # Partial initialisation, not a broken module. Something imported
        # `analysis.<name>` *first*, so that module is mid-execution in
        # sys.modules, stalled on its own `from .knowledge import ...`, and has
        # no `register` yet. It will finish immediately after we return, so this
        # is the one failure worth retrying — see `ensure_registered`.
        _log.debug("knowledge: %s not ready yet (imported before analysis.knowledge); "
                   "will register on first use", name)
        return False

    try:
        register(EXPLAINERS)
    except Exception:
        _log.exception("knowledge: %s failed to register", name)
        return True

    for finding_id, original in _CORE_EXPLAINERS.items():
        if EXPLAINERS.get(finding_id) is not original:
            EXPLAINERS[finding_id] = original
            _log.warning("knowledge: %s tried to redefine %s; kept the original",
                         name, finding_id)
    return True


def ensure_registered() -> Dict[str, Explainer]:
    """The full registry, completing any registration that could not run yet.

    Idempotent, and after the first pass it is a single empty-set test. It has
    to exist because of the import cycle: `import analysis.knowledge_tonal`
    before `analysis.knowledge` leaves tonal half-built at the moment we would
    otherwise register it, and without a retry that process runs for the rest of
    its life missing a third of the explainers. Anything that reads `EXPLAINERS`
    from outside this module should go through here.
    """
    if not _PENDING:
        return EXPLAINERS
    for name in [n for n in _MODULES if n in _PENDING]:
        if _register_module(name):
            _PENDING.discard(name)
    return EXPLAINERS


ensure_registered()
