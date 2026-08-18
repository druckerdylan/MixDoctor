"""The questions the analyzer asks instead of guessing.

A measurement can tell you that the bottom end steps back in the intro. It
cannot tell you that the intro is *supposed* to have no bottom end. Those two
facts look identical from the file, and reporting the first as though it were
evidence of the second is how "the dominant issue is low end steps back in
intro" ends up at the top of a report about an arrangement that is working.

So the analyzer asks. Every **deviation** whose measurement cannot separate a
decision from a mistake carries a `Clarification`: one question, answerable
without opening the session, plus what a yes does to the report and what a no
means.

Four rules, and they are what make the questions worth reading.

1. **Never on a defect.** Clipping is not a decision. An inverted channel is
   not a decision. Offering to excuse one would be absurd and would teach the
   user that every finding here is negotiable. `attach()` raises rather than
   let one through — see `_assert_no_defect_questions`.

2. **Answerable from memory of the session, not from the session.** "Is the low
   end pulled back in the intro on purpose?" is a question a producer can
   answer in the queue at a coffee shop. "Is your 250 Hz cut intentional?" is
   not a question anybody can answer about their own mix, because nobody mixes
   in those terms. Where a band has to be named, the question says what living
   there *does* — warmth, articulation, weight — and puts the frequencies in
   brackets for the people who do think that way.

3. **Name the moment.** Where the finding knows a section or a timestamp, the
   question uses it. "in the intro (0:00-0:14)" is a question about a specific
   thing that happened; "in some sections" is a survey.

4. **Neutral.** The question must not carry its own answer. "Did you mean to
   leave the vocal this far back?" is an accusation with a question mark on it.
   "Is the lead meant to sit under the instrumental?" is a question. Every
   `if_intended` here is honest about what a yes actually does, because the
   report has to then do exactly that.

The count is capped. Fifteen questions is a form; `select()` picks the four
highest-impact unanswered ones and `pending()` holds the rest for a user who
asks for more.
"""

from __future__ import annotations

import math
import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .types import Clarification, ClarificationAnswer, Finding

__all__ = [
    "ASK_LIMIT",
    "attach",
    "select",
    "pending",
    "apply_answers",
    "has_question",
    "questioned_ids",
]


# How many questions a report is allowed to actually ask. Four is about the
# most a person will answer before it stops being a conversation and starts
# being a form; the rest stay on the findings and are shown on request.
ASK_LIMIT = 4


# ---------------------------------------------------------------------------
# Reading the finding back
#
# Every number in a question comes off the Finding that carries it, never from
# a fresh calculation — so a question can never cite a figure the report does
# not also show. Where a number is missing the clause containing it is dropped
# rather than printed as a zero, which is why every formatter here returns "".
# ---------------------------------------------------------------------------


def _fin(value, default: Optional[float] = None) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _n(value: Optional[float], nd: int = 1) -> str:
    """A number, or '' if there isn't one.

    Rounding to one place then stripping the trailing zero turns -0.01 into the
    string "-0", and "only -0 dB of clean gain left" is the sort of line that
    tells a producer nobody read this before it shipped. A value that rounds to
    nothing is zero, so it prints as zero — the sign on a rounding error is not
    information.
    """
    if value is None:
        return ""
    text = f"{value:.{nd}f}" if nd else f"{value:.0f}"
    if nd:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "-0.", "") else text


#: "0:22" / "1:48.1" at the head of an evidence row's detail. Used only to tell
#: two identically named sections apart.
_CLOCK_RE = re.compile(r"^\d+:\d{2}(\.\d+)?$")


def _mmss(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    s = max(seconds, 0.0)
    return f"{int(s // 60)}:{int(round(s % 60)):02d}"


def _ev(finding: Finding, *labels: str) -> Optional[float]:
    """The value of the first evidence row matching any of these labels.

    Exact match first, then case-insensitive prefix — evidence labels carry
    interpolated names ("Low-end share in intro"), so a prefix is how you find
    a row whose full label you cannot know in advance.
    """
    rows = list(finding.evidence or [])
    for want in labels:
        for row in rows:
            if row.label == want:
                return _fin(row.value)
    for want in labels:
        low = want.lower()
        for row in rows:
            if str(row.label).lower().startswith(low):
                return _fin(row.value)
    return None


def _ev_window(finding: Finding, label: str) -> Optional[Tuple[float, float]]:
    for row in finding.evidence or []:
        if str(row.label).lower().startswith(label.lower()) and row.target_range:
            lo, hi = row.target_range
            lo_f, hi_f = _fin(lo), _fin(hi)
            if lo_f is not None and hi_f is not None:
                return (lo_f, hi_f)
    return None


def _outside(finding: Finding, label: str) -> bool:
    """Is the named evidence row actually outside its own window?

    A question that quotes a figure sitting comfortably inside its window as the
    reason it is being asked argues against itself: "crest factor is 7.4 dB
    against 5.5 to 10 dB for Trap" reads as *nothing is wrong here*, and the
    producer stops trusting the next question too. Writers use this to pick the
    measurement that actually put the finding on the report.
    """
    value = _ev(finding, label)
    window = _ev_window(finding, label)
    if value is None or window is None:
        return False
    return value < window[0] or value > window[1]


def _window(finding: Finding, label: str, nd: int = 1, unit: str = "") -> str:
    win = _ev_window(finding, label)
    if win is None:
        return ""
    return f"{_n(win[0], nd)} to {_n(win[1], nd)}{unit}"


def _worst_moment_span(finding: Finding) -> str:
    """Just the single worst flagged span, which is what a question should name."""
    moments = list(finding.moments or [])
    if not moments:
        return ""
    worst = max(moments, key=lambda mo: _fin(mo.intensity, 0.0) or 0.0)
    lo = _fin(worst.t_start, 0.0) or 0.0
    hi = _fin(worst.t_end, lo) or lo
    return _mmss(lo) if hi - lo < 0.5 else f"{_mmss(lo)}-{_mmss(hi)}"


def _section(finding: Finding) -> str:
    """The section a moment is labeled with: 'intro' from 'intro: no low end'."""
    for mo in finding.moments or []:
        label = str(mo.label or "")
        if ":" in label:
            name = label.split(":", 1)[0].strip()
            if name:
                return name
    return ""


def _where(finding: Finding) -> str:
    """'the intro (0:00-0:14)' / 'the section at 1:12-1:38' / 'that section'.

    A section whose label is a bare index is not worth naming — 'section 3'
    tells a producer nothing they can recognize, and the clock does.
    """
    name = _section(finding)
    span = _worst_moment_span(finding)
    generic = (not name) or name.lower().startswith("section ")
    if generic:
        return f"the section at {span}" if span else "that section"
    return f"the {name} ({span})" if span else f"the {name}"


def _the(name: str) -> str:
    """'the intro' / 'section 3' — an article, unless the label is an index."""
    clean = str(name or "").strip()
    if not clean:
        return "that section"
    return clean if clean.lower().startswith("section ") else f"the {clean}"


def _at(finding: Finding) -> str:
    """' at 1:04-1:12' — an optional trailing clause, empty when unknown."""
    span = _worst_moment_span(finding)
    return f" at {span}" if span else ""


# ---------------------------------------------------------------------------
# What every yes does
#
# One promise, worded once, because the report has to keep it identically for
# every question — a user who answers four of these must not have to work out
# what each yes bought. `types.Finding.acknowledged` is the entire mechanism,
# and its own docstring is the contract: the finding stays in the report as an
# observation and stops counting against the score. Nothing here promises more
# than that flag can deliver.
#
# A rider may be appended where it is a plain consequence of the same flag —
# "the figure stays in Details" is true because the finding stays. "The report
# will re-measure X" would not be, and does not appear.
# ---------------------------------------------------------------------------


def _kept(subject: str, extra: str = "") -> str:
    """'It stays in the report as a description of the low end and stops …'"""
    base = (
        f"It stays in the report as a description of {subject} and stops "
        f"counting against the score."
    )
    return f"{base} {extra.strip()}".strip()


# ---------------------------------------------------------------------------
# The questions
#
# One writer per finding id. Each takes the finding and the genre's display
# label and returns the Clarification, reading every figure it quotes off the
# finding's own evidence.
# ---------------------------------------------------------------------------


def _q_section_collapse(f: Finding, genre: str) -> Clarification:
    where = _where(f)
    weak_name = _the(_section(f))
    swing = _n(_ev(f, "Low-end swing across sections"), 1)

    # Two evidence rows, "Low-end share in <weakest>" then "... in <strongest>",
    # in that order. The names matter more than the numbers here: a question
    # that says "against the chorus" is about the record, and one that says
    # "against the strongest section" is about the analyzer.
    shares = [
        row for row in (f.evidence or [])
        if str(row.label).lower().startswith("low-end share in ")
    ]

    def _named(row) -> str:
        return str(row.label).split(" in ", 1)[-1].strip()

    def _at(row) -> str:
        """The section's start clock, off the evidence row's own detail.

        The row reads "0:21.6-0:52.3, -13.6 LUFS, …", so the clock is already on
        the finding and does not have to be recalculated. The tenths are dropped
        because the rest of the question is written in m:ss and half a sentence
        in a second format reads like two people wrote it. Only used to break a
        tie — see below.
        """
        head = str(getattr(row, "detail", "") or "").split("-", 1)[0].strip()
        return head.split(".", 1)[0] if _CLOCK_RE.match(head) else ""

    if len(shares) > 1:
        # A record can have two verses, and the weakest and strongest section
        # can both be one. "-10.5 dB of the verse's own energy against -0.5 dB
        # in the verse" is a sentence with no meaning, so when the two names
        # collide the clock is what tells them apart. The clock trails the
        # phrase rather than sitting inside the possessive: "the verse at
        # 0:00's own energy" is the cure being worse than the disease.
        weak_at, strong_at = _at(shares[0]), _at(shares[1])
        collide = _named(shares[0]) == _named(shares[1]) and bool(weak_at and strong_at)
        weak_clock = f" at {weak_at}" if collide else ""
        strong_ref = _the(_named(shares[1])) + (f" at {strong_at}" if collide else "")
        against = (
            f"{_n(_fin(shares[0].value), 1)} dB of {weak_name}'s own energy{weak_clock} "
            f"against {_n(_fin(shares[1].value), 1)} dB in {strong_ref}"
        )
    else:
        against = (
            f"much less of {weak_name}'s own energy than the sections around it do "
            f"of theirs"
        )
    swing_clause = f" — a {swing} dB swing" if swing else ""

    return Clarification(
        question=f"Is the bottom end meant to step back in {where}?",
        context=(
            f"Sub and low bass carry {against}{swing_clause}, measured as each "
            f"section's share of its own level — so this is not the quiet parts "
            f"being quiet: {weak_name} sits within 9 LU of the loudest section and "
            f"still has little underneath it. An arrangement that pulls the bottom "
            f"out and a low end that went missing measure exactly the same."
        ),
        if_intended=_kept(
            "the arrangement",
            "The swing stays on the timeline so you can still see where it happens, "
            "and no fix is prescribed for a section that is doing what it was "
            "written to do."
        ),
        if_not=(
            "Then something is thinning the bottom there without being asked. The "
            "usual causes are a sidechain or bus compressor still keyed to an "
            "element that drops out in that section, a high-pass automated on the "
            "bass, or the sub simply not having been written in yet. Compare the "
            "section's sub level against the one either side of it before reaching "
            "for EQ."
        ),
    )


def _q_no_section_lift(f: Finding, genre: str) -> Clarification:
    name = _section(f)
    span = _worst_moment_span(f)
    part = (
        f"the {name} at {span}"
        if name and not name.lower().startswith("section ") and span
        else f"the loudest part at {span}" if span else "the loudest part"
    )
    lift = _n(_ev(f, "Loudest section vs median"), 1)
    lift_clause = f"only {lift} LU above the median section" if lift else "level with the rest"
    return Clarification(
        question=(
            f"Is this meant to hold one level all the way through, with {part} "
            f"arriving no bigger than the rest?"
        ),
        context=(
            f"The loudest section sits {lift_clause}, when a {genre} arrangement "
            f"usually puts 2-4 LU into it. A record built to stay relentless and a "
            f"hook whose lift never made it out of the mix produce the same number."
        ),
        if_intended=_kept(
            "the arrangement",
            "Loudness range is still reported as a figure, because whoever masters "
            "this will read it before deciding how hard the file can be pushed."
        ),
        if_not=(
            "Then the lift is not surviving the mix. It comes from arrangement and "
            "faders — elements entering, the spread widening, a level move into the "
            "section — and not from the master bus: no amount of limiting puts a "
            "lift back, and heavy limiting is usually what took it away."
        ),
    )


def _q_unmastered(f: Finding, genre: str) -> Clarification:
    lufs = _n(_ev(f, "Integrated loudness"), 1)
    crest = _n(_ev(f, "Crest factor"), 1)
    return Clarification(
        question=(
            "Is this an unmastered mix — a file going to mastering rather than "
            "out as it is?"
        ),
        context=(
            f"Crest factor is {crest or 'wide'} dB and integrated loudness "
            f"{lufs or 'well'} LUFS, under where {genre} releases sit. That pairing "
            f"is either a mix with its headroom deliberately intact or a master "
            f"nobody finished; nothing in the file distinguishes them."
        ),
        if_intended=_kept(
            "where this file is in its life",
            "The level gap is then described as headroom for mastering rather than "
            "as a shortfall, and the loudness findings stop asking you to close it."
        ),
        if_not=(
            "Then this is a finished file that will read quiet and soft next to "
            "anything else in the genre. Check the loudness findings for how much "
            "clean gain is actually available before any limiting: that is the part "
            "that is free."
        ),
    )


def _q_squashed(f: Finding, genre: str) -> Clarification:
    """Ask about the level staying put, and quote the figure that says it does.

    Three measurements can put this finding on a report — loudness range, crest
    factor and TT-DR — and on any given file some of them are inside their
    windows. Quoting a passing one as the reason ("crest factor is 7.4 dB
    against 5.5 to 10 dB for Trap") hands the producer the counter-argument in
    the same breath as the question, so the context is built from whichever of
    the three actually missed, in the order the question is about: the movement
    across the record first, then how hard the peaks have been flattened.
    """
    reasons = []
    if _outside(f, "Loudness range"):
        lra = _n(_ev(f, "Loudness range"), 1)
        window = _window(f, "Loudness range", 1, " LU")
        reasons.append(
            f"loudest to quietest spans {lra} LU"
            + (f" against {window} for {genre}" if window else "")
        )
    if _outside(f, "Crest factor"):
        crest = _n(_ev(f, "Crest factor"), 1)
        window = _window(f, "Crest factor", 1, " dB")
        reasons.append(
            f"crest factor is {crest} dB" + (f" against {window}" if window else "")
        )
    if _outside(f, "TT-DR"):
        dr = _n(_ev(f, "TT-DR value"), 1)
        window = _window(f, "TT-DR", 1, " dB")
        reasons.append(f"TT-DR reads {dr}" + (f" against {window}" if window else ""))

    if reasons:
        measured = reasons[0][:1].upper() + reasons[0][1:]
        if len(reasons) > 1:
            measured += ", and " + ", ".join(reasons[1:])
        measured += ". "
    else:
        measured = (
            f"The level barely moves across the record, against the range {genre} "
            f"usually keeps. "
        )

    return Clarification(
        question=(
            "Is this meant to run at one continuous level, with the quiet moments "
            "held up close to the loud ones?"
        ),
        context=(
            measured
            + "Bus compression that was chosen and bus compression that got away "
            "from you measure identically — the file cannot tell you which hand was "
            "on the fader."
        ),
        if_intended=_kept(
            "the compression",
            "The crest and TT-DR figures stay in Details, because a mastering "
            "engineer reads them before deciding how much further this can go."
        ),
        if_not=(
            "Then something in the chain is doing more than you think. Bypass the "
            "master bus and re-measure: if crest jumps more than a couple of dB, the "
            "limiter or the bus compressor is where the movement went, not the mix."
        ),
    )


def _q_micro_dynamics(f: Finding, genre: str) -> Clarification:
    micro = _n(_ev(f, "Micro dynamics"), 1)
    window = _window(f, "Micro dynamics", 1, " dB")
    gr = _n(_ev(f, "Estimated gain reduction"), 1)
    return Clarification(
        question=(
            "Are the hits meant to be this leveled — every one arriving at the "
            "same size?"
        ),
        context=(
            f"Inside a 50 ms window there is {micro or 'little'} dB between peak and "
            f"RMS"
            + (f", against {window} for {genre}" if window else "")
            + (f", and the crest collapse between loud and moderate passages puts "
               f"estimated gain reduction at {gr} dB" if gr else "")
            + ". A mix that is glued and a mix that is flattened measure the same; "
            "only the intent behind the threshold differs."
        ),
        if_intended=_kept(
            "the compression",
            "Punch and transient findings keep reporting their own numbers, because "
            "those are what the glue costs and they are worth seeing even when the "
            "trade is the one you wanted."
        ),
        if_not=(
            "Then something is taking the front off every hit. Work backwards: "
            "bypass the master chain first, then the drum bus — the stage where "
            "micro-dynamics come back is the one doing it. A longer attack buys the "
            "transient back without giving up the glue."
        ),
    )


def _q_no_punch(f: Finding, genre: str) -> Clarification:
    punch = _ev(f, "Punch index")
    floor = _ev_window(f, "Punch index")
    t2s = _n(_ev(f, "Transient to sustain"), 1)
    return Clarification(
        question=(
            "Is this meant to feel smooth and rolling rather than hitting hard?"
        ),
        context=(
            f"Punch scores {_n(punch, 2) or 'low'}"
            + (f" against a {_n(floor[0], 2)} floor for {genre}" if floor else f" for {genre}")
            + (f", and the average hit sits only {t2s} dB above its own level 50 ms "
               f"later" if t2s else "")
            + ". A deliberately soft, continuous feel and a transient that was "
            "flattened somewhere in the chain give the same reading."
        ),
        if_intended=_kept(
            "the feel of the record",
            "Attack time and smearing stay in Details as the numbers to compare "
            "against if you ever want the hits back."
        ),
        if_not=(
            "Then the impact is being lost on the way out. Look at the weakest band "
            "in the evidence — that is where it is going — then check the drum bus "
            "and the master for a slow-attack compressor or a limiter doing most of "
            "the work."
        ),
    )


def _q_pumping(f: Finding, genre: str) -> Clarification:
    index = _n(_ev(f, "Pumping index"), 2)
    return Clarification(
        question="Is the mix meant to breathe on the beat like this?",
        context=(
            f"The whole mix's envelope modulates in time with the tempo"
            + (f" at a pumping index of {index}" if index else "")
            + f", with gain reduction and micro-dynamics both agreeing. In dance and "
            f"hip-hop that movement is often the hook itself; it is also exactly what "
            f"a bus compressor with the wrong release does by accident."
        ),
        if_intended=_kept(
            "the groove",
            "The pumping rate stays reported next to the detected tempo, because a "
            "pump that is meant to be there should be locked to the grid and the two "
            "numbers are how you check."
        ),
        if_not=(
            "Then a bus compressor is riding the whole mix. Lengthen the release "
            "until the movement stops tracking the beat, or high-pass the "
            "compressor's sidechain input so the kick stops triggering it."
        ),
    )


def _q_too_narrow(f: Finding, genre: str) -> Clarification:
    width = _n(_ev(f, "Stereo width"), 2)
    window = _window(f, "Stereo width", 2)
    return Clarification(
        question=(
            "Is this meant to sit narrow and centered rather than spread across "
            "the field?"
        ),
        context=(
            f"Side/Mid measures {width or 'low'}"
            + (f" against the {window} window {genre} works in" if window else f" for {genre}")
            + ". A record placed deliberately tight and a record whose stereo "
            "content never made it out of the session look the same from here."
        ),
        if_intended=_kept(
            "the image",
            "Mono compatibility keeps being reported, and it will read well — narrow "
            "is the safe direction, and that is worth stating rather than hiding."
        ),
        if_not=(
            "Then the width is not reaching the master. Check for a mono-maker or a "
            "utility plugin left on the bus, and check whether the parts meant to be "
            "wide are actually panned or just double-tracked down the middle."
        ),
    )


def _q_too_wide(f: Finding, genre: str) -> Clarification:
    width = _n(_ev(f, "Stereo width"), 2)
    window = _window(f, "Stereo width", 2)
    loss = _ev(f, "Mono sum loss")
    mono = _n(abs(loss), 1) if loss is not None else ""
    return Clarification(
        question="Is the image meant to be this wide?",
        context=(
            f"Side/Mid measures {width or 'high'}"
            + (f" against the {window} window {genre} works in" if window else f" for {genre}")
            + (f", and folding to mono costs {mono} dB" if mono else "")
            + ". Width that was placed and a widener left across the master are the "
            "same measurement."
        ),
        if_intended=_kept(
            "the image",
            "The mono fold-down cost stays reported as a fact rather than a fault — "
            "clubs, phone speakers and a good deal of playback still sum, and it is "
            "better to know the number than to be surprised by it."
        ),
        if_not=(
            "Then something is spreading the image further than you asked. Listen in "
            "mono first: whatever disappears is what is over-wide. A widener on the "
            "master or on a bus that already carries stereo content is the usual "
            "culprit."
        ),
    )


def _q_vocal_buried(f: Finding, genre: str) -> Clarification:
    at = _at(f)
    v2i = _n(_ev(f, "Vocal vs instruments"), 1)
    window = _window(f, "Vocal vs instruments", 1, " dB")
    intel = _n(_ev(f, "Intelligibility index"), 2)
    return Clarification(
        question=(
            f"Is the lead meant to sit under the instrumental"
            + (f", as it does{at}" if at else "")
            + "?"
        ),
        context=(
            f"The lead measures {v2i or 'below'} dB against everything else"
            + (f", under the {window} window {genre} places a lead in" if window
               else f" for {genre}")
            + (f", at an intelligibility of {intel}" if intel else "")
            + ". A vocal placed inside the bed and a vocal that lost a fader fight "
            "measure the same — and on plenty of records the first one is the point."
        ),
        if_intended=_kept(
            "where the lead sits",
            "Intelligibility keeps being reported, because it is the price of the "
            "placement and worth knowing exactly; it is no longer reported as a "
            "problem."
        ),
        if_not=(
            "Then the lead is losing to the bed. Two to three dB usually settles it. "
            "If the level is right and it still disappears, it is being masked "
            "rather than buried — the low mids and the 1-4 kHz consonant region are "
            "where to look, and carving those beats pushing the vocal harder."
        ),
    )


def _q_vocal_too_loud(f: Finding, genre: str) -> Clarification:
    v2i = _n(_ev(f, "Vocal vs instruments"), 1)
    window = _window(f, "Vocal vs instruments", 1, " dB")
    return Clarification(
        question="Is the lead meant to sit this far out in front of the track?",
        context=(
            f"The lead measures {v2i or 'above'} dB against everything else"
            + (f", over the {window} window {genre} places a lead in" if window
               else f" for {genre}")
            + ". Front-and-center is a style, and it is also where a vocal ends up "
            "after being ridden up to escape something covering it."
        ),
        if_intended=_kept(
            "where the lead sits",
            "The instrumental's level under the vocal keeps being reported, because "
            "that is what a forward lead costs and it is the thing to watch if the "
            "track later starts to feel thin."
        ),
        if_not=(
            "Then the vocal is riding over the track. Pull the lead rather than "
            "pushing everything else up to meet it — and check why it got there: if "
            "it was to escape masking, carving the bed is the actual fix and the "
            "fader can come back down afterwards."
        ),
    )


def _q_vocal_inconsistent(f: Finding, genre: str) -> Clarification:
    spread = _n(_ev(f, "Vocal level spread"), 1)
    window = _window(f, "Vocal level spread", 1, " dB")
    return Clarification(
        question=(
            "Is the vocal meant to move with the performance rather than being "
            "held at a steady level?"
        ),
        context=(
            f"The lead swings {spread or 'widely'} dB between its quietest and "
            f"loudest tenth"
            + (f", against the {window} {genre} usually absorbs" if window else "")
            + ". An unridden performance and a vocal chain that is not holding the "
            "level produce the same spread; the difference is whether the loud "
            "moments are the ones you wanted loud."
        ),
        if_intended=_kept(
            "the performance",
            "The spread stays reported as a figure so mastering knows the range is "
            "deliberate and does not flatten it."
        ),
        if_not=(
            "Then the lead is wandering. Ride the fader first and compress second — "
            "clip gain or automation on the phrases that drop, then let the "
            "compressor handle what is left. A compressor asked to fix this much "
            "performance range on its own will take the life out of it."
        ),
    )


def _q_sub_hot(f: Finding, genre: str) -> Clarification:
    sub = _n(_ev(f, "Sub energy"), 1)
    window = _window(f, "Sub energy", 1, " dB")
    return Clarification(
        question="Is the bottom octave meant to be this big?",
        context=(
            f"Energy under 60 Hz measures {sub or 'high'} dB against the rest of the "
            f"band"
            + (f", above the {window} window {genre} masters sit in" if window
               else f", above where {genre} masters sit")
            + ". Deliberate weight and an 808 nobody filtered look identical here, "
            "and neither reproduces on a laptop speaker."
        ),
        if_intended=_kept(
            "the low end",
            "Sub-25 Hz rumble is still reported separately, because that part is "
            "almost never the same decision and it costs limiter headroom for "
            "something no system will play."
        ),
        if_not=(
            "Then there is more down there than you meant. Check below 30 Hz first — "
            "that is usually where it is and it is free to remove. If the weight sits "
            "in the 40-60 Hz range instead, it is the 808 or the kick, and it is "
            "costing headroom the master will want."
        ),
    )


def _q_sub_thin(f: Finding, genre: str) -> Clarification:
    sub = _n(_ev(f, "Sub energy"), 1)
    window = _window(f, "Sub energy", 1, " dB")
    return Clarification(
        question="Is the bottom octave meant to stay this light?",
        context=(
            f"Energy under 60 Hz measures {sub or 'low'} dB against the rest of the "
            f"band"
            + (f", under the {window} window {genre} masters sit in" if window
               else f", under where {genre} masters sit")
            + ". A bottom end kept deliberately light and a sub that never made it "
            "out of the session sound the same on a small speaker."
        ),
        if_intended=_kept(
            "the low end",
            "The figure stays visible so mastering treats the space as deliberate "
            "rather than as a hole to fill."
        ),
        if_not=(
            "Then the bottom is missing rather than restrained. Check for a "
            "high-pass left across the master or the bass bus, and confirm on "
            "headphones or a full-range system — small speakers cannot tell you "
            "anything about this."
        ),
    )


def _q_kick_bass_collision(f: Finding, genre: str) -> Clarification:
    kick = _n(_ev(f, "Kick fundamental"), 0)
    bass = _n(_ev(f, "Bass fundamental"), 0)
    duck = _n(_ev(f, "Sidechain ducking depth"), 1)
    pitches = (
        f"The kick sits at {kick} Hz and the bass at {bass} Hz" if kick and bass
        else "The kick and the bass are sitting on nearly the same note"
    )
    return Clarification(
        question=(
            "Are the kick and the bass meant to be one sound rather than two "
            "separate parts?"
        ),
        context=(
            f"{pitches}, sharing the same region"
            + (f", and the sub only moves {duck} dB when the kick lands" if duck else "")
            + ". A layered 808 that is deliberately the kick and two separate parts "
            "fighting for the same octave measure the same way."
        ),
        if_intended=_kept(
            "the low end",
            "The overlap is then described as layering, and the report stops asking "
            "for separation between two things that are one thing. Kick definition "
            "is still reported, because a layered low end still has to have an "
            "audible front edge."
        ),
        if_not=(
            "Then they are competing. Decide which one owns the fundamental and move "
            "the other: pitch the 808 away from the kick, or carve a narrow notch in "
            "one at the other's center. Sidechain ducking is the fastest version and "
            "the one that costs the least of either part."
        ),
    )


def _q_mud(f: Finding, genre: str) -> Clarification:
    ratio = _n(_ev(f, "150-400 Hz vs 60-120 Hz"), 1)
    window = _window(f, "150-400 Hz vs 60-120 Hz", 1, " dB")
    res = _ev(f, "Worst 150-500 Hz resonance")
    res_clause = (
        " A narrow resonance was found in the same region, which is rarely part of "
        "the same decision." if res and res > 0.0 else ""
    )
    return Clarification(
        question="Is the mix meant to sit this warm and thick through the low mids?",
        context=(
            f"150-400 Hz carries {ratio or 'more'} dB relative to the bass under it"
            + (f", against {window} for {genre}" if window else f" for {genre}")
            + ". Warmth that was dialled in and warmth that accumulated because "
            "twelve sources each brought a little are the same measurement."
            + res_clause
        ),
        if_intended=_kept(
            "the tone of the record",
            "Any narrow resonance found in the region is still listed on its own, "
            "because a single ringing note is a different problem from a warm "
            "balance and is almost never the thing anybody chose."
        ),
        if_not=(
            "Then it is accumulation, and it is almost never one source. A shelf on "
            "the bus takes the body off everything at once; find the two or three "
            "sources contributing most between 200 and 400 Hz and cut there instead."
        ),
    )


def _q_bright_transients(f: Finding, genre: str) -> Clarification:
    index = _n(_ev(f, "5-9 kHz burstiness index"), 2)
    ceiling = _ev_window(f, "5-9 kHz burstiness index")
    at = _at(f)
    return Clarification(
        question=(
            f"Are the hats and percussion meant to cut this hard up top"
            + (f", as they do{at}" if at else "")
            + "?"
        ),
        context=(
            f"The 5-9 kHz band is bursting"
            + (f" at {index}" if index else "")
            + (f" against a {_n(ceiling[1], 2)} ceiling" if ceiling else "")
            + f", and the bursts are percussion rather than consonants. A bright, "
            f"cutting top is a genre signature in plenty of {genre}; it is also what "
            f"a stock sample and a bright bus produce with nobody deciding anything."
        ),
        if_intended=_kept(
            "the top end",
            "No de-esser is prescribed — it is the wrong tool for percussion anyway. "
            "Fatigue at volume is still noted, because it is what the choice costs."
        ),
        if_not=(
            "Then the top is getting away from the arrangement. A transient shaper on "
            "the percussion group, or a dynamic cut around 6-8 kHz keyed to the hats, "
            "keeps the brightness and removes the spit. A static shelf on the master "
            "takes the air off everything else to fix one source."
        ),
    )


def _q_sibilance(f: Finding, genre: str) -> Clarification:
    index = _n(_ev(f, "5-9 kHz burstiness index"), 2)
    at = _at(f)
    return Clarification(
        question=(
            f"Is the vocal meant to be this bright and airy on the consonants"
            + (f", as it is{at}" if at else "")
            + "?"
        ),
        context=(
            f"The 5-9 kHz band bursts"
            + (f" at {index}" if index else "")
            + " on what the analyzer reads as consonants rather than percussion. A "
            "deliberately crisp, close vocal and one that needs de-essing sit at the "
            "same reading — the index cannot hear the difference between air you "
            "wanted and an 'ess' that spits."
        ),
        if_intended=_kept(
            "the vocal's top end",
            "No de-esser is prescribed for a brightness you chose, and the band stops "
            "being read as a fault in the vocal chain."
        ),
        if_not=(
            "Then the consonants need controlling. De-ess the vocal itself rather "
            "than the bus — a de-esser across the master pulls air off cymbals and "
            "everything else in the band — and split-band it so the fix only lands "
            "on the 'ess' and not on the whole phrase."
        ),
    )


def _q_upper_mid_edge(f: Finding, genre: str) -> Clarification:
    harsh = _n(_ev(f, "Harshness index"), 2)
    at = _at(f)
    return Clarification(
        question=(
            "Is the mix meant to bite this hard through 2-5 kHz — the edge that "
            "makes a record cut on a phone?"
        ),
        context=(
            f"2-5 kHz stands above the line its own neighbors draw"
            + (f", scoring {harsh}" if harsh else "")
            + (f", hardest{at}" if at else "")
            + ". That is a different thing from the mix being bright overall: "
            "deliberate cut-through and a resonance nobody has noticed measure the "
            "same, and this is the region the ear is least forgiving about."
        ),
        if_intended=_kept(
            "the presence region",
            "Fatigue at volume is still noted, because it is what the choice costs "
            "and it is the reason to check the mix quietly as well as loud."
        ),
        if_not=(
            "Then it will turn unpleasant on earbuds before anywhere else. Look for "
            "the narrow peak named in the evidence rather than shelving the whole "
            "region — a dynamic cut that only moves on the loud moments keeps the "
            "presence and loses the edge."
        ),
    )


def _q_congested(f: Finding, genre: str) -> Clarification:
    clarity = _n(_ev(f, "Clarity index"), 2)
    return Clarification(
        question=(
            "Is this meant to be a wall of sound — everything playing at once and "
            "reading as one texture rather than as separate parts?"
        ),
        context=(
            f"A large share of the mix's audible loudness sits under its own masking "
            f"threshold"
            + (f", with clarity at {clarity}" if clarity else "")
            + ". Density that was built on purpose and an arrangement where nobody "
            "made room are the same measurement — shoegaze and a crowded mix look "
            "alike from outside."
        ),
        if_intended=_kept(
            "the density",
            "The congested bands and the moments stay on the timeline, because even "
            "an intentional wall usually has one or two places where something you "
            "wanted heard is not."
        ),
        if_not=(
            "Then parts are present without being audible. The fix is arrangement "
            "before EQ: take something out, or move it in time or in octave. Where "
            "everything has to stay, carve the masker rather than boosting the "
            "maskee — boosting is what got the mix here."
        ),
    )


def _q_too_loud(f: Finding, genre: str) -> Clarification:
    lufs = _n(_ev(f, "Integrated loudness"), 1)
    window = _window(f, "Integrated loudness", 1, " LUFS")
    delta = _n(_ev(f, "Delta vs Spotify reference"), 1)
    return Clarification(
        question=(
            "Is this meant to be delivered louder than a streaming master — for a "
            "club, a car, a download or a CD?"
        ),
        context=(
            f"Integrated loudness is {lufs or 'above'} LUFS"
            + (f", over the {window} window {genre} masters sit in" if window
               else f", over where {genre} masters sit")
            + (f". Streaming normalizes to -14 LUFS, so {delta} LU of that is turned "
               f"straight back down on playback while whatever was done to reach it "
               f"stays in the file" if delta else
               ". Streaming normalizes the level back down while whatever was done to "
               "reach it stays in the file")
            + ". Pushing past the window deliberately and pushing past it by habit "
            "are the same number."
        ),
        if_intended=_kept(
            "the delivery level",
            "Each platform's normalization is still shown, because that is a fact "
            "about what listeners will hear rather than a judgment on the master."
        ),
        if_not=(
            "Then you are paying for level nobody will hear. Back the limiter off "
            "until integrated lands in the window — normalization gives that level "
            "back for free, and the dynamics you spend to get above it do not come "
            "back."
        ),
    )


def _q_cannot_reach_level(f: Finding, genre: str) -> Clarification:
    lufs = _n(_ev(f, "Integrated loudness"), 1)
    headroom = _n(_ev(f, "Clean gain available"), 1)
    return Clarification(
        question=(
            "Is this meant to sit quieter than a released record — a mix going out "
            "for mastering, or a deliberately quiet master?"
        ),
        context=(
            f"Integrated loudness is {lufs or 'under the window'} LUFS with only "
            f"{headroom or 'a little'} dB of clean gain left before the true-peak "
            f"ceiling, so turning it up as far as the peaks allow still lands short "
            f"of where {genre} sits. Headroom kept on purpose and gain staging that "
            f"ran out give the same reading."
        ),
        if_intended=_kept(
            "the level",
            "The remaining gap is then framed as work for mastering rather than as a "
            "shortfall in the mix, and the report stops asking you to close it."
        ),
        if_not=(
            "Then it is the peaks stopping you, not the mix level. Find the two or "
            "three transients setting the ceiling — usually a kick or a snare — and "
            "control those individually. That buys more level than pushing the whole "
            "master into a limiter, and costs less."
        ),
    )


# ---------------------------------------------------------------------------
# Frequency balance: one shape, parameterised by band
#
# A band deviation is the hardest thing here to ask about well, because the
# honest measurement — "the low mids sit 3.4 dB over the reference" — is not a
# thing anybody decided. What they decided was that the mix should sound warm.
# So each band carries the adjective a producer would actually use for living
# there, what the region does musically, and the fix that applies when the
# answer is no. The frequencies stay in the question for the people who do
# think in Hz.
# ---------------------------------------------------------------------------


class _BandSense:
    """What one macro band is, in the words somebody would use about their mix.

    `where` carries the frequencies *and* what lives in them in one breath, so
    the question can name the band without asking anybody to translate. `hot`
    and `thin` are adjectives that fit "the mix sounds this ___", because that
    is the sentence a producer would say about their own record.
    """

    __slots__ = ("where", "hot", "thin", "fix_hot", "fix_thin")

    def __init__(self, where: str, hot: str, thin: str, fix_hot: str, fix_thin: str):
        self.where = where
        self.hot = hot
        self.thin = thin
        self.fix_hot = fix_hot
        self.fix_thin = fix_thin


_BAND_SENSE: Dict[str, _BandSense] = {
    "sub": _BandSense(
        where="under 60 Hz, the weight you feel rather than hear",
        hot="heavy underneath",
        thin="light underneath",
        fix_hot="check what is running below 30 Hz before touching the 808 itself. "
                "That part is inaudible on almost everything and costs limiter "
                "headroom for nothing",
        fix_thin="check for a high-pass left across the master or the bass bus, and "
                 "confirm on something that can actually reproduce 40 Hz",
    ),
    "low_bass": _BandSense(
        where="60-120 Hz, where the kick and the bass have their fundamentals",
        hot="big at the bottom",
        thin="light at the bottom",
        fix_hot="the kick and the bass are usually both sitting in it. Decide which "
                "one owns 60-120 Hz and move the other out of its way",
        fix_thin="either the kick has no fundamental or something is high-passing it, "
                 "and without this band the mix reads weightless on every small "
                 "speaker",
    ),
    "upper_bass": _BandSense(
        where="120-250 Hz, the body of the bass and the low end of guitars and piano",
        hot="thick",
        thin="lean",
        fix_hot="usually the bass body stacking with the low end of guitars and keys. "
                "Carve the accompaniment rather than the bass, which is the part that "
                "has to be there",
        fix_thin="the bass is probably all sub and no body, which vanishes the moment "
                 "the track is played on a phone — a phone cannot reproduce the sub "
                 "that is carrying it",
    ),
    "low_mid": _BandSense(
        where="250-500 Hz, the body in a voice, a snare or a guitar",
        hot="warm",
        thin="hollow",
        fix_hot="almost always accumulation across many sources. Find the two or "
                "three biggest contributors between 250 and 500 Hz rather than "
                "shelving the bus, which takes the body off everything at once",
        fix_thin="a scoop here reads thin and small on small speakers, and it is "
                 "usually a bus EQ doing it rather than any one source",
    ),
    "mid": _BandSense(
        where="500 Hz-1 kHz, the core of most instruments",
        hot="full in the middle",
        thin="scooped in the middle",
        fix_hot="check 300-600 Hz for boxiness and count how many sources are sharing "
                "the same octave; the fix is usually one of them leaving rather than "
                "an EQ move",
        fix_thin="a scooped middle sounds impressive on its own and disappears in a "
                 "playlist. Check whether a smile curve is sitting on the bus",
    ),
    "upper_mid": _BandSense(
        where="1-2 kHz, where attack and articulation live",
        hot="forward",
        thin="recessed",
        fix_hot="this is the most fatiguing region there is, and a narrow dynamic cut "
                "that only moves on the loud moments beats a static one",
        fix_thin="articulation lives here, and a mix light in it reads dull and "
                 "distant no matter how much air is added above it",
    ),
    "presence": _BandSense(
        where="2-5 kHz, the bite that lets a mix cut on a phone",
        hot="bright and cutting",
        thin="soft-edged",
        fix_hot="a dynamic cut keyed to the loud moments keeps the bite and loses the "
                "edge; a static cut here makes everything sound covered",
        fix_thin="intelligibility lives here, and adding air above it will not fix a "
                 "mix that is soft through 2-5 kHz",
    ),
    "brilliance": _BandSense(
        where="5-10 kHz, hats, cymbals and consonant tails",
        hot="crisp up top",
        thin="dark up top",
        fix_hot="hats and cymbals dominate this band, so a transient shaper on the "
                "percussion group beats a shelf on the master",
        fix_thin="check for a low-pass, or a tape or saturation stage rolling the top "
                 "off, before reaching for a shelf",
    ),
    "air": _BandSense(
        where="above 10 kHz, the air over everything else",
        hot="open and airy",
        thin="closed at the top",
        fix_hot="air is cheap to add and easy to overdo. Check whether an exciter or "
                "an air shelf is on the master doing more than you remember",
        fix_thin="worth confirming on headphones first — most rooms and many monitors "
                 "cannot tell you anything above 12 kHz",
    ),
}

# The subject of a sentence about this band, verb included. Listed rather than
# inferred, because "the low mids" and "the low bass" both end in an s and only
# one of them takes a plural verb.
_BAND_SUBJECT: Dict[str, str] = {
    "sub": "The sub sits",
    "low_bass": "The low bass sits",
    "upper_bass": "The upper bass sits",
    "low_mid": "The low mids sit",
    "mid": "The mids sit",
    "upper_mid": "The upper mids sit",
    "presence": "The presence band sits",
    "brilliance": "The brilliance band sits",
    "air": "The air band sits",
}


def _split_band_id(finding_id: str) -> Optional[Tuple[str, str]]:
    """'frequency_balance.low_mid_hot' -> ('low_mid', 'hot')."""
    if not finding_id.startswith("frequency_balance."):
        return None
    rest = finding_id.split(".", 1)[1]
    for direction in ("hot", "thin"):
        suffix = f"_{direction}"
        if rest.endswith(suffix):
            band = rest[: -len(suffix)]
            if band in _BAND_SENSE:
                return band, direction
    return None


def _q_frequency_balance(f: Finding, genre: str) -> Optional[Clarification]:
    parts = _split_band_id(str(f.id))
    if parts is None:
        return None
    band, direction = parts
    sense = _BAND_SENSE[band]
    subject = _BAND_SUBJECT.get(band, f"The {band.replace('_', ' ')} band sits")
    hot = direction == "hot"
    adjective = sense.hot if hot else sense.thin

    deviation = _ev(f, "Deviation from target")
    dev_clause = (
        f"{subject} {abs(deviation):.1f} dB "
        f"{'above' if deviation > 0 else 'below'} where the {genre} reference puts "
        f"{'them' if subject.endswith(' sit') else 'it'}. "
        if deviation is not None else ""
    )

    return Clarification(
        question=f"Is the mix meant to sound this {adjective} — {sense.where}?",
        context=(
            f"{dev_clause}This is a question about the tone of the record rather than "
            f"about a number: a balance chosen at the desk and a balance inherited "
            f"from the room, the monitors or a preset all measure the same from the "
            f"file."
        ),
        if_intended=_kept(
            "the tone of the record",
            "The band still appears on the balance chart against the reference, "
            "because the comparison is the useful part; it just stops being a fault."
        ),
        if_not=(
            "Then the region is carrying "
            + ("more" if hot else "less")
            + f" than you meant: {sense.fix_hot if hot else sense.fix_thin}."
        ),
    )


# ---------------------------------------------------------------------------
# The register
# ---------------------------------------------------------------------------

_WRITERS: Dict[str, Callable[[Finding, str], Optional[Clarification]]] = {
    "low_end.section_collapse": _q_section_collapse,
    "low_end.sub_energy_hot": _q_sub_hot,
    "low_end.sub_energy_thin": _q_sub_thin,
    "low_end.kick_bass_collision": _q_kick_bass_collision,
    "dynamic_range.no_section_lift": _q_no_section_lift,
    "dynamic_range.unmastered": _q_unmastered,
    "dynamic_range.squashed": _q_squashed,
    "compression.micro_dynamics_lost": _q_micro_dynamics,
    "compression.pumping": _q_pumping,
    "stereo_width.too_narrow": _q_too_narrow,
    "stereo_width.too_wide": _q_too_wide,
    "vocal_balance.buried": _q_vocal_buried,
    "vocal_balance.too_loud": _q_vocal_too_loud,
    "vocal_balance.inconsistent": _q_vocal_inconsistent,
    "mud.low_mid_buildup": _q_mud,
    "harshness.bright_transients": _q_bright_transients,
    "harshness.sibilance": _q_sibilance,
    "harshness.upper_mid_edge": _q_upper_mid_edge,
    "clarity.congested": _q_congested,
    "transients.no_punch": _q_no_punch,
    "loudness.too_loud": _q_too_loud,
    "loudness.cannot_reach_level": _q_cannot_reach_level,
}


def _writer(finding_id: str) -> Optional[Callable[[Finding, str], Optional[Clarification]]]:
    direct = _WRITERS.get(finding_id)
    if direct is not None:
        return direct
    if _split_band_id(finding_id) is not None:
        return _q_frequency_balance
    return None


def has_question(finding_id: str) -> bool:
    """Would this finding id carry a clarification, if it fired?"""
    return _writer(str(finding_id or "")) is not None


def questioned_ids() -> Tuple[str, ...]:
    """Every id with a written question, plus the frequency-balance family."""
    family = tuple(
        f"frequency_balance.{band}_{direction}"
        for band in _BAND_SENSE
        for direction in ("hot", "thin")
    )
    return tuple(sorted(_WRITERS)) + family


# ---------------------------------------------------------------------------
# Attaching, selecting, answering
# ---------------------------------------------------------------------------


def _assert_no_defect_questions(findings: Sequence[Finding]) -> None:
    """A defect may never carry a question. This is the guard, not a comment.

    Clipping is not a decision. An inverted channel is not a decision. There is
    nothing to ask about either, and offering to excuse one would teach the
    reader that everything in this report is negotiable — which is the opposite
    of what the defect/deviation split is for.

    Checked over the whole set rather than only over what this module writes,
    because a detector can construct a `Finding(clarification=...)` directly and
    this is the one place that sees every finding on its way out. It fires
    loudly on purpose: there is no correct behavior to fall back to, only a
    wrong question already written.
    """
    guilty = [
        f.id for f in findings
        if str(f.kind) == "defect" and f.clarification is not None
    ]
    if guilty:
        raise AssertionError(
            "clarify: defects must never carry a clarification — a defect is "
            "not a decision anybody made. Offenders: " + ", ".join(map(str, guilty))
        )


def attach(findings: Iterable[Finding], genre_label: str, ask: bool = True) -> List[Finding]:
    """Write a `Clarification` onto every deviation that should carry one.

    `genre_label` is the profile's display name ("Trap") and appears in the
    question text.

    A detector may pre-write its own clarification where it holds context the
    `Finding` does not carry out with it — the arrangement detector knows
    whether the weak section is an intro, a hook or a verse, and its question
    is better for saying so. Those are never overwritten; this fills the rest.

    `ask=False` strips every question instead of writing them. That is what
    `intent="reference"` wants: on a record somebody else finished there is no
    decision of yours to confirm, and asking would be nonsense.

    Mutates in place and returns the same list, so a caller can chain.
    """
    out = list(findings)
    if not ask:
        for finding in out:
            finding.clarification = None
        return out

    genre = str(genre_label or "the reference").strip() or "the reference"
    for finding in out:
        if finding.clarification is not None:
            continue                      # the detector wrote a better one
        # An answered finding keeps its question rather than losing it. It is
        # not asked again — `pending()` filters on `acknowledged` — but the
        # report can then show what was asked next to what was answered, which
        # is the difference between "you confirmed this" and an unexplained
        # finding sitting there with no severity.
        writer = _writer(str(finding.id))
        if writer is None:
            continue
        try:
            finding.clarification = writer(finding, genre)
        except Exception:
            # A question that cannot be written is not worth breaking a report
            # over — the finding stands on its own detail, as it always did.
            finding.clarification = None

    _assert_no_defect_questions(out)
    return out


def _ask_rank(finding: Finding) -> Tuple[float, float, str]:
    """Highest impact first, then the largest miss, then the id for stability."""
    return (
        -(_fin(finding.impact, 0.0) or 0.0),
        -(_fin(finding.miss_ratio, 0.0) or 0.0),
        str(finding.id),
    )


def pending(findings: Sequence[Finding]) -> List[Finding]:
    """Every unanswered finding carrying a question, most worth asking first."""
    return sorted(
        (
            f for f in findings
            if f.clarification is not None and not f.acknowledged
        ),
        key=_ask_rank,
    )


def select(findings: Sequence[Finding], limit: int = ASK_LIMIT) -> List[Finding]:
    """The questions to actually put in front of somebody. At most `limit`.

    Fifteen questions is a form. Four is a conversation, and the four are the
    unanswered deviations with the most recoverable points behind them — if the
    answer is "yes, on purpose", those are the ones that change the report most.

    One per dimension first, then fill. Two questions about the low end in a set
    of four reads as the analyzer laboring a point; spreading them means the
    four cover four different parts of the record.

    The API and the UI both call this, so the set of questions a user is shown
    is decided once and cannot drift between them.
    """
    ranked = pending(findings)
    if limit <= 0:
        return []

    primary: List[Finding] = []
    extra: List[Finding] = []
    seen: set = set()
    for finding in ranked:
        if finding.dimension in seen:
            extra.append(finding)
        else:
            seen.add(finding.dimension)
            primary.append(finding)

    chosen = primary[:limit]
    for finding in extra:
        if len(chosen) >= limit:
            break
        chosen.append(finding)
    return sorted(chosen, key=_ask_rank)


def apply_answers(
    findings: Sequence[Finding], answers: Iterable[ClarificationAnswer]
) -> List[Finding]:
    """Fold the user's answers back onto the findings.

    A yes sets `acknowledged`, which is the whole mechanism: scoring, ranking
    and the fix list all read that one flag, so what a yes *does* lives in
    those places and not here. A no clears it, so an answer can be changed.

    Findings with no matching answer are untouched, and an answer naming a
    finding that is not in the report is ignored rather than an error — a
    client may be posting answers against a slightly older report.
    """
    by_id: Dict[str, bool] = {}
    for answer in answers or []:
        by_id[str(answer.finding_id)] = bool(answer.intended)

    for finding in findings:
        if str(finding.id) in by_id:
            finding.acknowledged = by_id[str(finding.id)]
    return list(findings)
