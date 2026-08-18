"""Loudness, dynamic range and compression — what the numbers actually mean.

This is the half of the loudness story that a meter cannot tell you. The
measurement layer says *"short-term PSR bottoms out at 3.1 dB"*; this module
says that the limiter has stopped catching peaks and started manufacturing
harmonics, what that sounds like, and which knob to move.

Four ideas run through everything here, and almost nobody arrives already
holding all four:

* **Loudness is not peak.** A track can touch the ceiling and still be quiet.
  Peak is one instant; LUFS is weighted energy over time. They move
  independently, and confusing them is the root of most of these findings.
* **Streaming normalization changed the arithmetic.** Loudness above the
  target is handed back at playback. What you did to get there is not.
* **Micro and macro dynamics are different things.** Macro is the chorus
  lifting above the verse. Micro is the stick surviving inside the hit.
  Compression takes micro first, and micro is what makes a record sound alive.
* **Genre decides the target.** There is no universal correct crest factor,
  loudness or dynamic range, and any advice that gives one is wrong for most
  of the people reading it.

`register()` is called by `knowledge.py`. Nothing here touches
`knowledge.EXPLAINERS` at import time — that direction of dependency is a
circular import.
"""

from __future__ import annotations

from typing import Dict

from .knowledge import Explainer, FixStep, Resource, youtube_search

EXPLAINERS: Dict[str, Explainer] = {}


def _add(finding_id: str, explainer: Explainer) -> None:
    EXPLAINERS[finding_id] = explainer


# ---------------------------------------------------------------------------
# Limiter behavior
# ---------------------------------------------------------------------------

_add(
    "limiter.over_driven",
    Explainer(
        headline="Your limiter has stopped controlling peaks and started adding distortion of "
                 "its own.",
        what_it_is=(
            "A limiter's job is to catch the occasional moment that pokes above everything "
            "else. Whether it is still doing that job shows up in one number: the gap between "
            "the loudest instant and how loud the music around it actually feels. Music has "
            "that gap naturally — a snare sticks out above the bar it sits in. Push more level "
            "into a limiter and the gap shrinks, because the limiter is pulling the top down "
            "toward the body.\n\n"
            "Drive it harder still and it stops reacting to occasional peaks and starts "
            "holding the signal against the ceiling more or less all the time. From that point "
            "extra input buys you almost no extra output level. What it does buy is gain that "
            "moves further and faster, and gain that moves fast enough is no longer volume "
            "control — it is reshaping the waveform, which is another way of saying it is "
            "making distortion. That is the state this finding is describing."
        ),
        what_you_hear=(
            "The kick turns into a thud with no click on the front. Everything sounds equally "
            "loud, so nothing sounds loud. On sustained loud passages there is a fizz or grit "
            "sitting over the cymbals and the top of the vocal that was not in the mix. The "
            "reliable tell is time: it sounds impressive for ten seconds and tiring after "
            "thirty, because the ear never gets a moment of relief to measure the loud parts "
            "against."
        ),
        why_it_matters=(
            "The loudness is temporary and the damage is not. Streaming turns the whole track "
            "down to its target, so the level you bought with limiter drive is handed straight "
            "back — while the flattened transients and the generated harmonics stay in the "
            "file at any volume. This is also the specific failure that makes a master sound "
            "amateur while measuring correctly: the balance is fine, the spectrum is fine, and "
            "the record still sounds smaller than the reference it is being compared to."
        ),
        common_causes=(
            "The limiter's input was pushed up to hit a loudness number rather than set to "
            "what the material tolerates.",
            "Release is too fast for the tempo, so the gain recovers between every hit — fast "
            "gain movement under heavy reduction is what generates the harmonics.",
            "Two loudness stages in series, each 'only doing a couple of dB', with only the "
            "last one being watched.",
            "The low end is doing all the driving. Sub and kick carry most of the energy, so "
            "they trigger the limiter on every beat and everything else gets pulled down with "
            "them.",
            "The mix is not balanced, so the limiter is being asked to resolve a level fight "
            "that faders should have resolved.",
        ),
        how_to_fix=(
            FixStep(
                action="Back the limiter's input off until the gain-reduction meter flickers "
                       "instead of sitting.",
                detail=(
                    "The working target is one to two dB on the loudest hits, not four to six "
                    "held continuously. You will lose level doing this. The level is "
                    "recoverable later; the harmonics are not."
                ),
                needs="limiter",
            ),
            FixStep(
                action="Take the peaks off before the limiter with a clipper.",
                detail=(
                    "A clipper shaves the very tips of the drum transients, which are short "
                    "enough that flattening them is barely audible, and hands the limiter a "
                    "signal it no longer has to fight. This is why modern loud records can be "
                    "loud without sounding crushed — the clipper does the peak work and the "
                    "limiter only does the ceiling."
                ),
                needs="clipper",
                without=(
                    "Without a clipper, split the work across two gentle stages instead of one "
                    "hard one: a compressor on the mix bus taking a dB or two, then the "
                    "limiter taking a dB or two. Two small reductions sound very different "
                    "from one large one."
                ),
            ),
            FixStep(
                action="Slow the limiter's release, or switch it to auto.",
                detail=(
                    "A release short enough to recover between hits makes the gain move at "
                    "something close to audio rate, and gain moving at audio rate is a "
                    "modulation that generates its own harmonics. Auto-release exists "
                    "specifically to avoid this and is usually the right answer on a master."
                ),
                needs="limiter",
            ),
            FixStep(
                action="Find out which part of the spectrum is actually triggering it.",
                detail=(
                    "Put an EQ before the limiter and low-pass everything above about 150 Hz, "
                    "so only the bottom end reaches it. Watch the gain-reduction meter. If the "
                    "bottom alone makes it work nearly as hard as the full mix does, the low "
                    "end is what you are limiting and no limiter setting will fix that — the "
                    "sub level will. Take the low-pass back off afterwards."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Compress the low band separately so it stops dragging the whole mix "
                       "down.",
                detail=(
                    "Control the bottom with its own band and the limiter suddenly has far "
                    "less to do, because the peaks it was chasing were never in the part of "
                    "the spectrum you were listening to."
                ),
                needs="comp_multiband",
                without=(
                    "Put a compressor on the bass and kick bus rather than on the master, and "
                    "get the low end consistent there. A steady bottom end is what lets a "
                    "master limiter relax."
                ),
            ),
            FixStep(
                action="Accept the genre's loudness window as the finish line.",
                detail=(
                    "If the track will not reach the level you want with the limiter working "
                    "gently, the answer is not more drive. Every platform normalizes to a "
                    "target anyway, so the extra drive is buying distortion and no loudness."
                ),
            ),
        ),
        how_to_verify=(
            "Re-analyze: short-term peak-to-loudness should come back inside the genre range "
            "and the distortion index should drop. By ear, level-match the two versions — pull "
            "the louder one down until they sound equally loud — and A/B. The backed-off "
            "version should sound bigger on the drums. If it does, the drive was the problem."
        ),
        learn_more=(
            "Two related numbers get confused here. PLR is peak against the whole file's "
            "integrated loudness: a single figure for the record, and easy to flatter with a "
            "quiet intro. PSR is the same ratio measured over short windows, so it tells you "
            "what is happening moment to moment, and the low end of its distribution tells you "
            "what happens in the loudest bars — which is exactly where a limiter works "
            "hardest. That is why the analysis reads the tenth percentile rather than the "
            "average.\n\n"
            "The mechanism behind the distortion is worth understanding, because it explains "
            "every setting above. A limiter reduces gain, and gain reduction is multiplication "
            "of the signal by a control curve. If that curve moves slowly relative to the "
            "audio, you hear it as level changing. If it moves fast — heavy reduction plus a "
            "short release — the control curve itself has energy at audio frequencies, and "
            "multiplying two signals together creates sum and difference frequencies that were "
            "in neither of them. That is not a metaphor for distortion; it is the same "
            "arithmetic as ring modulation, which is why an over-driven limiter sounds gritty "
            "in a way that a gently-set one never does."
        ),
        minutes=25,
        resources=(
            youtube_search(
                "how much limiter gain reduction is too much mastering",
                "What a limiter sounds like once it stops catching and starts holding",
                "Engineers A/B-ing occasional gain reduction against continuous, which is the "
                "distinction this finding is built on and the one a meter alone will not teach you.",
            ),
            youtube_search(
                "clipper before limiter mastering loud without distortion",
                "Clipping the peaks so the limiter can go back to its job",
                "The clipper-then-limiter chain from the fix steps, demonstrated end to end — "
                "including how far you can shave a drum transient before anyone hears it.",
            ),
            youtube_search(
                "limiter attack release settings mastering explained",
                "Release time, and why a short one makes grit",
                "The setting behind the fizz on your loud sections: release short enough that the "
                "gain recovers between hits is the thing generating harmonics, not the ceiling.",
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Loudness
# ---------------------------------------------------------------------------

_add(
    "loudness.cannot_reach_level",
    Explainer(
        headline="Turning this up as far as the peaks allow still leaves it quieter than the "
                 "genre sits.",
        what_it_is=(
            "There are two different meanings of 'loud' and they do not track each other. "
            "Peak level is the single highest instant in the file — one sample, over in a "
            "fraction of a millisecond. Loudness, measured in LUFS, is the energy of the whole "
            "track averaged over time and weighted to match what the ear is actually sensitive "
            "to. A file can be touching the ceiling and still be quiet, because a handful of "
            "isolated spikes are up there while the body of the music sits far below them.\n\n"
            "That distance is what this finding measures. Adding clean gain — just turning the "
            "whole thing up — is free until the peaks reach the ceiling. Here you can spend "
            "all of that and still land under where releases in this genre sit, which means "
            "the shortfall is not a fader move. Something in the mix is peaking a long way "
            "above the music."
        ),
        what_you_hear=(
            "The mix sounds small even with the fader at the top, and turning the monitors up "
            "gives you more level without giving you more weight — the density you are chasing "
            "is not in the file to be found. Usually one element gives the cause away: a "
            "snare, a clap, a vocal consonant or a percussion one-shot that jumps out of the "
            "speakers while everything else stays where it is. In a playlist the track sounds "
            "like it is playing in the next room."
        ),
        why_it_matters=(
            "Streaming will turn quiet tracks up toward its target, so this may not sound "
            "quiet on Spotify — but that is a plaster, not a fix, and it is not applied "
            "everywhere. Anything that does not normalize, which includes DJ software, a "
            "client's video timeline, a sync brief and most download contexts, plays it as "
            "delivered. The bigger cost is what the peaks force you to do next: reaching genre "
            "level from here means a limiter removing a lot of peak, and that is precisely the "
            "squashing that ruins masters. Fixing the peaks in the mix means the loudness "
            "comes almost for free afterwards."
        ),
        common_causes=(
            "One percussive element with a huge, uncompressed transient — a clap, a rimshot, a "
            "sampled snare — setting the peak for the whole record.",
            "A vocal that was never compressed or ridden, where a single hard consonant "
            "defines the ceiling.",
            "Sub-bass with a peaky waveform. A raw sine 808 or an unfiltered DI has far more "
            "peak than it has audible weight.",
            "Inaudible low-frequency rumble or DC offset under the mix, adding peak level with "
            "no loudness whatsoever.",
            "Nothing on the mix bus, so no stage is bringing the elements into a shared level.",
            "One very short loud moment in the arrangement — an impact, a crash, a riser "
            "landing — that no listener registers as loud but the meter does.",
        ),
        how_to_fix=(
            FixStep(
                action="Find the peak before you process anything.",
                detail=(
                    "Play the loudest section with peak-hold on the master meter, then mute "
                    "elements one at a time. When the peak reading drops a long way, you have "
                    "found it. It is very often not the element you assumed."
                ),
            ),
            FixStep(
                action="High-pass anything that does not need to reach the bottom octave.",
                detail=(
                    "Rumble below about 30 Hz is peak level you cannot hear on any speaker and "
                    "cannot use. Removing it can hand back a dB or more of clean gain across "
                    "the whole mix for no audible change at all."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Control the offending element on its own channel, not on the master.",
                detail=(
                    "A fast-attack compressor on that one channel, taking three or four dB on "
                    "the hits only, brings its peak down into line with the rest of the mix. "
                    "Doing the same job on the master pulls the entire record down every time "
                    "that one sound happens."
                ),
                needs="comp_single",
            ),
            FixStep(
                action="Clip the tips of the drum transients rather than compressing them.",
                detail=(
                    "The first millisecond of a drum hit carries huge peak and almost no "
                    "perceived loudness, so shaving it is close to inaudible and buys real "
                    "headroom. This is the standard way modern records get level without "
                    "sounding compressed."
                ),
                needs="clipper",
                without=(
                    "Without a clipper, use a fast-attack compressor with a high threshold on "
                    "the drum bus, set so it only touches the loudest hits. Slower and less "
                    "surgical, same direction."
                ),
            ),
            FixStep(
                action="Put something gentle across the mix bus to bring the elements together.",
                detail=(
                    "A bus compressor with a slow attack and a dB or two of reduction on the "
                    "loud sections raises the body of the mix without touching the transients. "
                    "That is the difference between a mix that will master easily and one that "
                    "will not."
                ),
                needs="comp_bus",
                without=(
                    "Use your stock compressor on the mix bus: ratio around 2:1, attack 30 ms "
                    "or slower so the transients pass through, release set to recover before "
                    "the next beat, aiming for one to two dB of reduction in the loud sections."
                ),
            ),
            FixStep(
                action="Watch integrated loudness while you work rather than after.",
                detail=(
                    "A peak meter cannot show you this problem — that is the whole point of the "
                    "finding. Keep a loudness meter on the master so you can see the body "
                    "coming up as you tame the peaks."
                ),
                needs="meter_loudness",
                without=(
                    "Bounce and re-analyze after each round of changes. Slower, but a peak "
                    "meter genuinely cannot tell you whether this is improving."
                ),
            ),
        ),
        how_to_verify=(
            "Re-analyze. Integrated loudness should now reach the genre window with true peak "
            "still at or below -1.0 dBTP, and crest factor should still be inside the genre "
            "range. That last part matters: if crest has dropped below the range you did not "
            "solve it, you just squashed it, and you now have the opposite problem."
        ),
        learn_more=(
            "LUFS comes from ITU-R BS.1770. The signal is passed through a filter that "
            "approximates the ear — a shelf that lifts the upper mids and a high-pass that "
            "de-emphasizes the very bottom — then squared, averaged over time, and gated so "
            "that silence and quiet passages do not drag the number down. Peak measurement "
            "does none of that: no filtering, no averaging, no gate, one instant.\n\n"
            "Two consequences follow, and they explain most of this finding. First, energy "
            "down at 40 Hz contributes far less to LUFS than the same energy at 2 kHz, so a "
            "bass-heavy mix reads quieter than it feels and eats headroom doing it. Second, a "
            "spike lasting a millisecond raises peak by however tall it is and raises LUFS by "
            "essentially nothing — so a single stray transient can cost you several dB of "
            "usable level while contributing nothing a listener would call loudness. The gap "
            "between peak and loudness is the headroom you have not spent, and closing it at "
            "source is what gain staging actually means."
        ),
        minutes=30,
        resources=(
            Resource(
                kind="reference",
                label="ITU-R BS.1770 — the specification LUFS comes from",
                url="https://www.itu.int/rec/R-REC-BS.1770/en",
                note="The actual algorithm behind the loudness number in this report: the "
                     "ear-weighting filter, the gating that ignores quiet passages, and the "
                     "true-peak measurement, from the body that defines all three.",
                source="International Telecommunication Union",
            ),
            youtube_search(
                "why is my mix not loud enough peaks headroom gain staging",
                "Finding what is eating your headroom",
                "Producers hunting the one element that sets the peak for an entire mix — the "
                "first fix step here, and the one people skip on their way to the limiter.",
            ),
            youtube_search(
                "taming loud transients snare clap peaks in a mix",
                "Getting a single stray transient under control",
                "Treating the clap, rimshot or consonant that peaks far above the music on its own "
                "channel, instead of pulling the whole record down every time it happens.",
            ),
        ),
    ),
)

_add(
    "loudness.too_loud",
    Explainer(
        headline="You are pushing harder than the genre needs, and the platform hands that "
                 "loudness straight back.",
        what_it_is=(
            "Every major streaming service measures the loudness of your master once, before "
            "anyone plays it, and applies a fixed volume change so that every track in a "
            "playlist arrives at roughly the same loudness. Spotify, YouTube and Tidal aim at "
            "about -14 LUFS; Apple Music sits nearer -16. It is a plain volume change, not "
            "compression, and it is applied to the whole track.\n\n"
            "So delivering a master well above the target does not make it play louder. It "
            "makes the platform turn it down further. What arrives at the listener is your "
            "audio at the platform's level, including every compromise you made pushing it "
            "there. This finding is that the master sits above where releases in this genre "
            "sit — which is not the same as above -14, because the window is genre-derived and "
            "a techno record is supposed to be hotter than a folk record."
        ),
        what_you_hear=(
            "Nothing, until you level-match. Pull the loud version down so it is the same "
            "apparent loudness as a backed-off version and the differences arrive all at once: "
            "the loud one has rounder drum transients, a flatter low end, a narrower stereo "
            "image — limiters pull hardest on the widest moments — and less sense of depth "
            "between the front and back of the mix. It sounds smaller at the same loudness. "
            "That is the entire argument, and it is why unmatched A/B comparisons are useless: "
            "louder always wins for the first few seconds."
        ),
        why_it_matters=(
            "You are paying in dynamics for loudness the listener never receives. A master "
            "pushed to the edge, played back after normalization, is bit-for-bit the same "
            "audio turned down — squashed transients and all. Meanwhile a version of the same "
            "mix that was not pushed as hard, turned down less, sounds bigger at identical "
            "measured loudness because it kept the peaks. This is not an aesthetic preference; "
            "it is arithmetic, and it is why the loudness war ended when normalization "
            "arrived."
        ),
        common_causes=(
            "Chasing a number heard secondhand — usually -14 for everything — or chasing a "
            "reference track's file level rather than the level it plays back at.",
            "A/B-ing against a reference without matching levels first, so the master gets "
            "pushed a little further every session.",
            "Limiter input nudged up repeatedly over a long session. Each nudge is inaudible; "
            "the total is not.",
            "Several loudness stages compounding — a saturated mix bus, then a clipper, then a "
            "limiter, then a maximizer preset that assumes it is first in the chain.",
            "Monitoring quietly. A quiet monitor makes a mix feel undersized, and the fix "
            "reached for is the master rather than the volume knob.",
        ),
        how_to_fix=(
            FixStep(
                action="Pull the limiter's input down until integrated loudness lands in the "
                       "genre window, and leave the output ceiling at -1.0 dBTP.",
                detail=(
                    "The ceiling is not what is making this loud — the input drive is. Move "
                    "the input, not the ceiling."
                ),
                needs="limiter",
            ),
            FixStep(
                action="Turn the monitors up instead of the master.",
                detail=(
                    "Most over-limiting is a monitoring decision in disguise. If the mix only "
                    "feels big when it is squashed, check that you are not mixing quietly and "
                    "compensating with the master chain."
                ),
            ),
            FixStep(
                action="Level-match every reference before you judge against it.",
                detail=(
                    "Match the two to the same integrated loudness and only then switch "
                    "between them. Unmatched, the louder of any two things sounds better for "
                    "about five seconds and you will push to win that comparison every time."
                ),
                needs="reference_matching",
                without=(
                    "Import a commercial track from the genre onto a channel, pull its fader "
                    "until the two sound equally loud on a quick switch, and mark the fader "
                    "position. Crude, and still far better than comparing unmatched."
                ),
            ),
            FixStep(
                action="Meter integrated loudness across the whole track, not short-term across "
                       "the chorus.",
                detail=(
                    "Short-term readings in the loudest section always look hotter than the "
                    "record is. The number that platforms act on is integrated over the full "
                    "length, gated."
                ),
                needs="meter_loudness",
                without=(
                    "Bounce and re-analyze. A peak meter, a spectrum analyzer and your ears "
                    "cannot give you integrated loudness."
                ),
            ),
            FixStep(
                action="If backing off makes the mix feel weak, fix the balance rather than "
                       "putting the drive back.",
                detail=(
                    "A mix that only sounds big under heavy limiting is usually one where the "
                    "low end is eating the headroom, or where the arrangement has no contrast "
                    "so the limiter was supplying the density. Both are upstream problems."
                ),
            ),
        ),
        how_to_verify=(
            "Integrated loudness inside the genre window and true peak at or below -1.0 dBTP. "
            "Then the listening test that settles it: match the pushed version and the "
            "backed-off version to the same integrated loudness and switch between them "
            "blind. If the backed-off one sounds bigger — and it usually does — you were over "
            "the line."
        ),
        learn_more=(
            "Normalization is asymmetric, and that asymmetry is the whole strategy. Turning a "
            "loud track down is always safe, so platforms always do it in full: go 6 LU over "
            "the target and you lose exactly 6 LU. Turning a quiet track up is not always "
            "safe, because the gain could push peaks into clipping, so platforms limit how far "
            "they will raise a quiet track — Spotify will not push one up if doing so would "
            "clip it. So the penalty for being far under the target is real but small, and the "
            "reward for being over it is zero.\n\n"
            "The genre window matters more than the platform target here. Trap, techno and EDM "
            "releases genuinely sit hot, because dense, saturated, limiter-driven sound is part "
            "of what the genre is, and they take the normalization hit knowingly. Acoustic, "
            "jazz and classical releases sit far below the target and lose nothing, because "
            "normalization raises them and their dynamics are the point. That is why this "
            "analysis compares against where the genre's releases actually land rather than "
            "against a single number for all music."
        ),
        minutes=15,
        resources=(
            Resource(
                kind="reference",
                label="How playback loudness normalization works",
                url="https://en.wikipedia.org/wiki/Audio_normalization",
                note="Peak normalization and loudness normalization side by side, with the worked "
                     "example that makes this finding concrete — a platform simply subtracting "
                     "the difference between your master and its target.",
                source="Wikipedia",
            ),
            youtube_search(
                "spotify loudness normalization why louder masters get turned down",
                "What streaming actually does to a hot master",
                "The same master heard before and after a platform's normalization, which is the "
                "only comparison that shows you what the extra limiter drive bought you.",
            ),
            youtube_search(
                "level matched a b comparison loudness bias mixing",
                "Level-matched A/B, and why it changes your mind",
                "The technique the verify step depends on. Louder wins any unmatched comparison "
                "for about five seconds, which is long enough to make the decision for you.",
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Dynamic range
# ---------------------------------------------------------------------------

_add(
    "dynamic_range.squashed",
    Explainer(
        headline="There is less distance between the quiet and loud parts than this music "
                 "lives on, so nothing lands.",
        what_it_is=(
            "Two measurements, one story. Crest factor is the distance between the loudest "
            "instant in the file and its average level — how far the waveform sticks up above "
            "its own body. Loudness range is the same question over minutes instead of "
            "milliseconds: how far the loud sections sit above the quiet ones. When either "
            "falls under what the genre carries, the file has been compressed toward a single "
            "level. Look at the waveform and it is a rectangle.\n\n"
            "Neither number is a quality score on its own. A jazz record and a trap record are "
            "supposed to measure completely differently, and the trap record is not broken for "
            "being flatter. This fires against the genre's own range, which is why the same "
            "reading would be fine somewhere else."
        ),
        what_you_hear=(
            "Loud and boring. The mix stops feeling like it is moving. The chorus does not "
            "feel bigger than the verse even though you know you pushed the faders; the drums "
            "are clearly present but do not hit; the vocal stays at the same distance for four "
            "minutes. Then fatigue — people reach to turn it off before the second chorus and "
            "cannot say why, because nothing is technically wrong."
        ),
        why_it_matters=(
            "Impact is relative. A loud moment sounds loud because of the quieter moment "
            "before it — the ear adapts to whatever level has been present for a few seconds "
            "and judges everything against that. Take away the contrast and you take away the "
            "impact, and no amount of extra level replaces it, because the platform is going "
            "to normalize the whole thing anyway.\n\n"
            "It is also printed. Compression that has been rendered into a stereo file cannot "
            "be taken back out — a mastering engineer can make this louder or change its tone, "
            "but nobody can put movement back into a waveform that no longer has any. Turned "
            "down to the platform's target, a squashed master and a dynamic one arrive at the "
            "same loudness, and the dynamic one is the one that sounds bigger."
        ),
        common_causes=(
            "The master limiter holding continuous gain reduction rather than catching peaks.",
            "Compression stacked at every stage — channel, group, mix bus, master, limiter — "
            "each taking a couple of dB that sounded harmless in isolation.",
            "Parallel compression blended in too high, so the crushed path fills every gap the "
            "dry path leaves.",
            "The mix underneath was already flat. If every section was built at the same fader "
            "level, the master is only reporting what the arrangement did.",
            "Production from pre-mastered samples and loops, where every element arrived "
            "already limited.",
        ),
        how_to_fix=(
            FixStep(
                action="Bypass the entire master chain and listen to the mix underneath.",
                detail=(
                    "This is the first move because it decides which problem you have. If the "
                    "bare mix already has no movement, nothing you do on the master will put "
                    "it back — the work is faders and automation. If the bare mix breathes and "
                    "the master does not, the chain is the culprit."
                ),
            ),
            FixStep(
                action="Count your compressors and remove the ones that are not earning their "
                       "place.",
                detail=(
                    "Four stages taking two dB each is eight dB of gain reduction nobody ever "
                    "decided to apply. Bypass all of them, then re-enable one at a time and "
                    "keep only the ones you can hear doing something musical."
                ),
            ),
            FixStep(
                action="Slow the attack on the bus compressor so transients pass before it "
                       "clamps.",
                detail=(
                    "An attack of 30 ms or slower lets the front of each hit through and "
                    "compresses the body behind it, which is what makes a mix sound punchier "
                    "rather than flatter. Aim for a dB or two of reduction, released before "
                    "the next beat arrives."
                ),
                needs="comp_bus",
                without=(
                    "Your stock compressor does this job: ratio 2:1, attack 30 ms or slower, "
                    "release short enough to recover before the next beat, one to two dB of "
                    "reduction in the loud sections. The slow attack is the part that matters."
                ),
            ),
            FixStep(
                action="Get density from a parallel path instead of squeezing the main one.",
                detail=(
                    "Compressing a copy hard and blending it under the untouched signal adds "
                    "weight while the dry path keeps every transient intact. Serial "
                    "compression cannot do that — whatever it removes is removed from "
                    "everything."
                ),
                needs="comp_parallel",
                without=(
                    "Send the bus to an aux, compress the aux hard with a fast attack and high "
                    "ratio, and blend it under the dry signal until you hear weight arrive. "
                    "Keep the dry path untouched."
                ),
            ),
            FixStep(
                action="Exaggerate whatever attack is left with a transient shaper — but do that after the steps above, not instead of them.",
                detail=(
                    "Be clear about what this does, because it is easy to expect too much of "
                    "it. A transient shaper cannot put back an attack that is no longer in the "
                    "file; it can only take the difference that still exists between the front "
                    "of a hit and the body behind it and stretch it further apart. On drums "
                    "that were rounded off rather than erased, that is often enough, and it "
                    "does it without undoing the glue you wanted from the bus compressor. On "
                    "something that has been limited flat there is nothing left for it to "
                    "grab, and it will only add a click. So fix the cause first, then use this "
                    "to sharpen the result."
                ),
                needs="transient_shaper",
                without=(
                    "Without a transient shaper the only route is the cause: slow the attack "
                    "times on the drum bus compressor until you can hear the stick again. "
                    "Nothing you place after the compression can widen a gap the compression "
                    "already closed."
                ),
            ),
            FixStep(
                action="Build the contrast back in the arrangement, not on the master.",
                detail=(
                    "Ride the chorus up a couple of dB, pull a layer out of the verse, drop the "
                    "hats to half time somewhere. Contrast created upstream survives the "
                    "master chain; contrast you try to create with the master chain does not "
                    "exist."
                ),
            ),
        ),
        how_to_verify=(
            "Re-analyze: crest factor and loudness range back inside the genre range. By ear, "
            "set the monitor level, play from the verse into the chorus and do not touch "
            "anything — the chorus should feel like it grew. And level-match against the old "
            "version before deciding, or the louder one will win regardless."
        ),
        learn_more=(
            "Crest factor is peak divided by average level, written in dB. A pure sine wave "
            "measures about 3 dB. A single close-miked drum hit can measure 20. A finished pop "
            "master usually lands somewhere near 8 to 13, a jazz or classical record much "
            "higher. The number describes the material, not its quality, which is why a "
            "universal 'minimum DR' rule is meaningless and why this analysis compares against "
            "a genre range.\n\n"
            "Crest factor and loudness range measure different time scales and can disagree "
            "completely. A record can have a wide loudness range — the chorus genuinely lifts "
            "above the verse — while being crushed inside every individual bar, and the "
            "reverse is also possible: perfectly preserved transients across an arrangement "
            "where every section arrives at the same level. Reading only one of them is how "
            "people end up fixing the wrong thing."
        ),
        minutes=40,
        resources=(
            Resource(
                kind="reference",
                label="The loudness war, and what ended it",
                url="https://en.wikipedia.org/wiki/Loudness_war",
                note="Two decades of masters compressed toward a rectangle, why engineers "
                     "objected, and how playback normalization removed the reward — the history "
                     "sitting behind the waveform shape this finding is describing.",
                source="Wikipedia",
            ),
            youtube_search(
                "mix bus compression slow attack punch settings",
                "Bus compression that adds punch rather than removing it",
                "Attack and release demonstrated on a full mix, so you can hear the slow-attack "
                "setting that compresses the body and lets the front of each hit through.",
            ),
            youtube_search(
                "parallel compression mix bus glue without squashing",
                "Density from a parallel path instead of the main one",
                "The one route to a denser mix that does not cost transients: the dry signal stays "
                "untouched while the crushed copy underneath supplies the weight.",
            ),
        ),
    ),
)

_add(
    "dynamic_range.unmastered",
    Explainer(
        headline="Nothing here is broken — this just has not been mastered yet, and it will "
                 "feel small next to anything else.",
        what_it_is=(
            "This is the opposite end of the same axis as a squashed master. There is more "
            "distance between the peaks and the body of the music than finished records in "
            "this genre carry, and the overall level sits below where they sit. That "
            "combination is the fingerprint of a good mix that has not had its final stage: "
            "nothing has brought the level up, and nothing has brought the loudest moments "
            "into line with the rest.\n\n"
            "Neither figure would mean this on its own, which is why they are read together. "
            "Wide dynamics at genre level is a well-made dynamic record and entirely correct. "
            "Wide dynamics well under genre level is a mix waiting for a master."
        ),
        what_you_hear=(
            "On its own, played by itself, it sounds good — often better than the loud "
            "version, because nothing has been done to it. The problem only appears in "
            "company. Play it straight after a released track in the same genre without "
            "touching the volume and yours sounds distant, thin and undersized, and a listener "
            "reads that as a worse record rather than a quieter one."
        ),
        why_it_matters=(
            "Less than the other findings in this list, and it is worth being straight about "
            "that: this is a to-do, not damage. Streaming normalization raises quiet tracks "
            "toward the target, so on Spotify the level gap largely closes by itself. What "
            "does not close by itself is everything else mastering does — the final tonal "
            "decisions, the glue, the consistency across a release — and a track that has "
            "never been mastered has usually not had any of it. It also matters wherever "
            "normalization does not apply: DJ software, a client's timeline, a car A/B against "
            "the radio."
        ),
        common_causes=(
            "The mix is finished and no master has been applied. This is the usual answer and "
            "it is completely normal.",
            "A master chain exists in the session but is bypassed, or the limiter is in place "
            "with its ceiling set and no input drive.",
            "Headroom was left deliberately for a mastering engineer — in which case this "
            "finding is correct, expected, and should be ignored.",
            "One or two elements with huge peaks are holding the whole thing down, so the level "
            "stage was attempted and could not get anywhere.",
        ),
        how_to_fix=(
            FixStep(
                action="Decide whether this should be mastered by you at all.",
                detail=(
                    "If it is going to a mastering engineer, stop here. Leave the headroom, "
                    "take the limiter off, and send it. Everything below is only for when you "
                    "are finishing it yourself."
                ),
            ),
            FixStep(
                action="Do the tonal pass before the level pass.",
                detail=(
                    "Level is the last decision, not the first. Compare against a genre "
                    "reference at matched loudness and correct the balance first — once a "
                    "limiter is in the chain, every EQ move you make changes how hard it works "
                    "and you end up chasing your own tail."
                ),
                needs="reference_matching",
                without=(
                    "Drop a commercial track from the genre into the session, pull its fader "
                    "until it is as loud as yours, and switch between them. Fix the tonal "
                    "differences you hear before you add any level."
                ),
            ),
            FixStep(
                action="Glue before you limit.",
                detail=(
                    "A bus compressor with a slow attack, a low ratio and a dB or two of "
                    "reduction brings the elements into one performance rather than several. "
                    "It does almost nothing for loudness, and that is fine — that is not its "
                    "job."
                ),
                needs="comp_bus",
                without=(
                    "Your stock compressor on the mix bus at 2:1, attack 30 ms or slower, one "
                    "to two dB of reduction on the loud sections. Slow attack, gentle ratio, "
                    "no more."
                ),
            ),
            FixStep(
                action="Then bring the level up with a limiter, aiming at the genre window and "
                       "a -1.0 dBTP ceiling.",
                detail=(
                    "Go in small steps and stop as soon as you are inside the window. There is "
                    "no reward for going past it — the platform takes the difference back."
                ),
                needs="limiter",
            ),
            FixStep(
                action="Watch the dynamics as you add level, so you stop before you overshoot.",
                detail=(
                    "The failure mode from here is landing in the opposite problem. Keep an eye "
                    "on the crest and stop the moment the transients start rounding."
                ),
                needs="meter_loudness",
                without=(
                    "Bounce and re-analyze before you commit. It is much easier to notice "
                    "overshoot in a measurement than by ear during a long session."
                ),
            ),
        ),
        how_to_verify=(
            "Integrated loudness inside the genre window, crest factor still inside the genre "
            "window rather than below it, true peak at or under -1.0 dBTP. If crest ends up "
            "under the window you have overshot and now own the squashed-master problem "
            "instead — go back and take a dB off the limiter drive."
        ),
        learn_more=(
            "Dynamic range on its own tells you nothing about quality. A pristine classical "
            "recording and a rough bedroom demo can measure the same crest factor. The "
            "information lives entirely in the pairing with loudness:\n\n"
            "- Wide dynamics at genre level — a dynamic record, made on purpose.\n"
            "- Wide dynamics well below genre level — the level stage has not happened.\n"
            "- Narrow dynamics at genre level — the loudness was bought with the dynamics.\n"
            "- Narrow dynamics below genre level — something is compressed and it did not even "
            "buy loudness, which usually means the compression is upstream in the mix.\n\n"
            "This is also why DR meter scores quoted on their own, as a badge of quality, are "
            "close to meaningless. Without the loudness figure next to them they cannot "
            "distinguish any of the four cases above."
        ),
        minutes=20,
        resources=(
            Resource(
                kind="reference",
                label="How loud to aim, by genre and by platform",
                url="https://www.masteringthemix.com/pages/how-loud-should-you-master",
                note="Concrete loudness targets for the level stage this track has not had yet, "
                     "including why integrated and short-term readings answer different questions "
                     "and where to leave true peak.",
                source="Mastering The Mix",
            ),
            youtube_search(
                "how to master your own track step by step limiter loudness",
                "A first master, in order",
                "Full walkthroughs that follow the sequence above — tonal decisions, then glue, "
                "then level — rather than opening with a limiter and fixing tone through it.",
            ),
            youtube_search(
                "prepare your mix for mastering how much headroom to leave",
                "If someone else is mastering this",
                "What to send and what to strip off, for the case where this finding is correct "
                "and the right response is to change nothing and leave the headroom alone.",
            ),
        ),
    ),
)

_add(
    "dynamic_range.no_section_lift",
    Explainer(
        headline="The biggest moment in the song arrives at the same size as everything else.",
        what_it_is=(
            "The analysis splits the record into sections and measures the loudness of each "
            "one. In a song-shaped record the loudest section — the chorus, the hook, the drop "
            "— sits measurably above the middle of the record. That difference is the signal a "
            "listener reads as 'this is the payoff'. Here the loudest section barely rises "
            "above the median, so there is no arrival.\n\n"
            "Be aware of what is and is not certain in that. The loudness figures are direct "
            "measurements and are solid. Where the section boundaries were placed is an "
            "algorithm's judgment, so it may not agree with where you would say the chorus "
            "starts. The conclusion — that nothing in this record is much louder than anything "
            "else — does not depend on getting the boundaries exactly right."
        ),
        what_you_hear=(
            "The song never arrives. You can follow the form intellectually — you know which "
            "part is the chorus — but nothing about it feels bigger, and attention drifts by "
            "the second time round. The usual verbal version is 'it's fine, it just doesn't go "
            "anywhere'. A specific tell: the chorus has more parts playing in it than the verse "
            "and still does not sound louder, because everything already there got quieter to "
            "make room for them."
        ),
        why_it_matters=(
            "This one is worth saying plainly because it is the most commonly misdiagnosed "
            "finding in the whole analysis: no plugin on the master fixes it. A limiter cannot "
            "make one section louder than another — it does the exact opposite, it pulls them "
            "toward each other. Compression, saturation and loudness maximizing all reduce the "
            "difference you are trying to create. The lift has to be built in the arrangement "
            "and in the faders, upstream of everything on the master bus.\n\n"
            "Some music is not built this way at all, and the detector knows it: it does not "
            "fire on ambient, classical, jazz or lo-fi, where a record is one continuous "
            "texture by design. If your track is deliberately a single unbroken thing and this "
            "still fired, it is describing your intent rather than a fault, and you should "
            "ignore it."
        ),
        common_causes=(
            "The verse is already full. If everything is playing all the time, there is nothing "
            "left to add when the chorus arrives.",
            "Layers were added to the chorus without pulling anything down to make space, so "
            "the section got denser rather than louder.",
            "The mix bus compressor is doing more gain reduction in the chorus than in the "
            "verse, which cancels the lift precisely where you built it. This one is extremely "
            "common and invisible unless you watch the meter across the transition.",
            "Every section was gain-staged or normalized to the same level during production.",
            "The hook occupies the same instrumentation and the same register as the verse, so "
            "there is no change of color to carry the arrival even if the level moves.",
        ),
        how_to_fix=(
            FixStep(
                action="Take something out of the verse rather than adding to the chorus.",
                detail=(
                    "The lift is the difference, not the absolute level, and you have far more "
                    "room to make the verse smaller than to make the chorus bigger. Strip a "
                    "layer, drop the hats to half time, mute a pad, thin the bass — anything "
                    "that leaves space for the chorus to fill."
                ),
            ),
            FixStep(
                action="Automate the section change at the boundary.",
                detail=(
                    "Ride the key elements — or the whole mix bus — up by two or three dB when "
                    "the chorus lands, and put the verse back down when it returns. Two or "
                    "three dB is not subtle at a section boundary; it reads as a change of "
                    "size."
                ),
            ),
            FixStep(
                action="Check whether your bus compressor is eating the lift you just made.",
                detail=(
                    "Watch the gain-reduction meter across a verse-into-chorus transition. If "
                    "it pulls harder in the chorus, it is undoing your fader move by design. "
                    "Raise the threshold so it is not working in the loud section, slow the "
                    "attack, or move the automation after the compressor rather than before it."
                ),
            ),
            FixStep(
                action="Change the register, not only the level.",
                detail=(
                    "The most reliable chorus lift is not volume at all: an octave that was not "
                    "there before, a counter-line above the vocal, a fuller chord voicing, a "
                    "cymbal that only appears in the hook. The ear reads new information as "
                    "'bigger' more strongly than it reads two dB."
                ),
            ),
            FixStep(
                action="Filter the verse and release the filter at the chorus.",
                detail=(
                    "High-pass the supporting parts through the verse and take the filter off "
                    "when the chorus lands. The low end returning is one of the strongest "
                    "arrival cues there is, and it costs no level at all."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Keep the verse narrow and open the chorus out.",
                detail=(
                    "Width contrast works the same way level contrast does. A verse held "
                    "closer to the center makes a wide chorus feel like the room got bigger."
                ),
                needs="imager",
                without=(
                    "Do it with panning instead: pull the verse's supporting parts in toward "
                    "the center and spread the chorus doubles out wide. Same contrast, no "
                    "plugin, and it is more phase-safe than an imager anyway."
                ),
            ),
        ),
        how_to_verify=(
            "Re-analyze and the loudest section should sit a couple of LU above the median "
            "section. By ear, the stop condition is a fixed monitor level: start playback "
            "thirty seconds before the chorus and do not touch the volume knob. If you do not "
            "feel a change of size when it lands, it is not there yet, no matter what the "
            "faders say."
        ),
        learn_more=(
            "Perceived loudness is relative and the ear adapts within seconds. Sit in a "
            "constant level and your hearing recalibrates to it, so a section sounds big only "
            "in relation to what came immediately before. That is the reason two or three LU "
            "of arrangement lift beats six dB of extra limiter gain: the lift is a change, and "
            "the extra gain is just a new baseline you adapt to within a bar.\n\n"
            "It is also why the standard professional move is to make the verse smaller rather "
            "than the chorus bigger. You get the same difference, and you keep the headroom "
            "instead of spending it — a chorus pushed up costs peak level you then have to "
            "limit away, while a verse pulled down costs nothing and makes the master's job "
            "easier. Contrast is free in one direction and expensive in the other."
        ),
        minutes=60,
        resources=(
            youtube_search(
                "make the chorus hit harder arrangement contrast mixing",
                "Making a chorus feel bigger without making it louder",
                "Arrangement moves rather than plugin moves: what to take out of the verse so the "
                "chorus has somewhere to arrive, which is the fix this finding actually needs.",
            ),
            youtube_search(
                "volume automation mix bus chorus lift mixing",
                "Riding the section change",
                "Drawing the lift across a verse-into-chorus boundary, and checking that the bus "
                "compressor is not quietly undoing the fader move you just made.",
            ),
            youtube_search(
                "why doesn't my drop hit build tension arrangement",
                "When the biggest moment does not land",
                "The same problem from the electronic side, where the arrival is a drop: tension "
                "built before it is what makes it feel large, not level applied to it.",
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

_add(
    "compression.micro_dynamics_lost",
    Explainer(
        headline="Every hit has had its attack flattened, even though the shape of the song "
                 "survived.",
        what_it_is=(
            "Dynamics happen on two time scales and they fail independently.\n\n"
            "Macro dynamics are minutes: the chorus rising above the verse, the breakdown "
            "dropping away. Micro dynamics are milliseconds. Inside a single drum hit there is "
            "a very short spike at the front — the stick striking the skin, the pick catching "
            "the string — followed by a body that is much quieter. The distance between that "
            "spike and the body is the micro dynamic, and this measures it inside a window "
            "about the length of one hit.\n\n"
            "Compression takes micro dynamics first, always, because the transient is by "
            "definition the thing that crosses the threshold first. So a mix can keep a "
            "perfectly healthy macro shape — the chorus still lifts, the loudness range still "
            "measures correctly — while every individual hit inside it has been rounded off. "
            "That combination is exactly what this finding is."
        ),
        what_you_hear=(
            "The words people reach for are flat, dull, small, lifeless — the vocabulary you "
            "get when someone cannot point at a frequency. Concretely: the kick is a thud with "
            "no click on the front, the snare has body but no crack, the acoustic guitar has "
            "no pick sound, the vocal consonants stop pushing forward out of the track. "
            "Everything is audible and nothing is being played by hands. The mix passes every "
            "EQ test you throw at it, because the balance is not what is wrong."
        ),
        why_it_matters=(
            "This is the single most common reason a master sounds uncommercial while "
            "measuring correctly. Transients are how a listener judges distance and physical "
            "reality — a sharp attack reads as close, present and real, a rounded one reads as "
            "distant and synthetic — so losing them moves the whole record backwards in the "
            "room.\n\n"
            "It also costs loudness the wrong way round. The peaks were doing a lot of the work "
            "of making the drums feel loud; with them gone, the track has to run at a higher "
            "average level to feel the same, which means more compression, which removes more "
            "transient. The damage is cumulative and no single stage in the chain ever sounds "
            "wrong on its own."
        ),
        common_causes=(
            "Fast attack times on channel compressors. Anything under about 10 ms catches the "
            "transient itself rather than the body behind it.",
            "A master limiter doing continuous gain reduction instead of occasional. A limiter "
            "has an attack time near zero by design, so every dB it takes comes off the "
            "transient.",
            "Serial compression: five stages each removing a barely audible amount, none of "
            "them individually worth arguing about.",
            "Parallel compression blended so high that the crushed path is effectively the "
            "signal and the dry path is the garnish.",
            "Samples that arrived already limited. A loop bounced from someone else's master "
            "has no transient left to preserve.",
            "A transient shaper set to reduce attack, or a preset labeled 'punch' that is "
            "actually adding sustain.",
        ),
        how_to_fix=(
            FixStep(
                action="Slow the attack times, starting with the drum bus and the mix bus.",
                detail=(
                    "For anything that is supposed to hit, an attack of 20 to 30 ms lets the "
                    "transient through and compresses what follows it — which makes the hit "
                    "sound punchier, not softer. Fast attack is the right tool for taming a "
                    "single wild peak and the wrong tool for glue."
                ),
                needs="comp_single",
            ),
            FixStep(
                action="Audit the chain by bypassing one compressor at a time.",
                detail=(
                    "Solo the drums in the mix and bypass each dynamics processor in turn, "
                    "listening for the click on the front of the kick. One of them is usually "
                    "responsible for most of the loss, and it is rarely the one you suspect."
                ),
            ),
            FixStep(
                action="Move density into a parallel path instead of a serial one.",
                detail=(
                    "Compressing a copy hard and blending it underneath adds weight while the "
                    "untouched path keeps every transient. A serial compressor cannot do that "
                    "— whatever it removes is removed from the only signal there is."
                ),
                needs="comp_parallel",
                without=(
                    "Send the drum bus to an aux, compress the aux hard with a fast attack, and "
                    "blend it in under the dry signal. Keep the dry path completely untouched "
                    "— it is the thing carrying the attack."
                ),
            ),
            FixStep(
                action="Rebuild the attack with a transient shaper.",
                detail=(
                    "A transient shaper works on the envelope rather than on level, so it can "
                    "restore the front of a hit without disturbing the balance you already "
                    "settled."
                ),
                needs="transient_shaper",
                without=(
                    "Nothing puts back an attack that is gone, so the work here is undoing the "
                    "stage that removed it. Bypass the compressors on the drum bus one at a "
                    "time, find the one costing the most click, and slow its attack or take it "
                    "out."
                ),
            ),
            FixStep(
                action="Let a clipper take the peaks so the limiter can stop taking them.",
                detail=(
                    "A clipper removes the top fraction of a millisecond and leaves the rest of "
                    "the transient shape intact. A limiter reacting to the same peak pulls the "
                    "whole hit down and holds it there. Same peak reduction, completely "
                    "different result."
                ),
                needs="clipper",
                without=(
                    "Reduce the limiter's input until the gain-reduction meter flickers on the "
                    "loudest hits instead of moving continuously, and take the level loss."
                ),
            ),
            FixStep(
                action="Check the source material before processing anything.",
                detail=(
                    "If the sample itself arrived flat, nothing downstream restores it. Layer a "
                    "sharper one underneath or replace it — that is faster and sounds better "
                    "than any amount of shaping."
                ),
            ),
        ),
        how_to_verify=(
            "Re-analyze: micro-dynamics back inside the genre range. By ear, the stop condition "
            "is the stick — at the same monitor level as before, you should be able to hear the "
            "front edge of each kick and snare clearly in the full mix, not just when soloed."
        ),
        learn_more=(
            "The reason compression takes micro dynamics first is in the attack time. A "
            "compressor's detector needs a moment to respond; during that moment the transient "
            "passes through at full level and gain reduction only clamps down afterwards. So a "
            "slow attack keeps the spike and squashes the body, which increases the distance "
            "between them and makes the hit sound harder. A fast attack does the reverse: it "
            "catches the spike and leaves the body, which reduces the distance. Both are "
            "correct tools; they just do opposite jobs, and 'punch' presets frequently pick "
            "the wrong one.\n\n"
            "A limiter is the extreme case — its attack is effectively zero, because that is "
            "what holding a ceiling requires — so a limiter can only ever remove micro "
            "dynamics, never create them. Macro dynamics are untouched by any of this until "
            "the gain reduction becomes near-continuous, which is exactly why a mix can show a "
            "perfectly healthy loudness range and still sound dead. The two measurements are "
            "not different views of the same thing; they are different things."
        ),
        minutes=35,
        resources=(
            Resource(
                kind="reference",
                label="Attack and release, explained on a snare",
                url="https://www.soundonsound.com/techniques/compression-made-easy",
                note="A careful treatment of the exact mechanism this finding rests on: a slow "
                     "attack keeps the spike and squashes the body, a fast one does the reverse, "
                     "and 'punch' presets routinely pick the wrong one.",
                source="Sound On Sound",
            ),
            youtube_search(
                "compressor attack time transients punch explained",
                "Attack time, demonstrated on drums",
                "Hear the click on the front of a kick disappear and come back as the attack time "
                "moves — this finding in a single audible demonstration.",
            ),
            youtube_search(
                "parallel compression drums keep transients tutorial",
                "Weight without losing the front of the hit",
                "Moving density into a parallel path, which is the only way to add it without "
                "taking it out of the transients you are trying to keep.",
            ),
        ),
    ),
)

_add(
    "compression.pumping",
    Explainer(
        headline="The whole mix is rising and falling in time with the beat instead of "
                 "individual parts moving.",
        what_it_is=(
            "The analysis takes the level of the whole mix over time, removes the slow "
            "arrangement changes, and asks how much of the wobble that remains repeats in time "
            "with the beat — at the tempo itself, or at half it, or at double it. A compressor "
            "that pushes down on every kick and lets go before the next one imprints exactly "
            "that: a repeating rise and fall at the beat rate, applied to everything in the "
            "mix at once.\n\n"
            "This is the least certain finding in the analysis and it should be treated that "
            "way. Percussive music modulates its own level at the beat rate with no compressor "
            "involved at all — a loop of kicks *is* a periodic envelope, and no measurement "
            "taken from a stereo file can tell that apart from a compressor breathing. That is "
            "why it is only reported when two independent things agree with it: the estimated "
            "gain reduction is significant and the micro-dynamics are already at the bottom of "
            "the genre's range. Even then, read this as a reason to go and listen for "
            "something specific, not as a verdict."
        ),
        what_you_hear=(
            "Listen to something that should be steady while the beat plays — a sustained pad, "
            "a held vocal note, a reverb tail, a ride cymbal. If it ducks down and swells back "
            "in time with the kick, that is it, and once you have heard it you cannot unhear "
            "it. Over two bars it can pass as groove. Over a full record it sounds like the mix "
            "is breathing, and the giveaway is that the 'groove' is happening to elements that "
            "are not playing anything rhythmic."
        ),
        why_it_matters=(
            "The cause is nearly always the low end driving a full-band detector. Kick and bass "
            "carry most of the energy in almost any mix, so a single compressor watching the "
            "whole signal is effectively following the bottom octave — and when it clamps for "
            "the kick, it clamps the vocal, the hats and the reverb along with it. The mix ends "
            "up sounding smaller than the sum of its parts, and it wastes headroom: everything "
            "above 200 Hz is being turned down to control something below 100.\n\n"
            "Genre caveat that matters here. In house, techno and a lot of EDM, audible "
            "sidechain ducking is the sound and is put there deliberately. If the pumping is "
            "intentional and locked to the kick, this finding is describing your arrangement, "
            "not a fault. What is worth checking even then is whether a second, accidental "
            "compressor on the master is adding its own breathing on top of the one you meant."
        ),
        common_causes=(
            "One full-band compressor or limiter with the low end dominating its detector.",
            "A release time close to the beat interval, so the compressor finishes recovering "
            "just as the next hit arrives. This is the worst case and it happens by accident "
            "whenever release is dialled by feel at one tempo.",
            "Sidechain ducking set too deep or with too long a release, so the duck has not "
            "finished when the next kick lands.",
            "A master limiter taking several dB on every kick — a limiter pumping is a "
            "compressor pumping with the attack removed.",
            "Kick and bass occupying the same frequency and the same moment, so the combined "
            "low end is far louder than either needs to be and drives everything harder than "
            "the music does.",
        ),
        how_to_fix=(
            FixStep(
                action="Confirm it by ear before you change anything.",
                detail=(
                    "Solo a sustained element with the drums still audible behind it and listen "
                    "for the duck. If nothing ducks, the measurement was reading the groove "
                    "rather than a compressor, and you should stop here — that is a known limit "
                    "of this test, not a failure to find the problem."
                ),
            ),
            FixStep(
                action="Stop the low end from driving the detector.",
                detail=(
                    "Compress the low band on its own so that the kick and bass only duck each "
                    "other, and the rest of the spectrum stops moving with them. This is the "
                    "actual fix for most cases of this finding rather than a workaround."
                ),
                needs="comp_multiband",
                without=(
                    "If your compressor has a sidechain or detector high-pass, set it around "
                    "100-150 Hz so the bottom octave stops triggering it. If it does not, take "
                    "the compressor off the full mix and put it on a bus that excludes the kick "
                    "and bass instead."
                ),
            ),
            FixStep(
                action="Set the release against the tempo rather than by feel.",
                detail=(
                    "Work out the beat interval — 60 divided by the tempo, in seconds — and put "
                    "the release clearly on one side of it. Well under, and the gain is steady "
                    "again before the next hit. Well over, and it never gets the chance to "
                    "ripple. Landing near it is what makes a compressor breathe. Auto-release "
                    "exists precisely to avoid this and is usually the right choice on a bus."
                ),
                needs="comp_single",
            ),
            FixStep(
                action="Reduce the depth before you reduce anything else.",
                detail=(
                    "Pumping scales with how far the gain travels. Two dB of movement at an "
                    "awkward release is a groove; six dB is a fault. Raising the threshold is "
                    "often a faster fix than re-timing anything."
                ),
                needs="comp_single",
            ),
            FixStep(
                action="Fix the low end so it stops demanding the gain reduction in the first "
                       "place.",
                detail=(
                    "Duck the bass from the kick's own signal rather than letting them fight, "
                    "and the combined low end stops spiking. A compressor that is never asked "
                    "for six dB cannot pump."
                ),
                needs="comp_sidechain_ext",
                without=(
                    "Carve a narrow dip in the bass at the kick's fundamental with a static EQ, "
                    "or automate the bass down for the length of each kick. More work than a "
                    "sidechain, and it keeps the two out of each other's way just as well."
                ),
            ),
            FixStep(
                action="Split the work across stages instead of asking one compressor for all "
                       "of it.",
                detail=(
                    "Two dB from a bus compressor and two dB from a limiter breathe far less "
                    "than four dB from either alone, because neither one has to move fast."
                ),
            ),
        ),
        how_to_verify=(
            "By ear first: the sustained element test. A pad, a held note or a reverb tail "
            "should stay steady straight through the kick. On the meter, the gain-reduction "
            "needle should not trace a repeating shape at the beat. On re-analysis, the pumping "
            "index should fall and micro-dynamics should come back up. If the index stays high "
            "but nothing ducks audibly, the measurement was reading your groove and the track "
            "is fine."
        ),
        learn_more=(
            "Here is what 'tempo-locked modulation' actually means, because the phrase hides a "
            "lot. Plot the level of the mix against time and subtract a slow moving average — "
            "that removes the arrangement, since a chorus arriving is a trend rather than a "
            "ripple. What is left is a wiggle. Now run a frequency analysis on that wiggle. Not "
            "on the audio: on the level curve. Repeating movement shows up as energy piled into "
            "narrow bins at the beat rate and its multiples instead of spread evenly, and a "
            "compressor's gain ripple looks exactly like that.\n\n"
            "The measurement is done in dB rather than in raw level for a specific reason. A "
            "compressor *multiplies* the signal by its gain curve, and multiplication turns "
            "into addition once you take a logarithm — which is all dB is. So in dB the "
            "compressor's ripple is simply added on top of the music's own envelope, sitting "
            "there as a clean separable component, instead of being tangled up with it.\n\n"
            "What the analysis still cannot do is tell that ripple apart from a kick pattern, "
            "because a kick pattern produces the same shape. The real difference is that a "
            "compressor applies its ripple to everything in the mix, including parts that are "
            "not playing on the beat, while a kick pattern only moves the kick — and separating "
            "those needs the individual tracks, which a stereo bounce does not contain. Hence "
            "the corroboration requirement, the low confidence score, and the instruction to go "
            "and listen to a pad before you believe any of it."
        ),
        minutes=30,
        resources=(
            youtube_search(
                "compressor release time tempo pumping breathing fix",
                "Hearing pumping, then re-timing the release",
                "Isolated examples of a compressor breathing at the beat rate, plus setting "
                "release against the tempo rather than by feel — which is how it lands there.",
            ),
            youtube_search(
                "compressor sidechain high pass filter stop low end pumping",
                "Stopping the low end from driving the detector",
                "The detector high-pass fix demonstrated: filter the bottom octave out of what "
                "the compressor is listening to and the kick stops ducking the whole mix.",
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Per-stem compression
#
# These two only fire on the deep analysis, because they are measured on a
# separated source rather than on the two-track. That is the whole point of
# them: the master can measure fine while one element inside it has been
# flattened, and no amount of staring at the stereo file will tell you which.
# ---------------------------------------------------------------------------

_add(
    "compression.stem_drums_flat",
    Explainer(
        headline="The drums themselves are squashed — the damage is on the drum bus, not the master.",
        what_it_is=(
            "This is measured on the drums alone, pulled out of your mix as a separate source, "
            "and the number is crest factor: the distance between the loudest instant in the "
            "drums and their own average level. A drum hit is mostly empty space with a very "
            "short spike at the front, so healthy drums measure a long way apart — the spike is "
            "far above the body. Compress them hard and the spike comes down to meet the body, "
            "and the two numbers converge.\n\n"
            "The reason this is worth a finding of its own is that the two-track cannot show it "
            "to you. Your master's crest factor is the drums averaged in with everything else, "
            "and a sustained pad, a bass and a vocal all sitting at a steady level will drag "
            "that average toward the middle regardless of what the drums are doing. So a "
            "flattened drum bus under an otherwise untouched mix and a lightly-glued mix with "
            "healthy drums can produce the same reading on the master. Separating the drums out "
            "is what tells the two apart, and it says the drums are the ones that are flat."
        ),
        what_you_hear=(
            "The drums are loud and they do not hit. You can hear the kick, you can hear the "
            "snare, and neither of them arrives — there is no moment of impact, just a thud "
            "that is already at full level by the time you notice it. The groove stops pulling "
            "you along, because groove is carried by the front edge of each hit and that is the "
            "part that has gone. Turning the drum bus up makes this worse rather than better: "
            "the drums get louder and the mix gets more crowded, and they still do not punch. "
            "Cymbals and room tone come up between the hits, so the kit sounds washier and "
            "further away even though it is measurably louder."
        ),
        why_it_matters=(
            "It is the one form of compression damage that genuinely cannot be repaired later. "
            "Once an attack has been compressed off the source and printed into the mix, "
            "nothing on the master bus puts it back — a transient shaper on the master lifts "
            "the front of everything, including the bass note and the vocal syllable that "
            "happened to land on the same beat, which reads as a different problem rather than "
            "a fix. Mastering can make this louder and change its tone and it cannot make it "
            "punch.\n\n"
            "It also costs you loudness in a way that feels backwards. Punch comes from "
            "contrast, and contrast is exactly what a limiter is cheap to produce: a short "
            "spike costs almost no perceived loudness and buys a lot of impact. Flat drums have "
            "spent that headroom on sustained level instead, so you are paying full price for "
            "energy that the ear does not read as power."
        ),
        common_causes=(
            "A fast attack on the drum bus compressor. Anything under about 10 ms catches the "
            "stick before it gets out, which is precisely the part you wanted to keep.",
            "Compression at every stage — on the kick, on the snare, on the drum bus, on the "
            "mix bus, then the limiter. Each one takes a small bite out of the same transient.",
            "Parallel compression blended in too far, so the crushed copy is loud enough to "
            "fill the gaps the dry drums leave.",
            "Drum samples or loops that arrived already limited. A lot of commercial packs are "
            "printed hot, and you are compressing something that was compressed before you "
            "opened it.",
            "A clipper or saturator on the drum bus driven past the point of shaving peaks, so "
            "it is flattening the whole front of the hit rather than the tip of it.",
        ),
        how_to_fix=(
            FixStep(
                action="Solo the drum bus and bypass its compressor before you change anything.",
                detail=(
                    "You are checking whether the flatness arrived with the processing or with "
                    "the samples. If the bare drums already have no spike, no amount of attack "
                    "tuning will help and the fix is to replace or re-layer the sources."
                ),
            ),
            FixStep(
                action="Slow the attack on the drum bus compressor to 20-30 ms and re-set the amount.",
                detail=(
                    "This is the single move that matters. A compressor's detector takes time "
                    "to respond, and during that window the transient passes at full level. A "
                    "slow attack therefore keeps the spike and compresses the body behind it, "
                    "which pushes the two further apart and is what people mean by a compressor "
                    "adding punch. Then reduce the gain reduction to 2-4 dB on the loud hits, "
                    "released before the next beat lands."
                ),
                needs="comp_single",
            ),
            FixStep(
                action="Get the weight from a parallel path rather than from the main one.",
                detail=(
                    "Compress a copy of the drums hard and blend it underneath the untouched "
                    "drums. The dry path keeps every attack intact and the crushed path adds "
                    "the density, which is a thing serial compression structurally cannot do — "
                    "whatever it removes, it removes from the signal you are keeping."
                ),
                needs="comp_parallel",
                without=(
                    "Send the drum bus to an aux, put a stock compressor on the aux with a fast "
                    "attack and a high ratio, and blend that aux under the untouched drums "
                    "until the weight arrives. Keep the dry path clean."
                ),
            ),
            FixStep(
                action="Rebuild the attack on the individual drums, not on the bus.",
                detail=(
                    "A transient shaper on the kick and the snare separately lets you add the "
                    "front back to each one without lifting the room mics and cymbals with it. "
                    "On the bus it raises everything that starts on that beat, which is how "
                    "drums end up sounding clicky rather than punchy."
                ),
                needs="transient_shaper",
                without=(
                    "No transient shaper: layer a short, clean sample under the kick and snare "
                    "to supply the attack the compressed hit no longer has, and tuck it just "
                    "under the original so you hear the impact rather than the sample."
                ),
            ),
            FixStep(
                action="Let a clipper take the loudest peaks so the compressor and limiter can stop.",
                detail=(
                    "Two or three dB shaved off the very tops by a clipper is nearly inaudible "
                    "on drums and removes the peaks that were making everything downstream work "
                    "hard. The rest of the hit stays the shape it was."
                ),
                needs="clipper",
                without=(
                    "Without a clipper, pull the drum bus fader down a dB or two and make up "
                    "the level after the master limiter instead. Less input means less gain "
                    "reduction on the peaks, which is the same result by a slower route."
                ),
            ),
            FixStep(
                action="Re-render and re-run the deep analysis, not the standard one.",
                detail=(
                    "This finding only exists when the drums can be separated out. The standard "
                    "analysis will show you the master's crest factor, which is the number that "
                    "hid the problem in the first place."
                ),
            ),
        ),
        how_to_verify=(
            "Re-run the deep analysis: the drum stem's crest factor should come back inside "
            "your genre's window, and the master's own crest factor will usually barely move — "
            "that gap is the whole reason this finding exists. By ear, play the drums against "
            "the mix at conversational volume and listen for the stick and the beater rather "
            "than the body. If you can hear what hit the drum, the attack is back. If the "
            "drums now sound thin as well as punchy, you took too much body out along with the "
            "compression — put a little of the parallel path back."
        ),
        learn_more=(
            "Crest factor is peak divided by average, in dB, and on a single source it is close "
            "to a direct description of the sound. An unprocessed close-miked snare measures "
            "somewhere around 18-20 dB. A finished drum bus in a loud modern record usually "
            "sits nearer 10-14. A pure sine wave is 3, and anything approaching that is a "
            "waveform with no events in it at all.\n\n"
            "The useful thing about measuring it per source rather than on the master is that "
            "the master's figure is an average over things with wildly different natural crest "
            "factors — drums are spiky, a pad is nearly flat by nature, a vocal is somewhere "
            "between — and averaging them destroys the information you actually want. A mix "
            "with dead drums and a very dynamic vocal can land on the same master reading as a "
            "mix with live drums and an even vocal, and only one of those needs fixing. "
            "Separating the sources is what turns a summary statistic back into a diagnosis, "
            "and it is why the deep analysis is worth its runtime on a mix that measures fine "
            "and does not hit."
        ),
        minutes=35,
        resources=(
            Resource(
                kind="reference",
                label="Parallel compression, and where the technique came from",
                url="https://en.wikipedia.org/wiki/Parallel_compression",
                note="Why blending a crushed copy under an untouched one keeps the attack that a "
                     "serial compressor would have to remove — the New York drum trick your fix "
                     "steps are describing, set out plainly.",
                source="Wikipedia",
            ),
            youtube_search(
                "drum bus compression attack release punch tutorial",
                "Drum bus attack and release, on real kits",
                "Setting the attack slow enough to let the stick out before the gain clamps, "
                "which is the single move this finding turns on.",
            ),
            youtube_search(
                "transient designer kick snare attack punch",
                "Rebuilding attack per drum rather than across the bus",
                "Transient shaping applied to the kick and snare individually, so the room and "
                "cymbals do not rise with every hit and turn punch into click.",
            ),
        ),
    ),
)

_add(
    "compression.stem_vocals_flat",
    Explainer(
        headline="The vocal has been compressed until it stopped moving, which is why it sounds distant.",
        what_it_is=(
            "This is measured on the lead vocal alone, separated out of your mix, and it is not "
            "about whether the vocal is even from line to line. Evening a vocal out is correct "
            "and every record does it. This measures micro-dynamics: the movement *inside* the "
            "syllables — the difference between the front of a consonant and the vowel behind "
            "it, the small lift on a pushed word, the breath before a phrase. A singer produces "
            "that variation continuously, on a scale of milliseconds, and it is most of what "
            "tells your ear how hard someone is working.\n\n"
            "So the two things a compressor does to a vocal are separable, and only one of them "
            "is the problem. Riding the phrases to a consistent level is macro and it is "
            "wanted. Flattening the shape inside each word is micro and it is what this "
            "finding is. The analysis also requires corroboration before it will say so: the "
            "measured micro-dynamics have to be short of your genre's window *and* the vocal's "
            "own crest collapse and peak pinning have to independently look like several dB of "
            "gain reduction. An unusually even singer is not over-compression, and calling it "
            "that would be a taste judgment wearing a number."
        ),
        what_you_hear=(
            "The vocal sits at one distance for the whole song and never comes toward you. The "
            "big line in the last chorus reads exactly like the first line of the verse — you "
            "can tell the singer sang it harder, and you cannot feel it. Because nothing about "
            "it changes, your ear stops tracking it as a performance and files it as texture, "
            "and it starts to disappear into the arrangement even at a healthy level.\n\n"
            "The specific trap is what happens next: it reads as too quiet, so you push the "
            "fader, and the vocal gets louder without getting any closer. That is the tell. "
            "Distance is carried by dynamic movement, not by level, and raising a flat vocal "
            "only makes it a loud flat vocal. Breaths and mouth noise coming up as loud as the "
            "words is the other giveaway — the compressor is pulling the loud parts down onto "
            "the level of the things that were never meant to be heard."
        ),
        why_it_matters=(
            "The lead vocal is the thing the listener is actually following, and a vocal that "
            "does not move gives them nothing to follow. Every other problem on this list "
            "affects how a mix sounds; this one affects whether the song lands, because "
            "intensity in a vocal performance is communicated almost entirely through "
            "variation and you have removed the variation.\n\n"
            "It is also the specific reason the fader stops working. When a producer says the "
            "vocal is buried and pushing it up does not fix it, over-compression is one of the "
            "two usual causes — masking from the instrumental is the other. And like all "
            "compression damage on a source, it is printed: nothing on the master bus can "
            "restore movement to a vocal that no longer has any, and mastering will make it a "
            "louder flat vocal too."
        ),
        common_causes=(
            "Compressors stacked without counting them: one on the recorded track, one on the "
            "vocal chain preset, one on the vocal bus, then the master limiter. Each is doing a "
            "few dB that sounded fine on its own.",
            "A vocal chain preset. Most of them are built to sound impressive on a solo'd dry "
            "vocal, which means heavy, and the setting that flatters a demo flattens a mix.",
            "Too high a ratio in one stage. Two compressors at 3:1 doing 3 dB each sound "
            "natural; one at 10:1 doing 6 dB does not.",
            "A fast attack, so the front of every consonant is caught before it gets out — the "
            "vocal goes soft-edged and mumbly as well as flat.",
            "Compressing to fix a level problem that belongs in the edit. If the quiet phrases "
            "were never lifted by hand, the compressor is being asked to close a gap far bigger "
            "than it should be closing, and it crushes the loud lines to get there.",
            "Limiting or clipping the vocal bus for loudness, which removes the peaks that were "
            "carrying the sense of effort.",
        ),
        how_to_fix=(
            FixStep(
                action="Bypass every compressor in the vocal chain and listen to the raw vocal in the mix.",
                detail=(
                    "You need to hear what you started with before you decide how much of it to "
                    "give back. Note where it genuinely disappears — those are the places that "
                    "need work, and they are usually far fewer than the compression you have "
                    "on it implies."
                ),
            ),
            FixStep(
                action="Fix the level in the edit first, with clip gain, before any compressor.",
                detail=(
                    "Clip gain is the volume control on the audio itself, ahead of the fader "
                    "and ahead of the plugins. Go through and lift the quiet phrases and pull "
                    "the shouted ones, phrase by phrase, until the vocal is roughly even before "
                    "anything is compressing it. This is the move that lets you use half as "
                    "much compression, because the compressor is now shaping a performance "
                    "instead of rescuing one — and it costs nothing but time."
                ),
            ),
            FixStep(
                action="Rebuild the chain as two gentle stages instead of one heavy one.",
                detail=(
                    "An opto compressor first, slow and level-dependent, doing 2-3 dB to smooth "
                    "the phrases, then a faster one after it catching only the syllables that "
                    "still poke out, also 2-3 dB. Two stages each barely working sound like a "
                    "vocal; one stage doing six sounds like a compressor."
                ),
                needs="comp_opto",
                without=(
                    "Two instances of your stock compressor in series, ratio about 2.5:1, "
                    "3 dB of reduction each, the first with a slow attack (10-20 ms) and the "
                    "second faster. The point is the total, split across two gentle stages "
                    "rather than taken by one aggressive one."
                ),
            ),
            FixStep(
                action="Slow the attack until you can hear the consonants again.",
                detail=(
                    "Under about 5 ms the compressor is clamping down on the front of each "
                    "word, which is where the intelligibility and the sense of effort both "
                    "live. Start at 10-20 ms and listen specifically to the T and K sounds."
                ),
                needs="comp_single",
            ),
            FixStep(
                action="Get the loud lines under control with automation rather than more ratio.",
                detail=(
                    "Ride the fader down through the last chorus by a dB or two instead of "
                    "asking the compressor to absorb it. Automation removes the level "
                    "difference without touching what happens inside the words, which is "
                    "exactly the distinction this finding is about."
                ),
            ),
            FixStep(
                action="Check whether the master limiter is the thing flattening it.",
                detail=(
                    "In a dense mix the vocal is usually the loudest centered element, so it "
                    "takes the most gain reduction — the limiter ends up riding the vocal "
                    "specifically. Bypass the master chain and compare. If the vocal moves "
                    "again without it, the fix is less drive into the limiter, not less "
                    "compression on the vocal."
                ),
                needs="limiter",
            ),
        ),
        how_to_verify=(
            "Re-run the deep analysis: the vocal stem's micro-dynamics should come back inside "
            "your genre's window and the estimated gain reduction on it should drop. By ear, "
            "the test is a comparison rather than a level — play the first line of the verse "
            "and then the biggest line of the last chorus back to back and check that the "
            "second one feels harder sung, not just louder. Then check the fader works again: "
            "a healthy vocal moves closer when you raise it. If breaths and mouth noise are "
            "now noticeably quieter than the words, you have taken enough compression off."
        ),
        learn_more=(
            "The reason movement reads as distance has nothing to do with loudness. Your ear "
            "judges how far away a sound source is mostly from cues that have no relationship "
            "to level: how much reverb arrives with it relative to the direct sound, how much "
            "high frequency has survived the air between you, and how much the source varies "
            "moment to moment. A voice close to you is dynamically alive, because you are "
            "hearing the raw output of the throat with nothing between you to smooth it. A "
            "voice across a room has been averaged by the space it crossed. So a compressor "
            "that removes variation is, as far as the ear is concerned, simulating distance — "
            "and turning it up does not undo that any more than turning up a distant sound "
            "makes it close.\n\n"
            "This is also why the fix is nearly always upstream of the compressor rather than "
            "in it. The reason people over-compress vocals is that they are using one control "
            "to solve two different problems: the phrase-to-phrase level, which is an editing "
            "problem, and the syllable-to-syllable shape, which the compressor is supposed to "
            "leave mostly alone. Do the first with clip gain and automation, and the amount of "
            "compression you need for the second turns out to be small."
        ),
        minutes=30,
        resources=(
            Resource(
                kind="reference",
                label="Paul White on keeping a vocal alive under compression",
                url="https://www.soundonsound.com/techniques/vocal-production",
                note="Level automation before the compressor, moderate ratios, and parallel "
                     "compression for weight — an engineer arriving at the same order of "
                     "operations as the fix steps here, and saying why.",
                source="Sound On Sound",
            ),
            youtube_search(
                "clip gain vocal editing before compression tutorial",
                "Clip gain, phrase by phrase",
                "The editing pass that halves how much compression the vocal needs: lifting the "
                "quiet lines by hand before any plugin gets to see them.",
            ),
            youtube_search(
                "vocal compression two compressors in series 1176 la2a",
                "Two gentle stages instead of one heavy one",
                "The opto-then-fast chain from the fix steps, demonstrated with each stage doing "
                "only a couple of dB — the shape to rebuild your vocal chain toward.",
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(target: Dict[str, Explainer]) -> None:
    """Copy this module's explainers into `target`.

    Called by `knowledge.py` after its own definitions, so the import only ever
    runs one way and the two modules cannot deadlock on each other.
    """
    target.update(EXPLAINERS)
