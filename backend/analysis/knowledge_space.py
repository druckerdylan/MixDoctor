"""Phase, stereo width and low end — what they mean and what to do about them.

These are the findings people get most wrong, because all three are about
things you cannot hear on the system you mixed on. A polarity flip sounds
"wide" on headphones. Stereo bass sounds "big" in the room. A section that
lost its bottom sounds fine on a laptop. Every one of them is a problem that
only exists somewhere else — in a club, on a phone, after a mono fold-down —
which is exactly why a number on a dashboard does not help anyone.

So the writing here leads with the sound and the situation, and puts the
physics at the end for whoever wants it. The measured values live on the
finding; nothing in this file cites one, because the same words ship with
every track.

Registered into `knowledge.EXPLAINERS` by `register()`. Nothing here imports
that dict at module scope — knowledge.py owns it and calls us.
"""

from __future__ import annotations

from typing import Dict

from .knowledge import Explainer, FixStep


EXPLAINERS: Dict[str, Explainer] = {}


def _add(finding_id: str, explainer: Explainer) -> None:
    EXPLAINERS[finding_id] = explainer


# ---------------------------------------------------------------------------
# Phase
# ---------------------------------------------------------------------------

_add(
    "phase.polarity_inverted",
    Explainer(
        headline="One channel is upside down. Anywhere the two get added together, most of this mix deletes itself.",
        what_it_is=(
            "A waveform is a stream of numbers that swing above and below zero — the "
            "speaker cone pushing out, then pulling in. Polarity inversion means one "
            "channel's numbers have had their sign flipped: every push became a pull. "
            "The sound of that channel on its own is unchanged, because your ear does not "
            "care which way a cone moves first. But add the two channels together and the "
            "flipped one subtracts from the other instead of adding to it. Anything the "
            "two channels share — the vocal, the kick, the bass, everything centred — "
            "arrives as a number and its own negative, and cancels to silence."
        ),
        what_you_hear=(
            "On headphones, an odd, hollow spaciousness with no middle to it. The vocal "
            "seems to come from beside your head or from everywhere at once rather than "
            "from a point in front of you, and you cannot say where the kick is. It can "
            "read as impressively wide, which is why this ships. Sum it to mono and the "
            "illusion is over: the record largely disappears and what is left is the thin "
            "residue of whatever was panned off centre."
        ),
        why_it_matters=(
            "This is not a width choice with a downside, it is a fault. A club plays a "
            "summed signal to the subs and often to the whole room; a phone has one "
            "speaker; most Bluetooth speakers sum; a DJ mixer has a mono button that gets "
            "used. In every one of those situations the parts of the mix you spent the "
            "most time on are the parts that vanish, because they are the centred ones. "
            "It is also the cheapest problem on this list to fix — usually one button."
        ),
        common_causes=(
            "A balanced cable with pins 2 and 3 swapped on one side of a stereo pair, in "
            "an outboard insert or a patchbay.",
            "A polarity button left engaged on one channel of a mid/side or dual-mono "
            "chain, often from an earlier experiment.",
            "A cheap 'mono to stereo' or widener plugin that fakes width by inverting one "
            "side — that trick produces exactly this measurement.",
            "A stereo sample or one-shot that is a mono sound duplicated with one copy "
            "flipped, sold as 'wide'.",
            "A single element inverted upstream (a bottom snare mic, a re-amped guitar) on "
            "something loud enough to drag the whole correlation negative.",
        ),
        how_to_fix=(
            FixStep(
                action="Confirm it in two seconds: sum the master to mono and listen.",
                detail=(
                    "If the mix collapses to a thin shell instead of just losing its "
                    "space, this is real. A healthy mix in mono gets narrower and slightly "
                    "quieter; it does not lose its vocal and its bass."
                ),
            ),
            FixStep(
                action="Flip the polarity of the right channel and check mono again.",
                detail=(
                    "Every DAW has a polarity or phase-invert switch on its utility, gain "
                    "or channel strip plugin — the one marked with a circle and a slash. "
                    "If mono immediately fills back in, you are done."
                ),
            ),
            FixStep(
                action="If flipping the master does not fix it, the inversion is inside the mix.",
                detail=(
                    "That means one element is flipped, not one channel. Solo buses in "
                    "mono one at a time; the guilty one is the bus whose level drops when "
                    "you mono it. Fix it on that track, not on the master — flipping the "
                    "master would just move the fault onto everything else."
                ),
            ),
            FixStep(
                action="Watch a correlation meter while you do it.",
                detail=(
                    "A polarity-inverted master pins the meter hard negative. Fixed, it "
                    "should sit positive and mostly high. The meter finds this faster than "
                    "your ears do, because your ears are being fooled by the width."
                ),
                needs="meter_correlation",
                without=(
                    "No correlation meter — use the mono button as a switch instead. "
                    "Toggle stereo to mono every few bars and listen for level "
                    "disappearing rather than space disappearing."
                ),
            ),
            FixStep(
                action="Re-render and re-analyse before you send it anywhere.",
                detail=(
                    "A polarity flip can also live in the export path — a hardware insert "
                    "or a bounce through outboard. Only a measurement of the actual "
                    "rendered file proves the shipped audio is right."
                ),
            ),
        ),
        how_to_verify=(
            "Correlation should read positive across the track instead of pinned negative, "
            "and folding to mono should cost roughly 3 dB or less rather than gutting the "
            "mix. By ear: in mono, the vocal, kick and bass should all still be there and "
            "still be the loudest things."
        ),
        learn_more=(
            "Correlation is one number describing how much the two channels have in "
            "common, and it runs from +1 to -1. At +1 the channels are identical — a mono "
            "file, or anything panned centre. At 0 they are unrelated, which is what a "
            "genuinely wide mix reads on its busiest moments and is completely healthy; "
            "summing two unrelated signals costs about 3 dB because they add as energy "
            "rather than as level. At -1 they are mirror images, and summing them gives "
            "you zero. That is the whole mechanism.\n\n"
            "Worth separating two words that get used interchangeably. Polarity is a sign "
            "flip applied to the entire signal at once — every frequency, every instant. "
            "Phase is a time relationship, and a delay shifts different frequencies by "
            "different amounts of their own cycle, which is why a delayed copy cancels "
            "some frequencies and reinforces others rather than cancelling all of them. "
            "Polarity is a switch. Phase is a slider. This finding is the switch."
        ),
        minutes=15,
    ),
)

_add(
    "phase.mono_incompatible",
    Explainer(
        headline="Play this anywhere the two channels get added together and part of the mix quietly goes missing.",
        what_it_is=(
            "Mono fold-down is what happens when a system adds your left and right "
            "channels into one. Two channels that are genuinely different — real stereo "
            "content — lose about 3 dB when you do that, and that loss is pure arithmetic, "
            "not damage. Beyond that, something is cancelling: parts of the two channels "
            "are opposite enough that adding them subtracts. This finding fires when the "
            "fold-down costs more than arithmetic explains, or when the two channels have "
            "less in common than this genre's records normally do."
        ),
        what_you_hear=(
            "In stereo, a big, open mix. In mono, the balance changes rather than just the "
            "space: specific elements drop in level or disappear — usually the widened "
            "ones, so doubled guitars, wide pads, backing vocals, reverb tails and hats. "
            "Everything else appears to jump forward because the things around it got "
            "quieter. If you have ever mono'd a mix and thought 'the vocal suddenly sounds "
            "too loud', the vocal did not change; its surroundings cancelled."
        ),
        why_it_matters=(
            "Mono did not go away, it moved. Club systems sum the low end to the subs and "
            "small rooms often run fully mono; phones have one speaker; laptop speakers "
            "sit centimetres apart and behave like one; most portable Bluetooth speakers "
            "sum; smart speakers, TVs, shop and venue ceiling systems, and radio all sum. "
            "That is a large share of the listening your record will actually get, and it "
            "is getting a different balance from the one you approved. The mono check also "
            "catches cancellation that is quietly making your stereo mix mushy — if a "
            "doubled guitar loses 6 dB in mono, it is partly cancelling in stereo too."
        ),
        common_causes=(
            "A stereo widener or imager on the master or a bus, pushed past what it can "
            "do without cancelling.",
            "Haas widening: a copy of a mono source delayed by a few milliseconds and "
            "panned opposite. Wide in stereo, comb-filtered to pieces in mono.",
            "A chorus, doubler or 'stereo spread' run at full wet on a bus rather than as "
            "an effect on one part.",
            "A mid/side EQ boosting the side channel broadly to add air or size.",
            "Two mics on one source at different distances — DI and amp, top and bottom "
            "snare — panned apart and never time-aligned.",
            "Sample-pack loops and one-shots labelled 'wide' that are a mono sound with "
            "one side delayed or flipped.",
        ),
        how_to_fix=(
            FixStep(
                action="Find the culprit by soloing buses in mono, not in stereo.",
                detail=(
                    "Put the master in mono and solo each bus in turn while toggling mono "
                    "on and off. A healthy bus narrows. The guilty one loses level. Two "
                    "minutes of this beats an hour of guessing."
                ),
            ),
            FixStep(
                action="Put a correlation meter on the master and on each suspect bus.",
                detail=(
                    "You are looking for a bus that sits near zero or below while the "
                    "others sit high. Watch it through a full section — cancellation is "
                    "often intermittent, showing up only where the widened part plays."
                ),
                needs="meter_correlation",
                without=(
                    "No correlation meter: A/B each bus stereo against mono and judge by "
                    "level, not by width. Anything that gets quieter rather than narrower "
                    "is cancelling."
                ),
            ),
            FixStep(
                action="Replace fake width with real width on whatever you find.",
                detail=(
                    "Two separately performed takes panned apart are decorrelated but "
                    "never cancel, because they are different waveforms, not the same one "
                    "delayed. Same for two different synth voices, or two different mic "
                    "positions on genuinely different sources. Copies cancel; performances "
                    "do not."
                ),
            ),
            FixStep(
                action="Widen per band instead of across the whole spectrum.",
                detail=(
                    "Leave everything below roughly 120 Hz alone, keep the low mids modest, "
                    "and take whatever width you want from the top end where cancellation "
                    "costs the least and the ear notices it most."
                ),
                needs="imager",
                without=(
                    "Without an imager, turn the width down at its source instead: the wet "
                    "or mix control on whatever created it, or the level of the doubled "
                    "copy. A few dB usually recovers the fold-down entirely."
                ),
            ),
            FixStep(
                action="Make space in the centre rather than pushing things out of it.",
                detail=(
                    "A narrow cut in the mid channel where the wide part lives lets the "
                    "sides be heard without any of them being processed wider. It costs "
                    "nothing on fold-down, because you removed something rather than "
                    "adding a difference signal."
                ),
                needs="eq_mid_side",
                without=(
                    "Without mid/side EQ, pan real elements apart to open the centre — "
                    "move a doubled part, a percussion layer or a keys track off centre "
                    "instead of processing width into anything."
                ),
            ),
        ),
        how_to_verify=(
            "Folding to mono should cost about 3 dB or less, and correlation should stay "
            "positive through the track. The real test is by ear: switch between stereo "
            "and mono and the *balance* should not change. You should lose the space and "
            "keep the mix."
        ),
        learn_more=(
            "Correlation runs +1 to -1. At +1 the channels are identical, at 0 they are "
            "unrelated, at -1 they are mirror images and sum to nothing. The useful part is "
            "that 0 is not a failure — a wide, healthy record spends plenty of time near "
            "zero, because unrelated is not the same as opposite. Two unrelated signals "
            "combine as power rather than as amplitude, which is where the fixed 3 dB "
            "fold-down cost comes from; two opposite signals combine as nothing.\n\n"
            "The catch is that correlation is one broadband number averaged over time, so "
            "it is dominated by whatever is loudest. A mix can read comfortably positive "
            "while one narrow region cancels completely underneath it, because that region "
            "holds a small share of the total energy. That is a separate finding — band "
            "cancellation — and it is the reason a mix can pass the headline mono check "
            "and still lose its bass in a club."
        ),
        minutes=25,
    ),
)

_add(
    "phase.band_cancellation",
    Explainer(
        headline="One frequency range disappears in mono while everything around it survives — which is why the overall phase reading looks fine.",
        what_it_is=(
            "Plenty of playback systems add your left and right channels together into one "
            "signal — that is mono, and it is what a phone speaker, a Bluetooth speaker or a "
            "club's subs are given. Doing that always costs a couple of dB, about 3, and "
            "that much is harmless arithmetic. Losing much more than that means parts of the "
            "two channels are opposite enough that adding them subtracts instead.\n\n"
            "The catch this finding exists for: the usual phase reading is one number "
            "averaged across the whole spectrum and dominated by whatever is loudest. A "
            "well-centred mix can therefore look completely healthy while one narrow "
            "frequency range cancels away to nothing underneath it, simply because that "
            "range holds a small share of the total energy. So each band is folded down to "
            "mono separately here, and the one losing far more than its share is reported."
        ),
        what_you_hear=(
            "In stereo, nothing wrong. In mono, one specific thing goes wrong and it "
            "depends where the cancelling band is. Down low, the bottom drops out and the "
            "kick loses weight while the rest of the mix is untouched. In the low mids, the "
            "mix goes hollow or nasal, like it is being played through a tube. Up top, the "
            "air and the reverb tails vanish and cymbals turn dull and dry. If a delayed "
            "copy is the cause, you may also hear a static metallic colouration in stereo — "
            "a flanger frozen in one position."
        ),
        why_it_matters=(
            "This is harder to catch than a whole-mix phase problem precisely because "
            "everything else is fine, so it passes every casual check and ships. Then the "
            "system that sums — a club, a phone, a Bluetooth speaker — removes one layer of "
            "your mix and leaves the rest, which sounds less like a phase fault and more "
            "like a bad mix. When the affected band is the bottom, it is worse: the low end "
            "is the part you most need to survive summing, and it is also the part with the "
            "least energy spread to hide behind."
        ),
        common_causes=(
            "A doubled or widened source: the same audio present twice with one copy "
            "delayed, so specific frequencies cancel and others reinforce.",
            "A stereo reverb or delay on a mono element, where the wet dominates in one "
            "region — reverb tails are heavily decorrelated by design.",
            "A stereo-ising plugin on a single instrument, most damagingly on a bass, pad "
            "or 808.",
            "A 'wide' stereo sample whose two sides are near-copies with a small offset.",
            "Two mics on one source at different distances, never time-aligned — the "
            "distance sets exactly which frequencies cancel.",
            "Different EQ settings on the left and right of a pair: an ordinary EQ rotates "
            "phase around each band, so unmatched curves misalign the two channels in "
            "exactly the region you boosted.",
        ),
        how_to_fix=(
            FixStep(
                action="Isolate the band before you hunt for the source.",
                detail=(
                    "Put the master in mono, then high-pass and low-pass around the region "
                    "the finding names so you are hearing only that. Now mute tracks one at "
                    "a time. The band will jump in level when the offender goes."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="If the cancelling band is below roughly 120 Hz, mono the low end.",
                detail=(
                    "Set the crossover just above where the bass fundamental lives. This is "
                    "not a compromise down there — you cannot localise those frequencies "
                    "anyway, so nothing is lost and the cancellation goes with it."
                ),
                needs="mono_maker",
                without=(
                    "No mono maker: high-pass the side channel with a mid/side EQ around "
                    "120 Hz. With neither, do it at the source — take the stereo processing "
                    "off the bass and the kick and pan them dead centre."
                ),
            ),
            FixStep(
                action="Try the polarity trick on any source that exists twice.",
                detail=(
                    "Flip the polarity of one copy — the second mic, the double, the "
                    "widened layer — and listen in mono. Whichever way round is louder is "
                    "the correct one. This takes five seconds and resolves half of these "
                    "cases outright, because a straight inversion is the most common single "
                    "cause."
                ),
            ),
            FixStep(
                action="Time-align two mics on one source instead of EQing around the damage.",
                detail=(
                    "Nudge the further mic earlier until the low end is strongest in mono. "
                    "You are matching the arrival times; when they line up the cancellation "
                    "notches move out of the audible range instead of being filtered."
                ),
            ),
            FixStep(
                action="Where the source is a delay-based double, change the delay or replace it.",
                detail=(
                    "Moving the offset moves the cancellation to different frequencies, "
                    "which is a dodge rather than a fix. A second recorded take, or a "
                    "different synth voice, gives you the same width with no cancellation at "
                    "all."
                ),
            ),
            FixStep(
                action="Take the width out of that band and give it back somewhere safer.",
                detail=(
                    "Narrow the side channel in the affected region and widen a region that "
                    "is not cancelling — usually above 3 kHz. The mix keeps its size and "
                    "stops losing that layer when it is summed."
                ),
                needs="eq_mid_side",
                without=(
                    "Without mid/side EQ, reduce the wet level of whatever is widening that "
                    "region and add the sense of space back with a reverb send on the parts "
                    "you want wide, keeping their dry signal centred."
                ),
            ),
        ),
        how_to_verify=(
            "Fold each band to mono again: the offending band should lose no more than "
            "about 3 dB, like the rest. Practically, put a spectrum analyser on the master "
            "and switch between stereo and mono — the shape of the curve should stay the "
            "same and just drop slightly, rather than developing a dent."
        ),
        learn_more=(
            "This is comb filtering, and it is worth understanding because it explains most "
            "phase problems you will ever meet. Add a signal to a delayed copy of itself "
            "and each frequency is affected differently, because a fixed delay is a "
            "different fraction of the cycle at every frequency. Where the delay equals a "
            "whole number of cycles the two copies reinforce and you get up to +6 dB; where "
            "it equals half a cycle they oppose and you get a deep null. Those nulls repeat "
            "evenly up the spectrum, which is what the 'comb' refers to.\n\n"
            "The spacing is set by the delay. A 1 ms offset puts its first null at 500 Hz "
            "with more every 1 kHz above it. A 10 ms offset nulls at 50 Hz and every 100 Hz "
            "thereafter — dozens of notches through the musical range. This is why "
            "delay-based widening is destructive in a patterned, per-band way rather than "
            "just making things quieter, why the damage moves when you change the delay "
            "time, and why the cure is alignment or a genuinely different signal, never EQ."
        ),
        minutes=30,
    ),
)


# ---------------------------------------------------------------------------
# Stereo width
# ---------------------------------------------------------------------------

_add(
    "stereo_width.too_wide",
    Explainer(
        headline="The mix is spread wider than it can survive, and the extra width turns into missing level as soon as anything sums it.",
        what_it_is=(
            "Any stereo signal can be described two ways. Left and right is one. Mid and "
            "side is the other: mid is what the two channels have in common, side is the "
            "difference between them. Width here is the ratio of side energy to mid energy "
            "— zero is mono, and the higher it climbs the more of the record lives in the "
            "difference rather than in the shared part. That matters because a lot of "
            "playback adds the two channels together into one — phones, most Bluetooth "
            "speakers, club subs. When that happens the shared part is delivered and the "
            "difference part is thrown away, so 'how wide is this' and 'how much of this "
            "survives being summed' are the same measurement read from two ends."
        ),
        what_you_hear=(
            "On headphones it is impressive: open, enveloping, everywhere. Headphones never "
            "sum, which is why this keeps getting approved. On speakers the centre feels "
            "hollow — the vocal and kick lack authority, the mix seems to happen around you "
            "rather than in front of you, and turning it up makes it louder without making "
            "it bigger. You also lose the ability to point at anything; everything is "
            "smeared across the field instead of occupying a place."
        ),
        why_it_matters=(
            "Perceived power lives in the centre, so a side-heavy mix has to be pushed "
            "harder to sound as loud as a competitor, and the pushing costs you dynamics. "
            "On anything that sums, the side energy is not merely quieter, it is gone. And "
            "the widening usually crowds out what should be centred: vocal, kick, snare and "
            "bass all belong in the mid, and there is only so much room there.\n\n"
            "This is genre-dependent to an extreme degree. Ambient and cinematic records run "
            "near the top of the scale by design, with the two channels sharing very little "
            "— that is the point of the music, not a defect in it. Trap and hip-hop sit far "
            "narrower, at roughly a quarter of the width, because the record is a centred "
            "808, a centred kick and a centred vocal, and width is hats and ad-libs. 'Too "
            "wide' here means wide for what this genre's records do, not wide in the "
            "abstract."
        ),
        common_causes=(
            "A widener or imager on the master doing the job the arrangement should do.",
            "Widening applied across the whole spectrum, so the bass is being spread along "
            "with the hats.",
            "Every synth patch left on its factory unison setting, so nothing in the "
            "arrangement is genuinely centred.",
            "Stereo reverb louder than it seems, especially on the drum bus, pushing energy "
            "into the sides on every hit.",
            "Doubling by short delay rather than by a second performance.",
            "A mid/side EQ boost on the side channel used to add 'air'.",
            "Judging width on headphones, where side content is heard directly instead of "
            "as a room impression.",
        ),
        how_to_fix=(
            FixStep(
                action="Take the master widener off first and see what you actually have.",
                detail=(
                    "Everything else you do is guesswork while a broadband widener is "
                    "sitting on the output. Bypass it, let your ears adjust for a minute, "
                    "and judge the mix underneath."
                ),
            ),
            FixStep(
                action="Get the width from the arrangement instead of from a processor.",
                detail=(
                    "Width made of different signals in different places costs nothing on "
                    "fold-down. Width made of one signal processed to sound like two costs "
                    "you level. Pan real parts apart before you reach for anything."
                ),
            ),
            FixStep(
                action="Widen per band, never broadband.",
                detail=(
                    "Nothing below about 120 Hz, restraint through the low mids where most "
                    "of the mix's energy lives, and most of your width above 3 kHz. That "
                    "distribution sounds wider than a broadband setting at half the "
                    "fold-down cost."
                ),
                needs="imager",
                without=(
                    "Without an imager, work on the sources: pull back the wet on the "
                    "reverbs, choruses and doublers a couple of dB at a time and check "
                    "whether the centre gets denser. It will."
                ),
            ),
            FixStep(
                action="Force the low end to mono.",
                detail=(
                    "Crossover somewhere around 120-150 Hz, just above the bass "
                    "fundamental. This alone often brings the width figure back inside the "
                    "window, because low-frequency side energy is heavy."
                ),
                needs="mono_maker",
                without=(
                    "No mono maker: high-pass the side channel with a mid/side EQ, or "
                    "failing that remove stereo processing from the bass, kick and any "
                    "sub-carrying synth and pan them centre."
                ),
            ),
            FixStep(
                action="Lift the centre rather than pushing the sides.",
                detail=(
                    "A small boost in the mid channel where the lead lives makes the mix "
                    "feel wider, because width is a contrast between centre and sides and "
                    "you can create it from either direction. Boosting the mid is free on "
                    "fold-down; boosting the sides is not."
                ),
                needs="eq_mid_side",
                without=(
                    "Without mid/side EQ, get the same contrast by raising the centred "
                    "elements — vocal, kick, snare, bass — a touch and lowering the wide "
                    "ones, rather than processing anything."
                ),
            ),
            FixStep(
                action="A/B against a record from the same genre at matched loudness.",
                detail=(
                    "Width is the parameter you are least able to judge in isolation, "
                    "because your ear normalises to whatever it has been hearing. Ten "
                    "seconds of a reference resets it."
                ),
                needs="reference_matching",
                without=(
                    "Import a commercial track from the same genre onto a muted channel, "
                    "match its loudness by ear, and switch between them. Do it in mono as "
                    "well as stereo — the difference is stark once levels match."
                ),
            ),
        ),
        how_to_verify=(
            "Side-to-mid should land inside the genre's window, and summing to mono should "
            "cost only the couple of dB that summing always costs, not more. By ear: in "
            "mono, the balance between vocal, "
            "kick and everything else should be recognisably the mix you approved in "
            "stereo. If mono changes who is loudest, you are not done."
        ),
        learn_more=(
            "The mid/side transform is just arithmetic and it is completely reversible: mid "
            "is (L+R)/2, side is (L-R)/2, and you get back left as mid plus side and right "
            "as mid minus side. Nothing is lost in the conversion. What matters is that a "
            "mono system delivers the mid and discards the side entirely, so 'how wide is "
            "this' and 'how much of this survives mono' are literally the same question.\n\n"
            "There is a second reason to keep width out of the bottom. You locate sounds "
            "using the level difference and the timing difference between your two ears. A "
            "50 Hz wave is about seven metres long and your ears are twenty centimetres "
            "apart, so at that wavelength there is almost no level difference to detect and "
            "the timing difference is a negligible fraction of a cycle. Below roughly 120 "
            "Hz you are essentially not hearing direction at all — you hear where the "
            "harmonics come from. Spreading the fundamental therefore buys no perceptible "
            "space while costing you everything on fold-down."
        ),
        minutes=25,
    ),
)

_add(
    "stereo_width.too_narrow",
    Explainer(
        headline="Everything is stacked in the same place, so nothing has room to be heard.",
        what_it_is=(
            "The same mid/side measurement as its opposite: side energy — the difference "
            "between the channels — against mid energy, which is what they share. A low "
            "figure means the two channels are nearly the same signal. That is a real "
            "stereo file behaving almost like a mono one, with every part competing for the "
            "single position between the speakers.\n\n"
            "One thing to keep in view while fixing it: a lot of playback adds the two "
            "channels back together into one — phones, most Bluetooth speakers, club subs — "
            "and when that happens the shared part is delivered and the difference part is "
            "discarded. So there is a right and a wrong way to add width, and the difference "
            "between them only shows up on those systems."
        ),
        what_you_hear=(
            "Small, and congested in a way that EQ does not fix. Sounds appear to come from "
            "one point rather than from a field, parts mask each other because they are all "
            "in the same place, and reverb registers as a level rather than as a space. The "
            "giveaway is that turning it up makes it louder but never bigger — bigness "
            "comes from a spread the mix does not have."
        ),
        why_it_matters=(
            "Masking is positional as well as spectral. Two sounds sharing a frequency range "
            "are far easier to tell apart when they arrive from different directions, so a "
            "narrow mix has to be EQ-carved much harder to reach the same clarity, and the "
            "carving costs you tone. Most listening now happens on headphones and earbuds, "
            "where the absence of width is unmistakable.\n\n"
            "But check the genre before you act. Lo-fi and trap sit narrow legitimately, "
            "because the record is a centred 808, a centred kick and a centred vocal — a "
            "narrow trap mix is usually correct. Ambient, cinematic and orchestral want "
            "several times as much. This finding compares you against your genre, so the "
            "question to ask is not 'is this wide enough' but 'do the records I am competing "
            "with sit wider'."
        ),
        common_causes=(
            "Everything programmed or recorded in mono and left at centre pan, with small "
            "10-20% pans that crowd the middle without creating any separation.",
            "A mono maker or utility with its crossover set far too high — mono up to 500 "
            "Hz collapses guitars, keys and low synths.",
            "A widener left on a narrowing setting, or a mono switch engaged on the master "
            "bus and forgotten.",
            "A stereo source printed from only one of its channels.",
            "All reverbs and delays on mono sends returning to the centre, so the space is "
            "in the same place as the source.",
            "Heavy limiting: it turns down the loud centred hits and drags everything else "
            "with them, which compresses the image toward the middle.",
        ),
        how_to_fix=(
            FixStep(
                action="Pan properly, and commit.",
                detail=(
                    "Find the parts that duplicate a role — a guitar double, two hat layers, "
                    "backing vocals, a percussion pair — and put them opposite each other, "
                    "far out. Timid pans give you crowding without separation. This is the "
                    "one way of getting width that costs nothing when the mix is summed."
                ),
            ),
            FixStep(
                action="Check what your mono maker or utility is set to.",
                detail=(
                    "Nothing above about 150 Hz should be forced mono. A crossover left high "
                    "is a very common single cause of a narrow reading, and it is a one-knob "
                    "fix."
                ),
            ),
            FixStep(
                action="Put the effects out to the sides and keep the dry signal centred.",
                detail=(
                    "A short offset delay panned opposite the source, or a ping-pong, gives "
                    "you width that is genuinely different audio on each side. Keep the dry "
                    "in the middle so the source stays locatable and the width comes from "
                    "the space around it."
                ),
                needs="delay",
            ),
            FixStep(
                action="Widen above the low mids only, in small steps, checking mono each time.",
                detail=(
                    "Work upward from about 300 Hz. Small moves: the setting where you can "
                    "clearly hear the widening is usually well past the setting where it "
                    "starts costing you level in mono."
                ),
                needs="imager",
                without=(
                    "No imager: take the width from real stereo sources instead — a stereo "
                    "reverb return, a second recorded take panned opposite, or two different "
                    "synth voices playing the same part."
                ),
            ),
            FixStep(
                action="Lift the top of the side channel rather than the whole side signal.",
                detail=(
                    "A gentle high shelf on the sides opens the mix without touching the "
                    "region where cancellation actually hurts. Broadband side boosts do the "
                    "opposite: they add the most where the most energy is."
                ),
                needs="eq_mid_side",
                without=(
                    "Without mid/side EQ, achieve the same by sending only the bright "
                    "elements — hats, air, top of the guitars — to a stereo reverb or short "
                    "delay, leaving the low and mid content dry."
                ),
            ),
        ),
        how_to_verify=(
            "Side-to-mid should sit inside the genre's window. The better test: switch to "
            "mono and you should lose the space and nothing else. If mono suddenly sounds "
            "clearer or more solid than stereo, the width you added came from a processor "
            "rather than from the arrangement, and you have traded one problem for another."
        ),
        learn_more=(
            "Width is a clarity tool, not decoration. Two sounds occupying the same "
            "frequency range mask each other far less when they arrive from different "
            "directions — the auditory system uses the difference between your two ears to "
            "pull them apart, and the effect is worth several dB of apparent separation. "
            "That is why a well-panned mix needs less EQ carving than a centred one: you got "
            "the separation for free from geometry instead of paying for it in tone.\n\n"
            "The distinction that matters when you add width is where it comes from. Width "
            "built from genuinely different signals — two performances, two voices, a real "
            "stereo room — is unrelated between the channels but never opposite, so summing "
            "to mono costs it the couple of dB that summing always costs and nothing more. "
            "Width built from copies of the same signal, "
            "delayed or inverted, is opposite by construction and cancels. Both read as "
            "'wider' on a meter. Only one of them still exists on a phone."
        ),
        minutes=30,
    ),
)

_add(
    "stereo_width.channel_imbalance",
    Explainer(
        headline="The whole mix leans to one side, so the centre of your image is not in the centre.",
        what_it_is=(
            "The average level of the left channel against the right, measured across the "
            "entire track. A mix with a centred vocal, kick, snare and bass should come out "
            "close to even, because those elements dominate and they are supposed to be in "
            "the middle. This is not a guitar solo panned right for eight bars — a "
            "whole-file average washes that out. It is a constant offset that is present all "
            "the way through."
        ),
        what_you_hear=(
            "Very often, nothing at all. Your hearing rebuilds a 'centre' wherever the "
            "energy is within a few seconds of listening, and then everything sounds normal. "
            "That adaptation is exactly why this survives to release. What you notice "
            "instead is indirect: you keep nudging pans on other elements to compensate, the "
            "vocal never quite locks in the middle, and on headphones one ear feels like it "
            "is working harder than the other by the end of the track."
        ),
        why_it_matters=(
            "Every pan decision you make after the offset exists is slightly wrong, because "
            "you are placing things relative to a centre that has moved. A listener with one "
            "earbud in gets a materially different mix depending on which ear they chose. "
            "Vinyl cutting is affected, since an off-centre image loads the two groove walls "
            "unevenly. And an offset that is genuinely constant almost always means "
            "something upstream is misconfigured — the imbalance is the symptom, and the "
            "cause is usually still doing other damage. It also costs nothing to fix, which "
            "makes leaving it in indefensible."
        ),
        common_causes=(
            "A mono element panned very slightly off centre and never spotted — a lead vocal "
            "a few percent right will do it on its own.",
            "A pan law mismatch after moving a project between DAWs, or a mono source "
            "sitting on a stereo channel where the pan control behaves as a balance control.",
            "An analogue chain or interface where one leg has more gain, printed into the "
            "bounce.",
            "A plugin instantiated on one side of a pair, or with unlinked channel settings "
            "left asymmetric.",
            "A stereo sample or loop that is itself lopsided, sitting loud in the "
            "arrangement.",
            "An arrangement that genuinely is one-sided: something busy hard right with "
            "nothing answering it on the left. That is a mix decision, and it is worth "
            "making on purpose rather than by accident.",
        ),
        how_to_fix=(
            FixStep(
                action="Look at the meter, do not listen for it.",
                detail=(
                    "Your ears have already adapted and will tell you the mix is centred. "
                    "Play the whole track and compare the two channels' average levels on "
                    "the master meter — averages, not peaks, and over the full duration, not "
                    "one section."
                ),
            ),
            FixStep(
                action="Find which bus carries the offset before touching the master.",
                detail=(
                    "Mute buses one at a time and watch which one takes the lean with it. "
                    "Correcting on the master fixes the number and leaves the cause in "
                    "place, where it keeps misleading every decision you make."
                ),
            ),
            FixStep(
                action="Check the pan positions of anything that should be dead centre.",
                detail=(
                    "Lead vocal, kick, snare, bass. In most DAWs you can reset a pan control "
                    "to exact centre with a modifier-click, which removes the possibility of "
                    "a value that merely looks centred."
                ),
            ),
            FixStep(
                action="Verify it is in the file and not in your room.",
                detail=(
                    "Swap the left and right cables at your speakers and listen again. If "
                    "the lean follows the music, it is in the mix. If it stays in the same "
                    "physical side of the room, it is your room or your listening position, "
                    "and correcting the file would make things worse."
                ),
            ),
            FixStep(
                action="Only after all that, apply a balance trim on the master.",
                detail=(
                    "A master trim shifts everything, including the parts that were already "
                    "correct, so it is a last resort rather than a first move. If you use "
                    "one, re-check that the phantom centre lands between the speakers "
                    "afterwards."
                ),
            ),
        ),
        how_to_verify=(
            "The two channels' average levels over the whole track should sit within about a "
            "dB of each other. By ear, standing equidistant from both speakers, the vocal "
            "and kick should appear exactly between them. Note that mono is useless as a "
            "check here — summing turns an offset into a level, so the problem simply "
            "disappears."
        ),
        learn_more=(
            "This is about the phantom centre, which is an illusion rather than a place. A "
            "sound coming equally out of two speakers is not localised at either one; your "
            "hearing constructs a single source hanging between them. The illusion depends "
            "on the two arrivals being matched in level and time. Roughly a dB of level "
            "difference is enough to shift the phantom image noticeably off centre, and by "
            "about 15-18 dB it has collapsed entirely into the louder speaker.\n\n"
            "Your room produces exactly the same shift without anything being wrong with the "
            "file — a wall closer on one side, an asymmetric desk reflection, a listening "
            "position off the centre line. Since the two are indistinguishable by ear, an "
            "offset you can measure in the file matters far more than one you think you can "
            "hear in the room, and the correct order of operations is always: measure the "
            "file, verify by swapping the speakers, and only then reach for a trim."
        ),
        minutes=10,
    ),
)


# ---------------------------------------------------------------------------
# Low end
# ---------------------------------------------------------------------------

_add(
    "low_end.kick_bass_collision",
    Explainer(
        headline="Your kick and your bass are fighting over the same space, so instead of a hit and a note you get one thick blur.",
        what_it_is=(
            "A kick drum and a bass or 808 both put most of their energy into the same "
            "octave or two, roughly 30 to 120 Hz. When two sounds sit that close together in "
            "frequency and land at the same instant, they do not simply add up. Depending on "
            "where each waveform happens to be in its cycle, they reinforce at some moments "
            "and cancel at others — and since neither is a steady tone, that sum comes out "
            "different on every single hit. The low end stops being a level you can set and "
            "becomes a level that lurches.\n\n"
            "There is a second, related failure, and it is about the fundamental. Any "
            "pitched sound is a stack of frequencies: the lowest one is the fundamental and "
            "it is what sets the note you hear, and above it sit the harmonics that give the "
            "sound its character. A kick has a fundamental too, whether or not anyone tuned "
            "it. If the kick's fundamental lands within a few semitones of the bass note, "
            "the ear cannot separate them into two pitches at all — the kick just re-triggers "
            "the note the bass is already holding, and you get thickness where you wanted an "
            "impact.\n\n"
            "Be aware of how this was measured. With stems, the kick and the bass are two "
            "separated sources and their pitches and levels are read directly. Without "
            "stems, the kick's spectrum is reconstructed by subtracting a between-hits "
            "spectrum from an at-hits one — a good estimate, not an isolated kick — and the "
            "amount the bass ducks under each kick is inferred by watching the sub level, "
            "not read off a plugin. Treat the shape of the finding as solid and the exact "
            "frequencies as approximate when there are no stems."
        ),
        what_you_hear=(
            "The bottom is loud but you cannot tell what it is doing. The kick has no "
            "definite moment of impact; the 808's pitch is hard to name. The level of the "
            "low end jumps between hits for no reason you can point to, and the same pattern "
            "sounds fine in one bar and swampy in the next. Through a limiter it pumps: the "
            "combined peaks slam the limiter and take the vocal and drums down with them. On "
            "a small speaker both disappear together, because everything they had was in the "
            "one octave the speaker cannot produce."
        ),
        why_it_matters=(
            "Two sources stacked in the same octave produce peaks you get no loudness from, "
            "so they eat headroom and drive gain reduction that pulls the entire mix. In "
            "hip-hop, trap, house, techno and drum and bass, the relationship between the "
            "kick and the bass is not a detail of the mix — it is the record, and a blurred "
            "one reads as amateur instantly to anyone in the genre. Translation suffers "
            "twice over: on a phone the fundamentals are gone, and because the two sources "
            "were leaning on each other for weight rather than each having a defined job, "
            "there is nothing in the harmonics left to carry the part."
        ),
        common_causes=(
            "The 808 and the kick are on the same note, either because the 808 follows the "
            "key and the kick sample happens to be pitched there, or because nobody tuned "
            "the kick at all.",
            "Both sources are full-range: a kick with a long sub tail under an 808 that also "
            "has a click, so each is trying to be the whole low end.",
            "No sidechain at all, or a sidechain whose release is so long the bass never "
            "comes back before the next note.",
            "A long 808 tail under a fast kick pattern, so the previous note is still "
            "sounding when the next kick lands.",
            "Kick and bass hitting the same beat throughout the track, with no rhythmic "
            "space designed between them.",
            "A sub layer added under both of them, which triples the number of sources in "
            "the same octave.",
        ),
        how_to_fix=(
            FixStep(
                action="Fix it in the arrangement first — decide whether they need to land together at all.",
                detail=(
                    "Move the 808 note off the kick by a sixteenth, shorten the note before "
                    "the kick, or leave a gap. This costs no processing, works on every "
                    "system, and is what makes the pattern groove rather than merely stop "
                    "colliding. If you programmed the pattern, start here."
                ),
            ),
            FixStep(
                action="Split the job by frequency: decide who owns the sub and who owns the punch.",
                detail=(
                    "Give one source exclusive territory below about 60-80 Hz and the other "
                    "the 80-150 Hz thump plus its click higher up. In most modern genres the "
                    "808 owns the sub and the kick owns the punch; in house and techno it is "
                    "often the reverse. Either works. What does not work is both being "
                    "allowed everywhere. High-pass the one that does not own the sub and "
                    "commit to it."
                ),
                needs="eq_static",
            ),
            FixStep(
                action=(
                    "Make the bass dip out of the way for the length of each kick — the move "
                    "usually called sidechaining."
                ),
                detail=(
                    "Mechanically: a compressor sitting on the bass, listening to the kick "
                    "rather than to the bass, so every kick hit turns the bass down for a "
                    "moment and lets it back up. Nothing is removed and nothing is filtered; "
                    "the two just stop happening at full level at the same instant.\n\n"
                    "Fast attack so the kick's transient is uncovered immediately, and — the "
                    "parameter that actually matters — a release timed so the bass is fully "
                    "back before the next note. Too long a release is what makes a mix pump "
                    "and breathe instead of just getting out of the way. Aim for a couple of "
                    "dB more duck than you think you need, then back it off until you stop "
                    "hearing the movement."
                ),
                needs="comp_sidechain_ext",
                without=(
                    "Draw the dip by hand instead of having a compressor do it: on the bass "
                    "track, pull the volume automation (or the clip envelope) down a few dB "
                    "right where each kick lands and back up immediately after, so the bass is "
                    "at full level again before the next note. Same result as sidechaining, "
                    "tedious exactly once — draw one and copy it to every hit. Watch the "
                    "recovery rather than the depth: a dip that has not come back up by the "
                    "next note is what makes a track pump."
                ),
            ),
            FixStep(
                action="Duck one frequency band instead of the whole bass.",
                detail=(
                    "A dynamic EQ band on the bass, keyed from the kick and centred on the "
                    "kick's fundamental, removes the collision while the bass keeps its body "
                    "and its harmonics. This is the fix when full-range ducking makes the "
                    "track feel like it is breathing."
                ),
                needs="eq_dynamic",
                without=(
                    "Without a dynamic band, take a narrow static cut in the bass at the "
                    "kick's fundamental and accept that it is always there. On a sustained "
                    "808 this is barely audible; on a melodic bass, prefer the arrangement "
                    "fix instead."
                ),
            ),
            FixStep(
                action="Tune the kick to a note that is in the key but not the note the bass plays.",
                detail=(
                    "An octave or a fifth away keeps them related and lets the ear hear two "
                    "separate pitches. The same note, or anything within a few semitones of "
                    "it, is the case the ear cannot resolve — which is the problem in the "
                    "first place."
                ),
            ),
            FixStep(
                action="Recover the kick's impact with attack, not with more low end.",
                detail=(
                    "Once the sub is out of the kick, it can feel weak. The fix is transient "
                    "shaping and a little presence around 2-5 kHz, both of which survive on "
                    "a phone. Putting the sub back does not."
                ),
                needs="transient_shaper",
                without=(
                    "Without a transient shaper, get the attack back with a narrow boost "
                    "around 2-5 kHz on the kick and a slightly slower compressor attack so "
                    "the initial hit passes through uncompressed."
                ),
            ),
            FixStep(
                action="Check the result in mono and on the smallest speaker you own.",
                detail=(
                    "A properly separated kick and bass still read as two distinct things on "
                    "a laptop, because each has its own harmonics up where the speaker "
                    "works. If they still merge there, the split has not happened."
                ),
            ),
        ),
        how_to_verify=(
            "Watch a spectrum analyser across 30-120 Hz over a bar: the level should pulse "
            "with the pattern rather than lurching unpredictably between hits. The limiter's "
            "gain reduction should stop spiking on the moments where kick and bass "
            "coincide. And the plain test — play four bars and say out loud what the kick is "
            "doing and what the bass is doing. If you can only describe one sound, keep "
            "going."
        ),
        learn_more=(
            "The reason two low sources are so much worse than two high ones is cycle "
            "length. Two tones close together in frequency produce beating: their sum swings "
            "between reinforcement and cancellation at a rate equal to the gap between them. "
            "At 50 Hz a single cycle lasts 20 milliseconds, so a kick lasting a tenth of a "
            "second is only a handful of cycles long. Whether it lands in step or opposed to "
            "the bass note beneath it is decided by timing on the order of a few "
            "milliseconds — which is the difference between one hit and the next in any real "
            "performance or any swung grid.\n\n"
            "Higher up the spectrum, cycles are so short that hundreds pass during a single "
            "note and the ear averages the interaction into a steady tone. Down here you "
            "hear each interaction individually, as a different level every time. That is "
            "why turning one of them down never fixes this: the problem is not the balance "
            "between them, it is that both are present in the same octave at the same "
            "instant. The three real fixes — move them apart in time, move them apart in "
            "frequency, or make one duck the other — all attack that, and nothing else does."
        ),
        minutes=45,
    ),
)

_add(
    "low_end.not_mono",
    Explainer(
        headline="Your bass is spread across the two channels, and a club system will subtract most of it.",
        what_it_is=(
            "How much of the energy below about 120 Hz is common to both channels versus how "
            "much is the difference between them. Bass that is mono lives entirely in the "
            "part every playback system reproduces. Bass that has been widened has some of "
            "its energy in the difference signal, and that portion gets subtracted rather "
            "than added the moment anything sums the channels."
        ),
        what_you_hear=(
            "In your room, often 'bigger' — which is precisely why people do it. What you "
            "are actually getting is a low end that wanders: it changes level when you move "
            "your head, it does not lock with the kick, and its weight is inconsistent from "
            "note to note. In mono it thins out or partly disappears, and the kick suddenly "
            "sounds like it is playing over the bass rather than with it."
        ),
        why_it_matters=(
            "A club rig feeds its subs a summed mono signal, and it does so for a reason: at "
            "those wavelengths, running subs in stereo creates cancellation zones across the "
            "dance floor, so nobody does it. Whatever you put into the sides of your bass "
            "never reaches the subwoofers — it is not merely quieter, it is arithmetically "
            "removed. Vinyl is stricter still. A record groove encodes the shared content as "
            "side-to-side stylus movement and the difference content as vertical movement, "
            "and large low-frequency vertical movement makes the cutter lift out of the "
            "groove or cut through the wall. Cutting engineers mono the bass themselves or "
            "send the master back.\n\n"
            "There is a headroom argument too. The bottom octave is the most expensive part "
            "of the mix in energy terms, and any of it spent on content that will be "
            "cancelled downstream is wasted twice: once in your limiter, once in the room."
        ),
        common_causes=(
            "A widener or imager on the master or the bass bus applied across the whole "
            "spectrum instead of above a crossover.",
            "A stereo bass patch: unison detune, a chorus, or two oscillators panned apart.",
            "A stereo reverb or delay on the 808 or the kick.",
            "A sampled 808 or kick one-shot that is already a wide stereo file — very common "
            "in commercial sample packs.",
            "Bass recorded as DI plus amp and panned apart, so the two arrival times "
            "decorrelate the low end.",
            "A mid/side EQ with a broad boost on the side channel that reaches down into the "
            "bass.",
        ),
        how_to_fix=(
            FixStep(
                action="Put a mono maker on the master or the bass bus with the crossover around 100-150 Hz.",
                detail=(
                    "Set it just above where the bass fundamental actually lives. This is the "
                    "direct fix and it takes one knob."
                ),
                needs="mono_maker",
                without=(
                    "No mono maker: use a mid/side EQ and high-pass the side channel around "
                    "120 Hz — same result, one more step. With neither, fix it at the "
                    "source: remove the stereo processing from the bass and kick and pan them "
                    "dead centre."
                ),
            ),
            FixStep(
                action="Fix it on the instrument, not just on the master.",
                detail=(
                    "Mono-ing the output works, but the widener upstream is still shaping "
                    "what reaches your limiter and still spending headroom on content that is "
                    "about to be summed away. Turn it off at the source and the fix is ahead "
                    "of everything else in the chain."
                ),
            ),
            FixStep(
                action="Choose the crossover by what is actually down there, not by habit.",
                detail=(
                    "Too high and you collapse guitars, keys and low synths into the middle "
                    "and the mix goes small. Too low and you leave the widened bass "
                    "fundamental sitting in the sides, which is the thing you were trying to "
                    "remove. Sweep it while watching where the bass energy stops."
                ),
                needs="meter_spectrum",
            ),
            FixStep(
                action="If you want a wide bass sound, put the width in the harmonics.",
                detail=(
                    "Split the bass into two layers: keep everything below the crossover mono "
                    "and untouched, and saturate and widen a duplicate above it. You hear the "
                    "width — because direction is something you perceive from harmonics, not "
                    "from fundamentals — and you lose nothing on fold-down."
                ),
                needs="saturation",
            ),
            FixStep(
                action="Confirm with a correlation meter on the low band specifically.",
                detail=(
                    "The broadband figure will not show this: the bass is a small share of "
                    "the mix's overall correlation. Band-limit the meter, or put it after a "
                    "low-pass on a parallel path."
                ),
                needs="meter_correlation",
                without=(
                    "No correlation meter: switch the master to mono and watch the bass level "
                    "on any meter. If it drops when you sum, it was in the sides."
                ),
            ),
        ),
        how_to_verify=(
            "Almost all of the energy below 120 Hz should be mono — bass-led genres want "
            "essentially all of it. Practically: switch to mono and the bass should not "
            "change level at all, only the space around it. Moving your head between the "
            "speakers should also stop changing the bass level."
        ),
        learn_more=(
            "Stereo bass is not a feature you are giving up; it is a feature that was never "
            "there. You locate a sound from the difference in level and the difference in "
            "arrival time between your two ears. A 50 Hz wave is about seven metres long. "
            "Your ears are about twenty centimetres apart, so at that wavelength the wave "
            "passes both ears at essentially the same pressure and essentially the same point "
            "in its cycle. There is nearly nothing for the auditory system to work with, and "
            "below roughly 100-150 Hz human directional hearing is effectively absent. What "
            "you perceive as the position of a bass is the position of its harmonics.\n\n"
            "So widening the fundamental adds no spatial impression you can actually hear, "
            "while adding a difference signal that anything downstream is free to subtract. "
            "It costs headroom, costs level on every summed system, and risks a vinyl "
            "rejection, in exchange for an effect that only exists as an artefact on "
            "near-field speakers in your own room."
        ),
        minutes=15,
    ),
)

_add(
    "low_end.sub_energy_hot",
    Explainer(
        headline="There is more weight under 60 Hz than the mix can carry, and most listeners will never hear the part costing you the most.",
        what_it_is=(
            "The share of the mix's total energy sitting in the bottom octave, roughly 20 to "
            "60 Hz. It is expressed as a proportion of the whole, so it does not change when "
            "you turn the track up — it describes balance, not level. Every genre has its "
            "own window for this, and the windows are very far apart: trap and hip-hop are "
            "allowed several times more bottom-octave energy than folk or jazz, because "
            "there the sub is the instrument. 'Hot' here means hot for this genre."
        ),
        what_you_hear=(
            "On a system with real extension, a bloated one-note bottom that swamps the "
            "kick's attack and makes every bass note sound like the same note. On everything "
            "else, nothing — which is the trap. The clue on small speakers is inverted: a "
            "track with too much sub sounds *quiet* on a phone, because the limiter has been "
            "turned down by energy the phone cannot reproduce, so all the audible parts "
            "arrive lower."
        ),
        why_it_matters=(
            "Excess sub is the single biggest thing standing between your mix and "
            "competitive loudness. A limiter measures energy, and energy down there is "
            "enormous relative to how loud it seems, so every bass note pulls the whole mix "
            "down — your snare gets quieter because your 808 is loud. It also masks upward: "
            "too much bottom octave dulls the low mids and the mix loses definition without "
            "anything being wrong up there. And on a big system, too much sub is not more "
            "bass; it is the point where the venue's own limiter engages and the entire "
            "record ducks."
        ),
        common_causes=(
            "An 808 or sub synth with no high-pass and a long tail, layered under a kick that "
            "also carries sub.",
            "A sub layer added 'for weight' and never checked on a system that can actually "
            "play it.",
            "Mixing on headphones or monitors that roll off below 60 Hz, so the excess is "
            "invisible while you work.",
            "A low shelf boost on the master or the mix bus, added to make the mix feel "
            "bigger on small speakers.",
            "A room mode or a null at your listening position making the sub sound thin to "
            "you while it is hot in the file — extremely common, and the reason this is a "
            "measurement rather than a judgement call.",
            "Pitch-shifted 808s whose fundamental has dropped below where the patch was "
            "designed to sit.",
        ),
        how_to_fix=(
            FixStep(
                action="Look at it instead of listening to it.",
                detail=(
                    "Put a spectrum analyser on the master and watch the bottom octave "
                    "against the rest. Your room is the least reliable thing in the chain "
                    "down here; the analyser is not affected by where you are sitting."
                ),
                needs="meter_spectrum",
            ),
            FixStep(
                action="High-pass under the lowest note the bass actually plays.",
                detail=(
                    "Not at a round number — find the note. Sweep a steep high-pass upward "
                    "until you hear the *character of the note* change rather than just the "
                    "weight, then back off from there. Everything below that point was "
                    "costing you and doing nothing."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Shelve rather than cut when you need less, not none.",
                detail=(
                    "A wide low shelf a couple of dB down on the bass bus keeps the note's "
                    "shape intact. A steep filter in the same place removes the bottom of the "
                    "note and leaves you with a bass that sounds smaller but no clearer."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Even out the loud notes rather than lowering all of them.",
                detail=(
                    "Sub level is rarely uniform — a few notes in the pattern are far hotter "
                    "than the rest, usually the lowest ones, and they are what set the "
                    "measurement. A dynamic band across 30-60 Hz brings those down and leaves "
                    "the rest alone, so you keep the weight and lose the bloat."
                ),
                needs="eq_dynamic",
                without=(
                    "Without a dynamic band, automate the bass volume per note. There are "
                    "fewer offending notes than you expect — often three or four in the "
                    "pattern — and fixing those is faster than reshaping the whole part."
                ),
            ),
            FixStep(
                action="A/B against a genre reference at matched loudness.",
                detail=(
                    "Sub excess is nearly impossible to judge alone and completely obvious in "
                    "a level-matched comparison. Match by ear on the mids, then listen to the "
                    "bottom."
                ),
                needs="reference_matching",
                without=(
                    "Load a commercial track from the same genre onto a muted channel, pull "
                    "its level until the vocals sit at the same apparent loudness as yours, "
                    "and switch between them. The bottom-octave difference will be "
                    "unmistakable."
                ),
            ),
            FixStep(
                action="Check on the smallest speaker you own after cutting.",
                detail=(
                    "If the mix loses nothing on a laptop when the sub comes down, you did not "
                    "need what you removed. If it does lose something, you cut into the "
                    "harmonics, not the sub — go back up."
                ),
            ),
        ),
        how_to_verify=(
            "The bottom-octave share should come back inside the genre's window. Two things "
            "should then be true: the limiter's gain reduction should stop tracking the bass "
            "line note for note, and the mix should get audibly louder at the same limiter "
            "setting than it was before. If it did not get louder, you cut the wrong thing."
        ),
        learn_more=(
            "This all follows from how differently the ear and a meter treat low frequencies. "
            "Human hearing is dramatically less sensitive down there — at normal listening "
            "levels a 30 Hz tone needs something like 30-40 dB more energy than a 1 kHz tone "
            "to seem equally loud. A peak meter and a limiter apply no such weighting; they "
            "count energy at 30 Hz exactly as they count energy at 3 kHz. So bottom-octave "
            "content is the worst trade in the mix: maximum measured level for minimum "
            "perceived loudness, and the measured level is what governs how loud your master "
            "is allowed to be.\n\n"
            "The same curves explain a mistake worth avoiding. Hearing flattens out as "
            "playback gets louder, so bass sounds proportionally stronger the louder you "
            "monitor. Judge your sub balance while mixing loud and you will consistently "
            "under-deliver it; judge it quiet and you will consistently overdo it. This is "
            "why sub is one of the few things worth setting against a measurement and a "
            "reference rather than against your own ears in the moment."
        ),
        minutes=25,
    ),
)

_add(
    "low_end.sub_energy_thin",
    Explainer(
        headline="The bottom octave is not pulling its weight, so the mix will feel small on anything that can actually play it.",
        what_it_is=(
            "The same measurement as its opposite: the share of total energy in roughly the "
            "20-60 Hz octave, on the low side of what this genre's records carry. Worth "
            "separating two things that get confused. A mix can be clean down there and "
            "correct — folk, jazz and acoustic windows sit far lower than trap's, and nobody "
            "wants an 808 under an acoustic guitar. This fires when the bottom octave is "
            "under what *this* genre expects, which means something the style relies on is "
            "missing rather than tastefully absent."
        ),
        what_you_hear=(
            "On laptop speakers, phones and earbuds — nothing wrong. Often it sounds cleaner, "
            "which is how this survives. On monitors with extension, in a car, on a decent "
            "pair of headphones or on any club system, the record sounds like a demo: the "
            "kick clicks but never lands, the bass is audible but weightless, and the whole "
            "mix sits on top of the speaker instead of moving any air. There is no floor "
            "under the arrangement."
        ),
        why_it_matters=(
            "You cannot add this in mastering, because there is nothing to boost. A low shelf "
            "on a bass with no fundamental raises the room noise and the rumble and delivers "
            "no note. It also changes how the master behaves under a limiter: a mix with no "
            "bottom hits the limiter with mids and highs instead, which are the frequencies "
            "the ear is most sensitive to, so it turns harsh and fatiguing long before it "
            "gets loud. Records that feel effortlessly loud usually have a well-controlled "
            "bottom octave, not an absent one."
        ),
        common_causes=(
            "High-passing by reflex — a filter on every channel including the kick and the "
            "bass, which are the two that belong down there.",
            "Mixing on headphones or small monitors with no extension below 60 Hz, so you "
            "keep reducing what you cannot hear in order to make things sound 'clean'.",
            "A bass patch that is all harmonics: saturated or distorted so hard the "
            "fundamental has been replaced by upper content.",
            "A room null at the listening position exactly where the bass lives — you hear a "
            "hole, boost around it, then cut the fundamental to stop the boom you created.",
            "A kick sample chosen for its click, with nothing under 80 Hz, and no sub layer "
            "added beneath it.",
            "A mono maker or multiband with its crossover set high and its low band turned "
            "down.",
        ),
        how_to_fix=(
            FixStep(
                action="Find out whether the fundamental exists before boosting anything.",
                detail=(
                    "Solo the bass and the kick with a spectrum analyser and look at where "
                    "their energy actually stops. If there is nothing below 60 Hz, no EQ move "
                    "will produce any — you need to add a source, not lift one."
                ),
                needs="meter_spectrum",
            ),
            FixStep(
                action="Check every high-pass filter in the session.",
                detail=(
                    "This is the most common single cause and the fastest fix. A high-pass "
                    "belongs on the things that should not be down there — vocals, guitars, "
                    "keys, overheads — not on the two things that should. Take them off the "
                    "kick and bass and re-measure before doing anything else."
                ),
            ),
            FixStep(
                action="Add a fundamental rather than boosting one that is not there.",
                detail=(
                    "A sine or triangle layer at the bass's root, following the notes, or a "
                    "sub-synth or octaver tracking the part. Keep it mono, keep it short, and "
                    "envelope it so it only sounds when the bass does — an untamed sub layer "
                    "turns this finding straight into its opposite."
                ),
            ),
            FixStep(
                action="Saturate the bass so the note survives where the fundamental cannot go.",
                detail=(
                    "Saturation generates harmonics above the fundamental, and those "
                    "harmonics are what let a phone speaker convey a 40 Hz note. Do this and "
                    "you can afford to keep a real fundamental at a sensible level instead of "
                    "over-boosting it to be heard on small speakers."
                ),
                needs="saturation",
            ),
            FixStep(
                action="Only now, if it is still short, use a wide low shelf.",
                detail=(
                    "Gentle and broad, on the bass bus rather than the master. Once there is "
                    "content to lift, a small shelf does a lot; before there is content, it "
                    "does nothing but raise noise."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Sanity-check the room, because it is probably why this happened.",
                detail=(
                    "Bounce it and listen in a car and on headphones. If the bass is fine "
                    "everywhere except where you mix, the room has a null and you have been "
                    "compensating for it in every mix you have ever made."
                ),
            ),
        ),
        how_to_verify=(
            "The bottom-octave share should land inside the genre's window. Then check both "
            "ends of the range: the bass note should be clearly identifiable on a laptop "
            "speaker, which proves the harmonics are there, and it should be felt on a system "
            "with extension, which proves the fundamental is. A car is the fastest single "
            "check for the second one."
        ),
        learn_more=(
            "The reason a bass line is recognisable on a phone at all is an effect called the "
            "missing fundamental. Play the harmonics of a 50 Hz note — 100, 150, 200 Hz — "
            "without the 50 Hz itself and the ear still reports the pitch as 50 Hz, because "
            "pitch is derived from the spacing of the harmonic series rather than from the "
            "presence of its lowest member. Your phone speaker produces nothing below a few "
            "hundred hertz and you still hear the bass line.\n\n"
            "That splits the low end into two jobs that are easy to confuse. The harmonics "
            "carry the *note* — what it is, which system it plays on, whether the part is "
            "intelligible. The fundamental carries the *weight* — the physical sensation, "
            "which only exists on systems that reach that low. A thin mix usually has the "
            "harmonics and no weight. A muddy one often has the weight and no harmonics. "
            "Diagnosing which of the two is missing, before reaching for a shelf, is most of "
            "the skill in low-end work."
        ),
        minutes=30,
    ),
)

_add(
    "low_end.subsonic_rumble",
    Explainer(
        headline="There is energy below the bottom of hearing eating your headroom, and nobody will ever hear what it is costing you.",
        what_it_is=(
            "Content below about 25 Hz. Human hearing is usually quoted as bottoming out at "
            "20 Hz, and in practice anything under 25 is felt at best and on most systems not "
            "even that. This is measured as a share of the mix's total energy, against a "
            "ceiling derived from the genre's own target curve — which matters, because trap, "
            "hip-hop and EDM legitimately carry far more energy near the bottom than folk "
            "does, and the ceiling moves accordingly. So this does not fire because you have "
            "a big sub. It fires when there is content below where even a big sub lives."
        ),
        what_you_hear=(
            "Nothing. Not 'subtle' — nothing. No home speaker reproduces it, headphones do "
            "not, phones do not, and a large PA converts it into cabinet excursion rather "
            "than sound. What you might notice is indirect: your limiter working harder than "
            "the mix looks like it should, and woofers visibly moving during passages that "
            "are supposed to be quiet."
        ),
        why_it_matters=(
            "This is pure cost with no return. It occupies peak level and drives gain "
            "reduction, so every dB of it is a dB of loudness you do not get, and each time "
            "it triggers the limiter it pulls the audible part of the mix down with it. On a "
            "big system it consumes amplifier power and driver excursion, which makes the "
            "sub you *do* want quieter and more distorted — and that distortion lands up in "
            "the range where the music is, so inaudible content produces audible damage. On "
            "vinyl it moves the cutting stylus a long way for no sound at all. A steep "
            "high-pass here is one of the very few completely free wins in mastering: "
            "measurable headroom back, nothing audible removed."
        ),
        common_causes=(
            "Microphone handling, footsteps, HVAC and traffic on acoustic recordings, printed "
            "into the track and never filtered out.",
            "A synth patch with a sub oscillator an octave lower than intended, or an 808 "
            "pitched down until its fundamental fell below hearing.",
            "DC offset or very low-frequency drift from an analogue chain, a converter or a "
            "poorly behaved plugin.",
            "The envelope thump of a heavily compressed source, where the gain change itself "
            "becomes a sub-audio waveform.",
            "Time-stretching and pitch-shifting artefacts, particularly on drum loops.",
            "A filter self-oscillating at the very bottom of its range, or an amplitude LFO "
            "slow enough to register as energy rather than as movement.",
        ),
        how_to_fix=(
            FixStep(
                action="High-pass the master at 25 Hz with a steep slope.",
                detail=(
                    "24 dB/octave or steeper. Slope matters more than frequency here: a gentle "
                    "12 dB/octave filter at 25 Hz is still passing plenty at 15 Hz, which is "
                    "the content you are trying to remove. Nothing audible goes with it."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Find the source anyway, rather than leaving the master filter to do the work.",
                detail=(
                    "Solo buses with your analyser's range set to show the very bottom. It is "
                    "almost always one track. Fixing it there turns the master filter into a "
                    "safety net instead of a repair, and stops the rumble from driving "
                    "compressors upstream that you will never see it in."
                ),
                needs="meter_spectrum",
            ),
            FixStep(
                action="Check for DC offset while you are down there.",
                detail=(
                    "A waveform sitting above or below the zero line has a constant offset "
                    "that costs headroom on one side of every peak and reads as energy at 0 "
                    "Hz. Most DAWs have DC-offset removal in their offline processing, and a "
                    "high-pass takes care of it too."
                ),
            ),
            FixStep(
                action="Use a linear-phase filter if you are worried about the bass transient.",
                detail=(
                    "This is one of the few places linear phase clearly earns its cost. A "
                    "steep filter that low rotates phase up through the region the kick and "
                    "bass occupy, which can soften the attack slightly; a linear-phase "
                    "high-pass does not."
                ),
                needs="eq_linear_phase",
                without=(
                    "Without a linear-phase EQ, use a gentler slope placed lower — around 18 "
                    "dB/octave at 20 Hz. You leave a little behind and keep the bass transient "
                    "shape untouched, which is the right trade on a drum-led mix."
                ),
            ),
            FixStep(
                action="Re-check the limiter after the filter, not before.",
                detail=(
                    "The point of the exercise is what it gives back. The peak level should "
                    "drop, and the limiter's gain-reduction meter — the display showing how "
                    "much it is having to turn the mix down — should move less for the same "
                    "output loudness."
                ),
                needs="limiter",
            ),
        ),
        how_to_verify=(
            "The sub-25 Hz share should fall under the genre's ceiling. The listening test is "
            "a negative one: switch the filter in and out and the mix should sound identical. "
            "If you can hear the filter, it is set too high or too steep — that is not "
            "evidence the rumble was musical. And you should be able to reach the same "
            "loudness while your limiter is working measurably less hard than before."
        ),
        learn_more=(
            "'Below hearing' does not mean 'not there'. Hearing rolls off steeply below about "
            "40 Hz and is essentially gone by 20, but a level meter has no such roll-off — it "
            "counts energy at 10 Hz exactly as it counts energy at 1 kHz, and so does your "
            "limiter's detector. That mismatch between what the meter counts and what the ear "
            "counts is the entire reason subsonic content is such a bad deal: maximum "
            "measured level, zero perceived loudness.\n\n"
            "The loudspeaker side is worse. To produce a given sound pressure, a driver's cone "
            "excursion has to rise steeply as frequency falls — roughly four times the "
            "movement for each octave down. Sub-audio content therefore drives a woofer "
            "towards its mechanical limits while producing almost no sound at all, and a "
            "driver near its limits distorts. Those distortion products are not at 15 Hz; they "
            "are harmonics, spread up into the range you can hear. So inaudible energy makes "
            "the audible part of your record measurably worse on exactly the large systems "
            "where it was supposed to pay off."
        ),
        minutes=10,
    ),
)

_add(
    "low_end.section_collapse",
    Explainer(
        # Not "at the point people are listening hardest" any more. This same
        # explainer now sits under "Intro runs emptier than most intros", where
        # that clause asserts the opposite of what the finding says — and the
        # headline is what `report._start_here` prints as the one-line summary,
        # so the contradiction lands on the first page.
        headline="One section carries far less bottom end than the rest of the record, and which section it is decides whether that is an arrangement or a fault.",
        what_it_is=(
            "The analysis splits the track into sections and measures each one's low-end "
            "energy as a share of that section's own level, then compares the sections that "
            "are carrying the record — the ones nearly as loud as the loudest. Because it is "
            "a share rather than an absolute, a quiet intro cannot trigger it: quiet parts "
            "being quiet is not a fault. What it catches is a section that is as loud as its "
            "neighbours and has far less bottom under it. Something that was holding up the "
            "low end stopped doing so there.\n\n"
            "Which section it is decides how loudly this is said. An intro or an outro with "
            "the bottom pulled back is how arrangements work, so one has to swing about twice "
            "as far as the rest of the record — and last long enough for anybody to sit in it "
            "— before it is mentioned at all, and then only as a note. A chorus or a drop is "
            "the section the record is built to arrive at, so it is held to the genre's own "
            "figure. Everything in between is reported and asked about, because the "
            "measurement genuinely cannot tell a written arrangement from a mistake."
        ),
        what_you_hear=(
            "The track suddenly feels smaller and higher, as if the speakers changed size "
            "mid-song. It usually lands on a drop or a chorus and reads as anticlimax: the "
            "section is as loud as the one before it and has less impact, which is the "
            "opposite of what you built it for. Sometimes you notice it the other way round — "
            "the vocal or the lead becomes suddenly exposed and a bit harsh, because the "
            "weight that was balancing it has gone."
        ),
        why_it_matters=(
            "A drop works by contrast, and most of the contrast people actually respond to is "
            "in the low end — it is the part you feel rather than hear, so it registers as "
            "physical arrival. Losing it for eight bars in the biggest section of the record "
            "is the difference between a release and a demo. It is also close to invisible on "
            "small speakers, since the section you broke sounds the same as everything else on "
            "a laptop, so it survives every check made on the wrong system. And because the "
            "ear is insensitive down there, the loudness meter barely moves — the section "
            "still measures loud while feeling empty."
        ),
        common_causes=(
            "The bass or 808 muted, or its automation left down, in that section — usually "
            "left over from an arrangement where a different element carried it.",
            "A sidechain keyed from something that stops playing there, so either the duck "
            "vanished or the duck became permanent.",
            "A filter automation sweep drawn for a build with no return point, leaving the "
            "following section filtered indefinitely.",
            "A bus compressor or multiband reacting to the section's different density and "
            "pulling the low band down exactly where it is busiest.",
            "The section built from a different set of loops or stems that were never "
            "level-matched against the rest.",
            "A drop where every element was stripped for impact, including the one that was "
            "quietly holding the floor.",
        ),
        how_to_fix=(
            FixStep(
                action="Go to the named section and look at the bass and kick tracks first.",
                detail=(
                    "Is anything muted, automated down, or filtered there? This is an editing "
                    "accident far more often than it is a mixing problem, and confirming it "
                    "takes under a minute."
                ),
            ),
            FixStep(
                action="Check every automation lane that crosses the section boundary.",
                detail=(
                    "Filter sweeps and volume rides drawn for a build are the usual culprits, "
                    "because a sweep with no return point stays where you left it. Look for "
                    "lanes whose last node is before the boundary."
                ),
            ),
            FixStep(
                action="Loop across the boundary and listen for the bottom, not the level.",
                detail=(
                    "The level does not change — that is the whole point of the finding, and "
                    "why it is easy to miss. Listen specifically for whether the floor under "
                    "the arrangement is still there when the new section starts."
                ),
            ),
            FixStep(
                action="If a sidechain is involved, check what it is keyed from.",
                detail=(
                    "A duck keyed from a kick that drops out in this section either stops "
                    "working or, if the key is a different element there, starts ducking "
                    "against the wrong rhythm. Key it from a dedicated silent trigger track so "
                    "it does not depend on the arrangement."
                ),
                needs="comp_sidechain_ext",
                without=(
                    "If you ducked the bass with volume automation instead, check that the "
                    "automation shape still matches the kick pattern in this section — a "
                    "copied envelope over a changed pattern ducks against nothing."
                ),
            ),
            FixStep(
                action="Watch what your bus dynamics do across the boundary.",
                detail=(
                    "A multiband's low band can be pushed much harder by a denser section and "
                    "quietly remove the bass precisely where you wanted the most. Compare its "
                    "gain reduction in the two sections; if the low band is working "
                    "significantly harder, that is your cause."
                ),
                needs="comp_multiband",
                without=(
                    "With a single bus compressor, watch its gain reduction across the "
                    "boundary. If it jumps, it is reacting to arrangement density rather than "
                    "to level — slow the attack, lower the ratio, or high-pass its detector so "
                    "the bass stops driving it."
                ),
            ),
            FixStep(
                action="If the section is deliberately stripped, put a floor back under it.",
                detail=(
                    "Sparse should mean fewer elements, not less bandwidth. A sub layer under "
                    "the lead, a low-passed pad, or simply bringing up the kick's own bottom "
                    "keeps the section feeling full-size while staying empty in the "
                    "arrangement."
                ),
            ),
        ),
        how_to_verify=(
            "Play the whole track watching the low band on an analyser and the loud sections "
            "should carry a similar share of bottom as each other. By ear, loop the boundary: "
            "crossing into the section should feel like the arrangement changed, not like the "
            "playback system changed."
        ),
        learn_more=(
            "There is a reason losing the low end reads as losing *size* rather than losing "
            "bass. Perceived bigness tracks bandwidth — the distance between the lowest and "
            "highest frequencies present. Take away the bottom octave and the mids and highs "
            "do not change level at all, but the spectrum gets narrower and the whole thing "
            "sounds like it is coming out of a smaller box. Nothing got quieter; the source "
            "appeared to shrink.\n\n"
            "It interacts badly with how loudness is measured. Loudness meters weight low "
            "frequencies down, in imitation of the ear, so removing the bottom octave barely "
            "moves the number — which is exactly why the collapsed section still measures as "
            "one of the loudest parts of your track. The gap between what the meter reports "
            "and what a listener feels is widest precisely here, in the low end, in the "
            "biggest section of the record. That gap is what this finding exists to close."
        ),
        minutes=20,
    ),
)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(target: Dict[str, Explainer]) -> None:
    """Copy these explainers into the shared registry.

    Called by `knowledge.py` after its own module body has run, which is what
    keeps this a one-way dependency: we import the dataclasses from there, and
    it never imports anything from here at module scope.
    """
    target.update(EXPLAINERS)
