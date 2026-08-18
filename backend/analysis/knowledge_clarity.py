"""Teaching content for clarity, transients and vocal balance.

These six findings share one idea and it is the most important one in the
product: **masking**. A loud sound does not just sit next to a quieter one, it
raises the level at which the quieter one becomes audible at all. That is why a
mix can hit every frequency target and still sound like a blanket, why turning
the vocal up makes the mix louder without making the words clearer, and why the
fix almost always belongs on the part that is *winning*.

`clarity.congested` carries the full explanation because it is where the
concept is load-bearing. The others reference it rather than repeat it.

Registered into `knowledge.EXPLAINERS` by `knowledge.py` calling `register()`.
Nothing here touches that dict at import time — this module is imported *by*
knowledge.py, so reaching back into it would be a cycle.
"""

from __future__ import annotations

from typing import Dict

from .knowledge import Explainer, FixStep, Resource, youtube_search

__all__ = ["EXPLAINERS", "register"]


EXPLAINERS: Dict[str, Explainer] = {}


def _add(finding_id: str, explainer: Explainer) -> None:
    EXPLAINERS[finding_id] = explainer


# ---------------------------------------------------------------------------
# Clarity
# ---------------------------------------------------------------------------

_add(
    "clarity.congested",
    Explainer(
        headline="Every part is in there, but you cannot follow any single one of them.",
        what_it_is=(
            "This is masking, and it is worth twenty minutes of your life to understand "
            "properly because it explains most of what people mean by 'my mix sounds "
            "amateur'.\n\n"
            "Your ear does not measure frequencies one at a time. The inner ear is a strip "
            "that responds to different pitches along its length, and a loud sound does not "
            "just occupy its own spot — it excites the neighboring region too. Anything "
            "quieter landing in that region stops being heard as a separate sound. It is not "
            "made quieter. It becomes invisible, while still consuming level, headroom and "
            "space in your mix.\n\n"
            "Two facts do most of the work. First, masking is **asymmetric**: a loud sound "
            "hides things ABOVE it in pitch far more than things below it. A heavy low end "
            "eats the midrange; a bright hi-hat barely touches the bass. Second, the spread "
            "gets **wider as the masker gets louder** — the same bass part that sat politely "
            "under the vocals at mix level swallows them once the master limiter has pushed "
            "everything up.\n\n"
            "This finding is the share of your mix's audible energy sitting under its own "
            "masking threshold — the level a sound would have to reach, given everything "
            "around it, before you could hear it as a separate thing. It is measured from the "
            "stereo mix, so it can see that energy is buried without being able to say which "
            "instrument buried it."
        ),
        what_you_hear=(
            "The mix sounds full and busy and slightly tiring, and no individual part sounds "
            "wrong when you solo it. You keep turning things up to hear them and the mix just "
            "gets louder. Detail vanishes first: the shaker, the second guitar layer, the "
            "reverb tail, the top of the piano. Late in a session it feels like the mix has "
            "gone 'flat' or 'small' and adding more is the instinct — which makes it worse. "
            "A congested mix also loses depth: everything sits at the same apparent distance, "
            "because the cues the ear uses for depth are exactly the quiet details being "
            "masked."
        ),
        why_it_matters=(
            "Masked energy is pure cost. It uses headroom, it makes the limiter work harder, "
            "and it contributes nothing a listener can hear — so a congested mix is both less "
            "clear AND quieter than a clean one at the same peak level. It gets worse "
            "downstream: lossy encoders decide what to throw away using a masking model very "
            "like this one, so the material sitting under threshold is the first thing an MP3 "
            "or AAC encoder discards, and it discards it crudely. Small speakers, phones and "
            "car systems compress the dynamic range further, which raises the quiet material "
            "into the masker's shadow and makes the clutter worse than it is on your monitors."
        ),
        common_causes=(
            "Too many parts occupying the same octave. Three synth layers, a guitar and a "
            "vocal all living in 200 Hz-2 kHz is an arrangement problem no EQ fully solves.",
            "Boosting instead of cutting. Each part got a lift where it sounded good in solo, "
            "so now everything is loud in the same three places and none of it stands out.",
            "Low-mid buildup around 200-500 Hz, which masks upward into the entire vocal and "
            "guitar range — the single most common cause, and it is invisible on a frequency "
            "curve because no one band looks unreasonable.",
            "Reverb and delay on many sources at once. Tails fill the gaps between notes, and "
            "the gaps between notes are where the ear separates parts.",
            "Heavy bus or master compression. Compressing everything together raises the quiet "
            "material up into the loud material's masking shadow — the whole mix gets denser "
            "and less separated at the same time.",
            "Saturation on many channels, each adding harmonics upward into the midrange.",
        ),
        how_to_fix=(
            FixStep(
                action="Decide who owns each region before touching an EQ.",
                detail=(
                    "For each frequency area — low end, low mids, midrange, presence, top — "
                    "pick one part that owns it and one that supports it. Everything else "
                    "gets cut there. This is the whole job; the plugin choices below are just "
                    "how you execute it. If two parts genuinely need the same octave at the "
                    "same time, that is an arrangement decision, not a mixing one."
                ),
            ),
            FixStep(
                action="Cut the masker, never boost the maskee.",
                detail=(
                    "This is the rule that makes masking actionable. The threshold that is "
                    "hiding a part is set by the LOUD part, so lowering the loud one drops the "
                    "threshold and the quiet one becomes audible at the level it already had. "
                    "Boosting the quiet one instead raises the whole mix, spends headroom, and "
                    "turns it into a masker of something else. Subtractive moves make a mix "
                    "clearer and quieter; that is the correct direction."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Look an octave BELOW whatever you cannot hear.",
                detail=(
                    "Because masking spreads upward, the thing burying your vocal is usually "
                    "not in the vocal's band. A wide 2-3 dB cut around 200-400 Hz on the "
                    "guitar, synth or drum bus opens up the 1-3 kHz range without anyone "
                    "touching 1-3 kHz. Wide and shallow beats narrow and deep here — you are "
                    "reducing a region's total loudness, not removing a resonance."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Make the cut happen only when the two parts collide.",
                detail=(
                    "Put a dynamic EQ on the masker, sidechained from the part being buried, "
                    "with a band across the contested region. It only ducks while both are "
                    "sounding, so the masker keeps its full weight everywhere else. This is "
                    "the difference between a mix that is clear and one that is thin."
                ),
                needs="eq_dynamic",
                without=(
                    "With static EQ only, make the cut modest — 2-3 dB — and automate the band "
                    "off, or automate the whole channel down, in the sections where the part "
                    "it was fighting is not playing."
                ),
            ),
            FixStep(
                action="Let a resonance suppressor take the narrow stuff.",
                detail=(
                    "Congestion is usually a mix of one broad region being too loud and a "
                    "handful of narrow resonances stacking inside the same critical band. "
                    "Do the broad region by hand as above, and let the suppressor track the "
                    "narrow ones — set depth conservatively and leave frequency alone."
                ),
                needs="resonance_suppressor",
                without=(
                    "Sweep a narrow boosted bell slowly through 150 Hz-1 kHz on the busiest "
                    "bus. Anything that jumps out and rings is a resonance: cut it a few dB "
                    "at the width you found it, then move on. Do not do more than three."
                ),
            ),
            FixStep(
                action="Shorten and filter the reverb, and give it pre-delay.",
                detail=(
                    "Tails are broadband and constant, which makes them near-perfect maskers. "
                    "High-pass the reverb return at 300-400 Hz, low-pass it around 6-8 kHz, "
                    "cut the decay, and add 20-40 ms of pre-delay so the dry transient lands "
                    "in silence before the space arrives. You will usually find you can then "
                    "use more reverb, not less."
                ),
                needs="reverb",
            ),
            FixStep(
                action="Separate in space and time, not just in frequency.",
                detail=(
                    "Two parts in the same band stop masking each other if they are not in the "
                    "same place or not playing at the same moment. Pan them apart, or use "
                    "mid/side EQ to keep the center for the lead and the sides for the layers. "
                    "Failing that, edit: shorten one part so the other has a gap to speak in."
                ),
                needs="eq_mid_side",
                without=(
                    "Pan the competing parts apart, and mute the least important of them "
                    "through the busiest section. Deleting a part is a legitimate mix move and "
                    "usually the fastest one."
                ),
            ),
            FixStep(
                action="Back off the mix-bus compression before you judge any of this.",
                detail=(
                    "Bus compression pulls quiet material up into the loud material's shadow, "
                    "so it manufactures congestion while sounding like glue. Halve the gain "
                    "reduction, re-listen, and only add it back once the balance is right."
                ),
                needs="comp_single",
            ),
        ),
        how_to_verify=(
            "Level-match your fixed version against the old one and A/B them. This matters, "
            "because the fixed one will be quieter and louder always wins a blind comparison: "
            "put both bounces in a session, drop a gain plugin on the louder one, and trim it "
            "until the two feel equally loud. The fix is working when the new version "
            "sounds quieter AND more detailed at the same time. Then play at conversation "
            "volume and try to follow one background part for a full bar without turning "
            "anything up; if you can, you are done. Re-analyze and the masking index should "
            "sit inside your genre's ceiling with worst-band congestion coming down with it."
        ),
        learn_more=(
            "The measurement behind this is a real psychoacoustic model, not an analogy. The "
            "spectrum is projected onto around 40 auditory filters spaced one ERB apart — an "
            "ERB is the bandwidth the ear genuinely resolves at that pitch, which is roughly a "
            "sixth of an octave in the midrange and far wider proportionally down low. That is "
            "why 40 Hz and 55 Hz are effectively the same event to your ear while 2 kHz and "
            "2.2 kHz are not, and why low-end arrangement problems are so much less forgiving "
            "than midrange ones.\n\n"
            "Each filter then spreads energy onto its neighbors: steeply downward — roughly "
            "27 dB for every filter-width of distance — and shallowly upward, with the upward "
            "slope flattening further as the masker gets louder. Anything falling under the "
            "resulting threshold is counted as masked. There is a time dimension too: a loud "
            "hit masks what follows it for 100-200 ms and, weakly, what comes 5-20 ms before "
            "it, which is why a dense mix loses the notes that fall just behind the snare.\n\n"
            "The practical consequence worth remembering: masking depends on LEVEL, so it is "
            "not a fixed property of your arrangement. A mix that separates fine at mix level "
            "can collapse once ten more decibels of loudness have been pushed into it, because "
            "every masker's shadow got wider on the way. If "
            "your mix falls apart when you push it into the limiter, this is usually why, and "
            "the answer is upstream in the balance rather than in the limiter."
        ),
        minutes=45,
        resources=(
            Resource(
                kind="reference",
                label="Auditory masking",
                url="https://en.wikipedia.org/wiki/Auditory_masking",
                source="Wikipedia",
                note=(
                    "The mechanism itself, from somewhere that is not us: the threshold "
                    "shift, the upward spread, and why a masker's shadow gets wider as it "
                    "gets louder. Read it once and the fix steps above stop being rules to "
                    "remember."
                ),
            ),
            youtube_search(
                "how to fix a cluttered mix subtractive EQ low mids",
                "Subtractive EQ on a whole mix, in real time",
                "Engineers pulling 2-3 dB out of 200-500 Hz and letting the midrange come "
                "back — the 'cut the masker' step demonstrated on real material, which is "
                "the part that is hard to believe until you hear it.",
            ),
            youtube_search(
                "too many parts in the same frequency range arrangement fix crowded mix",
                "When the answer is the arrangement, not the EQ",
                "For the case this explainer admits no EQ solves: two parts that genuinely "
                "need the same octave at the same time, and what to re-voice, shorten or "
                "mute instead of filtering.",
            ),
        ),
    ),
)

_add(
    "clarity.stem_masking",
    Explainer(
        headline="Two specific parts are fighting over the same frequency range, and one of "
                "them is losing.",
        what_it_is=(
            "Masking: a loud part raises the level at which a quieter part becomes audible, so "
            "the quiet one stops being heard as a separate sound even though it is still there "
            "and still using up headroom.\n\n"
            "What makes this finding different from the general clarity one is that it is not "
            "an inference. The track was separated into its sources — drums, bass, vocals and "
            "everything else — and those sources were compared against each other directly, "
            "band by band, only over the frames where both were actually playing. So instead "
            "of 'something in the low mids is buried', it can say which part is doing it to "
            "which. That is a direct measurement of the two sources, and it is why this reads "
            "as a named pair rather than a region.\n\n"
            "Note what this means about your spectrum analyzer: it will show nothing. This "
            "finding typically fires precisely when the spectrum of the finished mix looks "
            "healthy. Your ear resolves pitch in bands rather than continuously, and two parts "
            "stacked inside one of those bands add up to a single, perfectly reasonable-looking "
            "hump from outside. You cannot see this without pulling the sources apart, which is "
            "the reason the deep pass exists."
        ),
        what_you_hear=(
            "The buried part sounds like it is 'not doing anything' in the mix even though it "
            "was fine when you made it. Push its fader and the mix gets louder and thicker but "
            "that part still does not emerge — that is the signature, and it is what "
            "distinguishes masking from a simple level problem. Mute the masker for a second "
            "and the buried part leaps out at the level it always had. Do that test; it is the "
            "fastest way to feel what masking actually is."
        ),
        why_it_matters=(
            "You are paying full price in headroom, loudness and encoder bitrate for a part "
            "nobody can hear. Worse, the usual instinct — turn the buried part up — makes the "
            "mix louder and denser without making it clearer, and that is how mixes end up "
            "over-limited: you keep adding to solve a problem that adding cannot solve. Fixing "
            "it the other way round typically gives you back a dB or more of headroom and the "
            "mix gets clearer at the same time."
        ),
        common_causes=(
            "Bass and kick sharing a fundamental, so the two lowest-frequency parts occupy one "
            "auditory filter and neither has a defined pitch or attack.",
            "A wide pad, synth or guitar layer with full-range low mids under a lead, adding "
            "energy at 150-500 Hz that masks upward across the whole vocal range.",
            "Sample layers stacked for weight, each contributing the same 100-250 Hz body.",
            "Saturation or amp simulation on a supporting part, filling the spectrum between "
            "its own harmonics and turning a narrow source into a broadband one.",
            "Something that was mixed to sound big in solo. Big in solo almost always means "
            "broadband, and broadband parts are the best maskers there are.",
        ),
        how_to_fix=(
            FixStep(
                action="Mute the masker and listen once. Confirm it before you fix it.",
                detail=(
                    "Separation is a model and it can leak — a hi-hat can end up partly in "
                    "'other'. Ten seconds of muting tells you whether the named part is really "
                    "the one covering the other. Once you have heard it, you cannot unhear it."
                ),
            ),
            FixStep(
                action="Cut the masker in the contested band. Do not boost the buried part.",
                detail=(
                    "The threshold hiding the buried part is set by the loud one, so lowering "
                    "the loud one is what uncovers it — at the level it already had, without "
                    "adding anything to the mix. Wide bell, 2-4 dB, centered in the band the "
                    "finding names. If the masker is a supporting part, be braver: 6 dB out of "
                    "a pad's low mids is normal and nobody notices except as clarity."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Make that cut dynamic, keyed off the part being buried.",
                detail=(
                    "The measurement only counts frames where both parts sound, so the "
                    "collision is intermittent by definition and a permanent cut is a "
                    "permanent compromise. A dynamic band on the masker, sidechained from the "
                    "maskee, with a few dB of range across the contested area gives the masker "
                    "its full body back whenever the other part is not playing."
                ),
                needs="eq_dynamic",
                without=(
                    "With static EQ, keep the cut to 2-3 dB and automate it out of the sections "
                    "where the buried part is not present, so the masker is not thinned for "
                    "the whole record to solve a problem that happens in the choruses."
                ),
            ),
            FixStep(
                action="For a bass-and-kick pair, duck instead of carving.",
                detail=(
                    "Down low there is no room to separate two parts by frequency — the "
                    "auditory filters are too wide, and you would carve away the note. Trigger "
                    "a compressor on the bass from the kick, fast attack, release timed to let "
                    "the bass return before the next hit. Ten to twenty milliseconds of the "
                    "bass stepping aside is inaudible as ducking and completely audible as "
                    "punch."
                ),
                needs="comp_sidechain_ext",
                without=(
                    "No external sidechain: pick which one owns the fundamental instead. "
                    "High-pass the bass above the kick's fundamental, or tune the kick to sit "
                    "a fifth below the bass note, and let one part own the sub."
                ),
            ),
            FixStep(
                action="Move one of them out of the center.",
                detail=(
                    "Masking is strongest between sources in the same place. If the contested "
                    "band is above roughly 200 Hz, widening or panning the masker while the "
                    "buried part holds the center buys separation without either losing level. "
                    "Below 200 Hz this does not work and will cost you mono compatibility."
                ),
                needs="eq_mid_side",
                without=(
                    "Pan the masker off center if it is a supporting part, or use a short "
                    "delay of 10-25 ms on one side of it to widen it away from the center."
                ),
            ),
            FixStep(
                action="If the parts are both essential and both broadband, fix the arrangement.",
                detail=(
                    "No EQ resolves two parts that genuinely need the same octave for the same "
                    "bars. Change the voicing so one plays an octave up, shorten one so it "
                    "answers the other instead of doubling it, or cut it from that section. "
                    "This is the honest fix and often the fastest."
                ),
            ),
        ),
        how_to_verify=(
            "Mute-test again: when the fix is working, muting the masker should no longer "
            "reveal a hidden part — because it is not hidden any more. At conversation volume "
            "you should be able to follow the previously buried part through the busiest "
            "section. Re-run the deep analysis and the pair should either be gone or its "
            "overlap materially reduced, and your peak level should have come down slightly "
            "without the mix sounding smaller."
        ),
        learn_more=(
            "This measurement is deliberately conservative and it is worth knowing how. It "
            "only reports a pair when the masker is well above the maskee, IN a band the "
            "maskee genuinely occupies, for a real share of the frames where both are playing "
            "— because a part 15 dB up in a band the other one barely uses is not masking "
            "anything, and 15 dB up for four percent of the track is not either.\n\n"
            "It also only reports masking WITHIN a band. Real masking spreads upward in "
            "frequency well beyond the masker's own band, and that spread is the bigger effect "
            "in practice, but attributing it correctly between two separated sources needs a "
            "full excitation model this comparison does not run. Two consequences for you: "
            "what is named here is real, and it is a floor rather than the whole story. If you "
            "fix the named pair and the section still sounds crowded, look one band DOWN from "
            "wherever the crowding is — that is where the unattributed masking lives."
        ),
        minutes=30,
        resources=(
            Resource(
                kind="reference",
                label="Q. Can static EQ tackle masking effectively?",
                url="https://www.soundonsound.com/sound-advice/q-can-static-eq-tackle-masking-effectively",
                source="Sound On Sound",
                note=(
                    "Hugh Robjohns on the exact trade-off the fix steps above are making: a "
                    "static cut thins the masker for the whole record to solve a collision "
                    "that only happens in some bars, which is the argument for keying the "
                    "cut off the buried part instead."
                ),
            ),
            youtube_search(
                "dynamic EQ sidechain to unmask two instruments",
                "Setting up the sidechained band, in your DAW",
                "The routing is the fiddly part and it differs per plugin — these walk "
                "through getting the buried part into the trigger input, so you can aim one "
                "band at the specific pair named above.",
            ),
            youtube_search(
                "sidechain kick and bass so they stop clashing",
                "The low-end pair, which needs ducking rather than carving",
                "If the pair here is kick and bass, EQ is the wrong tool — there is no room "
                "to separate them by frequency. These cover the attack and release timing "
                "that makes one step aside for the other without it reading as pumping.",
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Transients
# ---------------------------------------------------------------------------

_add(
    "transients.no_punch",
    Explainer(
        headline="Your drums are loud, but they do not hit.",
        what_it_is=(
            "Punch is not loudness. It is CONTRAST — how far the sound drops in the moment "
            "after it starts.\n\n"
            "Every hit has two parts. The transient is the first few milliseconds: the beater "
            "on the skin, the stick on the rim, the click. The sustain is everything after it: "
            "the body, the ring, the tail. Your ear reads impact from the difference between "
            "the two, not from the peak. A kick that is loud and then GONE hits harder than a "
            "louder kick that hangs around for 80 ms, and it does so at a lower level. This is "
            "why 'turn the drums up' so rarely fixes a mix that lacks punch: it raises both "
            "halves and the contrast between them does not change.\n\n"
            "The measurement here is exactly that relationship — the level at the hit against "
            "the level 50 ms later, averaged across every hit found in the track, then checked "
            "band by band so kick weight and snare snap are judged separately. A mix can keep "
            "one and "
            "lose the other, and which one it lost tells you what happened."
        ),
        what_you_hear=(
            "The track feels like a wall rather than a groove. It is busy, it is loud, and it "
            "does not make you move. Individual hits sound soft-edged — the kick reads as a "
            "'thump held down' instead of a strike, the snare loses its crack and keeps its "
            "body. The tell is that it gets WORSE at low listening volume: punchy drums stay "
            "punchy when you turn a mix down, while a mix living on level alone turns into "
            "mush the moment it stops being loud. On phone speakers, which cannot reproduce "
            "the low fundamentals at all, a mix with no transients has almost no rhythm left."
        ),
        why_it_matters=(
            "Transients are how a listener locates the beat in time, so losing them costs you "
            "the groove — the thing most people actually respond to. They also cost you "
            "loudness, counter-intuitively: every platform normalizes playback to a target "
            "loudness, so what a listener hears is not your peak level but your contrast at a "
            "fixed loudness. Played back next to a flattened mix at the same measured loudness, "
            "a mix with real transients sounds bigger and hits harder, every time — the "
            "flattened one just turns down. And the damage compounds: with no "
            "peaks left, the limiter has nothing to grab, so it starts pulling down sustained "
            "material instead, which is what turns a flat mix into a pumping one.\n\n"
            "How much punch you need is genre-dependent and the bar moves a long way. Drum & "
            "bass, techno and trap live or die on it. Ambient, jazz and classical are supposed "
            "to breathe rather than hit, and the floor for them is far lower."
        ),
        common_causes=(
            "Compressor attack too fast. This is the number one cause by a distance: an attack "
            "under about 5 ms clamps down during the transient itself, so the compressor "
            "removes the exact 3 ms that made the hit a hit and leaves the sustain untouched.",
            "Too much of it, in too many places — a compressor on the kick, another on the drum "
            "bus, another on the mix bus, each taking a little of the same attack.",
            "The limiter doing 4-6 dB of gain reduction. It only reduces peaks, and the peaks "
            "are the transients.",
            "Release too fast, so the compressor recovers between hits and pulls the sustain "
            "back up — the gap that made the punch gets filled in by the compressor itself.",
            "Reverb, room mics or long sample tails filling the 50 ms after the hit. The hit is "
            "intact; there is simply nothing to fall to.",
            "Layered samples that are a few milliseconds out of alignment, which smears one "
            "sharp attack into a soft one.",
            "Low-end masking: a sustained 808 or bass note covering the kick's attack. The "
            "transient is there and inaudible, which reads identically to it being gone.",
        ),
        how_to_fix=(
            FixStep(
                action="Work out which half is wrong before reaching for anything.",
                detail=(
                    "Solo the drums and listen to the space between hits. If the hits sound "
                    "soft, the attack is being removed — go to the compressors. If the hits "
                    "sound fine but there is no gap after them, the sustain is too high — go "
                    "to reverb, tails and bass. These need opposite fixes and guessing wrong "
                    "costs you an hour."
                ),
            ),
            FixStep(
                action="Slow every compressor attack until you can hear the stick again.",
                detail=(
                    "Start with the drum bus. Attack up to 20-30 ms so the transient passes "
                    "through untouched and the compressor only grabs the body behind it. Then "
                    "set release to recover just before the next hit — too fast and it pumps "
                    "the sustain back up, too slow and it never lets go. Done right, a "
                    "compressor with a slow attack INCREASES punch, because it lowers the "
                    "sustain while leaving the transient alone."
                ),
                needs="comp_single",
            ),
            FixStep(
                action="Get density from a parallel path instead of the main one.",
                detail=(
                    "Crush a duplicate hard — fast attack, high ratio, 10 dB of reduction — and "
                    "blend it under the untouched signal. The dry path keeps the transient, the "
                    "squashed path adds weight and sustain, and you control the ratio with one "
                    "fader. This is how commercial drums get both."
                ),
                needs="comp_parallel",
                without=(
                    "Build it by hand: send the drum bus to an aux, put a stock compressor on "
                    "the aux with a fast attack and heavy reduction, and blend that aux in "
                    "underneath the untouched drums. Same result, one extra channel."
                ),
            ),
            FixStep(
                action="Add attack and cut sustain directly.",
                detail=(
                    "A transient shaper works on envelope shape rather than level, so it can "
                    "restore punch without changing the balance. Push attack a few dB and — "
                    "this matters more — pull sustain DOWN. Reducing sustain increases contrast "
                    "without adding any peak level, so it costs you no headroom at all."
                ),
                needs="transient_shaper",
                without=(
                    "Use an expander on the drum bus with a fast attack and a short hold: it "
                    "pushes down what sits below the threshold, which is the tail, and leaves "
                    "the hit alone. On individual samples, drawing a short volume envelope that "
                    "cuts the tail does the same job for free."
                ),
            ),
            FixStep(
                action="Take peaks off with a clipper so the limiter stops eating them.",
                detail=(
                    "A clipper shaving 1-2 dB off drum peaks reduces what the limiter has to "
                    "do, and clipping a transient by a decibel is far less audible than "
                    "limiting one. Do it on the drum bus, before the master chain."
                ),
                needs="clipper",
                without=(
                    "Reduce the input to your master limiter until gain reduction on drum hits "
                    "is 1-2 dB rather than 4-6. Take the loudness back from balance and from "
                    "cutting sustain, not from the limiter."
                ),
            ),
            FixStep(
                action="Clear the 50 ms after each hit.",
                detail=(
                    "Punch is a hole as much as a peak. Shorten drum reverb decay, add 20-40 ms "
                    "of pre-delay so the transient lands before the space does, high-pass the "
                    "reverb return above 300 Hz, and trim overlapping sample tails. Often the "
                    "drums did not lose their attack at all — something moved into the gap."
                ),
                needs="reverb",
            ),
            FixStep(
                action="Check whether the kick's attack is masked rather than missing.",
                detail=(
                    "Mute the bass and listen to the kick. If it suddenly punches, the "
                    "transient was always there and a sustained low note was covering it — that "
                    "is a masking fix (duck the bass under the kick), not a transient one. No "
                    "amount of transient shaping will beat a bass note sitting on top of the "
                    "attack."
                ),
            ),
        ),
        how_to_verify=(
            "Turn the monitors DOWN to conversation level. The kick and snare should still "
            "read as strikes and the groove should still pull; if the drums vanish into the "
            "track at low volume, the contrast is still not there. Solo the drum bus and listen "
            "for audible drop-away between hits. Then re-analyze: punch should clear your "
            "genre's floor, the weakest band should come up with it, and attack time should "
            "not have gone the wrong way. Trim the two bounces to the same apparent loudness "
            "before you judge by ear — the punchier version will usually measure quieter, and "
            "louder always wins an untrimmed comparison."
        ),
        learn_more=(
            "Attack time on a compressor is not a delay, it is how long the compressor takes to "
            "reach full gain reduction after crossing the threshold. So a 1 ms attack has the "
            "gain almost fully pulled down while the transient is still happening, and a 30 ms "
            "attack lets the whole transient through before clamping the body behind it. That "
            "is the entire mechanism, and it is why the slower setting sounds punchier despite "
            "compressing less of the sound.\n\n"
            "Underneath it is a quirk of hearing: the ear integrates loudness over roughly 30 "
            "to 50 ms, but it detects the ONSET of a sound in a couple of milliseconds. So a "
            "3 ms spike barely registers as loudness while registering fully as impact. That "
            "asymmetry is the free lunch in mixing — you can raise perceived impact without "
            "raising measured loudness, which is exactly what a transient shaper's attack "
            "control and a well-set clipper are both exploiting.\n\n"
            "It is also why per-band punch is measured separately. Over-compression flattens "
            "the high bands first, because that is where the shortest transients live; low-end "
            "masking flattens the low band first while the snare stays crisp. Whichever band "
            "is weakest is naming the cause."
        ),
        minutes=30,
        resources=(
            youtube_search(
                "compressor attack and release settings for punchy drums",
                "Hearing what attack time actually does to a hit",
                "The number one cause on the list above, demonstrated: the same drum bus at "
                "1 ms and at 30 ms, until you can recognize a clamped transient by ear "
                "rather than by meter.",
            ),
            youtube_search(
                "why does my mix lose punch when I make it louder",
                "Punch against loudness, which is the trade this finding is about",
                "Why the flattened version loses to the dynamic one once a platform "
                "normalizes them to the same loudness, and where to take the level back "
                "from instead of the limiter.",
            ),
            Resource(
                kind="reference",
                label="What is a transient in audio production?",
                url="https://www.izotope.com/en/learn/what-is-a-transient-audio-production.html",
                source="iZotope",
                note=(
                    "A tool-by-tool tour of what acts on the first few milliseconds of a hit "
                    "— compression, transient shaping, parallel paths, saturation — worth "
                    "reading before you pick which of the steps above is yours, since they "
                    "sound similar and do opposite things to the envelope."
                ),
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Vocal balance
# ---------------------------------------------------------------------------

_VOCAL_INFERENCE_NOTE = (
    "One thing to know about how this was measured. If you ran the standard analysis, there "
    "were no separated parts to work with, so the vocal level was estimated from the CENTER of "
    "the stereo field — the content that appears equally in both channels. That is a fair "
    "proxy for a lead vocal, because leads are almost always centered, but it cannot tell a "
    "singer from a centered synth, a mono bass or a snare, and it is meaningless on a mono "
    "file. Treat it as directional. Running the deep analysis separates the vocal as an actual "
    "source and compares it against the real instrumental, which turns this from an inference "
    "into a measurement — worth doing before you make a big move on the strength of it."
)

_add(
    "vocal_balance.buried",
    Explainer(
        headline="The words are hard to make out — and a listener who has to work for them "
                "will just skip the track.",
        what_it_is=(
            "This is the loudness of the lead against everything else, judged against where "
            "records in your genre actually place it. It is A-weighted, meaning it is measured "
            "the way the ear weights loudness rather than the way a meter weights energy — "
            "otherwise the sub would own the entire budget and every mix would look "
            "vocal-starved.\n\n"
            "Being buried is rarely just a fader being too low. Most of the time it is masking: "
            "other parts are raising the level at which the vocal becomes audible, so the vocal "
            "is present, at a reasonable level, and still not heard. The signature of that is "
            "unmistakable — you push the vocal fader, the mix gets louder, and the words are "
            "still not clearer. If that is happening, the fix is not on the vocal.\n\n"
            + _VOCAL_INFERENCE_NOTE
        ),
        what_you_hear=(
            "You can hear that someone is singing but you cannot catch the lyric unless you "
            "already know it — and because you wrote it, you do, which is why this is one of "
            "the hardest problems to hear in your own mix. The reliable symptoms: you keep "
            "reaching for the vocal fader and never settle; the vocal seems fine loud and "
            "disappears when you turn the mix down; consonants go first, so it sounds like the "
            "singer is mumbling when they are not. Ends of phrases drop out before beginnings "
            "do."
        ),
        why_it_matters=(
            "A listener deciding whether to keep listening does it in the first fifteen "
            "seconds, on a phone, at low volume, while doing something else. They will not "
            "lean in. Playback conditions make it worse than it is on your monitors: phone and "
            "laptop speakers reproduce almost nothing below 300 Hz, so the vocal's body is gone "
            "and only the part competing hardest with guitars and cymbals survives, and "
            "car and earbud listening adds noise across exactly the midrange where "
            "intelligibility lives.\n\n"
            "How forward the vocal should be is genre-dependent and the difference is large. "
            "Pop and country are vocal-forward — the lead sits clearly on top and everything "
            "else is arranged around it. In rock the vocal shares 2-4 kHz with the guitars and "
            "the whole mix is that negotiation. In EDM, techno and a lot of hip-hop the vocal "
            "is a texture inside the track rather than above it. Do not carry one genre's "
            "instinct into another."
        ),
        common_causes=(
            "Masking from the low mids. A 200-500 Hz buildup on guitars, keys or a pad spreads "
            "upward across the whole vocal range — you push the vocal up, the mix gets boomy, "
            "and the words are no clearer.",
            "Competition at 1-4 kHz specifically, where consonants live. Distorted guitars, "
            "bright synths, hi-hats and cymbals all camp there, and consonants are what carry "
            "intelligibility — vowels are loud and can survive being covered.",
            "Too much dynamic range on the vocal, so loud syllables sit right and quiet ones "
            "fall under the instrumental. The average reads fine and half the words do not.",
            "Reverb and delay pushing the vocal back. Wet sounds read as distant, and the tail "
            "of one word masks the start of the next.",
            "The master limiter. In a dense mix the vocal is often the loudest centered thing, "
            "so it takes the most gain reduction — the limiter is effectively riding it down.",
            "Mixing the vocal early against a half-finished instrumental and never re-checking "
            "it once everything else arrived.",
            "Monitoring loud. The ear's frequency response flattens at high SPL, so a vocal "
            "balanced at 95 dB is always too quiet at 65 dB, never the other way round.",
        ),
        how_to_fix=(
            FixStep(
                action="Move the fader first, and check it at low volume.",
                detail=(
                    "Turn the monitors down until the mix is barely above conversation level "
                    "and set the vocal there. Low-level listening is a far better judge of "
                    "vocal balance than loud listening, because it strips out the low end that "
                    "is flattering you. If a small fader move fixes it, stop — you do not have "
                    "a masking problem."
                ),
            ),
            FixStep(
                action="If the fader does not fix it, cut the competition instead.",
                detail=(
                    "A wide 2-3 dB cut around 200-500 Hz on the guitar, synth or keys bus opens "
                    "the vocal without touching the vocal, because masking spreads upward. Then "
                    "a smaller cut around 2-4 kHz on whatever is brightest. Cutting the masker "
                    "makes the vocal audible at the level it already had; boosting the vocal "
                    "just makes everything louder."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Make the room for the vocal appear only when it sings.",
                detail=(
                    "Put a dynamic EQ on the instrumental bus, sidechained from the vocal, with "
                    "a wide band across 300 Hz-4 kHz and a few dB of range. The backing gets "
                    "its full weight in the instrumental sections and steps aside under the "
                    "lines. This is the single highest-value move on this list and it is "
                    "inaudible as an effect."
                ),
                needs="eq_dynamic",
                without=(
                    "With static EQ, automate instead: a 1-2 dB dip on the instrumental bus "
                    "through the verses and choruses, back up in the gaps. Slower to set up, "
                    "and it does the same job."
                ),
            ),
            FixStep(
                action="Raise the quiet syllables rather than the loud ones.",
                detail=(
                    "If the vocal only disappears in places, the problem is its own dynamic "
                    "range. An opto compressor with a slow, level-dependent release evens a "
                    "vocal out without sounding compressed — aim for 3-6 dB on the loud lines "
                    "and let it do nothing on the quiet ones. First though, go through with "
                    "clip gain — the volume control on the audio itself, ahead of the fader — "
                    "and lift the quiet phrases by hand. A compressor cannot rescue a phrase "
                    "that never crosses its threshold in the first place."
                ),
                needs="comp_opto",
                without=(
                    "Use two stock compressors in series doing 3 dB each rather than one doing "
                    "6 — gentler ratios, and the second one catches what the first let past. "
                    "Before either of them, lift the quiet phrases by hand with clip gain (the "
                    "volume control on the audio itself, ahead of the fader), so the "
                    "compressors are not being asked to fix a problem that belongs in the edit."
                ),
            ),
            FixStep(
                action="Dry the vocal out before you turn it up.",
                detail=(
                    "Cut the reverb send, add 30-60 ms of pre-delay so the word lands before "
                    "the space, and high-pass the return above 400 Hz. A drier vocal reads as "
                    "closer and more intelligible at the same fader level, and you will find "
                    "you can add the space back once the level is right."
                ),
                needs="reverb",
            ),
            FixStep(
                action="Check what the master limiter is doing to it.",
                detail=(
                    "Bypass the master chain and re-listen to the balance. If the vocal is "
                    "clearly better without it, the limiter is riding the vocal down: reduce "
                    "the drive, or fix the density that is making the limiter work that hard, "
                    "rather than pushing the vocal up into it."
                ),
                needs="limiter",
            ),
        ),
        how_to_verify=(
            "Play it to conversation level on the worst speaker you own — a phone is ideal — "
            "and check you can follow every line. The stricter test: play it to someone who has "
            "not heard the song and ask them to tell you the first verse back. Re-analyze and "
            "the vocal-to-instrument ratio should sit inside your genre's window with the "
            "intelligibility index coming up alongside it. If you fixed it by cutting other "
            "parts rather than raising the vocal, your peak level should also have dropped "
            "slightly, which is the sign you did it the right way round."
        ),
        learn_more=(
            "Intelligibility is not carried by the loud part of a voice. Vowels are loud, "
            "sustained and low — they tell you a voice is there. Consonants are quiet, brief "
            "and concentrated in roughly 1-4 kHz, and they are what carry the actual words. "
            "'Bat', 'cat' and 'sat' differ only in a few milliseconds of quiet high-frequency "
            "noise. That is why the intelligibility figure here weighs the consonant band "
            "against what is competing with it rather than looking at overall vocal level: a "
            "vocal can be plenty loud and still unintelligible if a hi-hat or a distorted "
            "guitar owns 2-4 kHz.\n\n"
            "It also explains a common mixing mistake. Boosting 200-400 Hz makes a vocal sound "
            "bigger and richer in solo, and does nothing whatsoever for whether the words land "
            "— while adding exactly the energy that masks upward into its own consonants. If "
            "you want the words clearer, the move is almost always to remove something in the "
            "consonant band from everything else, not to add anything to the vocal."
        ),
        minutes=25,
        resources=(
            youtube_search(
                "how to make vocals cut through the mix without turning them up",
                "Getting the words out without touching the vocal fader",
                "Every technique in these works on the parts around the lead, which is where "
                "the fix lives once you have confirmed that pushing the fader makes the mix "
                "louder and the lyric no clearer.",
            ),
            youtube_search(
                "carve space for vocals sidechain dynamic EQ instrumental bus",
                "Ducking the backing under the lead",
                "How to build the highest-value move on the fix list: a wide band on the "
                "instrumental bus keyed from the vocal, so the space only opens while the "
                "vocal is actually singing and the track keeps its weight in between.",
            ),
            Resource(
                kind="reference",
                label="Intelligibility",
                url="https://en.wikipedia.org/wiki/Intelligibility_(communication)",
                source="Wikipedia",
                note=(
                    "The speech-science behind why this is not a volume problem: "
                    "intelligibility rides on brief, quiet consonants and on the "
                    "signal-to-noise ratio around them, which is why a louder vocal is not "
                    "automatically a clearer one."
                ),
            ),
        ),
    ),
)

_add(
    "vocal_balance.too_loud",
    Explainer(
        headline="The vocal is sitting on top of the track instead of inside it.",
        what_it_is=(
            "The same measurement as a buried vocal, missed from the other side: the lead is "
            "further above the instrumental than records in your genre sit.\n\n"
            "Too loud is a real fault, not just a taste call, for two reasons. Masking works in "
            "both directions — a vocal that is too far up covers the arrangement underneath it, "
            "so the track sounds thinner than it is. And on a master, the loudest centered "
            "element decides how hard the limiter works; a vocal riding 3 dB above where it "
            "belongs is a vocal that costs the whole record its loudness.\n\n"
            + _VOCAL_INFERENCE_NOTE
        ),
        what_you_hear=(
            "The vocal sounds stuck on top rather than performed in the room — like it is in "
            "front of the speakers while the band is behind them. The track feels small and "
            "thin under it, and the instruments seem to lose their arrangement even though "
            "nothing about them changed. Detail you did not want becomes conspicuous: breaths, "
            "lip noise, the sibilance you already de-essed, the tail of the tuning correction. "
            "If it sounds like a demo with a good vocal on it rather than a record, this is "
            "usually why."
        ),
        why_it_matters=(
            "It costs loudness directly. Streaming normalizes to a target, so a master whose "
            "limiter is being driven by one over-loud element ends up quieter and flatter than "
            "one where the vocal sits in place — you pay for those extra decibels of vocal with "
            "decibels of everything else. It costs weight too: the arrangement you spent days "
            "on is being masked by the one part sitting above it.\n\n"
            "The right amount is genre-dependent and the range is wide. Pop and country put the "
            "vocal clearly forward. Rock, indie and most electronic music sit it inside the "
            "band, and a vocal that would be correct on a pop record sounds amateur on a rock "
            "one. Check a reference in your own genre before deciding this finding is wrong."
        ),
        common_causes=(
            "The mix was built around the vocal rather than the vocal being placed into the "
            "mix. Every level was set to keep the vocal clear, so the vocal is by definition on "
            "top of all of them.",
            "Heavy vocal compression. Compression raises average level while leaving the peak "
            "where it was, so the vocal gets louder than the fader position or the peak meter "
            "suggests — the ear hears the average.",
            "Presence and air boosts. A brighter vocal reads as louder at identical level, so "
            "the problem is often not the fader at all.",
            "Mixing on headphones or at low volume, both of which make a vocal feel like it "
            "needs more than it does.",
            "A dry vocal with no space around it. Dryness reads as closeness, which reads as "
            "loud.",
            "Referencing against an unmatched track — a quieter reference makes your vocal seem "
            "correctly placed when it is not.",
        ),
        how_to_fix=(
            FixStep(
                action="Level-match against a reference in the same genre before changing anything.",
                detail=(
                    "Match loudness, not peaks, and switch between them at the same chorus. "
                    "Vocal placement is one of the strongest genre conventions there is, and it "
                    "is almost impossible to judge from memory."
                ),
                needs="reference_matching",
                without=(
                    "Import a commercial track from the same genre onto a track in your "
                    "session, put a gain plugin on it, and trim by ear until the two feel "
                    "equally loud on the choruses. Then switch between them every few seconds. "
                    "The vocal difference will be obvious within three passes."
                ),
            ),
            FixStep(
                action="Fix brightness before you fix level.",
                detail=(
                    "If the vocal is loud but not big, you have a presence problem rather than "
                    "a level one. Pull 2-5 kHz back by a couple of dB and re-listen: the vocal "
                    "often drops into place without the fader moving, and it keeps the weight "
                    "that simply lowering it would have cost you."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Reduce the vocal compression rather than the vocal fader.",
                detail=(
                    "Heavy compression is what made the vocal feel loud without the peaks "
                    "moving. Back the gain reduction off by half and re-balance. You may lose "
                    "some consistency doing this — an automation pass is the right way to get "
                    "it back, not more compression."
                ),
                needs="comp_single",
            ),
            FixStep(
                action="Push it back with space instead of level.",
                detail=(
                    "Add a short plate or room at a low level and a slap delay behind the "
                    "vocal. Distance cues let a vocal stay audible while stopping it sitting in "
                    "front of the speakers, and they usually solve this more convincingly than "
                    "any fader move because the real complaint is depth rather than volume."
                ),
                needs="reverb",
            ),
            FixStep(
                action="Bring the arrangement up to the vocal instead of pulling the vocal down.",
                detail=(
                    "If the vocal is where you want it and the track feels small under it, the "
                    "instrumental is the thing that is wrong. Raise the drums and bass and "
                    "re-check — the ratio is what is being measured, and either side of it can "
                    "move."
                ),
            ),
            FixStep(
                action="Re-check the master limiter afterwards.",
                detail=(
                    "A vocal that was riding the limiter gives back real headroom once it sits "
                    "down. Expect gain reduction to drop and the master to get louder at the "
                    "same setting — that is the loudness you were paying for."
                ),
                needs="limiter",
            ),
        ),
        how_to_verify=(
            "At conversation volume the vocal should still be completely intelligible but "
            "should no longer be the only thing you notice — you should be able to follow the "
            "bass line without effort while the vocal is singing. Level-matched against a "
            "genre reference, the vocal should sit at a similar distance. Re-analyze: the "
            "ratio inside the genre window, and master limiter gain reduction lower than "
            "before at the same output."
        ),
        learn_more=(
            "Perceived vocal level and measured vocal level come apart in ways worth knowing, "
            "because they are why this is so easy to get wrong.\n\n"
            "Compression is the biggest one. It reduces peaks and lets you raise the average, "
            "so a heavily compressed vocal sits several dB higher in perceived loudness at the "
            "same peak reading — your meter says nothing changed and your ear says the vocal "
            "came forward. Brightness is the second: hearing is most sensitive around 2-4 kHz, "
            "so energy there reads as loudness far more than the same energy at 200 Hz, which "
            "is why an air boost sounds like a level boost. And distance perception runs mostly "
            "on the ratio of dry to reflected sound, so a dry vocal is heard as close and a "
            "close sound is heard as loud, independent of its actual level.\n\n"
            "The practical upshot: when a vocal feels too loud, the fader is the third thing to "
            "try, not the first. Brightness, compression and dryness are usually what actually "
            "moved it forward, and each of them can be corrected without giving up the presence "
            "the fader would have taken with it."
        ),
        minutes=20,
        resources=(
            youtube_search(
                "how loud should the vocal be compare to reference track",
                "Settling this against a record instead of from memory",
                "Vocal placement is a genre convention, and this finding is a genre "
                "judgment — these cover level-matching a reference properly, which is the "
                "only way to tell 'too loud' from 'correct for pop'.",
            ),
            youtube_search(
                "vocal sounds like it is on top of the track add depth reverb delay",
                "Pushing a vocal back with space rather than level",
                "The fix that keeps the presence the fader would have taken with it: "
                "distance cues from short reverb and slap delay, for when the real complaint "
                "is depth rather than volume.",
            ),
        ),
    ),
)

_add(
    "vocal_balance.inconsistent",
    Explainer(
        headline="The vocal keeps drifting forward and back, so the listener has to keep "
                "re-focusing.",
        what_it_is=(
            "This is not an EQ problem and not a tone problem. It is about the vocal holding "
            "ONE place in the mix from the first line to the last.\n\n"
            "Two separate things can trigger it, and they have opposite fixes. The first is "
            "level spread: the difference between the vocal's loud moments and its quiet ones, "
            "measured only over the parts where it is actually singing, so silence between "
            "phrases does not count. The second is intelligibility: whether the quiet, "
            "high-frequency part of a voice that carries consonants — and therefore the actual "
            "words — is winning against what is around it. If the spread is what failed, the "
            "vocal "
            "itself is wandering. If intelligibility is what failed while the level looks "
            "steady, the vocal is holding still and the arrangement is moving underneath it — "
            "which is the classic 'fine in the verse, gone in the chorus' problem and needs "
            "fixing on the other parts.\n\n"
            "How much wander is acceptable scales with the genre. A folk record with a wide "
            "dynamic range earns a vocal that moves; a trap record where everything sits in a "
            "few decibels does not.\n\n"
            + _VOCAL_INFERENCE_NOTE
        ),
        what_you_hear=(
            "Some lines jump out and others disappear, and you find yourself rewinding to catch "
            "a word. Often the first line of a phrase is loud and the last word of it is gone, "
            "because singers relax at the end of a breath. It is more tiring to listen to than "
            "a vocal that is simply too quiet — a consistently quiet vocal fades into the "
            "background and stays there, while a wandering one keeps demanding attention and "
            "then withholding it. On headphones you may not notice at all; in a car or on a "
            "phone half the words go."
        ),
        why_it_matters=(
            "Consistency is most of what separates a mixed vocal from a raw one, and it is the "
            "difference a listener attributes to 'production quality' without being able to "
            "name it. It has a technical cost too: the loud moments set how hard the master "
            "limiter works, so a vocal with a wide spread drives the limiter with its peaks "
            "while its quiet parts are still too quiet — you get less loudness AND worse "
            "intelligibility from the same performance."
        ),
        common_causes=(
            "No volume automation. This is the usual answer, and it is a work problem rather "
            "than a skill problem: nobody has done the pass yet.",
            "One compressor being asked to do everything. Compression handles syllable-level "
            "movement well and phrase-level movement badly, and pushing it hard enough to catch "
            "the quiet phrases crushes the loud ones and drags up breaths and room.",
            "Mic technique. A singer moving toward and away from the mic changes tone as well "
            "as level — closer means more low end from the proximity effect, further means more "
            "room — so a fader move restores the level but not the sound.",
            "Comping across takes recorded on different days, at different distances, or with "
            "different preamp settings.",
            "A static vocal under a changing arrangement. The vocal level never moves, the "
            "chorus arrives with twice the instrumentation, and the vocal is effectively 3 dB "
            "quieter without a fader touching it.",
            "A de-esser or dynamic EQ over-reacting on some lines, pulling whole words down "
            "rather than just the sibilants.",
        ),
        how_to_fix=(
            FixStep(
                action="Even the level out in the edit, before any plugin. Phrase by phrase, "
                       "then word by word.",
                detail=(
                    "Clip gain is the volume control on the audio itself, ahead of the fader "
                    "and ahead of every plugin — every DAW has one, usually a handle on the "
                    "clip or a value in its info box. Go through the vocal with your eyes on "
                    "the waveform and level the phrases against each other so they LOOK even. "
                    "Then "
                    "catch the individual words that still jump or vanish. This is the single "
                    "most effective thing on this list, it takes twenty to forty minutes, and "
                    "there is no plugin that replaces it — every compressor downstream works "
                    "better once the input is even, because it is no longer being asked to fix "
                    "edit-level problems with a threshold."
                ),
            ),
            FixStep(
                action="Understand why compression alone cannot fix this.",
                detail=(
                    "A compressor reacts to level crossing a threshold. A phrase that is 5 dB "
                    "quiet for four bars sits below the threshold the entire time, so the "
                    "compressor does literally nothing to it. Lower the threshold to catch it "
                    "and you now crush every loud phrase and pull up breaths, headphone bleed "
                    "and room. Compression fixes syllable-to-syllable. Automation fixes "
                    "phrase-to-phrase. They are not substitutes and you need both."
                ),
            ),
            FixStep(
                action="Split the compression across two gentle stages.",
                detail=(
                    "An opto first, slow and level-dependent, doing 3-4 dB to smooth the broad "
                    "movement, then a faster compressor doing another 2-3 dB to catch peaks. "
                    "Two stages at 3 dB sound natural where one at 6 dB sounds squashed, "
                    "because each is working in the gentle part of its curve."
                ),
                needs="comp_opto",
                without=(
                    "Two instances of your stock compressor in series: the first with a low "
                    "ratio (2:1), slow attack and slow release doing 3 dB, the second faster "
                    "with a higher ratio doing 2-3 dB on peaks only. Same principle, and it "
                    "beats one compressor doing 6 dB every time."
                ),
            ),
            FixStep(
                action="Ride the vocal against the arrangement, not against itself.",
                detail=(
                    "Write automation that lifts the vocal 1-2 dB where the instrumentation "
                    "gets denser — the choruses, the double-tracked sections, wherever the "
                    "guitars come in — and drops it back where the track thins out. A vocal at "
                    "a constant level in a track that changes density is not actually constant "
                    "to the listener. Do this pass with the monitors quiet."
                ),
            ),
            FixStep(
                action="If it is the words rather than the level, fix the competition.",
                detail=(
                    "When the vocal's own level is steady but the words still come and go, the "
                    "problem is what surrounds it. Put a dynamic EQ on the instrumental bus "
                    "keyed from the vocal, a wide band across 1-4 kHz, a few dB of range. The "
                    "backing steps aside for the consonants only."
                ),
                needs="eq_dynamic",
                without=(
                    "Automate a 1-2 dB dip in 1-4 kHz on the instrumental bus through the "
                    "sections where the words get lost, and check the de-esser is not clamping "
                    "whole words rather than just the sibilants."
                ),
            ),
            FixStep(
                action="Accept that some of it is a re-take.",
                detail=(
                    "If a line changed distance from the mic, its tone changed too, and no "
                    "amount of level restores that. Re-recording two or three words takes five "
                    "minutes and beats an hour of trying to automate around it. Being honest "
                    "about which lines those are is a mixing skill."
                ),
            ),
        ),
        how_to_verify=(
            "Listen through once at conversation volume with your hand off the mouse and note "
            "every point where you strain or wince. Fewer than two in a full song is finished. "
            "Then check it in the car or on a phone, which is where inconsistency is most "
            "brutal. Re-analyze: the vocal's level spread should fall inside your genre's "
            "allowance and the intelligibility index should clear its floor — and if only one "
            "of those two moved, you fixed the wrong half."
        ),
        learn_more=(
            "The spread here is measured as the 90th percentile of the vocal's level minus the "
            "10th, over the frames where it is actually sounding. Percentiles rather than "
            "maximum and minimum, deliberately: one shouted word or one whispered aside should "
            "not define the whole vocal, and using the extremes would make every expressive "
            "performance look broken. So the number describes where the bulk of the vocal "
            "lives, and a genuinely dynamic performance with one big moment still measures as "
            "consistent — as it should.\n\n"
            "The reason automation beats compression for this is a difference in what each one "
            "can see. A compressor knows only the signal's level right now against a threshold, "
            "with a fixed attack and release; it has no idea whether it is in a verse or a "
            "chorus, whether the line matters, or whether the singer stepped back. You know all "
            "of that. Automation is you supplying the context the compressor cannot have, which "
            "is why the professional order is always the same: edit the level first, compress "
            "the residue second, then ride the result against the arrangement. Doing it in any "
            "other order means each stage is fighting the last one's mistakes."
        ),
        minutes=40,
        resources=(
            youtube_search(
                "clip gain vocal editing level phrases before compression",
                "The edit pass, which is the actual fix here",
                "This is a performance and automation problem, not a tone one — these show "
                "the phrase-by-phrase clip gain workflow that no plugin replaces, and why "
                "every compressor after it behaves better.",
            ),
            youtube_search(
                "vocal volume automation riding the fader mixing tutorial",
                "Riding the vocal against the arrangement",
                "Automation is the half of this a compressor cannot do, because it has no "
                "idea a chorus just arrived. These cover writing and refining the rides, "
                "which is what fixes phrase-to-phrase drift.",
            ),
            Resource(
                kind="reference",
                label="Vocal Production",
                url="https://www.soundonsound.com/techniques/vocal-production",
                source="Sound On Sound",
                note=(
                    "Paul White arriving at the same order of work from the other end: fix "
                    "the level disparities with automation first and leave the compressor to "
                    "shape tone rather than to level a performance it cannot see the shape "
                    "of."
                ),
            ),
        ),
    ),
)


def register(target: Dict[str, Explainer]) -> None:
    """Copy these explainers into the shared registry.

    Called by `knowledge.py` after its own definitions are in place. Copying
    rather than mutating a module-level import keeps this file free of any
    reference to `knowledge.EXPLAINERS`, which would be a circular import.
    """
    target.update(EXPLAINERS)
