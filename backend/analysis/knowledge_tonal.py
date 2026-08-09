"""Teaching content for the tonal-balance findings.

Mud, harshness, sibilance, and the per-band frequency-balance family. These are
the findings a producer clicks on most, and they are the ones where a restated
measurement is least useful: "the low mids are 4 dB over target" tells you
nothing you can act on unless you already know what the low mids do.

Three things this file tries to do that a spec sheet does not:

1. **Name the sound before the science.** Boxy, honky, woolly, glassy. A
   producer recognises those instantly and recognises "excess 300-600 Hz" only
   after they already know the answer.
2. **Say what lives in the band.** Every one of the eighteen band explainers
   names what is usually responsible, because "presence is low" is a
   measurement and "your vocal sounds like it is behind the speakers and you
   keep pushing the fader to fix it" is a diagnosis.
3. **Never quote this track's numbers.** These strings ship identically for
   every upload. The measured values live on the finding; the meaning lives
   here.

Registered into `knowledge.EXPLAINERS` by `register()`, which knowledge.py
calls. Nothing here touches that dict at import time — doing so would create a
circular import.
"""

from __future__ import annotations

from typing import Dict

from .knowledge import Explainer, FixStep

EXPLAINERS: Dict[str, Explainer] = {}


def _add(finding_id: str, explainer: Explainer) -> None:
    EXPLAINERS[finding_id] = explainer


# ---------------------------------------------------------------------------
# Mud
# ---------------------------------------------------------------------------

_add(
    "mud.low_mid_buildup",
    Explainer(
        headline="Your mix sounds covered — there is a wall of energy sitting between the bass and everything else.",
        what_it_is=(
            "Mud is a ratio, not a level. The region roughly between 150 and 400 Hz is the "
            "body of almost everything you recorded: the harmonics of the bass, the thump "
            "behind the kick's click, the low end of guitars, keys and pads, the chest of the "
            "singer, the shell of the toms. It is supposed to be busy. So the measurement asks "
            "three questions rather than one — how loud that region is against the bass below "
            "it, how loud it is against the mids above it, and whether 300-600 Hz specifically "
            "is standing above the mix's own average level. A big low end is not mud. A big low "
            "end with nothing separating it from the region above it is."
        ),
        what_you_hear=(
            "Thick, covered, indistinct. There is plenty going on and you cannot pick any of it "
            "out. The most reliable tell is what happens when you turn it up: a clear mix gains "
            "detail with volume, a muddy one only gains weight. Consonants go first, so the "
            "vocal becomes hard to follow before it becomes hard to hear. The kick loses its "
            "point. Acoustic guitars start to sound like a cardboard box with strings on it. It "
            "is usually more obvious on a phone or a laptop than on your monitors, because a "
            "small speaker cannot reproduce the sub that was distracting you from it."
        ),
        why_it_matters=(
            "This region carries a large share of the total energy in a mix, so your limiter is "
            "mostly reacting to it — every kick and every bass note pulls the whole track down "
            "and you spend loudness on something nobody wants to hear. It also masks upwards far "
            "more strongly than it masks downwards, which is why the instinctive fix of turning "
            "the vocal up does not work. The vocal gets louder, the buildup gets no quieter, and "
            "the mix is now loud and still covered."
        ),
        common_causes=(
            "Nothing is high-passed. Eight sources each carrying a modest amount of 200 Hz "
            "content sum into a wall that none of them contains on its own.",
            "The room. Untreated rooms resonate hardest right through this region, and that "
            "resonance is printed into every microphone you put in it.",
            "Proximity effect: any directional mic lifts the low end as it gets closer, so a "
            "singer who leans in has a different low-mid balance line to line.",
            "'Warm' processing — tape, tube and console emulations mostly add their harmonic "
            "content low, and three of them in a chain add it three times.",
            "Reverb and delay returns with no high-pass. Tails live longest exactly here and "
            "almost nobody solos a return to check.",
            "A layered bass or 808 stack where every layer contributes its own overtones "
            "between 150 and 300 Hz.",
        ),
        how_to_fix=(
            FixStep(
                action="First decide whether this is a ratio problem or a tilt problem.",
                detail=(
                    "Compare against a released track in the same genre at matched volume and "
                    "listen for whether the reference has a clear gap between its bass and its "
                    "mids that yours does not. If your entire low end is heavy, that is tilt "
                    "and one broad move fixes it. If only the region above the bass is heavy, "
                    "that is mud, and one broad move will make the track thin."
                ),
                needs="reference_matching",
                without=(
                    "Drop a released track you like onto a muted channel, pull its fader until "
                    "the two feel equally loud, and switch between them at low volume. The "
                    "level match matters more than the tool — louder always sounds better, and "
                    "that is how mixes get talked into being muddy."
                ),
            ),
            FixStep(
                action="Take a little out of several sources instead of a lot out of one.",
                detail=(
                    "A wide bell — the hump-shaped EQ band, where Q sets how wide the hump is "
                    "and a low number means wide, so Q 0.7 to 1 covers most of an octave — "
                    "somewhere between 200 and 350 Hz, pulling "
                    "1.5 to 2 dB, on the three or four biggest contributors. Separate sources "
                    "pile up in a region rather than adding neatly, so four of them cut 2 dB "
                    "each take about as much out of it as one cut by 8 — and none of the four "
                    "ends up sounding thin. This is the whole trick: mud is a stacking problem, "
                    "so it needs a distributed fix."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="High-pass everything that does not need low end at all.",
                detail=(
                    "Roughly 80-120 Hz on a lead vocal and higher on backing vocals, 100-200 Hz "
                    "on electric guitars, keys and pads, 300 Hz and up on hats and shakers, and "
                    "250-300 Hz on every reverb and delay return. None of what you remove was "
                    "audible as bass; it was only adding to the pile."
                ),
            ),
            FixStep(
                action="Put a dynamic band on the persistent offender rather than a static cut.",
                detail=(
                    "A band around 200-300 Hz that only pulls when the region crosses its "
                    "threshold leaves a sparse verse warm and acts only in the dense chorus, "
                    "which is where the problem actually is. Set the threshold so the gain meter "
                    "moves in full sections and sits still in the quiet ones."
                ),
                needs="eq_dynamic",
                without=(
                    "Use a static cut and then automate the EQ off in the sparse sections. The "
                    "region is usually only a problem when everything is playing at once, and a "
                    "cut that is right for the chorus will hollow out the intro."
                ),
            ),
            FixStep(
                action="If the buildup is one ringing note rather than a broad heaviness, treat it as a resonance.",
                detail=(
                    "A suppressor tracks the peak and ducks it only while it rings, which is the "
                    "right behaviour for a room mode or a resonant cabinet — those come and go "
                    "with the note being played."
                ),
                needs="resonance_suppressor",
                without=(
                    "Find it by hand: a narrow bell boosted 6 dB, swept slowly from 150 to 400 "
                    "Hz until one spot rings much louder than its neighbours. Invert that boost "
                    "into a 3-4 dB cut at the same frequency, on the source, and check it in the "
                    "full arrangement rather than solo — a notch that sounds right solo is "
                    "usually twice as deep as it needs to be."
                ),
            ),
        ),
        how_to_verify=(
            "Play the busiest section quietly on a small speaker or a phone. You should be able "
            "to follow the lead without turning it up. In a fresh analysis, 150-400 Hz should "
            "come back inside your genre's window against the bass beneath it, and 300-600 Hz "
            "should sit close to the average level across the rest of your own mix. One warning "
            "sign: if the track "
            "now sounds small and thin rather than clear, you took the energy out of one place "
            "instead of several. Put it back and spread the cut."
        ),
        learn_more=(
            "Two mechanisms make this region behave the way it does. The first is that masking "
            "is asymmetric — it works far better upwards than downwards. Inside your inner ear, "
            "sound is sorted by frequency along a membrane, and the wave that does the sorting "
            "travels from the high-frequency end toward the low, so a loud low frequency ends up "
            "shading everything above it on the way past. Energy at 250 Hz covers up sounds a "
            "kilohertz and higher; energy at a kilohertz barely touches 250 Hz. That is why the "
            "region below the mids is the one that hides things, and why a mix can measure "
            "bright and still sound like there is a blanket over it. The second mechanism is how "
            "energy adds up. Two separate sources putting a similar amount into the same region "
            "are not twice as loud as one, but they are about 3 dB more — and that 3 dB repeats "
            "every time the count doubles, so a level that is harmless on one track is not "
            "harmless on twenty. Both facts point the same way: fix it at the sources, in small "
            "amounts, in several places."
        ),
        minutes=30,
    ),
)


# ---------------------------------------------------------------------------
# Harshness
# ---------------------------------------------------------------------------

_add(
    "harshness.upper_mid_edge",
    Explainer(
        headline="The mix is tiring — after half a minute you want to turn it down, and listeners will.",
        what_it_is=(
            "2 to 5 kHz is the region your ear is most sensitive to, and this measurement is "
            "deliberately not 'is that region loud'. It asks how far the region stands above the "
            "line its own neighbours draw: the analyser looks at the span below it and the span "
            "above it, works out the slope your mix is on, and measures the departure. A mix "
            "with an ordinary smooth downward tilt scores near zero no matter how bright it is. "
            "A mix with a shelf or a spike sitting on top of that slope scores high even if it "
            "is dark overall. Brightness is the slope of the whole spectrum; harshness is one "
            "region leaving it. You can be brighter than your reference and less harsh than it, "
            "and that is usually exactly what you are aiming for. A second figure reported "
            "alongside — sharpness — weights the spectrum the way the ear does, so a mix has to "
            "clear both."
        ),
        what_you_hear=(
            "Not brightness. Glare. An edge on vocal consonants, a hard white front on distorted "
            "guitars, cymbals that feel like they are being pushed at your face. The tell is "
            "duration rather than level: the mix sounds exciting for twenty seconds, and then "
            "you notice you have turned it down, and the volume you can comfortably sit at keeps "
            "drifting lower. On nearfield monitors with a dip in this region — which is a lot of "
            "them — you may not hear it at all. On earbuds, laptop speakers and car tweeters, "
            "all of which peak here, it is the first thing anyone notices."
        ),
        why_it_matters=(
            "This is the region that decides how long someone listens, and playback hardware "
            "stacks its own peak on top of yours, so the listener gets more of it than you did. "
            "It also poisons your own work: a fatiguing mix gets monitored quieter, low-volume "
            "monitoring hides the low end, and the next hour of decisions is made on bad "
            "information. And a harsh master gets turned down by the listener, which costs you "
            "the loudness you spent the whole chain fighting for."
        ),
        common_causes=(
            "A presence boost added on the mix bus or master to make the track read on a laptop. "
            "It works for about thirty seconds.",
            "The same boost on several sources at once — vocal at 3k, guitars at 3k, snare at "
            "3k. Each is defensible on its own. The sum is not.",
            "Distortion, saturation or clipping applied to something low. Harmonics land far "
            "above the note that made them, so an overdriven bass or a hard-clipped drum bus "
            "shows up as an edge in the upper mids.",
            "One resonant source: a bright condenser's presence bump, an amp-sim cabinet "
            "resonance, a snare ring that nobody damped.",
            "A limiter working too hard. Its release lets the midrange swell back between kicks, "
            "and that movement reads as edge rather than as pumping.",
            "'Air' and 'exciter' plugins whose band is centred lower than the name suggests.",
        ),
        how_to_fix=(
            FixStep(
                action="Work out whether it is a spike or a shelf. They need opposite tools.",
                detail=(
                    "Take a bell with a high Q, boost it about 6 dB, and sweep slowly from 2 to "
                    "5 kHz across the full mix. If one spot screams far louder than everywhere "
                    "else, a single source has a resonance and you should fix it there. If the "
                    "whole span comes up evenly, the region has been shelved up and no notch "
                    "will help."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="If it is a spike, duck it dynamically instead of notching it out.",
                detail=(
                    "The peak is only a problem while the source that carries it is playing. A "
                    "suppressor tracks it and leaves the rest of the arrangement alone, which "
                    "matters here more than anywhere else — a static notch in this region is "
                    "audible as a hole the moment the offending source stops."
                ),
                needs="resonance_suppressor",
                without=(
                    "Notch it at the frequency you found: narrow bell, 3 to 4 dB down, on the "
                    "source rather than the mix bus. Judge the depth in the full arrangement, "
                    "never solo."
                ),
            ),
            FixStep(
                action="If it is a shelf, stagger the sources rather than cutting the sum.",
                detail=(
                    "Decide which single instrument owns 2-5 kHz — usually the lead vocal — and "
                    "move everyone else's clarity boost somewhere it does not fight: guitars "
                    "around 1.5-2 kHz, snare up at 5-6 kHz, hats above 8 kHz. You get the same "
                    "intelligibility for a fraction of the energy in the region the ear is most "
                    "sensitive to."
                ),
            ),
            FixStep(
                action="Pull the region only when it is actually loud.",
                detail=(
                    "A dynamic band around 3 kHz on the mix bus, threshold set so it does "
                    "nothing in the verse and 1-2 dB in the chorus, fixes the loud sections "
                    "without dulling the quiet ones. Slow-ish release, or the movement itself "
                    "becomes audible."
                ),
                needs="eq_dynamic",
                without=(
                    "Use a static wide cut of about 1 dB, Q around 0.7, near 3 kHz, and automate "
                    "it deeper only in the loudest sections. Small and wide beats large and "
                    "narrow when the whole region is up."
                ),
            ),
            FixStep(
                action="Bypass your saturators and clippers one at a time.",
                detail=(
                    "If the edge disappears when a distortion stage comes out, the fix is less "
                    "drive there, not EQ afterwards. Distortion products are spread across the "
                    "spectrum and interleaved with the signal, so you cannot filter them out "
                    "without taking the wanted content with them."
                ),
            ),
            FixStep(
                action="Hold the region with a multiband band instead of an EQ move if it only peaks occasionally.",
                detail=(
                    "One band across 2-5 kHz, catching 1-2 dB on peaks and nothing the rest of "
                    "the time. Attack slow enough to let transients through, or you will trade "
                    "the edge for a blunt snare."
                ),
                needs="comp_multiband",
                without=(
                    "Turn down the level feeding your limiter by a dB or two and listen again. "
                    "If the edge softens, the limiter was generating it and no EQ placed after "
                    "it will fix the cause."
                ),
            ),
        ),
        how_to_verify=(
            "Play the loudest section at conversational volume for a full minute and see whether "
            "you reach for the volume knob. On earbuds, consonants and cymbals should sit behind "
            "the vocal rather than in front of it. In a fresh analysis the harshness index "
            "should come back under your genre's ceiling — rock, punk and metal are allowed "
            "noticeably more up here than R&B, acoustic, jazz or lo-fi, and the ceiling moves "
            "with the genre you selected. The overall tilt should barely move. If the mix got "
            "darker, your cut was too wide and you removed brightness instead of edge."
        ),
        learn_more=(
            "Your ear canal is roughly a 25 mm tube closed at one end by the eardrum, which "
            "makes it a quarter-wave resonator with a resonance close to 3 kHz. It hands you "
            "something like 10-15 dB of free acoustic gain in exactly this region before the "
            "sound even reaches the eardrum, and the equal-loudness contours show the result: "
            "around 3 kHz you need roughly 10 dB less sound pressure than at 1 kHz to perceive "
            "the same loudness. Everything about this band follows from that one fact. It is "
            "where speech intelligibility lives, so it is genuinely valuable real estate; it is "
            "also where the ear has the least headroom, so two decibels too many is not a small "
            "error. Treat it as a budget with one tenant: something is allowed to be loud there, "
            "but only one thing at a time."
        ),
        minutes=25,
    ),
)

_add(
    "harshness.sibilance",
    Explainer(
        headline="The S and T sounds are jumping out in front of the words they belong to.",
        what_it_is=(
            "Sibilance is the noise burst inside consonants — s, sh, ch, t, z, f. Unlike a sung "
            "note it has no pitch: it is turbulent air over teeth and tongue, and it lands "
            "roughly between 5 and 9 kHz. Two things separate it from air and general "
            "brightness, and the measurement uses both. It sits above the line its neighbouring "
            "regions draw, and — more usefully — it is intermittent. It arrives in short bursts "
            "a few times a bar instead of running continuously, so the index compares the "
            "loudest frames in the band against the typical frame. A steady shimmer of cymbals "
            "and room, which is air, barely registers. A vocal that spits scores high even in a "
            "mix that is dull overall."
        ),
        what_you_hear=(
            "The 's' is louder than the word it is attached to. It whistles, spits, or in a bad "
            "case sounds like a short burst of static in front of the vowel. On monitors at a "
            "sensible volume it can pass as detail; on earbuds, where the driver sits a "
            "centimetre from the eardrum with a response that already peaks in the same region, "
            "it is a needle. In a car it is the part of the vocal that survives the road noise "
            "best, so the singer sounds like they are hissing at you."
        ),
        why_it_matters=(
            "A sibilant peak is still a peak. Your limiter sees it, ducks the whole mix for a "
            "fraction of a second and lets go, so every 's' punches a small hole in the track. "
            "Lossy encoders also deal badly with short, noisy, high-frequency events — an ess is "
            "exactly the signal that gets smeared into something longer and grainier than what "
            "you exported. And it is the most common single defect in vocal-forward music, which "
            "is why the tolerance for it is set tighter for R&B and soul than for rock."
        ),
        common_causes=(
            "A bright large-diaphragm condenser aimed straight at the mouth from close range. "
            "Moving the singer a few degrees off-axis at the source beats anything you can do "
            "afterwards.",
            "A high shelf or air boost across the whole vocal. It adds breath and sibilance in "
            "equal measure, because they are the same frequencies.",
            "Compression before anything that controls the top. Heavy vocal compression brings "
            "the body up, and a bright EQ after it then makes the ess disproportionate.",
            "Saturation or a parallel distortion bus on the vocal, generating new high harmonics "
            "out of the consonants.",
            "Stacked doubles and backing vocals: four takes saying 's' a few milliseconds apart "
            "sum into one long ess far louder than any single take.",
            "A de-esser placed before the thing that causes the problem. It cannot control what "
            "has not happened yet.",
        ),
        how_to_fix=(
            FixStep(
                action="Confirm it is actually the singer before you treat it.",
                detail=(
                    "Without separate stems this is measured on an extracted centre channel, not "
                    "on the real vocal track, so a centred hi-hat, shaker or tambourine can "
                    "inflate the reading. Loop a line and listen for whether the spit is on the "
                    "consonants. If it lands on the offbeats instead, the problem is percussion "
                    "and a de-esser is the wrong tool."
                ),
            ),
            FixStep(
                action="Put whatever controls the top last in the chain.",
                detail=(
                    "Order is most of this fix. Anything that brightens the vocal — EQ, "
                    "saturation, exciter — has to sit upstream of the de-essing, or you are "
                    "treating the version of the signal before the problem was created."
                ),
            ),
            FixStep(
                action="Find the frequency instead of guessing it.",
                detail=(
                    "Loop the worst word. Take a narrow bell, boost it 6 dB and sweep from 5 to "
                    "9 kHz until the ess becomes intolerable — that is your centre. Deeper "
                    "voices usually sit in the lower half of that span and brighter voices "
                    "higher, but the ranges overlap enough that sweeping is faster than "
                    "assuming."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Use a split-band de-esser, not a broadband one.",
                detail=(
                    "This is the difference that decides whether de-essing sounds invisible or "
                    "ruins the vocal. A broadband de-esser is a compressor with a filtered "
                    "sidechain: it listens to 5-9 kHz and, when it hears an ess, turns the "
                    "entire vocal down. The whole word flinches and dulls, and on a sustained "
                    "note it is obvious. A split-band de-esser divides the signal, ducks only "
                    "the band it is listening to, and leaves the vowel underneath completely "
                    "untouched. If your de-esser has a mode switch, that is the switch. Aim for "
                    "3-6 dB of reduction on the worst esses and none at all on the rest of the "
                    "line — if the meter moves during vowels, the threshold is too low."
                ),
                needs="deesser",
                without=(
                    "Automate it by hand: pull the clip gain or draw the fader down 2-4 dB on "
                    "each offending consonant. It is slow and it is still the best-sounding fix "
                    "there is, because it does literally nothing to the rest of the vocal. Keep "
                    "a static cut at the ess frequency as a last resort — it removes the ess and "
                    "every bit of breath on every other word too."
                ),
            ),
            FixStep(
                action="With dynamic EQ, use two narrow bands rather than one wide one.",
                detail=(
                    "Around 6 kHz catches 'sh' and 'ch'; around 8 kHz catches 's' and 't'. Fast "
                    "attack, fast release, and a range limit so neither band can ever pull more "
                    "than about 6 dB. A dynamic band is a de-esser you can place exactly, which "
                    "matters when a singer has two distinct problem frequencies."
                ),
                needs="eq_dynamic",
            ),
            FixStep(
                action="On a finished stereo mix, work on the centre only.",
                detail=(
                    "The esses are in the middle and the hats and cymbals mostly are not, so a "
                    "dynamic band applied to the mid channel takes the ess down without dulling "
                    "the top of the whole record. Keep it to a couple of dB — this is damage "
                    "control, not the real repair."
                ),
                needs="eq_mid_side",
                without=(
                    "Without mid/side control on a full mix, anything you do to the esses also "
                    "hits the hi-hats and cymbals. Hold the reduction under about 2 dB and "
                    "accept the compromise, or go back to the vocal track — this is one problem "
                    "that is dramatically cheaper to fix before the bounce."
                ),
            ),
            FixStep(
                action="Stop before it lisps.",
                detail=(
                    "The failure mode of de-essing is a singer who suddenly says 'th' instead of "
                    "'s', and it is more noticeable than the sibilance was. Loop a line and "
                    "toggle the processing: the consonant should still clearly be a consonant, "
                    "just no louder than the vowel beside it."
                ),
            ),
        ),
        how_to_verify=(
            "Loop the worst line and bypass and re-engage. The consonant should be present and "
            "no louder than the vowel next to it, and the word must not sound like a lisp. Then "
            "listen to the cymbals and hats: if the whole top of the mix moved with the vocal, "
            "your reduction is broadband and you want a split-band tool. In a fresh analysis the "
            "sibilance index should come back under the genre ceiling, which is set tighter for "
            "R&B and soul on purpose — that is where sibilance is most often the thing that gets "
            "a mix sent back."
        ),
        learn_more=(
            "Why intermittency is the right way to measure this: a fricative is turbulent noise "
            "lasting 50 to 150 milliseconds, with a broad spectral peak whose centre shifts with "
            "the singer's anatomy and mouth shape. Cymbal wash, breath and room tone occupy the "
            "same band and are near-continuous. Average the whole track and the two are "
            "indistinguishable, which is why a plain '5-9 kHz level' reading is useless here and "
            "why brightening a mix does not automatically make it sibilant. Look instead at how "
            "much louder the loudest frames are than the typical ones and they separate cleanly. "
            "The same fact explains why the correct tool is a dynamics processor rather than an "
            "EQ: the offending energy exists for perhaps two percent of the track's duration, "
            "and a static filter charges you for it across the other ninety-eight."
        ),
        minutes=20,
    ),
)


_add(
    "harshness.bright_transients",
    Explainer(
        headline=(
            "The top of your beat is spitting, and it is the hats and shakers doing it — "
            "not a vocal."
        ),
        what_it_is=(
            "This is the same measurement that finds sibilance, reported honestly about a "
            "different source. The analyser looks at 5-9 kHz and asks how much louder the "
            "loudest frames are than the typical ones. A steady shimmer of cymbals and room "
            "scores near zero however bright the mix is; something that arrives in short, hard "
            "bursts scores high. An 's' does that — and so does a closed hi-hat, a shaker, a "
            "tambourine and a rim click, which are all 30-60 millisecond noise bursts in "
            "exactly the same octave.\n\n"
            "So the number alone cannot tell you which. What decides it here is whether there "
            "is a lead vocal sitting up in the mix for those bursts to belong to. On this "
            "track there is not — either no voice was detected at all, or the voice that was "
            "detected is tucked well under the bed, which is where a hook goes when somebody "
            "is going to rap over the beat. Either way, the thing making the top burst is the "
            "percussion, and this finding says so instead of calling it sibilance and sending "
            "you to a de-esser.\n\n"
            "Worth being straight about the confidence: without separated stems this is an "
            "inference. It is a strong one — a bursty top octave on a track with no lead up "
            "front is hats far more often than it is anything else — but it is not a "
            "measurement of the hi-hat channel. Run the analysis with stem separation on and "
            "the question is answered directly, by reading the 5-10 kHz share off each "
            "separated source, and the confidence on this finding goes up accordingly."
        ),
        what_you_hear=(
            "A top end that pecks at you. Individual hat hits poke out in front of the beat "
            "instead of sitting in it, the shaker sounds like it is in a different, closer "
            "room than everything else, and the groove reads as busy rather than tight. "
            "Turn it up and it becomes tiring within a minute or two even though nothing "
            "sounds obviously wrong — that fatigue is the tell, because the ear is at its "
            "most sensitive right through this band.\n\n"
            "It is worst exactly where your beat will actually be heard. Earbuds and phone "
            "speakers both have a response peak in the same region, so a hat pattern that "
            "sounds crisp on monitors turns into a tick track on a bus. And if a rapper puts "
            "a vocal on top of it, their consonants land in this same octave and now have to "
            "fight the hats for it — the words lose their edge and the natural response is to "
            "brighten the vocal, which makes the whole top harder still."
        ),
        why_it_matters=(
            "Three concrete costs, in the order they bite.\n\n"
            "Your limiter sees every one of those bursts as a peak. It ducks the whole track "
            "for a few milliseconds and lets go, so a busy hat pattern punches dozens of small "
            "holes in the low end per bar. That is a large part of why a beat can measure loud "
            "and still feel weak — the kick is being modulated by the hats.\n\n"
            "Lossy encoders handle short, noisy, high-frequency events badly. A hat is close to "
            "the worst case, and what comes out of the encoder is smeared and grainier than "
            "what you exported. The louder and pointier the transient, the more obvious the "
            "damage.\n\n"
            "And it spends headroom on the part of the record that is not the record. If this "
            "is a beat, the topline is going to want that 5-9 kHz room for consonants. "
            "Whatever the hats are holding there, the vocal has to be pushed past."
        ),
        common_causes=(
            "A stock hi-hat sample that is already bright, then a high shelf across the drum "
            "bus on top of it. Most one-shots in a modern pack are pre-brightened, so the "
            "shelf you would add to a live kit is a second helping.",
            "A transient shaper or a clipper on the drum bus with attack pushed up. Both make "
            "the hat's leading edge sharper, which is precisely the thing this measures.",
            "Layered percussion: a closed hat, a shaker and a tambourine on the same 16ths. "
            "Three sources with the same spectral centre stack into one burst that is far "
            "louder than any of them, and the pattern reads as a single very loud hat.",
            "Bus compression with a slow attack on the drums. It lets the transient through "
            "untouched and then turns down the body behind it, which raises the transient's "
            "level relative to everything around it.",
            "Saturation or an exciter on the master. Both generate new high harmonics from the "
            "sharpest edges in the mix, and the sharpest edges in a beat are the hats.",
            "No dynamic control on the percussion group at all — normal, and usually fine, "
            "until the hats are the loudest thing in the top two octaves.",
        ),
        how_to_fix=(
            FixStep(
                action="Solo the hats and confirm it before you treat anything.",
                detail=(
                    "Without stems this attribution is an inference. Loop eight bars and "
                    "listen for where the spit lands. On the offbeats and the 16ths it is the "
                    "hats and shakers, and everything below applies. On the consonants of a "
                    "vocal it is sibilance and you want a de-esser instead. If you can, "
                    "re-run the analysis with stem separation on and let it answer directly."
                ),
            ),
            FixStep(
                action="Pull the attack down on the percussion bus with a transient shaper.",
                detail=(
                    "This is the right tool and it is not the obvious one. The problem is not "
                    "that the hats are loud, it is that their leading edge is far louder than "
                    "their body — which is exactly the ratio a transient shaper controls and "
                    "exactly the ratio an EQ cannot touch. Take the attack down 3-6 dB on the "
                    "hat and shaker group and leave the level alone. The pattern keeps its "
                    "place in the groove and stops pecking."
                ),
                needs="transient_shaper",
                without=(
                    "A fast compressor gets most of the way there: 4:1 or so, attack under "
                    "1 ms so it actually catches the transient, fast release, and aim for "
                    "2-4 dB of reduction on the hits and none between them. Slower than about "
                    "5 ms of attack and you are doing the opposite of what you want — the "
                    "transient passes through and the body gets ducked behind it."
                ),
            ),
            FixStep(
                action="Take the shelf off the hat bus before you add anything else.",
                detail=(
                    "If there is a high shelf on the drums or the master, bypass it and "
                    "listen. On a pack sample that is usually the entire problem, and "
                    "removing 2 dB of shelf you did not need beats adding a dynamic band to "
                    "fight it."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Put a dynamic band on the percussion group, not the master.",
                detail=(
                    "One band at 7-8 kHz, wide-ish (Q around 1.0), threshold set so it only "
                    "moves on the loudest hits, range limited to 4 dB. On the percussion group "
                    "it takes the peaks off the hats and leaves the vocal, the keys and the "
                    "snare's body completely alone. The same band on the master would duck the "
                    "whole record every time a hat lands, which is the mistake this step "
                    "exists to avoid."
                ),
                needs="eq_dynamic",
                without=(
                    "A multiband compressor's top band does the same job: crossover around "
                    "6 kHz, fast attack and release, 2-3 dB of reduction on the peaks. Put it "
                    "on the percussion group for the same reason."
                ),
            ),
            FixStep(
                action="Thin the layers before you process them.",
                detail=(
                    "If a closed hat, a shaker and a tambourine are all on the 16ths, the "
                    "cheapest fix is to keep one of them and drop the other two 6 dB, or "
                    "filter each to a different part of the band so they stop stacking. "
                    "Three bright sources on the same grid is an arrangement problem, and "
                    "no amount of dynamics processing turns it back into one source."
                ),
            ),
            FixStep(
                action="If this is a beat, leave the topline room on purpose.",
                detail=(
                    "Check the hats against where a vocal will sit: pull the group down 1-2 dB "
                    "and see whether the beat still reads. Most do, and the 5-9 kHz room you "
                    "just freed is what stops the rapper's consonants from having to be shoved "
                    "through the hat pattern later."
                ),
            ),
            FixStep(
                action="Keep it if the brightness is the point.",
                detail=(
                    "Hats cutting hard is a genre choice and a good one in plenty of records. "
                    "This is a difference from the reference, not a defect. If the beat is "
                    "supposed to feel fast and glassy and it does, the correct action is none "
                    "— just know the limiter and the encoder are both paying for it, and give "
                    "the master a little more headroom than you otherwise would."
                ),
            ),
        ),
        how_to_verify=(
            "Loop the busiest eight bars at a realistic volume and listen for a full minute. "
            "The hats should read as part of the groove rather than as individual events in "
            "front of it, and you should not feel like turning it down. Then watch the "
            "limiter's gain-reduction meter with the drums soloed: it should not be moving on "
            "every hat. In a fresh analysis the 5-9 kHz burstiness index should come back "
            "under the ceiling — which is set wider on a track with no lead vocal than on one "
            "with a singer, because percussion is supposed to cut and consonants are not. And "
            "check the low end afterwards: taking the peaks off a busy hat pattern usually "
            "makes the kick sound louder without you touching it, because the limiter has "
            "stopped ducking underneath it."
        ),
        learn_more=(
            "Why this measurement cannot name its own source, and why that matters. A "
            "fricative is turbulent noise lasting 50-150 ms with a broad spectral peak "
            "somewhere between 5 and 9 kHz. A closed hi-hat is a metal-on-metal impulse "
            "lasting 30-60 ms with a broad spectral peak in the same place. In a summed "
            "two-track they are the same kind of object: short, noisy, bright, intermittent. "
            "No statistic computed on the stereo file separates them, which is why the honest "
            "move is to look for corroborating evidence — is there a voice on this record at "
            "all, and if so is it sitting where a lead sits — rather than to assume.\n\n"
            "That distinction is not academic, because the two problems have opposite fixes. "
            "Sibilance wants a split-band de-esser on the vocal, which reduces a band for the "
            "50 ms an 's' lasts and leaves the vowel untouched. A spitting hat wants attack "
            "control on the percussion bus, which changes the shape of the transient and "
            "leaves the band alone. Apply the first to the second and you dull the whole top "
            "of the record every time a hat lands; apply the second to the first and you "
            "flatten the singer's diction. Getting the attribution right is most of getting "
            "the fix right, and it is the reason separating stems is worth the extra minute "
            "of processing on anything percussive."
        ),
        minutes=15,
    ),
)


# ---------------------------------------------------------------------------
# Frequency balance — one explainer per macro band, per direction.
#
# The detector emits at most one hot and one thin band per analysis, chosen by
# how far outside that band's own genre tolerance it sits. So these fire on the
# worst offender, and a band already claimed by a more specific detector (mud
# takes the upper bass and low mids, harshness takes presence, sibilance takes
# brilliance, the low-end detector takes sub and low bass) stands down here.
# ---------------------------------------------------------------------------

_add(
    "frequency_balance.sub_hot",
    Explainer(
        headline="There is more weight under the track than it needs, and most of it is doing nothing you can hear.",
        what_it_is=(
            "The sub band is the bottom two octaves of music — below roughly 60 Hz. Almost "
            "nothing has its fundamental down there except an 808, a sub-bass synth, the lowest "
            "part of a kick, and the bottom string of a five-string bass. You do not really hear "
            "this region as pitch; you feel it as pressure and weight. Everything else that ends "
            "up there is unintentional: mic-stand thumps, footsteps, HVAC, plosives, room rumble, "
            "and the DC-ish garbage under a badly rendered sample. The measurement compares this "
            "band against the level of your own midrange and against what your genre's curve "
            "expects, so it is not an absolute rule — it is 'more than this style of music uses'."
        ),
        what_you_hear=(
            "On a system that can actually reproduce it: the low end feels loose and unfocused, "
            "notes bloom into each other, and the rest of the mix seems oddly quiet for how hard "
            "the meters are working. On a laptop, a phone or most earbuds: nothing at all. That "
            "silence is the trap. The energy is still in the file, still eating your headroom, "
            "still triggering your limiter, and completely invisible on the speakers most "
            "producers check on."
        ),
        why_it_matters=(
            "Sub is the most expensive energy in a mix. It takes enormous amplitude to be "
            "perceived at a moderate loudness, so it dominates peak level and it is what your "
            "limiter spends its gain reduction on — which means the snare, the vocal and the "
            "guitars all get pulled down every time an 808 lands. On a club system it is worse "
            "than a balance problem: excess sub drives woofers into excursion, and the "
            "distortion that produces is spread up through the mids where you can hear it."
        ),
        common_causes=(
            "More than one source owning the bottom: a kick with a sub layer, an 808, and a sub "
            "synth all playing the same octave.",
            "An 808 or sub patch with a long tail that has not been shortened to the groove, so "
            "each note is still ringing when the next arrives.",
            "No high-pass anywhere in the chain, so rumble, plosives and handling noise from "
            "every mic'd source pile up beneath the music.",
            "A bass part written low enough that its fundamental falls below where most systems "
            "reproduce — the note is there, but only the meter knows.",
            "Mixing on headphones with a big low-end lift, or in a room with a null at the "
            "listening position, so the sub sounds correct while you add too much.",
        ),
        how_to_fix=(
            FixStep(
                action="High-pass the master at 25-30 Hz with a steep slope.",
                detail=(
                    "24 dB per octave. Nothing musical lives below that in almost any genre, and "
                    "nothing that does reproduces on any consumer system. It is free headroom "
                    "and the single fastest move here."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Pick one source to own the bottom octave and high-pass the others out of it.",
                detail=(
                    "Usually the 808 or the sub synth owns it and the kick keeps its punch "
                    "higher up, or the reverse — but not both. Two sources sharing the bottom "
                    "octave do not sound twice as big; they sound loose, because their waveforms "
                    "are long enough to add and cancel with each other note by note."
                ),
            ),
            FixStep(
                action="Duck the sub source from the kick.",
                detail=(
                    "A short, fast sidechain keyed off the kick — a few dB, released within the "
                    "length of the kick — lets the transient through cleanly and stops the two "
                    "adding into one long overload. This is the honest fix for kick and 808 "
                    "collision, as opposed to just turning one down."
                ),
                needs="comp_sidechain_ext",
                without=(
                    "Do it with volume automation: draw a short dip in the sub's level on each "
                    "kick hit. Tedious, but it is exactly what the sidechain would have done and "
                    "you get to shape the curve by hand."
                ),
            ),
            FixStep(
                action="Force the bottom to mono.",
                detail=(
                    "Below roughly 100 Hz, stereo information costs you level and translates "
                    "badly — a club system sums it anyway and a vinyl cut will refuse it. Mono "
                    "below 100-120 Hz makes the same sub read louder and tighter for no extra "
                    "energy."
                ),
                needs="mono_maker",
                without=(
                    "Fix it at the source instead: turn off chorus, unison detune, stereo "
                    "widening and stereo reverb on the bass patch. That is where a stereo "
                    "bottom end almost always comes from."
                ),
            ),
            FixStep(
                action="Check the result on an analyser rather than by ear.",
                detail=(
                    "Very few rooms and very few headphones tell the truth down here. Watch the "
                    "bottom of the spectrum while the track plays: what you are looking for is "
                    "the sub moving with the notes, not sitting as a constant floor under "
                    "everything."
                ),
                needs="meter_spectrum",
            ),
        ),
        how_to_verify=(
            "Re-analyse and the sub band should sit inside the window for your genre — hip-hop, "
            "trap and EDM are allowed far more here than rock, acoustic or jazz, and the "
            "tolerance is wider too, because this is where those genres actually differ. The "
            "more convincing test is loudness: with the same limiter settings the whole mix "
            "should now sit louder and the gain reduction meter should move less, because the "
            "limiter has stopped spending itself on energy nobody hears."
        ),
        learn_more=(
            "The equal-loudness contours are steepest at the bottom of the spectrum. At a "
            "moderate listening level a 30 Hz tone needs tens of decibels more sound pressure "
            "than a 1 kHz tone to be perceived as equally loud, and the gap widens as you listen "
            "quieter. Two consequences follow. First, sub costs enormously more amplitude — and "
            "therefore headroom, limiter work and speaker excursion — per unit of perceived "
            "loudness than anything else in the mix. Second, how much sub your track appears to "
            "have depends on how loudly it is played, which is why a mix that is perfect in the "
            "car is boomy on someone's desk. Judging the bottom on a meter is not a failure of "
            "ears; it is the only way to be consistent about a region whose perceived level "
            "moves with the volume knob."
        ),
        minutes=20,
    ),
)

_add(
    "frequency_balance.sub_thin",
    Explainer(
        headline="The track has no weight underneath it — it will feel small in a car and on a club system.",
        what_it_is=(
            "The sub band, below about 60 Hz, is the part of a mix you feel rather than hear. It "
            "is the fundamental of an 808 or sub synth, the bottom of a kick, and the lowest "
            "notes of a bass. Being under this band's target does not mean the track has no bass "
            "— you can have a full, present bass guitar and still measure thin here, because a "
            "bass guitar's fundamentals mostly live an octave higher. It means the octave below "
            "the bass is missing. The target comes from your genre's own curve, so a jazz or "
            "acoustic mix is expected to have very little; if this finding fired on one of those "
            "styles, the track is below even that."
        ),
        what_you_hear=(
            "On your laptop or on small monitors, nothing wrong at all — that is why this one "
            "survives to the master. Then the track plays in a car and sounds polite, or in a "
            "club and disappears next to everything around it. The kick reads as a click or a "
            "slap rather than a hit; the drop arrives and nothing lands underneath it; the whole "
            "record feels like a demo without anyone being able to say why."
        ),
        why_it_matters=(
            "In bass-led genres the sub is not support, it is the hook — the thing the listener "
            "came for. Streaming normalisation also works against you here: platforms turn "
            "everything down to a common loudness, so a track that made its impact by being "
            "loud rather than by being big arrives at the same level as everyone else's and now "
            "has nothing left. Weight is one of the very few things normalisation cannot take "
            "away from you."
        ),
        common_causes=(
            "Monitoring that does not reproduce the region. Most nearfields roll off around "
            "50-60 Hz and most laptops far higher, so the sub is simply not being judged.",
            "High-passing by reflex. An 80 or 100 Hz filter on every channel including the bass "
            "removes exactly the octave in question.",
            "A distorted, amp-simmed or heavily saturated bass. Those processes add harmonics "
            "upward and often thin the fundamental, so the bass gets more audible and less deep "
            "at the same time.",
            "The arrangement, not the mix: a bassline written or transposed high enough that no "
            "fundamental falls in the band. No EQ can boost a note that was never played.",
            "Mix-bus or multiband compression clamping the low band, so the sub is present in "
            "the quiet sections and squashed out of the loud ones.",
        ),
        how_to_fix=(
            FixStep(
                action="Establish whether it is missing or just inaudible to you.",
                detail=(
                    "Look at the bottom of a spectrum analyser while the track plays. If there "
                    "is energy moving with the bass notes, your monitoring is the problem and "
                    "you should trust the meter. If the region is flat and empty, the content is "
                    "genuinely not there and the rest of these steps apply."
                ),
                needs="meter_spectrum",
            ),
            FixStep(
                action="Check the notes before you check the EQ.",
                detail=(
                    "Find the lowest note the bass part plays and work out its fundamental. If "
                    "the part sits above the band, the fix is arrangement — drop the line an "
                    "octave, add a sub layer that plays only the root, or move the pattern down "
                    "— not a boost."
                ),
            ),
            FixStep(
                action="Move the high-pass on the bass and kick down, or take it off.",
                detail=(
                    "Those two are the sources that are supposed to have content down there. A "
                    "filter at 20-30 Hz protects against rumble and leaves the music alone; a "
                    "filter at 80 Hz on the bass is removing the thing this finding is about."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Add a sub layer rather than boosting one.",
                detail=(
                    "A sine following the bass root, or an octave-down layer gated to the same "
                    "rhythm, gives you clean controllable weight. Boosting a low shelf on a "
                    "source that has little content down there mostly amplifies rumble and "
                    "noise, because that is what is actually there."
                ),
            ),
            FixStep(
                action="Saturate the sub so it survives on small speakers.",
                detail=(
                    "Harmonic distortion on the sub creates content an octave and two octaves "
                    "up. Those harmonics are reproducible on a laptop and a phone, and the ear "
                    "reconstructs the missing fundamental from them — so the bass reads as deep "
                    "on systems that cannot produce a single cycle of it. Blend it in parallel "
                    "so the clean sub stays intact underneath."
                ),
                needs="saturation",
            ),
        ),
        how_to_verify=(
            "Re-analyse: the sub band should land inside your genre's window, which is a very "
            "different number for trap than for folk. Then confirm it physically — play it in a "
            "car or on a system with a real woofer, and the kick should feel like an impact "
            "rather than a tap. One more check: put a high-pass at 80 Hz across the master and "
            "bypass it repeatedly. You should clearly hear something leave. If nothing changes, "
            "there is still nothing down there."
        ),
        learn_more=(
            "The ear reconstructs pitch from harmonic spacing, not from the presence of the "
            "fundamental. Play the second, third and fourth harmonics of a 40 Hz note with no "
            "40 Hz at all and you still hear a 40 Hz note — the auditory system works out the "
            "common spacing and reports the pitch. This is called the missing fundamental, and "
            "it is the reason a phone speaker can convey a bassline it is physically incapable "
            "of reproducing. It is also the reason saturation is a better tool than EQ for a "
            "thin bottom end: EQ can only amplify what exists, while distortion generates the "
            "upper harmonics that let a small speaker imply the note. What it cannot do is make "
            "the track feel like anything on a system that does move air, which is why you fix "
            "the actual sub as well."
        ),
        minutes=25,
    ),
)

_add(
    "frequency_balance.low_bass_hot",
    Explainer(
        headline="The bottom of the mix is boomy — the kick and the bass are both fighting for the same octave.",
        what_it_is=(
            "Roughly 60 to 120 Hz is where the punch lives. It is the fundamental of a kick "
            "drum, the meat of a bass guitar's normal playing range, the bottom of a floor tom, "
            "and the lowest useful octave of a piano. Unlike the sub beneath it, you hear this "
            "region as much as you feel it, and it is reproduced by headphones, car systems and "
            "decent monitors, so it is the part of the low end most listeners actually get. Too "
            "much of it against your genre's curve is what people mean by boom."
        ),
        what_you_hear=(
            "Boom, and one-note bass: the bassline stops having a melody and becomes a series of "
            "similar-sounding thuds because the loudest note in the room is whichever one your "
            "speakers and room happen to favour. The kick sounds big on its own and swallows the "
            "bass in the arrangement. When the mix is loud, the whole track breathes on every "
            "kick, because that is what the limiter is following."
        ),
        why_it_matters=(
            "This region reproduces on nearly everything, so unlike excess sub it is audible to "
            "every listener rather than only the ones with a woofer. It is also where kick and "
            "bass have to share, and a shared, unmanaged 60-120 Hz is the most common reason a "
            "low end sounds powerful in solo and mushy in the mix. Worse, it is the region your "
            "room lies about most, so it is easy to spend an hour making it worse."
        ),
        common_causes=(
            "Kick and bass with the same fundamental. Both are correct alone; together they sum "
            "on some notes and cancel on others.",
            "A room null at the mixing position. If your room cancels 80 Hz where you sit, you "
            "will boost 80 Hz until it sounds right and put far too much into the file.",
            "Boosting the kick to hear it on a laptop. Small speakers cannot reproduce it, so no "
            "amount of boost makes it audible there — it only shows up on real systems.",
            "A bass amp sim or bass preset with a low shelf baked in, stacked with the shelf you "
            "then added.",
            "Long kick samples. A kick with a 300 ms tail overlaps the bass note that follows "
            "it, and the overlap is all in this band.",
        ),
        how_to_fix=(
            FixStep(
                action="Give the kick and the bass different homes inside the octave.",
                detail=(
                    "Pick one to own 60-80 Hz and the other to own 100-120 Hz, then cut each "
                    "where the other lives — a couple of dB is usually enough. It does not "
                    "matter which way round; it matters that you decide. Genre tends to answer "
                    "it for you: in hip-hop the 808 takes the bottom and the kick sits above, in "
                    "rock the kick takes it."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Duck the bass from the kick rather than carving both.",
                detail=(
                    "A fast sidechain of 2-4 dB, released inside the length of the kick, lets "
                    "the kick's fundamental through cleanly and gives the bass everything back "
                    "immediately after. It fixes the collision only at the moments the collision "
                    "happens, which static EQ cannot do."
                ),
                needs="comp_sidechain_ext",
                without=(
                    "Use a dynamic dip drawn as automation on the bass channel, timed to the "
                    "kick, or shorten the kick sample so the two overlap less in the first "
                    "place. Shortening the kick costs nothing and often solves it outright."
                ),
            ),
            FixStep(
                action="Check the region on headphones before you cut anything.",
                detail=(
                    "Headphones have no room modes. If the boom is present on headphones it is "
                    "in the mix; if it only exists on your speakers, it is your room, and "
                    "EQ-ing the file to fix a room will wreck it everywhere else."
                ),
            ),
            FixStep(
                action="Cut where it is heavy rather than boosting everything else.",
                detail=(
                    "A wide bell around the offending region, 2-3 dB, on the source that is "
                    "actually carrying it — not on the master. Cutting keeps your headroom; "
                    "boosting the other nine tracks to compete is how a mix ends up needing 6 dB "
                    "of limiting."
                ),
            ),
            FixStep(
                action="Hold the band with a multiband if it is only heavy in the loud sections.",
                detail=(
                    "One band across the low end, 1-2 dB of reduction on the choruses and "
                    "nothing on the verses, keeps the arrangement consistent without flattening "
                    "the kick's transient. Slow attack, or you will remove the punch you are "
                    "trying to protect."
                ),
                needs="comp_multiband",
                without=(
                    "Automate the low-end EQ or the bass fader down by a decibel in the dense "
                    "sections. The problem is almost always arrangement density rather than a "
                    "constant excess."
                ),
            ),
        ),
        how_to_verify=(
            "Play a section with a moving bassline and check that you can hum it. If every note "
            "sounds like the same weight, the region is still uncontrolled. Re-analyse: the low "
            "bass band should sit inside your genre's tolerance, which is a touch wider for "
            "hip-hop and trap than for anything else. Watch your limiter as well — it should now "
            "be reacting to the whole mix rather than lurching on every kick."
        ),
        learn_more=(
            "Small rooms have discrete resonances at low frequencies. Between two parallel walls "
            "you get a standing wave whose lowest mode sits at roughly 172 divided by the "
            "distance in metres — so a 3.4 m wall gives you about 50 Hz, with more modes above "
            "it. At a peak, a frequency can read 10 dB too loud; at a null, 10 dB too quiet, and "
            "the peaks and nulls are metres apart. This is why the low end is the region where "
            "monitoring failures cause the most damage: you are not making a mistake of taste, "
            "you are correcting for something that only exists in your chair. Nulls are the "
            "dangerous ones, because they make you add. Headphones, a spectrum analyser and a "
            "reference track are all immune to the room, and using all three is cheaper than "
            "treating it."
        ),
        minutes=25,
    ),
)

_add(
    "frequency_balance.low_bass_thin",
    Explainer(
        headline="The kick has no body — it clicks instead of hitting.",
        what_it_is=(
            "60 to 120 Hz is the punch band: the fundamental of a kick drum, the working range "
            "of a bass guitar, the weight of a floor tom. It is distinct from the sub below it, "
            "which you feel but barely hear. This is the part of the low end that actually "
            "reproduces on headphones, car systems and monitors, so it is where most listeners "
            "get their impression of whether a record hits. Being under target here while the "
            "sub is fine is a specific and very common shape — a mix with weight but no impact."
        ),
        what_you_hear=(
            "The kick is a click or a slap with nothing behind it. The bass is audible as pitch "
            "but not as force. The track has a floor of sub you can feel, and then a gap, and "
            "then the mids — so it sounds simultaneously heavy and weak, and it never quite "
            "sounds like it is being played in a room. Turning the whole low end up does not fix "
            "it, it just makes the sub louder."
        ),
        why_it_matters=(
            "Groove is carried here. The sense that a drummer hit something comes from the decay "
            "of the fundamental over the hundred milliseconds after the transient, and if that "
            "part is missing the listener gets the attack without the consequence. It also "
            "affects perceived loudness: this region reproduces on almost every playback system, "
            "so a mix that is thin here sounds quieter than its measured loudness on exactly the "
            "devices most people use."
        ),
        common_causes=(
            "High-passing the kick too high. Filters at 100 Hz get recommended for 'clarity' and "
            "they remove the body along with the rumble.",
            "Scooping 100 Hz to make room for the sub. It does make room; it also removes the "
            "part of the kick you can hear on headphones.",
            "A kick sample chosen on laptop speakers. Those cannot reproduce the band, so you "
            "pick a sample by its click and only find out later.",
            "A room peak in this region making you cut it. If your position exaggerates 90 Hz, "
            "you will pull it out of the file to make your room sound right.",
            "Heavy compression or limiting on the drum bus with a fast attack, which flattens the "
            "body more than the transient and leaves you with attack and no weight.",
        ),
        how_to_fix=(
            FixStep(
                action="Move the high-pass on the kick and bass down before adding anything.",
                detail=(
                    "Often the body is not missing, it is filtered. Sweep the filter downward "
                    "while the full mix plays and stop where the kick stops gaining weight — "
                    "usually far lower than the number people quote."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Add body with a wide bell on the kick, not a shelf on the master.",
                detail=(
                    "1.5 to 3 dB, Q around 1, somewhere between 70 and 100 Hz depending on the "
                    "sample. A master shelf raises the sub with it and gives you a bigger "
                    "version of the same imbalance."
                ),
            ),
            FixStep(
                action="Layer instead of boosting if the sample has nothing to boost.",
                detail=(
                    "A second kick layer, low-passed so it contributes only weight, tucked under "
                    "the original and aligned so their waveforms start in the same direction. "
                    "Check the alignment: two kicks out of polarity cancel exactly the region "
                    "you are trying to add."
                ),
            ),
            FixStep(
                action="Get the punch back from the compressor before you get it from an EQ.",
                detail=(
                    "A transient shaper can restore the sustain portion of the kick — the body — "
                    "without touching the balance of the mix, which an EQ boost cannot claim. "
                    "Push sustain rather than attack; the attack is not what is missing here."
                ),
                needs="transient_shaper",
                without=(
                    "Lengthen the attack time on your drum-bus compressor so the first "
                    "20-30 ms passes uncompressed, and slow the release so it is not clamping "
                    "down again during the kick's decay. Over-fast compression on drums removes "
                    "the body first."
                ),
            ),
            FixStep(
                action="Confirm on something with a real driver.",
                detail=(
                    "Headphones or a monitor with a woofer, not a laptop. This band is precisely "
                    "the one small speakers cannot show you, and it is the one you are currently "
                    "making decisions about."
                ),
            ),
        ),
        how_to_verify=(
            "Re-analyse and the low-bass band should come back inside the window for your genre. "
            "By ear, the kick should read as a hit with a short decay rather than a click, and "
            "you should be able to feel the rhythm of the bassline as well as follow its pitch. "
            "If the mix now sounds boomy on your speakers but correct on headphones, your room "
            "was why the band was thin in the first place — trust the headphones."
        ),
        learn_more=(
            "A kick drum is two events glued together. The transient — the beater against the "
            "head — is a broadband click lasting a few milliseconds with most of its audible "
            "energy in the upper mids, and it is what tells you the drum was struck. The body is "
            "the resonance of the shell and head, mostly between 60 and 120 Hz, decaying over "
            "perhaps 100-200 ms, and it is what tells you how hard. Change the ratio and you "
            "change what instrument it sounds like: all click and no body is a practice pad, all "
            "body and no click is a thud you cannot place in time. Compression alters that ratio "
            "before it alters anything else, because the body is the long part and the attack "
            "escapes ahead of the detector — which is why 'the kick lost its weight' is far more "
            "often a compressor problem than an EQ problem."
        ),
        minutes=20,
    ),
)

_add(
    "frequency_balance.upper_bass_hot",
    Explainer(
        headline="Warmth has turned into a blanket — the mix sounds thick and slow.",
        what_it_is=(
            "120 to 250 Hz is the warmth region, and it is the busiest, most over-subscribed "
            "band in a mix. Everything has something here: the fundamental of a bass playing its "
            "higher notes, the body of an acoustic guitar, the low end of an electric guitar, "
            "the chest of a male vocal, the shell of a rack tom, the left hand of a piano, the "
            "bottom of a pad. Nothing sounds wrong when you solo it. The band exists exactly "
            "where 'full and warm' turns into 'thick and slow', and the difference between those "
            "two is usually two or three decibels of total energy."
        ),
        what_you_hear=(
            "Woolly. The mix has weight but no definition — the notes of the bass smear "
            "together, the kick's front edge softens, and the singer sounds like they are "
            "speaking into their own chest. Instruments stop being separable: you can tell "
            "there is a guitar and a keyboard, but not where one ends. Turn the mix down and it "
            "seems to lose almost nothing, which is a good sign that the energy is bunched down "
            "here rather than spread across the spectrum."
        ),
        why_it_matters=(
            "This is the band directly beneath the region that carries intelligibility, and low "
            "frequencies mask upwards, so an excess here buries the vocal and the snare without "
            "showing up as anything you can point at. It is also heavy: it costs limiter "
            "headroom in the same way sub does, but unlike sub it is audible on every device, so "
            "the cost is paid twice. If this band and the region just above it are both hot, you "
            "will normally see the low-mid buildup finding instead — that one is the same energy "
            "described as a ratio."
        ),
        common_causes=(
            "Proximity effect. Every directional microphone lifts the low end as it gets closer, "
            "and a close-mic'd vocal or guitar cab picks up several decibels here that were "
            "never in the room.",
            "Nothing high-passed. Guitars, keys, pads and backing vocals all contribute here and "
            "none of them need to.",
            "Layering. Three synth layers built to sound big each carry their own body, and the "
            "stack has three times the body and the same top.",
            "Tape, tube and console emulations, which mostly add their harmonics low. Two or "
            "three in a chain compound.",
            "A small tracking room. Rooms this size resonate strongly through here and print it "
            "into every microphone.",
        ),
        how_to_fix=(
            FixStep(
                action="High-pass the sources that have no business down here.",
                detail=(
                    "Backing vocals, hats and percussion, most keys and pads, and every reverb "
                    "and delay return. Sweep the filter up until the source thins, then back off "
                    "a little. Judged in the mix rather than solo, most sources tolerate a much "
                    "higher filter than they seem to on their own."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Fix proximity effect with a filter rather than a shelf.",
                detail=(
                    "On a close-mic'd vocal, a high-pass around 100-120 Hz removes the "
                    "proximity lift without touching the chest tone that makes the voice sound "
                    "like a person. A broad low shelf takes both, which is why de-proximity-ing "
                    "with a shelf makes singers sound weedy."
                ),
            ),
            FixStep(
                action="Spread the cut across the biggest contributors rather than concentrating it.",
                detail=(
                    "Two decibels off three or four sources beats six off one. Separate sources "
                    "pile up in this region rather than adding neatly, so taking a small amount "
                    "from many removes roughly as much in total as a deep cut on one — and every "
                    "track still sounds like itself afterwards."
                ),
            ),
            FixStep(
                action="Use a dynamic band on whichever source is inconsistent.",
                detail=(
                    "A vocalist who moves relative to the mic is only heavy here on some lines. "
                    "A dynamic band set to act only when the region gets loud tracks that, "
                    "instead of thinning the lines that were already correct."
                ),
                needs="eq_dynamic",
                without=(
                    "Ride the clip gain line by line, or split the vocal into two tracks — close "
                    "lines and normal lines — and filter them differently. Clip-gain riding on a "
                    "moving singer is the single highest-value hour in most vocal mixes."
                ),
            ),
        ),
        how_to_verify=(
            "Play the fullest section and count instruments. You should be able to point at each "
            "one, and the bass should have distinguishable notes. Re-analyse: the upper-bass "
            "band should sit inside your genre's tolerance, which is tight here in every genre — "
            "this is not a region where much variation is allowed, because it is where "
            "everything overlaps. If the mix now sounds clear but small, you cut too much in one "
            "place instead of a little in several."
        ),
        learn_more=(
            "Proximity effect is worth understanding, because it explains a large share of "
            "unwanted energy in this band. Directional microphones work by sensing the "
            "difference in pressure between the front and the back of the diaphragm. Close to a "
            "source, the sound field is spherical and the pressure falls off steeply across that "
            "short distance, which produces a large difference at low frequencies where the "
            "wavelength is long compared to the mic. The result is a low-end lift that grows as "
            "the source gets closer and can reach 6-10 dB at a few centimetres. Omnidirectional "
            "microphones sense absolute pressure and do not have the effect at all. That is why "
            "a singer who leans in gets warmer, why radio voices sound the way they do, and why "
            "a high-pass filter — not a shelf — is the correct compensation: you are undoing a "
            "shelving lift that starts low and rises toward the bottom."
        ),
        minutes=25,
    ),
)

_add(
    "frequency_balance.upper_bass_thin",
    Explainer(
        headline="The mix sounds hollow — there is sub, there is top, and nothing joining them.",
        what_it_is=(
            "120 to 250 Hz is the region that connects the bass to the music. It is the "
            "fundamental of a bass playing anywhere above its lowest notes, the body of "
            "acoustic and electric guitars, the chest of a voice, the tone of a tom. Whatever "
            "makes an instrument sound like a physical object with a size lives here. Because "
            "it is also the region that most often gets over-cut in the name of clarity, being "
            "underweight here is usually a self-inflicted wound rather than something the "
            "recording lacked."
        ),
        what_you_hear=(
            "Hollow, brittle, disconnected. The sub is there and the top is there but they do "
            "not sound like the same record. Voices sound small and slightly false, as though "
            "the singer is smaller than they are. Acoustic guitars turn tinny. Toms lose their "
            "pitch. The classic experience is a mix that sounds impressively clean for ten "
            "seconds and then feels weak the moment you play a commercial track after it."
        ),
        why_it_matters=(
            "Everything in the mix loses its sense of physical size at once, and no single track "
            "sounds wrong, so it is very hard to diagnose from inside the session. It also "
            "pushes you into a second mistake: with the body gone, the mix feels quiet, so you "
            "add limiting to compensate, and now it is thin and squashed. Most 'small-sounding' "
            "masters are this band, not loudness."
        ),
        common_causes=(
            "High-passing everything at 200 Hz because someone said to. The advice is a "
            "shorthand for 'filter what does not need low end', not a number to apply to every "
            "channel.",
            "Over-correcting a mud complaint. Mud lives a little higher, and a cut wide enough "
            "to fix it usually takes this band down with it.",
            "Scoop presets — metal guitar tones, some synth patches — used on sources that are "
            "not being backed by anything else filling the region.",
            "Mixing on headphones or speakers with a bump here, so you cut what your monitoring "
            "added.",
            "Mid/side widening. Pushing the sides up and the centre down thins exactly the "
            "instruments that carry this band, because they are mostly centred.",
        ),
        how_to_fix=(
            FixStep(
                action="Audit your high-pass filters before you add anything back.",
                detail=(
                    "Go through every channel and note where each filter sits. The fix is "
                    "usually five or six filters coming down by 40-60 Hz each, not one boost on "
                    "the master. Filters stack: eight sources each filtered at 200 Hz remove far "
                    "more from the region than the number on any single one suggests."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Put the weight back on the sources that are meant to carry it.",
                detail=(
                    "The bass and the main rhythm instrument. A wide bell of 1.5-2 dB around "
                    "150-200 Hz on those two does more, and sounds more natural, than the same "
                    "boost on the mix bus, because it restores the balance inside the "
                    "arrangement rather than tilting the whole record."
                ),
            ),
            FixStep(
                action="Check whether this happened while you were fixing mud.",
                detail=(
                    "If you cut this band to clear a covered mix, the cut was too wide. Narrow "
                    "it and move it up: the muddy region is centred higher, and you can usually "
                    "get the same clarity with a tighter, shallower move that leaves the body "
                    "alone."
                ),
            ),
            FixStep(
                action="Check the centre against the sides.",
                detail=(
                    "If the sides are much fuller than the middle through this band, a widener "
                    "is the cause and no amount of boosting will fix the hollowness — narrow "
                    "the image below about 300 Hz first, then re-judge the level."
                ),
                needs="eq_mid_side",
                without=(
                    "Sum the mix to mono and listen. If it collapses to something noticeably "
                    "thinner and smaller rather than just narrower, side-channel processing is "
                    "eating the centre and that is the thing to fix first."
                ),
            ),
        ),
        how_to_verify=(
            "A/B against a released track in your genre at matched loudness and listen "
            "specifically to whether voices and guitars sound like the same physical size on "
            "both. Re-analyse: the band should land inside the tolerance for your genre — metal "
            "and some EDM curves genuinely scoop here, so the target is not the same everywhere, "
            "and if this fired on one of those you are below even the scooped expectation. Watch "
            "for the opposite failure: if the mix gets warm but covered, you have gone past the "
            "target and into mud."
        ),
        learn_more=(
            "High-pass filters stack in a way that surprises people. A filter does not stop dead "
            "at its corner frequency; it slopes, typically 12 or 24 dB per octave, and it is "
            "already attenuating an octave above the corner. Put the same filter on eight "
            "channels and each one takes its own share out of the region above the corner, so "
            "the sum loses far more than any single reading suggests. They also each add phase "
            "shift near the corner, and eight overlapping phase rotations in the same region "
            "smear the timing of everything that passes through them, which reads as a loss of "
            "weight over and above the level change. The practical rule: filter aggressively on "
            "the tracks that contribute nothing down there, and leave the ones that carry the "
            "body of the arrangement almost alone."
        ),
        minutes=20,
    ),
)

_add(
    "frequency_balance.low_mid_hot",
    Explainer(
        headline="The mix sounds boxy, like it is playing inside a cardboard box.",
        what_it_is=(
            "250 to 500 Hz is the box. Physically it is where small enclosures resonate — a "
            "drum shell, a guitar body, an untreated bedroom, a vocal booth made of duvets — so "
            "when several sources are all carrying energy here they start to sound like they "
            "were recorded inside the same container. Musically it holds the body of a snare, "
            "the lower midrange of guitars and keys, the vowel weight of a voice, and the "
            "fundamentals of a lot of the notes in a typical arrangement. Some of it is "
            "essential; the mix goes lifeless without it. What this finding says is that there "
            "is more of it than your genre's curve expects."
        ),
        what_you_hear=(
            "Boxy is the word everyone uses and it is exactly right: the mix sounds like it is "
            "coming out of a container rather than out of the air. The snare sounds like someone "
            "hitting a box; the vocal sounds like the singer has hands cupped around the mic; "
            "guitars sound cheap in a way that is hard to name. It is more obvious on speakers "
            "than headphones, and more obvious quiet than loud."
        ),
        why_it_matters=(
            "Boxiness reads as amateur faster than almost anything else, because it is the "
            "signature of a small untreated space and listeners recognise it without being able "
            "to name it. It also sits directly underneath the region where vocal intelligibility "
            "lives, so it costs clarity as well as tone. If the region below this one is heavy "
            "too, you will normally see the low-mid buildup finding instead — that is the same "
            "energy assessed as a ratio, and it is the more serious version of the problem."
        ),
        common_causes=(
            "The recording room. Untreated rooms have their strongest audible resonance right "
            "here, and it is on every track that was recorded with a microphone.",
            "Undamped drums. A snare or tom with a ringing shell puts a narrow, pitched peak "
            "into this band on every hit.",
            "Cheap or poorly placed close mics, particularly on cabinets, where a few "
            "centimetres of position change alters this band more than any EQ move you can make "
            "afterwards.",
            "Too many midrange-heavy layers doing the same job — three rhythm parts where the "
            "arrangement needs one.",
            "Heavy saturation on the mix bus, which adds harmonic content densest in the low "
            "mids.",
        ),
        how_to_fix=(
            FixStep(
                action="Find out whether it is broad or a single ringing note.",
                detail=(
                    "Sweep a narrow boosted bell through 250-500 Hz on the full mix. A single "
                    "spot that rings much louder than the rest is a resonance — a shell, a "
                    "cabinet, or the room — and needs a narrow fix on one source. A whole "
                    "region that is evenly heavy needs a wide, shallow move spread across "
                    "several sources."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="If it is a resonance, duck it dynamically instead of notching it flat.",
                detail=(
                    "A room or shell resonance only rings when the source is playing, so a "
                    "tracking suppressor removes it during the ring and leaves the source's "
                    "tone alone the rest of the time. A permanent notch removes body the source "
                    "needed in the passages where the resonance was not excited."
                ),
                needs="resonance_suppressor",
                without=(
                    "Notch at the frequency you found — narrow bell, 3 to 4 dB, on the source. "
                    "Then check the sparse sections: if the source now sounds hollow there, "
                    "automate the notch out for those bars."
                ),
            ),
            FixStep(
                action="Fix drum ring at the drum, not at the desk.",
                detail=(
                    "A strip of tape or a moongel on the head takes out a shell resonance in "
                    "five seconds and costs nothing in tone. An EQ notch that does the same job "
                    "removes the resonance from every other sound bleeding into that mic too."
                ),
            ),
            FixStep(
                action="Thin several sources a little instead of one a lot.",
                detail=(
                    "A wide bell around 350-450 Hz taking 1.5-2 dB from the two or three "
                    "sources contributing most is nearly always better than a deep cut on the "
                    "mix bus. A mix-bus cut here takes the body out of the snare and the vocal "
                    "at the same time as it takes out the box."
                ),
            ),
        ),
        how_to_verify=(
            "Listen to the snare and the vocal in the full mix. The snare should sound like a "
            "drum with a head rather than a struck box, and the vocal should sound like it is in "
            "the room with you rather than in front of a wall. Re-analyse: the low-mid band "
            "should come back inside your genre's tolerance — noting that lo-fi deliberately "
            "allows a great deal here, because in that style the box is the aesthetic. If the "
            "mix now sounds clean but weightless, you cut too wide."
        ),
        learn_more=(
            "Boxiness is a resonance problem more often than a level problem, and resonances "
            "have a Q — a measure of how narrow and how long-ringing they are. An enclosure "
            "resonates at a frequency set by its dimensions and it keeps ringing after the "
            "excitation stops, which is why boxy sources sound smeared in time as well as "
            "wrong in tone. It also explains why the ear picks it out of a dense mix so easily: "
            "several unrelated sources all carrying the same narrow resonance is a strong cue "
            "that they share a physical space, and your hearing is extremely good at spotting "
            "shared room signatures — it is how you localise anything indoors. The corollary is "
            "practical: matching the width of your cut to the width of the resonance matters "
            "more than the depth. A narrow ring needs a narrow notch, and a wide shallow "
            "heaviness needs a wide shallow cut, and using the wrong shape for either one causes "
            "a second problem while solving the first."
        ),
        minutes=20,
    ),
)

_add(
    "frequency_balance.low_mid_thin",
    Explainer(
        headline="The mix is thin — clean, but with no substance behind it.",
        what_it_is=(
            "250 to 500 Hz is the lower body of the arrangement: the weight of a snare, the "
            "meat of a rhythm guitar, the lower vowels of a voice, the fundamental of a lot of "
            "melody notes. Producers cut this region constantly, and for good reason — it is "
            "the first place a mix gets crowded. Under-cutting it makes a mix boxy; over-cutting "
            "it makes a mix that sounds superb for ten seconds and gutless for the other three "
            "minutes. This finding says you are on the second side of that line for your genre."
        ),
        what_you_hear=(
            "Thin, brittle, hi-fi in a bad way. The snare becomes a crack with no drum behind "
            "it. Guitars sound like a small radio. The voice loses the part that makes it sound "
            "like a body producing a sound. The characteristic tell is comparison: on its own "
            "the mix sounds clean and detailed, and the moment you play a commercial track it "
            "sounds like a demo, because the reference has substance underneath its clarity."
        ),
        why_it_matters=(
            "Thinness costs perceived loudness. Energy in the lower mids is a large part of what "
            "makes a record feel big at a given meter reading, so a scooped mix has to be pushed "
            "harder to feel level with anything else — and it takes the limiting worse than a "
            "full one would. It is also the failure mode that is hardest to notice in your own "
            "session, because every individual track sounds cleaner after each cut and you never "
            "hear the sum going missing."
        ),
        common_causes=(
            "A reflex 300 Hz cut on every channel. It is the most-repeated EQ tip in existence "
            "and applied to twenty channels it removes the middle of the record.",
            "Over-correcting boxiness or mud with a wide cut on the mix bus.",
            "Smiley-face EQ: bass and treble up, mids down. It sounds impressive on monitors at "
            "volume and disappears everywhere else.",
            "Genre presets built for a scooped style, used on a track that does not have a wall "
            "of guitars filling the region back in.",
            "Multiband compression with an over-aggressive low-mid band, which only removes the "
            "region in the loud sections — so the chorus gets thinner than the verse, which is "
            "backwards.",
        ),
        how_to_fix=(
            FixStep(
                action="Count your cuts before you add anything.",
                detail=(
                    "Look at how many channels have something pulled between 250 and 500 Hz. If "
                    "the answer is most of them, the fix is undoing three or four of those, not "
                    "boosting the master. Each one was a reasonable decision; the sum was not a "
                    "decision at all."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Put the body back on the sources that carry the song.",
                detail=(
                    "Usually the snare and the main rhythm instrument. A wide bell of 1.5-2 dB "
                    "around 300-400 Hz on those, checked in the mix rather than solo. Restoring "
                    "it inside the arrangement keeps the balance between parts; a master-bus "
                    "boost raises the parts you deliberately thinned as well."
                ),
            ),
            FixStep(
                action="A/B against a reference at matched loudness, not at matched fader.",
                detail=(
                    "This band is where the difference between your mix and a commercial one is "
                    "most audible and least visible, and level differences will mislead you "
                    "completely. Match until they feel equally loud, then switch quickly and "
                    "listen for substance rather than clarity."
                ),
                needs="reference_matching",
                without=(
                    "Do it manually: reference track on a muted channel, fader pulled until the "
                    "two feel the same volume, switch between them every few seconds. Listen "
                    "specifically to the snare and the rhythm part, which is where the "
                    "difference shows first."
                ),
            ),
            FixStep(
                action="Check the multiband and bus compression.",
                detail=(
                    "If a low-mid band is doing several decibels of reduction in the choruses, "
                    "the mix is getting thinner exactly when it should be getting bigger. "
                    "Reduce the ratio or raise the threshold until that band is barely moving, "
                    "then re-judge."
                ),
                needs="comp_multiband",
                without=(
                    "Bypass your bus compressor and listen to whether the body comes back. If it "
                    "does, slow the attack and reduce the ratio — fast, heavy bus compression "
                    "flattens the mid weight before it flattens anything else."
                ),
            ),
        ),
        how_to_verify=(
            "Play the loudest section against a reference at matched loudness. Yours should not "
            "sound noticeably smaller. Re-analyse: the band should sit inside your genre's "
            "window, remembering that metal genuinely scoops here and lo-fi genuinely piles it "
            "on, so the target you are being held to is not a universal one. The failure "
            "direction to watch for is boxiness — if the mix stops being thin and starts "
            "sounding like a container, you have gone a decibel or two too far."
        ),
        learn_more=(
            "Equal-loudness contours explain why a scooped mix is a trap. Your ear's sensitivity "
            "is not flat, and it is much less flat at low listening levels than at high ones: "
            "quiet listening loses the extremes of the spectrum first and leaves the midrange "
            "relatively intact. So a mix with its mids scooped and its extremes boosted sounds "
            "spectacular at the volume you mixed it at — where your ears are compensating least "
            "— and hollow at the volume most people play music. The midrange is the part that "
            "survives every playback situation: quiet listening, phone speakers, car noise, a "
            "bar. Anything you put there will be heard, and anything you take from there is "
            "invisible on a good system and fatal on a bad one. That is the reason experienced "
            "engineers check quietly and on small speakers, which are two ways of asking the "
            "same question about the same band."
        ),
        minutes=20,
    ),
)

_add(
    "frequency_balance.mid_hot",
    Explainer(
        headline="The mix sounds honky and nasal, like it is coming through a telephone.",
        what_it_is=(
            "500 Hz to 1 kHz is the middle of everything. It is the core tone of guitars, the "
            "fundamental of a snare, the main vowel region of a voice, the body of horns and "
            "strings, and the part of the spectrum small speakers reproduce best. It is also the "
            "region that identifies a sound: take it away and instruments become hard to name. "
            "Too much of it against your genre's curve does something specific rather than "
            "generic — it makes the mix nasal, because nasality in a human voice is literally an "
            "excess of energy in this band."
        ),
        what_you_hear=(
            "Honk. A pinched, nasal, hand-over-the-mouth quality; a mix that sounds like a small "
            "transistor radio or a phone speaker even when played through good ones. Guitars "
            "sound cheap, the snare sounds like a knock rather than a hit, and vocals sound like "
            "the singer has a cold. It is more obvious at low volume, and it is the one tonal "
            "fault that laptop speakers exaggerate rather than hide."
        ),
        why_it_matters=(
            "Your ear treats this region as the identity of a sound, so an excess makes "
            "everything sound like it was recorded through the same cheap device — the mix loses "
            "variety between instruments. It also happens to be the region where most consumer "
            "playback is already peaking, so what reads as slightly forward on your monitors is "
            "considerably worse on the devices people use. And it is fatiguing in a different "
            "way from harshness: not painful, just relentless."
        ),
        common_causes=(
            "Amp sims and cabinet impulse responses, most of which have a pronounced mid hump. "
            "Two guitars with the same IR double it.",
            "Boosting the midrange to make the mix translate on a laptop. It works on the "
            "laptop and it is wrong everywhere else.",
            "A dynamic mic close to a source, which tends to be forward here by design.",
            "Too many sources occupying the same midrange with no arrangement decision about "
            "which one leads.",
            "Heavy saturation or distortion on the mix bus, which fills the middle before it "
            "fills anywhere else.",
        ),
        how_to_fix=(
            FixStep(
                action="Sweep for the honk before you cut anything.",
                detail=(
                    "A narrow boosted bell moved slowly through 500 Hz to 1 kHz on the full mix "
                    "will find a specific spot that sounds like a mouth shape. That is your "
                    "centre. Broad cuts placed by guesswork in this region take the identity out "
                    "of the instruments along with the honk."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="If it is a guitar cabinet, change the capture rather than the EQ.",
                detail=(
                    "Swapping the impulse response, or moving the mic a couple of centimetres "
                    "toward the edge of the cone, changes this band more than any EQ move and "
                    "does not leave a hole where the tone used to be. Same for a real cab in a "
                    "room."
                ),
            ),
            FixStep(
                action="Decide who owns the middle, and pull the others down rather than everyone equally.",
                detail=(
                    "In most arrangements this band belongs to the lead — the voice or the main "
                    "instrument. Take 1.5-2 dB out of the supporting parts in the same region "
                    "and the lead comes forward without being turned up, which is a cleaner "
                    "result than boosting it."
                ),
            ),
            FixStep(
                action="Hold the region dynamically if it only honks when the arrangement is full.",
                detail=(
                    "A dynamic band across the middle with a modest range, threshold set so it "
                    "does nothing in the verse, keeps the sparse sections' tone intact and "
                    "controls the dense ones — where the honk is caused by pile-up rather than "
                    "by any single source."
                ),
                needs="eq_dynamic",
                without=(
                    "Use a static cut of about 1 dB with a wide Q, and automate an extra dB in "
                    "the fullest sections only. Wide and shallow, because this region is where "
                    "narrow cuts are most audible as damage."
                ),
            ),
        ),
        how_to_verify=(
            "Play the mix on a laptop or phone speaker, which reproduce almost nothing but this "
            "band. It should sound balanced rather than shouty. Then check on your monitors that "
            "the instruments still sound like themselves — if the guitars have become vague or "
            "the vocal has lost character, the cut is too deep or too wide. Re-analyse: the mid "
            "band should be inside your genre's tolerance, which is one of the tightest in the "
            "set, because this is where the ear is least forgiving of error in either direction."
        ),
        learn_more=(
            "The telephone band is 300 Hz to 3.4 kHz, and that is not an arbitrary choice — it "
            "is the range engineers found could carry speech intelligibly with the least "
            "bandwidth. 500 Hz to 1 kHz sits in the middle of it and carries the first formant "
            "of most vowels, which is the resonance of the throat and mouth cavity that tells "
            "you which vowel is being said. Because your auditory system is specialised for "
            "extracting speech, it treats energy in that band as informational rather than "
            "musical, and it is very sensitive to the shape of it. That is why an excess here "
            "reads as a specific mouth-like quality — nasal, cupped, honky — rather than as a "
            "vague tonal problem, and it is why the same excess on a guitar makes the guitar "
            "sound like it is speaking. It is also why this is a poor place to make broad "
            "corrections: you are editing the region your hearing scrutinises hardest."
        ),
        minutes=20,
    ),
)

_add(
    "frequency_balance.mid_thin",
    Explainer(
        headline="The mix is scooped — it will sound impressive on your speakers and vanish on everyone else's.",
        what_it_is=(
            "500 Hz to 1 kHz is the region a phone, a laptop, a car dash speaker, a TV and a bar "
            "PA all reproduce best, and in many cases the only region they reproduce properly at "
            "all. It is also the identity band: the core of guitars, the fundamental of a snare, "
            "the vowel body of a voice. Being under target here is the classic scoop, and it is "
            "the most seductive mistake in mixing because removing it makes everything sound "
            "cleaner and more separated on a good system — right up until the track leaves your "
            "room."
        ),
        what_you_hear=(
            "On your monitors: wide, clean, expensive. On a phone, a laptop or a car radio: "
            "hollow, distant and quiet, as if someone opened a window between you and the band. "
            "The vocal loses its body and you compensate by pushing the fader, which makes it "
            "loud and still weak. A useful test is playing the track from across the room — "
            "distance strips everything except this band, and a scooped mix nearly disappears."
        ),
        why_it_matters=(
            "Most listening happens on devices that cannot reproduce anything else. If you have "
            "put the mix's substance below 200 Hz and above 3 kHz, most of your audience will "
            "receive a track made of the parts you left out. Perceived loudness suffers too: "
            "loudness metering is weighted toward the midrange because hearing is, so a scooped "
            "mix reads quieter than it measures and gets pushed harder into the limiter to "
            "compensate."
        ),
        common_causes=(
            "Smiley-face EQ, whether applied deliberately or arrived at one channel at a time.",
            "Scooping guitars out of habit. It works when a wall of them is filling the region "
            "collectively and it fails when there are two.",
            "Stereo widening. Most wideners raise the sides and reduce the middle, and the "
            "instruments that carry this band are almost all centred.",
            "Chasing separation. Every source cut in the mids sounds more distinct on its own, "
            "and the mix gets emptier every time.",
            "Mixing loud. At high levels your ears flatten out and the mids feel more prominent "
            "than they are, so you take them out.",
        ),
        how_to_fix=(
            FixStep(
                action="Test it on the worst speaker you own before you change anything.",
                detail=(
                    "A phone speaker only plays this band. If the mix is thin and distant there "
                    "but full on your monitors, the diagnosis is confirmed and you know exactly "
                    "how much you need to put back — enough that the phone sounds like music."
                ),
            ),
            FixStep(
                action="Undo the scoops at the sources rather than boosting the bus.",
                detail=(
                    "Find the three or four channels with the deepest mid cuts and halve them. "
                    "A mix-bus boost here raises the whole middle including the parts you meant "
                    "to keep out of the way, and the separation you gained by scooping "
                    "disappears all at once instead of selectively."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Check whether the middle went missing rather than being cut.",
                detail=(
                    "If the sides are much fuller than the centre through this band, a widener "
                    "or a mid/side move is the cause. Narrowing the image back toward the middle "
                    "restores the band without any EQ at all."
                ),
                needs="eq_mid_side",
                without=(
                    "Sum to mono and compare. If mono sounds dramatically weaker rather than "
                    "just narrower, side-heavy processing is thinning the centre and that is "
                    "what to fix first."
                ),
            ),
            FixStep(
                action="Mix this band at low volume.",
                detail=(
                    "Turn the monitors down to conversation level and set the balance of the "
                    "lead, the snare and the rhythm part there. At low levels your hearing is at "
                    "its most midrange-focused, which is exactly the perspective this band needs "
                    "and exactly the one loud monitoring denies you."
                ),
            ),
        ),
        how_to_verify=(
            "Play it on a phone speaker and on your monitors within a minute of each other. The "
            "phone version should sound like a complete, if small, record — not like a distant "
            "one. Re-analyse: the mid band should sit inside your genre's tolerance, which is "
            "the tightest in the set because this is where errors are least forgivable. Then "
            "check for the opposite failure: if the mix has become honky or the instruments have "
            "started to blur together, you have overshot."
        ),
        learn_more=(
            "There is a reason loudness meters weight this region. Broadcast loudness "
            "measurement applies a filter that approximates the frequency response of human "
            "hearing before it integrates anything, which means midrange energy contributes far "
            "more to a loudness reading than the same energy at the extremes. Your perception "
            "works the same way, so 'how loud does this feel' and 'how much midrange does this "
            "have' are much closer to the same question than most people expect. That produces a "
            "counterintuitive but reliable result: adding midrange usually makes a mix feel "
            "louder without touching a limiter, and scooping it forces you to use one. If you "
            "are fighting for loudness and losing, an under-served midrange is the first place "
            "to look, well before you reach for another dB of limiting."
        ),
        minutes=20,
    ),
)

_add(
    "frequency_balance.upper_mid_hot",
    Explainer(
        headline="The mix is shouting at you — it sounds aggressive in a way that gets tiring fast.",
        what_it_is=(
            "1 to 2 kHz is the shout band. It carries the pick attack of a guitar, the crack of "
            "a snare, the lower half of vocal consonants, and the part of a voice that raises "
            "when someone speaks over noise. It is the region a megaphone, a telephone and a "
            "cheap PA all emphasise, and it is where a mix reads as forward, urgent and "
            "aggressive. It is next door to the harshness region but a different problem: 2-5 "
            "kHz is edge and glare, this is shout and honk — you hear it as effort rather than "
            "as brightness."
        ),
        what_you_hear=(
            "The mix sounds like it is being pushed at you, or like everything is being played a "
            "little too hard. Voices sound strained even when the performance was relaxed. "
            "Guitars sound thin and aggressive rather than full. Snares crack without body. At "
            "quiet volume it seems fine — the giveaway is that it gets worse faster than the "
            "rest of the mix as you turn it up."
        ),
        why_it_matters=(
            "The ear's sensitivity is already climbing steeply through this band toward its peak "
            "just above it, so an excess costs more perceived loudness than it looks like on a "
            "spectrum. It is also the region that most cheap playback hardware emphasises, so "
            "the listener with the worst speakers gets the worst version. And it consumes "
            "loudness: a mix that shouts feels loud at a lower measured level, but it also makes "
            "people turn it down, which loses you exactly what you were buying."
        ),
        common_causes=(
            "Several sources all boosted here to cut through. Vocal, guitars and snare all "
            "reaching for the same shelf of intelligibility.",
            "Distortion on midrange sources, which produces harmonics that pile into this band "
            "from below.",
            "EQ decisions made to fix a mix on small speakers, which naturally emphasise this "
            "region — you end up adding what they already have.",
            "Over-compressed bus processing, which raises the mid content between transients "
            "because it is the most continuous part of the signal.",
            "Amp sims and presence controls, which are usually centred lower than the word "
            "'presence' suggests.",
        ),
        how_to_fix=(
            FixStep(
                action="Decide which one source is allowed to be forward here.",
                detail=(
                    "Usually the lead vocal or the lead instrument. Take 1.5-2 dB out of the "
                    "supporting parts in this band and the lead cuts through with less total "
                    "energy in the mix — the opposite of what boosting the lead achieves."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Move the competing boosts apart instead of removing them.",
                detail=(
                    "Guitars generally read as clear at 1.5-2 kHz, snares at 3-5 kHz, hats "
                    "higher still, vocals at 2-4 kHz. Staggering them gets you the same "
                    "definition from each source with a fraction of the overlap, and overlap is "
                    "what makes this band shout."
                ),
            ),
            FixStep(
                action="Check whether distortion is putting it there.",
                detail=(
                    "Bypass saturators and amp sims one at a time. Harmonic distortion from "
                    "low-mid sources lands squarely in this band, and if the shout goes away "
                    "when a drive stage does, EQ is the wrong tool — reduce the drive instead."
                ),
            ),
            FixStep(
                action="Control it dynamically if it only shows up in the loud sections.",
                detail=(
                    "A dynamic band around 1.5 kHz doing 1-2 dB in the choruses and nothing in "
                    "the verses. This band gets worse with density, so a fixed cut set for the "
                    "chorus will make the verses sound weak and distant."
                ),
                needs="eq_dynamic",
                without=(
                    "Automate a wide, shallow EQ cut into the busiest sections only, or ride the "
                    "supporting parts down a decibel there. Either is better than a permanent "
                    "cut sized for the worst moment."
                ),
            ),
        ),
        how_to_verify=(
            "Play the loudest section at a volume slightly above comfortable. It should sound "
            "energetic rather than strained, and a sung line should still sound like a person "
            "singing rather than a person shouting. Re-analyse: the upper-mid band should sit "
            "inside your genre's tolerance, which is one of the tightest, and check the "
            "harshness reading at the same time — the two problems often travel together and "
            "fixing this one partially fixes that one."
        ),
        learn_more=(
            "Speech has a natural mechanism called the Lombard effect: in noise, people do not "
            "just get louder, they shift energy upward into the 1-3 kHz region, because that is "
            "where speech cuts through noise best. Your hearing is tuned to detect exactly that "
            "shift — it is how you know someone is straining without seeing them. When a mix has "
            "excess energy in the same band, the same perceptual machinery fires, and the mix "
            "sounds like it is working hard even though nothing in the performance was. That is "
            "why 'shouty' is such a consistent description of this band across people who have "
            "never discussed it, and why the fix is almost always to reduce competition rather "
            "than to reduce one source: the shout comes from several sources all reaching for "
            "the same shortcut to intelligibility at once."
        ),
        minutes=20,
    ),
)

_add(
    "frequency_balance.upper_mid_thin",
    Explainer(
        headline="The mix sounds soft and distant — you can hear it but you cannot quite follow it.",
        what_it_is=(
            "1 to 2 kHz is where definition begins. It holds the pick and the fret noise of a "
            "guitar, the stick on a snare, the lower half of consonants, and the attack of "
            "almost every instrument that has one. It is not brightness — brightness is an "
            "octave or two higher — it is articulation: the difference between hearing that "
            "something was played and hearing what was played. Being under target here produces "
            "a mix that is perfectly pleasant and slightly unintelligible."
        ),
        what_you_hear=(
            "Soft, polite, far away. Lyrics get harder to follow, particularly at low volume. "
            "Guitars lose their pick and turn into a wash. Drums lose the sense of being struck. "
            "The mix has bass and it has air, and the middle of it feels like it is behind a "
            "curtain — which is why this often coexists with the temptation to keep raising the "
            "vocal fader without ever getting the words back."
        ),
        why_it_matters=(
            "Intelligibility lives across 1-4 kHz, and this band is the low half of it. A "
            "listener who cannot follow a lyric without concentrating does not concentrate; they "
            "skip. The band also carries a lot of what makes a mix feel like it is happening "
            "close to you rather than at a distance, so an under-served upper midrange makes a "
            "record sound smaller and less present, which no amount of loudness will fix."
        ),
        common_causes=(
            "Cutting to remove honk with too wide a bell, so the definition went with the "
            "nasality.",
            "Dark processing on the whole mix: tape emulation, some analogue channel strips, "
            "lo-fi chains, all of which soften here.",
            "Very dark source material — a ribbon mic, an off-axis dynamic, a muted guitar tone "
            "— that was never bright enough to begin with.",
            "Over-compression with a slow attack on the mix bus, which lets transients through "
            "but then pulls their tails down, dulling the attack in perception.",
            "A big low end making the middle look and sound relatively absent, so the band is "
            "not thin so much as buried.",
        ),
        how_to_fix=(
            FixStep(
                action="Check whether it is missing or masked.",
                detail=(
                    "Pull the low end down by 2 dB temporarily and listen to whether the "
                    "definition comes back. If it does, this band is not thin — it is being "
                    "covered from below, and the fix is downstairs rather than here."
                ),
            ),
            FixStep(
                action="Add it on the sources that carry the song, narrowly.",
                detail=(
                    "1.5-2 dB with a moderate Q around 1.5-2 kHz on the vocal, the snare and "
                    "the main rhythm instrument. Narrow, because a wide boost here shades into "
                    "the shout band above and swaps one problem for another."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Get the attack back rather than boosting the frequency.",
                detail=(
                    "If the definition was flattened by compression, a transient shaper "
                    "restores it without changing the tonal balance at all — more honest than "
                    "an EQ boost, because the problem was envelope rather than spectrum."
                ),
                needs="transient_shaper",
                without=(
                    "Slow the attack on the offending compressor so the first few milliseconds "
                    "pass untouched, and speed the release so it is not still holding the tail "
                    "down. If that is not enough, back off the ratio — heavy compression removes "
                    "articulation before it removes anything else."
                ),
            ),
            FixStep(
                action="Use saturation instead of EQ when the source has nothing to boost.",
                detail=(
                    "A dark source with no content in the band gains nothing from a boost except "
                    "noise. Light saturation generates new harmonics from the fundamentals that "
                    "are there, which produces definition that sits in the mix rather than "
                    "sitting on top of it."
                ),
                needs="saturation",
            ),
        ),
        how_to_verify=(
            "Play the track at conversation level and see whether you can follow the lyric "
            "without effort. Then check on a phone: definition should survive the transition, "
            "because this band is one of the few a phone reproduces. Re-analyse for the band "
            "inside your genre's tolerance — and watch the harshness index while you do it, "
            "because the failure mode of fixing this is pushing into the region above and "
            "trading distance for glare."
        ),
        learn_more=(
            "Intelligibility is not spread evenly across the spectrum. Measures like the "
            "articulation index weight bands by how much each contributes to understanding "
            "speech, and the weighting peaks between roughly 1 and 4 kHz — consonants carry the "
            "information that distinguishes words, and consonants live up there, while the "
            "vowels that carry most of the energy live much lower. That mismatch is the whole "
            "reason a vocal can be loud and unintelligible at the same time: you have plenty of "
            "the energy and none of the information. It also explains why turning a vocal up is "
            "such a poor fix — it raises the vowels along with everything else and increases the "
            "masking the consonants have to fight through. The efficient move is always to add a "
            "small amount where the information is, and leave the level alone."
        ),
        minutes=20,
    ),
)

_add(
    "frequency_balance.presence_hot",
    Explainer(
        headline="Everything is pushed to the front — the mix is in your face and it will wear listeners out.",
        what_it_is=(
            "2 to 5 kHz is the presence band: consonants, the beater click of a kick, the stick "
            "on a snare, the bite of a distorted guitar, the attack of everything. It is also "
            "where your hearing is at its most sensitive, so it is the most powerful two octaves "
            "in a mix and the easiest to overspend in. This finding is about level: the region "
            "is louder than your genre's curve expects. That is related to but distinct from "
            "harshness, which asks whether the region is peaking above its own neighbours. A "
            "smooth, even lift across the band reads as forward and aggressive; a spike inside "
            "it reads as glare. You can have either, or both."
        ),
        what_you_hear=(
            "The mix comes at you. Vocals and cymbals sit in front of the speakers rather than "
            "between them, transients feel sharp, and everything sounds close and immediate — "
            "which is genuinely exciting for the first thirty seconds. Then the volume you want "
            "starts drifting down. On earbuds and laptop speakers, which have their own peak "
            "here, it goes from forward to unpleasant."
        ),
        why_it_matters=(
            "This is the region that sets how long someone can listen, and how long someone can "
            "listen is a real commercial variable. It is also where the depth of a mix is "
            "decided: excess presence flattens the front-to-back dimension, because your brain "
            "reads a strong upper midrange as 'close' and everything ends up at the same "
            "distance. Genre matters here more than in most bands — rock, punk and metal carry "
            "considerably more presence than R&B, soul, jazz or lo-fi, and the target you are "
            "measured against reflects the style you selected."
        ),
        common_causes=(
            "Presence boosts on many sources at once, each one added to make that source 'cut'.",
            "A bright reference chased on monitors that are dull in this region.",
            "Exciters and enhancers, which usually work here even when the marketing says air.",
            "Distortion on midrange sources, which piles harmonics into the band from below.",
            "A high shelf on the mix bus that starts lower than intended. A shelf at 4 kHz has "
            "already begun lifting well below its corner.",
        ),
        how_to_fix=(
            FixStep(
                action="Establish whether it is a broad lift or a narrow peak.",
                detail=(
                    "Sweep a narrow boosted bell across the band. If one spot is far worse than "
                    "the rest, treat it as a resonance and fix it at the source. If the whole "
                    "band is evenly up, it was added deliberately somewhere and the fix is to "
                    "find and reduce that, not to notch."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Take it out where it went in.",
                detail=(
                    "Look for a mix-bus or master shelf first, then for the same boost repeated "
                    "on several channels. Reducing three source boosts by a decibel each is "
                    "audibly better than one broad cut on the master, which dulls the sources "
                    "that were correct along with the ones that were not."
                ),
            ),
            FixStep(
                action="Reduce the count of sources that live here rather than the level of all of them.",
                detail=(
                    "Give the band to the lead vocal and move the others: guitars get their "
                    "definition lower, hats and cymbals higher. Fewer things in the band is the "
                    "fix that keeps the mix exciting; turning the whole band down is the fix "
                    "that makes it dull."
                ),
            ),
            FixStep(
                action="Use a shelf rather than a bell if you are correcting the mix bus.",
                detail=(
                    "A gentle high shelf starting around 2.5-3 kHz, pulling about a decibel, "
                    "keeps the relationship between presence and the air above it intact. A wide "
                    "bell in the middle of the band leaves the top of it untouched, which pushes "
                    "the mix toward a hollow, slightly artificial character."
                ),
            ),
            FixStep(
                action="Match the tilt to a reference rather than to a number.",
                detail=(
                    "Capture the curve of a track in your genre and compare shapes. Presence "
                    "level is one of the strongest genre signatures there is, and a reference "
                    "settles it faster than any target curve — but apply the difference by hand, "
                    "in a decibel or two, rather than letting the match run at full depth."
                ),
                needs="eq_match",
                without=(
                    "A/B against a released track at matched loudness and listen specifically to "
                    "how far forward the vocal sits on each. That comparison is the whole "
                    "measurement; you are checking whether your mix is closer to the listener "
                    "than the record you are aiming at."
                ),
            ),
        ),
        how_to_verify=(
            "Listen to the full mix at a comfortable level for a couple of minutes without "
            "touching the volume. If you find yourself lowering it, you are not done. Re-analyse "
            "for the presence band inside the tolerance for your genre, and check the harshness "
            "index at the same time — if the level came down and harshness did not, there is a "
            "peak inside the band that a broad move will never reach."
        ),
        learn_more=(
            "There is a useful distinction hiding in this finding. Level and peakiness are "
            "different measurements of the same band, and they call for different fixes. Level "
            "asks how much total energy is in 2-5 kHz compared to what the genre expects, and it "
            "responds to shelves and broad moves. Peakiness — the harshness index — asks whether "
            "the band departs from the slope its own neighbours describe, and it responds only "
            "to narrow, targeted work. A mix can be high on level and low on peakiness, which is "
            "a bright, forward, perfectly listenable record. It can also be low on level and "
            "high on peakiness, which is a dull mix with a nasty spike in it, and turning the "
            "whole band down makes that one worse rather than better. Knowing which of the two "
            "you have determines whether you reach for a shelf or a scalpel, and getting that "
            "backwards is the most common way this region gets mishandled."
        ),
        minutes=20,
    ),
)

_add(
    "frequency_balance.presence_thin",
    Explainer(
        headline="The vocal sounds like it is behind the speakers — and turning it up will not bring it forward.",
        what_it_is=(
            "2 to 5 kHz is presence: the consonants that carry the words, the click of the "
            "beater, the stick on the snare, the bite of a guitar. Your ear is more sensitive "
            "here than anywhere else, which means a very small amount of energy in this band "
            "buys a great deal of perceived closeness. Being under target has a specific "
            "consequence that most producers experience without diagnosing: a source with no "
            "presence sounds distant, and the instinctive fix — raising its fader — makes it "
            "louder without making it any closer or any clearer."
        ),
        what_you_hear=(
            "The lead sounds like it is behind the band, or behind a curtain. You can hear it "
            "perfectly well and you still cannot make out the words at low volume. The mix "
            "sounds pleasant, smooth, and slightly like it is happening in another room. The "
            "reliable tell is the fader: if you have pushed the vocal up several times over the "
            "course of the session and it is now loud but still not clear, you have a presence "
            "problem rather than a level problem."
        ),
        why_it_matters=(
            "Your brain uses high-frequency content as a distance cue — air absorbs high "
            "frequencies over distance, so a dull source reads as far away no matter how loud it "
            "is. This is why turning it up fails: you are making a distant thing louder, and the "
            "listener still places it at the back. It also costs you intelligibility, and an "
            "unintelligible lyric is a skipped track. Genre matters: R&B, soul and lo-fi "
            "deliberately sit further back here than rock or country, and the target reflects "
            "that."
        ),
        common_causes=(
            "A dark source: a ribbon mic, an off-axis dynamic, a muffled guitar tone, a sample "
            "that was already dull.",
            "Over-de-essing. A de-esser with a low band or a broadband mode pulls the whole "
            "region down on every consonant, which is exactly the content this band needs.",
            "Tape, tube and 'analogue warmth' processing stacked across the chain, all of which "
            "soften the top of the midrange.",
            "Fixing harshness with too wide a cut. The harsh spike was narrow; the correction "
            "was not.",
            "A heavy low end making the presence region relatively quiet even though it was "
            "never cut — masking rather than absence.",
        ),
        how_to_fix=(
            FixStep(
                action="Try presence before you try level.",
                detail=(
                    "Return the lead's fader to where it was two rounds of pushing ago, then add "
                    "1.5-3 dB with a moderately wide bell around 3-4 kHz. The correct result is "
                    "that the source comes forward and gets clearer while sitting at a lower "
                    "level than before — clarity and loudness are different controls and this is "
                    "the one you wanted."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Check what is removing it before you add more.",
                detail=(
                    "Bypass the de-esser, the tape emulation and any wide cut you made to fix "
                    "harshness, one at a time. It is common for this band to be perfectly well "
                    "served in the recording and taken out three times over in the chain, and "
                    "adding a boost on top of three subtractions gets you noise and phase "
                    "instead of presence."
                ),
            ),
            FixStep(
                action="Clear the masking as well as adding the band.",
                detail=(
                    "If cymbals, hats or bright synths occupy the same region, a decibel or two "
                    "out of them does more for the lead than the same amount added to it, "
                    "because it lowers what the consonants have to push through rather than "
                    "raising the whole voice."
                ),
            ),
            FixStep(
                action="Generate presence rather than boosting it when the source is genuinely dark.",
                detail=(
                    "Light saturation on the lead produces harmonics from the fundamentals that "
                    "are already there. It reads as presence but sits inside the mix rather than "
                    "on top of it, and it does not amplify the noise floor and room tone the way "
                    "a large EQ boost on a dark source does."
                ),
                needs="saturation",
            ),
            FixStep(
                action="Use an exciter if there is genuinely nothing in the band to work with.",
                detail=(
                    "Synthesised harmonics are the only way to create content that was never "
                    "captured. Blend it low — this is the region where artificiality is most "
                    "audible — and check on headphones, where added harmonics show up more "
                    "clearly than on speakers."
                ),
                needs="exciter",
                without=(
                    "Build one: duplicate the source, high-pass the copy around 1.5 kHz, drive "
                    "it hard into any distortion you have, and blend it quietly underneath the "
                    "original. That is what an exciter is, and doing it by hand gives you "
                    "control over exactly which part of the signal gets excited."
                ),
            ),
        ),
        how_to_verify=(
            "Play the track at conversation level and follow the lyric without reading it. Then "
            "check that the lead's fader can come back down a decibel and still be perfectly "
            "clear — that is the proof you fixed presence rather than level. Re-analyse for the "
            "band inside your genre's tolerance, and keep an eye on the harshness and sibilance "
            "readings while you do it, because the failure mode of this fix is adding edge and "
            "spit along with the clarity."
        ),
        learn_more=(
            "Distance perception in hearing rests on three cues, and only one of them is level. "
            "The second is the ratio of direct sound to reverberation: more room, further away. "
            "The third is high-frequency content, because air absorbs high frequencies over "
            "distance far more than low ones — a sound fifty metres away has measurably less "
            "top than the same sound at two metres. Your auditory system uses all three "
            "constantly and without your involvement, which is why a dull source is heard as "
            "distant rather than as dull, and why raising its level produces a loud distant "
            "sound rather than a close one. It is also the practical basis of depth in a mix: "
            "you place things at the back by rolling off their top and adding room, and you "
            "bring things to the front with presence and dryness. Level barely participates. "
            "Once you have heard it work you stop reaching for the fader first."
        ),
        minutes=20,
    ),
)

_add(
    "frequency_balance.brilliance_hot",
    Explainer(
        headline="The top of the mix is hissy and splashy — the cymbals are running the record.",
        what_it_is=(
            "5 to 10 kHz is the detail band. It holds the body of hi-hats and cymbals, string "
            "and fret noise, breath, the crispness of a snare, and the upper half of sibilant "
            "consonants. Almost nothing has a fundamental up here — it is all overtones and "
            "noise — which is why it controls the perceived texture of a mix rather than its "
            "tone. Too much of it against your genre's curve makes a record sound bright in a "
            "papery, thin way rather than in an expensive way."
        ),
        what_you_hear=(
            "Splashy cymbals that smear over everything, a hissy quality on the whole mix, and "
            "consonants that spit. The track sounds detailed on first listen and thin on the "
            "second, because your attention keeps going to the top instead of to the song. On "
            "earbuds it is sharp; in a car it is the only thing you hear clearly over the road "
            "noise."
        ),
        why_it_matters=(
            "This band and sibilance overlap almost entirely, so an excess here usually brings a "
            "vocal spit problem with it. It is also the region where cheap tweeters and earbuds "
            "are least controlled, so the same master lands very differently across devices. And "
            "there is a monitoring trap: hi-hats and cymbals almost always sound too quiet on "
            "monitors at a comfortable level and too loud on headphones, so this is a band "
            "producers routinely overshoot in one direction while checking in the other."
        ),
        common_causes=(
            "Hi-hats and cymbals mixed by ear on monitors and never checked on headphones.",
            "An 'air' boost placed lower than intended. Most air shelves start well below the "
            "frequency printed on them and lift this band first.",
            "Exciters and enhancers on the master.",
            "Heavy saturation, which adds high-order harmonics and lands most of them here.",
            "Compensating for dull monitors or an over-damped room by brightening the file "
            "rather than fixing the monitoring.",
        ),
        how_to_fix=(
            FixStep(
                action="Check it on headphones before you decide it is a mix problem.",
                detail=(
                    "This is the band most affected by room absorption and by where your head "
                    "is. If the top is only excessive on one system, fix the monitoring rather "
                    "than the master — and if it is excessive on both, the rest of these steps "
                    "apply."
                ),
            ),
            FixStep(
                action="Turn the source down before you EQ the sum.",
                detail=(
                    "It is usually the hats and cymbals, and 1-2 dB off the overhead or hat "
                    "channel does what a master shelf does without dulling the vocal's "
                    "consonants and the acoustic guitar's detail at the same time."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Find the air boost you forgot about.",
                detail=(
                    "Check the mix bus, the master, and any channel strip preset with an air or "
                    "presence control engaged. A shelf that starts an octave lower than its "
                    "label is one of the most common causes of an excessive top, and removing it "
                    "is better than adding a corrective cut underneath it."
                ),
            ),
            FixStep(
                action="Use a dynamic band if the top only gets splashy in the loud sections.",
                detail=(
                    "Cymbals are the loudest thing in a chorus and nothing in a verse, so a "
                    "fixed shelf sized for the chorus dulls the whole song. A dynamic high shelf "
                    "doing 1-2 dB only when the band crosses its threshold tracks the "
                    "arrangement."
                ),
                needs="eq_dynamic",
                without=(
                    "Automate the cymbal or overhead fader down by a decibel in the loud "
                    "sections. Simpler, and it fixes the balance rather than the tone, which is "
                    "usually the real problem."
                ),
            ),
            FixStep(
                action="Check whether it is sibilance rather than cymbals.",
                detail=(
                    "Mute the drums and listen to the vocal alone in the mix. If the top is "
                    "still hot, the singer is the source and a de-esser will do more than any "
                    "broad shelf, which would only dull the cymbals that were fine."
                ),
                needs="deesser",
                without=(
                    "Ride the vocal's clip gain down on the worst consonants, or pull a narrow "
                    "static cut at the ess frequency as a stopgap. Either targets the actual "
                    "cause rather than the whole top of the record."
                ),
            ),
        ),
        how_to_verify=(
            "Play the busiest section and see whether your attention goes to the song or to the "
            "cymbals. Then check on earbuds, where this band is most exposed. Re-analyse for the "
            "brilliance band inside your genre's tolerance — lo-fi allows enormously less here "
            "than EDM or pop, because a rolled-off top is that style rather than a fault — and "
            "check the sibilance reading at the same time, since the two usually move together."
        ),
        learn_more=(
            "This is the band where listeners genuinely disagree, and it is worth knowing why "
            "before you argue about it. High-frequency hearing declines with age and with "
            "exposure, and the loss starts at the top and works down, so two people in the same "
            "room can have several decibels of difference in how they perceive this region and "
            "both be listening accurately for themselves. Playback hardware adds its own "
            "variation: tweeter dispersion narrows as frequency rises, so moving your head "
            "changes this band more than any other, and earbud response up here varies wildly "
            "between models. The practical consequence is that this is the region where you "
            "should trust a measurement and a reference track more than your ears alone — not "
            "because your ears are wrong, but because they are one sample from a distribution "
            "with a very wide spread, and the analyser is at least the same instrument every "
            "time."
        ),
        minutes=20,
    ),
)

_add(
    "frequency_balance.brilliance_thin",
    Explainer(
        headline="The mix sounds dull and closed, like it is playing through a door.",
        what_it_is=(
            "5 to 10 kHz is where detail and texture live: the shape of a cymbal rather than "
            "just its wash, the sound of fingers on strings, breath around a voice, the crisp "
            "part of a snare. It is almost entirely overtones and noise, so it carries very "
            "little of a mix's energy and a great deal of its character. Being under target here "
            "produces a mix that is tonally correct and somehow lifeless — nothing is missing "
            "when you go looking for it, and the whole thing sounds like it is happening behind "
            "something."
        ),
        what_you_hear=(
            "Dull, closed, muffled. Cymbals become a grey wash without shape. Acoustic guitars "
            "lose their strings and become a body. The mix sounds acceptable when it is loud and "
            "dead when it is quiet, because the little detail there is stops being audible "
            "first. Next to a commercial reference it sounds like a cloth is over the tweeters."
        ),
        why_it_matters=(
            "Detail is a large part of what people mean by production value, and its absence is "
            "one of the reasons a mix can be technically correct and still sound like a demo. It "
            "also affects intelligibility at low volume, where this band and the presence band "
            "carry most of the information. Genre matters, though: lo-fi has an extremely wide "
            "tolerance here on purpose, and a rolled-off top in that style is the point rather "
            "than a fault."
        ),
        common_causes=(
            "Mixing on bright headphones or in a live room, so you cut what your monitoring was "
            "adding.",
            "Lossy or low-quality source material. A sample pulled from a compressed file has "
            "little content up here to begin with.",
            "Over-de-essing the whole mix, which pulls this band down every time anyone sings.",
            "Heavy tape emulation, which softens the top by design, stacked on multiple busses.",
            "Age and exposure. High-frequency hearing declines gradually, and it is normal to "
            "compensate by rolling off what you can no longer hear clearly.",
        ),
        how_to_fix=(
            FixStep(
                action="Check whether the content exists before boosting.",
                detail=(
                    "Watch the analyser across the band while the busiest section plays. If "
                    "there is content and it is simply low, a shelf will work. If the region is "
                    "close to empty, boosting only raises noise, and you need to generate "
                    "harmonics instead."
                ),
                needs="meter_spectrum",
            ),
            FixStep(
                action="Add it with a gentle shelf on the master, then check the vocal.",
                detail=(
                    "1 to 2 dB, shelf starting around 7-8 kHz. Then listen to the singer, "
                    "because everything you add here also adds to the esses — if the shelf "
                    "brings sibilance with it, the fix belongs on the sources instead."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Lift the sides rather than the whole image.",
                detail=(
                    "A high shelf on the side channel only brightens the cymbals, room and "
                    "reverb without touching the centred vocal. You get air and detail with no "
                    "additional sibilance, which is the cleanest version of this move."
                ),
                needs="eq_mid_side",
                without=(
                    "Put the shelf on the individual sources that should be bright — overheads, "
                    "acoustic guitars, percussion — rather than on the master. Slower, and it "
                    "achieves the same separation between what gets brighter and what stays "
                    "where it is."
                ),
            ),
            FixStep(
                action="Generate the harmonics if the region is genuinely empty.",
                detail=(
                    "An exciter synthesises content above what exists, which EQ by definition "
                    "cannot. Blend it in at a level where switching it off makes the mix sound "
                    "closed rather than making it sound normal — that is the point where you "
                    "have added detail rather than fizz."
                ),
                needs="exciter",
                without=(
                    "Send a copy of the mix or the bright sources to a high-passed parallel "
                    "channel, drive it into any saturation you have, and blend it back "
                    "quietly. Distortion creates new harmonics above the input, so this makes "
                    "top end where an EQ would only amplify the noise floor."
                ),
            ),
        ),
        how_to_verify=(
            "A/B against a released track in your genre at matched loudness and listen only to "
            "the cymbals and the room. Yours should have shape rather than wash. Re-analyse for "
            "the brilliance band inside your genre's window — remembering that lo-fi is allowed "
            "a great deal of darkness here and does not need this fixing. Check sibilance at the "
            "same time: the most common way this repair goes wrong is trading dullness for spit."
        ),
        learn_more=(
            "There is a real difference between boosting the top and creating it, and it matters "
            "most in this band. An EQ is a filter: it can only change the level of what is "
            "already present, so applied to a region with nothing but noise in it, it gives you "
            "louder noise. Distortion is a non-linear process, and non-linearity generates new "
            "frequencies that were not in the input — multiples of what went in. That is why a "
            "saturator or an exciter can bring apparent top end back to a dull source and an EQ "
            "cannot, and why the two sound different even when the analyser shows a similar "
            "curve: the harmonics from distortion are locked to the source material, appearing "
            "and disappearing with it, whereas an EQ boost is a fixed lift applied to whatever "
            "happens to be there, including hiss between the notes. Program-dependent brightness "
            "sounds like part of the recording. Static brightness sounds like a setting."
        ),
        minutes=20,
    ),
)

_add(
    "frequency_balance.air_hot",
    Explainer(
        headline="The very top of the mix is too hot — some of it is probably noise rather than music.",
        what_it_is=(
            "Above about 10 kHz there is almost no musical information left. There are no "
            "fundamentals up there, and very few identifiable sounds; what remains is the last "
            "harmonics of cymbals, breath and room, plus the sense of openness that engineers "
            "call air. It is also where everything that is not music collects: hiss from "
            "preamps, noise floors from samples, digital aliasing from distortion plugins "
            "without oversampling, and the top edge of anything that was resampled badly. Being "
            "over the target here often means the extra energy is not signal."
        ),
        what_you_hear=(
            "A glassy, brittle sheen, or a fine constant hiss you notice most in the gaps between "
            "phrases. Cymbals sound like they are made of glass rather than metal. On some "
            "systems it just reads as shiny and expensive, which is why it goes unnoticed; on a "
            "tweeter-forward system or good headphones it sounds artificial and slightly "
            "electronic. If the extra energy is aliasing rather than harmonics, it sounds subtly "
            "wrong in a way people describe as 'digital' — inharmonic, unrelated to the notes "
            "being played."
        ),
        why_it_matters=(
            "Two concrete costs. First, lossy encoders allocate bits across the spectrum, and "
            "energy up here is expensive to encode and nearly inaudible, so an over-bright "
            "master spends the listener's bitrate on the least useful part of the file and the "
            "rest of the mix gets less. Second, if the energy is noise or aliasing rather than "
            "music, you are pushing your limiter with something no listener wants. Genre "
            "matters: EDM and modern pop are built with a lot of air, jazz and lo-fi are not."
        ),
        common_causes=(
            "Noise floor: hiss from a preamp, a noisy sample library, or an amp sim left with "
            "its input high, all of which are broadband but only visible up here.",
            "Distortion, saturation or clipping running without oversampling. Anything that "
            "distorts creates harmonics above the input, and the ones too high for the session's "
            "sample rate to store get reflected back down into the top of the audible range "
            "instead of disappearing — an effect called aliasing.",
            "Air shelves stacked at several stages — one on the vocal, one on the bus, one on "
            "the master — each of them reasonable alone.",
            "Exciters on the master, which are designed to put energy exactly here.",
            "Compensating for hearing or monitoring that rolls off up top by adding what you "
            "cannot hear.",
        ),
        how_to_fix=(
            FixStep(
                action="Establish whether it is music or noise.",
                detail=(
                    "Watch the top of the analyser during a gap or a fade. Real air moves with "
                    "the cymbals and the vocal; noise sits at a constant level regardless of "
                    "what is playing. If it is constant, no EQ curve is the right fix — you have "
                    "a noise problem, not a balance problem."
                ),
                needs="meter_spectrum",
            ),
            FixStep(
                action="Turn on oversampling in everything that distorts — saturators, clippers, amp sims, some limiters.",
                detail=(
                    "Those plugins generate harmonics higher than the session's sample rate can "
                    "store, and without oversampling switched on, the excess is reflected back "
                    "down into the top of the audible range as content that has no musical "
                    "relationship to the notes that made it. Switching it on removes an entire "
                    "category of unwanted energy up here and costs nothing but CPU."
                ),
            ),
            FixStep(
                action="Remove the noise at the source rather than filtering the master.",
                detail=(
                    "Hiss is broadband — a low-pass on the master only hides the part of it you "
                    "can see. Gate or clean the noisy channels, or replace the noisy sample, and "
                    "the top of the mix comes back down without losing any of the cymbals."
                ),
                needs="noise_reduction",
                without=(
                    "Gate or automate the offending channels so they are silent when they are "
                    "not playing, which removes most of the accumulated hiss. Solo each channel "
                    "in a quiet passage to find which ones are contributing — it is usually one "
                    "or two, not all of them."
                ),
            ),
            FixStep(
                action="Count your air boosts and remove one rather than cutting on the master.",
                detail=(
                    "Air shelves are the easiest processing to apply twice without noticing, "
                    "because each one sounds like an improvement in isolation. Removing a "
                    "duplicate is always better than adding a corrective cut below it, which "
                    "leaves you with two filters fighting and extra phase shift for nothing."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Low-pass the extreme top if the content there is not musical.",
                detail=(
                    "A gentle filter starting around 18-19 kHz removes aliasing and ultrasonic "
                    "junk without touching anything anyone can hear. Do this only after you know "
                    "the energy is not real air, because on a mix that genuinely uses the region "
                    "it will read as a loss of openness."
                ),
            ),
        ),
        how_to_verify=(
            "Listen to a quiet passage or a tail on headphones at a slightly higher level than "
            "usual. There should be no constant hiss underneath it. Re-analyse for the air band "
            "inside your genre's window, which is generous for EDM and cinematic work and very "
            "tight for lo-fi. And listen once more to the cymbals: they should sound like metal "
            "rather than glass, which is the perceptual difference between harmonics and "
            "aliasing better than any meter will tell you."
        ),
        learn_more=(
            "Aliasing is worth understanding because it is invisible until you know what it is. "
            "A digital system can only represent frequencies below half its sample rate. When a "
            "non-linear process — any distortion, saturation or clipping — generates harmonics "
            "above that limit, they do not simply vanish: they are reflected back down into the "
            "audible band, landing at frequencies that have no harmonic relationship to the "
            "notes that created them. Harmonic distortion sounds musical because its products "
            "are multiples of the input. Aliasing sounds wrong because its products are "
            "arbitrary, and they move in the opposite direction to the pitch — play a rising "
            "line through an aliasing plugin and you can hear a second line descending "
            "underneath it. Oversampling fixes it by running the non-linearity at a higher "
            "internal rate, so the new harmonics fit below the internal limit and are filtered "
            "out before the signal returns to your session's rate. It is the difference between "
            "distortion that sounds like an instrument and distortion that sounds like a bug."
        ),
        minutes=20,
    ),
)

_add(
    "frequency_balance.air_thin",
    Explainer(
        headline="The top octave is missing — the mix sounds closed in rather than open.",
        what_it_is=(
            "Above roughly 10 kHz there is no melody and almost no identifiable sound. What "
            "lives there is the tail of cymbal harmonics, breath, room reflections and the "
            "faint texture that makes a recording feel like it was made in a real space. "
            "Engineers call it air because that is what it sounds like: the space around the "
            "sounds rather than the sounds themselves. It is worth being honest about the "
            "priority — plenty of great records have very little up here, and this is the least "
            "urgent of the band findings. What it is genuinely good at is telling you something "
            "upstream is wrong, because the most common reason a mix has no air is that "
            "something in the chain removed it."
        ),
        what_you_hear=(
            "Closed in. The mix sounds like it is behind glass, or like a good recording played "
            "back through a slightly veiled system. It is not dullness in the ordinary sense — "
            "the cymbals are there and the vocal is clear — it is the absence of the sense of "
            "space around them. Most people notice it only in comparison, which is why an A/B "
            "against a reference is the right diagnostic here."
        ),
        why_it_matters=(
            "The mix will sound smaller and less expensive than its peers on good systems, "
            "though on earbuds and phone speakers it changes very little. The stronger reason to "
            "care is diagnostic: a hard cliff at the top of the spectrum means a lossy source "
            "somewhere in your session, and if a stem or a sample went through a codec, that "
            "same source has other artefacts you cannot see on a spectrum. Air is the symptom "
            "that makes it visible."
        ),
        common_causes=(
            "A lossy source file. MP3 and AAC encoders low-pass their output, often around "
            "15-16 kHz, so any sample, stem or bounce that passed through one has a hard cliff "
            "and nothing above it.",
            "Heavy tape emulation, which rolls off the top by design and does it on every "
            "instance in the chain.",
            "A low-pass filter left on from something else, on a bus or a master.",
            "Mixing on bright headphones, so you removed what they were adding.",
            "Aggressive noise reduction, which takes the top off along with the hiss because "
            "that is where most of the hiss was.",
        ),
        how_to_fix=(
            FixStep(
                action="Look for a cliff before you look for a curve.",
                detail=(
                    "On an analyser, real content fades away gradually toward 20 kHz. A vertical "
                    "wall at a fixed frequency is a codec or a filter, not a mix decision. Find "
                    "which channel has it by soloing candidates — it is usually one sample or "
                    "one stem, and no processing recovers what a codec threw away."
                ),
                needs="meter_spectrum",
            ),
            FixStep(
                action="Re-source the file rather than trying to repair it.",
                detail=(
                    "If a stem or sample was lossy, replacing it with the original is the entire "
                    "fix and everything else is cosmetics. This is worth ten minutes of hunting: "
                    "the codec also introduced artefacts you cannot see, and they are on the "
                    "part of the file you can hear."
                ),
            ),
            FixStep(
                action="Add a gentle shelf only where there is content to lift.",
                detail=(
                    "1 to 2 dB starting around 12 kHz. Check the noise floor and the esses "
                    "immediately afterwards — a top shelf raises hiss and sibilance at exactly "
                    "the same rate as it raises the air you wanted, and on a quiet passage that "
                    "trade is often not worth it."
                ),
                needs="eq_static",
            ),
            FixStep(
                action="Lift the sides rather than the middle.",
                detail=(
                    "Air lives in the room, the reverb and the cymbals, which are mostly in the "
                    "sides; the vocal and its esses are in the middle. A high shelf on the side "
                    "channel alone gives you openness and width with none of the added "
                    "sibilance, which is the cleanest version of this move by a wide margin."
                ),
                needs="eq_mid_side",
                without=(
                    "Put the shelf on the overheads, room mics, reverb returns and percussion "
                    "instead of on the master. It takes longer and it targets the same content "
                    "the side channel would have."
                ),
            ),
            FixStep(
                action="Synthesise it if there is nothing there at all.",
                detail=(
                    "An exciter creates harmonics above the existing content, which is the only "
                    "way to get top end from material that has none. Keep it subtle — the "
                    "failure mode is a fizzy halo that sits on top of the mix rather than "
                    "belonging to it — and always check it on headphones."
                ),
                needs="exciter",
                without=(
                    "Run a parallel send, high-pass it steeply around 6-8 kHz, drive it into "
                    "whatever saturation you have, and blend it back very quietly. Distortion "
                    "generates new harmonics above what it is given, so it makes air where a "
                    "shelf would only amplify silence and hiss."
                ),
            ),
        ),
        how_to_verify=(
            "A/B against a released track at matched loudness and listen for space around the "
            "sounds rather than for brightness. Re-analyse for the air band inside your genre's "
            "window — the tolerance up here is the widest of any band precisely because opinion "
            "varies, and lo-fi is allowed to have essentially none. Then check the sibilance "
            "index and a quiet passage: if the esses got sharper or a hiss appeared, you bought "
            "the air with something you did not want."
        ),
        learn_more=(
            "Almost nothing up here is a sound in its own right. Above roughly 10 kHz you are "
            "hearing the harmonics of things whose fundamentals sit far below — a cymbal's "
            "shimmer, the noise of a breath, the first reflections off the walls of the room. "
            "That is why air reads as *space* rather than as brightness: it is mostly "
            "information about where a sound happened rather than what it was. It is also why "
            "adding it with an EQ works less often than people expect. A shelf lifts whatever "
            "is already there, and if the source never captured that detail there is nothing "
            "under the shelf but noise and cymbal wash, which is the sound of a mix that has "
            "been brightened rather than opened up.\n\n"
            "Two very different things produce a thin top octave, and they call for opposite "
            "responses. The ordinary one is that the balance is simply dark — a gentle slope "
            "away from the top, common in lo-fi, in a lot of hip-hop, and on anything tracked "
            "through warm analogue. That is a taste decision and it needs no fixing.\n\n"
            "The other one is worth knowing how to spot, because it is invisible until you do. "
            "Lossy codecs discard what a psychoacoustic model predicts you will not hear, and "
            "one of their first economies is to low-pass the signal — MP3 at moderate bitrates "
            "typically throws away everything above about 16 kHz. That leaves a vertical cliff "
            "on an analyser, and the cliff is a reliable fingerprint: no microphone, instrument "
            "or filter you would deliberately use produces a wall that steep at one frequency. "
            "A slope is taste; a wall is a codec. If you find a wall, the missing air is the "
            "least of it — the same encode left pre-echo on transients and quantisation noise "
            "in the mids, problems you can hear but cannot see. The cliff is the cheap detector "
            "for the expensive problem, so it is worth checking for even when the darkness "
            "turns out to be deliberate."
        ),
        minutes=15,
    ),
)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(target: Dict[str, Explainer]) -> None:
    """Copy this module's explainers into the shared registry.

    Called by knowledge.py after its own definitions. Copying rather than
    importing the shared dict here keeps this module free of a circular import
    and makes the merge order explicit at the call site.
    """
    target.update(EXPLAINERS)
