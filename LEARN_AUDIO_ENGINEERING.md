# Audio Engineering Learning Guide

A comprehensive guide to understanding all the audio concepts used in MixDoctor. Start from the top and work your way down.

---

## Table of Contents

1. [The Basics: What is a Mix?](#1-the-basics-what-is-a-mix)
2. [Frequency & The Spectrum](#2-frequency--the-spectrum)
3. [Loudness & Metering](#3-loudness--metering)
4. [Dynamics & Compression](#4-dynamics--compression)
5. [Stereo Field & Phase](#5-stereo-field--phase)
6. [Common Mix Problems & Solutions](#6-common-mix-problems--solutions)
7. [Mastering Basics](#7-mastering-basics)
8. [Recommended Learning Path](#8-recommended-learning-path)

---

## 1. The Basics: What is a Mix?

### What You're Actually Doing

Mixing is the process of combining multiple audio tracks (vocals, drums, bass, guitars, synths, etc.) into a single stereo file that sounds cohesive, balanced, and professional.

Think of it like cooking: you have individual ingredients (tracks), and you're combining them into a dish (the mix). The goal is balance - no single ingredient should overpower the others unless intentional.

### The Three Dimensions of a Mix

Every mix exists in three dimensions:

| Dimension | Controlled By | Think Of It As |
|-----------|--------------|----------------|
| **Volume** (up/down) | Faders, compression | How loud each element is |
| **Frequency** (low/high) | EQ | The "tone" - bass to treble |
| **Stereo** (left/right) | Panning, stereo width | Where sounds sit left to right |

A fourth dimension is often added:

| Dimension | Controlled By | Think Of It As |
|-----------|--------------|----------------|
| **Depth** (front/back) | Reverb, delay, volume | How close or far something sounds |

### Key Terms

- **Track/Channel**: A single audio source (one vocal, one kick drum, etc.)
- **Bus/Group**: Multiple tracks routed together (all drums to one "drum bus")
- **Master/Mix Bus**: The final stereo output where everything combines
- **Plugin**: Software that processes audio (EQ, compressor, reverb, etc.)
- **DAW**: Digital Audio Workstation (Ableton, Logic, Pro Tools, FL Studio, etc.)

---

## 2. Frequency & The Spectrum

### What is Frequency?

Frequency is how many times a sound wave vibrates per second, measured in **Hertz (Hz)**.

- **Low frequency** = slow vibrations = bass sounds (20-250 Hz)
- **High frequency** = fast vibrations = treble sounds (4,000-20,000 Hz)

Humans hear roughly **20 Hz to 20,000 Hz** (20 kHz).

### The Frequency Bands (MixDoctor Uses These)

| Band | Range | What Lives Here | Character |
|------|-------|-----------------|-----------|
| **Sub** | 20-60 Hz | Sub bass, kick drum fundamental | Felt more than heard, rumble |
| **Low** | 60-250 Hz | Bass guitar, kick body, bass synths | Warmth, power, boom |
| **Low-Mid** | 250-500 Hz | Vocal body, snare body, guitars | Warmth vs. mud |
| **Mid** | 500-2,000 Hz | Vocals, most instruments | Clarity, presence |
| **Presence** | 2,000-6,000 Hz | Vocal consonants, snare crack, guitar attack | Intelligibility, edge |
| **Air** | 6,000-20,000 Hz | Cymbals, breath, sparkle | Brightness, openness |

### The Most Important Concept: Masking

**Masking** occurs when two sounds occupy the same frequency range - they fight for space and neither is heard clearly.

Example: A bass guitar and kick drum both heavy in 60-100 Hz will sound muddy together.

**Solution**: Give each element its own "space" in the frequency spectrum using EQ.

### EQ (Equalization)

EQ is your primary tool for shaping frequency. Types of EQ moves:

| Type | What It Does | When To Use |
|------|--------------|-------------|
| **High-pass filter (HPF)** | Removes everything below a frequency | Clean up rumble on vocals, guitars |
| **Low-pass filter (LPF)** | Removes everything above a frequency | Tame harsh synths, create distance |
| **Bell/Peak** | Boost or cut a specific range | Surgical fixes, tonal shaping |
| **Shelf** | Boost or cut everything above/below a point | Broad tonal changes (add "air", reduce "mud") |

### EQ Golden Rules

1. **Cut before you boost** - Removing problems is better than adding "fixes"
2. **Use narrow Q for cutting, wide Q for boosting** - Surgical cuts, musical boosts
3. **Less is more** - Small moves (1-3 dB) usually sound more natural
4. **Always A/B compare** - Toggle the EQ on/off to make sure it's actually helping

### Practice Exercise

1. Solo a vocal track
2. Sweep a narrow bell boost (+6 dB, Q=4) slowly from 200 Hz to 5,000 Hz
3. Listen for frequencies that sound harsh or muddy
4. Cut those frequencies by 2-4 dB instead of boosting
5. Unsolo and listen in context

---

## 3. Loudness & Metering

### Why Loudness Matters

Your ears perceive loudness logarithmically - a sound must be 10x more powerful to sound "twice as loud." This is why we use **decibels (dB)**, a logarithmic scale.

### Types of Loudness Measurement

#### Peak Level (dBFS)

- Measures the **highest instantaneous point** in the waveform
- **dBFS** = decibels Full Scale (0 dBFS is the maximum before digital clipping)
- Used to prevent clipping
- **Target**: Keep peaks below -1 dBFS (leaves headroom for mastering)

#### RMS Level (dB)

- Measures **average loudness** over time
- Closer to how we perceive loudness
- Higher RMS = sounds louder, but less dynamic

#### LUFS (Loudness Units Full Scale)

**This is the modern standard.** LUFS measures perceived loudness the way humans actually hear it.

| Term | Meaning |
|------|---------|
| **Integrated LUFS** | Average loudness of the entire track |
| **Short-term LUFS** | Loudness over 3-second windows |
| **Momentary LUFS** | Loudness over 400ms windows |

#### Loudness Targets by Platform

| Platform | Target LUFS |
|----------|-------------|
| Spotify | -14 LUFS |
| Apple Music | -16 LUFS |
| YouTube | -14 LUFS |
| Club/DJ music | -6 to -9 LUFS |
| Film/TV | -24 LUFS |

### True Peak

**True Peak** measures what happens **between** samples when digital audio is converted to analog. It can be higher than your peak meter shows!

- **Target**: -1.0 dBTP (true peak) or lower
- Why: Prevents distortion when streaming services convert your audio

### Loudness Range (LRA)

Measures the **difference between loud and quiet parts** of your track.

| LRA | Description |
|-----|-------------|
| < 4 LU | Very compressed, little dynamics (EDM, pop) |
| 4-8 LU | Moderate dynamics (most modern music) |
| 8-12 LU | Good dynamics (rock, acoustic) |
| > 12 LU | Very dynamic (classical, jazz, film scores) |

### Crest Factor

The difference between peak level and RMS level. Higher crest factor = more dynamic, punchier transients.

| Crest Factor | Interpretation |
|--------------|----------------|
| < 6 dB | Over-compressed, squashed |
| 6-10 dB | Typical modern music |
| 10-14 dB | Punchy, dynamic |
| > 14 dB | Very dynamic, might need compression |

---

## 4. Dynamics & Compression

### What Are Dynamics?

Dynamics = the difference between loud and quiet parts.

- **High dynamics**: Big difference between soft and loud (expressive but inconsistent)
- **Low dynamics**: Everything similar volume (consistent but potentially lifeless)

### Compression Explained

A **compressor** automatically turns down loud parts, reducing dynamic range.

Think of it like an assistant riding the volume fader - when things get too loud, they turn it down.

### Compressor Parameters

| Parameter | What It Does | Setting Guidelines |
|-----------|--------------|-------------------|
| **Threshold** | Volume level where compression starts | Lower = more compression |
| **Ratio** | How much to reduce (4:1 means 4 dB over threshold becomes 1 dB) | 2:1-4:1 gentle, 8:1+ heavy |
| **Attack** | How fast compression kicks in | Fast (1-10ms) catches transients, Slow (20-50ms) lets punch through |
| **Release** | How fast compression lets go | Match to tempo/groove, typically 50-300ms |
| **Makeup Gain** | Boosts output to compensate for reduction | Match perceived loudness before/after |
| **Knee** | How gradually compression engages | Soft knee = smooth, Hard knee = aggressive |

### Compression Types by Use

| Type | Best For | Characteristics |
|------|----------|-----------------|
| **VCA** | Drums, mix bus | Clean, precise, fast |
| **FET (1176-style)** | Vocals, drums, aggressive | Colorful, punchy, fast |
| **Opto (LA-2A style)** | Vocals, bass | Smooth, slow, musical |
| **Tube/Variable-mu** | Mix bus, mastering | Glue, warmth, slow |

### Common Compression Mistakes

1. **Too much compression**: Sounds squashed, no life, pumping artifacts
2. **Attack too fast**: Kills transients, drums lose punch
3. **Release too slow**: Pumping, breathing, unnatural volume swings
4. **Not gain-matching**: Louder always sounds "better" - match levels to judge accurately

### Parallel Compression

Instead of compressing a signal directly, blend a heavily compressed copy with the original:

1. Send your track to a bus
2. Compress that bus heavily (high ratio, low threshold)
3. Blend the compressed bus under the original
4. Result: Weight and sustain without losing transients

---

## 5. Stereo Field & Phase

### Stereo Basics

Stereo = two channels (left and right) that create the illusion of width and space.

### Panning

Where a sound sits in the stereo field:

| Position | Use For |
|----------|---------|
| **Center** | Lead vocal, kick, snare, bass (low frequencies) |
| **Slight L/R** | Supporting elements, double-tracked parts |
| **Hard L/R** | Stereo pairs (overheads, room mics), wide synths |

**Rule**: Keep low frequencies (below 150-200 Hz) centered. Bass in the sides causes phase issues.

### Stereo Width

How "wide" the stereo image feels. Measured as a ratio (0 = mono, 1 = typical stereo, >1 = extra wide).

| Width | Description |
|-------|-------------|
| 0 - 0.2 | Nearly mono, focused |
| 0.3 - 0.5 | Moderate width, common for balanced mixes |
| 0.5 - 0.7 | Wide, spacious |
| > 0.8 | Very wide (can cause phase issues) |

### Stereo Correlation

Measures how similar the left and right channels are:

| Correlation | Meaning |
|-------------|---------|
| +1.0 | Perfectly identical (mono) |
| +0.5 to +1.0 | Good, mono-compatible |
| 0 to +0.5 | Wide stereo, might have issues in mono |
| 0 to -1.0 | Phase problems, will disappear in mono |
| -1.0 | Perfectly opposite (complete cancellation in mono) |

### Phase Explained

**Phase** refers to the timing relationship between waveforms. When two identical sounds are:

- **In phase**: They add together (louder)
- **Out of phase**: They cancel each other (quieter or silent)

### Mono Compatibility

Many playback systems sum stereo to mono (phone speakers, some club systems, Bluetooth speakers). If your mix has phase issues, elements will disappear or sound thin in mono.

**Always check your mix in mono!**

### Common Phase Problems

1. **Stereo widening plugins used excessively**: Creates out-of-phase information
2. **Multi-mic recordings**: Drum overheads, room mics at different distances
3. **Layered sounds**: Two similar synths slightly out of time

### Fixing Phase Issues

1. **Check in mono**: Flip your monitoring to mono regularly
2. **Phase flip button**: Try flipping the phase (180°) on one side
3. **Time alignment**: Nudge tracks to align transients
4. **Stereo width reduction**: Pull back stereo widening until mono sounds good

---

## 6. Common Mix Problems & Solutions

### Problem: Muddy Mix

**Symptoms**: Everything sounds thick, unclear, no definition

**Causes**:
- Too much energy in 200-500 Hz range
- Multiple elements fighting in low-mids
- Room reflections captured in recordings

**Solutions**:
1. High-pass filter everything that doesn't need low end (vocals, guitars, keys)
2. Cut 200-400 Hz on guitars, synths, and vocals (2-4 dB)
3. Let the bass and kick own the low end
4. Use narrow cuts to find and remove specific problem frequencies

### Problem: Harsh/Brittle Mix

**Symptoms**: Fatiguing to listen to, hurts ears at volume, sibilance

**Causes**:
- Too much energy in 2-6 kHz range
- Over-brightening during mixing
- Poor microphone choice or placement

**Solutions**:
1. Dynamic EQ or de-esser on vocals (4-8 kHz)
2. Low-pass filter harsh synths and samples
3. Cut 3-5 kHz on aggressive elements
4. Use saturation to add harmonics instead of EQ boosting

### Problem: No Punch

**Symptoms**: Drums sound flat, no impact, lifeless groove

**Causes**:
- Over-compression with fast attack
- Transients lost in processing
- Samples already over-processed

**Solutions**:
1. Slow down compressor attack (20-50ms) to let transients through
2. Use parallel compression instead of heavy direct compression
3. Transient shaper to add attack
4. Layer punchy one-shot samples under weak drums

### Problem: Thin/Weak Mix

**Symptoms**: No weight, sounds small, lacks power

**Causes**:
- Over-filtering low end
- Not enough low-mid body
- Poor arrangement (nothing filling low frequencies)

**Solutions**:
1. Check high-pass filter frequencies (might be too high)
2. Add subtle low shelf boost on mix bus (+1-2 dB)
3. Layer sub bass under existing bass
4. Saturate bass elements to add harmonics that translate on small speakers

### Problem: Buried Lead Vocal

**Symptoms**: Can't understand words, vocal gets lost

**Causes**:
- Competing elements in 1-5 kHz range
- Too much reverb
- Poor compression control

**Solutions**:
1. Cut 2-4 kHz on competing instruments (guitars, synths)
2. Sidechain reverb/delay to duck when vocal plays
3. Automate vocal level phrase by phrase
4. Use dynamic EQ to boost vocal presence only when it's playing

---

## 7. Mastering Basics

### What is Mastering?

Mastering is the final step - preparing your mix for distribution. It ensures:

- Consistent loudness across songs
- Proper format and levels for streaming/physical media
- Final polish and quality control

### Mastering is NOT

- A fix for a bad mix
- Just making it louder
- Applying presets

### Mastering Chain Basics

Typical order (adjust based on the mix's needs):

1. **Corrective EQ**: Fix any remaining tonal issues
2. **Compression**: Add glue and control dynamics
3. **Tonal EQ**: Shape the overall tone
4. **Stereo processing**: Adjust width if needed
5. **Limiter**: Set final loudness and prevent clipping

### Mastering Readiness Checklist

Before sending for mastering (or mastering yourself):

- [ ] Mix sounds good on multiple speaker systems
- [ ] No clipping (peaks below -1 dBFS)
- [ ] No excessive processing on mix bus
- [ ] Headroom left (-3 to -6 dBFS peak is ideal)
- [ ] No audible distortion, clicks, or pops
- [ ] Phase/mono compatibility checked

### When Is a Mix "Done"?

A mix is ready for mastering when:

1. Every element can be heard clearly
2. The balance feels right on multiple playback systems
3. No frequency range is over/under-represented
4. The emotion and energy of the song comes through
5. You've listened to it fresh (after a break) and still like it

---

## 8. Recommended Learning Path

### Week 1-2: Foundations

**Focus**: Frequency and EQ

**Do**:
- Learn the frequency ranges by heart
- Practice identifying frequencies (use training apps)
- EQ one track per day, focusing on cutting problems

**Watch**:
- "How to EQ" - any YouTube tutorial (start with Produce Like A Pro or Pensado's Place)
- Frequency ear training (SoundGym, TrainYourEars)

### Week 3-4: Dynamics

**Focus**: Compression and loudness

**Do**:
- Compress a vocal from scratch
- Compress drums, focusing on attack/release
- A/B compressed vs uncompressed while level-matched

**Watch**:
- "Compression Explained" tutorials
- Study different compressor types

### Week 5-6: Space and Stereo

**Focus**: Panning, width, and depth

**Do**:
- Create a mix using only volume and panning (no effects)
- Experiment with reverb and delay on different elements
- Check all mixes in mono

**Watch**:
- "Creating Depth in a Mix"
- "Stereo Width Explained"

### Week 7-8: Integration

**Focus**: Full mixes

**Do**:
- Mix a full song from start to finish
- Reference against professional mixes in the same genre
- Get feedback from other producers/engineers

**Read**:
- "Mixing Secrets for the Small Studio" by Mike Senior
- "The Mixing Engineer's Handbook" by Bobby Owsinski

### Ongoing Practice

1. **Reference constantly**: Compare your mixes to professional releases
2. **Mix in context**: Always check changes against the full mix
3. **Take breaks**: Ear fatigue causes bad decisions
4. **Trust your ears**: Meters guide, but ears decide
5. **Finish things**: A finished mediocre mix teaches more than an unfinished "perfect" one

---

## Quick Reference Card

### EQ Cheat Sheet

| Issue | Frequency | Action |
|-------|-----------|--------|
| Rumble | 20-60 Hz | High-pass filter |
| Boom/mud | 200-400 Hz | Cut 2-4 dB |
| Boxy | 300-600 Hz | Cut 2-4 dB |
| Honky/nasal | 800-1500 Hz | Cut 2-3 dB |
| Harsh | 2-5 kHz | Cut or use dynamic EQ |
| Sibilant | 5-9 kHz | De-esser |
| Dull | 8-12 kHz | Shelf boost 1-2 dB |

### Compression Starting Points

| Source | Ratio | Attack | Release |
|--------|-------|--------|---------|
| Lead Vocal | 3:1 - 4:1 | 10-25ms | 40-100ms |
| Drums (bus) | 4:1 | 10-30ms | Auto or 100-200ms |
| Bass | 4:1 - 6:1 | 20-40ms | 100-200ms |
| Mix Bus | 2:1 - 3:1 | 30-50ms | Auto or 100-300ms |

### Target Levels

| Metric | Target |
|--------|--------|
| Peak | < -1 dBFS |
| True Peak | < -1 dBTP |
| Integrated LUFS | -14 (streaming) to -8 (loud genres) |
| Loudness Range | 4-8 LU (most genres) |

---

## Glossary

| Term | Definition |
|------|------------|
| **dB** | Decibel - logarithmic unit for measuring audio level |
| **dBFS** | Decibels Full Scale - digital level where 0 is maximum |
| **Hz** | Hertz - frequency measurement (cycles per second) |
| **kHz** | Kilohertz - 1,000 Hz |
| **LUFS** | Loudness Units Full Scale - perceived loudness measurement |
| **LRA** | Loudness Range - difference between quiet and loud parts |
| **EQ** | Equalizer - adjusts frequency balance |
| **HPF** | High-Pass Filter - removes low frequencies |
| **LPF** | Low-Pass Filter - removes high frequencies |
| **Q** | Quality factor - width of an EQ band (higher = narrower) |
| **RMS** | Root Mean Square - average level measurement |
| **Transient** | The initial attack of a sound |
| **Bus** | A routing path where multiple tracks combine |
| **Clipping** | Distortion from exceeding 0 dBFS |
| **Headroom** | Space between your peaks and 0 dBFS |

---

*Last updated: February 2026*
*Created for MixDoctor - AI Mix Analysis*
