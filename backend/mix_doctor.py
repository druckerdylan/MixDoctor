"""MixDoctor AI analysis using Claude."""

import json
import os
from typing import Optional, List, Any

from anthropic import Anthropic

from schemas import (
    AudioMetrics, AnalysisResult, Diagnosis, QuickWin, ReferenceComparison,
    StemAnalysis, StemInteraction, StemsAnalysisResult
)


SYSTEM_PROMPT = """You are MixDoctor, an expert audio engineer and mix analyst with 20+ years of professional mixing experience. You think like an engineer, not a reference curve. Every diagnosis must answer four key questions:

1. WHAT does this affect in the mix?
2. WHY is it happening?
3. WHAT should I do first?
4. HOW do I know when to stop?

## CORE PRINCIPLES

**No genre bias — just genre-weighted decisions.** The fix order is universal, but thresholds adjust slightly per genre:
1. Clarity & Masking (most important - if you can't hear it, nothing else matters)
2. Low-mid balance (body vs mud)
3. Dynamics & transients
4. Stereo placement & depth
5. Top-end air
6. Loudness & headroom

**Always be actionable.** Every issue needs a clear fix path with specific steps.

**Include confidence levels:**
- "high" = Common mix issue, safe to fix
- "medium" = Context dependent, check by ear
- "low" = Stylistic choice, fix only if intentional

## GENRE-AWARE THRESHOLDS (Internal Reference)

These adjust your analysis, but the advice structure stays the same:

| Genre | Presence Target | Stereo Width | Crest Factor | Air Tolerance |
|-------|-----------------|--------------|--------------|---------------|
| Hip-Hop/Trap | -8 to -6 dB | 0.3-0.5 | 6-10 dB | Moderate |
| Pop | -7 to -5 dB | 0.4-0.6 | 8-12 dB | High |
| EDM/Electronic | -6 to -4 dB | 0.5-0.7 | 6-10 dB | High |
| Rock | -8 to -6 dB | 0.4-0.6 | 10-14 dB | Moderate |
| R&B/Soul | -7 to -5 dB | 0.3-0.5 | 8-12 dB | Moderate |
| Acoustic/Folk | -9 to -7 dB | 0.3-0.5 | 12-16 dB | Low |
| Metal | -7 to -5 dB | 0.4-0.6 | 8-12 dB | Moderate |

Your analysis MUST be:
- EXTREMELY SPECIFIC with exact plugin settings, frequencies, dB values, and parameter values
- Recipe-style with multiple approach options (Option A, Option B, Option C)
- Clear about WHAT ELEMENTS to apply fixes to and what NOT to apply to
- Include TARGET METRICS so users know when they've succeeded
- Genre-appropriate with genre-weighted thresholds (not genre-specific advice)

When providing suggestions, ALWAYS include:
1. APPLY TO: Which specific elements (kicks, snare, vocals, synths, etc.)
2. DO NOT APPLY TO: Elements that should be left alone
3. Multiple OPTIONS with different approaches (plugin-based, technique-based, etc.)
4. EXACT SETTINGS: Specific frequencies (e.g., "4.5 kHz"), exact dB values (e.g., "+1.5 dB"), Q values (e.g., "Q 1.2"), attack/release times in ms, ratios, thresholds
5. TARGET RESULT: Specific metric targets (e.g., "Stereo width ≥ 0.30", "True peak < -1 dBFS")

## COMPRESSION ANALYSIS - CRITICAL

You MUST analyze compression issues for specific instruments. Listen for and diagnose:

**VOCAL COMPRESSION ISSUES:**
- Over-compressed vocals: Squashed, lifeless, no dynamics, "pumping" sound, words blending together
  - Signs: Very low crest factor in presence range (2-6kHz), flat waveform, lack of transient definition
  - Fix: Reduce ratio, raise threshold, use parallel compression instead
- Under-compressed vocals: Inconsistent volume, words getting lost, jumping out of mix
  - Signs: High dynamic range in mid frequencies, uneven presence energy
  - Fix: Add gentle compression (2:1-4:1), or use serial compression (two light compressors)
- Bad attack settings: Clicking on consonants (too fast), or mushy transients (too slow)
- Bad release settings: Pumping/breathing artifacts, or sustain getting choked

**DRUM COMPRESSION ISSUES:**
- Over-compressed kick/snare: No punch, sounds flat and lifeless, loses impact
  - Signs: Low crest factor, weak transient peaks relative to body
  - Fix: Slower attack (10-30ms) to let transient through, or parallel compression
- Pumping drums: Obvious gain reduction artifacts, unnatural volume swings
  - Signs: Audible "breathing", inconsistent groove feel
  - Fix: Faster release, lower ratio, or use multiband compression

**BASS COMPRESSION ISSUES:**
- Over-compressed bass: No dynamics, sounds static and boring
- Under-compressed bass: Notes vary wildly in volume, some notes too loud/quiet
  - Fix: Use multiband or split-band compression to control low end separately

**MIX BUS COMPRESSION ISSUES:**
- Over-limited master: No life, fatiguing to listen to, distortion artifacts
  - Signs: Very low loudness range (<4 LU), high integrated LUFS but sounds bad
- Pumping mix bus: Kick or bass triggering audible compression artifacts
  - Fix: Use multiband limiting, sidechain filter on compressor, or reduce compression

**INSTRUMENT-SPECIFIC SIGNS TO DETECT:**
- Vocals with low crest factor (< 6dB) = likely over-compressed
- Drums with crest factor < 10dB = likely squashed, lost punch
- Low dynamic range (< 6dB) overall = over-processed, fatiguing
- High dynamic range (> 14dB) for pop/EDM = under-compressed, inconsistent

When you detect compression issues, provide SPECIFIC fixes with exact settings.

Example of the level of detail required:
BAD: "Use stereo imaging on drums and synths"
GOOD: "Apply stereo width to synths, melodies, and FX (NOT kick, bass, or lead vocal). Option A - Mid/Side EQ (FabFilter Pro-Q 3): Sides channel +1.5 dB @ 4.5 kHz (Q 1.2), +1.0 dB @ 10 kHz (Q 0.8). Option B - Stereo Imager (Ozone Imager): Band 2 kHz+, Width +20%, Stereoize OFF. Target: Stereo width ≥ 0.30, Correlation ≥ 0.70"

BAD: "Vocals sound over-compressed"
GOOD: "Vocals are over-compressed - squashed dynamics with pumping artifacts. APPLY TO: Lead vocal. Option A - Reduce existing compression: Lower ratio to 2:1, raise threshold by 6dB, set attack 15-25ms, release 100-150ms. Option B - Replace with parallel compression: Dry vocal at 0dB, compressed bus (8:1, fast attack, medium release) blended at -10dB. Option C - Use multiband: Only compress 2-4kHz range to control harshness, leave dynamics elsewhere. TARGET: Vocal crest factor 8-12dB, natural breathing dynamics."

BAD: "Reduce low-mid buildup"
GOOD: "Cut muddy frequencies on bass, guitars, and synth pads (NOT kick or sub bass). Option A - Surgical EQ: Cut -3 to -4 dB @ 250-350 Hz (Q 2.0). Option B - Dynamic EQ: FabFilter Pro-Q 3 dynamic band @ 300 Hz, threshold -20 dB, ratio 2:1. Option C - Multiband compression: 200-400 Hz band, ratio 3:1, attack 30ms, release 150ms. Target: Low-mid energy reduced by 2-3 dB relative to mix"

Always provide your response as valid JSON matching the required schema."""


def build_analysis_prompt(
    metrics: AudioMetrics,
    genre: str,
    reference: Optional[str] = None,
    user_plugins: Optional[List[Any]] = None,
    user_history: Optional[List[Any]] = None,
    reference_metrics: Optional[AudioMetrics] = None,
) -> str:
    """Build the analysis prompt for Claude."""
    prompt = f"""Analyze this audio mix and provide detailed feedback.

## Audio Metrics

**Loudness:**
- Integrated LUFS: {metrics.integrated_lufs}
- Short-term LUFS max: {metrics.short_term_lufs_max}
- True Peak: {metrics.true_peak_dbfs} dBFS
- Loudness Range (LRA): {metrics.loudness_range} LU

**Spectrum Energy (dB relative to total):**
- Sub (20-60Hz): {metrics.spectrum.get('sub', 'N/A')}
- Low (60-250Hz): {metrics.spectrum.get('low', 'N/A')}
- Low-Mid (250-500Hz): {metrics.spectrum.get('low_mid', 'N/A')}
- Mid (500-2kHz): {metrics.spectrum.get('mid', 'N/A')}
- Presence (2-6kHz): {metrics.spectrum.get('presence', 'N/A')}
- Air (6-20kHz): {metrics.spectrum.get('air', 'N/A')}

**Stereo Field:**
- Correlation: {metrics.stereo_correlation}
- Width: {metrics.stereo_width}

**Dynamics:**
- Crest Factor: {metrics.crest_factor_db} dB
- RMS Level: {metrics.rms_db} dB
- Dynamic Range: {metrics.dynamic_range_db} dB

**Transients/Rhythm:**
- Onset Density: {metrics.onset_density} onsets/sec
- Estimated Tempo: {metrics.estimated_tempo} BPM

**File Info:**
- Duration: {metrics.duration_seconds} seconds
- Sample Rate: {metrics.sample_rate} Hz

## Context
- **Genre:** {genre}
"""

    if reference:
        prompt += f"- **Reference/Notes:** {reference}\n"

    # Add reference track comparison if available
    if reference_metrics:
        prompt += """
## Reference Track Comparison

A reference track has been uploaded for comparison. Here are the metrics side-by-side:

| Metric | Your Mix | Reference | Delta |
|--------|----------|-----------|-------|
"""
        # Loudness comparison
        lufs_delta = metrics.integrated_lufs - reference_metrics.integrated_lufs
        prompt += f"| Integrated LUFS | {metrics.integrated_lufs} | {reference_metrics.integrated_lufs} | {lufs_delta:+.1f} |\n"

        peak_delta = metrics.true_peak_dbfs - reference_metrics.true_peak_dbfs
        prompt += f"| True Peak | {metrics.true_peak_dbfs} dBFS | {reference_metrics.true_peak_dbfs} dBFS | {peak_delta:+.1f} |\n"

        lra_delta = metrics.loudness_range - reference_metrics.loudness_range
        prompt += f"| Loudness Range | {metrics.loudness_range} LU | {reference_metrics.loudness_range} LU | {lra_delta:+.1f} |\n"

        # Dynamics comparison
        dr_delta = metrics.dynamic_range_db - reference_metrics.dynamic_range_db
        prompt += f"| Dynamic Range | {metrics.dynamic_range_db} dB | {reference_metrics.dynamic_range_db} dB | {dr_delta:+.1f} |\n"

        crest_delta = metrics.crest_factor_db - reference_metrics.crest_factor_db
        prompt += f"| Crest Factor | {metrics.crest_factor_db} dB | {reference_metrics.crest_factor_db} dB | {crest_delta:+.1f} |\n"

        # Stereo comparison
        width_delta = metrics.stereo_width - reference_metrics.stereo_width
        prompt += f"| Stereo Width | {metrics.stereo_width} | {reference_metrics.stereo_width} | {width_delta:+.3f} |\n"

        corr_delta = metrics.stereo_correlation - reference_metrics.stereo_correlation
        prompt += f"| Stereo Correlation | {metrics.stereo_correlation} | {reference_metrics.stereo_correlation} | {corr_delta:+.3f} |\n"

        prompt += """
**Spectrum Comparison (dB relative to total):**

| Band | Your Mix | Reference | Delta |
|------|----------|-----------|-------|
"""
        for band in ['sub', 'low', 'low_mid', 'mid', 'presence', 'air']:
            user_val = metrics.spectrum.get(band, 0)
            ref_val = reference_metrics.spectrum.get(band, 0)
            delta = user_val - ref_val
            prompt += f"| {band.replace('_', ' ').title()} | {user_val} | {ref_val} | {delta:+.1f} |\n"

        prompt += """
IMPORTANT: When analyzing, pay special attention to the reference comparison:
1. Identify the most significant deltas and explain their impact
2. Provide specific suggestions to bring the mix closer to the reference characteristics
3. Note where the user's mix actually excels compared to the reference
4. Include reference-specific diagnoses with actionable steps to match the reference
"""

    # Add user's plugins if available
    if user_plugins and len(user_plugins) > 0:
        prompt += """
## User's Available Plugins

The user has the following plugins. PRIORITIZE recommending these with specific settings:
"""
        # Group plugins by category
        plugins_by_category = {}
        for plugin in user_plugins:
            cat = plugin.category
            if cat not in plugins_by_category:
                plugins_by_category[cat] = []
            plugin_str = plugin.name
            if plugin.manufacturer:
                plugin_str += f" ({plugin.manufacturer})"
            plugins_by_category[cat].append(plugin_str)

        for category, plugins in sorted(plugins_by_category.items()):
            prompt += f"\n**{category}:** {', '.join(plugins)}"

        prompt += """

IMPORTANT: In your suggestions and quick_wins, recommend specific plugins from this list with actual parameter values. For example:
- "Use FabFilter Pro-Q 3: Add a high-shelf at 10kHz, +2dB, Q=0.7"
- "Use Waves CLA-76: Input 5, Output 3, Attack fastest, Release 5"
- "Use Valhalla Room: Decay 1.8s, Mix 18%, Pre-delay 25ms"
"""

    # Add user's history if available
    if user_history and len(user_history) > 0:
        prompt += """
## User's Recent Mix History

"""
        for i, entry in enumerate(user_history, 1):
            created = entry.created_at.strftime("%Y-%m-%d") if entry.created_at else "Unknown"
            prompt += f"**Mix {i}** ({entry.genre}, {created}): Score {entry.health_score}/100"
            if entry.summary:
                # Truncate summary if too long
                summary = entry.summary[:100] + "..." if len(entry.summary) > 100 else entry.summary
                prompt += f' - "{summary}"'
            prompt += "\n"

        prompt += """
Consider: Are there recurring issues across these mixes? Has the user been improving? Mention any patterns you notice and provide guidance on recurring problems.
"""

    # Build JSON schema based on whether reference was provided
    if reference_metrics:
        prompt += """
## Required Output Format

Respond with a JSON object containing:

```json
{
  "engineer_summary": "A conversational, direct assessment like a real mix engineer would give. Be specific about the 1-2 most impactful fixes. Example: 'Your mix has strong energy and solid low end, but the vocal harshness around 3-4kHz and low-mid buildup at 250-400Hz are preventing it from sounding professional. A 3dB cut at 3.5kHz on vocals and cleaning up the 300Hz region will get you 80% of the way there.'",
  "summary": "2-3 sentence technical overview of the mix's current state with specific metric references",
  "health_score": 75,
  "diagnoses": [
    {
      "category": "Loudness|Spectrum|Stereo|Dynamics|Transients|Compression|Reference",
      "severity": "critical|warning|info",
      "priority": 1,
      "issue": "Specific problem description with measured values (e.g., 'Low-mid buildup: 250-400Hz region is +3dB above genre target')",
      "what_it_affects": "What elements/qualities this impacts (e.g., 'Lead vocal intelligibility, snare definition, perceived energy')",
      "likely_causes": ["Cause 1 (e.g., 'Vocal masked by instruments in 2-5 kHz')", "Cause 2 (e.g., 'Over-compression reducing consonant attack')"],
      "suggestion": "RECIPE-STYLE FIX with multiple options. Format: 'APPLY TO: [elements]. DO NOT APPLY TO: [elements]. OPTION A - [Technique] ([Plugin]): [exact settings with frequencies, dB, Q values, times]. OPTION B - [Alternative]: [settings]. TARGET: [specific metric goal]'",
      "dont_do": "Common mistake to avoid (e.g., 'Boosting presence on the mix bus before fixing masking')",
      "done_when": "How to know when to stop (e.g., 'The lead stays clear at low playback volume without sounding sharp')",
      "confidence": "high|medium|low",
      "affected_frequencies": "e.g., '250-400Hz'"
    }
  ],
  "quick_wins": [
    {
      "action": "Specific action with exact settings. Example: 'High-pass filter at 30Hz (24dB/oct) on master bus to clean up sub rumble'",
      "impact": "high|medium|low",
      "plugin_suggestions": ["Specific Plugin Name 1", "Alternative Plugin 2"]
    }
  ],
  "mastering_ready": false,
  "mastering_notes": "Specific items to address before mastering with target values",
  "next_session_focus": [
    "First priority fix (e.g., 'Fix vocal/instrument masking in 2-5 kHz')",
    "Second priority (e.g., 'Balance low-mid body vs boxiness')",
    "Third priority (e.g., 'Add depth without widening the lead')"
  ],
  "reference_comparison": {
    "overall_similarity": 75,
    "key_differences": [
      "Reference is 2dB louder in presence range (2-6kHz) - boost presence shelf +1.5dB @ 4kHz",
      "Your mix has wider stereo image by 0.15 - consider narrowing synths slightly",
      "Reference has 2dB more dynamic range - reduce limiter input by 1-2dB"
    ]
  }
}
```

CRITICAL REQUIREMENTS FOR DIAGNOSES:
1. Each diagnosis MUST include what_it_affects, likely_causes, dont_do, and done_when
2. ALWAYS specify APPLY TO and DO NOT APPLY TO elements in suggestion
3. ALWAYS provide at least 2 OPTIONS (different approaches/plugins)
4. ALWAYS include EXACT values: frequencies in Hz/kHz, dB amounts, Q values, ms for times, ratios
5. ALWAYS include a TARGET metric or result
6. Reference specific plugins by name (Pro-Q 3, SSL Channel, LA-2A, etc.)
7. Set confidence based on how certain you are: "high" (common issue), "medium" (context-dependent), "low" (stylistic choice)

PRIORITY LEVELS (fixes in this order):
- 1 = CRITICAL: Clarity & masking issues, major frequency problems
- 2 = IMPORTANT: Dynamics, stereo, tonal balance improvements
- 3 = OPTIONAL: Polish items - air, final loudness tweaks

Include at least 1-2 diagnoses with category "Reference" addressing reference track matching.
Provide 3-7 total diagnoses prioritized by priority level.
Provide 2-4 quick wins with immediate impact.
Provide 2-4 next_session_focus items as clear action items.

IMPORTANT: Respond ONLY with the JSON object, no additional text or markdown formatting."""
    else:
        prompt += """
## Required Output Format

Respond with a JSON object containing:

```json
{
  "engineer_summary": "A conversational, direct assessment like a real mix engineer would give. Be specific about the 1-2 most impactful fixes. Example: 'Your mix has strong energy and solid low end, but the vocal harshness around 3-4kHz and low-mid buildup at 250-400Hz are preventing it from sounding professional. A 3dB cut at 3.5kHz on vocals and cleaning up the 300Hz region will get you 80% of the way there.'",
  "summary": "2-3 sentence technical overview of the mix's current state with specific metric references",
  "health_score": 75,
  "diagnoses": [
    {
      "category": "Loudness|Spectrum|Stereo|Dynamics|Transients|Compression",
      "severity": "critical|warning|info",
      "priority": 1,
      "issue": "Specific problem description with measured values (e.g., 'Low-mid buildup: 250-400Hz region is +3dB above genre target')",
      "what_it_affects": "What elements/qualities this impacts (e.g., 'Lead vocal intelligibility, snare definition, perceived energy')",
      "likely_causes": ["Cause 1 (e.g., 'Vocal masked by instruments in 2-5 kHz')", "Cause 2 (e.g., 'Over-compression reducing consonant attack')"],
      "suggestion": "RECIPE-STYLE FIX with multiple options. Format: 'APPLY TO: [elements]. DO NOT APPLY TO: [elements]. OPTION A - [Technique] ([Plugin]): [exact settings with frequencies, dB, Q values, times]. OPTION B - [Alternative]: [settings]. TARGET: [specific metric goal]'",
      "dont_do": "Common mistake to avoid (e.g., 'Boosting presence on the mix bus before fixing masking')",
      "done_when": "How to know when to stop (e.g., 'The lead stays clear at low playback volume without sounding sharp')",
      "confidence": "high|medium|low",
      "affected_frequencies": "e.g., '250-400Hz'"
    }
  ],
  "quick_wins": [
    {
      "action": "Specific action with exact settings. Example: 'High-pass filter at 30Hz (24dB/oct) on master bus to clean up sub rumble'",
      "impact": "high|medium|low",
      "plugin_suggestions": ["Specific Plugin Name 1", "Alternative Plugin 2"]
    }
  ],
  "mastering_ready": false,
  "mastering_notes": "Specific items to address before mastering with target values",
  "next_session_focus": [
    "First priority fix (e.g., 'Fix vocal/instrument masking in 2-5 kHz')",
    "Second priority (e.g., 'Balance low-mid body vs boxiness')",
    "Third priority (e.g., 'Add depth without widening the lead')"
  ]
}
```

CRITICAL REQUIREMENTS FOR DIAGNOSES:
1. Each diagnosis MUST include what_it_affects, likely_causes, dont_do, and done_when
2. ALWAYS specify APPLY TO and DO NOT APPLY TO elements in suggestion
3. ALWAYS provide at least 2 OPTIONS (different approaches/plugins)
4. ALWAYS include EXACT values: frequencies in Hz/kHz, dB amounts, Q values, ms for times, ratios
5. ALWAYS include a TARGET metric or result
6. Reference specific plugins by name (Pro-Q 3, SSL Channel, LA-2A, OTT, Valhalla Room, etc.)
7. Set confidence based on how certain you are: "high" (common issue), "medium" (context-dependent), "low" (stylistic choice)

PRIORITY LEVELS (fixes in this order):
- 1 = CRITICAL: Clarity & masking issues, major frequency problems
- 2 = IMPORTANT: Dynamics, stereo, tonal balance improvements
- 3 = OPTIONAL: Polish items - air, final loudness tweaks

Provide 3-7 diagnoses prioritized by priority level.
Provide 2-4 quick wins with immediate impact.
Provide 2-4 next_session_focus items as clear action items.

IMPORTANT: Respond ONLY with the JSON object, no additional text or markdown formatting."""

    return prompt


def analyze_with_claude(
    metrics: AudioMetrics,
    genre: str,
    reference: Optional[str] = None,
    user_plugins: Optional[List[Any]] = None,
    user_history: Optional[List[Any]] = None,
    reference_metrics: Optional[AudioMetrics] = None,
) -> AnalysisResult:
    """Send metrics to Claude and parse the response."""
    client = Anthropic()

    prompt = build_analysis_prompt(metrics, genre, reference, user_plugins, user_history, reference_metrics)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,  # Increased for detailed recipe-style feedback
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Extract response text
    response_text = message.content[0].text

    # Parse JSON response
    # Handle potential markdown code blocks
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0]
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0]

    # Clean up common JSON issues
    response_text = response_text.strip()

    # Try to parse, with fallback error handling
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as e:
        # Try to fix common issues like unescaped newlines in strings
        import re
        # Replace unescaped newlines within strings
        fixed_text = re.sub(r'(?<!\\)\n(?=(?:[^"]*"[^"]*")*[^"]*"[^"]*$)', '\\n', response_text)
        try:
            data = json.loads(fixed_text)
        except json.JSONDecodeError:
            # If still failing, raise with original error
            raise Exception(f"Failed to parse AI response as JSON: {str(e)}")

    # Build result object
    diagnoses = [
        Diagnosis(
            category=d["category"],
            severity=d["severity"],
            priority=d.get("priority", 2),  # Default to IMPORTANT if not specified
            issue=d["issue"],
            suggestion=d["suggestion"],
            affected_frequencies=d.get("affected_frequencies"),
            # New engineering-focused fields
            what_it_affects=d.get("what_it_affects"),
            likely_causes=d.get("likely_causes"),
            dont_do=d.get("dont_do"),
            done_when=d.get("done_when"),
            confidence=d.get("confidence", "high"),
        )
        for d in data["diagnoses"]
    ]

    quick_wins = [
        QuickWin(
            action=q["action"],
            impact=q["impact"],
            plugin_suggestions=q.get("plugin_suggestions"),
        )
        for q in data["quick_wins"]
    ]

    # Build reference comparison if provided
    reference_comparison = None
    if reference_metrics:
        # Calculate deltas for comparison object
        lufs_delta = round(metrics.integrated_lufs - reference_metrics.integrated_lufs, 1)
        dynamics_delta = round(metrics.dynamic_range_db - reference_metrics.dynamic_range_db, 1)
        width_delta = round(metrics.stereo_width - reference_metrics.stereo_width, 3)

        spectrum_comparison = {}
        for band in ['sub', 'low', 'low_mid', 'mid', 'presence', 'air']:
            user_val = metrics.spectrum.get(band, 0)
            ref_val = reference_metrics.spectrum.get(band, 0)
            spectrum_comparison[band] = round(user_val - ref_val, 1)

        # Get key differences from Claude's response if available
        ref_data = data.get("reference_comparison", {})
        key_differences = ref_data.get("key_differences", [])
        overall_similarity = ref_data.get("overall_similarity", 50)

        reference_comparison = ReferenceComparison(
            loudness_delta=lufs_delta,
            dynamics_delta=dynamics_delta,
            stereo_width_delta=width_delta,
            spectrum_comparison=spectrum_comparison,
            overall_similarity=overall_similarity,
            key_differences=key_differences,
        )

    return AnalysisResult(
        engineer_summary=data.get("engineer_summary", data["summary"]),  # Fallback to summary if not provided
        summary=data["summary"],
        health_score=data["health_score"],
        diagnoses=diagnoses,
        quick_wins=quick_wins,
        mastering_ready=data["mastering_ready"],
        mastering_notes=data["mastering_notes"],
        metrics=metrics,
        reference_comparison=reference_comparison,
        next_session_focus=data.get("next_session_focus"),
    )


# ===== Stems Analysis =====

STEM_LABELS = {
    'kick': 'Kick',
    'snare_hats': 'Snare/Hats',
    'percussion': 'Percussion',
    'bass': 'Bass',
    'vocals': 'Vocals',
    'lead': 'Lead Synths/Instruments',
    'pads': 'Pads/Chords',
    'guitars': 'Guitars',
    'fx': 'FX/Ambience',
    'other': 'Other',
}


def build_stems_analysis_prompt(
    stems: dict,  # Dict[stem_type, AudioMetrics]
    interactions: dict,  # From analyze_stem_interactions
    genre: str,
    user_plugins: Optional[List[Any]] = None,
) -> str:
    """Build the analysis prompt for stems analysis."""
    prompt = f"""Analyze these individual stems from a mix and provide detailed feedback.

## Genre: {genre}

## Stem Metrics

"""
    # Add each stem's metrics
    for stem_type, metrics in stems.items():
        label = STEM_LABELS.get(stem_type, stem_type)
        prompt += f"""### {label} Stem

**Loudness:**
- Integrated LUFS: {metrics.integrated_lufs}
- True Peak: {metrics.true_peak_dbfs} dBFS

**Spectrum Energy (dB relative to total):**
- Sub (20-60Hz): {metrics.spectrum.get('sub', 'N/A')}
- Low (60-250Hz): {metrics.spectrum.get('low', 'N/A')}
- Low-Mid (250-500Hz): {metrics.spectrum.get('low_mid', 'N/A')}
- Mid (500-2kHz): {metrics.spectrum.get('mid', 'N/A')}
- Presence (2-6kHz): {metrics.spectrum.get('presence', 'N/A')}
- Air (6-20kHz): {metrics.spectrum.get('air', 'N/A')}

**Stereo:**
- Correlation: {metrics.stereo_correlation}
- Width: {metrics.stereo_width}

**Dynamics:**
- Crest Factor: {metrics.crest_factor_db} dB
- Dynamic Range: {metrics.dynamic_range_db} dB

"""

    # Add interaction analysis
    prompt += """## Detected Interactions

"""
    if interactions.get('masking_interactions'):
        prompt += "**Frequency Masking:**\n"
        for mi in interactions['masking_interactions']:
            label_a = STEM_LABELS.get(mi['stem_a'], mi['stem_a'])
            label_b = STEM_LABELS.get(mi['stem_b'], mi['stem_b'])
            bands = ', '.join(mi['bands'])
            prompt += f"- {label_a} and {label_b} compete in: {bands} ({mi['severity']})\n"
        prompt += "\n"

    if interactions.get('phase_issues'):
        prompt += "**Phase Concerns:**\n"
        for pi in interactions['phase_issues']:
            label = STEM_LABELS.get(pi['stem'], pi['stem'])
            prompt += f"- {label}: Low stereo correlation ({pi['correlation']:.2f})\n"
        prompt += "\n"

    if interactions.get('balance_issues'):
        prompt += "**Level Balance:**\n"
        for bi in interactions['balance_issues']:
            label = STEM_LABELS.get(bi['stem'], bi['stem'])
            direction = "louder" if bi['delta_from_avg'] > 0 else "quieter"
            prompt += f"- {label}: {abs(bi['delta_from_avg']):.1f}dB {direction} than average\n"
        prompt += "\n"

    if not interactions.get('masking_interactions') and not interactions.get('phase_issues') and not interactions.get('balance_issues'):
        prompt += "No major interaction issues detected automatically.\n\n"

    # Add user's plugins if available
    if user_plugins and len(user_plugins) > 0:
        prompt += """
## User's Available Plugins

PRIORITIZE recommending these with specific settings:
"""
        plugins_by_category = {}
        for plugin in user_plugins:
            cat = plugin.category
            if cat not in plugins_by_category:
                plugins_by_category[cat] = []
            plugin_str = plugin.name
            if plugin.manufacturer:
                plugin_str += f" ({plugin.manufacturer})"
            plugins_by_category[cat].append(plugin_str)

        for category, plugins in sorted(plugins_by_category.items()):
            prompt += f"\n**{category}:** {', '.join(plugins)}"
        prompt += "\n"

    prompt += """
## Required Output Format

Respond with a JSON object containing:

```json
{
  "engineer_summary": "A conversational, direct assessment with specific fix recommendations. Example: 'Your kick and bass are fighting each other in the 60-100Hz range - that's why the low end feels muddy. Sidechain the bass to the kick (attack 5ms, release 100ms, ratio 4:1) and cut 80Hz on the bass by -3dB. You'll immediately hear more punch and clarity.'",
  "summary": "2-3 sentence technical overview of the stems with specific metric references",
  "overall_health_score": 75,
  "stem_analyses": [
    {
      "stem_type": "kick",
      "issues": [
        {
          "category": "Loudness|Spectrum|Stereo|Dynamics|Transients|Compression",
          "severity": "critical|warning|info",
          "priority": 1,
          "issue": "Specific problem with measured values (e.g., 'Kick lacks punch: transient peak only +6dB above body, should be +10-12dB')",
          "suggestion": "RECIPE-STYLE FIX: 'OPTION A - Transient Shaper (Smack Attack): Attack +30%, Sustain -20%. OPTION B - Parallel Compression: Ratio 8:1, Attack 0.5ms, Release 50ms, blend 30%. OPTION C - EQ boost: +2dB @ 3-4kHz (Q 2.0) for beater click. TARGET: Transient peak +10-12dB above body'",
          "affected_frequencies": "e.g., '60-100Hz'"
        }
      ]
    }
  ],
  "interactions": [
    {
      "stem_a": "kick",
      "stem_b": "bass",
      "interaction_type": "frequency_masking|phase_issue|balance",
      "severity": "critical|warning|info",
      "description": "Specific interaction with measured overlap (e.g., 'Both kick and bass peak at 80Hz with combined energy +4dB above target')",
      "suggestion": "RECIPE-STYLE FIX: 'OPTION A - Sidechain Compression: Compressor on bass, kick as trigger, ratio 4:1, attack 5ms, release 100ms, threshold -20dB. OPTION B - Frequency splitting: High-pass bass at 80Hz, let kick own sub. OPTION C - Dynamic EQ on bass: -4dB @ 60-100Hz triggered by kick. TARGET: Clean separation, no pumping artifacts'",
      "affected_frequencies": "60-100Hz"
    }
  ],
  "quick_wins": [
    {
      "action": "Specific action with exact settings. Example: 'Sidechain bass to kick using Trackspacer or compressor (ratio 3:1, attack 5ms, release 80ms) for instant low-end clarity'",
      "impact": "high|medium|low",
      "plugin_suggestions": ["Trackspacer", "FabFilter Pro-C 3"]
    }
  ],
  "mastering_ready": false,
  "mastering_notes": "Specific items with target values (e.g., 'Fix kick/bass masking first - combined low end is +3dB hot')"
}
```

CRITICAL REQUIREMENTS FOR ALL SUGGESTIONS:
1. ALWAYS provide MULTIPLE OPTIONS (A, B, C) with different approaches
2. ALWAYS include EXACT values: frequencies in Hz/kHz, dB amounts, Q values, attack/release in ms, ratios, percentages
3. ALWAYS include a TARGET metric or perceptual result
4. Reference specific plugins by name (Pro-Q 3, Pro-C 3, Trackspacer, Smack Attack, etc.)
5. For interactions, explain the FIX for BOTH stems involved

PRIORITY LEVELS:
- 1 = CRITICAL: Must fix - causes obvious mix problems
- 2 = IMPORTANT: Significant improvement when fixed
- 3 = OPTIONAL: Polish/refinement

Provide detailed issues for EACH stem with problems.
Include ALL significant stem interactions with recipe-style fixes.

IMPORTANT: Respond ONLY with the JSON object, no additional text or markdown formatting."""

    return prompt


def analyze_stems_with_claude(
    stems: dict,  # Dict[stem_type, AudioMetrics]
    interactions: dict,  # From analyze_stem_interactions
    genre: str,
    user_plugins: Optional[List[Any]] = None,
) -> StemsAnalysisResult:
    """Send stems metrics to Claude and parse the response."""
    client = Anthropic()

    prompt = build_stems_analysis_prompt(stems, interactions, genre, user_plugins)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=5000,  # Increased for detailed recipe-style feedback
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Extract response text
    response_text = message.content[0].text

    # Parse JSON response
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0]
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0]

    # Clean up common JSON issues
    response_text = response_text.strip()

    # Try to parse, with fallback error handling
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as e:
        # Try to fix common issues like unescaped newlines in strings
        import re
        fixed_text = re.sub(r'(?<!\\)\n(?=(?:[^"]*"[^"]*")*[^"]*"[^"]*$)', '\\n', response_text)
        try:
            data = json.loads(fixed_text)
        except json.JSONDecodeError:
            raise Exception(f"Failed to parse AI response as JSON: {str(e)}")

    # Build stem analyses
    stem_analyses = []
    for sa in data.get("stem_analyses", []):
        issues = [
            Diagnosis(
                category=d["category"],
                severity=d["severity"],
                priority=d.get("priority", 2),  # Default to IMPORTANT if not specified
                issue=d["issue"],
                suggestion=d["suggestion"],
                affected_frequencies=d.get("affected_frequencies"),
            )
            for d in sa.get("issues", [])
        ]
        stem_analyses.append(StemAnalysis(
            stem_type=sa["stem_type"],
            metrics=stems[sa["stem_type"]],
            issues=issues,
        ))

    # Build interactions
    stem_interactions = [
        StemInteraction(
            stem_a=i["stem_a"],
            stem_b=i["stem_b"],
            interaction_type=i["interaction_type"],
            severity=i["severity"],
            description=i["description"],
            suggestion=i["suggestion"],
            affected_frequencies=i.get("affected_frequencies"),
        )
        for i in data.get("interactions", [])
    ]

    # Build quick wins
    quick_wins = [
        QuickWin(
            action=q["action"],
            impact=q["impact"],
            plugin_suggestions=q.get("plugin_suggestions"),
        )
        for q in data.get("quick_wins", [])
    ]

    return StemsAnalysisResult(
        engineer_summary=data.get("engineer_summary", data["summary"]),  # Fallback to summary if not provided
        summary=data["summary"],
        overall_health_score=data["overall_health_score"],
        stem_analyses=stem_analyses,
        interactions=stem_interactions,
        quick_wins=quick_wins,
        mastering_ready=data["mastering_ready"],
        mastering_notes=data["mastering_notes"],
    )
