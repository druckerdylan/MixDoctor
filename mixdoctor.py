#!/usr/bin/env python3
"""
MixDoctor - Expert mixing engineer and audio QA analyst
Analyzes mix diagnostics and returns prioritized fix plans.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("Error: anthropic package not installed. Run: pip install anthropic")
    sys.exit(1)

MIXDOCTOR_SYSTEM_PROMPT = """You are "MixDoctor", an expert mixing engineer and audio QA analyst with 20+ years of experience across genres.
Your job: analyze mix diagnostics and return a prioritized, step-by-step fix plan that a producer can execute in any DAW.

IMPORTANT RULES
- Be objective and specific. No fluff. No vague advice like "use your ears."
- Prefer actionable moves: exact frequency ranges, dB ranges, compressor behavior, and routing suggestions.
- When unsure, state the uncertainty and give 2–3 plausible causes with tests to confirm.
- Do not assume access to the audio file; you only have the data provided.
- Output must follow the exact JSON schema specified. Do not include any extra keys.

ANALYSIS FRAMEWORK
When analyzing mix data, consider these domains in order:

1. GAIN STAGING
   - Check peak/RMS relationships
   - Identify headroom issues
   - Flag clipping or limiting artifacts

2. CLEANUP
   - Resonances that need surgical EQ
   - Mud accumulation (150-350Hz)
   - Harsh frequencies (2-5kHz)
   - Sibilance issues (5-9kHz)

3. DYNAMICS
   - Transient preservation (kick/snare punch)
   - Compression balance
   - LRA (loudness range) appropriateness for genre

4. TONE / FREQUENCY BALANCE
   - Spectrum distribution vs genre norms
   - Tonal balance deviations
   - Low-end clarity and definition

5. SPACE
   - Depth and dimension
   - Reverb/delay balance
   - Front-to-back placement

6. STEREO IMAGE
   - Width appropriateness
   - Mono compatibility
   - Phase issues
   - Mid/side balance

7. LOUDNESS / MASTERING PREP
   - LUFS targets for platform
   - True peak headroom
   - Crest factor health

PRIORITY LEVELS
- P0: Critical - mix sounds broken, must fix before any listening
- P1: Important - noticeable issues that affect commercial viability
- P2: Polish - refinements that separate good from great

GENRE CONTEXT
Adjust expectations based on genre:
- EDM/Hip-hop: Aggressive limiting OK, -6 to -8 LUFS, heavy sub
- Rock/Metal: -9 to -11 LUFS, mid-forward, transient punch
- Jazz/Classical: -14 to -18 LUFS, high LRA, natural dynamics
- Pop: -8 to -10 LUFS, vocal-forward, polished highs

PLATFORM TARGETS
- Spotify: -14 LUFS integrated (normalized), -1dB true peak
- Apple Music: -16 LUFS integrated, -1dB true peak
- YouTube: -14 LUFS integrated, -1dB true peak
- Club: -6 to -8 LUFS, -0.3dB true peak, mono sub below 80Hz

OUTPUT FORMAT
You must respond with valid JSON only. No markdown, no explanation outside the JSON.
"""

OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["summary", "diagnoses", "quick_wins", "mastering_readiness"],
    "properties": {
        "summary": {
            "type": "object",
            "required": ["overall_health", "critical_issues", "main_focus"],
            "properties": {
                "overall_health": {
                    "type": "string",
                    "enum": ["critical", "poor", "fair", "good", "excellent"]
                },
                "critical_issues": {"type": "integer"},
                "main_focus": {"type": "string"}
            }
        },
        "diagnoses": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "priority", "category", "symptom", "likely_cause", "tests", "actions", "expected_outcome", "risk"],
                "properties": {
                    "id": {"type": "string"},
                    "priority": {"type": "string", "enum": ["P0", "P1", "P2"]},
                    "category": {
                        "type": "string",
                        "enum": ["gain_staging", "cleanup", "dynamics", "tone", "space", "stereo", "loudness"]
                    },
                    "symptom": {"type": "string"},
                    "likely_cause": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "tests": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["step", "target", "operation", "parameters"],
                            "properties": {
                                "step": {"type": "integer"},
                                "target": {"type": "string"},
                                "operation": {"type": "string"},
                                "parameters": {"type": "object"}
                            }
                        }
                    },
                    "expected_outcome": {"type": "string"},
                    "risk": {"type": "string"}
                }
            }
        },
        "quick_wins": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["action", "impact", "time_estimate"],
                "properties": {
                    "action": {"type": "string"},
                    "impact": {"type": "string", "enum": ["high", "medium", "low"]},
                    "time_estimate": {"type": "string"}
                }
            },
            "minItems": 3,
            "maxItems": 6
        },
        "mastering_readiness": {
            "type": "object",
            "required": ["ready", "blockers", "recommendations"],
            "properties": {
                "ready": {"type": "boolean"},
                "blockers": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "recommendations": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "suggested_headroom_db": {"type": "number"}
            }
        }
    }
}


def create_analysis_prompt(mix_data: dict) -> str:
    """Create the analysis prompt from mix data."""
    prompt = f"""Analyze the following mix diagnostic data and provide a complete fix plan.

MIX DIAGNOSTIC DATA:
```json
{json.dumps(mix_data, indent=2)}
```

Respond with a JSON object following this exact schema:
```json
{json.dumps(OUTPUT_SCHEMA, indent=2)}
```

Remember:
- Order diagnoses by priority (P0 first) then by the workflow order (gain staging → cleanup → dynamics → tone → space → stereo → loudness)
- Be specific with frequencies (e.g., "200-350Hz" not "low mids")
- Include dB values where applicable (e.g., "-3dB cut" not "reduce")
- Quick wins should be impactful changes achievable in under 10 minutes each
- For mastering readiness, consider: headroom, dynamic range, tonal balance, phase issues

Respond with valid JSON only."""
    return prompt


def analyze_mix(mix_data: dict, model: str = "claude-sonnet-4-20250514") -> dict:
    """Send mix data to Claude for analysis."""
    client = anthropic.Anthropic()

    message = client.messages.create(
        model=model,
        max_tokens=8192,
        system=MIXDOCTOR_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": create_analysis_prompt(mix_data)}
        ]
    )

    response_text = message.content[0].text

    # Clean up response if wrapped in markdown code blocks
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.startswith("```")]
        response_text = "\n".join(lines)

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        return {
            "error": "Failed to parse response as JSON",
            "details": str(e),
            "raw_response": response_text
        }


def validate_input(data: dict) -> list:
    """Validate input data and return list of warnings."""
    warnings = []

    # Check for minimum useful data
    useful_fields = [
        "genre", "integrated_lufs", "true_peak", "spectrum",
        "resonances", "stereo", "masking", "user_notes"
    ]

    present = [f for f in useful_fields if f in data and data[f]]

    if len(present) < 2:
        warnings.append("Limited data provided - analysis may be less accurate")

    # Validate LUFS if present
    if "integrated_lufs" in data:
        lufs = data["integrated_lufs"]
        if lufs > 0:
            warnings.append(f"Integrated LUFS of {lufs} is positive - this seems incorrect")
        elif lufs < -30:
            warnings.append(f"Integrated LUFS of {lufs} is very quiet - verify measurement")

    # Validate true peak
    if "true_peak" in data:
        tp = data["true_peak"]
        if tp > 0:
            warnings.append(f"True peak of {tp}dB indicates clipping")

    return warnings


def main():
    parser = argparse.ArgumentParser(
        description="MixDoctor - Analyze mix diagnostics and get fix recommendations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mixdoctor input.json
  mixdoctor input.json -o report.json
  mixdoctor input.json --model claude-opus-4-20250514
  cat diagnostics.json | mixdoctor -

Input JSON should contain any available mix metrics:
  - genre, reference_tracks, target_platform
  - integrated_lufs, short_term_lufs, true_peak, lra, crest_factor
  - stems: {drums, bass, vocals, music} with peak_db and rms_db
  - spectrum: {sub, low, low_mid, mid, presence, air} values
  - resonances: [{freq_hz, q, severity, timestamps}]
  - stereo: {correlation, width_by_band, mid_side_balance}
  - masking: ["vocal masked by guitars 2-4k", ...]
  - flags: {sibilance, harshness, mud, phase_issues, mono_compatibility}
  - user_notes: "vocal sounds dull, kick weak on phone"
        """
    )

    parser.add_argument(
        "input",
        type=str,
        help="Input JSON file with mix diagnostics (use '-' for stdin)"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file (default: stdout)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Claude model to use (default: claude-sonnet-4-20250514)"
    )

    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output"
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate input, don't analyze"
    )

    parser.add_argument(
        "--schema",
        action="store_true",
        help="Print output JSON schema and exit"
    )

    args = parser.parse_args()

    # Print schema if requested
    if args.schema:
        print(json.dumps(OUTPUT_SCHEMA, indent=2))
        return 0

    # Read input
    try:
        if args.input == "-":
            mix_data = json.load(sys.stdin)
        else:
            with open(args.input, "r") as f:
                mix_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input: {e}", file=sys.stderr)
        return 1

    # Validate input
    warnings = validate_input(mix_data)
    for w in warnings:
        print(f"Warning: {w}", file=sys.stderr)

    if args.validate_only:
        if warnings:
            print("Validation completed with warnings", file=sys.stderr)
            return 1
        print("Validation passed", file=sys.stderr)
        return 0

    # Analyze
    try:
        result = analyze_mix(mix_data, model=args.model)
    except anthropic.APIConnectionError:
        print("Error: Could not connect to Anthropic API", file=sys.stderr)
        return 1
    except anthropic.AuthenticationError:
        print("Error: Invalid API key. Set ANTHROPIC_API_KEY environment variable.", file=sys.stderr)
        return 1
    except anthropic.APIError as e:
        print(f"Error: API error: {e}", file=sys.stderr)
        return 1

    # Output
    indent = 2 if args.pretty else None
    output_json = json.dumps(result, indent=indent)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
            f.write("\n")
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(output_json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
