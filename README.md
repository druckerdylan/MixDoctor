# MixDoctor

AI-powered audio mix analysis. Available as both a CLI tool and a full-stack web application.

## Web Application

Upload your tracks and receive professional feedback with a modern web interface.

### Features

- **Audio Analysis**: Extracts loudness (LUFS), spectrum balance, stereo field, dynamics, and transient information
- **AI Feedback**: Uses Claude to provide genre-aware, actionable mix suggestions
- **Quick Wins**: Prioritized fixes for maximum impact
- **Mastering Readiness**: Know when your mix is ready for mastering

### Tech Stack

- **Backend**: FastAPI, librosa, pyloudnorm, pydub
- **Frontend**: React, TypeScript, Vite, Tailwind CSS
- **AI**: Claude Sonnet via Anthropic API

### Setup

#### Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Run the server
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`. View docs at `http://localhost:8000/docs`.

#### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Run dev server
npm run dev
```

The frontend will be available at `http://localhost:5173`.

### API Endpoints

#### POST /analyze

Upload an audio file for analysis.

**Form Data:**
- `file`: Audio file (MP3, WAV, M4A, FLAC, AIFF - max 250MB)
- `genre`: Music genre (e.g., "EDM", "Hip Hop", "Rock")
- `reference`: (Optional) Reference track or notes

**Response:**
```json
{
  "summary": "Overview of the mix",
  "health_score": 75,
  "diagnoses": [...],
  "quick_wins": [...],
  "mastering_ready": false,
  "mastering_notes": "...",
  "metrics": {...}
}
```

#### GET /health

Health check endpoint.

### Supported Audio Formats

- MP3
- WAV
- M4A (AAC)
- FLAC
- AIFF

---

## CLI Tool

Expert mixing engineer and audio QA analyst CLI tool. Analyzes mix diagnostics and returns prioritized, actionable fix plans.

### Installation

```bash
cd mixdoctor
pip install -r requirements.txt
chmod +x mixdoctor.py
```

Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### Usage

```bash
# Analyze a mix diagnostic file
./mixdoctor.py input.json

# Pretty-print output
./mixdoctor.py input.json --pretty

# Save to file
./mixdoctor.py input.json -o report.json --pretty

# Read from stdin
cat diagnostics.json | ./mixdoctor.py -

# Use a different model
./mixdoctor.py input.json --model claude-opus-4-20250514

# Validate input only
./mixdoctor.py input.json --validate-only

# Print output schema
./mixdoctor.py --schema
```

### Input Format

Provide a JSON file with any available mix metrics:

```json
{
  "genre": "pop",
  "reference_tracks": ["Artist - Song"],
  "target_platform": "Spotify",

  "integrated_lufs": -10.5,
  "short_term_lufs": -8.2,
  "true_peak": -0.5,
  "lra": 6.2,
  "crest_factor": 9.1,

  "stems": {
    "drums": {"peak_db": -4.2, "rms_db": -15.1},
    "bass": {"peak_db": -6.8, "rms_db": -17.2},
    "vocals": {"peak_db": -3.5, "rms_db": -16.8},
    "music_bus": {"peak_db": -2.1, "rms_db": -13.2}
  },

  "spectrum": {
    "sub_20_60": -6.5,
    "low_60_120": -1.2,
    "low_mid_120_350": 1.8,
    "mid_350_2k": 0.5,
    "presence_2k_6k": -0.8,
    "air_6k_16k": -2.1
  },

  "tonal_balance_deviation": {
    "sub": -1.5,
    "low": 0.8,
    "low_mid": 2.1,
    "mid": 0.2,
    "presence": -1.5,
    "air": -2.0
  },

  "resonances": [
    {"freq_hz": 320, "q": 7.2, "severity": "high", "timestamps": ["1:20", "2:45"]},
    {"freq_hz": 2800, "q": 5.5, "severity": "medium", "timestamps": ["throughout"]}
  ],

  "stereo": {
    "correlation": 0.65,
    "width_by_band": {"low": 0.1, "mid": 0.55, "high": 0.78},
    "mid_side_balance": {
      "low": {"mid": -3.0, "side": -20.0},
      "mid_band": {"mid": -5.0, "side": -8.0},
      "high": {"mid": -7.0, "side": -6.0}
    }
  },

  "transients": {
    "kick_transient_score": 0.72,
    "snare_transient_score": 0.68,
    "limiter_gr_db": -2.5,
    "clipping_flags": false
  },

  "masking": [
    "vocals masked by synths 1-3kHz",
    "bass masked by kick 50-80Hz"
  ],

  "flags": {
    "sibilance": true,
    "harshness": false,
    "mud": true,
    "phase_issues": false,
    "mono_compatibility": "ok"
  },

  "arrangement_notes": [
    {"section": "verse 2", "issue": "energy_drop", "timestamp": "1:45-2:15"}
  ],

  "user_notes": "Vocals sound harsh in the chorus. Bass is boomy."
}
```

### Input Fields Reference

| Field | Description |
|-------|-------------|
| `genre` | Musical genre for context-aware recommendations |
| `reference_tracks` | Reference tracks for tonal comparison |
| `target_platform` | Target platform (Spotify, Apple, YouTube, Club) |
| `integrated_lufs` | Integrated loudness in LUFS |
| `short_term_lufs` | Short-term LUFS measurement |
| `true_peak` | True peak level in dBTP |
| `lra` | Loudness Range in LU |
| `crest_factor` | Peak-to-RMS ratio in dB |
| `stems` | Per-stem peak and RMS measurements |
| `spectrum` | Energy levels per frequency band |
| `tonal_balance_deviation` | Deviation from reference curve per band |
| `resonances` | Detected resonance peaks |
| `stereo` | Stereo correlation, width, and mid/side balance |
| `transients` | Transient scores and limiter activity |
| `masking` | Detected frequency masking issues |
| `flags` | Problem flags (sibilance, harshness, mud, phase) |
| `arrangement_notes` | Arrangement-level issues |
| `user_notes` | Free-form notes about perceived issues |

### Output Format

```json
{
  "summary": {
    "overall_health": "fair",
    "critical_issues": 2,
    "main_focus": "Low-mid cleanup and vocal clarity"
  },
  "diagnoses": [
    {
      "id": "D001",
      "priority": "P0",
      "category": "cleanup",
      "symptom": "Muddy low-mids with +4.2dB deviation at 120-350Hz",
      "likely_cause": "Accumulation from multiple sources (guitars, bass, vocals) competing in the low-mid range",
      "confidence": "high",
      "tests": [
        "Solo each instrument and check 200-350Hz buildup",
        "Use a spectrum analyzer to identify the worst offenders",
        "Bypass bass and check if muddiness reduces"
      ],
      "actions": [
        {
          "step": 1,
          "target": "bass",
          "operation": "high_pass_filter",
          "parameters": {"freq_hz": 40, "slope": "18dB/oct"}
        },
        {
          "step": 2,
          "target": "guitars",
          "operation": "eq_cut",
          "parameters": {"freq_hz": 250, "gain_db": -3, "q": 1.5}
        }
      ],
      "expected_outcome": "Clearer low-end definition, improved kick/bass separation",
      "risk": "Over-cutting can make the mix thin; check on multiple speakers"
    }
  ],
  "quick_wins": [
    {
      "action": "Add 2-3dB high shelf at 8kHz on vocals",
      "impact": "high",
      "time_estimate": "2 minutes"
    }
  ],
  "mastering_readiness": {
    "ready": false,
    "blockers": [
      "True peak too hot (-0.2dB), need -1dB minimum",
      "Low-mid buildup will be exacerbated by limiting"
    ],
    "recommendations": [
      "Fix low-mid issues before mastering",
      "Reduce master output by 1dB for true peak headroom"
    ],
    "suggested_headroom_db": -3.0
  }
}
```

### Priority Levels

- **P0**: Critical - mix sounds broken, must fix before any listening
- **P1**: Important - noticeable issues affecting commercial viability
- **P2**: Polish - refinements that separate good from great

### Platform Targets

| Platform | Integrated LUFS | True Peak |
|----------|-----------------|-----------|
| Spotify | -14 LUFS | -1 dBTP |
| Apple Music | -16 LUFS | -1 dBTP |
| YouTube | -14 LUFS | -1 dBTP |
| Club | -6 to -8 LUFS | -0.3 dBTP |

## License

MIT
