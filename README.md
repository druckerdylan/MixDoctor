# MixDoctor

Upload a mix, get a mastering engineer's read on it — backed by numbers, not vibes.

MixDoctor decodes your audio once, measures it against genre-specific targets, and
returns a scored report: 14 dimensions, timestamped findings, and a prioritised fix
list you can work through with a DAW open.

## Architecture: measure deterministically, reason with AI

The system is split into three layers, and the split is the whole design.

```
audio file
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 1. DSP  (analysis/dsp/*)                                │
│    Decode once → 10 independent measurements,           │
│    + section analysis (always) and stem separation      │
│      (opt-in) — see "Depth passes".                     │
│    Pure numbers. No genre knowledge, no opinions.       │
└─────────────────────────────────────────────────────────┘
    │  Measurements
    ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Detectors  (analysis/detectors.py, targets.py)       │
│    Join measurements against the genre's target windows │
│    → Findings (severity, confidence, evidence, moments) │
│    → 14 DimensionScores → health score + grade          │
│    Deterministic rules. Same input, same verdict.       │
└─────────────────────────────────────────────────────────┘
    │  Findings + Dimensions
    ▼
┌─────────────────────────────────────────────────────────┐
│ 3. AI engineer  (engineer.py, capabilities.py)          │
│    Claude turns findings into prose and prescriptions,  │
│    shaped by the capabilities the producer owns.        │
│    Language and judgement only — every number it cites  │
│    comes from layer 1. OPTIONAL at every point.         │
└─────────────────────────────────────────────────────────┘
    │  MixAnalysis
    ▼
  FastAPI  →  React
```

**Why it matters:** the AI never measures anything. It is handed numbers that were
already computed and is asked to explain and prioritise them. That means the model
cannot hallucinate a loudness figure or invent a resonance — and if the API key is
missing, rate-limited, or the model declines, the report still ships. Every finding
carries a deterministic `detail` sentence built from the measurements, so a report
without the AI layer is *worse prose*, never *wrong data*.

Three invariants hold this together, enforced in `analysis/engine.py`:

1. **Decode once.** `core.load_audio` resamples to 48 kHz and hands the same buffer
   to all ten measurements. Decoding per measurement would dominate runtime.
2. **Never raise on valid audio.** Every measurement runs under `_stage`, which logs
   a failure, records a warning, and substitutes a neutral result. A vocal detector
   crashing must not cost you your clipping report.
3. **Nothing leaves that JSON cannot carry.** `_sanitize` walks the payload and
   replaces NaN/±inf before the model is re-validated. One NaN in a series field is
   a 500 on the endpoint and a blank screen in the browser.

### The 14 dimensions

Each is scored 0–100 against the selected genre's targets. The weight is how much
that dimension can move the overall health score — "how much does this ruin the
record", not "how hard is it to fix".

| # | Dimension | Label | Weight | What it measures |
|---|-----------|-------|--------|------------------|
| 1 | `clipping` | Clipping & True Peak | 1.00 | Flat-topped runs, inter-sample overs, distortion residual |
| 2 | `phase` | Phase & Mono Compatibility | 1.00 | Correlation, polarity, per-band mono fold-down loss |
| 3 | `loudness` | Loudness | 0.80 | Integrated LUFS, LRA, true peak, PLR |
| 4 | `limiter` | Limiter Behaviour | 0.80 | PSR percentiles — how hard the ceiling is being worked |
| 5 | `dynamic_range` | Dynamic Range | 0.70 | Crest factor, macro/micro dynamics, DR value |
| 6 | `compression` | Compression | 0.55 | Pumping index and rate, gain-reduction estimate |
| 7 | `frequency_balance` | Frequency Balance | 0.85 | 9 macro bands + 31 third-octave bands vs the genre curve |
| 8 | `mud` | Mud & Low-Mid Buildup | 0.75 | 150–400 Hz against the low end and against 1–3 kHz |
| 9 | `harshness` | Harshness & Sibilance | 0.65 | 2–5 kHz edge, 5–9 kHz sibilance, Zwicker sharpness |
| 10 | `low_end` | Kick / 808 Relationship | 0.75 | Kick/bass fundamental collision, sidechain, sub rumble |
| 11 | `vocal_balance` | Vocal Balance | 0.70 | Centre-vs-sides energy, intelligibility, consistency |
| 12 | `stereo_width` | Stereo Width | 0.40 | Side/Mid ratio, per-band width, L/R balance |
| 13 | `transients` | Transient Impact | 0.50 | Attack time, punch index, smearing |
| 14 | `clarity` | Mix Clarity | 0.85 | Spectral flatness/contrast, masking, band congestion |

The roll-up is a weighted mean pulled 35% toward the worst dimension, then charged a
compounding penalty for each additional critical/major finding. A pure mean buries a
single catastrophic dimension; a pure minimum throws away everything else measured.

**Genres** (24): acoustic, alternative, ambient, cinematic, classical, country, dnb,
edm, folk, hip_hop, house, indie, jazz, lofi, metal, orchestral, other, pop, punk,
rnb, rock, soul, techno, trap. Unknown names fall back to `other`.

**Platform targets** (7): Spotify, Apple Music, YouTube, Tidal, Amazon Music,
SoundCloud, Club / DJ.

## Running it locally

### Backend

```bash
cd backend

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # add your ANTHROPIC_API_KEY

uvicorn main:app --reload
```

API on `http://localhost:8000`, interactive docs at `/docs`.

`ffmpeg` is only needed for containers `libsndfile` cannot open. WAV, FLAC, AIFF,
OGG and (with a current libsndfile) MP3 decode natively without it.

> **The report arrives in two parts, on purpose.** `POST /analyze` returns every
> measured number in ~3 s. The engineer's write-up is a second call, `POST /engineer`,
> which takes **3–5 minutes** — generation runs at ~75 output tokens/sec and a real
> report is 9k–14k tokens, so it is bound by output length, not by anything tunable.
> Folding the two together would exceed the 30–60 s timeout most proxies enforce.
> The UI renders the score, findings and timeline immediately and fills the fix plan
> in behind them. `MIXDOCTOR_RUN_AI=0` disables the second call entirely.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

App on `http://localhost:5173`. In dev it points at `http://localhost:8000`
automatically — no `.env` needed.

To build the UI with no backend running at all, load
`backend/testdata/sample_analysis.json`. It is a real captured payload (schema v2,
`mix_problem.wav` analysed as trap **with `separate_stems=true`**), so the stem
surfaces have real data: 4 separated sources and 5 named masking pairs. It carries a
real engineer report with 8 prescriptions, so `FixStack` renders too.

Two caveats on that file: the fixture is 27 s, so `SectionMap` shows a single section
(the arrangement pass needs ≥30 s) — use a full-length track to exercise that surface
properly. And the engineer block is carried from an earlier capture of the same file
and genre rather than regenerated, because the AI layer was unkeyed when it was
refreshed; every `finding_id` it references is still raised by the current run, which
the refresh script asserts before writing.

### Environment variables

**Backend** (`backend/.env`):

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `ANTHROPIC_API_KEY` | For the AI layer | — | Unset: analysis runs, prose is skipped, a warning is attached |
| `MIXDOCTOR_RUN_AI` | No | `1` | `0`/`false`/`no`/`off` skips the engineer consult entirely |
| `MIXDOCTOR_ANALYSIS_WORKERS` | No | `min(3, cpu_count)` | DSP thread-pool size. More than 3 is slower — BLAS already threads underneath |
| `JWT_SECRET_KEY` | Production | insecure dev default | Signing key for auth tokens |
| `DATABASE_URL` | No | `sqlite:///./mixdoctor.db` | SQLAlchemy URL |
| `LOG_LEVEL` | No | `INFO` | Root log level |

**Frontend** (`frontend/.env`):

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `VITE_API_BASE` | No | localhost:8000 in dev, Railway in prod | Backend base URL override |
| `VITE_KOFI_USERNAME` | No | — | Ko-fi handle. Renders the tip jar as `https://ko-fi.com/<handle>` |
| `VITE_DONATE_URL` | No | — | Full donation URL (Stripe, PayPal.me, GitHub Sponsors). **Takes precedence** over `VITE_KOFI_USERNAME` |

**The donate surface is opt-in and fails closed.** With *both* variables unset,
`DONATE_URL` resolves to `null` and every donate surface — the footer `DonateLink`
and the post-report `DonatePanel` — returns `null` and renders zero bytes. There is
no dead button and no link to a 404. Setting either one turns both surfaces on; the
label follows the platform ("Buy me a coffee" for Ko-fi, "Support this" otherwise).

## API

Base URL `http://localhost:8000`. Auth is optional everywhere; sending a bearer
token additionally passes the user's plugin list to the AI layer and writes the
result to their history.

### `POST /analyze`

`multipart/form-data`:

| Field | Required | Notes |
|-------|----------|-------|
| `file` | yes | Audio file, max 250 MB |
| `genre` | yes | Genre key or a loose name — `trap`, `Rock`, `Hip Hop` all resolve |
| `reference_file` | no | Second track to compare tonal balance, level and width against |
| `notes` | no | Free-form producer context, passed to the AI layer |
| `separate_stems` | no | `true` runs the Demucs depth pass. **Off by default** — see [Stem separation](#stem-separation-opt-in) for what it costs and what it buys |

Returns a `MixAnalysis` (schema v2). Top-level shape:

```jsonc
{
  "schema_version": 2,
  "filename": "mix.wav",
  "genre": "trap",
  "health_score": 47.3,          // 0-100, weighted roll-up
  "grade": "F",                  // A+ .. F
  "ceiling_score": 92.1,         // reachable if every prescription is applied
  "mastering_ready": false,
  "mastering_blockers": ["..."], // each names the figure that produced it
  "dimensions":       [ /* 14 × DimensionScore */ ],
  "findings":         [ /* Finding: severity, confidence, evidence[], moments[] */ ],
  "measurements":     {
    /* the full DSP dataset, 10 measurement groups, plus the two depth passes: */
    "sections": { /* SectionAnalysis — always present, always attempted */ },
    "stems":    { /* StemAnalysis — `available:false` unless separate_stems was set */ }
  },
  "platform_targets": [ /* 7 × PlatformTarget */ ],
  "reference":        null,      // ReferenceDelta when reference_file was sent
  "engineer":         null,      // always null here — see POST /engineer
  "waveform_peaks":   [ /* 1400 floats, 0-1 */ ],
  "waveform_rms":     [ /* 1400 floats, 0-1 */ ],
  "analysis_ms":      2873,
  "warnings":         []         // mono/short-file/degraded-stage notices
}
```

### `POST /engineer`

Body: `{ "analysis": MixAnalysis, "plugins"?: OwnedPlugin[] }` — the analysis exactly
as `/analyze` returned it, plus optionally the producer's vault. A bare `MixAnalysis`
is still accepted for clients that predate the vault; it simply arrives with no
plugins and the brief falls back to stock DAW devices.

`plugins[]` is what makes the prescriptions specific to this producer — see
[The plugin capability system](#the-plugin-capability-system). An authenticated
request also merges the plugins saved on the account.

Returns an `EngineerReport` — verdict, the one thing, strengths, prescriptions with
exact DAW moves, session plan.

Stateless on purpose: the client posts back what it already holds, so there is no job
table to keep, it works across however many workers are running, and a dropped
connection costs a retry rather than orphaning a job. Takes 3–5 minutes.

`503` when the layer is switched off or unkeyed, `502` when the model could not be
reached — both non-fatal, since the client already has every measured finding.

The authoritative definition is `backend/analysis/types.py`; the TypeScript mirror
is `frontend/src/types/analysis.ts`. **They must be changed in the same commit** —
the wire contract is hand-maintained, not generated.

**Errors.** A file that cannot be analysed returns **422** with a sentence written
for the person who uploaded it, never a stack trace:

```json
{ "detail": "This file is silent (or near-silent) — nothing to measure." }
```

Also `400` for an unsupported extension or an empty upload, `413` over 250 MB, and
`500` only for genuine server bugs.

### `GET /genres`

`{"genres": [{"key": "trap", "label": "Trap"}, ...]}` — the UI selector's source.

### `GET /health`

`{"status": "healthy", "version": "2.0.0"}`

### Auth, plugins, history

`POST /auth/signup`, `POST /auth/login`, `GET /auth/me`;
`GET|POST /plugins`, `DELETE /plugins/{id}`;
`GET /history`, `GET /history/{id}`.

### Supported audio formats

`.wav` `.mp3` `.flac` `.aiff` `.aif` `.m4a` `.ogg` `.opus`

Anything decodable is resampled to 48 kHz mono/stereo internally; analysis is capped
at the first 10 minutes.

## Depth passes

Two passes go beyond the whole-file numbers. One is free and always runs; the other
is expensive and never runs unless you ask.

### Section analysis

`analysis/dsp/sections.py`. Always on, no flag, ~0.7–1.5 s.

A single set of numbers for a whole record hides the thing producers actually get
wrong: the chorus that fails to lift, the drop where the low end collapses, the verse
sitting 4 dB under everything around it. The pass segments the track on novelty in
the spectral/loudness feature stream, labels each span (`intro` / `verse` / `chorus` /
`drop` / `bridge` / `outro`), and re-measures level, spectrum and dynamics inside each
one. On a 218 s fixture it returns 8 sections; on a 27 s one it returns a single
section, because the arrangement detectors need ≥30 s before a loudness range is
measuring an arrangement rather than the file's edges.

It renders as `SectionMap` in the report and it is clickable — selecting a section
seeks the timeline to it.

### Stem separation (opt-in)

`analysis/dsp/separation.py`. **Off by default.** Set `separate_stems=true` on
`POST /analyze`, or flip the deep-analysis toggle in the intake form.

Demucs (`htdemucs`) splits the mix into vocals / drums / bass / other, and each source
is measured on its own. That converts four *inferences* into *measurements*:

| Without stems | With stems |
|---|---|
| Vocal level guessed from a centre-channel proxy | Vocal measured against the actual instrumental |
| Kick and 808 are one waveform | Two objects, with separate fundamentals and a real collision figure |
| Compression read off the master | Per-element gain reduction |
| Masking inferred from the summed spectrum | Source-against-source, with named masker/maskee pairs |

Findings built on stem data carry visibly higher `confidence` for exactly this reason.

**What it costs.** Measured on this machine (Apple Silicon, 8 cores), a 27 s stereo
fixture:

| | Cold (first call, model load) | Warm |
|---|---|---|
| Separation stage | 6.8 s | 4.9–5.4 s |
| **Total `analyze_mix`** | **9.7 s** | **8.2 s** |
| Same file, stems off | — | **2.4–2.9 s** |

So it is roughly a **3× wall-clock multiplier**, and it is the one stage that does not
parallelise with anything else. The first call also downloads ~80 MB of weights into
the torch hub cache. Cost scales with duration, not with file size.

**It degrades gracefully rather than failing.** If torch or demucs is missing, the
weights cannot be downloaded, or the model errors, the analysis still completes: the
stage returns `stems.available: false` with an explanatory warning propagated to
`warnings` on the report, and every non-stem finding is unaffected. Verified by
blocking the `demucs` import outright — the run completes in 3.4 s with the same
health score as a normal stems-off run and this warning attached:

```
Stem separation unavailable: the Demucs model could not be loaded
(ImportError: No module named 'demucs'). The mix was analysed without stems.
```

## The plugin capability system

`analysis/capabilities.py` + `frontend/src/data/plugins.ts`.

A prescription should change *shape* based on the tools available, not just swap a
brand name in. Knowing someone owns "Pro-Q 3" is worth little; knowing they own
something with `eq_dynamic` changes the instruction:

```
resonance_suppressor  →  let Soothe track it; set depth, not frequency
eq_dynamic            →  dynamic band at 318 Hz, threshold -24, range -6
eq_static only        →  static notch, and automate it off in the sparse bars
```

Only the last one makes the producer do the work by hand — and telling someone who
owns Soothe to hand-notch is what makes a tool feel generic. **Names go stale;
capabilities do not.**

So the vault stores capability slugs, not just names. Every user starts with
`STOCK_CAPABILITIES` (static EQ, compressor, limiter, expander, reverb, delay,
saturation, spectrum meter) — the devices every DAW ships — so the capability list is
never empty and there is always an executable plan. `CAPABILITY_FOR_DIMENSION` then
maps each finding dimension to the tools that actually solve it, and the brief tells
the model both what the producer has *and* what is missing, per problem this mix
actually has.

**What it changes in practice.** Same mix (`mix_problem.wav`, trap), same findings —
only the vault differs:

| Dimension | Stock only | With Pro-Q 3 + soothe2 + Pro-MB + Pro-L 2 |
|---|---|---|
| `mud` | Static EQ | Dynamic EQ, Resonance suppressor, Multiband comp |
| `harshness` | Static EQ | Resonance suppressor, Dynamic EQ |
| `low_end` | **nothing purpose-built** | External sidechain, Dynamic EQ, Multiband comp |
| `loudness` | **nothing purpose-built** | True-peak limiter |
| `limiter` | **nothing purpose-built** | True-peak limiter, Multiband comp |

The stock brief carries an explicit *"not available to them"* list and an instruction
not to write a move whose settings only make sense on a tool they do not own. The
equipped brief unlocks external sidechain for this mix's kick/808 collision — which is
the correct fix for it and is simply not reachable with stock devices.

The vault opens from the header (`PluginVaultTrigger` in `Shell`) and from the intake
form, persists locally, and is posted as `plugins[]` on `POST /engineer`.

## Performance

A 20-second stereo fixture, measured on an 8-core laptop:

| Stage | Time |
|-------|------|
| Decode | ~47 ms |
| All 10 measurements (3 threads) | ~2.8 s |
| Detectors + scoring + waveform | ~5 ms |
| **Total, `POST /analyze`** | **~2.4-2.9 s** |
| Section analysis (included above) | ~0.7–1.5 s |
| Stem separation (`separate_stems=true`, *not* included above) | +4.9 s warm, +6.8 s cold |
| **Total, `POST /analyze` with stems** | **~7.9-9.7 s** |
| `POST /engineer` (`effort=high`) | 3–5 min, output-token bound |

Transient analysis dominates the stems-off DSP pass — it is the beat tracker, and it
sets the wall-clock time almost single-handedly. With stems on, separation dominates
everything: it is a single-threaded neural forward pass that overlaps with nothing.

## Calibration

**The honest limitation of this product is `backend/analysis/targets.py`.** Every
number in it — the 1/3-octave anchor curves, the loudness and dynamics windows, the
mud and harshness caps — is hand-set from engineering practice. That is a defensible
starting point and it is not the same thing as *measured*. A hand-set curve encodes
what a good engineer believes trap should look like; it does not encode what the trap
records people actually release look like. Where those two disagree, MixDoctor is
confidently telling a producer to move toward a number nobody in the genre is hitting.

`backend/tools/calibrate.py` closes that gap for anyone with a library of commercial
masters. It fits the targets to real records:

```bash
cd backend
python tools/calibrate.py --genre trap --input ~/Music/trap-references/ \
    --out calibration-trap.md --apply
```

**What it does.** Walks the directory recursively for every supported extension and
measures each file with the *existing* DSP layer — `analysis.engine.measure_reference`,
the same four measurements the reference-track comparison uses. Nothing in the tool
reimplements measurement, so a change to the DSP changes the calibration with it.
Measurement runs on a process pool (one file per worker, progress on stderr).

Then it aggregates:

* **The fitted target curve** — median and interquartile range of the 1/3-octave
  spectrum across the corpus, normalised to the same 800/1000/1250 Hz reference
  `dsp/spectral.py` uses, so it drops onto `targets.target_curve()` unconverted.
* **Distributions** for integrated LUFS, LRA, true peak, crest factor,
  micro-dynamics, PSR p10, stereo width, correlation, mud ratio and harshness index.

The markdown report puts the fitted values next to the hand-set ones and flags every
place the hand-set value falls outside the corpus interquartile range. **Those flags
are the list of things that are wrong.** Scalar windows are additionally judged on
what fraction of the corpus actually falls inside them — a window centred correctly
but too narrow fires on records that are fine, and the report says so.

**What it refuses to do.** Files under 60 s, silent files, mono files in a genre
mastered in stereo, dual-mono, anything under −35 LUFS, and anything that will not
decode are skipped, each with its reason listed in the report. Below **5** usable
masters the tool stops and writes nothing — a "fitted" curve from two tracks is worse
than a hand-set one, because it looks empirical. `--min-files`, `--min-duration` and
`--allow-mono` relax those deliberately.

`--apply` writes `backend/analysis/targets_fitted.py`: a generated, importable module
holding the fitted anchors in the exact tuple-of-tuples format `_CURVES` uses, the
full unreduced 31-band curve, the per-band IQR, and a ready-to-paste `_register(...)`
call. **It never touches `targets.py`.** Re-running for another genre merges into the
same file rather than replacing it. Replacing a hand-set opinion with a fitted one is
a decision a human makes after reading the report, not one a script makes by
overwriting a file.

### Known measurement bias — read this before pasting anything

`core.stft_power` runs an 8192-point FFT at 48 kHz (5.86 Hz bins) and a 1/3-octave
band's level is the *summed* power of the bins inside it. A bin count is an integer
and a band width is not, so the narrow low bands read systematically hot or shy —
arithmetic on the grid, nothing to do with the audio. The tool derives the size of
that error rather than guessing at it (`_grid_bias_db()`):

| Band | Width | Bins | Predicted bias |
|---|---:|---:|---:|
| 20 Hz | 4.6 Hz | 0.79 | +1.02 dB |
| 31.5 Hz | 7.3 Hz | 1.24 | +2.06 dB |
| 40 Hz | 9.3 Hz | 1.58 | −1.99 dB |
| 63 Hz | 14.6 Hz | 2.49 | +0.81 dB |
| 125 Hz | 28.9 Hz | 4.94 | −0.92 dB |

From 160 Hz up there are dozens of bins per band and the effect falls below 0.01 dB.
The bias is identical for every file, so it never widens the IQR — it offsets the
fitted median of those bands by a fixed amount. The report stars them, prints the
predicted offset next to each, and tells you to subtract it before believing the
number. Everywhere else the fitted value is simply the better one.

### `--self-test`

Proves the harness itself is correct with no real corpus available:

```bash
cd backend
python tools/calibrate.py --self-test
```

`testdata/reference_trap.wav`, `reference_pop.wav` and `reference_folk.wav` are noise
shaped to sit exactly on their genre's target curve, which makes them ground truth.
The self-test expands each into a synthetic corpus — five variants at different levels
and different spectral tilts (−0.6 … +0.6 dB/decade, **median zero**), two of them in a
subdirectory, plus one file per admission rule that must be rejected — and runs the
whole production path over it: directory walk, skip rules, process pool, median/IQR
aggregation, comparison, `--apply`, and the generated module's own import.

The assertion is not "one file measures right" but "the median of a corpus that
disagrees with itself recovers the truth". Measured: worst per-band error **0.40–0.54
dB** across the 25 grid-free bands (rms 0.18–0.23 dB), and every band the tool flags
as contradicting `targets.py` is a grid-limited one — it invents no disagreements
where the measurement can speak for itself.

## Tests / fixtures

`backend/testdata/` holds six 44.1 kHz fixtures: `mix_clean`, `mix_problem`,
`mix_phase` (20 s stereo), `mix_mono` (20 s mono), `mix_short` (1.2 s), and
`mix_silent` (2 s of silence, which must fail with a clean 422).

Three 48 kHz 45 s `reference_*.wav` files are the ground truth for the whole scoring
system. Each is noise shaped to its genre's **target curve**, then run through a
synthetic mastering chain in the same causal order a real master is built:

    spectrum -> arrangement dynamics -> limiting until crest hits the genre figure
    -> peak set to -1 dBFS

Loudness is an *output* of that chain rather than a knob, so it lands inside the genre
window on its own. That matters: it means the reference is correct on spectrum,
dynamics, loudness and peak simultaneously, which is the only way it can validate a
scorer that judges all four.

| Fixture | LUFS (genre window) | Crest (window) | Score at its own genre |
|---|---|---|---|
| `reference_trap` | -6.7 (-9.5 … -6.0) | 7.4 (5.5 … 10.0) | **90.1 (A)** |
| `reference_pop`  | -8.2 (-11.0 … -7.5) | 10.3 (8.0 … 13.0) | **90.2 (A)** |
| `reference_folk` | -11.6 (-16.0 … -11.0) | 14.3 (11.0 … 18.0) | **91.8 (A)** |

**The cross-genre gradient is the real test.** Judging the same trap reference against
progressively more distant genres should degrade smoothly, and it does:

| `reference_trap` judged as | trap | pop | rock | folk | ambient |
|---|---|---|---|---|---|
| Score | **90.1 (A)** | 87.5 (A-) | 76.4 (C+) | 71.8 (C) | 65.1 (D+) |

A correct mix scores A; the same file scores worse the further its genre context moves
from what it was built for. If a change to the detectors or targets breaks either of
those two tables, that is a regression — check it before shipping.

The residual minor findings on these files (a few hundred clipped samples, limiter
drive) are the analyser correctly catching the `tanh` limiting in the generator. They
are real, not noise.

> Regenerating them: match the genre's loudness and crest windows, not just its curve.
> An earlier cut fitted spectrum only, rendered everything near -15 LUFS, and
> `reference_trap` scored 66.5 (D+) for being 6 LU too quiet — a defect in the fixture
> that looked exactly like a defect in the analyser.

```bash
cd backend
python -c "
from analysis.engine import analyze_mix
a = analyze_mix('testdata/mix_problem.wav', 'trap', run_ai=False)
print(a.health_score, a.grade, len(a.findings))
"
```

## Deploying

Both halves auto-deploy from a push to `main`: **Railway** builds `backend/`
(nixpacks, see `nixpacks.toml`), **Vercel** builds `frontend/` (see
`frontend/vercel.json`). There is no deploy CLI step — `git push` is the deploy.

### Environment variables

**Railway (backend)**

| Variable | Required | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | for the write-up | Without it, `/analyze` still returns every measured finding; `/engineer` returns 503 and the UI says so. |
| `JWT_SECRET_KEY` | yes in production | Falls back to an insecure dev default otherwise. |
| `DATABASE_URL` | **yes** | See the persistence warning below. |
| `MIXDOCTOR_ENABLE_STEMS` | no | `1` to offer deep analysis. Requires `requirements-stems.txt` installed *and* the RAM to run it. |
| `MIXDOCTOR_RUN_AI` | no | `0` sheds the AI layer without a deploy. |

**Vercel (frontend)**

| Variable | Required | Notes |
|---|---|---|
| `VITE_API_BASE` | yes | The Railway URL. Without it a production build falls back to the hard-coded default in `src/config.ts`. |
| `VITE_KOFI_USERNAME` | no | Ko-fi username. Unset ⇒ every donate surface renders nothing. |
| `VITE_DONATE_URL` | no | Full URL for Stripe/BMAC/GitHub Sponsors instead. Takes precedence. |

### Persistence — read this before shipping accounts

`DATABASE_URL` defaults to `sqlite:///./mixdoctor.db`, and Railway's filesystem
is **ephemeral**. On the default setting every deploy silently wipes user
accounts, saved plugin vaults and analysis history. Add a Railway Postgres
service and set `DATABASE_URL` to its connection string before anyone signs up.

Anonymous use is unaffected: the plugin vault is `localStorage`-first and
analysis needs no account.

### Source separation in production

Not enabled by default, on purpose. `torch` + `demucs` add ~800 MB to the image
and htdemucs wants a couple of GB of RAM per job — enough to OOM a small
instance. Measured here: ~5 s on Apple Silicon (MPS), materially slower on a
shared vCPU.

To enable, on an instance with the headroom:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # Linux: skip the CUDA payload
pip install -r backend/requirements-stems.txt
```

then set `MIXDOCTOR_ENABLE_STEMS=1`. `GET /capabilities` reports the resolved
state, and the frontend hides the deep-analysis toggle when it is off, so the
control is never offered where it cannot run.

### The write-up takes minutes

`POST /engineer` is a separate call precisely because it runs 2-4 minutes
(output-token bound, see [Performance](#performance)). Confirm your proxy will
hold a request that long. Vercel's serverless functions will not — which is why
the frontend calls Railway directly rather than proxying through Vercel.

## License

MIT
