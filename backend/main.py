"""FastAPI backend for MixDoctor.

`POST /analyze` is the whole product: an upload and a genre in, a `MixAnalysis`
out. Three things in here are load-bearing and easy to undo by accident:

* **The upload is streamed to disk in chunks, never `await file.read()`.** A
  250 MB read materialises 250 MB of `bytes` in the worker before anything is
  written, which is an instant OOM on a small dyno — and the size check that
  was supposed to prevent it ran *after* the read.
* **Only `core`'s own errors become a 422.** They carry a sentence written for
  the person who uploaded the file ("Track is 0.40s. At least 1 second of audio
  is needed."), so the message is passed through verbatim. Anything else is a
  bug on our side and is a 500 with a generic message.
* **CORS accepts any localhost port via regex.** The fixed list broke every
  time Vite picked a different port.
* **`separate_stems` is opt-in and stays that way.** Source separation is the
  only stage in the analysis that costs more than everything else put together,
  so the default request is ~3 s and the deep one is a deliberate choice the
  caller makes per upload. See the `/analyze` docstring for the measured cost.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import (
    Body, Depends, FastAPI, File, Form, HTTPException, Query, Response, UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session

from starlette.concurrency import run_in_threadpool

import budget
import engineer
import report as report_builder
from analysis import capabilities, core, engine, knowledge, targets
from analysis.types import (
    ClarificationAnswer, EngineerReport, MixAnalysis, OwnedPlugin,
)
from auth import decode_token
from database import get_db, init_db
from models import AnalysisHistory, User, UserPlugin
from schemas import HealthResponse

load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("mixdoctor")

if not os.environ.get("ANTHROPIC_API_KEY"):
    logger.warning(
        "ANTHROPIC_API_KEY is not set. Analysis still runs and every finding is "
        "measured; only the engineer's write-up is skipped."
    )

app = FastAPI(
    title="Mix Diagnostic API",
    description="Deep audio mix analysis",
    version="2.0.0",
)

# The named origins are the deployed frontends. The regex covers local dev on
# any port — Vite hands out 5173, 5174, 5175... depending on what is free, and
# a fixed list means a broken app on whichever one you get today.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # The new home. Both apex and www — Vercel serves whichever you set as
        # primary and redirects the other, but the browser sends Origin from
        # whatever the user typed, so a missing one is a 403 on your own site.
        "https://mixdiagnostic.com",
        "https://www.mixdiagnostic.com",
        # The Vercel hostnames stay: they keep working after a custom domain is
        # attached, and preview deployments are served from them.
        "https://mix-doctor.vercel.app",
        "https://mixdoctor.vercel.app",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from routers import auth as auth_router  # noqa: E402
from routers import history as history_router  # noqa: E402
from routers import plugins as plugins_router  # noqa: E402

app.include_router(auth_router.router)
app.include_router(plugins_router.router)
app.include_router(history_router.router)

# Whatever the decoder can actually open, rather than a second hand-maintained
# list that drifts out of sync with it.
SUPPORTED_FORMATS = set(core.SUPPORTED_FORMATS)

MAX_FILE_SIZE = 250 * 1024 * 1024          # 250 MB
UPLOAD_CHUNK = 1024 * 1024                 # 1 MB

security = HTTPBearer(auto_error=False)


def _run_ai_enabled() -> bool:
    """Whether the engineer's write-up may be requested at all.

    **Off unless explicitly enabled.** Every `/engineer` call costs the operator
    roughly 40 cents of Anthropic credit, uncapped and unauthenticated, so a
    default of on means a public deployment bills its owner for every visitor —
    and for every refresh. Defaulting off is the only safe posture for a site
    anyone can reach.

    Nothing else is lost when it is off. The measured findings, all 51
    explainers, the clarification questions, the score card and the downloadable
    report are deterministic and need no API at all. The write-up is the
    track-specific layer on top; `/engineer` returns a clean 503 and the UI says
    so rather than breaking.

    Set `MIXDOCTOR_RUN_AI=1` to turn it on for a private instance, or once
    there is a way to charge for it.
    """
    return (os.environ.get("MIXDOCTOR_RUN_AI", "0").strip().lower()
            in {"1", "true", "yes", "on"})


# The consult is deliberately NOT part of /analyze.
#
# Measured on this codebase: generation runs at ~75 output tokens/sec, and a
# real report is 9k-14k tokens, so the call takes 2-3 minutes and is bound by
# output length rather than anything we can tune away. Effort only moves it
# between ~128s and ~182s. That is far past the 30-60s timeout most proxies
# (Railway, Vercel) enforce, so a synchronous /analyze would not merely be slow
# — it would 504 in production.
#
# Splitting it is also the better product: measurement finishes in ~3s, so the
# score, the findings and the timeline are on screen almost immediately, and
# the prescriptions fill in behind them.
MAX_ANALYSIS_BODY_BYTES = 4 * 1024 * 1024


@app.on_event("startup")
async def startup_event() -> None:
    init_db()
    logger.info(
        "MixDoctor API ready (analysis workers=%d, AI layer=%s)",
        engine.ANALYSIS_WORKERS,
        "on" if (_run_ai_enabled() and os.environ.get("ANTHROPIC_API_KEY")) else "off",
    )


# ---------------------------------------------------------------------------
# Upload handling
# ---------------------------------------------------------------------------


def _check_extension(filename: Optional[str], label: str) -> str:
    if not filename:
        raise HTTPException(status_code=400, detail=f"No filename provided for the {label}.")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported {label} format: {ext or '(none)'}. "
                f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
            ),
        )
    return ext


async def _spool(upload: UploadFile, ext: str, label: str) -> str:
    """Stream an upload to a temp file, enforcing the size limit as we go.

    The limit is checked against bytes already written rather than against a
    `Content-Length` header (which a client controls) or against a fully-read
    body (which is the OOM). The partial file is removed before raising, so a
    rejected 300 MB upload leaves nothing behind on disk either.
    """
    handle = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    path = handle.name
    written = 0
    try:
        while True:
            chunk = await upload.read(UPLOAD_CHUNK)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"The {label} is larger than the "
                        f"{MAX_FILE_SIZE // (1024 * 1024)} MB limit."
                    ),
                )
            handle.write(chunk)
    except HTTPException:
        handle.close()
        _remove(path)
        raise
    except Exception:
        handle.close()
        _remove(path)
        logger.exception("failed to spool the %s upload", label)
        raise HTTPException(status_code=400, detail=f"Could not read the uploaded {label}.")
    finally:
        if not handle.closed:
            handle.close()

    if written == 0:
        _remove(path)
        raise HTTPException(status_code=400, detail=f"The uploaded {label} is empty.")
    return path


def _remove(path: Optional[str]) -> None:
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            logger.warning("could not remove temp file %s", path)


def _current_user(
    credentials: Optional[HTTPAuthorizationCredentials], db: Session
) -> Optional[User]:
    if not credentials:
        return None
    user_id = decode_token(credentials.credentials)
    if user_id is None:
        return None
    return db.query(User).filter(User.id == user_id).first()


def _persist(
    db: Session, user: User, filename: str, genre: str, result: MixAnalysis
) -> None:
    """Write the analysis to history. A DB failure must not fail the response.

    `AnalysisHistory` predates the v2 schema, so the new report is adapted onto
    it rather than the other way round: `health_score` is an Integer column so
    the float is rounded, `summary` takes the engineer's verdict when there is
    one and the worst finding's own sentence when there is not, and `diagnoses`
    takes the findings. `metrics` stores the score fields alongside the raw
    measurements so the history list can be rendered without re-deriving them.
    """
    try:
        summary = ""
        if result.engineer and result.engineer.verdict:
            summary = result.engineer.verdict
        elif result.findings:
            summary = f"{result.findings[0].title}. {result.findings[0].detail}"
        else:
            summary = next(
                (d.headline for d in result.dimensions if d.headline),
                "No issues detected.",
            )

        entry = AnalysisHistory(
            user_id=user.id,
            filename=filename,
            genre=genre,
            health_score=int(round(result.health_score)),
            summary=summary[:4000],
            diagnoses=[f.model_dump() for f in result.findings],
            metrics={
                "schema_version": result.schema_version,
                "health_score": result.health_score,
                "grade": result.grade,
                "ceiling_score": result.ceiling_score,
                "mastering_ready": result.mastering_ready,
                "mastering_blockers": result.mastering_blockers,
                "dimensions": [d.model_dump() for d in result.dimensions],
                "measurements": result.measurements.model_dump(),
                "platform_targets": [p.model_dump() for p in result.platform_targets],
                "reference": result.reference.model_dump() if result.reference else None,
                "analysis_ms": result.analysis_ms,
            },
        )
        db.add(entry)
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("could not persist the analysis to history")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="healthy", version="2.0.0")


def _stems_enabled() -> bool:
    """Whether this deployment can actually separate a mix.

    Two gates, both of which must pass. `MIXDOCTOR_ENABLE_STEMS` is the
    operator's switch; the import check is reality.

    Torch and Demucs are deliberately NOT in requirements.txt. They add ~800 MB
    to the image and htdemucs wants a couple of GB of RAM per job, which OOMs a
    small dyno — so the default deployment ships without them and the analysis
    layer is written to notice (separation.py imports torch lazily, inside the
    functions that need it). Install them and flip the flag on a box with the
    headroom, and the feature lights up with no code change.
    """
    if os.environ.get("MIXDOCTOR_ENABLE_STEMS", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    try:
        import importlib.util

        return all(importlib.util.find_spec(m) is not None for m in ("torch", "demucs"))
    except Exception:
        return False



def _budget_snapshot() -> Dict[str, Any]:
    """What the cap looks like right now, for /capabilities.

    Opens its own short-lived session: this runs inside the capabilities
    handler, which has no `db` dependency, and adding one there would make a
    trivial probe hold a connection.
    """
    from database import SessionLocal

    limit = budget.budget_usd()
    if limit <= 0.0:
        return {"enabled": False, "period": budget.budget_period()}
    db = SessionLocal()
    try:
        state = budget.check_budget(db)
        return {
            "enabled": True,
            "period": budget.budget_period(),
            "budget_usd": round(limit, 2),
            "remaining_usd": round(state.remaining, 2),
            "reports_left": state.reports_left,
        }
    except Exception:
        logger.exception("budget: snapshot failed")
        return {"enabled": True, "period": budget.budget_period(), "error": True}
    finally:
        db.close()


@app.get("/capabilities")
async def capabilities_probe() -> Dict[str, Any]:
    """What this particular deployment can do.

    The frontend reads this to decide whether to *offer* deep analysis at all.
    Hiding a control the server cannot honour is better than degrading after
    the user has already waited for an upload.
    """
    return {
        "version": app.version,
        "stems": _stems_enabled(),
        "ai": bool(_run_ai_enabled() and os.environ.get("ANTHROPIC_API_KEY")),
        "ai_budget": _budget_snapshot(),
        "max_upload_bytes": MAX_FILE_SIZE,
        "formats": sorted(SUPPORTED_FORMATS),
    }


@app.get("/genres")
async def list_genres() -> Dict[str, List[Dict[str, str]]]:
    """Genre keys and labels for the UI selector."""
    return {"genres": [{"key": key, "label": label} for key, label in targets.all_genres()]}


# ---------------------------------------------------------------------------
# Teaching content
# ---------------------------------------------------------------------------

#: Built once and served as bytes. The content is fixed at deploy time, so
#: re-encoding ~half a megabyte of JSON per request would be pure waste.
_KNOWLEDGE_CACHE: Optional[bytes] = None


def _knowledge_payload() -> Dict[str, Any]:
    """Every explainer, plus the vocabulary needed to read the fix steps.

    **Fix steps ship unresolved, on purpose.** `knowledge.applicable_steps()`
    exists and is not used here: resolving `needs` server-side would require
    knowing which plugins this person owns, and for an anonymous user we do not
    — the vault is localStorage-first and never reaches us. Resolving it with
    the stock list as a stand-in would be worse than not resolving it, because
    every step gated behind a tool they *do* own would silently render as its
    `without` fallback. So each step carries `action`, `detail`, `needs` and
    `without` verbatim and the client matches them against its own vault.

    `capabilities` is here so the UI can say "needs a dynamic EQ" rather than
    "needs eq_dynamic", and `stock_capabilities` so a client with an empty vault
    still resolves the same way the server would: those are assumed, not owned.

    **`resources` ships `source` alongside every link, and the client is
    expected to render it.** These are outbound links to other people's work,
    which is exactly why they need no licence — a hyperlink is not a copy — and
    exactly why the publisher has to be named wherever the link appears. Serving
    the attribution as a field rather than leaving the UI to invent one is what
    makes that guarantee hold at every call site instead of at the ones somebody
    remembered.
    """
    explainers = knowledge.ensure_registered()

    payload: Dict[str, Any] = {}
    for finding_id, explainer in sorted(explainers.items()):
        payload[finding_id] = {
            "id": finding_id,
            "dimension": finding_id.split(".", 1)[0],
            "headline": explainer.headline,
            "what_it_is": explainer.what_it_is,
            "what_you_hear": explainer.what_you_hear,
            "why_it_matters": explainer.why_it_matters,
            "common_causes": list(explainer.common_causes),
            "how_to_fix": [
                {
                    "action": step.action,
                    "detail": step.detail,
                    "needs": step.needs,
                    "without": step.without,
                }
                for step in explainer.how_to_fix
            ],
            "how_to_verify": explainer.how_to_verify,
            "learn_more": explainer.learn_more,
            "minutes": explainer.minutes,
            # Read through `getattr` and always emitted, even when empty. An
            # explainer nobody has written links for yet is normal, and a client
            # that can rely on the array existing renders "no resources" as an
            # empty loop instead of a special case.
            "resources": [
                {
                    "kind": r.kind,
                    "label": r.label,
                    "url": r.url,
                    "note": r.note,
                    # Never dropped when blank. `source` is the citation shown
                    # beside the link, and the client must not have to decide
                    # whether a missing key means unattributed or unknown.
                    "source": r.source,
                }
                for r in (getattr(explainer, "resources", ()) or ())
            ],
        }

    return {
        "version": app.version,
        "count": len(payload),
        "explainers": payload,
        "capabilities": {
            slug: {"label": label, "unlocks": unlocks}
            for slug, (label, unlocks) in sorted(capabilities.CAPABILITIES.items())
        },
        # What advice is allowed to assume. A step with no `needs`, or one
        # naming a slug in here, is executable by everybody.
        "stock_capabilities": list(capabilities.STOCK_CAPABILITIES),
    }


@app.get("/knowledge")
async def knowledge_base() -> Response:
    """The teaching layer: what every finding means and how to fix it.

    Static per deploy and keyed by finding id, so the client fetches it once and
    joins it against whatever analysis is on screen. That is the point — the
    explanation for a finding must not depend on an API call that can be slow,
    rate-limited or switched off, because "true peak above delivery ceiling"
    with nothing behind it is the thing this endpoint exists to stop shipping.
    """
    global _KNOWLEDGE_CACHE
    if _KNOWLEDGE_CACHE is None:
        _KNOWLEDGE_CACHE = json.dumps(
            _knowledge_payload(), ensure_ascii=False
        ).encode("utf-8")

    return Response(
        content=_KNOWLEDGE_CACHE,
        media_type="application/json",
        # An hour is short next to "changes only on deploy", and it is the
        # difference between fetching this once and fetching it per analysis.
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.post("/analyze", response_model=MixAnalysis)
async def analyze(
    file: UploadFile = File(..., description="Audio file to analyze"),
    genre: str = Form(..., description="Music genre, e.g. 'trap', 'Rock', 'Hip Hop'"),
    intent: str = Form(
        "full_mix",
        description=(
            "What the file is: full_mix, beat, instrumental, stem, reference or demo. "
            "Genre sets the reference the mix is compared against; intent sets which "
            "comparisons are worth making at all — a beat's absent lead is correct, a "
            "stem has no mix balance, and nobody is being asked to fix a released "
            "record. Unknown values fall back to full_mix."
        ),
    ),
    reference_file: Optional[UploadFile] = File(
        None, description="Optional reference track to compare against"
    ),
    notes: Optional[str] = Form(None, description="Optional context from the producer"),
    separate_stems: bool = Form(
        False,
        description=(
            "Run source separation and measure vocals/drums/bass/other on their own. "
            "Costs roughly 3x the request time; off by default."
        ),
    ),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
) -> MixAnalysis:
    """Analyze one mix and return the full report.

    Authentication is optional. When present, the user's plugin list is handed
    to the AI layer so the moves it prescribes name devices they actually own,
    and the result is written to their history.

    **`separate_stems` costs real latency and is off by default.** The default
    path is ~2-3 s for a 20 s clip and ~10 s for a four-minute track, because
    the eleven measurements run in parallel on a thread pool. Turning this on
    adds a Demucs pass over the whole file *before* any of it can be judged,
    and that pass does not parallelise with anything.

    Measured end to end on this codebase, on Apple Silicon with MPS:

        20 s clip        1.9 s  ->   5.4 s   (2.9 s separating)
        4:00 track      10.2 s  ->  52.4 s   (36.8 s separating)
        5:16 track      12.5 s  ->  69.0 s   (48.7 s separating)

    That is roughly 9 s of separation per minute of audio with the model
    resident, plus a one-off ~3 s the first time the weights are loaded, and
    several times all of it on a CPU-only host. **A full-length track will
    exceed a 30 s proxy timeout**, so treat this as a background or
    explicitly-opted-into path rather than something to turn on by default in a
    UI. Past six minutes the DSP layer separates the loudest three minutes only
    and says so in `measurements.stems.warnings`. Set
    `MIXDOCTOR_SEPARATION_TIMEOUT_S` to bound it for your deployment — a blown
    budget truncates the excerpt and names the span it did cover, rather than
    failing the request.

    What the latency buys is confidence, not extra findings. Vocal balance and
    kick-versus-808 stop being inferred from a centre estimate and a
    reconstructed spectrum and become direct measurements (confidence 0.55 to
    0.90 and 0.70 to 0.93 respectively), masking gains a named source on each
    side of it, and per-element compression becomes reportable at all. Every
    failure path — no model, no network on the first run, no GPU, budget
    exhausted — comes back as `measurements.stems.available == false` plus a
    warning, with the rest of the report unchanged.
    """
    ext = _check_extension(file.filename, "audio file")
    ref_ext = (
        _check_extension(reference_file.filename, "reference file")
        if reference_file and reference_file.filename
        else None
    )

    user = _current_user(credentials, db)
    user_plugins: List[UserPlugin] = []
    if user:
        user_plugins = (
            db.query(UserPlugin)
            .filter(UserPlugin.user_id == user.id)
            .order_by(UserPlugin.category, UserPlugin.name)
            .all()
        )

    path: Optional[str] = None
    ref_path: Optional[str] = None
    try:
        path = await _spool(file, ext, "audio file")
        if reference_file and ref_ext:
            ref_path = await _spool(reference_file, ref_ext, "reference file")

        try:
            # Off the event loop. `analyze_mix` is CPU-bound from end to end and
            # holds the thread for its whole duration; with `separate_stems` on
            # that is tens of seconds, and running it inline would stall every
            # other request on this worker — including the health check.
            result = await run_in_threadpool(
                lambda: engine.analyze_mix(
                    path,
                    genre,
                    intent=intent,
                    filename=file.filename or os.path.basename(path),
                    notes=notes,
                    reference_path=ref_path,
                    user_plugins=user_plugins,
                    # Never here — see the note by MAX_ANALYSIS_BODY_BYTES. The
                    # client asks for the write-up separately via POST /engineer.
                    run_ai=False,
                    # Honour the deployment gate, not just the request. A client
                    # asking for stems on a box that cannot run them should get
                    # a normal analysis, not a slow one that fails at the end.
                    separate_stems=bool(separate_stems) and _stems_enabled(),
                )
            )
        except (core.AudioTooShortError, core.SilentAudioError, ValueError) as exc:
            # These carry a message written for the producer, not a stack trace.
            raise HTTPException(status_code=422, detail=str(exc))
        except HTTPException:
            raise
        except Exception:
            logger.exception("analysis failed for %s", file.filename)
            raise HTTPException(
                status_code=500,
                detail="The analysis failed unexpectedly. Please try again.",
            )

        if user:
            _persist(db, user, file.filename or "upload", result.genre, result)

        return result
    finally:
        _remove(path)
        _remove(ref_path)


class EngineerRequest(BaseModel):
    """The wrapper body for `POST /engineer`.

    `plugins` is what makes the prescriptions specific to this producer: the
    capability slugs on each entry decide the *shape* of every move, not just
    which brand name gets printed. A bare `MixAnalysis` body is still accepted
    (see `_parse_engineer_body`), it simply arrives with no plugins.
    """

    analysis: MixAnalysis
    plugins: List[OwnedPlugin] = Field(
        default_factory=list,
        description=(
            "What the producer owns. `capabilities` should carry slugs from "
            "analysis.capabilities.CAPABILITIES; unknown slugs are dropped."
        ),
    )


def _parse_engineer_body(payload: Dict[str, Any]) -> Tuple[MixAnalysis, List[OwnedPlugin]]:
    """Accept the wrapper, fall back to a bare `MixAnalysis`.

    The client shipped before `plugins` existed posts the analysis object
    itself, and that build is in people's browsers right now. Try the wrapper
    first — it is the only shape with a top-level `analysis` key, so there is
    no ambiguity — and only if that fails treat the whole body as the analysis.
    Both errors are reported when neither shape validates, because "field
    required: analysis" on its own would be actively misleading to whoever is
    posting the older shape.
    """
    try:
        request = EngineerRequest.model_validate(payload)
        return request.analysis, list(request.plugins)
    except ValidationError as wrapper_error:
        try:
            return MixAnalysis.model_validate(payload), []
        except ValidationError as bare_error:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": (
                        "The body must be either {analysis: MixAnalysis, plugins?: OwnedPlugin[]} "
                        "or a bare MixAnalysis."
                    ),
                    "as_wrapper": wrapper_error.errors()[:5],
                    "as_bare_analysis": bare_error.errors()[:5],
                },
            )


def _db_plugins(db: Session, user: User) -> List[OwnedPlugin]:
    """The user's saved plugins as `OwnedPlugin`s.

    `models.UserPlugin` has name/manufacturer/category columns and no
    capability column, so these arrive capability-less and
    `engineer.enrich_plugins` fills in what it recognises by name. A client
    that posts real capability data always beats that guess — see
    `_merge_plugins`.
    """
    rows = (
        db.query(UserPlugin)
        .filter(UserPlugin.user_id == user.id)
        .order_by(UserPlugin.category, UserPlugin.name)
        .all()
    )
    return [
        OwnedPlugin(
            name=row.name,
            manufacturer=row.manufacturer,
            category=row.category or "",
            capabilities=[],
        )
        for row in rows
        if row.name
    ]


def _merge_plugins(
    posted: List[OwnedPlugin], stored: List[OwnedPlugin]
) -> List[OwnedPlugin]:
    """Posted list first, then anything saved that is not already in it.

    Posted entries lead because they are the ones that can carry capabilities;
    `engineer.coerce_plugins` does the de-duplication by name and keeps the
    richer record on a collision, so a plugin present in both places does not
    lose its capability list to the DB row.
    """
    return engineer.coerce_plugins([*posted, *stored])[: engineer.MAX_PLUGINS_IN_BRIEF]


@app.post("/engineer", response_model=EngineerReport)
async def engineer_consult(
    payload: Dict[str, Any] = Body(
        ...,
        description=(
            "{analysis: MixAnalysis, plugins?: OwnedPlugin[]}. A bare MixAnalysis is also "
            "accepted for backwards compatibility."
        ),
    ),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
) -> EngineerReport:
    """Second half of the report: the engineer's read on an existing analysis.

    Stateless by design — the client posts back the `MixAnalysis` it already
    holds rather than us keeping a job table. That survives a restart, works
    across however many workers are running, and means a dropped connection
    costs the user a retry instead of an orphaned job.

    The trade is that the measurements arrive from the client and cannot be
    trusted as authentic. That is acceptable: a caller who forges measurements
    gets a report about their own fiction, and nothing is persisted from it.
    What we do defend against is the prose fields being used as an injection
    channel, hence the clamp below.
    """
    if not _run_ai_enabled():
        raise HTTPException(
            status_code=503,
            detail="The engineer's write-up is currently disabled on this server.",
        )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="The engineer's write-up is unavailable: no API key is configured.",
        )

    analysis, posted_plugins = _parse_engineer_body(payload)

    user = _current_user(credentials, db)
    plugins = _merge_plugins(posted_plugins, _db_plugins(db, user) if user else [])
    plugins = _clamp_free_text(analysis, plugins)

    ctx = engineer.EngineerContext(
        measurements=analysis.measurements,
        findings=analysis.findings,
        dimensions=analysis.dimensions,
        genre=analysis.genre,
        # What the file is. Without it every brief posted here read as a full
        # mix, so a beat got told to bring its hook up — the exact failure the
        # intent paragraph exists to prevent. Older payloads omit the field and
        # the model defaults it to `full_mix`, which is the behaviour they had.
        intent=str(getattr(analysis, "intent", "full_mix") or "full_mix"),
        platform_targets=analysis.platform_targets,
        reference=analysis.reference,
        plugins=plugins,
        filename=analysis.filename,
        health_score=analysis.health_score,
        grade=analysis.grade,
        # `scores` is Optional on `MixAnalysis`, so a client posting a payload
        # from before the ScoreCard existed sends None and the brief falls back
        # to the labelled legacy composite. Nothing to guard beyond the getattr:
        # the findings carry `acknowledged`/`clarification` defaults of their
        # own, so an old payload simply has no confirmed choices and no
        # questions, and both new sections render as empty and drop out.
        scores=getattr(analysis, "scores", None),
        mastering_ready=analysis.mastering_ready,
        mastering_blockers=analysis.mastering_blockers,
    )

    # Refuse before spending. A cap that only notices after the API call is
    # not a cap — and this endpoint is unauthenticated, so "one more call" is
    # whatever the internet decides it is.
    state = budget.check_budget(db)
    if not state.allowed:
        raise HTTPException(
            status_code=503,
            detail=(
                "The engineer's write-up is unavailable: this server's AI budget for the "
                "period is spent. Your measured findings are complete."
                if state.reason == "exhausted"
                else "The engineer's write-up is currently disabled on this server."
            ),
        )

    usage: Dict[str, int] = {}
    report = await run_in_threadpool(
        lambda: engineer.consult_engineer(ctx, usage_sink=usage)
    )

    # Bill whatever the call reported, success or not. A run that generated
    # tokens and then failed to parse still cost money.
    if usage:
        cost = budget.record_spend(
            db,
            input_tokens=usage.get("input_tokens", 0),
            cached_tokens=usage.get("cached_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )
        logger.info(
            "engineer: consult cost $%.4f; $%.2f of $%.2f budget remains this %s",
            cost, budget.remaining_usd(db), budget.budget_usd(), budget.budget_period(),
        )

    if report is None:
        # consult_engineer never raises; None means unavailable, refused, or
        # unparseable. The client already has every measured finding, so this
        # degrades the page rather than breaking it.
        raise HTTPException(
            status_code=502,
            detail="The engineer could not be reached. Your measured findings are still complete.",
        )
    return report


class ReassessRequest(BaseModel):
    """The body for `POST /reassess`: a report, plus what the producer said about it."""

    analysis: MixAnalysis
    answers: List[ClarificationAnswer] = Field(
        default_factory=list,
        description=(
            "One entry per question answered. `finding_id` must name a finding on the "
            "posted analysis that carries a `clarification`; anything else is ignored. "
            "`intended: false` clears a previous yes, so an answer can be changed."
        ),
    )


@app.post("/reassess", response_model=MixAnalysis)
async def reassess(
    payload: ReassessRequest = Body(
        ...,
        description="{analysis: MixAnalysis, answers: ClarificationAnswer[]}",
    ),
) -> MixAnalysis:
    """Re-score a report against the producer's answers about what was deliberate.

    The analyzer can measure that the low end steps back in the intro. It cannot
    measure that you wrote it that way. This is where it stops guessing: each
    ambiguous deviation carries a question, and a yes here marks that finding
    `acknowledged`, drops it out of `scores.reference_match` and out of the
    dimension roll-up, and leaves it in the report as an observation with its
    numbers intact.

    **No audio is uploaded and no DSP runs** — the measurements arrive on the
    posted analysis and are carried through untouched, so this is a few
    milliseconds rather than a few seconds. Nothing about the file changed; only
    what the report makes of it did.

    Stateless in exactly the way `/engineer` and `/report` are: the client posts
    back the `MixAnalysis` it already holds, so there is no job table to outlive
    a restart, no session to lose, and answers can be revised as often as the
    producer likes. The same caveat applies — the measurements come from the
    client and are taken at face value — and it costs nothing here, because a
    caller who forges them gets a re-score of their own fiction and nothing is
    persisted.

    Two things a yes cannot do, both enforced in `engine.apply_answers`:
    `scores.technical` does not move, and `mastering_ready` does not move. Only
    deviations are answerable, a defect is never a deviation, and "I meant it"
    is not a fix for a clipped render.
    """
    try:
        return await run_in_threadpool(
            engine.apply_answers, payload.analysis, payload.answers
        )
    except Exception:
        logger.exception("reassessment failed for %s", payload.analysis.filename)
        raise HTTPException(
            status_code=500,
            detail=(
                "The report could not be re-scored. The analysis you have on screen "
                "is unaffected."
            ),
        )


@app.post("/report")
async def download_report(
    payload: Dict[str, Any] = Body(
        ...,
        description=(
            "{analysis: MixAnalysis, plugins?: OwnedPlugin[]}. A bare MixAnalysis is also "
            "accepted, exactly as on POST /engineer."
        ),
    ),
    format: str = Query(
        "markdown",
        description=(
            "'markdown' (default) returns the document as text/markdown. 'html' returns a "
            "single self-contained page — inline CSS, no external requests — with print rules "
            "so the browser's own Save-as-PDF produces a clean document."
        ),
    ),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
) -> Response:
    """The full write-up as a document the producer can keep.

    Stateless in the same way `/engineer` is, and for the same reasons: the
    client posts back the `MixAnalysis` it already holds, so there is no job
    table to outlive a restart and no orphaned state when a connection drops.

    **No AI is involved and no API key is needed.** Every section is built from
    the measurements plus `analysis.knowledge`, which is the whole point of the
    explainer layer — a report that stops working when a third party is down is
    not a report. Where `analysis.engineer` did run and the client posts the
    result back inside the payload, its prescriptions are woven into the defect
    and deviation sections and attributed to it.

    Plugins matter here as much as they do on `/engineer`: the fix steps are
    resolved against the capabilities the producer actually owns, so somebody
    with a dynamic EQ and somebody without get different instructions rather
    than the same one with a caveat.
    """
    analysis, posted_plugins = _parse_engineer_body(payload)

    user = _current_user(credentials, db)
    plugins = _merge_plugins(posted_plugins, _db_plugins(db, user) if user else [])

    wants_html = str(format or "").strip().lower() in {"html", "htm"}
    try:
        # CPU-bound string assembly over a large payload — small next to the
        # analysis itself, but it still holds the thread, and this endpoint is
        # hit while the results page is interactive.
        body = await run_in_threadpool(
            report_builder.render_html if wants_html else report_builder.render_markdown,
            analysis,
            plugins,
        )
    except Exception:
        logger.exception("report generation failed for %s", analysis.filename)
        raise HTTPException(
            status_code=500,
            detail="The document could not be generated. Your analysis on screen is unaffected.",
        )

    filename = report_builder.suggested_filename(analysis, "html" if wants_html else "md")
    return Response(
        content=body,
        # Starlette appends `; charset=utf-8` to any text/* media type itself,
        # so spelling it out here produces a duplicated parameter.
        media_type="text/html" if wants_html else "text/markdown",
        headers={
            # `inline` for HTML because the client opens it in a tab to print;
            # `attachment` for Markdown because that one is a download.
            "Content-Disposition": (
                f'{"inline" if wants_html else "attachment"}; filename="{filename}"'
            ),
            "Cache-Control": "no-store",
        },
    )


MAX_PLUGINS = 200
MAX_PLUGIN_NAME = 120


def _clamp_free_text(
    analysis: MixAnalysis,
    plugins: Optional[List[OwnedPlugin]] = None,
    limit: int = 600,
) -> List[OwnedPlugin]:
    """Bound the model-facing strings carried in from the client.

    Finding titles and details are interpolated into the prompt, and so now are
    plugin names. Length is the lever that matters here — a clamped string
    cannot smuggle in a wall of instructions, and the output is
    schema-constrained regardless.

    Plugin names are the newest surface and the softest one: they are free text
    the user typed, they land in a list the model is told to prescribe from, and
    a list with no cap is a way to push the actual measurements out of context.
    Names are clipped, capability slugs are filtered against the known
    vocabulary (so an invented slug cannot describe itself into the brief), and
    the list is capped. Returns the clamped list; the analysis is clamped in
    place, as before.
    """
    for finding in analysis.findings:
        finding.title = finding.title[:200]
        finding.detail = finding.detail[:limit]
        for ev in finding.evidence:
            ev.label = ev.label[:120]
            ev.detail = ev.detail[:240]
    for dim in analysis.dimensions:
        dim.headline = dim.headline[:limit]
    analysis.filename = analysis.filename[:200]
    analysis.genre = analysis.genre[:60]
    analysis.mastering_blockers = [b[:240] for b in analysis.mastering_blockers[:12]]

    clamped: List[OwnedPlugin] = []
    for plugin in (plugins or [])[:MAX_PLUGINS]:
        name = " ".join(str(plugin.name or "").split())[:MAX_PLUGIN_NAME]
        if not name:
            continue
        manufacturer = " ".join(str(plugin.manufacturer or "").split())[:80] or None
        clamped.append(
            OwnedPlugin(
                name=name,
                manufacturer=manufacturer,
                category=" ".join(str(plugin.category or "").split())[:60],
                capabilities=capabilities.normalise(plugin.capabilities or []),
            )
        )
    return clamped


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
