"""FastAPI backend for MixDoctor."""

import os
import tempfile
from typing import Optional, List

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from schemas import AnalysisResult, HealthResponse, StemsAnalysisResult
from audio_analyzer import analyze_audio, analyze_stem_interactions
from mix_doctor import analyze_with_claude, analyze_stems_with_claude
from database import init_db, get_db
from auth import get_current_user_optional
from models import User, UserPlugin, AnalysisHistory

# Load environment variables
load_dotenv()

# Verify API key is available
if not os.environ.get("ANTHROPIC_API_KEY"):
    print("WARNING: ANTHROPIC_API_KEY not set. Analysis will fail.")

app = FastAPI(
    title="MixDoctor API",
    description="AI-powered audio mix analysis",
    version="1.0.0",
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:5174",  # Vite alternate port
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:3000",
        "https://mix-doctor.vercel.app",  # Production frontend
        "https://mixdoctor.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from routers import auth as auth_router
from routers import plugins as plugins_router
from routers import history as history_router

app.include_router(auth_router.router)
app.include_router(plugins_router.router)
app.include_router(history_router.router)

# Supported audio formats
SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".aiff", ".aif"}
MAX_FILE_SIZE = 250 * 1024 * 1024  # 250MB

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup."""
    init_db()
    print("Database initialized.")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="1.0.0")


@app.get("/test-ai")
async def test_ai():
    """Test if Claude API is working."""
    try:
        from anthropic import Anthropic
        import httpx

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return {"status": "error", "error": "ANTHROPIC_API_KEY not set"}

        client = Anthropic(
            api_key=api_key,
            timeout=httpx.Timeout(30.0, connect=10.0)
        )

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=50,
            messages=[{"role": "user", "content": "Say 'API working' in 3 words or less"}]
        )
        return {"status": "success", "response": message.content[0].text}
    except Exception as e:
        import traceback
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}


# Bearer token extraction for optional auth
security = HTTPBearer(auto_error=False)


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_mix(
    file: UploadFile = File(..., description="Audio file to analyze"),
    genre: str = Form(..., description="Music genre (e.g., 'EDM', 'Hip Hop', 'Rock')"),
    reference: Optional[str] = Form(None, description="Optional reference track or notes"),
    reference_file: Optional[UploadFile] = File(None, description="Optional reference audio file for comparison"),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
):
    """
    Analyze an audio mix and return AI-powered feedback.

    Upload an audio file (MP3, WAV, M4A, FLAC, AIFF) and specify the genre.
    Optionally upload a reference track for comparison.
    Returns detailed analysis including:
    - Health score (0-100)
    - Specific diagnoses with severity levels
    - Quick wins for immediate improvement
    - Mastering readiness assessment
    - Reference comparison (if reference track provided)

    If authenticated, the analysis will use the user's plugins and history
    to provide personalized recommendations.
    """
    # Get current user (optional)
    current_user: Optional[User] = None
    if credentials:
        from auth import decode_token
        user_id = decode_token(credentials.credentials)
        if user_id:
            current_user = db.query(User).filter(User.id == user_id).first()

    # Fetch user's plugins and history if authenticated
    user_plugins = []
    user_history = []
    if current_user:
        user_plugins = db.query(UserPlugin).filter(
            UserPlugin.user_id == current_user.id
        ).order_by(UserPlugin.category, UserPlugin.name).all()

        user_history = db.query(AnalysisHistory).filter(
            AnalysisHistory.user_id == current_user.id
        ).order_by(AnalysisHistory.created_at.desc()).limit(5).all()

    # Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {ext}. Supported: {', '.join(SUPPORTED_FORMATS)}",
        )

    # Save uploaded file to temp location
    temp_file = None
    temp_reference_file = None
    try:
        # Create temp file with correct extension
        temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        content = await file.read()

        # Check file size
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB",
            )

        temp_file.write(content)
        temp_file.close()

        # Analyze audio
        try:
            print(f"Analyzing audio file: {temp_file.name}")
            metrics = analyze_audio(temp_file.name)
            print(f"Audio analysis complete: {metrics.integrated_lufs} LUFS")
        except Exception as e:
            import traceback
            print(f"Audio analysis error: {str(e)}")
            print(traceback.format_exc())
            raise HTTPException(
                status_code=422,
                detail=f"Failed to analyze audio: {str(e)}",
            )

        # Analyze reference file if provided
        reference_metrics = None
        if reference_file and reference_file.filename:
            ref_ext = os.path.splitext(reference_file.filename)[1].lower()
            if ref_ext not in SUPPORTED_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported reference format: {ref_ext}. Supported: {', '.join(SUPPORTED_FORMATS)}",
                )

            temp_reference_file = tempfile.NamedTemporaryFile(suffix=ref_ext, delete=False)
            ref_content = await reference_file.read()

            if len(ref_content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"Reference file too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB",
                )

            temp_reference_file.write(ref_content)
            temp_reference_file.close()

            try:
                reference_metrics = analyze_audio(temp_reference_file.name)
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"Failed to analyze reference audio: {str(e)}",
                )

        # Get AI analysis (pass plugins and history for personalization)
        try:
            result = analyze_with_claude(
                metrics,
                genre,
                reference,
                user_plugins=user_plugins,
                user_history=user_history,
                reference_metrics=reference_metrics,
            )
        except Exception as e:
            import traceback
            print(f"AI analysis error: {str(e)}")
            print(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"AI analysis failed: {str(e)}",
            )

        # Save to history if authenticated
        if current_user:
            history_entry = AnalysisHistory(
                user_id=current_user.id,
                filename=file.filename,
                genre=genre,
                health_score=result.health_score,
                summary=result.summary,
                diagnoses=[d.model_dump() for d in result.diagnoses],
                metrics=metrics.model_dump(),
            )
            db.add(history_entry)
            db.commit()

        return result

    finally:
        # Clean up temp files
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        if temp_reference_file and os.path.exists(temp_reference_file.name):
            os.unlink(temp_reference_file.name)


VALID_STEM_TYPES = {
    'kick', 'snare_hats', 'percussion', 'bass', 'vocals',
    'lead', 'pads', 'guitars', 'fx', 'other'
}


@app.post("/analyze-stems", response_model=StemsAnalysisResult)
async def analyze_stems(
    genre: str = Form(..., description="Music genre"),
    stem_types: str = Form(..., description="Comma-separated stem types"),
    stems: List[UploadFile] = File(..., description="Stem audio files"),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
):
    """
    Analyze individual stems from a mix.

    Upload 2+ stem files with their types for detailed analysis.
    Returns per-stem feedback, interaction analysis, and overall assessment.
    """
    # Parse stem types
    stem_type_list = [s.strip() for s in stem_types.split(',')]

    # Validate we have at least 2 stems
    if len(stems) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 stems are required for analysis",
        )

    if len(stems) != len(stem_type_list):
        raise HTTPException(
            status_code=400,
            detail=f"Number of files ({len(stems)}) must match number of stem types ({len(stem_type_list)})",
        )

    # Validate stem types
    for st in stem_type_list:
        if st not in VALID_STEM_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stem type: {st}. Valid types: {', '.join(VALID_STEM_TYPES)}",
            )

    # Get current user for plugins
    current_user: Optional[User] = None
    user_plugins = []
    if credentials:
        from auth import decode_token
        user_id = decode_token(credentials.credentials)
        if user_id:
            current_user = db.query(User).filter(User.id == user_id).first()
            if current_user:
                user_plugins = db.query(UserPlugin).filter(
                    UserPlugin.user_id == current_user.id
                ).order_by(UserPlugin.category, UserPlugin.name).all()

    # Process each stem
    temp_files = []
    stems_metrics = {}

    try:
        for i, (stem_file, stem_type) in enumerate(zip(stems, stem_type_list)):
            # Validate file
            if not stem_file.filename:
                raise HTTPException(status_code=400, detail=f"No filename for stem {i + 1}")

            ext = os.path.splitext(stem_file.filename)[1].lower()
            if ext not in SUPPORTED_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported format for {stem_type}: {ext}",
                )

            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            content = await stem_file.read()

            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"Stem {stem_type} too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB",
                )

            temp_file.write(content)
            temp_file.close()
            temp_files.append(temp_file.name)

            # Analyze stem
            try:
                metrics = analyze_audio(temp_file.name)
                stems_metrics[stem_type] = metrics
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"Failed to analyze {stem_type} stem: {str(e)}",
                )

        # Analyze interactions between stems
        interactions = analyze_stem_interactions(stems_metrics)

        # Get AI analysis
        try:
            result = analyze_stems_with_claude(
                stems_metrics,
                interactions,
                genre,
                user_plugins=user_plugins,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"AI analysis failed: {str(e)}",
            )

        return result

    finally:
        # Clean up temp files
        for temp_path in temp_files:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
