from datetime import datetime
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any


class AudioMetrics(BaseModel):
    """Extracted audio analysis metrics."""
    # Loudness
    integrated_lufs: float
    short_term_lufs_max: float
    true_peak_dbfs: float
    loudness_range: float

    # Spectrum energy by band
    spectrum: Dict[str, float]  # sub, low, low_mid, mid, presence, air

    # Stereo
    stereo_correlation: float
    stereo_width: float

    # Dynamics
    crest_factor_db: float
    rms_db: float
    dynamic_range_db: float

    # Transients
    onset_density: float  # onsets per second
    estimated_tempo: float

    # File info
    duration_seconds: float
    sample_rate: int

    # Clipping
    clipped_samples: int = 0
    clip_percentage: float = 0.0
    max_consecutive_clips: int = 0
    has_clipping: bool = False
    clipping_severity: str = "good"

    # Phase
    mono_compatible: bool = True
    phase_status: str = "good"
    mono_energy_loss_db: float = 0.0


class Diagnosis(BaseModel):
    """A single mix diagnosis item."""
    category: str
    severity: str  # critical, warning, info (kept for backwards compatibility)
    priority: int  # 1=CRITICAL (must fix), 2=IMPORTANT (improves quality), 3=OPTIONAL (polish)
    issue: str
    suggestion: str
    affected_frequencies: Optional[str] = None
    # Engineering-focused fields
    what_it_affects: Optional[str] = None  # What elements/qualities this impacts
    likely_causes: Optional[List[str]] = None  # Why this is happening
    dont_do: Optional[str] = None  # Common mistakes to avoid
    done_when: Optional[str] = None  # How to know when to stop
    confidence: Optional[str] = "high"  # high, medium, low - how certain the diagnosis is


class QuickWin(BaseModel):
    """An actionable quick fix."""
    action: str
    impact: str  # high, medium, low
    plugin_suggestions: Optional[List[str]] = None


class ReferenceComparison(BaseModel):
    """Comparison metrics between user mix and reference track."""
    loudness_delta: float  # LUFS difference
    dynamics_delta: float  # Dynamic range difference
    stereo_width_delta: float  # Stereo width difference
    spectrum_comparison: Dict[str, float]  # Per-band delta
    overall_similarity: int  # 0-100 score
    key_differences: List[str]  # Human-readable differences


class AnalysisResult(BaseModel):
    """Complete analysis result from MixDoctor."""
    engineer_summary: str  # Conversational summary like a real engineer would give
    summary: str
    health_score: int  # 0-100
    diagnoses: List[Diagnosis]
    quick_wins: List[QuickWin]
    mastering_ready: bool
    mastering_notes: str
    metrics: AudioMetrics
    reference_comparison: Optional[ReferenceComparison] = None
    next_session_focus: Optional[List[str]] = None  # Clear next actions for the user


class AnalyzeRequest(BaseModel):
    """Request body for analysis (genre and reference come as form fields)."""
    genre: str
    reference: Optional[str] = None


# ===== Stems Analysis Schemas =====

class StemAnalysis(BaseModel):
    """Analysis result for a single stem."""
    stem_type: str
    metrics: AudioMetrics
    issues: List[Diagnosis]


class StemInteraction(BaseModel):
    """Interaction analysis between two stems."""
    stem_a: str
    stem_b: str
    interaction_type: str  # frequency_masking, phase_issue, balance
    severity: str  # critical, warning, info
    description: str
    suggestion: str
    affected_frequencies: Optional[str] = None


class StemsAnalysisResult(BaseModel):
    """Complete analysis result for stems upload."""
    engineer_summary: str  # Conversational summary like a real engineer would give
    summary: str
    overall_health_score: int  # 0-100
    stem_analyses: List[StemAnalysis]
    interactions: List[StemInteraction]
    quick_wins: List[QuickWin]
    mastering_ready: bool
    mastering_notes: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str


# ===== Auth Schemas =====

class UserCreate(BaseModel):
    """User registration request."""
    email: EmailStr
    username: str
    password: str


class UserLogin(BaseModel):
    """User login request."""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """User info response."""
    id: int
    email: str
    username: str
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str
    user: UserResponse


# ===== Plugin Schemas =====

class PluginCreate(BaseModel):
    """Create plugin request."""
    name: str
    category: str  # EQ, Compressor, Limiter, Reverb, Delay, Saturator, etc.
    manufacturer: Optional[str] = None


class PluginResponse(BaseModel):
    """Plugin response."""
    id: int
    name: str
    category: str
    manufacturer: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


# ===== History Schemas =====

class AnalysisHistoryResponse(BaseModel):
    """Analysis history list item."""
    id: int
    filename: str
    genre: str
    health_score: int
    summary: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class AnalysisHistoryDetail(BaseModel):
    """Detailed analysis history entry."""
    id: int
    filename: str
    genre: str
    health_score: int
    summary: Optional[str]
    diagnoses: Optional[List[Dict[str, Any]]]
    metrics: Optional[Dict[str, Any]]
    created_at: datetime

    class Config:
        from_attributes = True
