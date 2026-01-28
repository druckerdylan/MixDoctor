"""Analysis history endpoints."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from models import User, AnalysisHistory
from auth import get_current_user
from schemas import AnalysisHistoryResponse, AnalysisHistoryDetail

router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=List[AnalysisHistoryResponse])
async def list_history(
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get the user's analysis history."""
    history = (
        db.query(AnalysisHistory)
        .filter(AnalysisHistory.user_id == current_user.id)
        .order_by(AnalysisHistory.created_at.desc())
        .limit(limit)
        .all()
    )

    return [
        AnalysisHistoryResponse(
            id=h.id,
            filename=h.filename,
            genre=h.genre,
            health_score=h.health_score,
            summary=h.summary,
            created_at=h.created_at,
        )
        for h in history
    ]


@router.get("/{history_id}", response_model=AnalysisHistoryDetail)
async def get_history_detail(
    history_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get detailed analysis history entry."""
    entry = db.query(AnalysisHistory).filter(
        AnalysisHistory.id == history_id,
        AnalysisHistory.user_id == current_user.id,
    ).first()

    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="History entry not found",
        )

    return AnalysisHistoryDetail(
        id=entry.id,
        filename=entry.filename,
        genre=entry.genre,
        health_score=entry.health_score,
        summary=entry.summary,
        diagnoses=entry.diagnoses,
        metrics=entry.metrics,
        created_at=entry.created_at,
    )
