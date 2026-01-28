"""Plugin management endpoints."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from models import User, UserPlugin
from auth import get_current_user
from schemas import PluginCreate, PluginResponse

router = APIRouter(prefix="/plugins", tags=["plugins"])


@router.get("", response_model=List[PluginResponse])
async def list_plugins(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get all plugins for the current user."""
    plugins = db.query(UserPlugin).filter(UserPlugin.user_id == current_user.id).order_by(UserPlugin.category, UserPlugin.name).all()
    return [
        PluginResponse(
            id=p.id,
            name=p.name,
            category=p.category,
            manufacturer=p.manufacturer,
            created_at=p.created_at,
        )
        for p in plugins
    ]


@router.post("", response_model=PluginResponse, status_code=status.HTTP_201_CREATED)
async def add_plugin(
    plugin_data: PluginCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Add a plugin to the user's collection."""
    # Check for duplicate (same name for this user)
    existing = db.query(UserPlugin).filter(
        UserPlugin.user_id == current_user.id,
        UserPlugin.name == plugin_data.name,
    ).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Plugin already exists",
        )

    plugin = UserPlugin(
        user_id=current_user.id,
        name=plugin_data.name,
        category=plugin_data.category,
        manufacturer=plugin_data.manufacturer,
    )
    db.add(plugin)
    db.commit()
    db.refresh(plugin)

    return PluginResponse(
        id=plugin.id,
        name=plugin.name,
        category=plugin.category,
        manufacturer=plugin.manufacturer,
        created_at=plugin.created_at,
    )


@router.delete("/{plugin_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_plugin(
    plugin_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Remove a plugin from the user's collection."""
    plugin = db.query(UserPlugin).filter(
        UserPlugin.id == plugin_id,
        UserPlugin.user_id == current_user.id,
    ).first()

    if not plugin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Plugin not found",
        )

    db.delete(plugin)
    db.commit()
