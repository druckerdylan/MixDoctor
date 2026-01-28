"""SQLAlchemy models for MixDoctor."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship

from database import Base


class User(Base):
    """User account model."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(50), nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    plugins = relationship("UserPlugin", back_populates="user", cascade="all, delete-orphan")
    history = relationship("AnalysisHistory", back_populates="user", cascade="all, delete-orphan")


class UserPlugin(Base):
    """User's available plugins."""
    __tablename__ = "user_plugins"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    category = Column(String(100), nullable=False)  # EQ, Compressor, Limiter, etc.
    manufacturer = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    user = relationship("User", back_populates="plugins")


class AnalysisHistory(Base):
    """History of mix analyses."""
    __tablename__ = "analysis_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    genre = Column(String(100), nullable=False)
    health_score = Column(Integer, nullable=False)
    summary = Column(Text, nullable=True)
    diagnoses = Column(JSON, nullable=True)  # Store as JSON array
    metrics = Column(JSON, nullable=True)  # Store AudioMetrics as JSON
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    user = relationship("User", back_populates="history")
