"""
Database setup using SQLAlchemy + aiosqlite (SQLite)
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, JSON
from datetime import datetime

DATABASE_URL = "sqlite+aiosqlite:///data/annotator.db"

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    path = Column(String)
    file_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class OnnxModel(Base):
    __tablename__ = "onnx_models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    path = Column(String)
    classes_path = Column(String, nullable=True)
    input_shape = Column(JSON, nullable=True)
    output_shape = Column(JSON, nullable=True)
    classes = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, index=True)
    dataset_name = Column(String, index=True)
    image_name = Column(String, index=True)
    detections = Column(JSON, default=list)  # List of detection dicts
    status = Column(String, default="pending")  # pending | validated | exported
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


async def init_db():
    """Create all tables on startup."""
    import os
    os.makedirs("data", exist_ok=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """Dependency to get DB session."""
    async with AsyncSessionLocal() as session:
        yield session
