"""Worker-owned metadata; request and response bodies live in Runner log files."""

from datetime import datetime
from typing import Optional
from uuid import NAMESPACE_URL, uuid5

from pydantic import JsonValue
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from llmperf_backend.storage.base import Base


def worker_identity(runner_id: str, step_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"llmperf:worker:{runner_id}:{step_id}"))


class WorkerRecord(Base):
    __tablename__ = "workers"
    __table_args__ = (
        UniqueConstraint("runner_id", "step_id", name="uq_worker_runner_step"),
        CheckConstraint(
            "state IN ('created', 'submitted', 'succeeded', 'failed', 'cancelled', 'indeterminate')",
            name="ck_worker_state",
        ),
        Index("ix_worker_runner_state", "runner_id", "state"),
        Index("ix_worker_runner_created", "runner_id", "created_at", "id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    runner_id: Mapped[str] = mapped_column(
        ForeignKey("runners.id", ondelete="CASCADE"), nullable=False
    )
    step_id: Mapped[str] = mapped_column(String(128), nullable=False)
    generation: Mapped[int] = mapped_column(Integer, nullable=False)
    state: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    submitted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    worker_metadata: Mapped[dict[str, JsonValue]] = mapped_column(
        "metadata", JSONB, default=dict, nullable=False
    )
    log_key: Mapped[Optional[str]] = mapped_column(String(256))
    completion_hash: Mapped[Optional[str]] = mapped_column(String(64))
