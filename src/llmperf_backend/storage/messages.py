"""Generic Runner messages and the chat-history specialization."""

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from pydantic import JsonValue
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from llmperf_backend.storage.base import Base


class MessageRecord(Base):
    __tablename__ = "runner_messages"
    __table_args__ = (
        Index(
            "ix_runner_message_destination", "runner_id", "destination", "created_at"
        ),
    )
    __mapper_args__ = {
        "polymorphic_on": "message_type",
        "polymorphic_identity": "message",
    }

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    runner_id: Mapped[str] = mapped_column(
        ForeignKey("runners.id", ondelete="CASCADE"), nullable=False
    )
    message_type: Mapped[str] = mapped_column(String(32), nullable=False)
    source: Mapped[Optional[str]] = mapped_column(String(128))
    destination: Mapped[str] = mapped_column(
        String(128), default="runner", nullable=False
    )
    topic: Mapped[str] = mapped_column(String(64), default="output", nullable=False)
    payload: Mapped[dict[str, JsonValue]] = mapped_column(
        JSONB, default=dict, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class MessageReceiptRecord(Base):
    __tablename__ = "runner_message_receipts"
    message_id: Mapped[str] = mapped_column(
        ForeignKey("runner_messages.id", ondelete="CASCADE"), primary_key=True
    )
    consumer: Mapped[str] = mapped_column(String(128), primary_key=True)
    acknowledged_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class ConversationMessageRecord(MessageRecord):
    """Chat-specific identity and ordering over the common message envelope."""

    __tablename__ = "conversation_messages"
    __mapper_args__ = {"polymorphic_identity": "conversation"}
    __table_args__ = (
        UniqueConstraint(
            "instance_id", "conversation", "position", name="uq_conversation_position"
        ),
        CheckConstraint(
            "position >= 0 AND role IN ('user', 'assistant')",
            name="ck_message_position_role",
        ),
    )
    id: Mapped[str] = mapped_column(
        ForeignKey("runner_messages.id", ondelete="CASCADE"), primary_key=True
    )
    instance_id: Mapped[str] = mapped_column(
        ForeignKey("task_instances.id", ondelete="CASCADE"), nullable=False
    )
    conversation: Mapped[str] = mapped_column(String(64), nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
