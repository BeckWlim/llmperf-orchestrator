"""Asynchronous PostgreSQL persistence for benchmark Runners and metrics."""

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    CheckConstraint,
    delete,
    func,
    or_,
    select,
    text,
    update,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, aliased, mapped_column

from llmperf_backend.models import DatabaseConfig
from llmperf_backend.planner import as_utc, next_fire_details
from llmperf_backend.task_compiler import TaskCompileContext, compile_task_definition


QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"
TERMINAL_STATUSES = {SUCCEEDED, FAILED, CANCELLED}
PLAN_ACTIVE = "active"
PLAN_PAUSED = "paused"
PLAN_COMPLETED = "completed"
PLAN_CANCELLED = "cancelled"
PLAN_STATUSES = {PLAN_ACTIVE, PLAN_PAUSED, PLAN_COMPLETED, PLAN_CANCELLED}
TASK_INSTANCE_PENDING_STATES = {"planned", "active"}
TASK_INSTANCE_TERMINAL_STATES = {"completed", "failed", "cancelled"}
JSON_DOCUMENT = JSONB
RUNNER_CLAIM_LOCK_ID = 0x4C4C4D50


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def json_safe(value: Any) -> Any:
    """Make benchmark JSON acceptable to PostgreSQL's strict JSON encoder."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    scalar_value = getattr(value, "item", None)
    if callable(scalar_value):
        return json_safe(scalar_value())
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


class Base(DeclarativeBase):
    pass


class BenchmarkCampaignRecord(Base):
    __tablename__ = "benchmark_campaigns"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    tags: Mapped[Dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False)
    created_by: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )


class BenchmarkRunnerPlanRecord(Base):
    __tablename__ = "benchmark_runner_plans"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'paused', 'completed', 'cancelled')",
            name="ck_runner_plan_status",
        ),
        CheckConstraint(
            "ends_at IS NOT NULL OR max_occurrences IS NOT NULL",
            name="ck_runner_plan_boundary",
        ),
        CheckConstraint(
            "ends_at IS NULL OR ends_at > starts_at",
            name="ck_runner_plan_time_range",
        ),
        CheckConstraint(
            "overlap_policy IN ('queue', 'skip')",
            name="ck_runner_plan_overlap_policy",
        ),
        Index(
            "ix_runner_plan_due",
            "next_fire_at",
            postgresql_where=text("status = 'active'"),
        ),
        Index("ix_runner_plan_campaign", "campaign_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    campaign_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("benchmark_campaigns.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default=PLAN_ACTIVE, nullable=False)
    timezone: Mapped[str] = mapped_column(String(64), nullable=False)
    recurrence: Mapped[Dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False)
    overlap_policy: Mapped[str] = mapped_column(String(20), nullable=False)
    runner_template: Mapped[Dict[str, Any]] = mapped_column(
        JSON_DOCUMENT, nullable=False
    )
    template_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    starts_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ends_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    max_occurrences: Mapped[Optional[int]] = mapped_column(Integer)
    next_fire_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_fire_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    occurrence_cursor: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    emitted_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    skipped_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    misfire_grace_seconds: Mapped[int] = mapped_column(
        Integer, default=60, nullable=False
    )
    created_by: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class BenchmarkTaskDefinitionRecord(Base):
    __tablename__ = "benchmark_task_definitions"
    __table_args__ = (
        Index("ix_task_definition_campaign", "campaign_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    campaign_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("benchmark_campaigns.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    compiler: Mapped[str] = mapped_column(String(40), nullable=False)
    config: Mapped[Dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False)
    runner_template: Mapped[Dict[str, Any]] = mapped_column(
        JSON_DOCUMENT, nullable=False
    )
    created_by: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class BenchmarkTaskInstanceRecord(Base):
    __tablename__ = "benchmark_task_instances"
    __table_args__ = (
        UniqueConstraint(
            "definition_id", "instance_key", name="uq_task_instance_key"
        ),
        CheckConstraint(
            "state IN ('planned', 'active', 'completed', 'failed', 'cancelled')",
            name="ck_task_instance_state",
        ),
        Index("ix_task_instance_campaign", "campaign_id", "compiler", "state"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    definition_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("benchmark_task_definitions.id", ondelete="CASCADE"),
        nullable=False,
    )
    campaign_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("benchmark_campaigns.id", ondelete="CASCADE"),
        nullable=False,
    )
    compiler: Mapped[str] = mapped_column(String(40), nullable=False)
    instance_key: Mapped[str] = mapped_column(String(512), nullable=False)
    state: Mapped[str] = mapped_column(String(20), default="planned", nullable=False)
    spec: Mapped[Dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False)
    checkpoint: Mapped[Dict[str, Any]] = mapped_column(
        JSON_DOCUMENT, default=dict, nullable=False
    )
    outcome: Mapped[Dict[str, Any]] = mapped_column(
        JSON_DOCUMENT, default=dict, nullable=False
    )
    error: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class BenchmarkRunnerDispatchRecord(Base):
    __tablename__ = "benchmark_runner_dispatches"
    __table_args__ = (
        UniqueConstraint(
            "runner_plan_id", "dispatch_key", name="uq_runner_dispatch_plan_key"
        ),
        UniqueConstraint(
            "task_instance_id", "node_id", name="uq_runner_dispatch_task_node"
        ),
        UniqueConstraint("runner_id", name="uq_runner_dispatch_runner"),
        CheckConstraint(
            "state IN ('blocked', 'pending', 'emitted', 'cancelled')",
            name="ck_runner_dispatch_state",
        ),
        CheckConstraint(
            "parent_dispatch_id IS NULL OR parent_dispatch_id <> id",
            name="ck_runner_dispatch_not_self_parent",
        ),
        CheckConstraint(
            "(runner_plan_id IS NOT NULL AND task_instance_id IS NULL AND "
            "dispatch_key IS NOT NULL AND node_id IS NULL) OR "
            "(runner_plan_id IS NULL AND task_instance_id IS NOT NULL AND "
            "dispatch_key IS NULL AND node_id IS NOT NULL)",
            name="ck_runner_dispatch_owner",
        ),
        Index(
            "ix_runner_dispatch_due",
            "due_at",
            postgresql_where=text("state = 'pending'"),
        ),
        Index("ix_runner_dispatch_campaign", "campaign_id", "created_at"),
        Index("ix_runner_dispatch_parent", "parent_dispatch_id"),
        Index("ix_runner_dispatch_task", "task_instance_id", "node_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    campaign_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("benchmark_campaigns.id", ondelete="CASCADE"),
        nullable=False,
    )
    runner_plan_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("benchmark_runner_plans.id", ondelete="CASCADE")
    )
    dispatch_key: Mapped[Optional[str]] = mapped_column(String(80))
    task_instance_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("benchmark_task_instances.id", ondelete="CASCADE"),
    )
    node_id: Mapped[Optional[str]] = mapped_column(String(80))
    due_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    state: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    parent_dispatch_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("benchmark_runner_dispatches.id", ondelete="SET NULL"),
    )
    runner_template: Mapped[Dict[str, Any]] = mapped_column(
        JSON_DOCUMENT, nullable=False
    )
    lineage: Mapped[Dict[str, Any]] = mapped_column(
        JSON_DOCUMENT, default=dict, nullable=False
    )
    runner_id: Mapped[Optional[str]] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    emitted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class BenchmarkRunnerRecord(Base):
    __tablename__ = "benchmark_runners"
    __table_args__ = (
        UniqueConstraint(
            "runner_plan_id", "plan_occurrence", name="uq_runner_plan_occurrence"
        ),
        Index(
            "ix_runners_queue_created_at",
            "created_at",
            postgresql_where=text("status = 'queued'"),
        ),
        Index("ix_runners_status_created_at", "status", "created_at"),
        Index(
            "ix_runners_running_campaign",
            "campaign_id",
            postgresql_where=text("status = 'running'"),
        ),
        Index("ix_runner_plan_time", "runner_plan_id", "scheduled_for"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    campaign_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("benchmark_campaigns.id", ondelete="SET NULL"),
        index=True,
    )
    runner_plan_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("benchmark_runner_plans.id", ondelete="SET NULL"),
    )
    plan_occurrence: Mapped[Optional[int]] = mapped_column(Integer)
    scheduled_for: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    plan_template_version: Mapped[Optional[int]] = mapped_column(Integer)
    label: Mapped[Optional[str]] = mapped_column(String(200))
    created_by: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    benchmark_config: Mapped[Dict[str, Any]] = mapped_column(
        JSON_DOCUMENT, nullable=False
    )
    user_metadata: Mapped[Dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    cancel_requested: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    scheduler_id: Mapped[Optional[str]] = mapped_column(String(100))
    process_id: Mapped[Optional[int]] = mapped_column(Integer)
    exit_code: Mapped[Optional[int]] = mapped_column(Integer)
    summary: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON_DOCUMENT)
    request_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    stdout: Mapped[Optional[str]] = mapped_column(Text)
    stderr: Mapped[Optional[str]] = mapped_column(Text)


class BenchmarkRequestRecord(Base):
    __tablename__ = "benchmark_request_results"
    __table_args__ = (
        CheckConstraint(
            "sequence >= 0",
            name="ck_request_result_sequence",
        ),
    )

    runner_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("benchmark_runners.id", ondelete="CASCADE"),
        nullable=False,
        primary_key=True,
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False, primary_key=True)
    metrics: Mapped[Dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class BenchmarkRunnerEventRecord(Base):
    __tablename__ = "benchmark_runner_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    runner_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("benchmark_runners.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    message: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class BenchmarkRunnerPlanEventRecord(Base):
    __tablename__ = "benchmark_runner_plan_events"
    __table_args__ = (
        Index("ix_runner_plan_event_time", "runner_plan_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    runner_plan_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("benchmark_runner_plans.id", ondelete="CASCADE"),
        nullable=False,
    )
    event_type: Mapped[str] = mapped_column(String(30), nullable=False)
    occurrence: Mapped[Optional[int]] = mapped_column(Integer)
    scheduled_for: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    runner_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("benchmark_runners.id", ondelete="SET NULL")
    )
    message: Mapped[Optional[str]] = mapped_column(Text)
    details: Mapped[Dict[str, Any]] = mapped_column(
        JSON_DOCUMENT, default=dict, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class UserRecord(Base):
    __tablename__ = "users"

    username: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[Optional[str]] = mapped_column(String(200))
    email: Mapped[Optional[str]] = mapped_column(String(320))
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_by: Mapped[str] = mapped_column(String(64), nullable=False)


class TrustedClientKeyRecord(Base):
    __tablename__ = "trusted_client_keys"

    key_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    username: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("users.username", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    public_key_pem: Mapped[str] = mapped_column(Text, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    valid_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    created_by: Mapped[str] = mapped_column(String(64), nullable=False)


class TrustedClientEventRecord(Base):
    __tablename__ = "trusted_client_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    key_id: Mapped[Optional[str]] = mapped_column(String(32))
    action: Mapped[str] = mapped_column(String(20), nullable=False)
    actor: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class Database:
    """Own the async engine and session factory."""

    def __init__(self, config: DatabaseConfig):
        engine_options: Dict[str, Any] = {
            "echo": config.echo,
            "pool_pre_ping": True,
        }
        engine_options.update(
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
        )
        self.engine: AsyncEngine = create_async_engine(config.url, **engine_options)
        self.sessions = async_sessionmaker(
            self.engine, expire_on_commit=False, class_=AsyncSession
        )

    async def create_schema(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def ping(self) -> bool:
        try:
            async with self.sessions() as session:
                await session.execute(select(1))
            return True
        except Exception:
            return False

    async def dispose(self) -> None:
        await self.engine.dispose()


def _runner_dict(runner: BenchmarkRunnerRecord) -> Dict[str, Any]:
    dispatch_lag = (
        (runner.started_at - runner.scheduled_for).total_seconds()
        if runner.started_at is not None and runner.scheduled_for is not None
        else None
    )
    return {
        "runner_id": runner.id,
        "campaign_id": runner.campaign_id,
        "runner_plan_id": runner.runner_plan_id,
        "plan_occurrence": runner.plan_occurrence,
        "scheduled_for": runner.scheduled_for,
        "dispatch_lag_seconds": dispatch_lag,
        "plan_template_version": runner.plan_template_version,
        "label": runner.label,
        "created_by": runner.created_by,
        "status": runner.status,
        "benchmark": runner.benchmark_config,
        "metadata": runner.user_metadata,
        "created_at": runner.created_at,
        "started_at": runner.started_at,
        "finished_at": runner.finished_at,
        "heartbeat_at": runner.heartbeat_at,
        "cancel_requested": runner.cancel_requested,
        "scheduler_id": runner.scheduler_id,
        "summary": runner.summary,
        "request_count": runner.request_count,
        "error_message": runner.error_message,
        "stdout": runner.stdout,
        "stderr": runner.stderr,
        "worker": {
            "process_id": runner.process_id,
            "exit_code": runner.exit_code,
        },
    }


def _runner_list_dict(runner: BenchmarkRunnerRecord) -> Dict[str, Any]:
    """Return the small, stable projection used by Runner listings."""

    benchmark = runner.benchmark_config or {}
    summary = runner.summary or {}
    results = summary.get("results") or {}
    outcome = summary.get("outcome") or {}
    dispatch_lag = (
        (runner.started_at - runner.scheduled_for).total_seconds()
        if runner.started_at is not None and runner.scheduled_for is not None
        else None
    )
    return {
        "runner_id": runner.id,
        "campaign_id": runner.campaign_id,
        "runner_plan_id": runner.runner_plan_id,
        "plan_occurrence": runner.plan_occurrence,
        "scheduled_for": runner.scheduled_for,
        "dispatch_lag_seconds": dispatch_lag,
        "label": runner.label,
        "created_by": runner.created_by,
        "status": runner.status,
        "provider": benchmark.get("provider"),
        "model": benchmark.get("model"),
        "requests": {
            "started": outcome.get(
                "requests_started", results.get("num_requests_started")
            ),
            "completed": outcome.get(
                "requests_completed", results.get("num_completed_requests")
            ),
            "failed": outcome.get("requests_failed", results.get("number_errors")),
            "error_rate": results.get("error_rate"),
        },
        "created_at": runner.created_at,
        "started_at": runner.started_at,
        "finished_at": runner.finished_at,
        "scheduler_id": runner.scheduler_id,
        "worker": {
            "process_id": runner.process_id,
            "exit_code": runner.exit_code,
        },
    }


def _runner_plan_dict(plan: BenchmarkRunnerPlanRecord) -> Dict[str, Any]:
    next_fire_local = (
        plan.next_fire_at.astimezone(ZoneInfo(plan.timezone)).isoformat()
        if plan.next_fire_at is not None
        else None
    )
    return {
        "runner_plan_id": plan.id,
        "campaign_id": plan.campaign_id,
        "name": plan.name,
        "status": plan.status,
        "timezone": plan.timezone,
        "recurrence": plan.recurrence,
        "overlap_policy": plan.overlap_policy,
        "runner": plan.runner_template,
        "template_version": plan.template_version,
        "starts_at": plan.starts_at,
        "ends_at": plan.ends_at,
        "max_occurrences": plan.max_occurrences,
        "next_fire_at": plan.next_fire_at,
        "next_fire_local": next_fire_local,
        "last_fire_at": plan.last_fire_at,
        "occurrence_cursor": plan.occurrence_cursor,
        "emitted_count": plan.emitted_count,
        "skipped_count": plan.skipped_count,
        "misfire_grace_seconds": plan.misfire_grace_seconds,
        "created_by": plan.created_by,
        "created_at": plan.created_at,
        "updated_at": plan.updated_at,
    }


def _task_definition_dict(
    definition: BenchmarkTaskDefinitionRecord,
) -> Dict[str, Any]:
    return {
        "task_definition_id": definition.id,
        "campaign_id": definition.campaign_id,
        "name": definition.name,
        "compiler": definition.compiler,
        "definition": definition.config,
        "runner": definition.runner_template,
        "created_by": definition.created_by,
        "created_at": definition.created_at,
    }


def _task_instance_dict(
    instance: BenchmarkTaskInstanceRecord,
) -> Dict[str, Any]:
    return {
        "task_instance_id": instance.id,
        "task_definition_id": instance.definition_id,
        "campaign_id": instance.campaign_id,
        "compiler": instance.compiler,
        "instance_key": instance.instance_key,
        "state": instance.state,
        "spec": instance.spec,
        "checkpoint": instance.checkpoint,
        "outcome": instance.outcome,
        "error": instance.error,
        "created_at": instance.created_at,
        "updated_at": instance.updated_at,
    }


def _dispatch_dict(dispatch: BenchmarkRunnerDispatchRecord) -> Dict[str, Any]:
    return {
        "dispatch_id": dispatch.id,
        "campaign_id": dispatch.campaign_id,
        "runner_plan_id": dispatch.runner_plan_id,
        "dispatch_key": dispatch.dispatch_key,
        "task_instance_id": dispatch.task_instance_id,
        "node_id": dispatch.node_id,
        "due_at": dispatch.due_at,
        "state": dispatch.state,
        "parent_dispatch_id": dispatch.parent_dispatch_id,
        "lineage": dispatch.lineage,
        "runner_id": dispatch.runner_id,
        "created_at": dispatch.created_at,
        "emitted_at": dispatch.emitted_at,
    }


class RunnerRepository:
    """Transactional Runner state and result operations."""

    def __init__(self, database: Database):
        self.database = database

    @staticmethod
    def _event(runner_id: str, status: str, message: Optional[str] = None):
        return BenchmarkRunnerEventRecord(
            runner_id=runner_id, status=status, message=message
        )

    @staticmethod
    def _plan_event(
        runner_plan_id: str,
        event_type: str,
        occurrence: Optional[int] = None,
        scheduled_for: Optional[datetime] = None,
        runner_id: Optional[str] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkRunnerPlanEventRecord:
        return BenchmarkRunnerPlanEventRecord(
            runner_plan_id=runner_plan_id,
            event_type=event_type,
            occurrence=occurrence,
            scheduled_for=scheduled_for,
            runner_id=runner_id,
            message=message,
            details=json_safe(details or {}),
        )

    def _dst_events(
        self,
        session: AsyncSession,
        runner_plan_id: str,
        occurrence: int,
        adjustments: Sequence[Dict[str, Any]],
        planner_id: Optional[str] = None,
    ) -> None:
        for adjustment in adjustments:
            details = dict(adjustment)
            details["planner_id"] = planner_id
            session.add(
                self._plan_event(
                    runner_plan_id,
                    "dst_adjusted",
                    occurrence=occurrence,
                    message=(
                        f"DST policy {adjustment['policy']} applied to "
                        f"{adjustment['local_time']} {adjustment['timezone']}"
                    ),
                    details=details,
                )
            )

    @staticmethod
    def _new_runner_plan(
        campaign_id: str,
        payload: Dict[str, Any],
        runner_template: Dict[str, Any],
        created_by: str,
        default_starts_at: datetime,
    ) -> Tuple[BenchmarkRunnerPlanRecord, Sequence[Dict[str, Any]]]:
        immediate_first = payload.get("starts_at") is None
        starts_at = as_utc(payload.get("starts_at") or default_starts_at)
        ends_at = as_utc(payload["ends_at"]) if payload.get("ends_at") else None
        recurrence = json_safe(payload["recurrence"])
        if immediate_first:
            first_fire, adjustments = starts_at, []
        else:
            first_fire, adjustments = next_fire_details(
                payload["timezone"], recurrence, starts_at, after=None
            )
        plan_status = PLAN_ACTIVE
        if ends_at is not None and first_fire > ends_at:
            plan_status = PLAN_COMPLETED
            first_fire = None
        return (
            BenchmarkRunnerPlanRecord(
                id=str(uuid4()),
                campaign_id=campaign_id,
                name=payload["name"],
                status=plan_status,
                timezone=payload["timezone"],
                recurrence=recurrence,
                overlap_policy=payload["overlap_policy"],
                runner_template=json_safe(runner_template),
                starts_at=starts_at,
                ends_at=ends_at,
                max_occurrences=payload.get("max_occurrences"),
                next_fire_at=first_fire,
                misfire_grace_seconds=payload.get("misfire_grace_seconds", 60),
                created_by=created_by,
            ),
            adjustments,
        )

    @staticmethod
    def _new_dispatch(
        campaign_id: str,
        due_at: Optional[datetime],
        runner_template: Dict[str, Any],
        lineage: Optional[Dict[str, Any]] = None,
        state: str = "pending",
        parent_dispatch_id: Optional[str] = None,
        runner_plan_id: Optional[str] = None,
        dispatch_key: Optional[str] = None,
        task_instance_id: Optional[str] = None,
        node_id: Optional[str] = None,
    ) -> BenchmarkRunnerDispatchRecord:
        return BenchmarkRunnerDispatchRecord(
            id=str(uuid4()),
            campaign_id=campaign_id,
            runner_plan_id=runner_plan_id,
            dispatch_key=dispatch_key,
            due_at=as_utc(due_at) if due_at is not None else None,
            state=state,
            parent_dispatch_id=parent_dispatch_id,
            task_instance_id=task_instance_id,
            node_id=node_id,
            runner_template=json_safe(runner_template),
            lineage=json_safe(lineage or {}),
        )

    async def create_runner(
        self,
        benchmark: Dict[str, Any],
        metadata: Dict[str, Any],
        created_by: str,
        campaign_id: Optional[str] = None,
        label: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        runner = BenchmarkRunnerRecord(
            id=str(uuid4()),
            campaign_id=campaign_id,
            label=label,
            created_by=created_by,
            status=QUEUED,
            benchmark_config=json_safe(benchmark),
            user_metadata=json_safe(metadata),
        )
        async with self.database.sessions() as session, session.begin():
            if campaign_id is not None:
                campaign = await session.get(BenchmarkCampaignRecord, campaign_id)
                if campaign is None:
                    return None
            session.add(runner)
            # No ORM relationship is declared between Runners and events. Flush the
            # parent explicitly so PostgreSQL never sees the event INSERT first.
            await session.flush()
            session.add(self._event(runner.id, QUEUED, "Runner accepted"))
            await session.flush()
        return _runner_dict(runner)

    async def create_runners(
        self, campaign_id: str, runners: Sequence[Dict[str, Any]], created_by: str
    ) -> Optional[List[Dict[str, Any]]]:
        records = [
            BenchmarkRunnerRecord(
                id=str(uuid4()),
                campaign_id=campaign_id,
                label=runner.get("label"),
                created_by=created_by,
                status=QUEUED,
                benchmark_config=json_safe(runner["benchmark"]),
                user_metadata=json_safe(runner.get("metadata", {})),
            )
            for runner in runners
        ]
        async with self.database.sessions() as session, session.begin():
            campaign = await session.get(BenchmarkCampaignRecord, campaign_id)
            if campaign is None:
                return None
            session.add_all(records)
            # Establish every parent row before inserting FK-dependent events.
            await session.flush()
            session.add_all(
                self._event(runner.id, QUEUED, "Runner accepted in Campaign")
                for runner in records
            )
            await session.flush()
        return [_runner_dict(runner) for runner in records]

    async def create_campaign(
        self,
        name: str,
        description: Optional[str],
        tags: Dict[str, Any],
        created_by: str,
    ) -> Dict[str, Any]:
        campaign = BenchmarkCampaignRecord(
            id=str(uuid4()),
            name=name,
            description=description,
            tags=json_safe(tags),
            created_by=created_by,
        )
        async with self.database.sessions() as session, session.begin():
            session.add(campaign)
            await session.flush()
        return self._campaign_dict(campaign)

    async def create_campaign_workload(
        self,
        name: str,
        description: Optional[str],
        tags: Dict[str, Any],
        runners: Sequence[Dict[str, Any]],
        runner_plans: Sequence[Dict[str, Any]],
        task_definitions: Sequence[Dict[str, Any]],
        created_by: str,
    ) -> Dict[str, Any]:
        """Create a Campaign and its validated workload in one transaction."""

        campaign = BenchmarkCampaignRecord(
            id=str(uuid4()),
            name=name,
            description=description,
            tags=json_safe(tags),
            created_by=created_by,
        )
        records = [
            BenchmarkRunnerRecord(
                id=str(uuid4()),
                campaign_id=campaign.id,
                label=runner.get("label"),
                created_by=created_by,
                status=QUEUED,
                benchmark_config=json_safe(runner["benchmark"]),
                user_metadata=json_safe(runner.get("metadata", {})),
            )
            for runner in runners
        ]
        plans_with_adjustments = []
        plans = []
        definition_records: List[BenchmarkTaskDefinitionRecord] = []
        instance_records: List[BenchmarkTaskInstanceRecord] = []
        dispatch_records: List[BenchmarkRunnerDispatchRecord] = []
        async with self.database.sessions() as session, session.begin():
            database_now = utcnow()
            if runner_plans or task_definitions:
                database_now = as_utc(
                    (await session.execute(select(func.now()))).scalar_one()
                )
            if runner_plans:
                plans_with_adjustments = [
                    self._new_runner_plan(
                        campaign.id,
                        runner_plan["plan"],
                        runner_plan["runner_template"],
                        created_by,
                        database_now,
                    )
                    for runner_plan in runner_plans
                ]
                plans = [item[0] for item in plans_with_adjustments]
            session.add(campaign)
            await session.flush()
            for submitted in task_definitions:
                definition_payload = dict(submitted["definition"])
                definition_payload.pop("runner", None)
                definition = BenchmarkTaskDefinitionRecord(
                    id=str(uuid4()),
                    campaign_id=campaign.id,
                    name=definition_payload["name"],
                    compiler="task-graph/v1",
                    config=json_safe(definition_payload),
                    runner_template=json_safe(submitted["runner_template"]),
                    created_by=created_by,
                )
                session.add(definition)
                await session.flush()
                blueprints = compile_task_definition(
                    TaskCompileContext(
                        definition_id=definition.id,
                        definition_name=definition.name,
                        definition=definition_payload,
                        runner_template=submitted["runner_template"],
                        database_now=database_now,
                        created_by=created_by,
                    )
                )
                for blueprint in blueprints:
                    instance_records.append(
                        BenchmarkTaskInstanceRecord(
                            id=blueprint.instance_id,
                            definition_id=definition.id,
                            campaign_id=campaign.id,
                            compiler=definition.compiler,
                            instance_key=blueprint.instance_key,
                            state="planned",
                            spec=json_safe(
                                {
                                    "dimensions": blueprint.dimensions,
                                    "trial_index": blueprint.trial_index,
                                }
                            ),
                            checkpoint={},
                            outcome={"nodes": {}},
                        )
                    )
                    for node in blueprint.nodes:
                        dispatch_records.append(
                            BenchmarkRunnerDispatchRecord(
                            id=node.dispatch_id,
                            campaign_id=campaign.id,
                            due_at=database_now if not node.dependencies else None,
                            state="pending" if not node.dependencies else "blocked",
                            parent_dispatch_id=(
                                node.dependencies[0] if node.dependencies else None
                            ),
                            task_instance_id=blueprint.instance_id,
                            node_id=node.node_id,
                            runner_template=json_safe(node.runner_template),
                            lineage=json_safe(
                                {
                                    "created_by": created_by,
                                    "dependencies": node.dependencies,
                                    "after_seconds": node.after_seconds,
                                    "role": node.role,
                                    "payload_id": node.payload_id,
                                }
                            ),
                        )
                        )
                definition_records.append(definition)
            session.add_all(records)
            session.add_all(plans)
            session.add_all(instance_records)
            session.add_all(dispatch_records)
            await session.flush()
            session.add_all(
                self._event(runner.id, QUEUED, "Runner accepted in Campaign")
                for runner in records
            )
            for plan, adjustments in plans_with_adjustments:
                session.add(
                    self._plan_event(
                        plan.id,
                        "created",
                        scheduled_for=plan.next_fire_at,
                        message="RunnerPlan accepted in Campaign",
                    )
                )
                self._dst_events(session, plan.id, 0, adjustments)
            await session.flush()
        return {
            "campaign": self._campaign_dict(campaign),
            "items": [_runner_dict(runner) for runner in records],
            "runner_plans": [_runner_plan_dict(plan) for plan in plans],
            "task_definitions": [
                _task_definition_dict(definition)
                for definition in definition_records
            ],
        }

    async def create_runner_plan(
        self,
        campaign_id: str,
        payload: Dict[str, Any],
        runner_template: Dict[str, Any],
        created_by: str,
    ) -> Optional[Dict[str, Any]]:
        async with self.database.sessions() as session, session.begin():
            campaign = await session.get(BenchmarkCampaignRecord, campaign_id)
            if campaign is None:
                return None
            database_now = as_utc(
                (await session.execute(select(func.now()))).scalar_one()
            )
            plan, adjustments = self._new_runner_plan(
                campaign_id,
                payload,
                runner_template,
                created_by,
                database_now,
            )
            session.add(plan)
            await session.flush()
            session.add(
                self._plan_event(
                    plan.id,
                    "created",
                    scheduled_for=plan.next_fire_at,
                    message="RunnerPlan created",
                )
            )
            self._dst_events(session, plan.id, 0, adjustments)
        return _runner_plan_dict(plan)

    async def list_runner_plans(
        self,
        status: Optional[str],
        campaign_id: Optional[str],
        limit: int,
        offset: int,
    ) -> List[Dict[str, Any]]:
        statement = select(BenchmarkRunnerPlanRecord)
        if status is not None:
            statement = statement.where(BenchmarkRunnerPlanRecord.status == status)
        if campaign_id is not None:
            statement = statement.where(
                BenchmarkRunnerPlanRecord.campaign_id == campaign_id
            )
        statement = (
            statement.order_by(BenchmarkRunnerPlanRecord.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        async with self.database.sessions() as session:
            plans = (await session.execute(statement)).scalars().all()
            return [_runner_plan_dict(plan) for plan in plans]

    async def get_runner_plan(self, runner_plan_id: str) -> Optional[Dict[str, Any]]:
        async with self.database.sessions() as session:
            plan = await session.get(BenchmarkRunnerPlanRecord, runner_plan_id)
            return _runner_plan_dict(plan) if plan is not None else None

    async def get_plan_events(
        self, runner_plan_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        async with self.database.sessions() as session:
            plan = await session.get(BenchmarkRunnerPlanRecord, runner_plan_id)
            if plan is None:
                return None
            statement = (
                select(BenchmarkRunnerPlanEventRecord)
                .where(BenchmarkRunnerPlanEventRecord.runner_plan_id == runner_plan_id)
                .order_by(BenchmarkRunnerPlanEventRecord.id.asc())
            )
            events = (await session.execute(statement)).scalars().all()
            return [
                {
                    "event_type": event.event_type,
                    "occurrence": event.occurrence,
                    "scheduled_for": event.scheduled_for,
                    "runner_id": event.runner_id,
                    "message": event.message,
                    "details": event.details,
                    "created_at": event.created_at,
                }
                for event in events
            ]

    async def change_runner_plan(
        self, runner_plan_id: str, action: str
    ) -> Optional[Dict[str, Any]]:
        transitions = {
            "pause": ({PLAN_ACTIVE}, PLAN_PAUSED),
            "resume": ({PLAN_PAUSED}, PLAN_ACTIVE),
            "cancel": ({PLAN_ACTIVE, PLAN_PAUSED}, PLAN_CANCELLED),
        }
        allowed, target = transitions[action]
        async with self.database.sessions() as session, session.begin():
            statement = (
                select(BenchmarkRunnerPlanRecord)
                .where(BenchmarkRunnerPlanRecord.id == runner_plan_id)
                .with_for_update()
            )
            plan = (await session.execute(statement)).scalar_one_or_none()
            if plan is None:
                return None
            if plan.status not in allowed:
                response = _runner_plan_dict(plan)
                response["transition_error"] = (
                    f"cannot {action} RunnerPlan in {plan.status} state"
                )
                return response
            plan.status = target
            plan.updated_at = utcnow()
            if action == "cancel":
                await session.execute(
                    update(BenchmarkRunnerDispatchRecord)
                    .where(
                        BenchmarkRunnerDispatchRecord.runner_plan_id == plan.id,
                        BenchmarkRunnerDispatchRecord.state == "pending",
                    )
                    .values(state="cancelled")
                )
            session.add(
                self._plan_event(
                    plan.id,
                    "cancelled" if action == "cancel" else f"{action}d",
                    scheduled_for=plan.next_fire_at,
                    message=f"RunnerPlan {action}d",
                )
            )
            await session.flush()
            return _runner_plan_dict(plan)

    async def materialize_due_work(
        self, limit: int, planner_id: Optional[str] = None
    ) -> int:
        """Materialize due plans and task nodes as ordinary queued Runners."""

        async with self.database.sessions() as session, session.begin():
            database_now = as_utc(
                (await session.execute(select(func.now()))).scalar_one()
            )
            statement = (
                select(BenchmarkRunnerPlanRecord)
                .where(
                    BenchmarkRunnerPlanRecord.status == PLAN_ACTIVE,
                    BenchmarkRunnerPlanRecord.next_fire_at <= func.now(),
                )
                .order_by(BenchmarkRunnerPlanRecord.next_fire_at.asc())
                .with_for_update(skip_locked=True)
                .limit(limit)
            )
            plans = list((await session.execute(statement)).scalars().all())
            for plan in plans:
                await self._materialize_plan(session, plan, database_now, planner_id)
            await session.flush()
            dispatch_statement = (
                select(BenchmarkRunnerDispatchRecord)
                .where(
                    BenchmarkRunnerDispatchRecord.state == "pending",
                    BenchmarkRunnerDispatchRecord.due_at <= func.now(),
                )
                .order_by(
                    BenchmarkRunnerDispatchRecord.due_at.asc(),
                    BenchmarkRunnerDispatchRecord.created_at.asc(),
                )
                .with_for_update(skip_locked=True)
                .limit(limit)
            )
            dispatches = list(
                (await session.execute(dispatch_statement)).scalars().all()
            )
            for dispatch in dispatches:
                await self._emit_dispatch(session, dispatch, database_now, planner_id)
            return len(dispatches)

    async def _emit_dispatch(
        self,
        session: AsyncSession,
        dispatch: BenchmarkRunnerDispatchRecord,
        database_now: datetime,
        planner_id: Optional[str],
    ) -> None:
        template = dispatch.runner_template
        lineage = dispatch.lineage or {}
        runner = BenchmarkRunnerRecord(
            id=str(uuid4()),
            campaign_id=dispatch.campaign_id,
            runner_plan_id=dispatch.runner_plan_id,
            plan_occurrence=lineage.get("plan_occurrence"),
            scheduled_for=dispatch.due_at,
            plan_template_version=lineage.get("plan_template_version"),
            label=template.get("label"),
            created_by=str(lineage.get("created_by") or "planner"),
            status=QUEUED,
            benchmark_config=json_safe(template["benchmark"]),
            user_metadata=json_safe(template.get("metadata", {})),
        )
        session.add(runner)
        await session.flush()
        dispatch.state = "emitted"
        dispatch.runner_id = runner.id
        dispatch.emitted_at = database_now
        session.add(
            self._event(runner.id, QUEUED, f"Emitted by Planner {planner_id or '-'}")
        )
        if dispatch.runner_plan_id is not None:
            plan = await session.get(BenchmarkRunnerPlanRecord, dispatch.runner_plan_id)
            if plan is not None:
                plan.emitted_count += 1
                plan.updated_at = database_now
                session.add(
                    self._plan_event(
                        plan.id,
                        "emitted",
                        occurrence=lineage.get("plan_occurrence"),
                        scheduled_for=dispatch.due_at,
                        runner_id=runner.id,
                        message="Runner emitted through durable dispatch",
                        details={"planner_id": planner_id},
                    )
                )

    async def _materialize_plan(
        self,
        session: AsyncSession,
        plan: BenchmarkRunnerPlanRecord,
        database_now: datetime,
        planner_id: Optional[str],
    ) -> int:
        planned_for = as_utc(plan.next_fire_at)
        grace_boundary = database_now - timedelta(seconds=plan.misfire_grace_seconds)
        skipped_start = plan.occurrence_cursor
        while planned_for < grace_boundary:
            plan.skipped_count += 1
            plan.occurrence_cursor += 1
            if self._plan_exhausted(plan):
                plan.status = PLAN_COMPLETED
                plan.next_fire_at = None
                break
            planned_for, adjustments = next_fire_details(
                plan.timezone, plan.recurrence, plan.starts_at, planned_for
            )
            self._dst_events(
                session,
                plan.id,
                plan.occurrence_cursor,
                adjustments,
                planner_id,
            )
            if plan.ends_at is not None and planned_for > as_utc(plan.ends_at):
                plan.status = PLAN_COMPLETED
                plan.next_fire_at = None
                break
            plan.next_fire_at = planned_for
        skipped = plan.occurrence_cursor - skipped_start
        if skipped:
            session.add(
                self._plan_event(
                    plan.id,
                    "misfire_skipped",
                    occurrence=skipped_start,
                    message=f"Skipped {skipped} expired occurrence(s)",
                    details={"count": skipped, "planner_id": planner_id},
                )
            )
        if plan.status == PLAN_COMPLETED or planned_for > database_now:
            plan.updated_at = database_now
            if plan.status == PLAN_COMPLETED:
                session.add(
                    self._plan_event(
                        plan.id,
                        "completed",
                        occurrence=plan.occurrence_cursor,
                        scheduled_for=plan.last_fire_at,
                        message="RunnerPlan reached its execution boundary",
                        details={"planner_id": planner_id},
                    )
                )
            return 0

        occurrence = plan.occurrence_cursor
        template = plan.runner_template
        should_emit = True
        if plan.overlap_policy == "skip":
            active_runners = (
                await session.execute(
                    select(func.count())
                    .select_from(BenchmarkRunnerRecord)
                    .where(
                        BenchmarkRunnerRecord.runner_plan_id == plan.id,
                        BenchmarkRunnerRecord.status.in_({QUEUED, RUNNING}),
                    )
                )
            ).scalar_one()
            pending_dispatches = (
                await session.execute(
                    select(func.count())
                    .select_from(BenchmarkRunnerDispatchRecord)
                    .where(
                        BenchmarkRunnerDispatchRecord.runner_plan_id == plan.id,
                        BenchmarkRunnerDispatchRecord.state == "pending",
                    )
                )
            ).scalar_one()
            should_emit = active_runners == 0 and pending_dispatches == 0

        dispatch = None
        if should_emit:
            dispatch = self._new_dispatch(
                plan.campaign_id,
                planned_for,
                template,
                {
                    "plan_occurrence": occurrence,
                    "plan_template_version": plan.template_version,
                    "created_by": plan.created_by,
                },
                runner_plan_id=plan.id,
                dispatch_key=str(occurrence),
            )
            session.add(dispatch)
        else:
            plan.skipped_count += 1

        plan.last_fire_at = planned_for
        plan.occurrence_cursor += 1
        if self._plan_exhausted(plan):
            plan.status = PLAN_COMPLETED
            plan.next_fire_at = None
        else:
            candidate, adjustments = next_fire_details(
                plan.timezone, plan.recurrence, plan.starts_at, planned_for
            )
            self._dst_events(
                session,
                plan.id,
                plan.occurrence_cursor,
                adjustments,
                planner_id,
            )
            if plan.ends_at is not None and candidate > as_utc(plan.ends_at):
                plan.status = PLAN_COMPLETED
                plan.next_fire_at = None
            else:
                plan.next_fire_at = candidate
        plan.updated_at = database_now
        session.add(
            self._plan_event(
                plan.id,
                "dispatch_scheduled" if dispatch is not None else "overlap_skipped",
                occurrence=occurrence,
                scheduled_for=planned_for,
                message=(
                    "Occurrence compiled into durable dispatch"
                    if dispatch is not None
                    else "Occurrence skipped by overlap_policy=skip"
                ),
                details={
                    "planner_id": planner_id,
                    "overlap_policy": plan.overlap_policy,
                },
            )
        )
        if plan.status == PLAN_COMPLETED:
            session.add(
                self._plan_event(
                    plan.id,
                    "completed",
                    occurrence=plan.occurrence_cursor,
                    scheduled_for=plan.last_fire_at,
                    message="RunnerPlan reached its execution boundary",
                    details={"planner_id": planner_id},
                )
            )
        return 0

    @staticmethod
    def _plan_exhausted(plan: BenchmarkRunnerPlanRecord) -> bool:
        return (
            plan.max_occurrences is not None
            and plan.occurrence_cursor >= plan.max_occurrences
        )

    @staticmethod
    def _campaign_dict(campaign: BenchmarkCampaignRecord) -> Dict[str, Any]:
        return {
            "campaign_id": campaign.id,
            "name": campaign.name,
            "description": campaign.description,
            "tags": campaign.tags,
            "created_by": campaign.created_by,
            "created_at": campaign.created_at,
        }

    async def get_campaign(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        async with self.database.sessions() as session:
            campaign = await session.get(BenchmarkCampaignRecord, campaign_id)
            return self._campaign_dict(campaign) if campaign is not None else None

    async def list_campaigns(self, limit: int, offset: int) -> List[Dict[str, Any]]:
        statement = (
            select(BenchmarkCampaignRecord)
            .order_by(BenchmarkCampaignRecord.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        async with self.database.sessions() as session:
            campaigns = (await session.execute(statement)).scalars().all()
            statuses: Dict[str, Dict[str, int]] = {
                campaign.id: {} for campaign in campaigns
            }
            plan_statuses: Dict[str, Dict[str, int]] = {
                campaign.id: {} for campaign in campaigns
            }
            task_statuses: Dict[str, Dict[str, int]] = {
                campaign.id: {} for campaign in campaigns
            }
            dispatch_statuses: Dict[str, Dict[str, int]] = {
                campaign.id: {} for campaign in campaigns
            }
            if campaigns:
                campaign_ids = [campaign.id for campaign in campaigns]
                status_statement = (
                    select(
                        BenchmarkRunnerRecord.campaign_id,
                        BenchmarkRunnerRecord.status,
                        func.count(),
                    )
                    .where(BenchmarkRunnerRecord.campaign_id.in_(campaign_ids))
                    .group_by(
                        BenchmarkRunnerRecord.campaign_id,
                        BenchmarkRunnerRecord.status,
                    )
                )
                for campaign_id, status, count in await session.execute(
                    status_statement
                ):
                    statuses[str(campaign_id)][str(status)] = int(count)
                plan_statement = (
                    select(
                        BenchmarkRunnerPlanRecord.campaign_id,
                        BenchmarkRunnerPlanRecord.status,
                        func.count(),
                    )
                    .where(BenchmarkRunnerPlanRecord.campaign_id.in_(campaign_ids))
                    .group_by(
                        BenchmarkRunnerPlanRecord.campaign_id,
                        BenchmarkRunnerPlanRecord.status,
                    )
                )
                for campaign_id, plan_status, count in await session.execute(
                    plan_statement
                ):
                    plan_statuses[str(campaign_id)][str(plan_status)] = int(count)
                task_statement = (
                    select(
                        BenchmarkTaskInstanceRecord.campaign_id,
                        BenchmarkTaskInstanceRecord.state,
                        func.count(),
                    )
                    .where(
                        BenchmarkTaskInstanceRecord.campaign_id.in_(campaign_ids)
                    )
                    .group_by(
                        BenchmarkTaskInstanceRecord.campaign_id,
                        BenchmarkTaskInstanceRecord.state,
                    )
                )
                for campaign_id, task_state, count in await session.execute(
                    task_statement
                ):
                    task_statuses[str(campaign_id)][str(task_state)] = int(
                        count
                    )
                dispatch_statement = (
                    select(
                        BenchmarkRunnerDispatchRecord.campaign_id,
                        BenchmarkRunnerDispatchRecord.state,
                        func.count(),
                    )
                    .where(BenchmarkRunnerDispatchRecord.campaign_id.in_(campaign_ids))
                    .group_by(
                        BenchmarkRunnerDispatchRecord.campaign_id,
                        BenchmarkRunnerDispatchRecord.state,
                    )
                )
                for campaign_id, dispatch_state, count in await session.execute(
                    dispatch_statement
                ):
                    dispatch_statuses[str(campaign_id)][str(dispatch_state)] = int(
                        count
                    )
            responses = []
            for campaign in campaigns:
                response = self._campaign_dict(campaign)
                response.update(
                    self._campaign_runtime(
                        statuses[campaign.id],
                        plan_statuses[campaign.id],
                        task_statuses[campaign.id],
                        dispatch_statuses[campaign.id],
                    )
                )
                responses.append(response)
            return responses

    @staticmethod
    def _campaign_runtime(
        statuses: Union[Sequence[str], Mapping[str, int]],
        plan_statuses: Union[Sequence[str], Mapping[str, int]] = (),
        task_statuses: Union[Sequence[str], Mapping[str, int]] = (),
        dispatch_statuses: Union[Sequence[str], Mapping[str, int]] = (),
    ) -> Dict[str, Any]:
        runner_states = (QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED)
        plan_states = (
            PLAN_ACTIVE,
            PLAN_PAUSED,
            PLAN_COMPLETED,
            PLAN_CANCELLED,
        )
        if isinstance(statuses, Mapping):
            counts = {status: int(statuses.get(status, 0)) for status in runner_states}
        else:
            counts = {status: statuses.count(status) for status in runner_states}
        if isinstance(plan_statuses, Mapping):
            plan_counts = {
                plan_status: int(plan_statuses.get(plan_status, 0))
                for plan_status in plan_states
            }
        else:
            plan_counts = {
                plan_status: plan_statuses.count(plan_status)
                for plan_status in plan_states
            }
        task_states = ("planned", "active", "completed", "failed", "cancelled")
        if isinstance(task_statuses, Mapping):
            task_counts = {
                task_state: int(task_statuses.get(task_state, 0))
                for task_state in task_states
            }
        else:
            task_counts = {
                task_state: task_statuses.count(task_state)
                for task_state in task_states
            }
        runner_count = sum(counts.values())
        runner_plan_count = sum(plan_counts.values())
        task_instance_count = sum(task_counts.values())
        pending_task_instances = sum(
            task_counts[state] for state in TASK_INSTANCE_PENDING_STATES
        )
        dispatch_states = ("blocked", "pending", "emitted", "cancelled")
        if isinstance(dispatch_statuses, Mapping):
            dispatch_counts = {
                state: int(dispatch_statuses.get(state, 0)) for state in dispatch_states
            }
        else:
            dispatch_counts = {
                state: dispatch_statuses.count(state) for state in dispatch_states
            }
        dispatch_count = sum(dispatch_counts.values())
        pending_dispatches = dispatch_counts["blocked"] + dispatch_counts["pending"]
        if counts[RUNNING]:
            campaign_status = RUNNING
        elif counts[QUEUED]:
            campaign_status = QUEUED
        elif pending_task_instances or pending_dispatches:
            campaign_status = "planned"
        elif plan_counts[PLAN_ACTIVE]:
            campaign_status = "planned"
        elif plan_counts[PLAN_PAUSED]:
            campaign_status = PLAN_PAUSED
        elif not runner_count and not runner_plan_count:
            if task_instance_count:
                campaign_status = (
                    CANCELLED
                    if task_counts["cancelled"] == task_instance_count
                    else "completed"
                )
            elif dispatch_count:
                campaign_status = (
                    CANCELLED
                    if dispatch_counts["cancelled"] == dispatch_count
                    else "completed"
                )
            else:
                campaign_status = "empty"
        elif not runner_count:
            campaign_status = (
                CANCELLED
                if plan_counts[PLAN_CANCELLED] == runner_plan_count
                else "completed"
            )
        elif counts[CANCELLED] == runner_count and (
            runner_count or plan_counts[PLAN_CANCELLED]
        ):
            campaign_status = CANCELLED
        else:
            campaign_status = "completed"

        if campaign_status in {RUNNING, QUEUED, "planned", PLAN_PAUSED}:
            campaign_outcome = "pending"
        elif task_counts["failed"]:
            campaign_outcome = (
                FAILED
                if task_counts["failed"] == task_instance_count
                else "partial_failed"
            )
        elif not runner_count:
            campaign_outcome = CANCELLED if campaign_status == CANCELLED else "no_runs"
        elif counts[FAILED] == runner_count:
            campaign_outcome = FAILED
        elif counts[FAILED]:
            campaign_outcome = "partial_failed"
        elif counts[SUCCEEDED] == runner_count:
            campaign_outcome = SUCCEEDED
        elif counts[CANCELLED]:
            campaign_outcome = CANCELLED
        else:
            campaign_outcome = "no_runs"
        return {
            "status": campaign_status,
            "outcome": campaign_outcome,
            "has_failures": bool(counts[FAILED] or task_counts["failed"]),
            "runner_count": runner_count,
            "status_counts": counts,
            "runner_plan_count": runner_plan_count,
            "runner_plan_status_counts": plan_counts,
            "task_instance_count": task_instance_count,
            "task_instance_status_counts": task_counts,
            "dispatch_count": dispatch_count,
            "dispatch_status_counts": dispatch_counts,
        }

    async def get_campaign_status(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        async with self.database.sessions() as session:
            campaign = await session.get(BenchmarkCampaignRecord, campaign_id)
            if campaign is None:
                return None
            statement = (
                select(BenchmarkRunnerRecord.status, func.count())
                .where(BenchmarkRunnerRecord.campaign_id == campaign_id)
                .group_by(BenchmarkRunnerRecord.status)
            )
            statuses = {
                str(runner_status): int(count)
                for runner_status, count in await session.execute(statement)
            }
            plan_statement = (
                select(BenchmarkRunnerPlanRecord.status, func.count())
                .where(BenchmarkRunnerPlanRecord.campaign_id == campaign_id)
                .group_by(BenchmarkRunnerPlanRecord.status)
            )
            plan_statuses = {
                str(plan_status): int(count)
                for plan_status, count in await session.execute(plan_statement)
            }
            task_statement = (
                select(BenchmarkTaskInstanceRecord.state, func.count())
                .where(BenchmarkTaskInstanceRecord.campaign_id == campaign_id)
                .group_by(BenchmarkTaskInstanceRecord.state)
            )
            task_statuses = {
                str(task_state): int(count)
                for task_state, count in await session.execute(task_statement)
            }
            dispatch_statement = (
                select(BenchmarkRunnerDispatchRecord.state, func.count())
                .where(BenchmarkRunnerDispatchRecord.campaign_id == campaign_id)
                .group_by(BenchmarkRunnerDispatchRecord.state)
            )
            dispatch_statuses = {
                str(dispatch_state): int(count)
                for dispatch_state, count in await session.execute(dispatch_statement)
            }
            response = self._campaign_dict(campaign)
            response.update(
                self._campaign_runtime(
                    statuses, plan_statuses, task_statuses, dispatch_statuses
                )
            )
            return response

    async def request_cancel_campaign(
        self, campaign_id: str
    ) -> Optional[Dict[str, Any]]:
        async with self.database.sessions() as session, session.begin():
            campaign = await session.get(BenchmarkCampaignRecord, campaign_id)
            if campaign is None:
                return None
            statement = (
                select(BenchmarkRunnerRecord)
                .where(BenchmarkRunnerRecord.campaign_id == campaign_id)
                .with_for_update()
            )
            runners = list((await session.execute(statement)).scalars().all())
            now = utcnow()
            plan_statement = (
                select(BenchmarkRunnerPlanRecord)
                .where(BenchmarkRunnerPlanRecord.campaign_id == campaign_id)
                .with_for_update()
            )
            plans = list((await session.execute(plan_statement)).scalars().all())
            task_statement = (
                select(BenchmarkTaskInstanceRecord)
                .where(BenchmarkTaskInstanceRecord.campaign_id == campaign_id)
                .with_for_update()
            )
            instances = list(
                (await session.execute(task_statement)).scalars().all()
            )
            dispatch_statement = (
                select(BenchmarkRunnerDispatchRecord)
                .where(BenchmarkRunnerDispatchRecord.campaign_id == campaign_id)
                .with_for_update()
            )
            dispatches = list(
                (await session.execute(dispatch_statement)).scalars().all()
            )
            for plan in plans:
                if plan.status not in {PLAN_ACTIVE, PLAN_PAUSED}:
                    continue
                plan.status = PLAN_CANCELLED
                plan.updated_at = now
                session.add(
                    self._plan_event(
                        plan.id,
                        "cancelled",
                        scheduled_for=plan.next_fire_at,
                        message="RunnerPlan cancelled with Campaign",
                    )
                )
            for runner in runners:
                if runner.status in TERMINAL_STATUSES:
                    continue
                runner.cancel_requested = True
                if runner.status == QUEUED:
                    runner.status = CANCELLED
                    runner.finished_at = now
                    message = "Cancelled with Campaign before execution"
                    event_status = CANCELLED
                else:
                    message = "Cancellation requested with Campaign"
                    event_status = RUNNING
                session.add(self._event(runner.id, event_status, message))
            for instance in instances:
                if instance.state in {"completed", "failed", "cancelled"}:
                    continue
                instance.state = "cancelled"
                instance.error = "Campaign cancelled"
                instance.updated_at = now
            for dispatch in dispatches:
                if dispatch.state in {"blocked", "pending"}:
                    dispatch.state = "cancelled"
            await session.flush()
            response = self._campaign_dict(campaign)
            response.update(
                self._campaign_runtime(
                    [runner.status for runner in runners],
                    [plan.status for plan in plans],
                    [instance.state for instance in instances],
                    [dispatch.state for dispatch in dispatches],
                )
            )
            return response

    async def get_runner(self, runner_id: str) -> Optional[Dict[str, Any]]:
        async with self.database.sessions() as session:
            runner = await session.get(BenchmarkRunnerRecord, runner_id)
            return _runner_dict(runner) if runner is not None else None

    async def list_runners(
        self,
        status: Optional[str],
        limit: int,
        offset: int,
        full: bool = False,
        campaign_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        statement = select(BenchmarkRunnerRecord)
        if status:
            statement = statement.where(BenchmarkRunnerRecord.status == status)
        if campaign_id:
            statement = statement.where(
                BenchmarkRunnerRecord.campaign_id == campaign_id
            )
        statement = statement.order_by(BenchmarkRunnerRecord.created_at.desc())
        statement = statement.limit(limit).offset(offset)
        async with self.database.sessions() as session:
            rows = (await session.execute(statement)).scalars().all()
            serializer = _runner_dict if full else _runner_list_dict
            return [serializer(row) for row in rows]

    async def claim_next(self, scheduler_id: str) -> Optional[Dict[str, Any]]:
        async with self.database.sessions() as session, session.begin():
            # Serialize the very small claim transaction across Backend replicas.
            # Without this lock, simultaneous slots can all observe the same campaign
            # counts before any claim commits and defeat fair sharing in a burst.
            await session.execute(
                select(func.pg_advisory_xact_lock(RUNNER_CLAIM_LOCK_ID))
            )
            running_runner = aliased(BenchmarkRunnerRecord)
            campaign_running_count = (
                select(func.count(running_runner.id))
                .where(
                    running_runner.status == RUNNING,
                    running_runner.campaign_id
                    == BenchmarkRunnerRecord.campaign_id,
                )
                .correlate(BenchmarkRunnerRecord)
                .scalar_subquery()
            )
            statement = (
                select(BenchmarkRunnerRecord)
                .where(BenchmarkRunnerRecord.status == QUEUED)
                .order_by(
                    campaign_running_count.asc(),
                    BenchmarkRunnerRecord.created_at.asc(),
                )
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            runner = (await session.execute(statement)).scalar_one_or_none()
            if runner is None:
                return None
            now = utcnow()
            runner.status = RUNNING
            runner.started_at = now
            runner.heartbeat_at = now
            runner.scheduler_id = scheduler_id
            session.add(
                self._event(runner.id, RUNNING, f"Claimed by Scheduler {scheduler_id}")
            )
            await session.flush()
            return _runner_dict(runner)

    async def set_process(self, runner_id: str, process_id: int) -> None:
        async with self.database.sessions() as session, session.begin():
            await session.execute(
                update(BenchmarkRunnerRecord)
                .where(BenchmarkRunnerRecord.id == runner_id)
                .values(process_id=process_id, heartbeat_at=utcnow())
            )

    async def set_logs(
        self, runner_id: str, exit_code: int, stdout: str, stderr: str
    ) -> None:
        async with self.database.sessions() as session, session.begin():
            await session.execute(
                update(BenchmarkRunnerRecord)
                .where(BenchmarkRunnerRecord.id == runner_id)
                .values(exit_code=exit_code, stdout=stdout, stderr=stderr)
            )

    async def heartbeat(self, runner_id: str) -> bool:
        async with self.database.sessions() as session, session.begin():
            statement = (
                update(BenchmarkRunnerRecord)
                .where(
                    BenchmarkRunnerRecord.id == runner_id,
                    BenchmarkRunnerRecord.status == RUNNING,
                )
                .values(heartbeat_at=utcnow())
                .returning(BenchmarkRunnerRecord.cancel_requested)
            )
            return bool((await session.execute(statement)).scalar_one_or_none())

    async def request_cancel(self, runner_id: str) -> Optional[Dict[str, Any]]:
        async with self.database.sessions() as session, session.begin():
            statement = (
                select(BenchmarkRunnerRecord)
                .where(BenchmarkRunnerRecord.id == runner_id)
                .with_for_update()
            )
            runner = (await session.execute(statement)).scalar_one_or_none()
            if runner is None:
                return None
            if runner.status in TERMINAL_STATUSES:
                return _runner_dict(runner)
            runner.cancel_requested = True
            if runner.status == QUEUED:
                runner.status = CANCELLED
                runner.finished_at = utcnow()
                await self._advance_task_instance(
                    session, runner, CANCELLED, runner.finished_at
                )
                session.add(
                    self._event(runner.id, CANCELLED, "Cancelled before execution")
                )
            else:
                session.add(self._event(runner.id, RUNNING, "Cancellation requested"))
            await session.flush()
            return _runner_dict(runner)

    async def _advance_task_instance(
        self,
        session: AsyncSession,
        runner: BenchmarkRunnerRecord,
        terminal_status: str,
        occurred_at: datetime,
        request_started_at: Optional[datetime] = None,
        request_completed_at: Optional[datetime] = None,
        request_prompt_hash: Optional[str] = None,
    ) -> None:
        """Advance a compiled invoke DAG without task-specific runtime branches."""

        context = (runner.user_metadata or {}).get("task")
        if not isinstance(context, Mapping):
            return
        instance_id = context.get("instance_id")
        node_id = context.get("node_id")
        payload_id = context.get("payload_id")
        if not all(isinstance(value, str) for value in (instance_id, node_id, payload_id)):
            return

        instance = (
            await session.execute(
                select(BenchmarkTaskInstanceRecord)
                .where(BenchmarkTaskInstanceRecord.id == instance_id)
                .with_for_update()
            )
        ).scalar_one_or_none()
        if instance is None:
            return
        source = (
            await session.execute(
                select(BenchmarkRunnerDispatchRecord)
                .where(BenchmarkRunnerDispatchRecord.runner_id == runner.id)
                .with_for_update()
            )
        ).scalar_one_or_none()
        if source is None:
            instance.state = "failed"
            instance.error = f"task node {node_id} has no originating Dispatch"
            return

        started_at = request_started_at or runner.started_at or occurred_at
        completed_at = request_completed_at or occurred_at
        source_lineage = dict(source.lineage or {})
        source_lineage["actual_started_at"] = started_at.isoformat()
        source_lineage["actual_completed_at"] = completed_at.isoformat()
        source.lineage = json_safe(source_lineage)

        outcome = dict(instance.outcome or {})
        nodes = dict(outcome.get("nodes") or {})
        nodes[node_id] = {
            "role": context.get("role"),
            "payload_id": payload_id,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "prompt_hash": request_prompt_hash,
            "status": terminal_status,
        }
        outcome["nodes"] = nodes
        instance.outcome = json_safe(outcome)
        instance.updated_at = occurred_at

        if instance.state in TASK_INSTANCE_TERMINAL_STATES:
            return
        if terminal_status != SUCCEEDED:
            instance.state = "cancelled" if terminal_status == CANCELLED else "failed"
            instance.error = f"task node {node_id} ended as {terminal_status}"
            await self._cancel_task_dispatches(session, instance_id)
            return
        if not request_prompt_hash:
            instance.state = "failed"
            instance.error = f"task node {node_id} omitted prompt_hash"
            await self._cancel_task_dispatches(session, instance_id)
            return

        checkpoint = dict(instance.checkpoint or {})
        payload_hashes = dict(checkpoint.get("payload_hashes") or {})
        expected_hash = payload_hashes.get(payload_id)
        if expected_hash is not None and expected_hash != request_prompt_hash:
            instance.state = "failed"
            instance.error = f"payload {payload_id} replay changed prompt_hash"
            await self._cancel_task_dispatches(session, instance_id)
            return
        payload_hashes[payload_id] = request_prompt_hash
        checkpoint["payload_hashes"] = payload_hashes
        instance.checkpoint = json_safe(checkpoint)

        rows = list(
            (
                await session.execute(
                    select(BenchmarkRunnerDispatchRecord, BenchmarkRunnerRecord)
                    .outerjoin(
                        BenchmarkRunnerRecord,
                        BenchmarkRunnerRecord.id == BenchmarkRunnerDispatchRecord.runner_id,
                    )
                    .where(
                        BenchmarkRunnerDispatchRecord.task_instance_id == instance_id
                    )
                    .with_for_update()
                )
            ).all()
        )
        statuses = {
            dispatch.id: (
                terminal_status
                if dispatch.id == source.id
                else runner_record.status
                if runner_record is not None
                else None
            )
            for dispatch, runner_record in rows
        }
        completions = {}
        for dispatch, runner_record in rows:
            if dispatch.id == source.id:
                completions[dispatch.id] = completed_at
                continue
            value = (dispatch.lineage or {}).get("actual_completed_at")
            if isinstance(value, str):
                try:
                    completions[dispatch.id] = as_utc(datetime.fromisoformat(value))
                    continue
                except ValueError:
                    pass
            if runner_record is not None and runner_record.finished_at is not None:
                completions[dispatch.id] = runner_record.finished_at

        for dispatch, _ in rows:
            if dispatch.state != "blocked":
                continue
            dependencies = list((dispatch.lineage or {}).get("dependencies") or [])
            if dependencies and all(statuses.get(item) == SUCCEEDED for item in dependencies):
                anchors = [completions.get(item) for item in dependencies]
                if any(anchor is None for anchor in anchors):
                    continue
                dispatch.due_at = max(anchors) + timedelta(
                    seconds=int((dispatch.lineage or {}).get("after_seconds", 0))
                )
                dispatch.state = "pending"

        instance.state = (
            "completed"
            if rows
            and all(
                dispatch.state == "emitted"
                and statuses.get(dispatch.id) == SUCCEEDED
                for dispatch, _ in rows
            )
            else "active"
        )
        instance.error = None

    async def _cancel_task_dispatches(
        self, session: AsyncSession, instance_id: str
    ) -> None:
        await session.execute(
            update(BenchmarkRunnerDispatchRecord)
            .where(
                BenchmarkRunnerDispatchRecord.task_instance_id == instance_id,
                BenchmarkRunnerDispatchRecord.state.in_(["blocked", "pending"]),
            )
            .values(state="cancelled")
        )
    async def complete_runner(
        self,
        runner_id: str,
        summary: Dict[str, Any],
        request_metrics: Sequence[Dict[str, Any]],
        exit_code: int,
        stdout: str,
        stderr: str,
        terminal_status: str = SUCCEEDED,
        error_message: Optional[str] = None,
    ) -> bool:
        if terminal_status not in {SUCCEEDED, FAILED}:
            raise ValueError(f"Unsupported result status: {terminal_status}")
        safe_summary = json_safe(summary)
        safe_requests = [json_safe(item) for item in request_metrics]
        async with self.database.sessions() as session, session.begin():
            statement = (
                select(BenchmarkRunnerRecord)
                .where(BenchmarkRunnerRecord.id == runner_id)
                .with_for_update()
            )
            runner = (await session.execute(statement)).scalar_one_or_none()
            if runner is None or runner.status != RUNNING or runner.cancel_requested:
                return False
            completed_at = utcnow()
            request_started_at = None
            request_completed_at = None
            request_prompt_hash = None
            context = (runner.user_metadata or {}).get("task")
            if isinstance(context, Mapping) and safe_requests:
                request_starts = []
                request_completions = []
                for request in safe_requests:
                    timing = request.get("request_timing") or {}
                    request_metadata = request.get("request_metadata") or {}
                    prompt_hash = request_metadata.get("prompt_hash")
                    if request_prompt_hash is None and isinstance(prompt_hash, str):
                        request_prompt_hash = prompt_hash
                    for timestamp_name, target in (
                        ("client_start_utc", request_starts),
                        ("completed_utc", request_completions),
                    ):
                        timestamp_value = timing.get(timestamp_name)
                        if not isinstance(timestamp_value, str):
                            continue
                        try:
                            target.append(
                                as_utc(datetime.fromisoformat(timestamp_value))
                            )
                        except ValueError:
                            pass
                if request_starts:
                    request_started_at = min(request_starts)
                if request_completions:
                    request_completed_at = max(request_completions)
            await session.execute(
                delete(BenchmarkRequestRecord).where(
                    BenchmarkRequestRecord.runner_id == runner_id
                )
            )
            session.add_all(
                BenchmarkRequestRecord(
                    runner_id=runner_id, sequence=index, metrics=metrics
                )
                for index, metrics in enumerate(safe_requests)
            )

            await session.execute(
                update(BenchmarkRunnerRecord)
                .where(BenchmarkRunnerRecord.id == runner_id)
                .values(
                    status=terminal_status,
                    finished_at=completed_at,
                    heartbeat_at=completed_at,
                    exit_code=exit_code,
                    summary=safe_summary,
                    request_count=len(safe_requests),
                    stdout=stdout,
                    stderr=stderr,
                    error_message=error_message,
                )
            )
            await self._advance_task_instance(
                session,
                runner,
                terminal_status,
                completed_at,
                request_started_at=request_started_at,
                request_completed_at=request_completed_at,
                request_prompt_hash=request_prompt_hash,
            )
            event_message = (
                "Results persisted"
                if terminal_status == SUCCEEDED
                else error_message or "Failed benchmark results persisted"
            )
            session.add(self._event(runner_id, terminal_status, event_message))
        return True

    async def finish_runner(
        self,
        runner_id: str,
        status: str,
        message: str,
        exit_code: Optional[int],
        stdout: str,
        stderr: str,
    ) -> bool:
        if status not in {FAILED, CANCELLED}:
            raise ValueError(f"Unsupported terminal status: {status}")
        async with self.database.sessions() as session, session.begin():
            statement = (
                select(BenchmarkRunnerRecord)
                .where(
                    BenchmarkRunnerRecord.id == runner_id,
                    BenchmarkRunnerRecord.status == RUNNING,
                )
                .with_for_update()
            )
            runner = (await session.execute(statement)).scalar_one_or_none()
            if runner is None:
                return False
            finished_at = utcnow()
            runner.status = status
            runner.finished_at = finished_at
            runner.heartbeat_at = finished_at
            runner.exit_code = exit_code
            runner.error_message = message
            runner.stdout = stdout
            runner.stderr = stderr
            await self._advance_task_instance(session, runner, status, finished_at)
            session.add(self._event(runner_id, status, message))
        return True

    async def requeue_runner(self, runner_id: str, message: str) -> None:
        async with self.database.sessions() as session, session.begin():
            result = await session.execute(
                update(BenchmarkRunnerRecord)
                .where(
                    BenchmarkRunnerRecord.id == runner_id,
                    BenchmarkRunnerRecord.status == RUNNING,
                )
                .values(
                    status=QUEUED,
                    scheduler_id=None,
                    process_id=None,
                    heartbeat_at=None,
                    started_at=None,
                )
            )
            if result.rowcount:
                session.add(self._event(runner_id, QUEUED, message))

    async def requeue_stale(self, stale_after_seconds: int) -> int:
        stale_before = utcnow() - timedelta(seconds=stale_after_seconds)
        async with self.database.sessions() as session, session.begin():
            statement = (
                update(BenchmarkRunnerRecord)
                .where(
                    BenchmarkRunnerRecord.status == RUNNING,
                    or_(
                        BenchmarkRunnerRecord.heartbeat_at.is_(None),
                        BenchmarkRunnerRecord.heartbeat_at < stale_before,
                    ),
                )
                .values(
                    status=QUEUED,
                    scheduler_id=None,
                    process_id=None,
                    heartbeat_at=None,
                    started_at=None,
                )
                .returning(BenchmarkRunnerRecord.id)
            )
            result = await session.execute(statement)
            runner_ids = list(result.scalars().all())
            session.add_all(
                self._event(
                    runner_id,
                    QUEUED,
                    "Stale Scheduler heartbeat; Runner requeued",
                )
                for runner_id in runner_ids
            )
            return len(runner_ids)

    async def get_results(
        self, runner_id: str, include_requests: bool
    ) -> Optional[Dict[str, Any]]:
        async with self.database.sessions() as session:
            runner = await session.get(BenchmarkRunnerRecord, runner_id)
            if runner is None:
                return None
            response: Dict[str, Any] = {
                "runner_id": runner.id,
                "status": runner.status,
                "summary": runner.summary,
                "request_count": runner.request_count,
            }
            if include_requests:
                statement = (
                    select(BenchmarkRequestRecord)
                    .where(BenchmarkRequestRecord.runner_id == runner_id)
                    .order_by(BenchmarkRequestRecord.sequence.asc())
                )
                records = (await session.execute(statement)).scalars().all()
                response["requests"] = [record.metrics for record in records]
            return response

    async def get_events(self, runner_id: str) -> Optional[List[Dict[str, Any]]]:
        if await self.get_runner(runner_id) is None:
            return None
        statement = (
            select(BenchmarkRunnerEventRecord)
            .where(BenchmarkRunnerEventRecord.runner_id == runner_id)
            .order_by(BenchmarkRunnerEventRecord.id.asc())
        )
        async with self.database.sessions() as session:
            records = (await session.execute(statement)).scalars().all()
            return [
                {
                    "status": record.status,
                    "message": record.message,
                    "created_at": record.created_at,
                }
                for record in records
            ]

    async def export_campaign(
        self, campaign_id: str, include_requests: bool
    ) -> Optional[Dict[str, Any]]:
        campaign = await self.get_campaign(campaign_id)
        if campaign is None:
            return None
        statement = (
            select(BenchmarkRunnerRecord)
            .where(BenchmarkRunnerRecord.campaign_id == campaign_id)
            .order_by(BenchmarkRunnerRecord.created_at.asc())
        )
        async with self.database.sessions() as session:
            runner_records = (await session.execute(statement)).scalars().all()
            plan_statement = (
                select(BenchmarkRunnerPlanRecord)
                .where(BenchmarkRunnerPlanRecord.campaign_id == campaign_id)
                .order_by(BenchmarkRunnerPlanRecord.created_at.asc())
            )
            plan_records = (await session.execute(plan_statement)).scalars().all()
            definition_statement = (
                select(BenchmarkTaskDefinitionRecord)
                .where(BenchmarkTaskDefinitionRecord.campaign_id == campaign_id)
                .order_by(BenchmarkTaskDefinitionRecord.created_at.asc())
            )
            definition_records = (
                (await session.execute(definition_statement)).scalars().all()
            )
            instance_statement = (
                select(BenchmarkTaskInstanceRecord)
                .where(BenchmarkTaskInstanceRecord.campaign_id == campaign_id)
                .order_by(
                    BenchmarkTaskInstanceRecord.definition_id.asc(),
                    BenchmarkTaskInstanceRecord.instance_key.asc(),
                )
            )
            instance_records = (
                (await session.execute(instance_statement)).scalars().all()
            )
            dispatch_statement = (
                select(BenchmarkRunnerDispatchRecord)
                .where(BenchmarkRunnerDispatchRecord.campaign_id == campaign_id)
                .order_by(BenchmarkRunnerDispatchRecord.created_at.asc())
            )
            dispatch_records = (
                (await session.execute(dispatch_statement)).scalars().all()
            )
            status_counts = {
                status: 0 for status in (QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED)
            }
            runners: List[Dict[str, Any]] = []
            request_map: Dict[str, List[Dict[str, Any]]] = {}
            if include_requests and runner_records:
                request_statement = (
                    select(BenchmarkRequestRecord)
                    .where(
                        BenchmarkRequestRecord.runner_id.in_(
                            [runner.id for runner in runner_records]
                        )
                    )
                    .order_by(
                        BenchmarkRequestRecord.runner_id.asc(),
                        BenchmarkRequestRecord.sequence.asc(),
                    )
                )
                records = (await session.execute(request_statement)).scalars().all()
                for record in records:
                    request_map.setdefault(record.runner_id, []).append(record.metrics)
            for runner in runner_records:
                status_counts[runner.status] = status_counts.get(runner.status, 0) + 1
                item = _runner_dict(runner)
                if include_requests:
                    item["requests"] = request_map.get(runner.id, [])
                runners.append(item)
            plan_status_counts = {
                plan_status: sum(plan.status == plan_status for plan in plan_records)
                for plan_status in (
                    PLAN_ACTIVE,
                    PLAN_PAUSED,
                    PLAN_COMPLETED,
                    PLAN_CANCELLED,
                )
            }
            task_status_counts = {
                task_state: sum(
                    instance.state == task_state for instance in instance_records
                )
                for task_state in (
                    "planned",
                    "active",
                    "completed",
                    "failed",
                    "cancelled",
                )
            }
            dispatch_status_counts = {
                dispatch_state: sum(
                    dispatch.state == dispatch_state for dispatch in dispatch_records
                )
                for dispatch_state in ("blocked", "pending", "emitted", "cancelled")
            }
            runtime = self._campaign_runtime(
                status_counts,
                plan_status_counts,
                task_status_counts,
                dispatch_status_counts,
            )
            instances = [
                _task_instance_dict(instance) for instance in instance_records
            ]
            dispatches = [_dispatch_dict(dispatch) for dispatch in dispatch_records]
            from llmperf.task_analysis import build_task_analyses

            task_analyses = build_task_analyses(
                instances,
                dispatches,
                {runner.id: _runner_dict(runner) for runner in runner_records},
            )
            return {
                "version": 6,
                "campaign": campaign,
                "aggregate": {
                    **runtime,
                    "completed_request_count": sum(
                        runner.request_count for runner in runner_records
                    ),
                },
                "runner_plans": [_runner_plan_dict(plan) for plan in plan_records],
                "task_definitions": [
                    _task_definition_dict(definition)
                    for definition in definition_records
                ],
                "task_instances": instances,
                "dispatches": dispatches,
                "task_analyses": task_analyses,
                "runners": runners,
            }

    async def get_trusted_client_by_key_id(
        self, key_id: str
    ) -> Optional[Dict[str, Any]]:
        now = utcnow()
        statement = (
            select(TrustedClientKeyRecord, UserRecord)
            .join(UserRecord, UserRecord.username == TrustedClientKeyRecord.username)
            .where(
                TrustedClientKeyRecord.key_id == key_id,
                TrustedClientKeyRecord.enabled.is_(True),
                or_(
                    TrustedClientKeyRecord.valid_until.is_(None),
                    TrustedClientKeyRecord.valid_until > now,
                ),
                UserRecord.enabled.is_(True),
            )
        )
        async with self.database.sessions() as session:
            row = (await session.execute(statement)).one_or_none()
            if row is None:
                return None
            key, user = row
            return {
                "username": user.username,
                "key_id": key.key_id,
                "public_key_pem": key.public_key_pem,
                "role": user.role,
            }

    async def list_trusted_clients(self) -> List[Dict[str, Any]]:
        async with self.database.sessions() as session:
            users = (
                (
                    await session.execute(
                        select(UserRecord).order_by(UserRecord.username.asc())
                    )
                )
                .scalars()
                .all()
            )
            keys = (
                (
                    await session.execute(
                        select(TrustedClientKeyRecord).order_by(
                            TrustedClientKeyRecord.created_at.asc()
                        )
                    )
                )
                .scalars()
                .all()
            )
            keys_by_user: Dict[str, List[Dict[str, Any]]] = {}
            for key in keys:
                keys_by_user.setdefault(key.username, []).append(
                    {
                        "key_id": key.key_id,
                        "enabled": key.enabled,
                        "valid_until": key.valid_until,
                        "created_at": key.created_at,
                        "created_by": key.created_by,
                    }
                )
            return [
                {
                    "username": user.username,
                    "display_name": user.display_name,
                    "email": user.email,
                    "role": user.role,
                    "enabled": user.enabled,
                    "created_at": user.created_at,
                    "updated_at": user.updated_at,
                    "updated_by": user.updated_by,
                    "keys": keys_by_user.get(user.username, []),
                }
                for user in users
            ]

    async def upsert_trusted_client(
        self,
        username: str,
        key_id: str,
        public_key_pem: str,
        role: str,
        display_name: Optional[str],
        email: Optional[str],
        actor: str,
        previous_key_grace_seconds: int,
    ) -> Optional[Dict[str, Any]]:
        now = utcnow()
        async with self.database.sessions() as session, session.begin():
            conflicting_key = await session.get(TrustedClientKeyRecord, key_id)
            if conflicting_key is not None and conflicting_key.username != username:
                return None
            user = await session.get(UserRecord, username)
            action = "created" if user is None else "updated"
            if user is None:
                user = UserRecord(
                    username=username,
                    display_name=display_name,
                    email=email,
                    role=role,
                    enabled=True,
                    updated_by=actor,
                )
                session.add(user)
                await session.flush()
            else:
                user.display_name = display_name
                user.email = email
                user.role = role
                user.enabled = True
                user.updated_at = now
                user.updated_by = actor

            current_keys = (
                (
                    await session.execute(
                        select(TrustedClientKeyRecord).where(
                            TrustedClientKeyRecord.username == username,
                            TrustedClientKeyRecord.enabled.is_(True),
                            TrustedClientKeyRecord.key_id != key_id,
                            or_(
                                TrustedClientKeyRecord.valid_until.is_(None),
                                TrustedClientKeyRecord.valid_until > now,
                            ),
                        )
                    )
                )
                .scalars()
                .all()
            )
            if current_keys:
                action = "rotated"
                expires_at = now + timedelta(seconds=previous_key_grace_seconds)
                for current_key in current_keys:
                    current_key.valid_until = expires_at

            key = await session.get(TrustedClientKeyRecord, key_id)
            if key is None:
                key = TrustedClientKeyRecord(
                    key_id=key_id,
                    username=username,
                    public_key_pem=public_key_pem,
                    enabled=True,
                    valid_until=None,
                    created_by=actor,
                )
                session.add(key)
            else:
                key.public_key_pem = public_key_pem
                key.enabled = True
                key.valid_until = None

            session.add(
                TrustedClientEventRecord(
                    username=username,
                    key_id=key_id,
                    action=action,
                    actor=actor,
                )
            )
            await session.flush()
            return {
                "username": user.username,
                "display_name": user.display_name,
                "email": user.email,
                "role": user.role,
                "enabled": user.enabled,
                "active_key_id": key.key_id,
                "updated_at": user.updated_at,
                "updated_by": user.updated_by,
            }

    async def revoke_trusted_client(
        self, username: str, actor: str
    ) -> Optional[Dict[str, Any]]:
        async with self.database.sessions() as session, session.begin():
            user = await session.get(UserRecord, username)
            if user is None:
                return None
            user.enabled = False
            user.updated_at = utcnow()
            user.updated_by = actor
            keys = (
                (
                    await session.execute(
                        select(TrustedClientKeyRecord).where(
                            TrustedClientKeyRecord.username == username
                        )
                    )
                )
                .scalars()
                .all()
            )
            for key in keys:
                key.enabled = False
            session.add(
                TrustedClientEventRecord(
                    username=username,
                    key_id=None,
                    action="revoked",
                    actor=actor,
                )
            )
            await session.flush()
            return {
                "username": user.username,
                "role": user.role,
                "enabled": user.enabled,
                "updated_at": user.updated_at,
                "updated_by": user.updated_by,
            }

    async def list_trusted_client_events(
        self, limit: int = 100
    ) -> List[Dict[str, Any]]:
        statement = (
            select(TrustedClientEventRecord)
            .order_by(TrustedClientEventRecord.id.desc())
            .limit(limit)
        )
        async with self.database.sessions() as session:
            records = (await session.execute(statement)).scalars().all()
            return [
                {
                    "username": record.username,
                    "key_id": record.key_id,
                    "action": record.action,
                    "actor": record.actor,
                    "created_at": record.created_at,
                }
                for record in records
            ]
