"""Asynchronous PostgreSQL persistence for benchmark Runners and metrics."""

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    CheckConstraint,
    delete,
    func,
    or_,
    select,
    update,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from llmperf_backend.models import DatabaseConfig


QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"
TERMINAL_STATUSES = {SUCCEEDED, FAILED, CANCELLED}
JSON_DOCUMENT = JSON().with_variant(JSONB, "postgresql")


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


class BenchmarkRunnerRecord(Base):
    __tablename__ = "benchmark_runners"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    campaign_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("benchmark_campaigns.id", ondelete="SET NULL"),
        index=True,
    )
    label: Mapped[Optional[str]] = mapped_column(String(200))
    created_by: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
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
        if not config.url.startswith("sqlite+"):
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
    return {
        "runner_id": runner.id,
        "campaign_id": runner.campaign_id,
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
    return {
        "runner_id": runner.id,
        "campaign_id": runner.campaign_id,
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
            statuses: Dict[str, List[str]] = {campaign.id: [] for campaign in campaigns}
            if campaigns:
                status_statement = (
                    select(
                        BenchmarkRunnerRecord.campaign_id,
                        BenchmarkRunnerRecord.status,
                        func.count(),
                    )
                    .where(
                        BenchmarkRunnerRecord.campaign_id.in_(
                            [campaign.id for campaign in campaigns]
                        )
                    )
                    .group_by(
                        BenchmarkRunnerRecord.campaign_id,
                        BenchmarkRunnerRecord.status,
                    )
                )
                for campaign_id, status, count in await session.execute(
                    status_statement
                ):
                    statuses[str(campaign_id)].extend([str(status)] * int(count))
            responses = []
            for campaign in campaigns:
                response = self._campaign_dict(campaign)
                response.update(self._campaign_runtime(statuses[campaign.id]))
                responses.append(response)
            return responses

    @staticmethod
    def _campaign_runtime(statuses: Sequence[str]) -> Dict[str, Any]:
        counts = {
            status: statuses.count(status)
            for status in (QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED)
        }
        if not statuses:
            campaign_status = "empty"
        elif counts[RUNNING] or (counts[QUEUED] and len(statuses) != counts[QUEUED]):
            campaign_status = RUNNING
        elif counts[QUEUED]:
            campaign_status = QUEUED
        elif counts[FAILED]:
            campaign_status = FAILED
        elif counts[SUCCEEDED] == len(statuses):
            campaign_status = SUCCEEDED
        elif counts[CANCELLED] == len(statuses):
            campaign_status = CANCELLED
        else:
            campaign_status = "completed"
        return {
            "status": campaign_status,
            "runner_count": len(statuses),
            "status_counts": counts,
        }

    async def get_campaign_status(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        async with self.database.sessions() as session:
            campaign = await session.get(BenchmarkCampaignRecord, campaign_id)
            if campaign is None:
                return None
            statement = select(BenchmarkRunnerRecord.status).where(
                BenchmarkRunnerRecord.campaign_id == campaign_id
            )
            statuses = list((await session.execute(statement)).scalars().all())
            response = self._campaign_dict(campaign)
            response.update(self._campaign_runtime(statuses))
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
            await session.flush()
            response = self._campaign_dict(campaign)
            response.update(
                self._campaign_runtime([runner.status for runner in runners])
            )
            return response

    async def get_runner(self, runner_id: str) -> Optional[Dict[str, Any]]:
        async with self.database.sessions() as session:
            runner = await session.get(BenchmarkRunnerRecord, runner_id)
            return _runner_dict(runner) if runner is not None else None

    async def list_runners(
        self, status: Optional[str], limit: int, offset: int, full: bool = False
    ) -> List[Dict[str, Any]]:
        statement = select(BenchmarkRunnerRecord)
        if status:
            statement = statement.where(BenchmarkRunnerRecord.status == status)
        statement = statement.order_by(BenchmarkRunnerRecord.created_at.desc())
        statement = statement.limit(limit).offset(offset)
        async with self.database.sessions() as session:
            rows = (await session.execute(statement)).scalars().all()
            serializer = _runner_dict if full else _runner_list_dict
            return [serializer(row) for row in rows]

    async def claim_next(self, scheduler_id: str) -> Optional[Dict[str, Any]]:
        async with self.database.sessions() as session, session.begin():
            statement = (
                select(BenchmarkRunnerRecord)
                .where(BenchmarkRunnerRecord.status == QUEUED)
                .order_by(BenchmarkRunnerRecord.created_at.asc())
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
                session.add(
                    self._event(runner.id, CANCELLED, "Cancelled before execution")
                )
            else:
                session.add(self._event(runner.id, RUNNING, "Cancellation requested"))
            await session.flush()
            return _runner_dict(runner)

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
                    finished_at=utcnow(),
                    heartbeat_at=utcnow(),
                    exit_code=exit_code,
                    summary=safe_summary,
                    request_count=len(safe_requests),
                    stdout=stdout,
                    stderr=stderr,
                    error_message=error_message,
                )
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
            result = await session.execute(
                update(BenchmarkRunnerRecord)
                .where(
                    BenchmarkRunnerRecord.id == runner_id,
                    BenchmarkRunnerRecord.status == RUNNING,
                )
                .values(
                    status=status,
                    finished_at=utcnow(),
                    heartbeat_at=utcnow(),
                    exit_code=exit_code,
                    error_message=message,
                    stdout=stdout,
                    stderr=stderr,
                )
            )
            if not result.rowcount:
                return False
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
                item = {
                    "runner_id": runner.id,
                    "label": runner.label,
                    "status": runner.status,
                    "benchmark": runner.benchmark_config,
                    "metadata": runner.user_metadata,
                    "created_at": runner.created_at,
                    "started_at": runner.started_at,
                    "finished_at": runner.finished_at,
                    "summary": runner.summary,
                    "request_count": runner.request_count,
                    "error_message": runner.error_message,
                }
                if include_requests:
                    item["requests"] = request_map.get(runner.id, [])
                runners.append(item)
            return {
                "version": 1,
                "campaign": campaign,
                "aggregate": {
                    "runner_count": len(runner_records),
                    "status_counts": status_counts,
                    "completed_request_count": sum(
                        runner.request_count for runner in runner_records
                    ),
                },
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
