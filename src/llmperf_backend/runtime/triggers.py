"""Pure eligibility conditions shared by workflow and Runner progression."""

import asyncio
from uuid import NAMESPACE_URL, uuid5
from llmperf_backend.runtime.messages import RunnerMessage, RunnerMessageBus

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Mapping, Optional, Sequence


def utc_instant(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("trigger timestamps must include a UTC offset")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class Completion:
    succeeded: bool
    completed_at: datetime


@dataclass(frozen=True)
class Eligibility:
    ready: bool
    due_at: Optional[datetime] = None
    blocked: bool = False


class Trigger(ABC):
    """Evaluate durable facts without performing I/O or allocating execution slots."""

    @abstractmethod
    def evaluate(
        self, now: datetime, completions: Mapping[str, Completion]
    ) -> Eligibility: ...


@dataclass(frozen=True)
class DateTrigger(Trigger):
    due_at: datetime

    def evaluate(
        self, now: datetime, completions: Mapping[str, Completion]
    ) -> Eligibility:
        due_at = utc_instant(self.due_at)
        return Eligibility(utc_instant(now) >= due_at, due_at)


@dataclass(frozen=True)
class DelayTrigger(Trigger):
    anchor: datetime
    seconds: float

    def __post_init__(self) -> None:
        import math

        utc_instant(self.anchor)
        if not math.isfinite(self.seconds) or self.seconds < 0:
            raise ValueError("trigger delay must be finite and nonnegative")

    def evaluate(
        self, now: datetime, completions: Mapping[str, Completion]
    ) -> Eligibility:
        due_at = utc_instant(self.anchor) + timedelta(seconds=self.seconds)
        return DateTrigger(due_at).evaluate(now, completions)


@dataclass(frozen=True)
class DependencyTrigger(Trigger):
    dependencies: Sequence[str]
    created_at: datetime
    delay_seconds: float = 0.0

    def evaluate(
        self, now: datetime, completions: Mapping[str, Completion]
    ) -> Eligibility:
        if any(
            identifier in completions and not completions[identifier].succeeded
            for identifier in self.dependencies
        ):
            return Eligibility(False, blocked=True)
        if any(identifier not in completions for identifier in self.dependencies):
            return Eligibility(False)
        anchor = max(
            (completions[identifier].completed_at for identifier in self.dependencies),
            default=self.created_at,
        )
        return DelayTrigger(anchor, self.delay_seconds).evaluate(now, completions)


class TimerSource:
    def __init__(self, bus: RunnerMessageBus):
        self.bus = bus

    async def fire(self, trigger_id: str, due_at: datetime) -> None:
        due_instant = utc_instant(due_at)
        while True:
            remaining_seconds = (
                due_instant - datetime.now(timezone.utc)
            ).total_seconds()
            if remaining_seconds <= 0:
                break
            # Recheck wall time after adjustments; asyncio waits themselves use
            # the event loop's monotonic clock. This task occupies no request slot.
            await asyncio.sleep(min(remaining_seconds, 60.0))
        await self.bus.publish(
            RunnerMessage(
                runner_id=self.bus.runner_id,
                id=str(
                    uuid5(
                        NAMESPACE_URL,
                        f"{self.bus.runner_id}:timer:{trigger_id}:{due_instant.isoformat()}",
                    )
                ),
                source=trigger_id,
                topic="timer.due",
                payload={"trigger_id": trigger_id, "due_at": due_instant.isoformat()},
            )
        )

    async def periodic(
        self,
        trigger_id: str,
        starts_at: datetime,
        every_seconds: float,
        max_occurrences: int,
        ends_at: Optional[datetime] = None,
    ) -> None:
        import math

        if (
            not math.isfinite(every_seconds)
            or every_seconds <= 0
            or max_occurrences < 1
        ):
            raise ValueError(
                "periodic timers require a positive interval and finite occurrence bound"
            )
        origin = utc_instant(starts_at)
        end = utc_instant(ends_at) if ends_at is not None else None
        for occurrence in range(max_occurrences):
            due_at = origin + timedelta(seconds=occurrence * every_seconds)
            if end is not None and due_at > end:
                break
            await self.fire(f"{trigger_id}:{occurrence}", due_at)
