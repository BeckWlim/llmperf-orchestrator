"""Geographic recurrence calculation and RunnerPlan materialization loop."""

import asyncio
from datetime import date, datetime, time, timedelta, timezone
import logging
import os
import socket
from typing import Any, Dict, List, Mapping, Optional, Tuple
from uuid import uuid4
from zoneinfo import ZoneInfo

from llmperf_backend.models import PlannerConfig


LOGGER = logging.getLogger(__name__)
UTC = timezone.utc
WEEKDAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("datetime must include a UTC offset")
    return value.astimezone(UTC)


def _local_candidates(day: date, clock: time, zone: ZoneInfo) -> List[datetime]:
    """Return valid UTC instants for one local wall time, ordered earliest first."""

    naive = datetime.combine(day, clock.replace(tzinfo=None))
    candidates = []
    for fold in (0, 1):
        local = naive.replace(tzinfo=zone, fold=fold)
        instant = local.astimezone(UTC)
        round_trip = instant.astimezone(zone)
        if round_trip.replace(tzinfo=None) == naive and instant not in candidates:
            candidates.append(instant)
    return sorted(candidates)


def _calendar_matches(day: date, recurrence: Mapping[str, Any]) -> bool:
    frequency = recurrence["frequency"]
    if frequency == "daily":
        return True
    weekdays = {WEEKDAYS[value] for value in recurrence.get("weekdays") or []}
    return day.weekday() in weekdays


def next_fire_at(
    timezone_name: str,
    recurrence: Mapping[str, Any],
    starts_at: datetime,
    after: Optional[datetime] = None,
) -> datetime:
    """Return the first valid occurrence at/after start and strictly after `after`."""

    fire_at, _ = next_fire_details(timezone_name, recurrence, starts_at, after)
    return fire_at


def next_fire_details(
    timezone_name: str,
    recurrence: Mapping[str, Any],
    starts_at: datetime,
    after: Optional[datetime] = None,
) -> Tuple[datetime, List[Dict[str, Any]]]:
    """Return the next occurrence and deterministic DST decisions made for it."""

    start = as_utc(starts_at)
    lower = (
        start
        if after is None
        else max(start, as_utc(after) + timedelta(microseconds=1))
    )
    if recurrence["kind"] == "interval":
        seconds = int(recurrence["every_seconds"])
        if lower <= start:
            return start, []
        elapsed = (lower - start).total_seconds()
        steps = int(elapsed // seconds)
        candidate = start + timedelta(seconds=steps * seconds)
        if candidate < lower:
            candidate += timedelta(seconds=seconds)
        return candidate, []

    zone = ZoneInfo(timezone_name)
    raw_clock = recurrence["local_time"]
    clock = time.fromisoformat(raw_clock) if isinstance(raw_clock, str) else raw_clock
    interval = int(recurrence.get("interval") or 1)
    start_local_day = start.astimezone(zone).date()
    day = lower.astimezone(zone).date()
    adjustments = []
    for _ in range(366 * 200):
        distance = (day - start_local_day).days
        period_matches = distance >= 0 and (
            distance % interval == 0
            if recurrence["frequency"] == "daily"
            else (distance // 7) % interval == 0
        )
        if period_matches and _calendar_matches(day, recurrence):
            candidates = _local_candidates(day, clock, zone)
            local_wall_time = datetime.combine(
                day, clock.replace(tzinfo=None)
            ).isoformat()
            if not candidates:
                adjustments.append(
                    {
                        "reason": "nonexistent_local_time",
                        "policy": "skip",
                        "local_time": local_wall_time,
                        "timezone": timezone_name,
                    }
                )
            else:
                candidate = candidates[0]
                if candidate >= lower:
                    if len(candidates) > 1:
                        adjustments.append(
                            {
                                "reason": "ambiguous_local_time",
                                "policy": "first",
                                "local_time": local_wall_time,
                                "timezone": timezone_name,
                                "selected": candidate.isoformat(),
                                "candidates": [
                                    value.isoformat() for value in candidates
                                ],
                            }
                        )
                    return candidate, adjustments
        day += timedelta(days=1)
    raise ValueError("unable to find a bounded calendar occurrence")


def preview_fires(
    timing: Mapping[str, Any],
    count: int,
    default_starts_at: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Preview bounded occurrences in UTC and configured local time."""

    zone = ZoneInfo(str(timing["timezone"]))
    immediate_first = timing.get("starts_at") is None
    starts_at = as_utc(
        timing.get("starts_at") or default_starts_at or datetime.now(UTC)
    )
    ends_at = as_utc(timing["ends_at"]) if timing.get("ends_at") else None
    maximum = timing.get("max_occurrences")
    limit = min(count, int(maximum)) if maximum is not None else count
    items = []
    previous = None
    for occurrence in range(limit):
        if immediate_first and previous is None:
            fire_at, adjustments = starts_at, []
        else:
            fire_at, adjustments = next_fire_details(
                str(timing["timezone"]), timing["recurrence"], starts_at, previous
            )
        if ends_at is not None and fire_at > ends_at:
            break
        items.append(
            {
                "occurrence": occurrence,
                "scheduled_for": fire_at,
                "local_time": fire_at.astimezone(zone).isoformat(),
                "adjustments": adjustments,
            }
        )
        previous = fire_at
    return items


class Planner:
    """Poll due RunnerPlans and atomically materialize ordinary queued Runners."""

    def __init__(self, repository: Any, config: PlannerConfig):
        self.repository = repository
        self.config = config
        self.planner_id = f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:8]}"
        self._stop: Optional[asyncio.Event] = None
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if not self.config.enabled:
            LOGGER.info("Planner is disabled")
            return
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="llmperf-planner")
        LOGGER.info("Planner %s started", self.planner_id)

    async def stop(self) -> None:
        if self._stop is None:
            return
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        self._task = None
        self._stop = None
        LOGGER.info("Planner %s stopped", self.planner_id)

    def status(self) -> Dict[str, Any]:
        state = (
            "disabled"
            if not self.config.enabled
            else ("running" if self._stop is not None else "stopped")
        )
        return {
            "planner_id": self.planner_id,
            "status": state,
            "poll_interval_seconds": self.config.poll_interval_seconds,
            "batch_size": self.config.batch_size,
        }

    async def _run(self) -> None:
        if self._stop is None:
            raise RuntimeError("Planner must be started before running")
        while not self._stop.is_set():
            try:
                emitted = await self.repository.materialize_due_plans(
                    self.config.batch_size, self.planner_id
                )
                if emitted:
                    LOGGER.info("Planner materialized %d Runner(s)", emitted)
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.exception("Planner failed to materialize due RunnerPlans")
            try:
                await asyncio.wait_for(
                    self._stop.wait(), self.config.poll_interval_seconds
                )
            except asyncio.TimeoutError:
                pass
