"""Pure bounded timetable expansion shared by validation and protocol plugins."""

from datetime import datetime, timedelta
from typing import List
from zoneinfo import ZoneInfo


def expand_geographic_schedule(
    timezone_name: str,
    starts_at: datetime,
    every_seconds: int,
    duration_days: int,
) -> List[datetime]:
    """Expand elapsed intervals through a bounded number of local calendar days."""

    zone = ZoneInfo(timezone_name)
    local_start = starts_at.astimezone(zone)
    local_end_naive = local_start.replace(tzinfo=None) + timedelta(days=duration_days)
    local_end = local_end_naive.replace(tzinfo=zone)
    end = local_end.astimezone(ZoneInfo("UTC"))
    current = starts_at.astimezone(ZoneInfo("UTC")) + timedelta(seconds=every_seconds)
    observations = []
    while current <= end:
        observations.append(current)
        current += timedelta(seconds=every_seconds)
    return observations
