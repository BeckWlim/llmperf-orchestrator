"""Combine durable Worker counts with explicitly sampled execution telemetry."""

from typing import Any, Dict, Sequence

from llmperf_backend.runtime.scheduler import Scheduler
from llmperf_backend.storage.repository import RunnerRepository


async def attach_progress(
    records: Sequence[Dict[str, Any]],
    repository: RunnerRepository,
    scheduler: Scheduler,
    *,
    campaigns: bool,
) -> None:
    identifiers = [
        record["campaign_id" if campaigns else "runner_id"] for record in records
    ]
    if campaigns:
        await _attach_campaign_progress(records, repository, scheduler, identifiers)
        return
    durable = await repository.worker_progress(
        campaign_ids=identifiers if campaigns else (),
        runner_ids=() if campaigns else identifiers,
    )
    snapshot = scheduler.worker_status(list(durable))
    live = snapshot["runners"]
    sampled = snapshot["pool_ready"] and (
        snapshot["snapshot_age_seconds"]
        <= scheduler.config.ray_health_interval_seconds
        + scheduler.config.ray_health_timeout_seconds
    )
    for runner_id, progress in durable.items():
        runner_live = live.get(runner_id, {})
        if "created" in runner_live:
            progress["created"] = runner_live["created"]
            progress["phase"] = runner_live["phase"]
        for field in ("admission_waiting", "queued", "running"):
            progress[field] = runner_live.get(field, 0) if sampled else None
    count_fields = (
        "planned",
        "created",
        "submitted",
        "not_submitted",
        "pending_results",
        "succeeded",
        "failed",
        "cancelled",
        "indeterminate",
        "admission_waiting",
        "queued",
        "running",
    )
    for record in records:
        members = (
            [
                item
                for item in durable.values()
                if item["campaign_id"] == record["campaign_id"]
            ]
            if campaigns
            else [durable[record["runner_id"]]]
        )
        progress_view: Dict[str, Any] = {}
        for field in count_fields:
            values = [member[field] for member in members]
            progress_view[field] = (
                None if any(value is None for value in values) else sum(values)
            )
        progress_view["creation_phases"] = {
            phase: sum(member["phase"] == phase for member in members)
            for phase in (
                "waiting",
                "preparation_queued",
                "preparing",
                "submitting",
                "aggregating",
            )
        }
        record["workers"] = progress_view
        # Slot capacity is global, never a private quota belonging to this Campaign.
        record["request_slots"] = {
            "scope": "global",
            "busy": snapshot["slots_busy"],
            "total": snapshot["slots_total"],
            "free": snapshot["slots_free"],
            "sampled": bool(sampled),
            "snapshot_age_seconds": snapshot["snapshot_age_seconds"],
        }


async def _attach_campaign_progress(
    records: Sequence[Dict[str, Any]],
    repository: RunnerRepository,
    scheduler: Scheduler,
    identifiers: Sequence[str],
) -> None:
    durable = await repository.campaign_worker_progress(identifiers)
    snapshot = scheduler.worker_status()
    live = snapshot["runners"]
    # Only resident Runners need individual rows to merge live creation telemetry.
    resident = await repository.worker_progress(runner_ids=list(live))
    sampled = snapshot["pool_ready"] and (
        snapshot["snapshot_age_seconds"]
        <= scheduler.config.ray_health_interval_seconds
        + scheduler.config.ray_health_timeout_seconds
    )
    live_fields = ("admission_waiting", "queued", "running")
    for item in durable.values():
        item.update({field: 0 if sampled else None for field in live_fields})
    for runner_id, runner_live in live.items():
        stored = resident.get(runner_id)
        if stored is None or stored["campaign_id"] not in durable:
            continue
        item = durable[stored["campaign_id"]]
        if "created" in runner_live:
            if stored["created"] is None:
                item["unknown_created"] -= 1
            item["created"] += runner_live["created"] - (stored["created"] or 0)
            phases = item["creation_phases"]
            phases[stored["phase"]] = phases.get(stored["phase"], 0) - 1
            phases[runner_live["phase"]] = phases.get(runner_live["phase"], 0) + 1
        if sampled:
            for field in live_fields:
                item[field] += runner_live.get(field, 0)
    for record in records:
        item = durable[record["campaign_id"]]
        if item.pop("unknown_created") > 0:
            item["created"] = None
        item["creation_phases"] = {
            phase: item["creation_phases"].get(phase, 0)
            for phase in (
                "waiting",
                "preparation_queued",
                "preparing",
                "submitting",
                "aggregating",
            )
        }
        record["workers"] = item
        record["request_slots"] = {
            "scope": "global",
            "busy": snapshot["slots_busy"],
            "total": snapshot["slots_total"],
            "free": snapshot["slots_free"],
            "sampled": bool(sampled),
            "snapshot_age_seconds": snapshot["snapshot_age_seconds"],
        }
