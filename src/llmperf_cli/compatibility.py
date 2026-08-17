"""Version-tolerant, whitelist-only projections for every CLI response."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from llmperf_cli.client import ClientError
from llmperf.user_config import display_environment_value


Projector = Callable[[Any, bool], Any]


@dataclass(frozen=True)
class CLIProjection:
    """The only payload type accepted by the CLI rendering boundary."""

    route: str
    payload: Any
    renderer: str


def _object(document: Any, route: str) -> Mapping[str, Any]:
    if not isinstance(document, Mapping):
        raise ClientError(f"{route} response must be an object")
    return document


def _pick(document: Mapping[str, Any], fields: Sequence[str]) -> Dict[str, Any]:
    return {field: document.get(field) for field in fields if field in document}


def _items(
    document: Any,
    route: str,
    projector: Callable[[Mapping[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    source = _object(document, route)
    raw_items = source.get("items")
    if not isinstance(raw_items, list):
        raise ClientError(f"{route} response must contain an items array")
    projected = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, Mapping):
            raise ClientError(f"{route} item {index} must be an object")
        projected.append(projector(item))
    result = {"items": projected}
    for field in ("limit", "offset"):
        if field in source:
            result[field] = source[field]
    return result


def project_health(document: Any, detailed: bool = False) -> Dict[str, Any]:
    """Normalize old/new health shapes while excluding configuration internals."""

    source = _object(document, "health")
    auth = source.get("auth") if isinstance(source.get("auth"), Mapping) else {}
    enabled = bool(auth.get("enabled"))
    reload_error = bool(auth.get("reload_error"))
    return {
        "status": source.get("status"),
        "database": source.get("database"),
        "planner": source.get("planner"),
        "providers": source.get("providers"),
        "auth": {
            "status": (
                "degraded"
                if reload_error
                else ("enabled" if enabled else "disabled")
            ),
            "enabled": enabled,
            "reload_error": reload_error,
        },
    }


def project_runner(document: Any, detailed: bool = False) -> Dict[str, Any]:
    source = _object(document, "runner")
    benchmark = (
        source.get("benchmark") if isinstance(source.get("benchmark"), Mapping) else {}
    )
    summary = source.get("summary") if isinstance(source.get("summary"), Mapping) else {}
    results = summary.get("results") if isinstance(summary.get("results"), Mapping) else {}
    outcome = summary.get("outcome") if isinstance(summary.get("outcome"), Mapping) else {}
    requests = source.get("requests") if isinstance(source.get("requests"), Mapping) else {}
    error = source.get("error") if isinstance(source.get("error"), Mapping) else {}
    if not error and isinstance(outcome.get("first_error"), Mapping):
        error = outcome["first_error"]
    message = outcome.get("message") or source.get("message") or source.get("error_message")
    if not error and source.get("error_message"):
        error = {"code": None, "message": source["error_message"]}
    projected = {
        "runner_id": source.get("runner_id"),
        "status": source.get("status"),
        "label": source.get("label"),
        "provider": source.get("provider", benchmark.get("provider")),
        "model": source.get("model", benchmark.get("model")),
        "requests": {
            "started": requests.get(
                "started", outcome.get("requests_started", results.get("num_requests_started"))
            ),
            "completed": requests.get(
                "completed",
                outcome.get("requests_completed", results.get("num_completed_requests")),
            ),
            "failed": requests.get(
                "failed", outcome.get("requests_failed", results.get("number_errors"))
            ),
            "error_rate": requests.get("error_rate", results.get("error_rate")),
        },
        "error": _pick(error, ("code", "type", "message")) if error else None,
        "message": message,
        "scheduler_id": source.get("scheduler_id"),
        "worker": _pick(
            source.get("worker") if isinstance(source.get("worker"), Mapping) else {},
            ("process_id", "exit_code"),
        ),
        "started_at": source.get("started_at"),
        "finished_at": source.get("finished_at"),
    }
    if "plan_occurrence" in source:
        projected["plan_occurrence"] = source.get("plan_occurrence")
    if detailed:
        projected.update(
            _pick(
                source,
                (
                    "campaign_id",
                    "runner_plan_id",
                    "plan_occurrence",
                    "scheduled_for",
                    "dispatch_lag_seconds",
                    "created_at",
                    "cancel_requested",
                ),
            )
        )
    return projected


def _runner_list_item(source: Mapping[str, Any]) -> Dict[str, Any]:
    projected = project_runner(source, detailed=True)
    projected["created_at"] = source.get("created_at")
    return projected


def _runner_list(document: Any, detailed: bool) -> Dict[str, Any]:
    return _items(document, "runner.list", _runner_list_item)


_CAMPAIGN_FIELDS = (
    "campaign_id",
    "name",
    "description",
    "status",
    "outcome",
    "has_failures",
    "runner_count",
    "status_counts",
    "runner_plan_count",
    "runner_plan_status_counts",
    "protocol_instance_count",
    "protocol_instance_status_counts",
    "dispatch_count",
    "dispatch_status_counts",
    "created_at",
)


def _campaign(source: Mapping[str, Any]) -> Dict[str, Any]:
    return _pick(source, _CAMPAIGN_FIELDS)


def _campaign_list(document: Any, detailed: bool) -> Dict[str, Any]:
    return _items(document, "campaign.list", _campaign)


def _campaign_status(document: Any, detailed: bool) -> Dict[str, Any]:
    source = _object(document, "campaign.status")
    campaign_source = source.get("campaign")
    if not isinstance(campaign_source, Mapping):
        raise ClientError("campaign.status response must contain a campaign object")
    aggregate = source.get("aggregate")
    merged = dict(campaign_source)
    if isinstance(aggregate, Mapping):
        merged.update(aggregate)
    runners = source.get("runners", [])
    if not isinstance(runners, list):
        raise ClientError("campaign.status response runners must be an array")
    return {
        "campaign": _campaign(merged),
        "runners": [project_runner(item, detailed=detailed) for item in runners],
    }


_PLAN_FIELDS = (
    "runner_plan_id",
    "campaign_id",
    "name",
    "status",
    "timezone",
    "overlap_policy",
    "starts_at",
    "ends_at",
    "max_occurrences",
    "next_fire_at",
    "next_fire_local",
    "last_fire_at",
    "occurrence_cursor",
    "emitted_count",
    "skipped_count",
    "misfire_grace_seconds",
    "created_at",
    "updated_at",
)


def _plan(source: Mapping[str, Any]) -> Dict[str, Any]:
    return _pick(source, _PLAN_FIELDS)


def _plan_one(document: Any, detailed: bool) -> Dict[str, Any]:
    return _plan(_object(document, "planner"))


def _plan_list(document: Any, detailed: bool) -> Dict[str, Any]:
    return _items(document, "planner.list", _plan)


def _plan_preview(document: Any, detailed: bool) -> Dict[str, Any]:
    source = _object(document, "planner.preview")
    result = _pick(source, ("start_mode", "effective_starts_at"))
    items = source.get("items")
    if not isinstance(items, list):
        raise ClientError("planner.preview response must contain an items array")
    result["items"] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ClientError(f"planner.preview item {index} must be an object")
        result["items"].append(
            _pick(item, ("occurrence", "scheduled_for", "local_time", "adjustments"))
        )
    return result


def _plan_events(document: Any, detailed: bool) -> Dict[str, Any]:
    return _items(
        document,
        "planner.events",
        lambda item: _pick(
            item,
            ("runner_plan_id", "event", "action", "status", "message", "created_at"),
        ),
    ) | {"runner_plan_id": _object(document, "planner.events").get("runner_plan_id")}


def _scheduler(document: Any, detailed: bool) -> Dict[str, Any]:
    source = _object(document, "scheduler.status")
    result = _pick(
        source,
        (
            "scheduler_id",
            "status",
            "max_concurrent_runners",
            "live_slots",
            "busy_slots",
            "worker_kind",
            "active_workers",
            "ray_mode",
            "ray_actor_num_cpus",
        ),
    )
    guard = source.get("performance_guard")
    if isinstance(guard, Mapping):
        guard_view = _pick(
            guard,
            (
                "enabled",
                "tripped",
                "reason",
                "max_host_memory_utilization",
                "resume_host_memory_utilization",
            ),
        )
        host_memory = guard.get("host_memory")
        if isinstance(host_memory, Mapping):
            guard_view["host_memory"] = _pick(
                host_memory,
                ("available", "total_bytes", "available_bytes", "utilization"),
            )
        result["performance_guard"] = guard_view
    runtime = source.get("ray_runtime")
    if isinstance(runtime, Mapping):
        runtime_view = _pick(
            runtime,
            (
                "status",
                "error",
                "alive_nodes",
                "object_store_available_ratio",
                "claim_blocked",
                "claim_block_reason",
            ),
        )
        for resource_field in ("cluster_resources", "available_resources"):
            resources = runtime.get(resource_field)
            if isinstance(resources, Mapping):
                runtime_view[resource_field] = _pick(
                    resources,
                    ("CPU", "GPU", "memory", "object_store_memory"),
                )
        result["ray_runtime"] = runtime_view
    return result


def _planner_runtime(document: Any, detailed: bool) -> Dict[str, Any]:
    return _pick(
        _object(document, "planner.runtime"),
        ("planner_id", "status", "poll_interval_seconds", "batch_size"),
    )


def _provider(source: Mapping[str, Any]) -> Dict[str, Any]:
    raw_discovery = source.get("model_discovery")
    if isinstance(raw_discovery, Mapping):
        discovery = _pick(
            raw_discovery,
            ("mode", "path", "cache_ttl_seconds", "static_model_count"),
        )
    else:
        discovery = {
            "mode": source.get("discovery"),
            "cache_ttl_seconds": source.get("model_cache_ttl_seconds"),
            "static_model_count": source.get("static_model_count"),
        }
        if source.get("models_path") is not None:
            discovery["path"] = source.get("models_path")
    typical_models = source.get("typical_models", [])
    if not isinstance(typical_models, list) or not all(
        isinstance(model, str) for model in typical_models
    ):
        raise ClientError("provider profile typical_models must be a string array")
    return {
        "id": source.get("id"),
        "adapter": source.get("adapter", source.get("llm_api")),
        "base_url": source.get("base_url", source.get("api_base")),
        "api_key_configured": source.get(
            "api_key_configured", source.get("has_api_key")
        ),
        "typical_models": list(typical_models[:3]),
        "model_discovery": discovery,
    }


def _provider_list(document: Any, detailed: bool) -> Dict[str, Any]:
    source = _object(document, "provider.list")
    result = _items(source, "provider.list", _provider)
    result.update(_pick(source, ("generation", "loaded_at")))
    return result


def _provider_reload(document: Any, detailed: bool) -> Dict[str, Any]:
    source = _object(document, "provider.reload")
    result = _provider_list(source, detailed)
    result["reloaded"] = bool(source.get("reloaded"))
    raw_changes = source.get("changes")
    if not isinstance(raw_changes, Mapping):
        raise ClientError("provider.reload response must contain a changes object")
    changes: Dict[str, Any] = {}
    for field in ("added", "updated", "removed"):
        values = raw_changes.get(field)
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            raise ClientError(
                f"provider.reload changes.{field} must be a string array"
            )
        changes[field] = list(values)
    result["changes"] = changes
    return result


def _provider_models(document: Any, detailed: bool) -> Dict[str, Any]:
    source = _object(document, "provider.models")
    result = _pick(source, ("provider", "source", "cached", "fetched_at", "expires_at"))
    models = source.get("models")
    if not isinstance(models, list) or not all(isinstance(model, str) for model in models):
        raise ClientError("provider.models response must contain a models string array")
    result["models"] = list(models)
    return result


def _auth_client(source: Mapping[str, Any]) -> Dict[str, Any]:
    return _pick(
        source,
        ("username", "display_name", "email", "role", "enabled", "created_at", "updated_at"),
    )


def _auth_list(document: Any, detailed: bool) -> Dict[str, Any]:
    return _items(document, "auth.list", _auth_client)


def _auth_one(document: Any, detailed: bool) -> Dict[str, Any]:
    return _auth_client(_object(document, "auth"))


def _auth_events(document: Any, detailed: bool) -> Dict[str, Any]:
    return _items(
        document,
        "auth.events",
        lambda item: _pick(item, ("username", "action", "actor", "created_at")),
    )


def _config(document: Any, detailed: bool) -> Dict[str, Any]:
    source = _object(document, "config")
    result = _pick(
        source,
        (
            "name",
            "value",
            "path",
            "exists",
            "items",
            "removed",
            "effective_next_run",
        ),
    )
    name = result.get("name")
    if isinstance(name, str) and "value" in result:
        result["value"] = display_environment_value(name, result["value"])
    items = result.get("items")
    if isinstance(items, Mapping):
        result["items"] = {
            str(item_name): display_environment_value(str(item_name), item_value)
            for item_name, item_value in items.items()
        }
    return result


def _campaign_one(document: Any, detailed: bool) -> Dict[str, Any]:
    return _campaign(_object(document, "campaign"))


def _campaign_start(document: Any, detailed: bool) -> Dict[str, Any]:
    source = _object(document, "campaign.start")
    result = _pick(source, ("campaign_id", "exported_to"))
    if detailed:
        result["runners"] = [project_runner(item, True) for item in source.get("runners", [])]
        result["runner_plans"] = [_plan(item) for item in source.get("runner_plans", [])]
        if isinstance(source.get("campaign_status"), Mapping):
            result["campaign_status"] = _campaign(source["campaign_status"])
    return result


def _runner_one(document: Any, detailed: bool) -> Dict[str, Any]:
    return project_runner(document, detailed=detailed)


def _runner_wait(document: Any, detailed: bool) -> Any:
    if isinstance(document, list):
        return [project_runner(item, detailed=detailed) for item in document]
    return project_runner(document, detailed=detailed)


def _runner_logs(document: Any, detailed: bool) -> Dict[str, Any]:
    source = _object(document, "runner.logs")
    result = project_runner(source, detailed=True)
    result["stdout"] = source.get("stdout")
    result["stderr"] = source.get("stderr")
    return result


def _export_result(document: Any, detailed: bool) -> Dict[str, Any]:
    return _pick(_object(document, "export"), ("exported_to",))


_ADAPTERS: Dict[str, Projector] = {
    "config.set": _config,
    "config.get": _config,
    "config.unset": _config,
    "config.list": _config,
    "config.path": _config,
    "health": project_health,
    "scheduler.status": _scheduler,
    "planner.runtime": _planner_runtime,
    "planner.preview": _plan_preview,
    "planner.create": _plan_one,
    "planner.list": _plan_list,
    "planner.status": _plan_one,
    "planner.events": _plan_events,
    "planner.pause": _plan_one,
    "planner.resume": _plan_one,
    "planner.cancel": _plan_one,
    "provider.list": _provider_list,
    "provider.reload": _provider_reload,
    "provider.models": _provider_models,
    "auth.list": _auth_list,
    "auth.add": _auth_one,
    "auth.revoke": _auth_one,
    "auth.events": _auth_events,
    "campaign.start": _campaign_start,
    "campaign.status": _campaign_status,
    "campaign.list": _campaign_list,
    "campaign.cancel": _campaign_one,
    "campaign.export": _export_result,
    "runner.start": _runner_one,
    "runner.status": _runner_one,
    "runner.list": _runner_list,
    "runner.cancel": _runner_one,
    "runner.wait": _runner_wait,
    "runner.logs": _runner_logs,
    "runner.export": _export_result,
}


def _route(arguments: Any) -> str:
    command = str(arguments.command)
    subcommand = getattr(arguments, f"{command}_command", None)
    return f"{command}.{subcommand}" if subcommand else command


def _renderer(route: str, arguments: Any) -> str:
    if route == "health" and not (
        getattr(arguments, "json", False) or getattr(arguments, "full", False)
    ):
        return "health"
    if route == "campaign.status" and not (
        getattr(arguments, "json", False)
        or getattr(arguments, "full", False)
        or getattr(arguments, "include_requests", False)
    ):
        return "campaign_status"
    if route == "campaign.list" and not getattr(arguments, "json", False):
        return "campaign_table"
    if route == "runner.list" and not (
        getattr(arguments, "json", False) or getattr(arguments, "full", False)
    ):
        return "runner_table"
    if route == "runner.status" and not (
        getattr(arguments, "json", False) or getattr(arguments, "full", False)
    ):
        return "runner_summary"
    if route == "runner.logs":
        return "runner_logs"
    if route == "provider.list" and not getattr(arguments, "json", False):
        return "provider_table"
    if route == "provider.models" and not getattr(arguments, "json", False):
        return "provider_models"
    if route == "provider.reload" and not getattr(arguments, "json", False):
        return "provider_reload"
    if route in {
        "campaign.start",
        "campaign.cancel",
        "campaign.export",
        "runner.start",
        "runner.cancel",
        "runner.export",
        "runner.wait",
    } and not getattr(arguments, "full", False):
        return "silent"
    return "json"


def adapt_cli_response(arguments: Any, document: Any) -> CLIProjection:
    """Adapt one command result; unregistered routes can never reach a renderer."""

    route = _route(arguments)
    projector = _ADAPTERS.get(route)
    if projector is None:
        raise ClientError(f"No CLI response adapter is registered for {route}")
    detailed = bool(
        getattr(arguments, "full", False)
        or getattr(arguments, "include_requests", False)
    )
    return CLIProjection(
        route=route,
        payload=projector(document, detailed),
        renderer=_renderer(route, arguments),
    )


def registered_routes() -> Tuple[str, ...]:
    """Expose the closed route set for contract tests."""

    return tuple(sorted(_ADAPTERS))
