"""Command-line task orchestration and export client."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import getpass
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import yaml
from tqdm import tqdm

from llmperf.logging import LOG_COLOR_MODES, LOG_LEVELS, configure_logging
from llmperf.version import PROTOCOL_VERSION
from llmperf.user_config import (
    UserConfigError,
    display_environment_value,
    read_environment_file,
    set_environment_value,
    unset_environment_value,
)
from llmperf_cli.client import ClientError, LLMPerfClient, write_json
from llmperf_cli.projections import (
    CLIProjection,
    adapt_cli_response,
    project_health,
    project_runner,
)
from llmperf_cli.environment import load_cli_environment, resolve_cli_environment_path

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
CAMPAIGN_TERMINAL_STATUSES = {"completed", "cancelled", "empty"}
UNSUCCESSFUL_OUTCOMES = {"partial_failed", "failed", "cancelled"}
LOGGER = logging.getLogger("llmperfctl")


class HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Preserve command examples while keeping argparse's standard layout."""


TOP_LEVEL_HELP = """\
Control LLMPerf benchmark Runners through the backend service.

Concepts:
  Runner     One durable benchmark execution and its results.
  RunnerPlan A bounded geographic-time rule that produces Runners.
  Planner    The backend component that materializes due RunnerPlans.
  Campaign   A named workload containing Runners and/or RunnerPlans.
  Scheduler  The backend component that assigns queued Runners to Workers.
  Worker     A backend-owned Ray execution handle; it is not started directly.

Typical workflow:
  1. Inspect providers:  llmperfctl provider list
  2. Discover models:    llmperfctl provider models <provider-id>
  3. Submit a Runner:    llmperfctl runner start -f runner.yaml
  4. Inspect results:    llmperfctl runner status <runner-id>

Run "llmperfctl <command> --help" for command-specific examples.
"""


def _command_parser(
    subparsers: Any,
    name: str,
    *,
    help: str,
    description: str,
    epilog: Optional[str] = None,
) -> argparse.ArgumentParser:
    """Create a consistently documented command parser."""

    return subparsers.add_parser(
        name,
        help=help,
        description=description,
        epilog=epilog,
        formatter_class=HelpFormatter,
    )


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ClientError(f"Unable to read YAML plan {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ClientError(f"YAML document must be a mapping: {path}")
    if document.get("version") != PROTOCOL_VERSION:
        raise ClientError(f"YAML version must be {PROTOCOL_VERSION!r}: {path}")
    return document


def load_runner_plan(path: Path) -> Dict[str, Any]:
    """Load a bare RunnerPlan or extract one from Campaign YAML."""

    document = load_yaml(path)
    if "runner_plans" not in document:
        document.pop("version")
        return document
    plans = document["runner_plans"]
    if not isinstance(plans, list) or len(plans) != 1:
        raise ClientError(
            "planner commands require exactly one entry in campaign.runner_plans"
        )
    if not isinstance(plans[0], dict):
        raise ClientError("campaign.runner_plans[0] must be a mapping")
    return dict(plans[0])


def print_json(document: Any) -> None:
    print(json.dumps(document, ensure_ascii=False, indent=2, default=str))


def print_health(document: Dict[str, Any]) -> None:
    """Render the stable health projection without internal configuration data."""

    auth = document.get("auth") or {}
    print(f"Backend: {_table_value(document.get('status'))}")
    print(
        f"Database: {_table_value(document.get('database'))}  "
        f"Planner: {_table_value(document.get('planner'))}"
    )
    print(
        f"Providers: {_table_value(document.get('providers'), '0')}  "
        f"Auth: {_table_value(auth.get('status'))}"
    )


def submit_with_artifact_progress(action):
    """Render one indicator for backend tokenizer and dataset resolution."""

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(action)
        if future.done():
            result = future.result()
            LOGGER.info(
                "Backend validation/submission completed in %.1fs",
                time.monotonic() - started,
            )
            return result
        with tqdm(
            total=None,
            desc="Backend artifact download/cache lookup",
            file=sys.stderr,
            dynamic_ncols=True,
            leave=False,
            bar_format="{desc}: {elapsed}",
        ) as progress:
            while not future.done():
                time.sleep(1)
                progress.refresh()
        result = future.result()
        LOGGER.info(
            "Backend validation/submission completed in %.1fs",
            time.monotonic() - started,
        )
        return result


def _table_value(value: Any, default: str = "-") -> str:
    if value is None or value == "":
        return default
    return str(value).replace("\n", " ")


def _truncate(value: Any, width: int) -> str:
    text = _table_value(value)
    if len(text) <= width:
        return text
    return f"{text[: max(0, width - 1)]}…"


def _compact_timestamp(value: Any) -> str:
    text = _table_value(value)
    if text == "-":
        return text
    return text.replace("T", " ")[:16]


def print_provider_table(document: Dict[str, Any]) -> None:
    print(
        f"{'ID':<16} {'ADAPTER':<10} {'KEY':<5} {'DISCOVERY':<11} "
        f"{'TYPICAL MODELS (UP TO 3)':<46} BASE URL"
    )
    for item in document.get("items", []):
        discovery = item.get("model_discovery") or {}
        typical_models = ", ".join(item.get("typical_models") or []) or "-"
        print(
            f"{_truncate(item.get('id'), 16):<16} "
            f"{_truncate(item.get('adapter'), 10):<10} "
            f"{('yes' if item.get('api_key_configured') else 'no'):<5} "
            f"{_truncate(discovery.get('mode'), 11):<11} "
            f"{_truncate(typical_models, 46):<46} "
            f"{_table_value(item.get('base_url'))}"
        )


def print_provider_models(document: Dict[str, Any]) -> None:
    cache_state = "cached" if document.get("cached") else "fresh"
    print(
        f"Provider: {_table_value(document.get('provider'))}  "
        f"Source: {_table_value(document.get('source'))}  "
        f"State: {cache_state}"
    )
    for model in document.get("models", []):
        print(model)


def print_provider_reload(document: Dict[str, Any]) -> None:
    changes = document.get("changes") or {}
    print(
        f"Provider Profiles reloaded (generation "
        f"{_table_value(document.get('generation'))})."
    )
    for field in ("added", "updated", "removed"):
        values = changes.get(field) or []
        print(f"{field.title()}: {', '.join(values) if values else '-'}")


def _validate_runner_list(document: Any, full: bool) -> Dict[str, Any]:
    if not isinstance(document, dict) or document.get("full") is not full:
        raise ClientError(
            "Runner list response does not match this CLI version; restart or update "
            "the backend service"
        )
    items = document.get("items")
    if not isinstance(items, list):
        raise ClientError("Runner list response must contain an items array")
    if full:
        return document
    required = {
        "runner_id",
        "status",
        "provider",
        "model",
        "requests",
        "created_at",
    }
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ClientError(f"Runner list item {index} must be an object")
        missing = sorted(required.difference(item))
        if missing:
            raise ClientError(
                f"Runner list item {index} is missing required fields: "
                f"{', '.join(missing)}"
            )
        if not isinstance(item["requests"], dict):
            raise ClientError(f"Runner list item {index}.requests must be an object")
    return document


def _validate_campaign_list(document: Any) -> Dict[str, Any]:
    """Validate the lightweight Campaign collection returned by the backend."""

    if not isinstance(document, dict):
        raise ClientError(
            "Campaign list response does not match this CLI version; restart or "
            "update the backend service"
        )
    items = document.get("items")
    if not isinstance(items, list):
        raise ClientError("Campaign list response must contain an items array")
    required = {
        "campaign_id",
        "name",
        "status",
        "outcome",
        "runner_count",
        "runner_plan_count",
        "status_counts",
        "created_at",
    }
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ClientError(f"Campaign list item {index} must be an object")
        missing = sorted(required.difference(item))
        if missing:
            raise ClientError(
                f"Campaign list item {index} is missing required fields: "
                f"{', '.join(missing)}"
            )
        if not isinstance(item["status_counts"], dict):
            raise ClientError(
                f"Campaign list item {index}.status_counts must be an object"
            )
    return document


def print_campaign_table(document: Dict[str, Any]) -> None:
    """Render aggregate Campaign state without nested metadata or tags."""

    items = document.get("items") or []
    if not items:
        print("No Campaigns found.")
        return

    columns = (
        ("STATUS", 9),
        ("OUTCOME", 14),
        ("CAMPAIGN ID", 36),
        ("RUNNERS/PLANS", 13),
        ("Q/R/OK/F/C", 12),
        ("CREATED", 16),
        ("NAME", 24),
    )
    print("  ".join(title.ljust(width) for title, width in columns).rstrip())
    print("  ".join("-" * width for _, width in columns).rstrip())
    for item in items:
        counts = item.get("status_counts") or {}
        states = "/".join(
            _table_value(counts.get(status), "0")
            for status in ("queued", "running", "succeeded", "failed", "cancelled")
        )
        values = (
            item.get("status"),
            item.get("outcome"),
            item.get("campaign_id"),
            f"{item.get('runner_count')}/{item.get('runner_plan_count')}",
            states,
            _compact_timestamp(item.get("created_at")),
            item.get("name"),
        )
        print(
            "  ".join(
                _truncate(value, width).ljust(width)
                for value, (_, width) in zip(values, columns)
            ).rstrip()
        )

    offset = document.get("offset", 0)
    limit = document.get("limit", len(items))
    print(f"\nShowing {len(items)} Campaign(s) (offset={offset}, limit={limit}).")


def _runner_request_summary(runner: Dict[str, Any]) -> str:
    requests = runner.get("requests") or {}
    started = requests.get("started")
    completed = requests.get("completed")
    failed = requests.get("failed")
    if started is None and completed is None and failed is None:
        return "-"
    return (
        f"{_table_value(started)} started; "
        f"{_table_value(completed, '0')} ok; "
        f"{_table_value(failed, '0')} err"
    )


def print_campaign_status(document: Dict[str, Any]) -> None:
    """Render one Campaign and its lightweight Runner summaries."""

    campaign = document.get("campaign") or {}
    counts = campaign.get("status_counts") or {}
    plan_counts = campaign.get("runner_plan_status_counts") or {}
    print(
        f"Campaign: {_table_value(campaign.get('name'))} "
        f"({_table_value(campaign.get('campaign_id'))})"
    )
    print(
        f"Status: {_table_value(campaign.get('status'))}  "
        f"Outcome: {_table_value(campaign.get('outcome'))}  "
        f"Runners: {_table_value(campaign.get('runner_count'), '0')}  "
        f"RunnerPlans: {_table_value(campaign.get('runner_plan_count'), '0')}  "
        f"TaskInstances: {_table_value(campaign.get('task_instance_count'), '0')}  "
        f"Dispatches: {_table_value(campaign.get('dispatch_count'), '0')}"
    )
    print(
        "Runner states: "
        + "  ".join(
            f"{status}={_table_value(counts.get(status), '0')}"
            for status in ("queued", "running", "succeeded", "failed", "cancelled")
        )
    )
    if campaign.get("runner_plan_count"):
        print(
            "Plan states: "
            + "  ".join(
                f"{status}={_table_value(plan_counts.get(status), '0')}"
                for status in ("active", "paused", "completed", "cancelled")
            )
        )
    if campaign.get("task_instance_count"):
        task_counts = campaign.get("task_instance_status_counts") or {}
        print(
            "Task instance states: "
            + "  ".join(
                f"{state}={_table_value(task_counts.get(state), '0')}"
                for state in ("planned", "active", "completed", "failed", "cancelled")
            )
        )
    if campaign.get("dispatch_count"):
        dispatch_counts = campaign.get("dispatch_status_counts") or {}
        print(
            "Dispatch states: "
            + "  ".join(
                f"{state}={_table_value(dispatch_counts.get(state), '0')}"
                for state in ("blocked", "pending", "emitted", "cancelled")
            )
        )

    runners = document.get("runners") or []
    if not runners:
        print("\nNo Runners materialized for this Campaign.")
        return

    columns = (
        ("ROUND", 5),
        ("STATUS", 9),
        ("RUNNER ID", 36),
        ("PROVIDER/MODEL", 24),
        ("SUMMARY", 28),
    )
    print("\n" + "  ".join(title.ljust(width) for title, width in columns).rstrip())
    print("  ".join("-" * width for _, width in columns).rstrip())
    for runner in runners:
        target = (
            "/".join(
                part
                for part in (
                    _table_value(runner.get("provider"), ""),
                    _table_value(runner.get("model"), ""),
                )
                if part
            )
            or "-"
        )
        occurrence = runner.get("plan_occurrence")
        values = (
            occurrence + 1 if isinstance(occurrence, int) else "-",
            runner.get("status"),
            runner.get("runner_id"),
            target,
            _runner_request_summary(runner),
        )
        print(
            "  ".join(
                _truncate(value, width).ljust(width)
                for value, (_, width) in zip(values, columns)
            ).rstrip()
        )


def print_runner_table(document: Dict[str, Any]) -> None:
    """Render the lightweight Runner collection without logs or nested JSON."""

    items = document.get("items") or []
    if not items:
        print("No Runners found.")
        return

    columns = (
        ("STATUS", 9),
        ("RUNNER ID", 36),
        ("PROVIDER/MODEL", 24),
        ("OK/ERR", 7),
        ("CREATED", 16),
        ("LABEL", 16),
    )
    print("  ".join(title.ljust(width) for title, width in columns).rstrip())
    print("  ".join("-" * width for _, width in columns).rstrip())
    for item in items:
        requests = item.get("requests") or {}
        target = (
            "/".join(
                part
                for part in (
                    _table_value(item.get("provider"), ""),
                    _table_value(item.get("model"), ""),
                )
                if part
            )
            or "-"
        )
        completed = _table_value(requests.get("completed"), "0")
        failed = _table_value(requests.get("failed"), "0")
        values = (
            item.get("status"),
            item.get("runner_id"),
            target,
            f"{completed}/{failed}",
            _compact_timestamp(item.get("created_at")),
            item.get("label"),
        )
        print(
            "  ".join(
                _truncate(value, width).ljust(width)
                for value, (_, width) in zip(values, columns)
            ).rstrip()
        )

    offset = document.get("offset", 0)
    limit = document.get("limit", len(items))
    print(f"\nShowing {len(items)} Runner(s) (offset={offset}, limit={limit}).")


def print_runner_summary(document: Dict[str, Any]) -> None:
    """Render one compact Runner outcome without dumping nested JSON."""

    requests = document.get("requests") or {}
    worker = document.get("worker") or {}
    print(
        f"Runner: {_table_value(document.get('runner_id'))}  "
        f"Status: {_table_value(document.get('status'))}"
    )
    print(
        f"Target: {_table_value(document.get('provider'))}/"
        f"{_table_value(document.get('model'))}  "
        f"Requests: started={_table_value(requests.get('started'))} "
        f"completed={_table_value(requests.get('completed'))} "
        f"failed={_table_value(requests.get('failed'))}"
    )
    print(
        f"Scheduler: {_table_value(document.get('scheduler_id'))}  "
        f"Worker: pid={_table_value(worker.get('process_id'))} "
        f"exit_code={_table_value(worker.get('exit_code'))}"
    )
    error = document.get("error") or {}
    message = error.get("message") or document.get("message")
    if message:
        print(f"Message: {_table_value(message)}")


def print_runner_logs(document: Dict[str, Any]) -> None:
    """Render persisted Worker streams with unambiguous stream boundaries."""

    worker = document.get("worker") or {}
    print(
        f"Runner: {_table_value(document.get('runner_id'))}  "
        f"Status: {_table_value(document.get('status'))}  "
        f"Scheduler: {_table_value(document.get('scheduler_id'))}  "
        f"Worker: pid={_table_value(worker.get('process_id'))} "
        f"exit_code={_table_value(worker.get('exit_code'))}"
    )
    for stream in ("stdout", "stderr"):
        content = document.get(stream)
        print(f"\n[{stream}]")
        if content:
            sys.stdout.write(content)
            if not content.endswith("\n"):
                sys.stdout.write("\n")
        else:
            print("(empty)")


def render_result(result: CLIProjection) -> None:
    """Render only registered projections; raw documents are rejected."""

    if not isinstance(result, CLIProjection):
        raise ClientError("CLI renderer accepts only registered projections")
    renderers = {
        "health": print_health,
        "campaign_status": print_campaign_status,
        "campaign_table": print_campaign_table,
        "runner_table": print_runner_table,
        "runner_summary": print_runner_summary,
        "runner_logs": print_runner_logs,
        "provider_table": print_provider_table,
        "provider_models": print_provider_models,
        "provider_reload": print_provider_reload,
        "json": print_json,
        "silent": lambda document: None,
    }
    renderer = renderers.get(result.renderer)
    if renderer is None:
        raise ClientError(f"Unsupported CLI renderer: {result.renderer}")
    renderer(result.payload)


def summarize_runner(runner: Dict[str, Any]) -> Dict[str, Any]:
    return project_runner(runner)


def campaign_status_view(client: LLMPerfClient, campaign_id: str) -> Dict[str, Any]:
    """Load aggregate Campaign state and all lightweight Runner summaries."""

    campaign = client.get_campaign(campaign_id)
    runners: List[Dict[str, Any]] = []
    limit = 200
    offset = 0
    while True:
        page = _validate_runner_list(
            client.list_runners(
                status=None,
                limit=limit,
                offset=offset,
                full=False,
                campaign_id=campaign_id,
            ),
            full=False,
        )
        items = page["items"]
        runners.extend(items)
        if len(items) < limit:
            break
        offset += len(items)
    runners.sort(
        key=lambda runner: (
            str(runner.get("scheduled_for") or runner.get("created_at") or ""),
            str(runner.get("runner_id") or ""),
        )
    )
    return {"campaign": campaign, "runners": runners}


def _has_unsuccessful_runner(document: Any) -> bool:
    if isinstance(document, list):
        return any(_has_unsuccessful_runner(item) for item in document)
    if isinstance(document, dict):
        if document.get("outcome") in UNSUCCESSFUL_OUTCOMES:
            return True
        if document.get("status") in {"failed", "cancelled"}:
            return True
        for key in ("campaign_status", "aggregate", "completed", "runners"):
            nested = document.get(key)
            if isinstance(nested, (dict, list)) and _has_unsuccessful_runner(nested):
                return True
    return False


def wait_for_runners(
    client: LLMPerfClient,
    runner_ids: List[str],
    poll_interval: float,
    timeout: Optional[float],
    full_output: bool = False,
) -> List[Dict[str, Any]]:
    started = time.monotonic()
    pending = set(runner_ids)
    results: Dict[str, Dict[str, Any]] = {}
    last_state: Dict[str, Any] = {}
    while pending:
        for runner_id in list(pending):
            runner = client.get_runner(runner_id)
            results[runner_id] = runner
            current_state = _runner_signature(runner)
            if last_state.get(runner_id) != current_state:
                _log_runner_state(runner, time.monotonic() - started)
                last_state[runner_id] = current_state
            if runner["status"] in TERMINAL_STATUSES:
                pending.remove(runner_id)
                report = summarize_runner(runner)
                message = report.get("message")
                if message:
                    LOGGER.info("Runner %s: %s", runner_id, message)
        if not pending:
            break
        if timeout is not None and time.monotonic() - started >= timeout:
            observations = ", ".join(
                _runner_observation(results[runner_id])
                for runner_id in sorted(pending)
                if runner_id in results
            )
            raise ClientError(
                f"Timed out waiting for {len(pending)} Runner(s) after "
                f"{time.monotonic() - started:.1f}s: {observations}"
            )
        time.sleep(poll_interval)
    completed = [results[runner_id] for runner_id in runner_ids]
    return (
        completed if full_output else [summarize_runner(runner) for runner in completed]
    )


def _runner_signature(runner: Dict[str, Any]) -> Any:
    report = summarize_runner(runner)
    worker = report.get("worker") or {}
    requests = runner.get("requests") or report.get("requests") or {}
    return (
        report.get("status"),
        report.get("scheduler_id"),
        worker.get("process_id"),
        worker.get("exit_code"),
        requests.get("started"),
        requests.get("completed"),
        requests.get("failed"),
        runner.get("cancel_requested"),
    )


def _runner_observation(runner: Dict[str, Any]) -> str:
    worker = runner.get("worker") or {}
    return (
        f"{runner.get('runner_id')} status={runner.get('status')} "
        f"scheduler={runner.get('scheduler_id') or '-'} "
        f"pid={worker.get('process_id') or '-'}"
    )


def _log_runner_state(runner: Dict[str, Any], elapsed: float) -> None:
    report = summarize_runner(runner)
    worker = report.get("worker") or {}
    requests = runner.get("requests") or report.get("requests") or {}
    LOGGER.info(
        "Runner %s status: %s elapsed=%.1fs plan=%s occurrence=%s scheduler=%s "
        "worker_pid=%s exit_code=%s requests[started=%s completed=%s failed=%s]",
        report.get("runner_id"),
        report.get("status"),
        elapsed,
        runner.get("runner_plan_id") or "-",
        (
            runner.get("plan_occurrence")
            if runner.get("plan_occurrence") is not None
            else "-"
        ),
        report.get("scheduler_id") or "-",
        worker.get("process_id") or "-",
        worker.get("exit_code") if worker.get("exit_code") is not None else "-",
        requests.get("started") if requests.get("started") is not None else "-",
        requests.get("completed") if requests.get("completed") is not None else "-",
        requests.get("failed") if requests.get("failed") is not None else "-",
    )


def wait_for_campaign(
    client: LLMPerfClient,
    campaign_id: str,
    poll_interval: float,
    timeout: Optional[float],
    initial_plans: Optional[List[Dict[str, Any]]] = None,
    initial_runners: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    started = time.monotonic()
    last_campaign = None
    last_plans = {
        str(plan["runner_plan_id"]): _plan_signature(plan)
        for plan in initial_plans or []
    }
    last_runners = {
        str(runner["runner_id"]): _runner_signature(runner)
        for runner in initial_runners or []
    }
    while True:
        campaign = client.get_campaign(campaign_id)
        current_status = str(campaign["status"])
        campaign_signature = _campaign_signature(campaign)
        elapsed = time.monotonic() - started
        if campaign_signature != last_campaign:
            runner_counts = campaign.get("status_counts") or {}
            plan_counts = campaign.get("runner_plan_status_counts") or {}
            LOGGER.info(
                "Campaign %s status=%s outcome=%s elapsed=%.1fs runners=%s "
                "[queued=%s running=%s succeeded=%s failed=%s cancelled=%s] "
                "plans=%s [active=%s paused=%s completed=%s cancelled=%s]",
                campaign_id,
                current_status,
                campaign.get("outcome") or "-",
                elapsed,
                campaign.get("runner_count", 0),
                runner_counts.get("queued", 0),
                runner_counts.get("running", 0),
                runner_counts.get("succeeded", 0),
                runner_counts.get("failed", 0),
                runner_counts.get("cancelled", 0),
                campaign.get("runner_plan_count", 0),
                plan_counts.get("active", 0),
                plan_counts.get("paused", 0),
                plan_counts.get("completed", 0),
                plan_counts.get("cancelled", 0),
            )
            last_campaign = campaign_signature
        plans = client.list_runner_plans(
            campaign_id=campaign_id, limit=200, offset=0
        ).get("items", [])
        for runner_plan in plans:
            runner_plan_id = str(runner_plan["runner_plan_id"])
            plan_signature = _plan_signature(runner_plan)
            if last_plans.get(runner_plan_id) == plan_signature:
                continue
            _log_plan_state(runner_plan, "updated")
            last_plans[runner_plan_id] = plan_signature
        runners = client.list_runners(
            campaign_id=campaign_id,
            limit=200,
            offset=0,
            full=False,
        ).get("items", [])
        for runner in runners:
            runner_id = str(runner["runner_id"])
            runner_signature = _runner_signature(runner)
            if last_runners.get(runner_id) == runner_signature:
                continue
            _log_runner_state(runner, elapsed)
            last_runners[runner_id] = runner_signature
        if current_status in CAMPAIGN_TERMINAL_STATUSES:
            LOGGER.info(
                "Campaign %s finished: status=%s outcome=%s elapsed=%.1fs "
                "runners=%s",
                campaign_id,
                current_status,
                campaign.get("outcome") or "-",
                elapsed,
                campaign.get("runner_count", 0),
            )
            return campaign
        if timeout is not None and time.monotonic() - started >= timeout:
            raise ClientError(
                f"Timed out waiting for Campaign {campaign_id} after "
                f"{elapsed:.1f}s ({current_status}); runners="
                f"{campaign.get('status_counts', {})}, plans="
                f"{campaign.get('runner_plan_status_counts', {})}"
            )
        time.sleep(poll_interval)


def _campaign_signature(campaign: Dict[str, Any]) -> Any:
    runner_counts = campaign.get("status_counts") or {}
    plan_counts = campaign.get("runner_plan_status_counts") or {}
    return (
        campaign.get("status"),
        campaign.get("outcome"),
        campaign.get("has_failures"),
        campaign.get("runner_count"),
        tuple(
            runner_counts.get(status, 0)
            for status in ("queued", "running", "succeeded", "failed", "cancelled")
        ),
        campaign.get("runner_plan_count"),
        tuple(
            plan_counts.get(status, 0)
            for status in ("active", "paused", "completed", "cancelled")
        ),
    )


def _plan_signature(runner_plan: Dict[str, Any]) -> Any:
    return tuple(
        runner_plan.get(field)
        for field in (
            "status",
            "occurrence_cursor",
            "emitted_count",
            "skipped_count",
            "next_fire_at",
        )
    )


def _log_plan_state(runner_plan: Dict[str, Any], action: str) -> None:
    LOGGER.info(
        "RunnerPlan %s %s: status=%s occurrence=%s emitted=%s skipped=%s "
        "next_fire=%s next_local=%s",
        runner_plan.get("runner_plan_id"),
        action,
        runner_plan.get("status"),
        runner_plan.get("occurrence_cursor", 0),
        runner_plan.get("emitted_count", 0),
        runner_plan.get("skipped_count", 0),
        runner_plan.get("next_fire_at"),
        runner_plan.get("next_fire_local"),
    )


def start_campaign(client: LLMPerfClient, arguments: argparse.Namespace) -> Any:
    campaign_file = Path(arguments.file).expanduser()
    LOGGER.info("Loading Campaign YAML: %s", campaign_file)
    plan = load_yaml(campaign_file)
    campaign_spec = plan.get("campaign")
    if not isinstance(campaign_spec, dict) or not campaign_spec.get("name"):
        raise ClientError("campaign.start file must define campaign.name")
    runners = plan.get("runners", [])
    runner_plans = plan.get("runner_plans", [])
    task_definitions = plan.get("task_definitions", [])
    if not isinstance(runners, list):
        raise ClientError("campaign.start runners must be a list")
    if not isinstance(runner_plans, list):
        raise ClientError("campaign.start runner_plans must be a list")
    if not isinstance(task_definitions, list):
        raise ClientError("campaign.start task_definitions must be a list")
    if not runners and not runner_plans and not task_definitions:
        raise ClientError(
            "campaign.start requires runners, runner_plans, or task_definitions"
        )
    prepared_runners = []
    for index, runner in enumerate(runners):
        if not isinstance(runner, dict):
            raise ClientError(f"plan.runners[{index}] must be a mapping")
        prepared_runners.append(dict(runner))
    prepared_plans = []
    for index, runner_plan in enumerate(runner_plans):
        if not isinstance(runner_plan, dict):
            raise ClientError(f"plan.runner_plans[{index}] must be a mapping")
        prepared_plans.append(dict(runner_plan))
    prepared_definitions = []
    for index, definition in enumerate(task_definitions):
        if not isinstance(definition, dict):
            raise ClientError(f"plan.task_definitions[{index}] must be a mapping")
        prepared_definitions.append(dict(definition))
    batch = submit_with_artifact_progress(
        lambda: client.start_campaign(
            campaign_spec, prepared_runners, prepared_plans, prepared_definitions
        )
    )
    campaign_id = batch["campaign"]["campaign_id"]
    LOGGER.info("Campaign created: %s", campaign_id)
    created = batch["items"]
    created_plans = batch["runner_plans"]
    created_definitions = batch.get("task_definitions", [])
    LOGGER.info("Submitted %d Runner(s) to Campaign %s", len(created), campaign_id)
    for runner in created:
        _log_runner_state(runner, 0)
    LOGGER.info(
        "Registered %d RunnerPlan(s) in Campaign %s",
        len(created_plans),
        campaign_id,
    )
    for runner_plan in created_plans:
        _log_plan_state(runner_plan, "registered")
    LOGGER.info(
        "Registered %d task definition(s) in Campaign %s",
        len(created_definitions),
        campaign_id,
    )
    result: Dict[str, Any] = {
        "campaign_id": campaign_id,
        "runners": created,
        "runner_plans": created_plans,
        "task_definitions": created_definitions,
    }
    should_wait = arguments.wait or bool(plan.get("wait"))
    if should_wait:
        LOGGER.info("Waiting for the complete Campaign workload")
        result["campaign_status"] = wait_for_campaign(
            client,
            campaign_id,
            arguments.poll_interval,
            arguments.timeout,
            initial_plans=created_plans,
            initial_runners=created,
        )
        if getattr(arguments, "full", False):
            completed_document = client.export_campaign(campaign_id)
            result["completed"] = completed_document["runners"]
    output = arguments.output or plan.get("export")
    if output:
        if not should_wait:
            raise ClientError("Aggregate export requires --wait or plan.wait: true")
        document = client.export_campaign(
            campaign_id, include_requests=arguments.include_requests
        )
        write_json(Path(output), document)
        LOGGER.info("Exported Campaign %s to %s", campaign_id, output)
        result["exported_to"] = str(Path(output))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmperfctl",
        description=TOP_LEVEL_HELP,
        formatter_class=HelpFormatter,
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("LLMPERF_URL", "http://127.0.0.1:8000"),
        help="Backend URL (default: LLMPERF_URL or http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("LLMPERF_TOKEN"),
        help="Bearer token (default: LLMPERF_TOKEN, then discovered private key)",
    )
    parser.add_argument(
        "--private-key",
        default=os.environ.get("LLMPERF_PRIVATE_KEY"),
        help="PEM private key used to sign short-lived trusted-client tokens",
    )
    parser.add_argument(
        "--ssh-dir",
        default=os.environ.get("LLMPERF_SSH_DIR", "~/.ssh"),
        help="Directory scanned for RSA private keys when no key/token is explicit",
    )
    parser.add_argument(
        "--no-key-discovery",
        action="store_true",
        help="Do not discover authentication keys from --ssh-dir",
    )
    parser.add_argument(
        "--auth-issuer",
        default=os.environ.get("LLMPERF_AUTH_ISSUER", "llmperfctl"),
        help="Issuer claim for locally signed authentication tokens",
    )
    parser.add_argument(
        "--auth-audience",
        default=os.environ.get("LLMPERF_AUTH_AUDIENCE", "llmperf-api"),
        help="Audience claim for locally signed authentication tokens",
    )
    parser.add_argument(
        "--auth-subject",
        default=os.environ.get("LLMPERF_AUTH_SUBJECT", getpass.getuser()),
        help="Subject claim for locally signed authentication tokens",
    )
    parser.add_argument(
        "--token-ttl",
        type=int,
        default=60,
        help="Lifetime in seconds for locally signed tokens (default: 60)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="Backend HTTP request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default=os.environ.get("LLMPERFCTL_LOG_LEVEL", "info").lower(),
        help="Operational stderr log level (default: info)",
    )
    parser.add_argument(
        "--color",
        choices=LOG_COLOR_MODES,
        default=os.environ.get("LLMPERFCTL_LOG_COLOR", "auto").lower(),
        help="Colorize operational logs: auto, always, or never (default: auto)",
    )
    commands = parser.add_subparsers(
        dest="command", required=True, title="commands", metavar="COMMAND"
    )

    config = _command_parser(
        commands,
        "config",
        help="Manage persistent llmperfctl environment settings",
        description=(
            "Manage the local CLI environment file without contacting the Backend."
        ),
        epilog="""\
Examples:
  llmperfctl config set LLMPERF_URL http://127.0.0.1:8000
  llmperfctl config set LLMPERF_TOKEN --stdin
  llmperfctl config list
  llmperfctl config path
""",
    )
    config_commands = config.add_subparsers(
        dest="config_command",
        required=True,
        title="config commands",
        metavar="COMMAND",
    )
    config_set = _command_parser(
        config_commands,
        "set",
        help="Persist one CLI environment setting",
        description="Write one setting to the selected CLI environment file.",
    )
    config_set.add_argument("name", metavar="NAME")
    config_set.add_argument("value", metavar="VALUE", nargs="?")
    config_set.add_argument(
        "--stdin",
        action="store_true",
        help="read VALUE from standard input (recommended for tokens)",
    )
    config_get = _command_parser(
        config_commands,
        "get",
        help="Show one persisted CLI setting",
        description="Read one setting, redacting sensitive values.",
    )
    config_get.add_argument("name", metavar="NAME")
    config_unset = _command_parser(
        config_commands,
        "unset",
        help="Remove one persisted CLI setting",
        description="Remove one setting from the selected CLI environment file.",
    )
    config_unset.add_argument("name", metavar="NAME")
    _command_parser(
        config_commands,
        "list",
        help="List persisted CLI settings",
        description="List persisted settings with sensitive values redacted.",
    )
    _command_parser(
        config_commands,
        "path",
        help="Show the CLI environment file path",
        description="Print the selected CLI environment file and whether it exists.",
    )

    health = _command_parser(
        commands,
        "health",
        help="Check backend availability and component counts",
        description=(
            "Show a filtered Backend health projection. Use --json for the same "
            "stable fields or --full for the detailed projection."
        ),
        epilog=(
            "Examples:\n"
            "  llmperfctl health\n"
            "  llmperfctl health --json\n"
            "  llmperfctl health --full"
        ),
    )
    health_output = health.add_mutually_exclusive_group()
    health_output.add_argument(
        "--json",
        action="store_true",
        help="Print the filtered health projection as JSON",
    )
    health_output.add_argument(
        "--full",
        action="store_true",
        help="Print the detailed registered projection (never the raw response)",
    )

    scheduler = _command_parser(
        commands,
        "scheduler",
        help="Inspect the backend Runner scheduler",
        description=(
            "Inspect the backend-owned Scheduler. The Scheduler claims queued "
            "Runners and supervises Ray-backed Workers; it is started with "
            "the backend, not through llmperfctl."
        ),
        epilog="Example:\n  llmperfctl scheduler status",
    )
    scheduler_commands = scheduler.add_subparsers(
        dest="scheduler_command",
        required=True,
        title="scheduler commands",
        metavar="COMMAND",
    )
    _command_parser(
        scheduler_commands,
        "status",
        help="Show Scheduler state and active capacity",
        description="Show Scheduler identity, state, capacity, and Worker module.",
        epilog="Example:\n  llmperfctl scheduler status",
    )

    planner = _command_parser(
        commands,
        "planner",
        help="Manage geographic-time Runner plans",
        description=(
            "A RunnerPlan is a bounded geographic-time rule that the Planner "
            "materializes into ordinary queued Runners."
        ),
        epilog="Example:\n  llmperfctl planner list",
    )
    planner_commands = planner.add_subparsers(
        dest="planner_command",
        required=True,
        title="planner commands",
        metavar="COMMAND",
    )
    _command_parser(
        planner_commands,
        "runtime",
        help="Show Planner runtime state",
        description="Show the backend Planner state and polling configuration.",
        epilog="Example:\n  llmperfctl planner runtime",
    )
    planner_preview = _command_parser(
        planner_commands,
        "preview",
        help="Preview RunnerPlan occurrence times",
        description="Validate a RunnerPlan YAML file and preview its occurrence times.",
        epilog=(
            "Example:\n  llmperfctl planner preview "
            "-f examples/example-runner-plan.yaml"
        ),
    )
    planner_preview.add_argument("-f", "--file", required=True, help="RunnerPlan YAML")
    planner_create = _command_parser(
        planner_commands,
        "create",
        help="Create a RunnerPlan",
        description="Create a bounded RunnerPlan within an existing Campaign.",
        epilog=(
            "Example:\n  llmperfctl planner create <campaign-id> "
            "-f examples/example-runner-plan.yaml"
        ),
    )
    planner_create.add_argument("campaign_id", metavar="CAMPAIGN_ID")
    planner_create.add_argument("-f", "--file", required=True, help="RunnerPlan YAML")
    planner_list = _command_parser(
        planner_commands,
        "list",
        help="List RunnerPlans",
        description="List RunnerPlans with optional status and Campaign filters.",
        epilog="Example:\n  llmperfctl planner list --status active",
    )
    planner_list.add_argument("--status")
    planner_list.add_argument("--campaign-id")
    planner_list.add_argument("--limit", type=int, default=50)
    planner_list.add_argument("--offset", type=int, default=0)
    planner_status = _command_parser(
        planner_commands,
        "status",
        help="Show one RunnerPlan",
        description="Show one persisted RunnerPlan and its current cursor.",
        epilog="Example:\n  llmperfctl planner status <runner-plan-id>",
    )
    planner_status.add_argument("runner_plan_id", metavar="RUNNER_PLAN_ID")
    planner_events = _command_parser(
        planner_commands,
        "events",
        help="Show RunnerPlan audit events",
        description="Show materialization and state-change events.",
        epilog="Example:\n  llmperfctl planner events <runner-plan-id>",
    )
    planner_events.add_argument("runner_plan_id", metavar="RUNNER_PLAN_ID")
    for action in ("pause", "resume", "cancel"):
        action_parser = _command_parser(
            planner_commands,
            action,
            help=f"{action.title()} one RunnerPlan",
            description=f"{action.title()} one persisted RunnerPlan.",
            epilog=f"Example:\n  llmperfctl planner {action} <runner-plan-id>",
        )
        action_parser.add_argument("runner_plan_id", metavar="RUNNER_PLAN_ID")

    provider = _command_parser(
        commands,
        "provider",
        help="Inspect provider profiles and discover their models",
        description=(
            "Inspect backend-owned Provider Profiles. A default provider is the "
            "profile ID selected by LLMPERF_DEFAULT_PROVIDER; it is not a special "
            "profile named 'default'."
        ),
        epilog="""\
Examples:
  llmperfctl provider list
  llmperfctl provider models deepseek
  llmperfctl provider models deepseek --refresh
  llmperfctl provider models deepseek --json
  llmperfctl provider reload
""",
    )
    provider_commands = provider.add_subparsers(
        dest="provider_command",
        required=True,
        title="provider commands",
        metavar="COMMAND",
    )
    provider_list = _command_parser(
        provider_commands,
        "list",
        help="List configured Provider Profiles",
        description=(
            "List public Provider Profile configuration without exposing API keys."
        ),
        epilog="Example:\n  llmperfctl provider list",
    )
    provider_list.add_argument(
        "--json",
        action="store_true",
        help="Render the stable Provider projection as JSON",
    )
    provider_models = _command_parser(
        provider_commands,
        "models",
        help="List models visible to one Provider Profile",
        description=(
            "Discover model IDs through the profile's remote /models endpoint or "
            "its administrator-configured static model list."
        ),
        epilog="""\
Examples:
  llmperfctl provider models deepseek
  llmperfctl provider models deepseek --refresh

Use "llmperfctl provider list" to find valid provider IDs.
""",
    )
    provider_models.add_argument(
        "provider_id", metavar="PROVIDER_ID", help="Configured Provider Profile ID"
    )
    provider_models.add_argument(
        "--refresh",
        action="store_true",
        help="Bypass the backend model-list TTL cache (requires operator role)",
    )
    provider_models.add_argument(
        "--json",
        action="store_true",
        help="Render the stable model projection as JSON",
    )
    provider_reload = _command_parser(
        provider_commands,
        "reload",
        help="Reload Provider Profiles without restarting the Backend",
        description=(
            "Validate and atomically replace only LLMPERF_PROVIDER_* settings. "
            "Running Runners keep their existing connection snapshot; newly "
            "claimed Runners use the new generation. Requires operator role."
        ),
        epilog="Example:\n  llmperfctl provider reload",
    )
    provider_reload.add_argument(
        "--json",
        action="store_true",
        help="Render the reload result as JSON",
    )

    auth = _command_parser(
        commands,
        "auth",
        help="Manage trusted CLI clients and audit events",
        description=(
            "Manage backend trusted-client public keys. Administrative writes "
            "require a superuser identity."
        ),
        epilog="""\
Examples:
  llmperfctl auth list
  llmperfctl auth add ci-agent --public-key ./ci-agent.pub --role operator
  llmperfctl auth revoke ci-agent
  llmperfctl auth events --limit 20
""",
    )
    auth_commands = auth.add_subparsers(
        dest="auth_command", required=True, title="auth commands", metavar="COMMAND"
    )
    _command_parser(
        auth_commands,
        "list",
        help="List trusted clients",
        description="List trusted client identities and their assigned roles.",
        epilog="Example:\n  llmperfctl auth list",
    )
    auth_add = _command_parser(
        auth_commands,
        "add",
        help="Add or replace a trusted client",
        description="Register a client's RSA public key and authorization role.",
        epilog=(
            "Example:\n"
            "  llmperfctl auth add ci-agent --public-key ./ci-agent.pub "
            "--role operator"
        ),
    )
    auth_add.add_argument("username", metavar="USERNAME", help="Trusted client ID")
    auth_add.add_argument(
        "--public-key", required=True, metavar="FILE", help="RSA public-key PEM file"
    )
    auth_add.add_argument(
        "--role",
        choices=["viewer", "operator", "superuser"],
        default="operator",
        help="Authorization role (default: operator)",
    )
    auth_add.add_argument("--display-name", help="Human-readable client name")
    auth_add.add_argument("--email", help="Contact email stored with the client")
    auth_revoke = _command_parser(
        auth_commands,
        "revoke",
        help="Revoke a trusted client",
        description="Revoke a trusted client identity and its public key.",
        epilog="Example:\n  llmperfctl auth revoke ci-agent",
    )
    auth_revoke.add_argument("username", metavar="USERNAME", help="Trusted client ID")
    auth_events = _command_parser(
        auth_commands,
        "events",
        help="List trusted-client audit events",
        description="List recent trusted-client creation and revocation events.",
        epilog="Example:\n  llmperfctl auth events --limit 20",
    )
    auth_events.add_argument(
        "--limit", type=int, default=100, help="Maximum events to return (default: 100)"
    )

    campaign = _command_parser(
        commands,
        "campaign",
        help="Start and inspect groups of benchmark Runners",
        description=(
            "A Campaign is a durable workload containing immediate Runners, "
            "bounded RunnerPlans, or both. Use it to submit, inspect, cancel, "
            "or export one benchmark study."
        ),
        epilog="""\
Examples:
  llmperfctl campaign start -f campaign.yaml --wait
  llmperfctl campaign status <campaign-id>
  llmperfctl campaign list
  llmperfctl campaign export <campaign-id> -o results.json

See examples/example-campaign.yaml for the Campaign YAML shape.
""",
    )
    campaign_commands = campaign.add_subparsers(
        dest="campaign_command",
        required=True,
        title="campaign commands",
        metavar="COMMAND",
    )
    campaign_start = _command_parser(
        campaign_commands,
        "start",
        help="Create and distribute a Campaign workload",
        description=(
            "Validate Campaign YAML, then transactionally create the Campaign, "
            "queue immediate Runners, and register bounded RunnerPlans."
        ),
        epilog="""\
Examples:
  llmperfctl campaign start -f examples/example-campaign.yaml
  llmperfctl campaign start -f campaign.yaml --wait -o results.json
""",
    )
    campaign_start.add_argument(
        "-f", "--file", required=True, metavar="FILE", help="Campaign YAML file"
    )
    campaign_start.add_argument(
        "-w",
        "--wait",
        action="store_true",
        help="Wait for all bounded plans and their Runners to finish",
    )
    campaign_start.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="Status polling interval while waiting (default: 2)",
    )
    campaign_start.add_argument(
        "--timeout",
        type=float,
        metavar="SECONDS",
        help="Maximum total wait time",
    )
    campaign_start.add_argument(
        "--include-requests",
        action="store_true",
        help="Include per-request records in aggregate export",
    )
    campaign_start.add_argument(
        "--full",
        action="store_true",
        help="Print complete Runners, summaries, and captured Worker logs",
    )
    campaign_start.add_argument(
        "-o", "--output", metavar="FILE", help="Export aggregate JSON after waiting"
    )
    campaign_status = _command_parser(
        campaign_commands,
        "status",
        help="Show Campaign status and Runner summaries",
        description=(
            "Show Campaign lifecycle and execution outcome followed by every "
            "materialized Runner ID, state, target, and request summary."
        ),
        epilog=(
            "Examples:\n"
            "  llmperfctl campaign status <campaign-id>\n"
            "  llmperfctl campaign status <campaign-id> --json\n"
            "  llmperfctl campaign status <campaign-id> --full\n"
            "  llmperfctl campaign status <campaign-id> --full --include-requests"
        ),
    )
    campaign_status.add_argument(
        "campaign_id", metavar="CAMPAIGN_ID", help="Campaign identifier"
    )
    campaign_status.add_argument(
        "--json",
        action="store_true",
        help="Print lightweight Campaign and Runner status as JSON",
    )
    campaign_status.add_argument(
        "--full",
        action="store_true",
        help="Print a detailed projection; use export for raw results",
    )
    campaign_status.add_argument(
        "--include-requests",
        action="store_true",
        help="Include per-request metrics; implies --full",
    )
    campaign_list = _command_parser(
        campaign_commands,
        "list",
        help="List Campaigns",
        description=(
            "List Campaigns as a compact table by default. Use --json for the "
            "same lightweight aggregate records."
        ),
        epilog=(
            "Examples:\n"
            "  llmperfctl campaign list\n"
            "  llmperfctl campaign list --limit 20\n"
            "  llmperfctl campaign list --json"
        ),
    )
    campaign_list.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum Campaigns to return (default: 50)",
    )
    campaign_list.add_argument(
        "--offset", type=int, default=0, help="Pagination offset (default: 0)"
    )
    campaign_list.add_argument(
        "--json",
        action="store_true",
        help="Print the lightweight Campaign list as JSON",
    )
    campaign_cancel = _command_parser(
        campaign_commands,
        "cancel",
        help="Cancel Campaign plans and active Runners",
        description=(
            "Cancel active RunnerPlans, cancel queued Runners immediately, and "
            "request cancellation of running Runners in the Campaign."
        ),
        epilog="Example:\n  llmperfctl campaign cancel <campaign-id>",
    )
    campaign_cancel.add_argument(
        "campaign_id", metavar="CAMPAIGN_ID", help="Campaign identifier"
    )
    campaign_export = _command_parser(
        campaign_commands,
        "export",
        help="Export aggregate Campaign results",
        description="Write Campaign metadata and Runner results to a JSON file.",
        epilog=(
            "Example:\n"
            "  llmperfctl campaign export <campaign-id> -o campaign-results.json"
        ),
    )
    campaign_export.add_argument(
        "campaign_id", metavar="CAMPAIGN_ID", help="Campaign identifier"
    )
    campaign_export.add_argument(
        "--include-requests",
        action="store_true",
        help="Include individual request records",
    )
    campaign_export.add_argument(
        "-o", "--output", required=True, metavar="FILE", help="Destination JSON file"
    )

    runner = _command_parser(
        commands,
        "runner",
        help="Start and inspect durable benchmark executions",
        description=(
            "A Runner is one durable benchmark execution. Starting a Runner queues "
            "it in the backend; the Scheduler selects it and creates a Ray-backed "
            "Worker execution handle."
        ),
        epilog="""\
Examples:
  llmperfctl runner start -f runner.yaml --wait
  llmperfctl runner status <runner-id>
  llmperfctl runner list --status failed
  llmperfctl runner logs <runner-id>
  llmperfctl runner export <runner-id> -o result.json

Use "llmperfctl runner <command> --help" for command-specific options.
""",
    )
    runner_commands = runner.add_subparsers(
        dest="runner_command",
        required=True,
        title="runner commands",
        metavar="COMMAND",
    )
    runner_start = _command_parser(
        runner_commands,
        "start",
        help="Submit one Runner from YAML",
        description=(
            "Validate Runner YAML, resolve its provider and tokenizer, and queue a "
            "durable Runner for the Scheduler."
        ),
        epilog="""\
Examples:
  llmperfctl runner start -f examples/example-smoke.yaml
  llmperfctl runner start -f runner.yaml --label smoke --wait
  llmperfctl runner start -f runner.yaml --campaign-id <campaign-id>
""",
    )
    runner_start.add_argument("-f", "--file", required=True, help="Runner YAML")
    runner_start.add_argument(
        "--campaign-id", metavar="CAMPAIGN_ID", help="Attach Runner to a Campaign"
    )
    runner_start.add_argument("--label", help="Override the Runner label from YAML")
    runner_start.add_argument(
        "-w",
        "--wait",
        action="store_true",
        help="Wait until the Runner reaches a terminal state",
    )
    runner_start.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="Status polling interval while waiting (default: 2)",
    )
    runner_start.add_argument(
        "--timeout",
        type=float,
        metavar="SECONDS",
        help="Maximum total wait time",
    )
    runner_start.add_argument(
        "--full",
        action="store_true",
        help="Print the complete Runner and captured Worker logs",
    )
    runner_status = _command_parser(
        runner_commands,
        "status",
        help="Show one Runner",
        description="Show current state and persisted details for one Runner.",
        epilog="""\
Examples:
  llmperfctl runner status <runner-id>
  llmperfctl runner status <runner-id> --wait
  llmperfctl runner status <runner-id> --json
  llmperfctl runner status <runner-id> --full
""",
    )
    runner_status.add_argument(
        "runner_id", metavar="RUNNER_ID", help="Runner identifier"
    )
    runner_status_output = runner_status.add_mutually_exclusive_group()
    runner_status_output.add_argument(
        "--json",
        action="store_true",
        help="Print the compact Runner projection as JSON",
    )
    runner_status_output.add_argument(
        "--full",
        action="store_true",
        help="Print a detailed projection; use logs/export for raw data",
    )
    runner_status.add_argument(
        "-w",
        "--wait",
        action="store_true",
        help="Reconnect and wait until the Runner reaches a terminal state",
    )
    runner_status.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="Status polling interval while waiting (default: 2)",
    )
    runner_status.add_argument(
        "--timeout",
        type=float,
        metavar="SECONDS",
        help="Maximum local wait time; does not cancel the Runner",
    )
    runner_list = _command_parser(
        runner_commands,
        "list",
        help="List and filter Runners",
        description=(
            "List Runners as a compact table by default. Use --json for the same "
            "lightweight records or --full for detailed projections."
        ),
        epilog="""\
Examples:
  llmperfctl runner list
  llmperfctl runner list --status running
  llmperfctl runner list --status failed --json
  llmperfctl runner list --full --limit 5
""",
    )
    runner_list.add_argument(
        "--status",
        metavar="STATUS",
        help="Filter by queued, running, succeeded, failed, or cancelled",
    )
    runner_list.add_argument(
        "--limit", type=int, default=20, help="Maximum Runners to return (default: 20)"
    )
    runner_list.add_argument(
        "--offset", type=int, default=0, help="Pagination offset (default: 0)"
    )
    runner_list_output = runner_list.add_mutually_exclusive_group()
    runner_list_output.add_argument(
        "--json", action="store_true", help="Print the lightweight list as JSON"
    )
    runner_list_output.add_argument(
        "--full",
        action="store_true",
        help="Request detailed records and print only registered projections",
    )
    runner_cancel = _command_parser(
        runner_commands,
        "cancel",
        help="Cancel one Runner",
        description=(
            "Cancel a queued Runner or request termination of its running Worker."
        ),
        epilog="Example:\n  llmperfctl runner cancel <runner-id>",
    )
    runner_cancel.add_argument(
        "runner_id", metavar="RUNNER_ID", help="Runner identifier"
    )
    runner_wait = _command_parser(
        runner_commands,
        "wait",
        help="Wait for one or more Runners",
        description="Poll Runners until each reaches a terminal state.",
        epilog="""\
Examples:
  llmperfctl runner wait <runner-id>
  llmperfctl runner wait <runner-id-1> <runner-id-2> --timeout 300
""",
    )
    runner_wait.add_argument(
        "runner_id", nargs="+", metavar="RUNNER_ID", help="Runner identifier(s)"
    )
    runner_wait.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="Status polling interval (default: 2)",
    )
    runner_wait.add_argument(
        "--timeout",
        type=float,
        metavar="SECONDS",
        help="Maximum total wait time",
    )
    runner_wait.add_argument(
        "--full",
        action="store_true",
        help="Print detailed Runner projections",
    )
    runner_logs = _command_parser(
        runner_commands,
        "logs",
        help="Show captured Worker output",
        description="Show Worker identity, exit code, stdout, and stderr for a Runner.",
        epilog="Example:\n  llmperfctl runner logs <runner-id>",
    )
    runner_logs.add_argument("runner_id", metavar="RUNNER_ID", help="Runner identifier")
    runner_export = _command_parser(
        runner_commands,
        "export",
        help="Export one Runner result",
        description="Write one Runner and its request results to a JSON file.",
        epilog="Example:\n  llmperfctl runner export <runner-id> -o result.json",
    )
    runner_export.add_argument(
        "runner_id", metavar="RUNNER_ID", help="Runner identifier"
    )
    runner_export.add_argument(
        "-o", "--output", required=True, metavar="FILE", help="Destination JSON file"
    )
    return parser


def execute_cli_config(arguments: argparse.Namespace) -> Dict[str, Any]:
    """Manage the local CLI dotenv without constructing an HTTP client."""

    path = resolve_cli_environment_path()
    if arguments.config_command == "path":
        return {"path": str(path), "exists": path.is_file()}
    if arguments.config_command == "set":
        use_stdin = bool(getattr(arguments, "stdin", False))
        if use_stdin == (arguments.value is not None):
            raise UserConfigError("Provide exactly one of VALUE or --stdin")
        value = sys.stdin.read().rstrip("\r\n") if use_stdin else arguments.value
        set_environment_value(path, arguments.name, value)
        return {
            "name": arguments.name,
            "value": display_environment_value(arguments.name, value),
            "path": str(path),
            "effective_next_run": True,
        }
    if arguments.config_command == "unset":
        removed = unset_environment_value(path, arguments.name)
        return {
            "name": arguments.name,
            "removed": removed,
            "path": str(path),
            "effective_next_run": removed,
        }
    values = read_environment_file(path)
    if arguments.config_command == "get":
        value = values.get(arguments.name)
        if value is None:
            raise UserConfigError(
                f"CLI configuration setting is not defined: {arguments.name}"
            )
        return {
            "name": arguments.name,
            "value": display_environment_value(arguments.name, value),
            "path": str(path),
        }
    return {
        "path": str(path),
        "items": {
            name: display_environment_value(name, value)
            for name, value in sorted(values.items())
        },
    }


def execute(client: LLMPerfClient, arguments: argparse.Namespace) -> Any:
    if arguments.command == "health":
        return client.health()
    if arguments.command == "scheduler":
        return client.get_scheduler_status()
    if arguments.command == "planner":
        if arguments.planner_command == "runtime":
            return client.get_planner_status()
        if arguments.planner_command == "preview":
            payload = load_runner_plan(Path(arguments.file).expanduser())
            payload.pop("name", None)
            payload.pop("runner", None)
            return client.preview_runner_plan(payload)
        if arguments.planner_command == "create":
            runner_plan = client.create_runner_plan(
                arguments.campaign_id,
                load_runner_plan(Path(arguments.file).expanduser()),
            )
            _log_plan_state(runner_plan, "registered")
            return runner_plan
        if arguments.planner_command == "list":
            return client.list_runner_plans(
                arguments.status,
                arguments.campaign_id,
                arguments.limit,
                arguments.offset,
            )
        if arguments.planner_command == "status":
            return client.get_runner_plan(arguments.runner_plan_id)
        if arguments.planner_command == "events":
            return client.get_runner_plan_events(arguments.runner_plan_id)
        runner_plan = client.change_runner_plan(
            arguments.runner_plan_id, arguments.planner_command
        )
        _log_plan_state(runner_plan, arguments.planner_command)
        return runner_plan
    if arguments.command == "provider":
        if arguments.provider_command == "list":
            return client.list_providers()
        if arguments.provider_command == "reload":
            return client.reload_providers()
        return client.list_provider_models(
            arguments.provider_id, refresh=arguments.refresh
        )
    if arguments.command == "auth":
        if arguments.auth_command == "list":
            return client.list_trusted_clients()
        if arguments.auth_command == "add":
            public_key_path = Path(arguments.public_key).expanduser()
            try:
                public_key = public_key_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise ClientError(
                    f"Unable to read public key {public_key_path}: {exc}"
                ) from exc
            return client.write_trusted_client(
                arguments.username,
                public_key,
                arguments.role,
                arguments.display_name,
                arguments.email,
            )
        if arguments.auth_command == "revoke":
            return client.revoke_trusted_client(arguments.username)
        return client.list_trusted_client_events(arguments.limit)
    if arguments.command == "campaign":
        if arguments.campaign_command == "start":
            return start_campaign(client, arguments)
        if arguments.campaign_command == "status":
            if arguments.full or arguments.include_requests:
                return client.export_campaign(
                    arguments.campaign_id,
                    include_requests=arguments.include_requests,
                )
            return campaign_status_view(client, arguments.campaign_id)
        if arguments.campaign_command == "list":
            return _validate_campaign_list(
                client.list_campaigns(arguments.limit, arguments.offset)
            )
        if arguments.campaign_command == "cancel":
            campaign = client.cancel_campaign(arguments.campaign_id)
            LOGGER.info(
                "Campaign %s cancellation result: status=%s outcome=%s "
                "runners=%s plans=%s",
                arguments.campaign_id,
                campaign.get("status"),
                campaign.get("outcome"),
                campaign.get("runner_count"),
                campaign.get("runner_plan_count"),
            )
            return campaign
        document = client.export_campaign(
            arguments.campaign_id, arguments.include_requests
        )
        write_json(Path(arguments.output), document)
        LOGGER.info(
            "Exported Campaign %s to %s",
            arguments.campaign_id,
            arguments.output,
        )
        return {"exported_to": arguments.output}
    if arguments.command == "runner":
        if arguments.runner_command == "start":
            runner_file = Path(arguments.file).expanduser()
            LOGGER.info("Loading Runner YAML: %s", runner_file)
            payload = load_yaml(runner_file)
            if arguments.campaign_id:
                payload["campaign_id"] = arguments.campaign_id
            if arguments.label:
                payload["label"] = arguments.label
            LOGGER.info(
                "Validating and submitting Runner (request timeout: %g seconds)",
                arguments.request_timeout,
            )
            created = submit_with_artifact_progress(
                lambda: client.start_runner(payload)
            )
            runner_id = created["runner_id"]
            runner_status = created.get("status", "submitted")
            benchmark = created.get("benchmark") or payload.get("benchmark") or {}
            LOGGER.info(
                "Runner accepted: %s (%s) campaign=%s provider=%s model=%s",
                runner_id,
                runner_status,
                created.get("campaign_id") or "-",
                benchmark.get("provider") or "-",
                benchmark.get("model") or "-",
            )
            if not arguments.wait:
                LOGGER.info(
                    "Runner start is non-blocking; track progress with: "
                    "llmperfctl runner status %s",
                    runner_id,
                )
                return created
            LOGGER.info("Waiting for Runner completion")
            return wait_for_runners(
                client,
                [runner_id],
                arguments.poll_interval,
                arguments.timeout,
                full_output=arguments.full,
            )[0]
        if arguments.runner_command == "status":
            if arguments.wait:
                return wait_for_runners(
                    client,
                    [arguments.runner_id],
                    arguments.poll_interval,
                    arguments.timeout,
                    full_output=arguments.full,
                )[0]
            runner = client.get_runner(arguments.runner_id)
            return runner if arguments.full else summarize_runner(runner)
        if arguments.runner_command == "list":
            document = client.list_runners(
                arguments.status,
                arguments.limit,
                arguments.offset,
                full=arguments.full,
            )
            return _validate_runner_list(document, full=arguments.full)
        if arguments.runner_command == "cancel":
            runner = client.cancel_runner(arguments.runner_id)
            _log_runner_state(runner, 0)
            return runner
        if arguments.runner_command == "wait":
            return wait_for_runners(
                client,
                arguments.runner_id,
                arguments.poll_interval,
                arguments.timeout,
                full_output=arguments.full,
            )
        if arguments.runner_command == "logs":
            return client.get_runner_logs(arguments.runner_id)
        document = client.export_runner(arguments.runner_id)
        write_json(Path(arguments.output), document)
        LOGGER.info("Exported Runner %s to %s", arguments.runner_id, arguments.output)
        return {"exported_to": arguments.output}
    raise ClientError(f"Unsupported command: {arguments.command}")


def main() -> None:
    environment_error = None
    try:
        environment_path = load_cli_environment()
    except RuntimeError as exc:
        environment_path = None
        environment_error = exc
    parser = build_parser()
    arguments = parser.parse_args()
    configure_logging(arguments.log_level, color=arguments.color)
    if environment_error is not None and arguments.command != "config":
        parser.exit(1, f"error: {environment_error}\n")
    if arguments.command == "config":
        try:
            result = execute_cli_config(arguments)
            render_result(adapt_cli_response(arguments, result))
        except UserConfigError as exc:
            parser.exit(1, f"error: {exc}\n")
        return
    try:
        if arguments.token and arguments.private_key:
            raise ClientError("Use either --token or --private-key, not both")
        token_provider = None
        token_providers = None
        if arguments.private_key:
            from llmperf_cli.auth import PrivateKeyTokenProvider

            token_provider = PrivateKeyTokenProvider(
                Path(arguments.private_key),
                arguments.auth_issuer,
                arguments.auth_audience,
                arguments.auth_subject,
                arguments.token_ttl,
            )
        elif not arguments.token and not arguments.no_key_discovery:
            from llmperf_cli.auth import discover_private_key_providers

            token_providers = discover_private_key_providers(
                Path(arguments.ssh_dir),
                arguments.auth_issuer,
                arguments.auth_audience,
                arguments.auth_subject,
                arguments.token_ttl,
            )
        client = LLMPerfClient(
            arguments.url,
            arguments.token,
            arguments.request_timeout,
            token_provider=token_provider,
            token_providers=token_providers,
        )
        subcommand = getattr(
            arguments,
            f"{arguments.command}_command",
            None,
        )
        command_label = (
            f"{arguments.command} {subcommand}" if subcommand else arguments.command
        )
        LOGGER.info(
            "llmperfctl process started: pid=%d backend=%s command=%s cli_env=%s",
            os.getpid(),
            arguments.url,
            command_label,
            str(environment_path) if environment_path is not None else "none",
        )
        result = execute(client, arguments)
        render_result(adapt_cli_response(arguments, result))
        if _has_unsuccessful_runner(result):
            raise SystemExit(2)
    except ClientError as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
