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
from llmperf_cli.client import ClientError, LLMPerfClient, write_json


TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
LOGGER = logging.getLogger("llmperfctl")


class HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Preserve command examples while keeping argparse's standard layout."""


TOP_LEVEL_HELP = """\
Control LLMPerf benchmark Runners through the backend service.

Concepts:
  Runner     One durable benchmark execution and its results.
  Campaign   A named group of Runners for a benchmark study.
  Scheduler  The backend component that assigns queued Runners to Workers.
  Worker     A temporary backend-owned process; it is not started directly.

Typical workflow:
  1. Inspect providers:  llmperfctl provider list
  2. Discover models:    llmperfctl provider models <provider-id>
  3. Submit a Runner:    llmperfctl runner start -f runner.yaml
  4. Inspect results:    llmperfctl runner status <runner-id> --summary

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
    return document


def print_json(document: Any) -> None:
    print(json.dumps(document, ensure_ascii=False, indent=2, default=str))


def submit_with_artifact_progress(action):
    """Render one indicator for backend tokenizer and dataset resolution."""

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(action)
        if future.done():
            return future.result()
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
        return future.result()


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


def summarize_runner(runner: Dict[str, Any]) -> Dict[str, Any]:
    benchmark = runner.get("benchmark") or {}
    summary = runner.get("summary") or {}
    results = summary.get("results") or {}
    outcome = summary.get("outcome") or {}
    error = outcome.get("first_error")
    if error is None and runner.get("error_message"):
        error = {"code": None, "message": runner["error_message"]}
    return {
        "runner_id": runner.get("runner_id"),
        "status": runner.get("status"),
        "label": runner.get("label"),
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
        "error": error,
        "message": outcome.get("message") or runner.get("error_message"),
        "scheduler_id": runner.get("scheduler_id"),
        "worker": runner.get("worker"),
        "started_at": runner.get("started_at"),
        "finished_at": runner.get("finished_at"),
    }


def _has_unsuccessful_runner(document: Any) -> bool:
    if isinstance(document, list):
        return any(
            isinstance(item, dict) and item.get("status") in {"failed", "cancelled"}
            for item in document
        )
    if isinstance(document, dict):
        if document.get("status") in {"failed", "cancelled"}:
            return True
        completed = document.get("completed")
        if isinstance(completed, list):
            return _has_unsuccessful_runner(completed)
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
    last_status: Dict[str, str] = {}
    while pending:
        for runner_id in list(pending):
            runner = client.get_runner(runner_id)
            results[runner_id] = runner
            current_status = str(runner["status"])
            if last_status.get(runner_id) != current_status:
                LOGGER.info("Runner %s status: %s", runner_id, current_status)
                last_status[runner_id] = current_status
            if runner["status"] in TERMINAL_STATUSES:
                pending.remove(runner_id)
                report = summarize_runner(runner)
                message = report.get("message")
                if message:
                    LOGGER.info("Runner %s: %s", runner_id, message)
        if not pending:
            break
        if timeout is not None and time.monotonic() - started >= timeout:
            raise ClientError(f"Timed out waiting for {len(pending)} Runner(s)")
        time.sleep(poll_interval)
    completed = [results[runner_id] for runner_id in runner_ids]
    return (
        completed if full_output else [summarize_runner(runner) for runner in completed]
    )


def start_campaign(client: LLMPerfClient, arguments: argparse.Namespace) -> Any:
    campaign_file = Path(arguments.file).expanduser()
    LOGGER.info("Loading Campaign YAML: %s", campaign_file)
    plan = load_yaml(campaign_file)
    campaign_spec = plan.get("campaign")
    if not isinstance(campaign_spec, dict) or not campaign_spec.get("name"):
        raise ClientError("campaign.start file must define campaign.name")
    runners = plan.get("runners")
    if not isinstance(runners, list) or not runners:
        raise ClientError("campaign.start file must define a non-empty runners list")
    prepared_runners = []
    for index, runner in enumerate(runners):
        if not isinstance(runner, dict):
            raise ClientError(f"plan.runners[{index}] must be a mapping")
        prepared_runners.append(dict(runner))
    batch = submit_with_artifact_progress(
        lambda: client.create_campaign_with_runners(
            campaign_spec, prepared_runners
        )
    )
    campaign_id = batch["campaign"]["campaign_id"]
    LOGGER.info("Campaign created: %s", campaign_id)
    created = batch["items"]
    runner_ids = [runner["runner_id"] for runner in created]
    LOGGER.info("Submitted %d Runner(s) to Campaign %s", len(created), campaign_id)
    result: Dict[str, Any] = {
        "campaign_id": campaign_id,
        "runners": created,
    }
    should_wait = arguments.wait or bool(plan.get("wait"))
    if should_wait:
        LOGGER.info("Waiting for %d Campaign Runner(s)", len(runner_ids))
        result["completed"] = wait_for_runners(
            client,
            runner_ids,
            arguments.poll_interval,
            arguments.timeout,
            full_output=getattr(arguments, "full", False),
        )
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

    _command_parser(
        commands,
        "health",
        help="Check backend availability and component counts",
        description="Return backend health, database state, and component counts.",
        epilog="Example:\n  llmperfctl health",
    )

    scheduler = _command_parser(
        commands,
        "scheduler",
        help="Inspect the backend Runner scheduler",
        description=(
            "Inspect the backend-owned Scheduler. The Scheduler claims queued "
            "Runners and starts temporary Worker processes; it is started with "
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
""",
    )
    provider_commands = provider.add_subparsers(
        dest="provider_command",
        required=True,
        title="provider commands",
        metavar="COMMAND",
    )
    _command_parser(
        provider_commands,
        "list",
        help="List configured Provider Profiles",
        description=(
            "List public Provider Profile configuration without exposing API keys."
        ),
        epilog="Example:\n  llmperfctl provider list",
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
            "A Campaign is a durable, named collection of Runners. Use it to "
            "submit a benchmark matrix, inspect aggregate status, cancel the "
            "group, or export combined results."
        ),
        epilog="""\
Examples:
  llmperfctl campaign start -f campaign.yaml --wait
  llmperfctl campaign status <campaign-id>
  llmperfctl campaign list
  llmperfctl campaign export <campaign-id> -o results.json

See examples/glm-campaign.yaml for the Campaign YAML shape.
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
        help="Create a Campaign and submit its Runners",
        description=(
            "Read Campaign YAML, create the Campaign, and transactionally submit "
            "its non-empty runners list."
        ),
        epilog="""\
Examples:
  llmperfctl campaign start -f examples/glm-campaign.yaml
  llmperfctl campaign start -f campaign.yaml --wait -o results.json
""",
    )
    campaign_start.add_argument(
        "-f", "--file", required=True, metavar="FILE", help="Campaign YAML file"
    )
    campaign_start.add_argument(
        "-w", "--wait", action="store_true", help="Wait for all Runners to finish"
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
        help="Show aggregate Campaign status",
        description="Show Campaign metadata and aggregate Runner status counts.",
        epilog=(
            "Examples:\n"
            "  llmperfctl campaign status <campaign-id>\n"
            "  llmperfctl campaign status <campaign-id> --full\n"
            "  llmperfctl campaign status <campaign-id> --full --include-requests"
        ),
    )
    campaign_status.add_argument(
        "campaign_id", metavar="CAMPAIGN_ID", help="Campaign identifier"
    )
    campaign_status.add_argument(
        "--full",
        action="store_true",
        help="Return complete Campaign and Runner result documents",
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
        description="List Campaigns with aggregate Runner status.",
        epilog="Example:\n  llmperfctl campaign list --limit 20",
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
    campaign_cancel = _command_parser(
        campaign_commands,
        "cancel",
        help="Cancel queued and running Campaign Runners",
        description=(
            "Cancel queued Runners immediately and request cancellation of running "
            "Runners in the Campaign."
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
            "it in the backend; the Scheduler selects it and launches a temporary "
            "Worker process."
        ),
        epilog="""\
Examples:
  llmperfctl runner start -f runner.yaml --wait
  llmperfctl runner status <runner-id> --summary
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
  llmperfctl runner start -f examples/glm-smoke.yaml
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
  llmperfctl runner status <runner-id> --summary
""",
    )
    runner_status.add_argument(
        "runner_id", metavar="RUNNER_ID", help="Runner identifier"
    )
    runner_status.add_argument(
        "--summary",
        action="store_true",
        help="Print a compact outcome instead of the complete Runner",
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
            "lightweight records or --full for complete records and logs."
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
        help="Request complete Runners, including summaries and logs",
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
        help="Print complete Runners and captured Worker logs",
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


def execute(client: LLMPerfClient, arguments: argparse.Namespace) -> Any:
    if arguments.command == "health":
        return client.health()
    if arguments.command == "scheduler":
        return client.get_scheduler_status()
    if arguments.command == "provider":
        if arguments.provider_command == "list":
            return client.list_providers()
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
            return client.get_campaign(arguments.campaign_id)
        if arguments.campaign_command == "list":
            return client.list_campaigns(arguments.limit, arguments.offset)
        if arguments.campaign_command == "cancel":
            return client.cancel_campaign(arguments.campaign_id)
        document = client.export_campaign(
            arguments.campaign_id, arguments.include_requests
        )
        write_json(Path(arguments.output), document)
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
            LOGGER.info("Runner accepted: %s (%s)", runner_id, runner_status)
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
                    full_output=not arguments.summary,
                )[0]
            runner = client.get_runner(arguments.runner_id)
            return summarize_runner(runner) if arguments.summary else runner
        if arguments.runner_command == "list":
            document = client.list_runners(
                arguments.status,
                arguments.limit,
                arguments.offset,
                full=arguments.full,
            )
            return _validate_runner_list(document, full=arguments.full)
        if arguments.runner_command == "cancel":
            return client.cancel_runner(arguments.runner_id)
        if arguments.runner_command == "wait":
            return wait_for_runners(
                client,
                arguments.runner_id,
                arguments.poll_interval,
                arguments.timeout,
                full_output=arguments.full,
            )
        if arguments.runner_command == "logs":
            runner = client.get_runner(arguments.runner_id)
            return {
                "runner_id": runner["runner_id"],
                "status": runner["status"],
                "scheduler_id": runner.get("scheduler_id"),
                "worker": runner.get("worker"),
                "stdout": runner.get("stdout"),
                "stderr": runner.get("stderr"),
            }
        document = client.export_runner(arguments.runner_id)
        write_json(Path(arguments.output), document)
        return {"exported_to": arguments.output}
    raise ClientError(f"Unsupported command: {arguments.command}")


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    configure_logging(arguments.log_level, color=arguments.color)
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
        result = execute(client, arguments)
        if (
            arguments.command == "runner"
            and arguments.runner_command == "list"
            and not arguments.json
            and not arguments.full
        ):
            print_runner_table(result)
        else:
            print_json(result)
        if _has_unsuccessful_runner(result):
            raise SystemExit(2)
    except ClientError as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
