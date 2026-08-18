"""HTTP client command contracts and stable Worker rendering tests."""

from argparse import Namespace
from datetime import datetime, timezone
from io import BytesIO
import json
import logging
import socket

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import pytest
from urllib.error import HTTPError

from llmperf_cli.__main__ import (
    _has_unsuccessful_runner,
    _validate_campaign_list,
    _validate_runner_list,
    build_parser,
    execute,
    print_health,
    print_campaign_status,
    print_campaign_table,
    print_runner_logs,
    print_runner_summary,
    print_runner_table,
    project_health,
    render_result,
    start_campaign,
    summarize_runner,
    wait_for_campaign,
    wait_for_runners,
)
from llmperf_cli.auth import discover_private_key_providers
from llmperf_cli.client import ClientError, LLMPerfClient
from llmperf_cli.compatibility import adapt_cli_response, registered_routes


class FakeClient:
    def __init__(self):
        self.payloads = []
        self.plan_payloads = []

    def start_runner(self, payload):
        self.payloads.append(payload)
        return {
            "runner_id": f"runner-{len(self.payloads)}",
            "status": "queued",
        }

    def start_campaign_runners(self, campaign_id, runners):
        assert campaign_id == "campaign-1"
        payloads = []
        for runner in runners:
            payload = dict(runner)
            payload["campaign_id"] = campaign_id
            payloads.append(self.start_runner(payload))
        return {"items": payloads}

    def start_campaign(
        self, campaign, runners, runner_plans, task_definitions=None
    ):
        assert campaign["name"] == "glm-study"
        payloads = []
        for runner in runners:
            payload = dict(runner)
            payload["campaign_id"] = "campaign-1"
            payloads.append(self.start_runner(payload))
        plans = []
        for runner_plan in runner_plans:
            self.plan_payloads.append(runner_plan)
            plans.append(
                {
                    "runner_plan_id": f"plan-{len(self.plan_payloads)}",
                    "campaign_id": "campaign-1",
                    "status": "active",
                }
            )
        return {
            "campaign": {"campaign_id": "campaign-1"},
            "items": payloads,
            "runner_plans": plans,
            "task_definitions": task_definitions or [],
        }


def test_campaign_start(tmp_path):
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        """
campaign:
  name: glm-study
runners:
  - label: concurrency-1
    benchmark:
      model: glm-test
""",
        encoding="utf-8",
    )
    arguments = Namespace(
        file=str(plan),
        wait=False,
        poll_interval=0.01,
        timeout=None,
        output=None,
        include_requests=False,
    )
    client = FakeClient()

    result = start_campaign(client, arguments)

    assert result["campaign_id"] == "campaign-1"
    assert result["runners"][0]["runner_id"] == "runner-1"
    assert client.payloads[0]["campaign_id"] == "campaign-1"


def test_planned_campaign(tmp_path):
    plan = tmp_path / "planned.yaml"
    plan.write_text(
        """
campaign:
  name: glm-study
runner_plans:
  - name: interval-study
    timezone: Asia/Shanghai
    starts_at: 2026-08-14T00:00:00+08:00
    max_occurrences: 8
    recurrence:
      kind: interval
      every_seconds: 30
    runner:
      benchmark:
        model: glm-test
""",
        encoding="utf-8",
    )
    arguments = Namespace(
        file=str(plan),
        wait=False,
        poll_interval=0.01,
        timeout=None,
        output=None,
        include_requests=False,
    )
    client = FakeClient()

    result = start_campaign(client, arguments)

    assert result["runners"] == []
    assert result["runner_plans"][0]["runner_plan_id"] == "plan-1"
    assert client.plan_payloads[0]["recurrence"]["every_seconds"] == 30


def test_sweep_campaign(tmp_path):
    plan = tmp_path / "sweep.yaml"
    plan.write_text(
        """
campaign:
  name: glm-study
task_definitions:
  - name: retention
    matrix: {delay: [0, 60, 300]}
    trials: 2
    payloads: {replay: {seed_namespace: replay}}
    sequence:
      - {kind: invoke, id: prime, role: prime, payload: replay}
      - {kind: invoke, id: warm, role: warm, payload: replay, after_seconds: {dimension: delay}}
    runner:
      benchmark: {model: glm-test}
""",
        encoding="utf-8",
    )
    arguments = Namespace(
        file=str(plan), wait=False, poll_interval=0.01, timeout=None,
        output=None, include_requests=False,
    )
    result = start_campaign(FakeClient(), arguments)

    assert result["task_definitions"][0]["matrix"]["delay"] == [0, 60, 300]


def test_residency_campaign(tmp_path):
    plan = tmp_path / "residency.yaml"
    plan.write_text(
        """
campaign:
  name: glm-study
task_definitions:
  - name: residency
    matrix: {observations: [24]}
    payloads: {replay: {seed_namespace: replay}}
    sequence:
      - {kind: invoke, id: prime, role: prime, payload: replay}
      - kind: repeat
        id: observations
        count: {dimension: observations}
        interval_seconds: 3600
        invoke: {kind: invoke, id: warm, role: warm, payload: replay}
    runner:
      benchmark: {model: glm-test}
""",
        encoding="utf-8",
    )
    arguments = Namespace(
        file=str(plan), wait=False, poll_interval=0.01, timeout=None,
        output=None, include_requests=False,
    )
    result = start_campaign(FakeClient(), arguments)

    definition = result["task_definitions"][0]
    assert definition["sequence"][1]["kind"] == "repeat"
    assert definition["matrix"]["observations"] == [24]


def test_promotion_campaign(tmp_path):
    plan = tmp_path / "promotion.yaml"
    plan.write_text(
        """
campaign:
  name: glm-study
task_definitions:
  - name: repeat-dose
    matrix:
      warmup_count: [0, 1, 2, 4]
      quiet_seconds: [0, 120, 600, 3600, 21600]
    trials: 5
    payloads: {replay: {seed_namespace: replay}}
    sequence:
      - {kind: invoke, id: prime, role: prime, payload: replay}
      - kind: repeat
        id: warmups
        count: {dimension: warmup_count}
        invoke: {kind: invoke, id: warmup, role: warmup, payload: replay}
      - {kind: invoke, id: probe, role: probe, payload: replay, after_seconds: {dimension: quiet_seconds}}
    runner:
      benchmark: {model: glm-test}
""",
        encoding="utf-8",
    )
    arguments = Namespace(
        file=str(plan), wait=False, poll_interval=0.01, timeout=None,
        output=None, include_requests=False,
    )
    result = start_campaign(FakeClient(), arguments)

    definition = result["task_definitions"][0]
    assert definition["matrix"]["warmup_count"] == [0, 1, 2, 4]
    assert definition["matrix"]["quiet_seconds"][-1] == 21600
def test_campaign_export_status():
    class CampaignClient:
        def export_campaign(self, campaign_id, include_requests=False):
            return {
                "campaign": {"campaign_id": campaign_id},
                "aggregate": {"runner_count": 1},
                "runners": [{"requests": []}] if include_requests else [{}],
            }

    arguments = build_parser().parse_args(
        ["campaign", "status", "campaign-1", "--full", "--include-requests"]
    )

    report = execute(CampaignClient(), arguments)

    assert report["campaign"]["campaign_id"] == "campaign-1"
    assert report["aggregate"]["runner_count"] == 1
    assert report["runners"][0]["requests"] == []


def _write_rsa_private_key(path):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    path.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    path.chmod(0o600)
    return private_key


def test_key_discovery(tmp_path):
    other_key = _write_rsa_private_key(tmp_path / "team-key")
    preferred_key = _write_rsa_private_key(tmp_path / "llmperfctl")
    insecure_path = tmp_path / "id_rsa"
    _write_rsa_private_key(insecure_path)
    insecure_path.chmod(0o644)
    (tmp_path / "llmperfctl.pub").write_text("ssh-rsa ignored", encoding="utf-8")
    (tmp_path / "known_hosts").write_text("ignored", encoding="utf-8")

    providers = discover_private_key_providers(
        tmp_path,
        "llmperfctl",
        "llmperf-api",
        "test-user",
    )

    assert [
        provider.private_key.public_key().public_numbers() for provider in providers
    ] == [
        preferred_key.public_key().public_numbers(),
        other_key.public_key().public_numbers(),
    ]


class _FakeResponse:
    def __init__(self, document):
        self.content = json.dumps(document).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.content


def test_key_fallback(monkeypatch):
    authorization_headers = []

    def fake_urlopen(request, timeout):
        authorization = request.get_header("Authorization")
        authorization_headers.append(authorization)
        if authorization == "Bearer first-token":
            raise HTTPError(
                request.full_url,
                401,
                "Unauthorized",
                None,
                BytesIO(b'{"detail":"invalid key"}'),
            )
        return _FakeResponse({"items": []})

    monkeypatch.setattr("llmperf_cli.client.urlopen", fake_urlopen)
    client = LLMPerfClient(
        "http://127.0.0.1:8000",
        token_providers=[lambda: "first-token", lambda: "second-token"],
    )

    assert client.list_trusted_clients() == {"items": []}
    assert client.list_trusted_clients() == {"items": []}
    assert authorization_headers == [
        "Bearer first-token",
        "Bearer second-token",
        "Bearer second-token",
    ]


def test_forbidden_key(monkeypatch):
    attempts = []

    def fake_urlopen(request, timeout):
        attempts.append(request)
        raise HTTPError(
            request.full_url,
            403,
            "Forbidden",
            None,
            BytesIO(b'{"detail":"superuser access required"}'),
        )

    monkeypatch.setattr("llmperf_cli.client.urlopen", fake_urlopen)
    client = LLMPerfClient(
        "http://127.0.0.1:8000",
        token_providers=[lambda: "first-token", lambda: "second-token"],
    )

    with pytest.raises(ClientError) as error:
        client.list_trusted_clients()

    assert error.value.status_code == 403
    assert len(attempts) == 1


def test_auth_defaults(monkeypatch):
    monkeypatch.delenv("LLMPERF_SSH_DIR", raising=False)
    arguments = build_parser().parse_args(["auth", "list"])

    assert arguments.ssh_dir == "~/.ssh"
    assert arguments.no_key_discovery is False


def test_main_help():
    output = build_parser().format_help()

    assert "Runner     One durable benchmark execution" in output
    assert "Planner    The backend component" in output
    assert "Scheduler  The backend component" in output
    assert "llmperfctl provider models <provider-id>" in output
    assert "llmperfctl runner start -f runner.yaml" in output


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["provider", "--help"], "LLMPERF_DEFAULT_PROVIDER"),
        (["runner", "--help"], "A Runner is one durable benchmark execution"),
        (["campaign", "--help"], "See examples/example-campaign.yaml"),
        (["runner", "start", "--help"], "Validate Runner YAML"),
    ],
)
def test_command_help(arguments, expected, capsys):
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(arguments)

    assert exit_info.value.code == 0
    assert expected in capsys.readouterr().out


ARG_CASES = [
    pytest.param(
        {
            "argv": ["health"],
            "expected": {"json": False, "full": False},
        },
        id="health-defaults",
    ),
    pytest.param(
        {
            "argv": ["provider", "models", "deepseek", "--refresh"],
            "expected": {
                "command": "provider",
                "provider_command": "models",
                "provider_id": "deepseek",
                "refresh": True,
                "json": False,
            },
        },
        id="provider-models",
    ),
    pytest.param(
        {
            "argv": ["provider", "reload", "--json"],
            "expected": {
                "command": "provider",
                "provider_command": "reload",
                "json": True,
            },
        },
        id="provider-reload",
    ),
    pytest.param(
        {
            "argv": ["scheduler", "status"],
            "expected": {"scheduler_command": "status"},
        },
        id="scheduler-status",
    ),
    pytest.param(
        {
            "argv": ["planner", "list", "--status", "active"],
            "expected": {
                "planner_command": "list",
                "status": "active",
                "limit": 50,
            },
        },
        id="planner-list",
    ),
    pytest.param(
        {
            "argv": ["campaign", "status", "campaign-1"],
            "expected": {
                "campaign_command": "status",
                "campaign_id": "campaign-1",
                "json": False,
            },
        },
        id="campaign-status",
    ),
    pytest.param(
        {
            "argv": [
                "campaign",
                "status",
                "campaign-1",
                "--full",
                "--include-requests",
            ],
            "expected": {
                "campaign_command": "status",
                "campaign_id": "campaign-1",
                "full": True,
                "include_requests": True,
            },
        },
        id="campaign-status-full",
    ),
    pytest.param(
        {
            "argv": ["campaign", "list"],
            "expected": {"limit": 50, "offset": 0, "json": False},
        },
        id="campaign-list-defaults",
    ),
    pytest.param(
        {
            "argv": [
                "runner",
                "start",
                "-f",
                "smoke.yaml",
                "-w",
                "--poll-interval",
                "0.5",
                "--timeout",
                "60",
            ],
            "expected": {
                "runner_command": "start",
                "wait": True,
                "poll_interval": 0.5,
                "timeout": 60,
            },
        },
        id="runner-wait",
    ),
    pytest.param(
        {
            "argv": [
                "runner",
                "status",
                "runner-1",
                "-w",
                "--poll-interval",
                "0.25",
                "--timeout",
                "30",
                "--summary",
            ],
            "expected": {
                "runner_command": "status",
                "runner_id": "runner-1",
                "wait": True,
                "poll_interval": 0.25,
                "timeout": 30,
                "summary": True,
            },
        },
        id="runner-status-wait",
    ),
    pytest.param(
        {
            "argv": ["runner", "list"],
            "expected": {"limit": 20, "json": False, "full": False},
        },
        id="runner-list-defaults",
    ),
]


@pytest.mark.parametrize("case", ARG_CASES)
def test_args(case):
    arguments = build_parser().parse_args(case["argv"])

    for name, expected in case["expected"].items():
        assert getattr(arguments, name) == expected


def test_nonblocking_start(tmp_path, caplog):
    runner_file = tmp_path / "runner.yaml"
    runner_file.write_text(
        "benchmark:\n  model: glm-test\n",
        encoding="utf-8",
    )
    arguments = build_parser().parse_args(["runner", "start", "-f", str(runner_file)])

    with caplog.at_level(logging.INFO, logger="llmperfctl"):
        result = execute(FakeClient(), arguments)

    assert result == {"runner_id": "runner-1", "status": "queued"}
    assert arguments.wait is False
    logs = caplog.text
    assert f"Loading Runner YAML: {runner_file}" in logs
    assert "Validating and submitting Runner (request timeout: 120 seconds)" in logs
    assert "Runner accepted: runner-1 (queued)" in logs
    assert "Runner start is non-blocking" in logs
    assert "llmperfctl runner status runner-1" in logs


def test_list_table(capsys):
    print_runner_table(
        {
            "items": [
                {
                    "runner_id": "b792140c-b58c-430f-9c01-2947f49305dc",
                    "status": "failed",
                    "provider": "aliyun",
                    "model": "glm-5.2",
                    "requests": {"completed": 0, "failed": 1},
                    "created_at": "2026-08-12T02:30:04.443524Z",
                    "label": "glm-smoke-1x1",
                    "summary": {"large": "must not be rendered"},
                    "stdout": "must not be rendered",
                    "stderr": "must not be rendered",
                }
            ],
            "limit": 20,
            "offset": 0,
            "full": False,
        }
    )

    output = capsys.readouterr().out
    assert "STATUS" in output
    assert "b792140c-b58c-430f-9c01-2947f49305dc" in output
    assert "aliyun/glm-5.2" in output
    assert "0/1" in output


def test_campaign_list_table(capsys):
    document = {
        "items": [
            {
                "campaign_id": "cc895606-89d4-4562-811d-2e12a1e1a7de",
                "name": "deepseek-v4-pro-kvcache-reality",
                "status": "completed",
                "outcome": "succeeded",
                "runner_count": 2,
                "runner_plan_count": 1,
                "status_counts": {
                    "queued": 0,
                    "running": 0,
                    "succeeded": 2,
                    "failed": 0,
                    "cancelled": 0,
                },
                "created_at": "2026-08-12T09:48:00.000000Z",
                "description": "must not be rendered",
                "tags": {"large": "must not be rendered"},
            }
        ],
        "limit": 50,
        "offset": 0,
    }

    print_campaign_table(_validate_campaign_list(document))

    output = capsys.readouterr().out
    assert "CAMPAIGN ID" in output
    assert "OUTCOME" in output
    assert "completed" in output
    assert "succeeded" in output
    assert "cc895606-89d4-4562-811d-2e12a1e1a7de" in output
    assert "0/0/2/0/0" in output
    assert "2/1" in output
    assert "deepseek-v4-pro-kvcac" in output
    assert "must not be rendered" not in output


def test_campaign_status_view(capsys):
    class CampaignClient:
        def get_campaign(self, campaign_id):
            return {
                "campaign_id": campaign_id,
                "name": "cache-study",
                "status": "completed",
                "outcome": "partial_failed",
                "runner_count": 2,
                "runner_plan_count": 1,
                "status_counts": {
                    "queued": 0,
                    "running": 0,
                    "succeeded": 1,
                    "failed": 1,
                    "cancelled": 0,
                },
                "runner_plan_status_counts": {
                    "active": 0,
                    "paused": 0,
                    "completed": 1,
                    "cancelled": 0,
                },
            }

        def list_runners(self, status, limit, offset, full=False, campaign_id=None):
            assert status is None
            assert limit == 200
            assert offset == 0
            assert full is False
            assert campaign_id == "campaign-1"
            return {
                "items": [
                    {
                        "runner_id": "runner-2",
                        "status": "failed",
                        "provider": "aliyun",
                        "model": "deepseek-v4-pro",
                        "requests": {
                            "started": None,
                            "completed": None,
                            "failed": None,
                        },
                        "plan_occurrence": 1,
                        "scheduled_for": "2026-08-13T08:55:02Z",
                        "created_at": "2026-08-13T08:55:02Z",
                    },
                    {
                        "runner_id": "runner-1",
                        "status": "succeeded",
                        "provider": "aliyun",
                        "model": "deepseek-v4-pro",
                        "requests": {
                            "started": 10,
                            "completed": 9,
                            "failed": 1,
                        },
                        "plan_occurrence": 0,
                        "scheduled_for": "2026-08-13T08:54:32Z",
                        "created_at": "2026-08-13T08:54:32Z",
                    },
                ],
                "limit": limit,
                "offset": offset,
                "full": full,
                "campaign_id": campaign_id,
            }

    arguments = build_parser().parse_args(["campaign", "status", "campaign-1"])
    document = execute(CampaignClient(), arguments)
    print_campaign_status(document)

    assert [runner["runner_id"] for runner in document["runners"]] == [
        "runner-1",
        "runner-2",
    ]
    output = capsys.readouterr().out
    assert "Campaign: cache-study (campaign-1)" in output
    assert "Status: completed  Outcome: partial_failed" in output
    assert "succeeded=1" in output
    assert "ROUND" in output
    assert "runner-1" in output
    assert "10 started; 9 ok; 1 err" in output
    assert "runner-2" in output
    assert '"campaign"' not in output


def test_campaign_schema_error():
    with pytest.raises(ClientError, match="missing required fields"):
        _validate_campaign_list({"items": [{"campaign_id": "campaign-1"}]})


def test_bad_list_schema():
    with pytest.raises(ClientError, match="does not match this CLI version"):
        _validate_runner_list(
            {
                "items": [
                    {
                        "runner_id": "runner-1",
                        "status": "succeeded",
                        "benchmark": {"provider": "aliyun", "model": "glm-5.2"},
                        "summary": {"results": {}},
                    }
                ]
            },
            full=False,
        )


def test_full_list_request(monkeypatch):
    requested_urls = []

    def fake_urlopen(request, timeout):
        requested_urls.append(request.full_url)
        return _FakeResponse({"items": []})

    monkeypatch.setattr("llmperf_cli.client.urlopen", fake_urlopen)
    client = LLMPerfClient("http://127.0.0.1:8000")

    client.list_runners(status="failed", limit=5, offset=10, full=True)

    assert requested_urls == [
        "http://127.0.0.1:8000/api/v1/runners?status=failed&limit=5&offset=10&full=true"
    ]


def test_log_request(monkeypatch):
    requested_urls = []

    def fake_urlopen(request, timeout):
        requested_urls.append(request.full_url)
        return _FakeResponse(
            {
                "runner_id": "runner-1",
                "status": "failed",
                "scheduler_id": "scheduler-1",
                "worker": {"process_id": 42, "exit_code": 1},
                "stdout": "progress\n",
                "stderr": "failure\n",
            }
        )

    monkeypatch.setattr("llmperf_cli.client.urlopen", fake_urlopen)
    document = LLMPerfClient("http://127.0.0.1:8000").get_runner_logs("runner-1")

    assert requested_urls == [
        "http://127.0.0.1:8000/api/v1/runners/runner-1/logs"
    ]
    assert document["stderr"] == "failure\n"


def test_action_output(capsys):
    arguments = build_parser().parse_args(
        ["campaign", "start", "-f", "examples/example-campaign.yaml", "--wait"]
    )

    projection = adapt_cli_response(
        arguments, {"campaign_id": "campaign-1", "large": [1, 2, 3]}
    )
    render_result(projection)

    assert capsys.readouterr().out == ""

    with pytest.raises(ClientError, match="compatibility projections"):
        render_result({"campaign_id": "campaign-1"})


def test_health_projection(capsys):
    raw = {
        "status": "ok",
        "database": "connected",
        "planner": "running",
        "providers": 3,
        "auth": {
            "enabled": True,
            "generation": 9,
            "active_key_id": "must-not-project",
            "previous_key_active": True,
            "reload_error": False,
        },
        "config_source": "/internal/backend/config.yaml",
        "config_generation": 7,
    }

    projected = project_health(raw)
    print_health(projected)
    output = capsys.readouterr().out

    assert projected == {
        "status": "ok",
        "database": "connected",
        "planner": "running",
        "providers": 3,
        "auth": {
            "status": "enabled",
            "enabled": True,
            "reload_error": False,
        },
    }
    assert "Backend: ok" in output
    assert "Auth: enabled" in output
    assert "config_source" not in output
    assert "must-not-project" not in str(projected)

    class HealthClient:
        def health(self):
            return raw

    arguments = build_parser().parse_args(["health", "--json"])
    result = execute(HealthClient(), arguments)
    render_result(adapt_cli_response(arguments, result))
    assert json.loads(capsys.readouterr().out) == projected

    arguments = build_parser().parse_args(["health", "--full"])
    result = execute(HealthClient(), arguments)
    render_result(adapt_cli_response(arguments, result))
    assert json.loads(capsys.readouterr().out) == projected


def test_status_projection(capsys):
    raw = {
        "runner_id": "runner-1",
        "status": "running",
        "benchmark": {"provider": "aliyun", "model": "deepseek-v4-pro"},
        "summary": {"large": "must not reach the default projection"},
        "stdout": "must not be rendered",
    }

    class RunnerClient:
        def get_runner(self, runner_id):
            assert runner_id == "runner-1"
            return raw

    arguments = build_parser().parse_args(["runner", "status", "runner-1"])
    result = execute(RunnerClient(), arguments)
    render_result(adapt_cli_response(arguments, result))
    default_output = capsys.readouterr().out

    assert result == summarize_runner(raw)
    assert "Runner: runner-1  Status: running" in default_output
    assert '"summary"' not in default_output

    arguments = build_parser().parse_args(
        ["runner", "status", "runner-1", "--json"]
    )
    result = execute(RunnerClient(), arguments)
    render_result(adapt_cli_response(arguments, result))
    json_output = json.loads(capsys.readouterr().out)

    assert json_output == summarize_runner(raw)
    assert "summary" not in json_output


def test_adapter_route_registry():
    assert set(registered_routes()) == {
        "auth.add",
        "auth.events",
        "auth.list",
        "auth.revoke",
        "campaign.cancel",
        "campaign.export",
        "campaign.list",
        "campaign.start",
        "campaign.status",
        "config.get",
        "config.list",
        "config.path",
        "config.set",
        "config.unset",
        "health",
        "planner.cancel",
        "planner.create",
        "planner.events",
        "planner.list",
        "planner.pause",
        "planner.preview",
        "planner.resume",
        "planner.runtime",
        "planner.status",
        "provider.list",
        "provider.models",
        "provider.reload",
        "runner.cancel",
        "runner.export",
        "runner.list",
        "runner.logs",
        "runner.start",
        "runner.status",
        "runner.wait",
        "scheduler.status",
    }

    unknown = Namespace(command="unknown")
    with pytest.raises(ClientError, match="No CLI response adapter"):
        adapt_cli_response(unknown, {"secret": "must-not-render"})


def test_adapter_redaction():
    runner_arguments = build_parser().parse_args(
        ["runner", "status", "runner-1", "--full"]
    )
    runner_projection = adapt_cli_response(
        runner_arguments,
        {
            "runner_id": "runner-1",
            "status": "failed",
            "benchmark": {"provider": "aliyun", "model": "model-1"},
            "summary": {"private": "raw summary"},
            "metadata": {"token": "secret"},
            "stdout": "raw stdout",
            "stderr": "raw stderr",
        },
    )
    assert runner_projection.renderer == "json"
    assert "summary" not in runner_projection.payload
    assert "metadata" not in runner_projection.payload
    assert "stdout" not in runner_projection.payload

    scheduler_arguments = build_parser().parse_args(["scheduler", "status"])
    scheduler_projection = adapt_cli_response(
        scheduler_arguments,
        {
            "scheduler_id": "scheduler-1",
            "status": "running",
            "ray_address": "ray://internal.example:10001",
            "ray_runtime": {
                "status": "healthy",
                "claim_blocked": True,
                "claim_block_reason": "ray_object_store_low",
                "cluster_resources": {"CPU": 8, "node:private": 1},
            },
        },
    )
    assert scheduler_projection.payload["ray_runtime"] == {
        "status": "healthy",
        "claim_blocked": True,
        "claim_block_reason": "ray_object_store_low",
        "cluster_resources": {"CPU": 8},
    }
    assert "ray_address" not in scheduler_projection.payload

    auth_arguments = build_parser().parse_args(["auth", "list"])
    auth_projection = adapt_cli_response(
        auth_arguments,
        {
            "items": [
                {
                    "username": "operator",
                    "role": "operator",
                    "keys": [{"key_id": "private-key-id"}],
                    "public_key_pem": "must-not-render",
                }
            ]
        },
    )
    assert auth_projection.payload == {
        "items": [{"username": "operator", "role": "operator"}]
    }


def test_log_output(capsys):
    document = {
        "runner_id": "runner-1",
        "status": "failed",
        "scheduler_id": "scheduler-1",
        "worker": {"process_id": 42, "exit_code": 1},
        "stdout": "progress",
        "stderr": "failure\n",
    }

    print_runner_logs(document)

    output = capsys.readouterr().out
    assert "Runner: runner-1  Status: failed" in output
    assert "[stdout]\nprogress\n" in output
    assert "[stderr]\nfailure\n" in output


def test_datetime_payload(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data))
        return _FakeResponse(
            {"campaign": {"campaign_id": "campaign-1"}, "items": [], "runner_plans": []}
        )

    monkeypatch.setattr("llmperf_cli.client.urlopen", fake_urlopen)
    client = LLMPerfClient("http://127.0.0.1:8000")

    client.start_campaign(
        {"name": "planned"},
        [],
        [{"starts_at": datetime(2026, 8, 14, tzinfo=timezone.utc)}],
    )

    assert captured["runner_plans"][0]["starts_at"] == "2026-08-14T00:00:00+00:00"


def test_provider_encoding(monkeypatch):
    requested_urls = []
    requested_methods = []

    def fake_urlopen(request, timeout):
        requested_urls.append(request.full_url)
        requested_methods.append(request.get_method())
        return _FakeResponse({"models": []})

    monkeypatch.setattr("llmperf_cli.client.urlopen", fake_urlopen)
    client = LLMPerfClient("http://127.0.0.1:8000")

    client.list_provider_models("team/provider", refresh=True)

    assert requested_urls == [
        "http://127.0.0.1:8000/api/v1/providers/team%2Fprovider/models?refresh=true"
    ]
    assert requested_methods == ["GET"]

    client.reload_providers()
    assert requested_urls[-1] == "http://127.0.0.1:8000/api/v1/providers/reload"
    assert requested_methods[-1] == "POST"


def test_provider_output(capsys):
    profiles = {
        "generation": 2,
        "loaded_at": "2026-08-14T09:46:10+00:00",
        "items": [
            {
                "id": "zhipu",
                "adapter": "openai",
                "base_url": "https://open.bigmodel.cn/api/paas/v4",
                "api_key_configured": True,
                "typical_models": ["glm-5.2", "glm-5.3", "glm-5.4"],
                "model_discovery": {
                    "mode": "static",
                    "cache_ttl_seconds": 300,
                    "static_model_count": 1,
                },
            }
        ],
    }
    arguments = build_parser().parse_args(["provider", "list"])
    render_result(adapt_cli_response(arguments, profiles))
    output = capsys.readouterr().out
    assert "ID" in output
    assert "zhipu" in output
    assert "TYPICAL MODELS" in output
    assert "glm-5.2, glm-5.3, glm-5.4" in output
    assert not output.lstrip().startswith("{")

    models = {
        "provider": "zhipu",
        "source": "static",
        "cached": False,
        "fetched_at": "2026-08-14T09:46:10+00:00",
        "expires_at": "2026-08-14T09:51:10+00:00",
        "models": ["glm5.2"],
    }
    arguments = build_parser().parse_args(["provider", "models", "zhipu"])
    render_result(adapt_cli_response(arguments, models))
    output = capsys.readouterr().out
    assert "Provider: zhipu" in output
    assert "glm5.2" in output
    assert not output.lstrip().startswith("{")

    arguments = build_parser().parse_args(
        ["provider", "models", "zhipu", "--json"]
    )
    render_result(adapt_cli_response(arguments, models))
    assert json.loads(capsys.readouterr().out)["models"] == ["glm5.2"]


def test_request_timeout(monkeypatch):
    def fake_urlopen(request, timeout):
        raise socket.timeout("timed out")

    monkeypatch.setattr("llmperf_cli.client.urlopen", fake_urlopen)
    client = LLMPerfClient("http://127.0.0.1:8000", timeout=30)

    with pytest.raises(ClientError) as error:
        client.start_runner({"benchmark": {"model": "test"}})

    message = str(error.value)
    assert "timed out after 30 seconds" in message
    assert "backend may still be processing" in message
    assert "--request-timeout SECONDS" in message


def test_http_detail(monkeypatch):
    def fake_urlopen(request, timeout):
        raise HTTPError(
            request.full_url,
            422,
            "Unprocessable Entity",
            {"X-Request-ID": "request-123"},
            BytesIO(
                json.dumps(
                    {
                        "detail": [
                            {
                                "loc": ["body", "benchmark", "tokenizer", "id"],
                                "msg": "repository does not exist",
                                "type": "value_error",
                                "input": "sensitive request body",
                            },
                            {
                                "loc": ["body", "benchmark", "model"],
                                "msg": "unsupported model",
                                "type": "value_error",
                            },
                        ]
                    }
                ).encode("utf-8")
            ),
        )

    monkeypatch.setattr("llmperf_cli.client.urlopen", fake_urlopen)
    client = LLMPerfClient("http://127.0.0.1:8000")

    with pytest.raises(ClientError) as error:
        client.start_runner({"benchmark": {"model": "test"}})

    message = str(error.value)
    assert "HTTP 422 Unprocessable Entity" in message
    assert "POST /api/v1/runners" in message
    assert "request_id=request-123" in message
    assert "body.benchmark.tokenizer.id" in message
    assert "repository does not exist [value_error]" in message
    assert "body.benchmark.model" in message
    assert "sensitive request body" not in message
    assert error.value.method == "POST"
    assert error.value.path == "/api/v1/runners"


def test_wait_summary(caplog):
    class FailedRunnerClient:
        def get_runner(self, runner_id):
            return {
                "runner_id": runner_id,
                "status": "failed",
                "label": "smoke",
                "benchmark": {"provider": "aliyun", "model": "deepseek-v4-pro"},
                "scheduler_id": "scheduler-1",
                "worker": {"process_id": 62791, "exit_code": 1},
                "summary": {
                    "results": {
                        "num_requests_started": 1,
                        "num_completed_requests": 0,
                        "number_errors": 1,
                        "error_rate": 1.0,
                    },
                    "outcome": {
                        "status": "failed",
                        "requests_started": 1,
                        "requests_completed": 0,
                        "requests_failed": 1,
                        "first_error": {
                            "code": -1,
                            "message": "invalid stream",
                        },
                        "message": "No benchmark requests completed",
                    },
                },
                "stdout": "large output",
                "stderr": "large error output",
            }

    with caplog.at_level(logging.INFO, logger="llmperfctl"):
        reports = wait_for_runners(FailedRunnerClient(), ["runner-1"], 0.01, 1)

    assert reports == [
        {
            "runner_id": "runner-1",
            "status": "failed",
            "label": "smoke",
            "provider": "aliyun",
            "model": "deepseek-v4-pro",
            "requests": {
                "started": 1,
                "completed": 0,
                "failed": 1,
                "error_rate": 1.0,
            },
            "error": {"code": -1, "message": "invalid stream"},
            "message": "No benchmark requests completed",
            "scheduler_id": "scheduler-1",
            "worker": {"process_id": 62791, "exit_code": 1},
            "started_at": None,
            "finished_at": None,
        }
    ]
    assert "large output" not in str(reports)
    assert "Runner runner-1 status: failed" in caplog.text
    assert "scheduler=scheduler-1 worker_pid=62791 exit_code=1" in caplog.text
    assert "requests[started=1 completed=0 failed=1]" in caplog.text
    assert "Runner runner-1: No benchmark requests completed" in caplog.text
    assert _has_unsuccessful_runner(reports) is True
    assert (
        _has_unsuccessful_runner({"status": "completed", "outcome": "partial_failed"})
        is True
    )
    assert (
        _has_unsuccessful_runner({"status": "completed", "outcome": "succeeded"})
        is False
    )


def test_campaign_wait_logs(caplog):
    class CampaignClient:
        def __init__(self):
            self.index = -1
            self.campaigns = [
                self._campaign("planned", 0, 0, 0, "active"),
                self._campaign("running", 1, 0, 1, "active"),
                self._campaign("completed", 1, 1, 0, "completed"),
            ]

        @staticmethod
        def _campaign(status, runner_count, succeeded, running, plan_status):
            return {
                "status": status,
                "outcome": "succeeded" if status == "completed" else "pending",
                "has_failures": False,
                "runner_count": runner_count,
                "status_counts": {
                    "queued": 0,
                    "running": running,
                    "succeeded": succeeded,
                    "failed": 0,
                    "cancelled": 0,
                },
                "runner_plan_count": 1,
                "runner_plan_status_counts": {
                    "active": int(plan_status == "active"),
                    "paused": 0,
                    "completed": int(plan_status == "completed"),
                    "cancelled": 0,
                },
            }

        def get_campaign(self, campaign_id):
            self.index += 1
            return self.campaigns[self.index]

        def list_runner_plans(self, **kwargs):
            completed = self.index == 2
            return {
                "items": [
                    {
                        "runner_plan_id": "plan-1",
                        "status": "completed" if completed else "active",
                        "occurrence_cursor": self.index,
                        "emitted_count": self.index,
                        "skipped_count": 0,
                        "next_fire_at": None if completed else f"fire-{self.index}",
                        "next_fire_local": None if completed else f"local-{self.index}",
                    }
                ]
            }

        def list_runners(self, **kwargs):
            if self.index == 0:
                return {"items": []}
            succeeded = self.index == 2
            return {
                "items": [
                    {
                        "runner_id": "runner-1",
                        "runner_plan_id": "plan-1",
                        "plan_occurrence": 0,
                        "status": "succeeded" if succeeded else "running",
                        "scheduler_id": "scheduler-1",
                        "worker": {
                            "process_id": 70001,
                            "exit_code": 0 if succeeded else None,
                        },
                        "requests": {
                            "started": 1,
                            "completed": 1 if succeeded else 0,
                            "failed": 0,
                        },
                    }
                ]
            }

    with caplog.at_level(logging.INFO, logger="llmperfctl"):
        result = wait_for_campaign(CampaignClient(), "campaign-1", 0, 1)

    assert result["status"] == "completed"
    assert result["outcome"] == "succeeded"
    assert "status=planned" in caplog.text
    assert "runners=1 [queued=0 running=1" in caplog.text
    assert "RunnerPlan plan-1 updated" in caplog.text
    assert "occurrence=2 emitted=2" in caplog.text
    assert "Runner runner-1 status: running" in caplog.text
    assert "plan=plan-1 occurrence=0 scheduler=scheduler-1" in caplog.text
    assert "worker_pid=70001 exit_code=0" in caplog.text
    assert (
        "Campaign campaign-1 finished: status=completed outcome=succeeded"
        in caplog.text
    )


def test_wait_runner_reconnect():
    class SucceededRunnerClient:
        def get_runner(self, runner_id):
            return {
                "runner_id": runner_id,
                "status": "succeeded",
                "label": "recovered",
                "benchmark": {
                    "provider": "aliyun",
                    "model": "deepseek-v4-pro",
                },
                "summary": {
                    "results": {
                        "num_requests_started": 8,
                        "num_completed_requests": 8,
                        "number_errors": 0,
                        "error_rate": 0,
                    },
                    "outcome": {
                        "status": "succeeded",
                        "requests_started": 8,
                        "requests_completed": 8,
                        "requests_failed": 0,
                        "message": "8 benchmark requests completed",
                    },
                },
            }

    arguments = build_parser().parse_args(
        ["runner", "status", "runner-1", "--wait", "--summary"]
    )

    result = execute(SucceededRunnerClient(), arguments)

    assert result["runner_id"] == "runner-1"
    assert result["status"] == "succeeded"
    assert result["requests"]["completed"] == 8


def test_worker_error_summary():
    report = summarize_runner(
        {
            "runner_id": "runner-2",
            "status": "failed",
            "benchmark": {"provider": "test", "model": "model"},
            "error_message": "worker exited with code 1",
        }
    )

    assert report["error"] == {
        "code": None,
        "message": "worker exited with code 1",
    }
