from argparse import Namespace
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
    _validate_runner_list,
    build_parser,
    execute,
    print_runner_table,
    start_campaign,
    summarize_runner,
    wait_for_runners,
)
from llmperf_cli.auth import discover_private_key_providers
from llmperf_cli.client import ClientError, LLMPerfClient


class FakeClient:
    def __init__(self):
        self.payloads = []

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

    def create_campaign_with_runners(self, campaign, runners):
        assert campaign["name"] == "glm-study"
        payloads = []
        for runner in runners:
            payload = dict(runner)
            payload["campaign_id"] = "campaign-1"
            payloads.append(self.start_runner(payload))
        return {
            "campaign": {"campaign_id": "campaign-1"},
            "items": payloads,
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


def test_campaign_status_full_uses_export_report():
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


def test_auth_defaults():
    arguments = build_parser().parse_args(["auth", "list"])

    assert arguments.ssh_dir == "~/.ssh"
    assert arguments.no_key_discovery is False


def test_main_help():
    output = build_parser().format_help()

    assert "Runner     One durable benchmark execution" in output
    assert "Scheduler  The backend component" in output
    assert "llmperfctl provider models <provider-id>" in output
    assert "llmperfctl runner start -f runner.yaml" in output


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["provider", "--help"], "LLMPERF_DEFAULT_PROVIDER"),
        (["runner", "--help"], "A Runner is one durable benchmark execution"),
        (["campaign", "--help"], "See examples/glm-campaign.yaml"),
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
            "argv": ["provider", "models", "deepseek", "--refresh"],
            "expected": {
                "command": "provider",
                "provider_command": "models",
                "provider_id": "deepseek",
                "refresh": True,
            },
        },
        id="provider-models",
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
            "argv": ["campaign", "status", "campaign-1"],
            "expected": {
                "campaign_command": "status",
                "campaign_id": "campaign-1",
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
    assert "must not be rendered" not in output


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


def test_provider_encoding(monkeypatch):
    requested_urls = []

    def fake_urlopen(request, timeout):
        requested_urls.append(request.full_url)
        return _FakeResponse({"models": []})

    monkeypatch.setattr("llmperf_cli.client.urlopen", fake_urlopen)
    client = LLMPerfClient("http://127.0.0.1:8000")

    client.list_provider_models("team/provider", refresh=True)

    assert requested_urls == [
        "http://127.0.0.1:8000/api/v1/providers/team%2Fprovider/models?refresh=true"
    ]


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


def test_wait_summary(caplog):
    class FailedRunnerClient:
        def get_runner(self, runner_id):
            return {
                "runner_id": runner_id,
                "status": "failed",
                "label": "smoke",
                "benchmark": {"provider": "aliyun", "model": "deepseek-v4-pro"},
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
            "scheduler_id": None,
            "worker": None,
            "started_at": None,
            "finished_at": None,
        }
    ]
    assert "large output" not in str(reports)
    assert "Runner runner-1 status: failed" in caplog.text
    assert "Runner runner-1: No benchmark requests completed" in caplog.text
    assert _has_unsuccessful_runner(reports) is True


def test_status_wait_reconnects_to_existing_runner():
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
