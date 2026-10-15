import asyncio
import json

import pytest

from llmperf_backend.providers import (
    ProviderConfigError,
    ProviderModelDiscovery,
    ProviderRegistry,
)
from llmperf_backend.config import load_provider_profiles
from llmperf_backend.provider_files import load_provider_file

PROVIDER_ENVIRONMENT = {
    "LLMPERF_PROVIDER_DEEPSEEK_URL": "https://models.example/v1",
    "LLMPERF_PROVIDER_DEEPSEEK_KEY": "deepseek-secret",
    "LLMPERF_PROVIDER_GLM_URL": "https://glm.example/v1",
    "LLMPERF_PROVIDER_GLM_KEY": "glm-secret",
    "LLMPERF_PROVIDER_CATALOG_MODELS": "static-b,static-a",
    "LLMPERF_PROVIDER_ANTHROPIC_ADAPTER": "anthropic",
    "LLMPERF_PROVIDER_ANTHROPIC_KEY": "anthropic-secret",
    "LLMPERF_PROVIDER_ANTHROPIC_MODELS": "claude-test",
}


@pytest.mark.parametrize("suffix", ["yaml", "yml", "json"])
def test_provider_document(tmp_path, suffix):
    path = tmp_path / f"providers.{suffix}"
    document = {
        "version": "1.0.0",
        "providers": {
            "acme": {
                "adapter": "openai",
                "base_url": "https://acme.example/v1/",
                "api_key_env": "ACME_KEY",
                "models": ["model-a", "model,b"],
                "discovery": {"mode": "static", "cache_ttl_seconds": 60},
            },
            "claude": {"adapter": "anthropic", "models": ["claude-test"]},
        },
    }
    if suffix == "json":
        content = json.dumps(document)
    else:
        import yaml

        content = yaml.safe_dump(document)
    path.write_text(content)
    registry = ProviderRegistry(load_provider_file(path, {"ACME_KEY": "private-value"}))
    assert registry.require("acme").static_models == ("model-a", "model,b")
    assert registry.require("acme").api_base == "https://acme.example/v1"
    assert registry.require("claude").api_key_env == "ANTHROPIC_API_KEY"
    assert "private-value" not in json.dumps(registry.public_document())
    worker_environment = registry.worker_environment(
        "claude",
        registry.worker_environment(
            "acme", {"ACME_KEY": "private-value", "KEEP": "yes"}
        ),
    )
    assert worker_environment == {"KEEP": "yes"}
    selected_environment = registry.worker_environment(
        "acme", {"ACME_KEY": "private-value"}
    )
    assert selected_environment["OPENAI_API_KEY"] == "private-value"
    assert "ACME_KEY" not in selected_environment


@pytest.mark.parametrize(
    "content",
    [
        'version: "1.0.0"\nproviders: {}',
        'version: "1.0.0"\nproviders: {acme: {models: [model], api_key: secret-literal, api_key_env: ACME_KEY}}',
        'version: "1.0.0"\nproviders: {acme: {models: [model], discovery: {cache_ttl_seconds: "5"}}}',
        'version: "1.0.0"\nproviders: {acme: {base_url: "https://user:secret-literal@example.com"}}',
        'version: "1.0.0"\nproviders: {acme: {models: [model], modles: [typo]}}',
        'version: "1.0.0"\nproviders: {acme: {models: [model]}, acme: {models: [other]}}',
        'version: "1.0.0"\nproviders: {ac_me: {models: [model]}, ac-me: {models: [other]}}',
        'version: "1.0.0"\nproviders: {acme: {models: [""]}}',
        'version: "1.0.0"\nproviders: {acme: {base_url: "https://example.com", discovery: {mode: static}}}',
        'version: "1.0.0"\nproviders: [ secret-literal',
    ],
)
def test_invalid_document(tmp_path, content):
    path = tmp_path / "providers.yaml"
    path.write_text(content)
    with pytest.raises(ProviderConfigError) as failure:
        load_provider_file(path, {})
    assert "secret-literal" not in str(failure.value)
    registry = ProviderRegistry.from_loader(lambda: load_provider_file(path, {}))
    assert registry.list_public() == []
    assert registry.public_document()["configuration_error"]


def test_json_duplicate(tmp_path):
    path = tmp_path / "providers.json"
    path.write_text(
        '{"version":"1.0.0","providers":{"acme":{"models":["a"]},"acme":{"models":["b"]}}}'
    )
    with pytest.raises(ProviderConfigError, match="duplicate"):
        load_provider_file(path, {})


@pytest.mark.parametrize("environment", [{}, {"ACME_KEY": ""}, {"ACME_KEY": "   "}])
def test_empty_credentials(tmp_path, environment):
    path = tmp_path / "providers.yaml"
    path.write_text(
        'version: "1.0.0"\nproviders: {acme: {models: [model], api_key_env: ACME_KEY}}'
    )
    registry = ProviderRegistry.from_loader(
        lambda: load_provider_file(path, environment)
    )
    assert registry.require("acme").api_key == ""
    assert registry.list_public()[0]["api_key_configured"] is False
    assert (
        registry.worker_environment("acme", {"OPENAI_API_KEY": "unrelated-secret"})
        == {}
    )


def test_startup_recovery(tmp_path, caplog):
    path = tmp_path / "providers.yaml"
    registry = ProviderRegistry.from_loader(lambda: load_provider_file(path, {}))
    assert registry.list_public() == []
    assert registry.public_document()["configuration_error"]
    path.write_text('version: "1.0.0"\nproviders: [ secret-literal')
    with pytest.raises(ProviderConfigError):
        registry.reload()
    assert registry.generation == 0
    path.write_text('version: "1.0.0"\nproviders: {acme: {models: [model]}}')
    assert registry.reload()["configuration_error"] is None
    assert registry.require("acme").static_models == ("model",)
    assert "secret-literal" not in caplog.text


def test_loader_failure(caplog):
    def broken_loader():
        raise ValueError("private-secret")

    registry = ProviderRegistry.from_loader(broken_loader)
    with pytest.raises(ProviderConfigError) as failure:
        registry.reload()
    assert "private-secret" not in str(failure.value)
    assert "private-secret" not in caplog.text


@pytest.mark.parametrize("key", ["inline-secret", "", "   "])
def test_inline_credentials(tmp_path, key):
    from llmperf_backend.provider_files import ProviderFileProfile

    path = tmp_path / "providers.json"
    path.write_text(
        json.dumps(
            {
                "version": "1.0.0",
                "providers": {"acme": {"models": ["model"], "api_key": key}},
            }
        )
    )
    registry = ProviderRegistry.from_loader(
        lambda: load_provider_file(path, {"OPENAI_API_KEY": "unrelated-secret"})
    )
    profile = registry.require("acme")
    assert profile.api_key == key.strip()
    assert "inline-secret" not in repr(profile)
    assert "inline-secret" not in repr(
        ProviderFileProfile.model_validate({"api_key": key})
    )
    assert "inline-secret" not in json.dumps(registry.public_document())
    worker = registry.worker_environment("acme", {"OPENAI_API_KEY": "unrelated-secret"})
    assert worker.get("OPENAI_API_KEY", "") == key.strip()


def test_default_credentials(tmp_path):
    path = tmp_path / "providers.yaml"
    path.write_text('version: "1.0.0"\nproviders: {local: {models: [model]}}')
    registry = ProviderRegistry.from_loader(lambda: load_provider_file(path, {}))
    assert registry.require("local").api_key == ""
    assert registry.public_document()["configuration_error"] is None


def test_shared_provider_parser(tmp_path):
    path = tmp_path / "providers.json"
    path.write_text(
        json.dumps(
            {
                "version": "1.0.0",
                "providers": {
                    "test": {
                        "adapter": "anthropic",
                        "base_url": "https://example.com/v1/",
                        "api_key": "  secret  ",
                        "models": ["a", "b"],
                        "discovery": {
                            "mode": "static",
                            "path": "catalog",
                            "cache_ttl_seconds": 42,
                        },
                        "worker_api_base_env": "CUSTOM_URL",
                        "worker_api_key_env": "CUSTOM_KEY",
                    }
                },
            }
        )
    )
    legacy = ProviderRegistry.from_environment(
        {
            "LLMPERF_PROVIDER_TEST_ADAPTER": " ANTHROPIC ",
            "LLMPERF_PROVIDER_TEST_URL": "https://example.com/v1/",
            "LLMPERF_PROVIDER_TEST_KEY": "  secret  ",
            "LLMPERF_PROVIDER_TEST_MODELS": "a, b",
            "LLMPERF_PROVIDER_TEST_DISCOVERY": " STATIC ",
            "LLMPERF_PROVIDER_TEST_PATH": "catalog",
            "LLMPERF_PROVIDER_TEST_TTL": "42",
            "LLMPERF_PROVIDER_TEST_URLVAR": "CUSTOM_URL",
            "LLMPERF_PROVIDER_TEST_KEYVAR": "CUSTOM_KEY",
        }
    )
    assert legacy.require("test") == load_provider_file(path, {})[0]


@pytest.mark.parametrize(
    "field, value",
    [
        ("ADAPTER", "invalid"),
        ("DISCOVERY", "invalid"),
        ("TTL", "-1"),
        ("TTL", "86401"),
        ("TTL", "secret-literal"),
        ("URLVAR", ""),
        ("KEYVAR", ""),
    ],
)
def test_invalid_legacy_fields(field, value):
    with pytest.raises(ProviderConfigError) as failure:
        ProviderRegistry.from_environment(
            {
                "LLMPERF_PROVIDER_TEST_MODELS": "a",
                f"LLMPERF_PROVIDER_TEST_{field}": value,
            }
        )
    assert "secret-literal" not in str(failure.value)


def test_file_reload(tmp_path):
    environment_path = tmp_path / "backend.env"
    environment_path.write_text(
        "ACME_KEY=old-secret\nLLMPERF_PROVIDERS_FILE=providers.json\nLLMPERF_PROVIDER_LEGACY_MODELS=ignored\n"
    )
    provider_path = tmp_path / "providers.json"
    document = {
        "version": "1.0.0",
        "providers": {"acme": {"models": ["a"], "api_key_env": "ACME_KEY"}},
    }
    provider_path.write_text(json.dumps(document))

    def load_profiles():
        return load_provider_profiles(environment_path, process_environment={})

    registry = ProviderRegistry.from_loader(load_profiles)
    snapshot, _ = registry.profile_snapshot("acme")
    assert registry.get("legacy") is None
    environment_path.write_text(
        "ACME_KEY=new-secret\nLLMPERF_PROVIDERS_FILE=providers.json\n"
    )
    result = registry.reload()
    assert result["changes"]["updated"] == ["acme"]
    assert registry.require("acme").api_key == "new-secret"
    assert snapshot.api_key == "old-secret"
    assert (
        load_provider_profiles(
            environment_path, process_environment={"ACME_KEY": "process-secret"}
        )[0].api_key
        == "process-secret"
    )
    provider_path.write_text('{"bad":"secret-literal"}')
    with pytest.raises(ProviderConfigError):
        registry.reload()
    assert registry.generation == 1
    assert registry.require("acme").api_key == "new-secret"
    document["providers"] = {"other": {"models": ["b"]}}
    provider_path.write_text(json.dumps(document))
    registry.reload()
    assert "ACME_KEY" not in registry.worker_environment(
        "other", {"ACME_KEY": "new-secret"}
    )


def test_provider_file_selection(tmp_path):
    environment_path = tmp_path / "backend.env"
    environment_path.write_text("LLMPERF_PROVIDER_LEGACY_MODELS=a\n")
    assert (
        load_provider_profiles(environment_path, process_environment={})[0].provider_id
        == "legacy"
    )
    provider_path = tmp_path / "providers.yaml"
    provider_path.write_text('version: "1.0.0"\nproviders: {new: {models: [a]}}')
    assert (
        load_provider_profiles(environment_path, process_environment={})[0].provider_id
        == "new"
    )
    (tmp_path / "providers.json").write_text("{}")
    with pytest.raises(ProviderConfigError, match="Multiple"):
        load_provider_profiles(environment_path, process_environment={})
    selected = load_provider_profiles(
        environment_path,
        process_environment={"LLMPERF_PROVIDERS_FILE": str(provider_path)},
    )
    assert selected[0].provider_id == "new"
    with pytest.raises(ProviderConfigError, match="Unable to read"):
        load_provider_profiles(
            environment_path,
            process_environment={"LLMPERF_PROVIDERS_FILE": "missing.yaml"},
        )


def test_public_profile():
    registry = ProviderRegistry.from_environment(PROVIDER_ENVIRONMENT)

    profiles = {item["id"]: item for item in registry.list_public()}
    assert set(profiles) == {"anthropic", "catalog", "deepseek", "glm"}
    assert profiles["deepseek"]["api_key_configured"] is True
    assert "api_key" not in profiles["deepseek"]
    assert profiles["deepseek"]["adapter"] == "openai"
    assert profiles["deepseek"]["model_discovery"]["mode"] == "openai"
    assert profiles["catalog"]["model_discovery"]["mode"] == "static"
    assert profiles["catalog"]["typical_models"] == ["static-b", "static-a"]
    assert profiles["anthropic"]["typical_models"] == ["claude-test"]
    assert profiles["deepseek"]["typical_models"] == []

    anthropic = registry.require("anthropic")
    assert anthropic.api_key_env == "ANTHROPIC_API_KEY"
    assert anthropic.model_cache_ttl_seconds == 300

    benchmark = registry.resolve_benchmark(
        {"provider": "deepseek", "model": "deepseek-reasoner"}
    )
    assert benchmark["provider"] == "deepseek"
    assert benchmark["model"] == "deepseek-reasoner"
    assert benchmark["adapter"] == "openai"

    static_benchmark = registry.resolve_benchmark(
        {"provider": "catalog", "model": "static-a"}
    )
    assert static_benchmark["model"] == "static-a"
    with pytest.raises(
        ProviderConfigError, match="configured models: static-b, static-a"
    ):
        registry.resolve_benchmark({"provider": "catalog", "model": "static-c"})


def test_model_preview():
    registry = ProviderRegistry.from_environment(
        {"LLMPERF_PROVIDER_TEST_MODELS": "model-d,model-c,model-b,model-a"}
    )

    profile = registry.list_public()[0]

    assert profile["typical_models"] == ["model-d", "model-c", "model-b"]
    assert profile["model_discovery"]["static_model_count"] == 4


def test_worker_credentials():
    registry = ProviderRegistry.from_environment(PROVIDER_ENVIRONMENT)
    base_environment = dict(PROVIDER_ENVIRONMENT)
    base_environment["UNRELATED"] = "retained"

    environment = registry.worker_environment("glm", base_environment)

    assert environment["OPENAI_API_BASE"] == "https://glm.example/v1"
    assert environment["OPENAI_API_KEY"] == "glm-secret"
    assert environment["UNRELATED"] == "retained"
    assert not any(name.startswith("LLMPERF_PROVIDER_") for name in environment)
    assert "deepseek-secret" not in environment.values()
    assert "anthropic-secret" not in environment.values()


@pytest.mark.parametrize(
    "environment",
    [
        {"LLMPERF_PROVIDER_BAD_URL": "https://models.example/v1?key=x"},
        {"LLMPERF_PROVIDER_BAD_KEYVAR": "INVALID-NAME"},
    ],
)
def test_unsafe_routing(environment):
    with pytest.raises(ProviderConfigError):
        ProviderRegistry.from_environment(environment)


def test_unknown_provider_fields():
    with pytest.raises(ProviderConfigError, match="At least one provider"):
        ProviderRegistry.from_environment(
            {
                "LLMPERF_PROVIDER_DEEPSEEK_APIBASE": "https://models.example/v1",
                "LLMPERF_PROVIDER_DEEPSEEK_APIKEY": "old-secret",
            }
        )


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps({"data": [{"id": "model-b"}, {"id": "model-a"}]}).encode(
            "utf-8"
        )


def test_remote_discovery(monkeypatch):
    registry = ProviderRegistry.from_environment(PROVIDER_ENVIRONMENT)
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        assert timeout == 10.0
        return _FakeResponse()

    monkeypatch.setattr("llmperf_backend.providers.urlopen", fake_urlopen)
    discovery = ProviderModelDiscovery(registry)

    models = discovery._fetch_openai_models(registry.require("deepseek"))

    assert models == ("model-b", "model-a")
    assert len(requests) == 1
    assert requests[0].full_url == "https://models.example/v1/models"
    assert requests[0].get_header("Authorization") == "Bearer deepseek-secret"


def test_static_discovery():
    registry = ProviderRegistry.from_environment(PROVIDER_ENVIRONMENT)
    discovery = ProviderModelDiscovery(registry)

    first = asyncio.run(discovery.models("catalog"))
    second = asyncio.run(discovery.models("catalog"))

    assert first["models"] == ["static-a", "static-b"]
    assert first["cached"] is False
    assert second["cached"] is True


def test_atomic_reload():
    active = {
        "LLMPERF_PROVIDER_TEST_URL": "https://old.example/v1",
        "LLMPERF_PROVIDER_TEST_KEY": "old-secret",
        "LLMPERF_PROVIDER_TEST_KEYVAR": "TEST_PROVIDER_KEY",
    }
    registry = ProviderRegistry.from_environment(
        active, reload_loader=lambda: dict(active)
    )

    active.update(
        {
            "LLMPERF_PROVIDER_TEST_URL": "https://new.example/v1",
            "LLMPERF_PROVIDER_TEST_KEY": "new-secret",
            "LLMPERF_PROVIDER_EXTRA_MODELS": "model-b,model-a",
        }
    )
    result = registry.reload()

    assert result["generation"] == 1
    assert result["changes"] == {
        "added": ["extra"],
        "updated": ["test"],
        "removed": [],
    }
    assert "new-secret" not in json.dumps(result)
    environment = registry.worker_environment("test", {})
    assert environment["OPENAI_API_BASE"] == "https://new.example/v1"
    assert environment["TEST_PROVIDER_KEY"] == "new-secret"

    active["LLMPERF_PROVIDER_TEST_URL"] = "not-a-url"
    with pytest.raises(ProviderConfigError):
        registry.reload()
    assert registry.generation == 1
    assert registry.require("test").api_base == "https://new.example/v1"

    for name in (
        "LLMPERF_PROVIDER_TEST_URL",
        "LLMPERF_PROVIDER_TEST_KEY",
        "LLMPERF_PROVIDER_TEST_KEYVAR",
    ):
        active.pop(name, None)
    removed = registry.reload()
    assert removed["changes"]["removed"] == ["test"]
    environment = registry.worker_environment(
        "extra", {"TEST_PROVIDER_KEY": "must-not-leak", "KEEP": "yes"}
    )
    assert "TEST_PROVIDER_KEY" not in environment
    assert environment["KEEP"] == "yes"
