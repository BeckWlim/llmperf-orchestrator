import asyncio
import json

import pytest

from llmperf_backend.providers import (
    ProviderConfigError,
    ProviderModelDiscovery,
    ProviderRegistry,
)


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
        {"provider": "deepseek", "model": "deepseek-reasoner", "llm_api": "litellm"}
    )
    assert benchmark["provider"] == "deepseek"
    assert benchmark["model"] == "deepseek-reasoner"
    assert benchmark["llm_api"] == "openai"

    static_benchmark = registry.resolve_benchmark(
        {"provider": "catalog", "model": "static-a"}
    )
    assert static_benchmark["model"] == "static-a"
    with pytest.raises(ProviderConfigError, match="configured models: static-b, static-a"):
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
