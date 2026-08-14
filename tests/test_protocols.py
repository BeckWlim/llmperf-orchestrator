from datetime import datetime, timezone

from llmperf_backend.models import CacheResidencyProtocolCreate, dump_model
from llmperf_backend.protocols import get_protocol_plugin, registered_protocols
from llmperf_backend.protocols.base import ProtocolCompileContext


def _runner_template():
    return {
        "label": "residency",
        "metadata": {},
        "benchmark": {
            "provider": "test",
            "model": "cache-model",
            "llm_api": "openai",
            "mean_input_tokens": 4096,
            "stddev_input_tokens": 0,
            "mean_output_tokens": 16,
            "stddev_output_tokens": 0,
        },
    }


def test_geographic_protocol():
    definition = CacheResidencyProtocolCreate.model_validate(
        {
            "name": "three-days-hourly",
            "protocol": "cache-residency/v1",
            "schedule": {
                "kind": "geographic",
                "timezone": "Asia/Shanghai",
                "starts_at": "2026-08-15T00:00:00+08:00",
                "every_seconds": 3600,
                "duration_days": 3,
            },
            "mapping": "one_to_one",
            "chains": 1,
            "seed": 123,
            "cold_control": True,
            "runner": {},
        }
    )
    config = dump_model(definition)
    config.pop("runner")
    plugin = get_protocol_plugin("cache-residency/v1")

    instances = plugin.compile(
        ProtocolCompileContext(
            campaign_id="campaign-1",
            definition_id="definition-1",
            definition_name=definition.name,
            protocol=definition.protocol,
            config=config,
            runner_template=_runner_template(),
            database_now=datetime(2026, 8, 14, tzinfo=timezone.utc),
            created_by="test",
        )
    )

    assert registered_protocols() == ("cache-residency/v1", "cache-retention/v1")
    assert len(instances) == 1
    instance = instances[0]
    assert len(instance.spec["observations"]) == 72
    assert instance.spec["observations"][0]["offset_seconds"] == 3600
    assert instance.spec["observations"][-1]["offset_seconds"] == 72 * 3600
    assert len(instance.dispatches) == 1 + 72 * 2
    prime = instance.dispatches[0]
    assert prime.runner_template["benchmark"]["max_completed_requests"] == 72
    mapping_keys = prime.runner_template["metadata"]["protocol"]["mapping_keys"]
    assert len(mapping_keys) == 72
    warm_dispatches = [
        dispatch
        for dispatch in instance.dispatches
        if dispatch.role.startswith("warm:")
    ]
    warm_mappings = [
        dispatch.runner_template["metadata"]["protocol"]["mapping_key"]
        for dispatch in warm_dispatches
    ]
    assert warm_mappings == mapping_keys
    assert all(
        child.parent_dispatch_id == parent.dispatch_id
        for parent, child in zip(instance.dispatches, instance.dispatches[1:])
    )
