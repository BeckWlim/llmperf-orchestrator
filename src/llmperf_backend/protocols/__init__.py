"""Built-in durable Campaign protocol plugins."""

from llmperf_backend.protocols.cache_residency import CacheResidencyPlugin
from llmperf_backend.protocols.cache_retention import CacheRetentionPlugin
from llmperf_backend.protocols.registry import (
    get_protocol_plugin,
    register_protocol_plugin,
    registered_protocols,
)


register_protocol_plugin(CacheRetentionPlugin())
register_protocol_plugin(CacheResidencyPlugin())


__all__ = ["get_protocol_plugin", "registered_protocols"]
