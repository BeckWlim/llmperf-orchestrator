"""Registration and lookup for Campaign protocol compilers."""

from typing import Dict

from llmperf_backend.protocols.base import ProtocolPlugin


_PLUGINS: Dict[str, ProtocolPlugin] = {}


def register_protocol_plugin(plugin: ProtocolPlugin) -> None:
    if plugin.protocol in _PLUGINS:
        raise ValueError(f"protocol plugin already registered: {plugin.protocol}")
    _PLUGINS[plugin.protocol] = plugin


def get_protocol_plugin(protocol: str) -> ProtocolPlugin:
    try:
        return _PLUGINS[protocol]
    except KeyError as exc:
        raise ValueError(f"unsupported protocol plugin: {protocol}") from exc


def registered_protocols() -> tuple[str, ...]:
    return tuple(sorted(_PLUGINS))
