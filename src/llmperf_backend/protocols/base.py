"""Generic compile-time contract for durable Campaign protocol plugins."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol


@dataclass(frozen=True)
class ProtocolCompileContext:
    campaign_id: str
    definition_id: str
    definition_name: str
    protocol: str
    config: Dict[str, Any]
    runner_template: Dict[str, Any]
    database_now: datetime
    created_by: str


@dataclass(frozen=True)
class DispatchBlueprint:
    dispatch_id: str
    role: str
    state: str
    due_at: Optional[datetime]
    parent_dispatch_id: Optional[str]
    runner_template: Dict[str, Any]
    lineage: Dict[str, Any]


@dataclass(frozen=True)
class InstanceBlueprint:
    instance_id: str
    instance_key: str
    spec: Dict[str, Any]
    checkpoint: Dict[str, Any]
    outcome: Dict[str, Any]
    dispatches: List[DispatchBlueprint]


class ProtocolPlugin(Protocol):
    protocol: str

    def compile(self, context: ProtocolCompileContext) -> List[InstanceBlueprint]:
        """Compile a validated definition into a durable, finite dispatch graph."""
