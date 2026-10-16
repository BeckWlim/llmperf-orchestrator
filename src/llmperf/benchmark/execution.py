"""Prepared atomic steps: the boundary between workload preparation and execution."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Protocol, Sequence

from llmperf.clients.base import LLMResponse
from llmperf.clients.models import RequestConfig


@dataclass(frozen=True)
class RequestStep:
    step_id: str
    request: RequestConfig
    dependencies: tuple[str, ...] = ()
    delay_seconds: float = 0.0
    due_at: Optional[datetime] = None
    message_sources: tuple[str, ...] = ()


class BatchExecutor(Protocol):
    def __call__(
        self, steps: Sequence[RequestStep], window: int, timeout_seconds: float
    ) -> List[LLMResponse]: ...
