"""Prepared atomic steps: the boundary between workload preparation and execution."""

from dataclasses import dataclass
from datetime import datetime
from functools import update_wrapper
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    ParamSpec,
    Protocol,
    Sequence,
    Tuple,
)

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


BenchmarkResult = Tuple[Dict[str, Any], List[Dict[str, Any]]]


@dataclass(frozen=True)
class PreparedBatch:
    """Yield prepared work so asynchronous execution occupies no preparation thread."""

    steps: Sequence[RequestStep]
    window: int
    timeout_seconds: float
    executor: Optional[BatchExecutor] = None


BenchmarkStages = Generator[PreparedBatch, List[LLMResponse], BenchmarkResult]
P = ParamSpec("P")


class staged_benchmark(Generic[P]):
    """Keep the synchronous API while exposing a resumable preparation boundary."""

    def __init__(self, stages: Callable[P, BenchmarkStages]):
        self.stages = stages
        update_wrapper(self, stages)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> BenchmarkResult:
        stages = self.stages(*args, **kwargs)
        try:
            batch = next(stages)
            while True:
                if batch.executor is None:
                    raise RuntimeError(
                        "Prepared batches require an asynchronous executor"
                    )
                responses = batch.executor(
                    batch.steps, batch.window, batch.timeout_seconds
                )
                batch = stages.send(responses)
        except StopIteration as finished:
            return finished.value
        finally:
            stages.close()
