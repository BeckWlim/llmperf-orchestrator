"""Worker-template primitives and deterministic compilation for one Runner."""

from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from llmperf.configuration.duration import Duration
from llmperf.benchmark.execution import RequestStep


MAX_REQUEST_STEPS = 100_000


class RequestDependency(BaseModel):
    """Wait for a committed result, optionally reusing its structured message input/output."""

    model_config = ConfigDict(extra="forbid")
    step: str = Field(min_length=1, max_length=120)
    reuse_messages: bool = False

    @model_validator(mode="before")
    @classmethod
    def parse_name(cls, value: object) -> object:
        return {"step": value} if isinstance(value, str) else value


class RequestStepSpec(BaseModel):
    """One Worker template override and its readiness requirements."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    input: Optional[str] = Field(default=None, min_length=1, max_length=100)
    depends_on: list[RequestDependency] = Field(default_factory=list)
    after: Duration = Field(default_factory=lambda: Duration(value=0))
    at: Optional[datetime] = None

    @property
    def dependencies(self) -> tuple[str, ...]:
        return tuple(dependency.step for dependency in self.depends_on)

    @property
    def message_sources(self) -> tuple[str, ...]:
        return tuple(
            dependency.step
            for dependency in self.depends_on
            if dependency.reuse_messages
        )

    @model_validator(mode="after")
    def validate_dependencies(self) -> "RequestStepSpec":
        validate_dependencies(self.depends_on)
        return self

    @field_validator("at")
    @classmethod
    def validate_instant(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is not None and value.tzinfo is None:
            raise ValueError("step at must include a UTC offset")
        return value


class RequestPlan(BaseModel):
    """Expand a Worker template once by default, or repeat its complete step graph."""

    model_config = ConfigDict(extra="forbid")
    repeat: int = Field(default=1, ge=1, le=MAX_REQUEST_STEPS)
    every: Optional[Duration] = None
    input: str = Field(default="default", min_length=1, max_length=100)
    steps: Optional[list[RequestStepSpec]] = Field(
        default=None, min_length=1, max_length=10_000
    )

    @property
    def worker_count(self) -> int:
        return self.repeat * (len(self.steps) if self.steps is not None else 1)

    @model_validator(mode="after")
    def validate_graph(self) -> "RequestPlan":
        if self.every is not None and self.steps is not None:
            raise ValueError("worker.every cannot combine with explicit steps")
        if self.worker_count > MAX_REQUEST_STEPS:
            raise ValueError("request expansion exceeds 100000 Workers")
        seen: set[str] = set()
        for step in self.steps or ():
            if step.name in seen or not set(step.dependencies) <= seen:
                raise ValueError(
                    "steps need unique names and dependencies on earlier steps"
                )
            seen.add(step.name)
        return self

    def expand(self) -> list[RequestStepSpec]:
        """Scope dependencies and reused messages separately in every repetition."""
        if self.steps is None:
            return [
                RequestStepSpec(name=str(index), input=self.input)
                for index in range(self.repeat)
            ]
        expanded: list[RequestStepSpec] = []
        for repetition in range(self.repeat):
            prefix = f"{repetition}." if self.repeat > 1 else ""
            for step in self.steps:
                expanded.append(
                    step.model_copy(
                        update={
                            "name": prefix + step.name,
                            "input": step.input or self.input,
                            "depends_on": [
                                dependency.model_copy(
                                    update={"step": prefix + dependency.step}
                                )
                                for dependency in step.depends_on
                            ],
                        },
                        deep=True,
                    )
                )
        return expanded


def request_plan(benchmark: Mapping[str, Any]) -> RequestPlan:
    """Decode the plan portion of a validated execution document."""
    return RequestPlan.model_validate(
        {
            name: benchmark[name]
            for name in ("repeat", "every", "steps", "input")
            if name in benchmark
        }
    )


def planned_worker_count(benchmark: Mapping[str, Any]) -> int:
    """Count already-validated definitions without expanding or preparing any requests."""
    probe = benchmark.get("cache_probe")
    if probe is not None:
        return int(probe["trials"]) * (1 + int(probe.get("repeats_after_prime", 1)))
    steps = benchmark.get("steps")
    return int(benchmark.get("repeat", 1)) * (len(steps) if steps is not None else 1)


def planned_token_budget(benchmark: Mapping[str, Any]) -> int:
    """Conservative token estimate including message reuse along each step lineage."""
    input_tokens = int(benchmark.get("mean_input_tokens", 550))
    output_tokens = int(benchmark.get("mean_output_tokens", 150))
    if benchmark.get("cache_probe") is not None:
        return planned_worker_count(benchmark) * (input_tokens + output_tokens)
    plan = request_plan(benchmark)
    history_tokens: dict[str, int] = {}
    iteration_tokens = 0
    for step in plan.steps or [RequestStepSpec(name="request")]:
        input_definition = (benchmark.get("inputs") or {}).get(
            step.input or plan.input, {}
        )
        prompt = input_definition.get("text", benchmark.get("prompt"))
        tokens = input_definition.get("tokens", {})
        mean_tokens = (
            int(tokens.get("mean", input_tokens))
            if isinstance(tokens, dict)
            else int(tokens)
        )
        prompt_tokens = (
            max(mean_tokens, len(prompt.encode("utf-8")))
            if prompt is not None
            else mean_tokens
        )
        previous_tokens = sum(history_tokens[source] for source in step.message_sources)
        step_tokens = previous_tokens + prompt_tokens + output_tokens
        history_tokens[step.name] = step_tokens
        iteration_tokens += step_tokens
    return plan.repeat * iteration_tokens


def compile_requests(
    benchmark: Mapping[str, Any], prepared_steps: Sequence[RequestStep]
) -> Sequence[RequestStep]:
    """Bind prepared request data to the declared Worker graph; no execution or I/O."""
    if benchmark.get("steps") is None:
        return prepared_steps
    specifications = request_plan(benchmark).expand()
    if len(specifications) != len(prepared_steps):
        raise ValueError("Prepared Workers do not match the declared request expansion")
    compiled_steps: list[RequestStep] = []
    for specification, prepared_step in zip(specifications, prepared_steps):
        compiled_steps.append(
            RequestStep(
                specification.name,
                prepared_step.request.model_copy(deep=True),
                specification.dependencies,
                specification.after.seconds,
                specification.at,
                specification.message_sources,
            )
        )
    return compiled_steps


def validate_dependencies(dependencies: Sequence[RequestDependency]) -> None:
    if len({dependency.step for dependency in dependencies}) != len(dependencies):
        raise ValueError("depends_on cannot contain duplicate references")
