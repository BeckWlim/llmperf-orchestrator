"""Validated request and configuration models used by the backend."""

from datetime import datetime, time
from itertools import product
from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from llmperf.prompt_datasets import (
    validate_external_dataset_adapter,
)
from llmperf.version import PROTOCOL_VERSION, ProtocolVersion

DEFAULT_TOKENIZER_ID = "hf-internal-testing/llama-tokenizer"


class StrictModel(BaseModel):
    """Base model that rejects misspelled or unsupported configuration keys."""

    model_config = ConfigDict(extra="forbid")


class ServerConfig(StrictModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: Literal["critical", "error", "warning", "info", "debug", "trace"] = (
        "info"
    )
    workers: int = Field(default=1, ge=1)
    reload: bool = False


class TokenizerSpec(StrictModel):
    """A Hugging Face tokenizer requested by one benchmark."""

    source: Literal["huggingface"] = "huggingface"
    id: str = Field(min_length=1, max_length=200)
    revision: Optional[str] = Field(default=None, min_length=1, max_length=200)
    use_fast: bool = True


class ResolvedTokenizerSpec(StrictModel):
    """Backend-owned tokenizer identity persisted with a resolved Runner."""

    source: Literal["huggingface"] = "huggingface"
    id: str = Field(min_length=1, max_length=200)
    revision: str = Field(min_length=1, max_length=200)
    use_fast: bool = True
    requested_revision: str = Field(min_length=1, max_length=200)
    immutable_revision: bool
    selection: Literal["explicit", "global_default"]
    accuracy: Literal["compatible", "approximate"]


class CacheProbeConfig(StrictModel):
    """A deterministic, paired prompt-cache experiment within one Runner."""

    mode: Literal["exact_repeat", "shared_prefix", "early_mutation", "late_mutation"]
    trials: int = Field(default=20, ge=2, le=10_000)
    repeats_after_prime: int = Field(default=1, ge=1, le=100)
    schedule: Literal["randomized_family_blocks"] = "randomized_family_blocks"
    shared_prefix_tokens: Optional[int] = Field(default=None, gt=0)
    mutation_token_offset: Optional[int] = Field(default=None, ge=0)
    delay_seconds: float = Field(default=0, ge=0, le=60)
    persist_prompt_text: bool = False
    allow_approximate_tokenizer: bool = False
    tokenizer_divergence_warning_ratio: float = Field(default=0.05, ge=0, le=1)
    bootstrap_samples: int = Field(default=2_000, ge=100, le=100_000)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1)
    minimum_counter_coverage: float = Field(default=0.8, ge=0, le=1)


class CompiledTaskContext(StrictModel):
    """Compiler-owned identity for one atomic task-graph invocation."""

    definition_id: str = Field(min_length=1, max_length=36)
    instance_id: str = Field(min_length=1, max_length=36)
    node_id: str = Field(min_length=1, max_length=120)
    role: str = Field(min_length=1, max_length=64)
    payload_id: str = Field(min_length=1, max_length=64)
    payload_seed: int = Field(ge=0, le=2_147_483_647)
    trial_index: int = Field(ge=0)
    dimensions: Dict[str, int] = Field(default_factory=dict)


class DatasetSpec(StrictModel):
    """A backend-resolved Hugging Face dataset artifact."""

    source: Literal["huggingface"] = "huggingface"
    id: str = Field(min_length=1, max_length=200)
    filename: str = Field(min_length=1, max_length=500)
    revision: Optional[str] = Field(default=None, min_length=1, max_length=200)
    adapter: str = Field(min_length=1, max_length=64)

    @field_validator("adapter")
    @classmethod
    def validate_adapter(cls, value: str) -> str:
        try:
            return validate_external_dataset_adapter(value)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc


class _BenchmarkFields(StrictModel):
    """Fields and validation shared by submitted and resolved benchmarks."""

    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1)
    timeout_seconds: int = Field(default=90, gt=0)
    max_completed_requests: int = Field(default=10, gt=0)
    concurrent_requests: int = Field(default=1, gt=0)
    mean_input_tokens: int = Field(default=550, gt=0)
    stddev_input_tokens: int = Field(default=150, ge=0)
    shared_prefix_tokens: int = Field(default=0, ge=0)
    dataset: Optional[DatasetSpec] = None
    dataset_prompt_mode: Literal["sample", "concatenate"] = "sample"
    dataset_repeat_count: int = Field(default=1, ge=1)
    dataset_seed: int = 11111
    mean_output_tokens: int = Field(default=150, gt=0)
    stddev_output_tokens: int = Field(default=80, ge=0)
    additional_sampling_params: Dict[str, Any] = Field(default_factory=dict)
    cache_probe: Optional[CacheProbeConfig] = None

    @model_validator(mode="after")
    def validate_cache_probe(self) -> "_BenchmarkFields":
        if self.dataset is not None and self.shared_prefix_tokens:
            raise ValueError("dataset and shared_prefix_tokens cannot be used together")
        if self.dataset is None and self.dataset_prompt_mode != "sample":
            raise ValueError("dataset_prompt_mode requires dataset")
        if self.dataset_prompt_mode == "concatenate" and self.dataset_repeat_count != 1:
            raise ValueError(
                "concatenated dataset prompts require dataset_repeat_count=1"
            )
        if self.cache_probe is None:
            return self
        if self.cache_probe.mode == "shared_prefix":
            prefix = self.cache_probe.shared_prefix_tokens or self.shared_prefix_tokens
            if not prefix:
                raise ValueError(
                    "shared_prefix cache_probe requires shared_prefix_tokens"
                )
            if prefix >= self.mean_input_tokens:
                raise ValueError(
                    "cache_probe shared_prefix_tokens must be less than mean_input_tokens"
                )
        return self


class BenchmarkConfig(_BenchmarkFields):
    """Defaults matching the existing token benchmark command-line options."""

    tokenizer: TokenizerSpec = Field(
        default_factory=lambda: TokenizerSpec(id=DEFAULT_TOKENIZER_ID)
    )


class ResolvedBenchmarkConfig(_BenchmarkFields):
    """Backend-owned execution fields added after Provider and artifact resolution."""

    adapter: Literal["openai", "anthropic", "litellm", "sagemaker", "vertexai"]
    tokenizer: ResolvedTokenizerSpec


class CompiledBenchmarkConfig(ResolvedBenchmarkConfig):
    """Internal one-request benchmark emitted only by the task compiler."""

    task_context: CompiledTaskContext

    @model_validator(mode="after")
    def validate_compiled_shape(self) -> "CompiledBenchmarkConfig":
        if self.cache_probe is not None:
            raise ValueError("compiled benchmark cannot define cache_probe")
        if self.max_completed_requests != 1 or self.concurrent_requests != 1:
            raise ValueError("compiled benchmark must contain exactly one request")
        return self


class DatabaseConfig(StrictModel):
    """Asynchronous SQLAlchemy connection settings."""

    url: str = Field(default="postgresql+asyncpg:///llmperf", min_length=1)
    echo: bool = False
    auto_create_schema: bool = True
    pool_size: int = Field(default=5, ge=1)
    max_overflow: int = Field(default=10, ge=0)

    @model_validator(mode="after")
    def validate_postgresql(self) -> "DatabaseConfig":
        if not self.url.startswith("postgresql+asyncpg://"):
            raise ValueError("database.url must use postgresql+asyncpg")
        return self


class SchedulerConfig(StrictModel):
    """Durable benchmark scheduling settings."""

    enabled: bool = True
    max_concurrent_runners: int = Field(default=1, ge=1)
    poll_interval_seconds: float = Field(default=1.0, gt=0)
    cancel_grace_seconds: float = Field(default=5.0, gt=0)
    stale_after_seconds: int = Field(default=300, gt=0)
    log_bytes_limit: int = Field(default=1_000_000, ge=1024)
    ray_address: Optional[str] = Field(default=None, min_length=1)
    ray_num_cpus: int = Field(default=8, ge=1, le=1024)
    ray_actor_num_cpus: float = Field(default=1.0, gt=0, le=64)
    ray_object_store_memory_bytes: int = Field(default=268_435_456, ge=78_643_200)
    ray_health_interval_seconds: float = Field(default=5.0, gt=0, le=300)
    ray_health_timeout_seconds: float = Field(default=3.0, gt=0, le=60)
    artifact_resolution_timeout_seconds: float = Field(default=60.0, gt=0, le=3600)

    @field_validator("ray_address", mode="before")
    @classmethod
    def empty_ray_address(cls, value: Any) -> Any:
        return None if value == "" else value


class PerformanceGuardConfig(StrictModel):
    """Fail-closed limits for runtime fan-out and submitted workloads."""

    enabled: bool = True
    max_runner_concurrency: int = Field(default=32, ge=1)
    max_effective_concurrency: int = Field(default=32, ge=1)
    max_campaign_runners: int = Field(default=1_000, ge=1)
    max_campaign_provider_requests: int = Field(default=10_000, ge=1)
    max_campaign_token_budget: int = Field(default=100_000_000, ge=1)
    warning_ratio: float = Field(default=0.8, gt=0, lt=1)
    max_host_memory_utilization: float = Field(default=0.90, gt=0, lt=1)
    resume_host_memory_utilization: float = Field(default=0.80, gt=0, lt=1)
    min_ray_object_store_available_ratio: float = Field(default=0.10, ge=0, lt=1)
    resume_ray_object_store_available_ratio: float = Field(default=0.20, gt=0, le=1)
    sample_interval_seconds: float = Field(default=1.0, gt=0, le=60)

    @model_validator(mode="after")
    def validate_runtime_watermarks(self) -> "PerformanceGuardConfig":
        if self.resume_host_memory_utilization >= self.max_host_memory_utilization:
            raise ValueError(
                "resume_host_memory_utilization must be lower than "
                "max_host_memory_utilization"
            )
        if (
            self.resume_ray_object_store_available_ratio
            <= self.min_ray_object_store_available_ratio
        ):
            raise ValueError(
                "resume_ray_object_store_available_ratio must be greater than "
                "min_ray_object_store_available_ratio"
            )
        return self


class PlannerConfig(StrictModel):
    """Lightweight RunnerPlan materialization settings."""

    enabled: bool = True
    poll_interval_seconds: float = Field(default=1.0, gt=0)
    batch_size: int = Field(default=20, ge=1, le=1_000)


class AuthConfig(StrictModel):
    """Fixed public-key JWT verification settings."""

    enabled: bool = False
    public_key_path: Optional[str] = None
    algorithm: Literal["RS256"] = "RS256"
    bootstrap_subject: str = Field(default="llmperf-admin", min_length=1)
    issuer: str = Field(default="llmperfctl", min_length=1)
    audience: str = Field(default="llmperf-api", min_length=1)
    leeway_seconds: int = Field(default=5, ge=0, le=60)
    reload_interval_seconds: float = Field(default=1.0, ge=0, le=300)
    previous_key_grace_seconds: int = Field(default=120, ge=0, le=3600)


class AppConfig(StrictModel):
    version: ProtocolVersion
    environment: str = "development"
    server: ServerConfig = Field(default_factory=ServerConfig)
    benchmark: BenchmarkConfig
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    performance_guard: PerformanceGuardConfig = Field(
        default_factory=PerformanceGuardConfig
    )
    auth: AuthConfig = Field(default_factory=AuthConfig)

    @model_validator(mode="after")
    def validate_ray_runtime_shape(self) -> "AppConfig":
        if (
            self.scheduler.enabled
            and self.scheduler.ray_address is None
            and self.server.workers != 1
        ):
            raise ValueError(
                "embedded Ray requires server.workers=1; configure scheduler.ray_address "
                "for a multi-process Backend"
            )
        actor_capacity = int(
            self.scheduler.ray_num_cpus / self.scheduler.ray_actor_num_cpus
        )
        minimum_actor_demand = (
            self.server.workers * self.scheduler.max_concurrent_runners
        )
        if self.scheduler.enabled and minimum_actor_demand > actor_capacity:
            raise ValueError(
                "Ray actor capacity must cover at least one actor per Scheduler slot: "
                f"demand={minimum_actor_demand}, capacity={actor_capacity}"
            )
        return self


class YAMLValidationRequest(StrictModel):
    yaml_content: str = Field(min_length=1)


class BenchmarkRunnerSpec(StrictModel):
    """One persistent Runner in a single or Campaign request."""

    benchmark: Optional[BenchmarkConfig] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    label: Optional[str] = Field(default=None, min_length=1, max_length=200)


class BenchmarkRunnerCreate(BenchmarkRunnerSpec):
    """Start one Runner, optionally attached to a Campaign."""

    version: ProtocolVersion = PROTOCOL_VERSION
    campaign_id: Optional[str] = Field(default=None, min_length=1, max_length=36)


class BenchmarkRunnerBatchCreate(StrictModel):
    """Atomically start multiple Runners in one Campaign."""

    version: ProtocolVersion = PROTOCOL_VERSION
    runners: List[BenchmarkRunnerSpec] = Field(min_length=1, max_length=100)


class DimensionReference(StrictModel):
    dimension: str = Field(min_length=1, max_length=64)


TaskValue = Union[int, DimensionReference]


class TaskPayloadSpec(StrictModel):
    seed_namespace: str = Field(min_length=1, max_length=64)


class InvokeStep(StrictModel):
    kind: Literal["invoke"] = "invoke"
    id: str = Field(min_length=1, max_length=64)
    role: str = Field(min_length=1, max_length=64)
    payload: str = Field(min_length=1, max_length=64)
    after_seconds: TaskValue = 0


class RepeatStep(StrictModel):
    kind: Literal["repeat"] = "repeat"
    id: str = Field(min_length=1, max_length=64)
    count: TaskValue
    interval_seconds: TaskValue = 0
    invoke: InvokeStep


class ParallelStep(StrictModel):
    kind: Literal["parallel"] = "parallel"
    after_seconds: TaskValue = 0
    invokes: List[InvokeStep] = Field(min_length=1, max_length=100)


TaskStep = Annotated[
    Union[InvokeStep, RepeatStep, ParallelStep], Field(discriminator="kind")
]


class TaskDefinitionCreate(StrictModel):
    """A finite compile-time recipe that expands into atomic invoke nodes."""

    name: str = Field(min_length=1, max_length=200)
    matrix: Dict[str, List[int]] = Field(default_factory=dict)
    trials: int = Field(default=1, ge=1, le=1_000)
    seed: int = Field(default=11111, ge=0, le=2_147_483_647)
    payloads: Dict[str, TaskPayloadSpec] = Field(min_length=1, max_length=100)
    sequence: List[TaskStep] = Field(min_length=1, max_length=100)
    runner: BenchmarkRunnerSpec

    @model_validator(mode="after")
    def validate_graph(self) -> "TaskDefinitionCreate":
        for name, values in self.matrix.items():
            if not name or len(name) > 64:
                raise ValueError("task matrix dimension names must contain 1-64 chars")
            if not values or len(values) > 100:
                raise ValueError("task matrix dimensions require 1-100 values")
            if len(set(values)) != len(values):
                raise ValueError("task matrix dimension values must be unique")
        step_ids: List[str] = []
        for step in self.sequence:
            if isinstance(step, InvokeStep):
                invokes = [step]
                step_ids.append(step.id)
            elif isinstance(step, RepeatStep):
                invokes = [step.invoke]
                step_ids.extend([step.id, step.invoke.id])
            else:
                invokes = step.invokes
                step_ids.extend(item.id for item in invokes)
            for invoke in invokes:
                if invoke.payload not in self.payloads:
                    raise ValueError(
                        f"task invoke {invoke.id} references unknown payload "
                        f"{invoke.payload}"
                    )
            values_to_check = (
                [step.after_seconds]
                if isinstance(step, InvokeStep)
                else (
                    [
                        step.count,
                        step.interval_seconds,
                        step.invoke.after_seconds,
                    ]
                    if isinstance(step, RepeatStep)
                    else [
                        step.after_seconds,
                        *(invoke.after_seconds for invoke in step.invokes),
                    ]
                )
            )
            for value in values_to_check:
                if (
                    isinstance(value, DimensionReference)
                    and value.dimension not in self.matrix
                ):
                    raise ValueError(
                        f"task expression references unknown dimension {value.dimension}"
                    )
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("task step IDs must be unique")
        combinations = 1
        for values in self.matrix.values():
            combinations *= len(values)
        if combinations * self.trials > 10_000:
            raise ValueError("task definition cannot exceed 10000 instances")

        dimension_names = sorted(self.matrix)
        matrix_values = (
            product(*(self.matrix[name] for name in dimension_names))
            if dimension_names
            else [()]
        )
        for coordinate in matrix_values:
            dimensions = dict(zip(dimension_names, coordinate))

            def resolve(value: TaskValue) -> int:
                if isinstance(value, DimensionReference):
                    return int(dimensions[value.dimension])
                return int(value)

            planned_span = 0
            for step in self.sequence:
                if isinstance(step, InvokeStep):
                    delay = resolve(step.after_seconds)
                elif isinstance(step, RepeatStep):
                    count = resolve(step.count)
                    interval = resolve(step.interval_seconds)
                    node_delay = resolve(step.invoke.after_seconds)
                    if not 0 <= count <= 100:
                        raise ValueError(
                            "expanded repeat count must be between 0 and 100"
                        )
                    if interval < 0 or node_delay < 0:
                        raise ValueError("task delays cannot be negative")
                    delay = count * (interval + node_delay)
                else:
                    parallel_delay = resolve(step.after_seconds)
                    child_delays = [
                        resolve(item.after_seconds) for item in step.invokes
                    ]
                    if parallel_delay < 0 or any(item < 0 for item in child_delays):
                        raise ValueError("task delays cannot be negative")
                    delay = parallel_delay + max(child_delays)
                if delay < 0:
                    raise ValueError("task delays cannot be negative")
                planned_span += delay
                if planned_span > 21_600:
                    raise ValueError("task planned span cannot exceed six hours")
        if self.runner.benchmark is not None:
            if self.runner.benchmark.cache_probe is not None:
                raise ValueError("compiled task runner cannot define cache_probe")
        return self


class RunnerPlanRecurrence(StrictModel):
    """Bounded interval or geographic-calendar recurrence."""

    kind: Literal["interval", "calendar"]
    every_seconds: Optional[int] = Field(default=None, ge=1, le=31_536_000)
    frequency: Optional[Literal["daily", "weekly"]] = None
    interval: Optional[int] = Field(default=None, ge=1, le=366)
    local_time: Optional[time] = None
    weekdays: Optional[
        List[Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]]
    ] = None

    @model_validator(mode="after")
    def validate_shape(self) -> "RunnerPlanRecurrence":
        if self.kind == "interval":
            if self.every_seconds is None:
                raise ValueError("interval recurrence requires every_seconds")
            if (
                self.frequency is not None
                or self.interval is not None
                or self.local_time is not None
                or self.weekdays
            ):
                raise ValueError("interval recurrence cannot contain calendar fields")
            return self
        if self.every_seconds is not None:
            raise ValueError("calendar recurrence cannot contain every_seconds")
        if self.frequency is None or self.local_time is None:
            raise ValueError("calendar recurrence requires frequency and local_time")
        if self.frequency == "weekly" and not self.weekdays:
            raise ValueError("weekly recurrence requires weekdays")
        if self.frequency == "daily" and self.weekdays:
            raise ValueError("daily recurrence cannot contain weekdays")
        return self


class RunnerPlanTiming(StrictModel):
    """Geographic recurrence and mandatory execution boundary."""

    timezone: str = Field(min_length=1, max_length=64)
    starts_at: Optional[datetime] = None
    ends_at: Optional[datetime] = None
    max_occurrences: Optional[int] = Field(default=None, ge=1, le=100_000)
    recurrence: RunnerPlanRecurrence
    overlap_policy: Literal["queue", "skip"] = "queue"
    misfire_grace_seconds: int = Field(default=60, ge=0, le=86_400)

    @model_validator(mode="after")
    def validate_timing(self) -> "RunnerPlanTiming":
        if self.starts_at is not None and self.starts_at.tzinfo is None:
            raise ValueError("starts_at must include a UTC offset")
        if self.ends_at is not None and self.ends_at.tzinfo is None:
            raise ValueError("ends_at must include a UTC offset")
        if self.ends_at is None and self.max_occurrences is None:
            raise ValueError("runner plan requires ends_at or max_occurrences")
        if self.starts_at is None and self.misfire_grace_seconds == 0:
            raise ValueError(
                "an immediate runner plan requires misfire_grace_seconds greater "
                "than zero"
            )
        if (
            self.starts_at is not None
            and self.ends_at is not None
            and self.ends_at <= self.starts_at
        ):
            raise ValueError("ends_at must be later than starts_at")
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown IANA timezone: {self.timezone}") from exc
        return self


class RunnerPlanPreview(RunnerPlanTiming):
    count: int = Field(default=10, ge=1, le=100)


class RunnerPlanCreate(RunnerPlanTiming):
    name: str = Field(min_length=1, max_length=200)
    runner: BenchmarkRunnerSpec


class BenchmarkCampaignCreate(StrictModel):
    """Create a durable grouping for multiple benchmark Runners."""

    name: str = Field(min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=2000)
    tags: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkCampaignStart(StrictModel):
    """Atomically create one Campaign and its initial workload."""

    version: ProtocolVersion = PROTOCOL_VERSION
    campaign: BenchmarkCampaignCreate
    runners: List[BenchmarkRunnerSpec] = Field(default_factory=list, max_length=100)
    runner_plans: List[RunnerPlanCreate] = Field(default_factory=list, max_length=100)
    task_definitions: List[TaskDefinitionCreate] = Field(
        default_factory=list, max_length=20
    )

    @model_validator(mode="after")
    def validate_workload(self) -> "BenchmarkCampaignStart":
        workload_size = (
            len(self.runners) + len(self.runner_plans) + len(self.task_definitions)
        )
        if workload_size == 0:
            raise ValueError(
                "campaign requires runners, runner_plans, or task_definitions"
            )
        if workload_size > 100:
            raise ValueError("campaign workload cannot contain more than 100 items")
        instance_count = 0
        for definition in self.task_definitions:
            combinations = 1
            for values in definition.matrix.values():
                combinations *= len(values)
            instance_count += combinations * definition.trials
        if instance_count > 10_000:
            raise ValueError("campaign tasks cannot exceed 10000 total instances")
        return self


class TrustedClientWrite(StrictModel):
    """Superuser-managed user profile and trusted CLI public key."""

    public_key: str = Field(min_length=1)
    role: Literal["viewer", "operator", "superuser"] = "operator"
    display_name: Optional[str] = Field(default=None, max_length=200)
    email: Optional[str] = Field(default=None, max_length=320)


def validate_app_config(data: Dict[str, Any]) -> AppConfig:
    """Validate one application configuration document."""

    return AppConfig.model_validate(data)


def dump_model(model: BaseModel) -> Dict[str, Any]:
    """Serialize one validated request model to JSON-compatible values."""

    return model.model_dump(mode="json")


def app_config_schema() -> Dict[str, Any]:
    """Return the JSON Schema for the YAML root model."""

    return AppConfig.model_json_schema()
