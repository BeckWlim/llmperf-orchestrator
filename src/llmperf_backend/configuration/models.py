"""Validated request and configuration models used by the backend."""

from datetime import datetime, time, timezone
import hashlib
from math import prod
import re
import secrets
from typing import Annotated, Any, Dict, List, Literal, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    WithJsonSchema,
    field_validator,
    model_validator,
)
from llmperf.configuration.duration import Duration
from llmperf_backend.compiler.requests import (
    RequestPlan,
    RequestStepSpec as RequestStepSpec,
)

from llmperf.datasets.prompts import (
    validate_external_dataset_adapter,
)
from llmperf.version import PROTOCOL_VERSION, ProtocolVersion
from llmperf_backend.compiler.tasks import (
    TaskDefinitionCreate as CompilerTaskDefinitionCreate,
)

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
    delay: Duration = Field(default_factory=lambda: Duration(value=0))
    persist_prompt_text: bool = False
    allow_approximate_tokenizer: bool = False
    tokenizer_divergence_warning_ratio: float = Field(default=0.05, ge=0, le=1)
    bootstrap_samples: int = Field(default=2_000, ge=100, le=100_000)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1)
    minimum_counter_coverage: float = Field(default=0.8, ge=0, le=1)

    @field_validator("delay")
    @classmethod
    def validate_delay(cls, duration: Duration) -> Duration:
        if duration.seconds > 60:
            raise ValueError("cache probe delay must not exceed 60 seconds")
        return duration


class TaskExecutionContext(StrictModel):
    """Compiler-owned identity shared by atomic and batch executions."""

    definition_id: str = Field(min_length=1, max_length=36)
    instance_id: str = Field(min_length=1, max_length=36)
    node_id: str = Field(min_length=1, max_length=120)
    role: str = Field(min_length=1, max_length=64)
    trial_index: int = Field(ge=0)
    dimensions: Dict[str, int] = Field(default_factory=dict)


class CompiledTaskContext(TaskExecutionContext):
    execution_kind: Literal["atomic"] = "atomic"
    payload_id: str = Field(min_length=1, max_length=64)
    payload_seed: int = Field(ge=0, le=2_147_483_647)
    prompt_text: Optional[str] = None
    message_sources: List[str] = Field(default_factory=list)


class BatchTaskContext(TaskExecutionContext):
    execution_kind: Literal["batch"] = "batch"
    dataset_seed: int = Field(ge=0, le=18_446_744_073_709_551_615)
    seed_version: Literal["batch-v1", "family-suffix-v1"] = "batch-v1"
    shared_prefix_seed: Optional[int] = Field(default=None, ge=0, le=2**64 - 1)
    prefix_seed_version: Optional[Literal["family-prefix-v1"]] = None


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


class RunnerExecutionSettings(StrictModel):
    """Runner lifecycle settings, independent of the Worker request template."""

    timeout: Duration = Field(default_factory=lambda: Duration(value=90))
    concurrent_requests: Optional[int] = Field(default=None, gt=0)

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, duration: Duration) -> Duration:
        if duration.seconds <= 0:
            raise ValueError("timeout must be positive")
        return duration

    @field_validator("concurrent_requests", mode="before")
    @classmethod
    def validate_optional_window(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() in {"", "null"}:
            return None
        return value


class RunnerGenerationSettings(StrictModel):
    """Generation and measurement settings inherited by every Worker."""

    mean_output_tokens: int = Field(default=150, gt=0)
    stddev_output_tokens: int = Field(default=80, ge=0)
    output_token_count: Literal["provider", "local"] = "provider"
    additional_sampling_params: Dict[str, Any] = Field(default_factory=dict)


class InputTokens(StrictModel):
    """Generation target for new input only, excluding reused messages."""

    mean: int = Field(default=550, gt=0)
    stddev: int = Field(default=150, ge=0)

    @model_validator(mode="before")
    @classmethod
    def parse_count(cls, value: object) -> object:
        return (
            {"mean": value, "stddev": 0}
            if isinstance(value, int) and not isinstance(value, bool)
            else value
        )


class InputTemplateConfig(StrictModel):
    """Runner-owned input data and generation settings."""

    text: Optional[str] = Field(default=None, min_length=1)
    tokens: InputTokens = Field(default_factory=InputTokens)
    shared_prefix_tokens: int = Field(default=0, ge=0)
    dataset: Optional[DatasetSpec] = None
    dataset_prompt_mode: Literal["sample", "concatenate"] = "sample"
    dataset_repeat_count: int = Field(default=1, ge=1)
    dataset_seed: int = 11111

    @model_validator(mode="after")
    def validate_source(self) -> "InputTemplateConfig":
        if self.text is not None and (
            self.dataset is not None or self.shared_prefix_tokens
        ):
            raise ValueError(
                "literal input text cannot combine with a dataset or generated prefix"
            )
        if self.shared_prefix_tokens >= self.tokens.mean:
            raise ValueError(
                "shared_prefix_tokens must be smaller than the new-input token target"
            )
        if self.dataset is None and self.dataset_prompt_mode != "sample":
            raise ValueError("dataset_prompt_mode requires dataset")
        if (
            self.dataset is not None
            and self.shared_prefix_tokens
            and self.dataset_prompt_mode != "concatenate"
        ):
            raise ValueError("dataset shared_prefix_tokens requires concatenate mode")
        if self.dataset_prompt_mode == "concatenate" and self.dataset_repeat_count != 1:
            raise ValueError(
                "concatenated dataset prompts require dataset_repeat_count=1"
            )
        return self

    def execution_fields(self) -> Dict[str, Any]:
        document = self.model_dump(mode="json", exclude={"tokens", "text"})
        return {
            **document,
            "prompt": self.text,
            "mean_input_tokens": self.tokens.mean,
            "stddev_input_tokens": self.tokens.stddev,
        }


class _RequestFields(RequestPlan):
    """Input construction and finite Worker expansion."""

    prompt: Optional[str] = Field(default=None, min_length=1)
    mean_input_tokens: int = Field(default=550, gt=0)
    stddev_input_tokens: int = Field(default=150, ge=0)
    shared_prefix_tokens: int = Field(default=0, ge=0)
    shared_prefix_seed: Optional[int] = Field(default=None, ge=0, le=2**64 - 1)
    dataset: Optional[DatasetSpec] = None
    dataset_prompt_mode: Literal["sample", "concatenate"] = "sample"
    dataset_repeat_count: int = Field(default=1, ge=1)
    dataset_seed: int = 11111
    cache_probe: Optional[CacheProbeConfig] = None
    inputs: Dict[str, InputTemplateConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_cache_probe(self) -> "_RequestFields":
        required_inputs = (
            {step.input or self.input for step in self.steps}
            if self.steps
            else {self.input}
        )
        available_inputs = self.inputs.keys() if self.inputs else {"default"}
        if not required_inputs <= available_inputs:
            raise ValueError("request references an unknown Runner input")
        if self.cache_probe is not None and (
            self.steps is not None or self.repeat != 1 or self.prompt is not None
        ):
            raise ValueError(
                "cache_probe defines its own requests; it cannot combine with repeat, steps, or prompt"
            )
        if (
            self.dataset is not None
            and self.shared_prefix_tokens
            and self.dataset_prompt_mode != "concatenate"
        ):
            raise ValueError("dataset shared_prefix_tokens requires concatenate mode")
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


class RequestTemplateConfig(RequestPlan):
    """Execution logic and message composition; all input data belongs to Runner."""

    cache_probe: Optional[CacheProbeConfig] = None

    @model_validator(mode="after")
    def validate_probe(self) -> "RequestTemplateConfig":
        if self.cache_probe is not None and (
            self.steps is not None or self.repeat != 1
        ):
            raise ValueError("cache_probe cannot combine with steps or repeat")
        return self


class BenchmarkConfig(
    _RequestFields, RunnerExecutionSettings, RunnerGenerationSettings
):
    """Normalized execution/default settings; not the public Runner authoring shape."""

    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1)
    tokenizer: TokenizerSpec = Field(
        default_factory=lambda: TokenizerSpec(id=DEFAULT_TOKENIZER_ID)
    )


class ResolvedBenchmarkConfig(
    _RequestFields, RunnerExecutionSettings, RunnerGenerationSettings
):
    """Backend-owned execution fields added after Provider and artifact resolution."""

    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1)
    adapter: Literal["openai", "anthropic", "litellm", "sagemaker", "vertexai"]
    tokenizer: ResolvedTokenizerSpec

    @model_validator(mode="after")
    def validate_message_adapter(self) -> "ResolvedBenchmarkConfig":
        if (
            any(step.message_sources for step in self.steps or ())
            and "messages" in self.additional_sampling_params
        ):
            raise ValueError("sampling messages cannot override reused messages")
        if self.adapter == "vertexai" and any(
            step.message_sources for step in self.steps or ()
        ):
            raise ValueError("vertexai does not support structured message reuse")
        return self


class CompiledBenchmarkConfig(ResolvedBenchmarkConfig):
    """Internal one-request benchmark emitted only by the task compiler."""

    task_context: CompiledTaskContext

    @model_validator(mode="after")
    def validate_compiled_shape(self) -> "CompiledBenchmarkConfig":
        if self.steps is not None:
            raise ValueError(
                "compiled invoke cannot contain an internal request workflow"
            )
        if self.cache_probe is not None:
            raise ValueError("compiled benchmark cannot define cache_probe")
        if self.worker_count != 1 or self.concurrent_requests != 1:
            raise ValueError("compiled benchmark must contain exactly one request")
        return self


class BatchBenchmarkConfig(ResolvedBenchmarkConfig):
    """Compiler-owned batch execution with the declared Worker template."""

    task_context: BatchTaskContext

    @model_validator(mode="after")
    def validate_batch(self) -> "BatchBenchmarkConfig":
        if self.cache_probe is not None:
            raise ValueError("compiled batch cannot define cache_probe")
        if self.dataset_seed != self.task_context.dataset_seed:
            raise ValueError("batch dataset seed must match task context")
        if self.shared_prefix_seed != self.task_context.shared_prefix_seed:
            raise ValueError("batch prefix seed must match task context")
        return self


class GroupedBenchmarkConfig(ResolvedBenchmarkConfig):
    """Compiler-owned parallel invocations sharing one Runner's request pool."""

    task_contexts: List[CompiledTaskContext] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_group(self) -> "GroupedBenchmarkConfig":
        count = len(self.task_contexts)
        if (
            self.cache_probe is not None
            or self.worker_count != count
            or self.concurrent_requests != count
        ):
            raise ValueError(
                "grouped benchmark requires one concurrent request per task context"
            )
        if len({context.instance_id for context in self.task_contexts}) != 1:
            raise ValueError("grouped benchmark cannot cross task instances")
        if len({context.node_id for context in self.task_contexts}) != count:
            raise ValueError("grouped task node IDs must be unique")
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
    max_concurrent_runners: int = Field(default=1024, ge=1, le=16384)
    preparation_threads: int = Field(default=8, ge=1, le=64)
    poll_interval_seconds: float = Field(default=1.0, gt=0)
    cancel_grace_seconds: float = Field(default=5.0, gt=0)
    stale_after_seconds: int = Field(default=300, gt=0)
    log_bytes_limit: int = Field(default=1_000_000, ge=1024)
    ray_address: Optional[str] = Field(default=None, min_length=1)
    ray_num_cpus: int = Field(default=8, ge=1, le=1024)
    ray_actor_num_cpus: float = Field(default=1.0, gt=0, le=64)
    ray_actor_max_concurrency: int = Field(default=8, ge=1, le=1024)
    preparation_workers: int = Field(default=0, ge=0, le=64)
    preparation_chunk_size: int = Field(default=16, ge=1, le=256)
    preparation_buffer_chunks: int = Field(default=8, ge=1, le=64)
    ray_object_store_memory_bytes: int = Field(default=268_435_456, ge=78_643_200)
    ray_health_interval_seconds: float = Field(default=5.0, gt=0, le=300)
    ray_health_timeout_seconds: float = Field(default=3.0, gt=0, le=60)
    artifact_resolution_timeout_seconds: float = Field(default=60.0, gt=0, le=3600)

    @property
    def request_actor_capacity(self) -> int:
        return int(
            (self.ray_num_cpus - self.preparation_workers) / self.ray_actor_num_cpus
        )

    @field_validator("ray_address", mode="before")
    @classmethod
    def empty_ray_address(cls, value: Any) -> Any:
        return None if value == "" else value


class PerformanceGuardConfig(StrictModel):
    """Runtime protection, static request limits, and opt-in campaign budgets."""

    enabled: bool = True
    campaign_admission_enabled: bool = False
    max_runner_concurrency: int = Field(default=64, ge=1)
    max_effective_concurrency: int = Field(default=64, ge=1)
    max_campaign_runners: int = Field(default=1_000, ge=1)
    max_campaign_provider_requests: int = Field(default=10_000, ge=1)
    max_campaign_token_budget: int = Field(default=100_000_000, ge=1)
    warning_ratio: float = Field(default=0.8, gt=0, lt=1)
    max_host_memory_utilization: float = Field(default=0.90, gt=0, lt=1)
    resume_host_memory_utilization: float = Field(default=0.80, gt=0, lt=1)
    min_ray_object_store_available_ratio: float = Field(default=0.10, ge=0, lt=1)
    resume_ray_object_store_available_ratio: float = Field(default=0.20, gt=0, le=1)
    sample_interval_seconds: float = Field(default=1.0, gt=0, le=60)

    @property
    def workload_limits(self) -> Dict[str, int]:
        """Return thresholds for always-checked estimates and request limits."""

        return {
            "planned_runners": self.max_campaign_runners,
            "provider_requests": self.max_campaign_provider_requests,
            "token_budget": self.max_campaign_token_budget,
            "max_runner_concurrency": self.max_runner_concurrency,
            "effective_concurrency": self.max_effective_concurrency,
        }

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
        if self.scheduler.enabled and self.server.workers != 1:
            raise ValueError("Runtime coordinator requires server.workers=1")
        actor_capacity = self.scheduler.request_actor_capacity
        if self.scheduler.enabled and actor_capacity < 1:
            raise ValueError("Ray CPU budget must place at least one shared actor")
        return self


class YAMLValidationRequest(StrictModel):
    yaml_content: str = Field(min_length=1)


class BenchmarkRunnerSpec(RunnerExecutionSettings, RunnerGenerationSettings):
    """One persistent Runner in a single or Campaign request."""

    provider: Optional[str] = Field(default=None, min_length=1, max_length=64)
    model: Optional[str] = Field(default=None, min_length=1)
    tokenizer: Optional[TokenizerSpec] = None
    inputs: Optional[Dict[str, InputTemplateConfig]] = Field(
        default=None, min_length=1, max_length=100
    )
    request: Optional[RequestTemplateConfig] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    label: Optional[str] = Field(default=None, min_length=1, max_length=200)

    @model_validator(mode="after")
    def validate_message_input(self) -> "BenchmarkRunnerSpec":
        if self.inputs is not None:
            plan = self.request or RequestTemplateConfig()
            required_inputs = (
                {step.input or plan.input for step in plan.steps}
                if plan.steps
                else {plan.input}
            )
            if not required_inputs <= self.inputs.keys():
                raise ValueError("request references an unknown Runner input")
        if (
            self.request is not None
            and any(step.message_sources for step in self.request.steps or ())
            and "messages" in self.additional_sampling_params
        ):
            raise ValueError("sampling messages cannot override reused messages")
        return self

    def execution_benchmark(
        self, defaults: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Normalize public Worker templates into the persisted execution document."""
        template = dict(defaults or {})
        request_template = self.request or RequestTemplateConfig()
        template.update(request_template.model_dump(mode="json"))
        if self.inputs is not None:
            selected_input = template.get("input", "default")
            baseline_input = self.inputs.get(selected_input) or next(
                iter(self.inputs.values())
            )
            template.update(baseline_input.execution_fields())
            template["inputs"] = {
                name: value.model_dump(mode="json")
                for name, value in self.inputs.items()
            }
        else:
            # Backend input defaults remain internal; no input data is authored in request.
            default_input_fields = InputTemplateConfig().execution_fields()
            for input_field, input_default in default_input_fields.items():
                template[input_field] = (defaults or {}).get(input_field, input_default)
            if defaults is not None and defaults.get("inputs"):
                template["inputs"] = defaults["inputs"]
        for target_field in ("provider", "model", "tokenizer"):
            target_value = getattr(self, target_field)
            if target_value is not None:
                template[target_field] = (
                    target_value.model_dump(mode="json")
                    if isinstance(target_value, TokenizerSpec)
                    else target_value
                )
            elif defaults is not None and target_field in defaults:
                template[target_field] = defaults[target_field]
        for field_name in (
            "timeout",
            "concurrent_requests",
            *RunnerGenerationSettings.model_fields,
        ):
            if field_name in self.model_fields_set or field_name not in template:
                field_value = getattr(self, field_name)
                template[field_name] = (
                    field_value.model_dump(mode="json")
                    if isinstance(field_value, Duration)
                    else field_value
                )
        return template

    def execution_document(
        self, defaults: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return {
            "label": self.label,
            "metadata": dict(self.metadata),
            "benchmark": self.execution_benchmark(defaults),
        }

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        if "runner_round" in value:
            raise ValueError(
                "metadata.runner_round is reserved for Backend round expansion"
            )
        if "runner_group" in value:
            raise ValueError(
                "metadata.runner_group is reserved for Backend matrix expansion"
            )
        return value


TaskDefinitionCreate = CompilerTaskDefinitionCreate[BenchmarkRunnerSpec]


class BenchmarkRunnerCreate(BenchmarkRunnerSpec):
    """Start one Runner, optionally attached to a Campaign."""

    version: ProtocolVersion = PROTOCOL_VERSION
    campaign_id: Optional[str] = Field(default=None, min_length=1, max_length=36)


class BenchmarkRunnerBatchCreate(StrictModel):
    """Atomically start multiple Runners in one Campaign."""

    version: ProtocolVersion = PROTOCOL_VERSION
    runners: List[BenchmarkRunnerSpec] = Field(min_length=1, max_length=100)


class RunnerPlanRecurrence(StrictModel):
    """Bounded interval or geographic-calendar recurrence."""

    kind: Literal["interval", "calendar"]
    every: Optional[Duration] = None
    frequency: Optional[Literal["daily", "weekly"]] = None
    interval: Optional[int] = Field(default=None, ge=1, le=366)
    local_time: Optional[time] = None
    weekdays: Optional[
        List[Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]]
    ] = None

    @model_validator(mode="after")
    def validate_shape(self) -> "RunnerPlanRecurrence":
        if self.kind == "interval":
            if self.every is None:
                raise ValueError("interval recurrence requires every")
            if not 0.000001 <= self.every.seconds <= 31_536_000:
                raise ValueError(
                    "interval every must be between one microsecond and 365 days"
                )
            if (
                self.frequency is not None
                or self.interval is not None
                or self.local_time is not None
                or self.weekdays
            ):
                raise ValueError("interval recurrence cannot contain calendar fields")
            return self
        if self.every is not None:
            raise ValueError("calendar recurrence cannot contain every")
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


def dated_campaign_seed() -> int:
    """Fresh UTC-date-based seed, resolved once and persisted with every Runner."""
    material = f"{datetime.now(timezone.utc).isoformat()}:{secrets.token_hex(16)}"
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")


class RunnerRounds(StrictModel):
    """Repeat the campaign's Runner templates behind completion barriers."""

    count: int = Field(ge=1, le=1000)
    seed: Annotated[
        int,
        WithJsonSchema(
            {
                "anyOf": [
                    {"type": "integer", "minimum": 0},
                    {"type": "string", "const": "date"},
                ]
            },
            mode="validation",
        ),
    ] = Field(default_factory=dated_campaign_seed, ge=0)
    prefix_seed_scope: Literal["round", "family"] = "round"
    phases: List[str] = Field(
        default_factory=lambda: ["default"], min_length=1, max_length=16
    )

    @field_validator("seed", mode="before")
    @classmethod
    def resolve_seed(cls, raw_seed: object) -> object:
        return dated_campaign_seed() if raw_seed == "date" else raw_seed

    @field_validator("phases")
    @classmethod
    def validate_phases(cls, phases: List[str]) -> List[str]:
        if len(set(phases)) != len(phases) or any(
            re.fullmatch(r"[A-Za-z0-9_-]{1,64}", phase) is None for phase in phases
        ):
            raise ValueError(
                "round phases need unique names containing 1-64 letters, digits, _ or -"
            )
        return phases


class RunnerGroup(StrictModel):
    """Compact immediate Runner templates, expanded by the Backend before admission."""

    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    phase: Optional[str] = Field(default=None, min_length=1, max_length=64)
    matrix: Dict[str, List[StrictInt]] = Field(min_length=1, max_length=8)
    runner: BenchmarkRunnerSpec

    @field_validator("matrix")
    @classmethod
    def validate_matrix(cls, matrix: Dict[str, List[int]]) -> Dict[str, List[int]]:
        for name, values in matrix.items():
            if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) is None:
                raise ValueError(
                    "matrix dimension names need 1-64 letters, digits, _ or -"
                )
            if not values or len(values) > 100 or len(set(values)) != len(values):
                raise ValueError("matrix dimensions need 1-100 unique integer values")
        if prod(len(values) for values in matrix.values()) > 10000:
            raise ValueError("Runner group cannot exceed 10000 matrix coordinates")
        return matrix

    @property
    def count(self) -> int:
        return prod(len(values) for values in self.matrix.values())


class BenchmarkCampaignCreate(StrictModel):
    """Create a durable grouping for multiple benchmark Runners."""

    name: str = Field(min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=2000)
    tags: Dict[str, Any] = Field(default_factory=dict)
    rounds: Optional[RunnerRounds] = None


class BenchmarkCampaignStart(StrictModel):
    """Atomically create one Campaign and its initial workload."""

    version: ProtocolVersion = PROTOCOL_VERSION
    campaign: BenchmarkCampaignCreate
    runners: List[BenchmarkRunnerSpec] = Field(default_factory=list, max_length=100)
    runner_groups: List[RunnerGroup] = Field(default_factory=list, max_length=100)
    runner_plans: List[RunnerPlanCreate] = Field(default_factory=list, max_length=100)
    task_definitions: List[TaskDefinitionCreate] = Field(
        default_factory=list, max_length=20
    )

    @property
    def immediate_runner_count(self) -> int:
        return len(self.runners) + sum(group.count for group in self.runner_groups)

    @model_validator(mode="after")
    def validate_workload(self) -> "BenchmarkCampaignStart":
        workload_size = (
            len(self.runners)
            + len(self.runner_groups)
            + len(self.runner_plans)
            + len(self.task_definitions)
        )
        if workload_size == 0:
            raise ValueError(
                "campaign requires runners, runner_groups, runner_plans, or task_definitions"
            )
        if workload_size > 100:
            raise ValueError("campaign workload cannot contain more than 100 items")
        rounds = self.campaign.rounds
        phases = rounds.phases if rounds is not None else ["default"]
        group_keys = [
            (group.name, group.phase or phases[0]) for group in self.runner_groups
        ]
        if len(set(group_keys)) != len(group_keys):
            raise ValueError("Runner group names must be unique within a phase")
        for group in self.runner_groups:
            if group.phase is not None and (
                rounds is None or group.phase not in phases
            ):
                raise ValueError(
                    "Runner group phase must name a declared campaign round phase"
                )
        if self.immediate_runner_count * (rounds.count if rounds else 1) > 10000:
            raise ValueError(
                "campaign rounds/groups cannot exceed 10000 expanded runners"
            )
        if rounds is not None:
            if (
                not self.immediate_runner_count
                or self.runner_plans
                or self.task_definitions
            ):
                raise ValueError(
                    "campaign rounds require only immediate Runner templates"
                )
            occupied_phases = {group.phase or phases[0] for group in self.runner_groups}
            if self.runners:
                occupied_phases.add(phases[0])
            if occupied_phases != set(phases):
                raise ValueError(
                    "each round phase must contain at least one Runner template"
                )
            if rounds.prefix_seed_scope == "family" and any(
                runner.request is not None and runner.request.cache_probe is not None
                for runner in [
                    *self.runners,
                    *(group.runner for group in self.runner_groups),
                ]
            ):
                raise ValueError(
                    "family-scoped rounds use ordinary inputs, not cache_probe"
                )
        instance_count = 0
        for definition in self.task_definitions:
            instance_count += definition.instance_count
            if (
                definition.runner.request is not None
                and definition.runner.request.cache_probe is not None
            ):
                raise ValueError("compiled task runner cannot define cache_probe")
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
