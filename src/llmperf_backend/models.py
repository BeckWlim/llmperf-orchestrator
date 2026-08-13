"""Validated request and configuration models used by the backend."""

from datetime import datetime, time
from typing import Any, Dict, List, Literal, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    """A server-resolved Hugging Face tokenizer selected by one benchmark."""

    source: Literal["huggingface"] = "huggingface"
    id: str = Field(min_length=1, max_length=200)
    revision: Optional[str] = Field(default=None, min_length=1, max_length=200)
    requested_revision: Optional[str] = Field(
        default=None, min_length=1, max_length=200
    )
    use_fast: bool = True
    immutable_revision: bool = False
    selection: Literal[
        "explicit", "model_binding", "provider_default", "global_default"
    ] = "global_default"
    accuracy: Literal["exact", "compatible", "approximate"] = "approximate"


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


class DatasetSpec(StrictModel):
    """A backend-resolved Hugging Face dataset artifact."""

    source: Literal["huggingface"] = "huggingface"
    id: str = Field(min_length=1, max_length=200)
    filename: str = Field(min_length=1, max_length=500)
    revision: Optional[str] = Field(default=None, min_length=1, max_length=200)
    format: Literal["sharegpt"] = "sharegpt"


class BenchmarkConfig(StrictModel):
    """Defaults matching the existing token benchmark command-line options."""

    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1)
    llm_api: Literal["openai", "anthropic", "litellm", "sagemaker", "vertexai"] = (
        "openai"
    )
    timeout_seconds: int = Field(default=90, gt=0)
    max_completed_requests: int = Field(default=10, gt=0)
    concurrent_requests: int = Field(default=1, gt=0)
    mean_input_tokens: int = Field(default=550, gt=0)
    stddev_input_tokens: int = Field(default=150, ge=0)
    shared_prefix_tokens: int = Field(default=0, ge=0)
    dataset: Optional[DatasetSpec] = None
    dataset_repeat_count: int = Field(default=1, ge=1)
    dataset_seed: int = 11111
    mean_output_tokens: int = Field(default=150, gt=0)
    stddev_output_tokens: int = Field(default=80, ge=0)
    additional_sampling_params: Dict[str, Any] = Field(default_factory=dict)
    tokenizer: TokenizerSpec = Field(
        default_factory=lambda: TokenizerSpec(id=DEFAULT_TOKENIZER_ID)
    )
    cache_probe: Optional[CacheProbeConfig] = None

    @model_validator(mode="before")
    @classmethod
    def mark_explicit_tokenizer(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        result = dict(data)
        tokenizer = result.get("tokenizer")
        if isinstance(tokenizer, dict):
            tokenizer = dict(tokenizer)
            tokenizer.setdefault("selection", "explicit")
            tokenizer.setdefault("accuracy", "compatible")
            result["tokenizer"] = tokenizer
        return result

    @model_validator(mode="after")
    def validate_cache_probe(self) -> "BenchmarkConfig":
        if self.cache_probe is None:
            return self
        if (
            self.tokenizer.accuracy == "approximate"
            and not self.cache_probe.allow_approximate_tokenizer
        ):
            raise ValueError(
                "cache_probe requires an explicit or model-bound tokenizer; set "
                "allow_approximate_tokenizer=true to acknowledge the global fallback"
            )
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
    working_directory: str = "."
    worker_module: str = "llmperf_backend.worker"
    log_bytes_limit: int = Field(default=1_000_000, ge=1024)


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
    version: int = Field(default=1, ge=1)
    environment: str = "development"
    server: ServerConfig = Field(default_factory=ServerConfig)
    benchmark: BenchmarkConfig
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)


class YAMLValidationRequest(StrictModel):
    yaml_content: str = Field(min_length=1)


class BenchmarkRunnerSpec(StrictModel):
    """One persistent Runner in a single or Campaign request."""

    benchmark: Optional[BenchmarkConfig] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    label: Optional[str] = Field(default=None, min_length=1, max_length=200)


class BenchmarkRunnerCreate(BenchmarkRunnerSpec):
    """Start one Runner, optionally attached to a Campaign."""

    campaign_id: Optional[str] = Field(default=None, min_length=1, max_length=36)


class BenchmarkRunnerBatchCreate(StrictModel):
    """Atomically start multiple Runners in one Campaign."""

    runners: List[BenchmarkRunnerSpec] = Field(min_length=1, max_length=100)


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

    campaign: BenchmarkCampaignCreate
    runners: List[BenchmarkRunnerSpec] = Field(default_factory=list, max_length=100)
    runner_plans: List[RunnerPlanCreate] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def validate_workload(self) -> "BenchmarkCampaignStart":
        workload_size = len(self.runners) + len(self.runner_plans)
        if workload_size == 0:
            raise ValueError("campaign requires runners or runner_plans")
        if workload_size > 100:
            raise ValueError("campaign workload cannot contain more than 100 items")
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
