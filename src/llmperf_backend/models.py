"""Validated request and configuration models used by the backend."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


DEFAULT_TOKENIZER_ID = "hf-internal-testing/llama-tokenizer"


class StrictModel(BaseModel):
    """Base model that rejects misspelled or unsupported configuration keys."""

    class Config:
        extra = "forbid"


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
    use_fast: bool = True


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
    mean_output_tokens: int = Field(default=150, gt=0)
    stddev_output_tokens: int = Field(default=80, ge=0)
    additional_sampling_params: Dict[str, Any] = Field(default_factory=dict)
    tokenizer: TokenizerSpec = Field(
        default_factory=lambda: TokenizerSpec(id=DEFAULT_TOKENIZER_ID)
    )


class DatabaseConfig(StrictModel):
    """Asynchronous SQLAlchemy connection settings."""

    url: str = Field(default="postgresql+asyncpg:///llmperf", min_length=1)
    echo: bool = False
    auto_create_schema: bool = True
    pool_size: int = Field(default=5, ge=1)
    max_overflow: int = Field(default=10, ge=0)


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

    runners: List[BenchmarkRunnerSpec] = Field(min_items=1, max_items=100)


class BenchmarkCampaignCreate(StrictModel):
    """Create a durable grouping for multiple benchmark Runners."""

    name: str = Field(min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=2000)
    tags: Dict[str, Any] = Field(default_factory=dict)


class TrustedClientWrite(StrictModel):
    """Superuser-managed user profile and trusted CLI public key."""

    public_key: str = Field(min_length=1)
    role: Literal["viewer", "operator", "superuser"] = "operator"
    display_name: Optional[str] = Field(default=None, max_length=200)
    email: Optional[str] = Field(default=None, max_length=320)


def validate_app_config(data: Dict[str, Any]) -> AppConfig:
    """Validate with either Pydantic 1.x or 2.x."""

    model_validate = getattr(AppConfig, "model_validate", None)
    if model_validate is not None:
        return model_validate(data)
    return AppConfig.parse_obj(data)


def dump_model(model: BaseModel) -> Dict[str, Any]:
    """Serialize with either Pydantic 1.x or 2.x."""

    model_dump = getattr(model, "model_dump", None)
    if model_dump is not None:
        return model_dump(mode="json")
    return model.dict()


def app_config_schema() -> Dict[str, Any]:
    """Return the JSON Schema for the YAML root model."""

    model_json_schema = getattr(AppConfig, "model_json_schema", None)
    if model_json_schema is not None:
        return model_json_schema()
    return AppConfig.schema()
