"""Typed normalization for provider token and prompt-cache accounting."""

from dataclasses import asdict, dataclass
import json
from typing import Any, Dict, Mapping, Optional

from llmperf import common_metrics


RAW_USAGE_LIMIT = 16_384


def _counter(value: Any) -> Optional[int]:
    """Return a valid non-negative integer counter, otherwise ``None``."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0 or int(value) != value:
        return None
    return int(value)


def _bounded_raw_usage(usage: Mapping[str, Any]) -> Dict[str, Any]:
    """Retain auditable usage without allowing an unbounded result document."""

    try:
        encoded = json.dumps(usage, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return {"truncated": True, "value": str(usage)[:RAW_USAGE_LIMIT]}
    if len(encoded.encode("utf-8")) <= RAW_USAGE_LIMIT:
        return dict(usage)
    return {
        "truncated": True,
        "size_bytes": len(encoded.encode("utf-8")),
        "preview": encoded[: RAW_USAGE_LIMIT // 2],
    }


@dataclass(frozen=True)
class NormalizedUsage:
    provider_input_tokens: Optional[int]
    provider_output_tokens: Optional[int]
    hit_tokens: Optional[int]
    miss_tokens: Optional[int]
    creation_tokens: Optional[int]
    source_schema: str
    complete: bool
    valid: bool
    validation_error: Optional[str]
    raw_usage: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            common_metrics.NORMALIZED_USAGE: self.to_dict(),
            common_metrics.RAW_USAGE: self.raw_usage,
        }
        if self.provider_input_tokens is not None:
            metrics[common_metrics.PROVIDER_INPUT_TOKENS] = self.provider_input_tokens
        if self.provider_output_tokens is not None:
            metrics[common_metrics.PROVIDER_OUTPUT_TOKENS] = self.provider_output_tokens
        if self.hit_tokens is not None:
            metrics[common_metrics.KV_CACHE_HIT_TOKENS] = self.hit_tokens
        if self.miss_tokens is not None:
            metrics[common_metrics.KV_CACHE_MISS_TOKENS] = self.miss_tokens
        if self.creation_tokens is not None:
            metrics[common_metrics.KV_CACHE_CREATION_TOKENS] = self.creation_tokens
        if self.hit_tokens is not None and self.miss_tokens is not None:
            denominator = self.hit_tokens + self.miss_tokens
            metrics[common_metrics.KV_CACHE_HIT_RATE] = (
                self.hit_tokens / denominator if denominator else None
            )
        return metrics


def normalize_usage(usage: Mapping[str, Any]) -> NormalizedUsage:
    """Normalize known OpenAI-compatible cache accounting schemas.

    Unknown or invalid counters are retained in ``raw_usage`` but never guessed.
    A miss count is derived from total input only for schemas where cached input is
    documented as a partition of input tokens.
    """

    raw = _bounded_raw_usage(usage)
    invalid_counter_names = []

    def checked(container: Mapping[str, Any], name: str, path: str) -> Optional[int]:
        value = container.get(name)
        counter = _counter(value)
        if value is not None and counter is None:
            invalid_counter_names.append(path)
        return counter

    provider_input = checked(usage, "prompt_tokens", "prompt_tokens")
    if provider_input is None:
        provider_input = checked(usage, "input_tokens", "input_tokens")
    provider_output = checked(usage, "completion_tokens", "completion_tokens")
    if provider_output is None:
        provider_output = checked(usage, "output_tokens", "output_tokens")

    hit = checked(usage, "prompt_cache_hit_tokens", "prompt_cache_hit_tokens")
    miss = checked(usage, "prompt_cache_miss_tokens", "prompt_cache_miss_tokens")
    creation = checked(
        usage, "prompt_cache_creation_tokens", "prompt_cache_creation_tokens"
    )
    if creation is None:
        creation = checked(
            usage, "cache_creation_input_tokens", "cache_creation_input_tokens"
        )
    schema = "deepseek_prompt_cache" if hit is not None else "unknown"

    if hit is None:
        details = usage.get("prompt_tokens_details")
        if isinstance(details, Mapping):
            hit = checked(
                details,
                "cached_tokens",
                "prompt_tokens_details.cached_tokens",
            )
            creation = (
                creation
                if creation is not None
                else checked(
                    details,
                    "cache_creation_tokens",
                    "prompt_tokens_details.cache_creation_tokens",
                )
            )
            if hit is not None:
                schema = "openai_prompt_tokens_details"
        if hit is None:
            details = usage.get("input_tokens_details")
            if isinstance(details, Mapping):
                hit = checked(
                    details,
                    "cached_tokens",
                    "input_tokens_details.cached_tokens",
                )
                creation = (
                    creation
                    if creation is not None
                    else checked(
                        details,
                        "cache_creation_tokens",
                        "input_tokens_details.cache_creation_tokens",
                    )
                )
                if hit is not None:
                    schema = "openai_input_tokens_details"

    if hit is not None and miss is None and provider_input is not None:
        if hit <= provider_input:
            miss = provider_input - hit

    validation_error = (
        "invalid non-negative integer counters: " + ", ".join(invalid_counter_names)
        if invalid_counter_names
        else None
    )
    if (
        validation_error is None
        and hit is not None
        and provider_input is not None
        and hit > provider_input
    ):
        validation_error = "cache hit tokens exceed provider input tokens"
    elif validation_error is None and (
        hit is not None
        and miss is not None
        and provider_input is not None
        and hit + miss > provider_input
    ):
        validation_error = "cache hit plus miss tokens exceed provider input tokens"

    return NormalizedUsage(
        provider_input_tokens=provider_input,
        provider_output_tokens=provider_output,
        hit_tokens=hit,
        miss_tokens=miss,
        creation_tokens=creation,
        source_schema=schema,
        complete=hit is not None and miss is not None and validation_error is None,
        valid=validation_error is None,
        validation_error=validation_error,
        raw_usage=raw,
    )
