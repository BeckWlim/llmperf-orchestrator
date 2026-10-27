"""Allow-listed diagnostic contracts shared by the Backend and CLI."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class WorkerSummary(BaseModel):
    worker_id: str = Field(min_length=1)
    step_id: str = Field(min_length=1)
    status: Literal[
        "created", "submitted", "succeeded", "failed", "cancelled", "indeterminate"
    ]
    created_at: Optional[datetime] = None
    submitted_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class WorkerPage(BaseModel):
    items: list[WorkerSummary]
    limit: int = Field(ge=1, le=200)
    offset: int = Field(ge=0)
    next_offset: Optional[int] = Field(default=None, ge=0)


class WorkerMessage(BaseModel):
    role: str
    content: str


class WorkerMetrics(BaseModel):
    end_to_end_latency_s: Optional[float] = None
    ttft_s: Optional[float] = None
    number_input_tokens: Optional[int] = None
    number_output_tokens: Optional[int] = None
    error_code: Optional[str | int] = None

    @field_validator("error_code")
    @classmethod
    def safe_error_code(cls, value: Optional[str | int]) -> Optional[str | int]:
        if isinstance(value, str) and (
            len(value) > 64 or not all(char.isalnum() or char in "_-" for char in value)
        ):
            return None
        return value


class WorkerLog(WorkerSummary):
    completion_available: bool
    input_messages: Optional[list[WorkerMessage]] = None
    output_text: Optional[str] = None
    metrics: Optional[WorkerMetrics] = None
    dependencies: list[str] = Field(default_factory=list)
    message_sources: list[str] = Field(default_factory=list)
    input_name: Optional[str] = None
    cancellation_reason: Optional[str] = None

    @model_validator(mode="after")
    def consistent_completion(self) -> "WorkerLog":
        fields = (self.input_messages, self.output_text, self.metrics)
        if self.completion_available and any(value is None for value in fields):
            raise ValueError(
                "Committed completion must include messages, text and metrics"
            )
        if not self.completion_available and any(value is not None for value in fields):
            raise ValueError("Unavailable completion cannot contain input/output")
        return self
