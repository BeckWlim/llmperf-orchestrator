"""Shared duration syntax: numbers mean seconds; objects carry explicit units."""

import math
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

DurationScalar = TypeVar("DurationScalar")


class DurationValue(BaseModel, Generic[DurationScalar]):
    model_config = ConfigDict(extra="forbid")
    value: DurationScalar
    unit: Literal["ms", "s", "m", "h", "d"] = "s"

    @model_validator(mode="before")
    @classmethod
    def normalize_scalar(cls, raw_value: object) -> object:
        if isinstance(raw_value, bool):
            raise ValueError("duration cannot be a boolean")
        if isinstance(raw_value, (int, float, str)):
            return {"value": raw_value, "unit": "s"}
        return raw_value

    @field_validator("value", mode="before")
    @classmethod
    def reject_boolean(cls, raw_value: object) -> object:
        if isinstance(raw_value, bool):
            raise ValueError("duration cannot be a boolean")
        return raw_value

    @property
    def unit_seconds(self) -> float:
        return {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}[self.unit]


class Duration(DurationValue[float]):
    @model_validator(mode="after")
    def validate_value(self) -> "Duration":
        if not math.isfinite(self.seconds) or self.value < 0:
            raise ValueError("duration must be finite and nonnegative")
        return self

    @property
    def seconds(self) -> float:
        return self.value * self.unit_seconds


def duration_seconds(raw_value: object) -> float:
    return Duration.model_validate(raw_value).seconds
