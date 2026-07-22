"""Shared annotation values used by durable and stateless requests."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

AnnotationProvider = Literal["openai", "openrouter", "anthropic", "google"]

AnnotationClassName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class AnnotationClass(BaseModel):
    """One exact label and optional model-facing description."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: AnnotationClassName = Field(max_length=200)
    description: str = Field(default="", max_length=2_000)


__all__ = ["AnnotationClass", "AnnotationProvider"]
