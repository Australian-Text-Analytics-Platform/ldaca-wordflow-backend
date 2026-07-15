"""Canonical serialized tokenization metadata."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..shared.json_data import JsonData


class TokenizationMetadata(BaseModel):
    """Exact tokenization specification shared by persisted node DTOs."""

    model_config = ConfigDict(extra="forbid")

    column_name: str = Field(min_length=1, max_length=500)
    model: str = Field(min_length=1, max_length=500)
    language: str | None = Field(default=None, max_length=100)
    params: dict[str, JsonData] = Field(default_factory=dict, max_length=100)


__all__ = ["TokenizationMetadata"]
