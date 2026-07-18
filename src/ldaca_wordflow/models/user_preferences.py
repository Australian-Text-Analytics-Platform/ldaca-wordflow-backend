"""Strict account-level preference resources."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

PREFERENCES_SCHEMA_VERSION = 1


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UserPreferences(_StrictModel):
    """Non-secret preferences synchronized for one authenticated user."""

    hidden_views: list[str] = Field(default_factory=list)
    favorite_workspaces: list[str] = Field(default_factory=list)
    default_tokenizer_model: str | None = None
    analysis_multi_tab_enabled: bool = False
    contextual_hints_enabled: bool = True

    @field_validator("hidden_views", "favorite_workspaces")
    @classmethod
    def unique_non_empty_values(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            normalized = value.strip()
            if normalized and normalized not in cleaned:
                cleaned.append(normalized)
        return cleaned

    @field_validator("default_tokenizer_model")
    @classmethod
    def normalize_default_tokenizer(cls, value: str | None) -> str | None:
        normalized = value.strip() if value is not None else ""
        return normalized or None


class UserPreferencesPatch(_StrictModel):
    """Partial update; only explicitly provided fields are changed."""

    hidden_views: list[str] = Field(default_factory=list)
    favorite_workspaces: list[str] = Field(default_factory=list)
    default_tokenizer_model: str | None = None
    analysis_multi_tab_enabled: bool = False
    contextual_hints_enabled: bool = True

    @field_validator("hidden_views", "favorite_workspaces")
    @classmethod
    def unique_non_empty_values(cls, values: list[str]) -> list[str]:
        return UserPreferences.unique_non_empty_values(values)

    @field_validator("default_tokenizer_model")
    @classmethod
    def normalize_default_tokenizer(cls, value: str | None) -> str | None:
        return UserPreferences.normalize_default_tokenizer(value)


class StoredUserPreferences(UserPreferences):
    """Schema-versioned representation persisted to preferences.toml."""

    schema_version: Literal[1] = Field(default=PREFERENCES_SCHEMA_VERSION, frozen=True)


__all__ = [
    "PREFERENCES_SCHEMA_VERSION",
    "StoredUserPreferences",
    "UserPreferences",
    "UserPreferencesPatch",
]
