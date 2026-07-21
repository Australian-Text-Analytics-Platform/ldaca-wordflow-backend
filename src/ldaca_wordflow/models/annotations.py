"""Strict contracts for stateless annotation previews and provider discovery."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StringConstraints,
    model_validator,
)

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
AnnotationProvider = Literal["openai", "openrouter", "anthropic", "google"]


class AnnotationClass(BaseModel):
    """One exact label and optional model-facing description."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: NonEmptyText = Field(max_length=200)
    description: str = Field(default="", max_length=2_000)


class AnnotationConfig(BaseModel):
    """Persistable provider-independent annotation configuration."""

    model_config = ConfigDict(extra="forbid")

    text_column: NonEmptyText = Field(max_length=500)
    annotation_column: NonEmptyText = Field(max_length=500)
    classes: list[AnnotationClass] = Field(min_length=1, max_length=200)
    provider: AnnotationProvider
    model: NonEmptyText = Field(max_length=500)
    instruction: NonEmptyText = Field(max_length=20_000)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    reasoning_enabled: bool = False
    reasoning_effort: Literal["low", "medium", "high"] = "medium"

    @model_validator(mode="after")
    def unique_classes(self) -> "AnnotationConfig":
        """Reject ambiguous labels before any provider request is made."""

        normalized = [item.name.casefold() for item in self.classes]
        if len(normalized) != len(set(normalized)):
            raise ValueError("annotation class names must be unique")
        return self


class AnnotationPreviewRequest(AnnotationConfig):
    """One stateless, one-based page preview with a request-only credential."""

    api_key: SecretStr | None = Field(
        default=None,
        min_length=1,
        max_length=4_000,
        json_schema_extra={"writeOnly": True},
    )
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=200)


class AnnotationPreviewLabel(BaseModel):
    """A provider label paired with its stable zero-based source row index."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_index: int = Field(ge=0)
    label: str | None


class AnnotationPreviewResource(BaseModel):
    """Direct stateless preview result for one source-node page."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    total_rows: int = Field(ge=0)
    labels: list[AnnotationPreviewLabel]


class AnnotationModelsRequest(BaseModel):
    """Optional request-only credential for provider model discovery."""

    model_config = ConfigDict(extra="forbid")

    api_key: SecretStr | None = Field(
        default=None,
        min_length=1,
        max_length=4_000,
        json_schema_extra={"writeOnly": True},
    )


class AnnotationModelsResource(BaseModel):
    """Sorted model identifiers returned by one configured provider."""

    model_config = ConfigDict(extra="forbid")

    provider: AnnotationProvider
    models: list[str]


__all__ = [
    "AnnotationClass",
    "AnnotationModelsRequest",
    "AnnotationModelsResource",
    "AnnotationPreviewRequest",
    "AnnotationPreviewResource",
    "AnnotationProvider",
]
