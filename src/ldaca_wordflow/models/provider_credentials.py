"""Strict provider-credential resources and write-only update commands."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

AnnotationProvider = Literal["openai", "openrouter", "anthropic", "google"]
CredentialSource = Literal["none", "user", "deployment"]

CredentialValue = Annotated[
    str,
    Field(
        min_length=1,
        max_length=4_000,
        json_schema_extra={"writeOnly": True},
    ),
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProviderCredentialPatch(_StrictModel):
    """Write-only partial update; omitted fields remain unchanged."""

    openai_api_key: CredentialValue | None = None
    openrouter_api_key: CredentialValue | None = None
    anthropic_api_key: CredentialValue | None = None
    google_api_key: CredentialValue | None = None
    data_portal_api_token: CredentialValue | None = None


class AnnotationCredentialStatus(_StrictModel):
    openai: bool
    openrouter: bool
    anthropic: bool
    google: bool


class DataPortalCredentialStatus(_StrictModel):
    user_configured: bool
    deployment_configured: bool


class ProviderCredentialSummary(_StrictModel):
    """Safe credential presence information; never contains secret values."""

    annotation: AnnotationCredentialStatus
    data_portal: DataPortalCredentialStatus


class _StoredAnnotationCredentials(_StrictModel):
    openai: SecretStr | None = None
    openrouter: SecretStr | None = None
    anthropic: SecretStr | None = None
    google: SecretStr | None = None


class _StoredDataPortalCredentials(_StrictModel):
    api_token: SecretStr | None = None


class StoredProviderCredentials(_StrictModel):
    """Private representation persisted in the per-user TOML file."""

    annotation: _StoredAnnotationCredentials = Field(
        default_factory=_StoredAnnotationCredentials
    )
    data_portal: _StoredDataPortalCredentials = Field(
        default_factory=_StoredDataPortalCredentials
    )


__all__ = [
    "AnnotationProvider",
    "AnnotationCredentialStatus",
    "CredentialSource",
    "DataPortalCredentialStatus",
    "ProviderCredentialPatch",
    "ProviderCredentialSummary",
    "StoredProviderCredentials",
]
