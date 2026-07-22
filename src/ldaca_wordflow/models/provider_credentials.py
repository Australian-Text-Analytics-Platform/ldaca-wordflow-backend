"""Strict provider-credential resources and write-only update commands."""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StringConstraints,
    field_validator,
    model_validator,
)

from ..domain.annotation import (
    AnnotationProvider,
    normalize_annotation_provider_base_url,
)

CredentialStorage = Literal["backend", "browser"]

CredentialValue = Annotated[
    SecretStr,
    Field(
        min_length=1,
        max_length=4_000,
        json_schema_extra={"writeOnly": True},
    ),
]

ProviderConfigurationName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataPortalCredentialPatch(_StrictModel):
    """Write-only Data Portal credential update."""

    data_portal_api_token: CredentialValue | None = None


class _AnnotationProviderConfigurationFields(_StrictModel):
    name: ProviderConfigurationName
    provider: AnnotationProvider
    base_url: str | None = Field(default=None, max_length=2_000)

    @field_validator("base_url", mode="before")
    @classmethod
    def normalize_base_url(cls, value: object) -> object:
        if value is None or not isinstance(value, str):
            return value
        return normalize_annotation_provider_base_url(value)

    @model_validator(mode="after")
    def validate_locator(self) -> "_AnnotationProviderConfigurationFields":
        if self.provider == "custom" and self.base_url is None:
            raise ValueError("Custom providers require a base URL")
        if self.provider != "custom" and self.base_url is not None:
            raise ValueError("Built-in providers cannot define a base URL")
        return self


class AnnotationProviderConfigurationCreate(_AnnotationProviderConfigurationFields):
    """Create command containing one write-only provider credential."""

    api_key: CredentialValue | None = None

    @model_validator(mode="after")
    def require_builtin_credential(self) -> "AnnotationProviderConfigurationCreate":
        if self.provider != "custom" and self.api_key is None:
            raise ValueError("Built-in providers require an API key")
        return self


class AnnotationProviderConfigurationResource(_AnnotationProviderConfigurationFields):
    """Safe provider-configuration metadata returned to clients."""

    id: uuid.UUID
    has_api_key: bool


class AnnotationProviderConfigurationRename(_StrictModel):
    name: ProviderConfigurationName


class DataPortalCredentialStatus(_StrictModel):
    user_configured: bool | None
    deployment_configured: bool


class ProviderCredentialSummary(_StrictModel):
    """Safe credential presence information; never contains secret values."""

    storage: CredentialStorage
    annotation_providers: list[AnnotationProviderConfigurationResource] | None
    data_portal: DataPortalCredentialStatus


class _StoredDataPortalCredentials(_StrictModel):
    api_token: SecretStr | None = None


class StoredAnnotationProviderConfiguration(_AnnotationProviderConfigurationFields):
    id: uuid.UUID
    api_key: SecretStr | None = None

    @model_validator(mode="after")
    def require_builtin_credential(self) -> "StoredAnnotationProviderConfiguration":
        if self.provider != "custom" and self.api_key is None:
            raise ValueError("Built-in providers require an API key")
        return self


class StoredProviderCredentials(_StrictModel):
    """Private representation persisted in the per-user TOML file."""

    schema_version: Literal[2]
    annotation_providers: list[StoredAnnotationProviderConfiguration] = Field(
        default_factory=list
    )
    data_portal: _StoredDataPortalCredentials = Field(
        default_factory=_StoredDataPortalCredentials
    )

    @model_validator(mode="after")
    def unique_annotation_provider_configurations(
        self,
    ) -> "StoredProviderCredentials":
        seen_ids: set[uuid.UUID] = set()
        identities: set[tuple[str, str | None, str]] = set()
        for configuration in self.annotation_providers:
            if configuration.id in seen_ids:
                raise ValueError("Annotation provider configuration IDs must be unique")
            seen_ids.add(configuration.id)
            identity = (
                configuration.provider,
                configuration.base_url if configuration.provider == "custom" else None,
                configuration.api_key.get_secret_value()
                if configuration.api_key is not None
                else "",
            )
            if identity in identities:
                raise ValueError("Annotation provider configuration identities must be unique")
            identities.add(identity)
        return self


__all__ = [
    "AnnotationProvider",
    "AnnotationProviderConfigurationCreate",
    "AnnotationProviderConfigurationRename",
    "AnnotationProviderConfigurationResource",
    "CredentialStorage",
    "DataPortalCredentialPatch",
    "DataPortalCredentialStatus",
    "ProviderCredentialSummary",
    "StoredAnnotationProviderConfiguration",
    "StoredProviderCredentials",
]
