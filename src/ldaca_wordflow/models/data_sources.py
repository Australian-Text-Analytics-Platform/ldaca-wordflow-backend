"""Strict resources for packaged samples and the LDaCA Data Portal."""

from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from ..shared.portable_names import (
    portable_collision_key,
    portable_name_error,
    portable_relative_path_parts,
)


def sample_destination_path(collection_id: str, raw_path: str) -> PurePosixPath:
    """Canonicalize one catalogue path to its exact published destination."""

    path = PurePosixPath(*portable_relative_path_parts(raw_path))
    parts = path.parts[1:] if path.parts[0] == collection_id else path.parts
    return PurePosixPath(
        *portable_relative_path_parts(PurePosixPath(*parts).as_posix())
    )


class SampleFile(BaseModel):
    """One integrity-pinned file in a remote sample collection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")


class SampleCollection(BaseModel):
    """One importable sample collection from the configured catalogue."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    name: str
    description: str = ""
    language: str = ""
    bundled: bool = False
    total_size_bytes: int = Field(ge=0)
    recommended_for: list[str] = Field(default_factory=list)
    files: list[SampleFile] = Field(max_length=10_000)
    installed: bool = False

    @model_validator(mode="after")
    def validate_manifest(self) -> "SampleCollection":
        """Require an internally consistent, platform-unambiguous manifest."""

        if portable_name_error(self.id, exact=True) is not None:
            raise ValueError("Sample collection ID is not portable")
        portable_relative_path_parts(self.id)
        if sum(file.size for file in self.files) != self.total_size_bytes:
            raise ValueError("total_size_bytes must equal the sum of file sizes")
        normalized_paths = [
            tuple(
                portable_collision_key(part)
                for part in sample_destination_path(self.id, file.path).parts
            )
            for file in self.files
        ]
        if len(normalized_paths) != len(set(normalized_paths)):
            raise ValueError("Sample file paths must be distinct")
        return self


class SampleCatalogueResource(BaseModel):
    """Validated sample catalogue plus per-user installation state."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    collections: list[SampleCollection] = Field(max_length=500)

    @model_validator(mode="after")
    def validate_collection_ids(self) -> "SampleCatalogueResource":
        ids = [portable_collision_key(collection.id) for collection in self.collections]
        if len(ids) != len(set(ids)):
            raise ValueError("Sample collection IDs must be distinct")
        return self


class DataPortalSearchMethod(StrEnum):
    """Supported Data Portal search semantics."""

    KEYWORD = "keyword"
    IDENTIFIER = "identifier"
    COLLECTION = "collection"
    FILE_FORMAT = "file_format"
    ALL = "all"


class DataPortalSearchRequest(BaseModel):
    """One one-based portal search with an optional transient user token."""

    model_config = ConfigDict(extra="forbid")

    method: DataPortalSearchMethod = DataPortalSearchMethod.KEYWORD
    query: str = Field(default="", max_length=2_000)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=25, ge=1, le=100)
    api_token: SecretStr | None = None


class DataPortalFeaturedRequest(BaseModel):
    """Optional transient token for configured featured collections."""

    model_config = ConfigDict(extra="forbid")

    api_token: SecretStr | None = None


class DataPortalRecord(BaseModel):
    """Normalized portal record independent of Oni JSON-LD shapes."""

    model_config = ConfigDict(extra="forbid")

    id: str
    crate_id: str | None = None
    title: str
    description: str | None = None
    types: list[str] = Field(default_factory=list)
    license: str | None = None
    importable: bool
    access: list[str] = Field(default_factory=list)
    collections: list[str] = Field(default_factory=list)
    file_formats: list[str] = Field(default_factory=list)


class DataPortalSearchResource(BaseModel):
    """Direct normalized portal result page."""

    model_config = ConfigDict(extra="forbid")

    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    total: int = Field(ge=0)
    items: list[DataPortalRecord]


class DataPortalImportSubmitRequest(BaseModel):
    """Portal import request with a credential excluded from durable import state."""

    model_config = ConfigDict(extra="forbid")

    identifier: str = Field(min_length=1, max_length=4_000)
    name: str | None = Field(default=None, min_length=1, max_length=500)
    api_token: SecretStr | None = None


__all__ = [
    "DataPortalFeaturedRequest",
    "DataPortalImportSubmitRequest",
    "DataPortalRecord",
    "DataPortalSearchRequest",
    "DataPortalSearchResource",
    "SampleCatalogueResource",
    "SampleCollection",
]
