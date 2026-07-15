"""Strict resources for the runtime-owned user file store."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..shared.json_data import JsonData


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileResource(_StrictModel):
    """One addressable regular file or directory below the user's data root."""

    name: str
    path: str
    type: Literal["file", "directory"]
    size_bytes: int | None = Field(default=None, ge=0)
    modified_at: float
    file_type: str | None = None
    preview_available: bool = False

    @model_validator(mode="after")
    def validate_kind_fields(self) -> "FileResource":
        if self.type == "directory":
            if self.size_bytes is not None or self.file_type is not None:
                raise ValueError("directory resources cannot contain file metadata")
            self.preview_available = False
        elif self.size_bytes is None or self.file_type is None:
            raise ValueError("file resources require size and file type metadata")
        return self


class FilePreviewRequest(_StrictModel):
    """One-based bounded preview request for a user-owned file."""

    path: str = Field(min_length=1)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=500)
    sheet_name: str | None = None


class FilePreviewResource(_StrictModel):
    """Typed page from a supported file format."""

    path: str
    file_type: str
    supported_types: list[str]
    columns: list[str]
    rows: list[dict[str, JsonData]]
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    total_rows: int = Field(ge=0)
    total_pages: int = Field(ge=0)
    sheet_names: list[str] | None = None
    selected_sheet: str | None = None


class CreateFolderRequest(_StrictModel):
    """Create one validated child under a relative parent path."""

    name: str = Field(min_length=1, max_length=500)
    parent_path: str = ""


class MoveFileRequest(_StrictModel):
    """Move one file into an existing relative directory."""

    source_path: str = Field(min_length=1)
    target_directory_path: str = ""


__all__ = [
    "CreateFolderRequest",
    "FilePreviewRequest",
    "FilePreviewResource",
    "FileResource",
    "MoveFileRequest",
]
