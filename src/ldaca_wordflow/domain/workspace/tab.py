"""Strict portable Tab state owned by a Workspace aggregate."""

from __future__ import annotations

import unicodedata
import uuid
from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
)


def _reject_control_characters(value: str) -> str:
    if any(unicodedata.category(character) == "Cc" for character in value):
        raise ValueError("Tab name cannot contain control characters")
    return value


TabName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
    AfterValidator(_reject_control_characters),
]
"""One non-unique display-label contract shared by HTTP and persistence."""


class AnalysisKind(StrEnum):
    """Function identity fixed when a Workspace Tab is created."""

    ANNOTATION = "annotation"
    CONCORDANCE = "concordance"
    QUOTATION = "quotation"
    SEQUENTIAL = "sequential"
    TOKEN_FREQUENCY = "token_frequency"
    TOPIC_MODELING = "topic_modeling"


class Tab(BaseModel):
    """Complete strict public and persisted Tab representation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: uuid.UUID
    kind: AnalysisKind
    name: TabName
    analysis_id: uuid.UUID | None
    created_at: AwareDatetime
    modified_at: AwareDatetime
    revision: int = Field(ge=1)

    @classmethod
    def create(
        cls,
        *,
        kind: AnalysisKind,
        name: str,
        timestamp: datetime,
    ) -> "Tab":
        return cls(
            id=uuid.uuid4(),
            kind=kind,
            name=name,
            analysis_id=None,
            created_at=timestamp,
            modified_at=timestamp,
            revision=1,
        )


__all__ = ["AnalysisKind", "Tab", "TabName"]
