"""Strict Workspace-owned Analysis requests and lifecycle records."""

from __future__ import annotations

import math
import uuid
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StringConstraints,
    model_validator,
)

from ...shared.json_data import JsonData
from ..background import BackgroundState, Failure, Progress


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


NonEmptyText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


AnalysisState = BackgroundState


class ValidAnalysisIntegrity(_StrictModel):
    status: Literal["valid"] = "valid"


class InvalidAnalysisIntegrity(_StrictModel):
    status: Literal["invalid"] = "invalid"
    code: Literal["analysis_input_missing"] = "analysis_input_missing"
    missing_input_ids: list[uuid.UUID]

    @model_validator(mode="after")
    def require_distinct_missing_inputs(self) -> "InvalidAnalysisIntegrity":
        if not self.missing_input_ids or len(self.missing_input_ids) != len(
            set(self.missing_input_ids)
        ):
            raise ValueError("Missing Analysis inputs must be distinct")
        return self


AnalysisIntegrity = Annotated[
    ValidAnalysisIntegrity | InvalidAnalysisIntegrity,
    Field(discriminator="status"),
]


class QuotationEngineType(StrEnum):
    LOCAL = "local"
    REMOTE = "remote"


class QuotationEngineSelection(_StrictModel):
    type: QuotationEngineType = QuotationEngineType.LOCAL
    engine_id: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def validate_selection(self) -> "QuotationEngineSelection":
        if self.type is QuotationEngineType.LOCAL:
            if self.engine_id is not None:
                raise ValueError("A local quotation engine has no engine_id")
        elif not self.engine_id:
            raise ValueError("A remote quotation engine requires an engine_id")
        return self


def _validate_node_columns(
    node_ids: list[uuid.UUID],
    node_columns: dict[uuid.UUID, str],
) -> None:
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Data Block IDs must be distinct")
    if set(node_columns) != set(node_ids):
        raise ValueError("Data Block columns must exactly match the requested IDs")
    if any(not column.strip() for column in node_columns.values()):
        raise ValueError("Every Data Block requires a non-empty source column")


class TokenFrequencyAnalysisRequest(_StrictModel):
    kind: Literal["token_frequency"] = "token_frequency"
    node_ids: list[uuid.UUID] = Field(min_length=1, max_length=2)
    node_columns: dict[uuid.UUID, NonEmptyText]
    stop_words: list[str] = Field(default_factory=list)
    token_limit: int = Field(default=25, ge=1, le=5000)
    node_tokenizer_models: dict[uuid.UUID, NonEmptyText] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_nodes(self) -> "TokenFrequencyAnalysisRequest":
        _validate_node_columns(self.node_ids, self.node_columns)
        if set(self.node_tokenizer_models) - set(self.node_ids):
            raise ValueError("Tokenizer models may reference only requested inputs")
        return self


class TopicModelingAnalysisRequest(_StrictModel):
    kind: Literal["topic_modeling"] = "topic_modeling"
    node_ids: list[uuid.UUID] = Field(min_length=1, max_length=2)
    node_columns: dict[uuid.UUID, NonEmptyText]
    min_topic_size: int = Field(default=10, ge=2)
    random_seed: int = 42
    representative_words_count: int = Field(default=5, ge=1, le=100)
    sample_fractions: list[float | None] | None = None

    @model_validator(mode="after")
    def validate_nodes_and_sampling(self) -> "TopicModelingAnalysisRequest":
        _validate_node_columns(self.node_ids, self.node_columns)
        if self.sample_fractions is not None:
            if len(self.sample_fractions) != len(self.node_ids):
                raise ValueError("Sample fractions must align with Data Block IDs")
            if any(
                fraction is not None
                and (not math.isfinite(fraction) or not 0 < fraction <= 1)
                for fraction in self.sample_fractions
            ):
                raise ValueError("Sample fractions must be finite and in (0, 1]")
        return self


class ConcordanceAnalysisRequest(_StrictModel):
    kind: Literal["concordance"] = "concordance"
    node_ids: list[uuid.UUID] = Field(min_length=1, max_length=2)
    node_columns: dict[uuid.UUID, NonEmptyText]
    search_word: NonEmptyText
    num_left_tokens: int = Field(default=10, ge=0, le=1000)
    num_right_tokens: int = Field(default=10, ge=0, le=1000)
    regex: bool = False
    whole_word: bool = False
    case_sensitive: bool = False
    search_mode: Literal["regex", "tokens"] = "regex"

    @model_validator(mode="after")
    def validate_nodes(self) -> "ConcordanceAnalysisRequest":
        _validate_node_columns(self.node_ids, self.node_columns)
        return self


class QuotationAnalysisRequest(_StrictModel):
    kind: Literal["quotation"] = "quotation"
    node_id: uuid.UUID
    column: NonEmptyText
    engine: QuotationEngineSelection = Field(default_factory=QuotationEngineSelection)


class SequentialAnalysisRequest(_StrictModel):
    kind: Literal["sequential"] = "sequential"
    node_id: uuid.UUID
    time_column: NonEmptyText
    group_by_columns: list[NonEmptyText] = Field(default_factory=list, max_length=3)
    frequency: Literal[
        "second",
        "minute",
        "hourly",
        "daily",
        "weekly",
        "monthly",
        "quarterly",
        "yearly",
        "custom",
    ] = "monthly"
    sort_by_time: bool = True
    column_type: Literal["datetime", "numeric"] = "datetime"
    numeric_origin: float | None = Field(default=None, allow_inf_nan=False)
    numeric_interval: float | None = Field(default=None, allow_inf_nan=False)
    custom_interval_value: int | None = Field(default=None, ge=1)
    custom_interval_unit: (
        Literal["seconds", "minutes", "hours", "days", "weeks"] | None
    ) = None
    case_sensitive: bool = True

    @model_validator(mode="after")
    def validate_interval(self) -> "SequentialAnalysisRequest":
        if self.column_type == "numeric" and (
            self.numeric_interval is None or self.numeric_interval <= 0
        ):
            raise ValueError("Numeric input requires a positive numeric_interval")
        if self.column_type == "datetime" and self.frequency == "custom" and (
            self.custom_interval_value is None or self.custom_interval_unit is None
        ):
            raise ValueError("Custom datetime frequency requires a value and unit")
        return self


class AnnotationClass(_StrictModel):
    name: NonEmptyText = Field(max_length=200)
    description: str = Field(default="", max_length=2000)


AnnotationProvider = Literal["openai", "openrouter", "anthropic", "google"]


class _AnnotationFields(_StrictModel):
    kind: Literal["annotation"] = "annotation"
    node_id: uuid.UUID
    text_column: NonEmptyText = Field(max_length=500)
    annotation_column: NonEmptyText = Field(max_length=500)
    classes: list[AnnotationClass] = Field(min_length=1, max_length=200)
    provider: AnnotationProvider
    model: NonEmptyText = Field(max_length=500)
    instruction: NonEmptyText = Field(max_length=20_000)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, allow_inf_nan=False)
    reasoning_enabled: bool = False
    reasoning_effort: Literal["low", "medium", "high"] = "medium"
    output_node_name: NonEmptyText = Field(max_length=500)

    @model_validator(mode="after")
    def unique_classes(self) -> "_AnnotationFields":
        normalized = [item.name.casefold() for item in self.classes]
        if len(normalized) != len(set(normalized)):
            raise ValueError("Annotation class names must be unique")
        return self


class AnnotationAnalysisRequest(_AnnotationFields):
    """Secret-free immutable Annotation request stored in a Workspace."""


class AnnotationAnalysisSubmission(_AnnotationFields):
    """Annotation creation command carrying one process-local credential."""

    api_key: SecretStr = Field(min_length=1)

    def persisted_request(self) -> AnnotationAnalysisRequest:
        return AnnotationAnalysisRequest.model_validate(
            self.model_dump(exclude={"api_key"})
        )


RootAnalysisRequest = Annotated[
    TokenFrequencyAnalysisRequest
    | TopicModelingAnalysisRequest
    | ConcordanceAnalysisRequest
    | QuotationAnalysisRequest
    | SequentialAnalysisRequest
    | AnnotationAnalysisRequest,
    Field(discriminator="kind"),
]

AnalysisSubmission = Annotated[
    TokenFrequencyAnalysisRequest
    | TopicModelingAnalysisRequest
    | ConcordanceAnalysisRequest
    | QuotationAnalysisRequest
    | SequentialAnalysisRequest
    | AnnotationAnalysisSubmission,
    Field(discriminator="kind"),
]


class ConcordanceDetachmentAnalysisRequest(_StrictModel):
    kind: Literal["concordance_detachment"] = "concordance_detachment"
    node_id: uuid.UUID
    selected_columns: list[NonEmptyText] = Field(min_length=1)
    name: NonEmptyText | None = Field(default=None, max_length=500)


class ConcordanceDispersionDetachmentAnalysisRequest(_StrictModel):
    kind: Literal["concordance_dispersion_detachment"] = (
        "concordance_dispersion_detachment"
    )
    node_id: uuid.UUID
    selected_columns: list[NonEmptyText] = Field(min_length=1)
    selected_bins: list[int] | None = None
    total_bins: int | None = Field(default=None, ge=1)
    selected_matched_texts: list[str] | None = None
    match_case_insensitive: bool = False
    name: NonEmptyText | None = Field(default=None, max_length=500)


class QuotationDetachmentAnalysisRequest(_StrictModel):
    kind: Literal["quotation_detachment"] = "quotation_detachment"
    node_id: uuid.UUID
    selected_columns: list[NonEmptyText] = Field(min_length=1)
    name: NonEmptyText | None = Field(default=None, max_length=500)


ChildAnalysisRequest = Annotated[
    ConcordanceDetachmentAnalysisRequest
    | ConcordanceDispersionDetachmentAnalysisRequest
    | QuotationDetachmentAnalysisRequest,
    Field(discriminator="kind"),
]

AnalysisRequest = RootAnalysisRequest | ChildAnalysisRequest


def persisted_submission(submission: AnalysisSubmission) -> RootAnalysisRequest:
    if isinstance(submission, AnnotationAnalysisSubmission):
        return submission.persisted_request()
    return submission


def analysis_input_ids(request: AnalysisRequest) -> tuple[uuid.UUID, ...]:
    if isinstance(
        request,
        (
            TokenFrequencyAnalysisRequest,
            TopicModelingAnalysisRequest,
            ConcordanceAnalysisRequest,
        ),
    ):
        return tuple(request.node_ids)
    return (request.node_id,)


class AnalysisArtifactRecord(_StrictModel):
    """Private portable identity for one Analysis-owned artifact."""

    name: NonEmptyText = Field(max_length=500)
    relative_path: NonEmptyText = Field(max_length=1000)
    media_type: str | None = Field(default=None, max_length=200)


class _AnalysisLifecycle(_StrictModel):
    id: uuid.UUID
    parent_analysis_id: uuid.UUID | None
    request: AnalysisRequest
    state: AnalysisState
    progress: Progress
    cancellation_requested_at: AwareDatetime | None
    error: Failure | None
    created_at: AwareDatetime
    started_at: AwareDatetime | None
    finished_at: AwareDatetime | None
    revision: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> "_AnalysisLifecycle":
        is_child = isinstance(
            self.request,
            (
                ConcordanceDetachmentAnalysisRequest,
                ConcordanceDispersionDetachmentAnalysisRequest,
                QuotationDetachmentAnalysisRequest,
            ),
        )
        if is_child != (self.parent_analysis_id is not None):
            raise ValueError("Only child Analysis requests require a parent")
        terminal = self.state in {
            AnalysisState.SUCCEEDED,
            AnalysisState.FAILED,
            AnalysisState.CANCELLED,
        }
        if terminal != (self.finished_at is not None):
            raise ValueError("Terminal Analysis state and finished_at must agree")
        if self.state is AnalysisState.QUEUED and self.started_at is not None:
            raise ValueError("Queued Analyses cannot have started_at")
        if self.state in {AnalysisState.RUNNING, AnalysisState.SUCCEEDED} and (
            self.started_at is None
        ):
            raise ValueError("Running and successful Analyses require started_at")
        if (self.state is AnalysisState.FAILED) != (self.error is not None):
            raise ValueError("Only failed Analyses contain a Failure")
        if self.state is AnalysisState.CANCELLED and (
            self.cancellation_requested_at is None
        ):
            raise ValueError("Cancelled Analyses require a cancellation request")
        if self.state is AnalysisState.SUCCEEDED and self.progress.fraction != 1.0:
            raise ValueError("Successful Analyses require complete Progress")
        if self.state is not AnalysisState.SUCCEEDED and self.progress.fraction == 1.0:
            raise ValueError("Only successful Analyses may persist complete Progress")
        if self.started_at is not None and self.started_at < self.created_at:
            raise ValueError("started_at cannot precede created_at")
        if (
            self.cancellation_requested_at is not None
            and self.cancellation_requested_at < self.created_at
        ):
            raise ValueError("cancellation_requested_at cannot precede created_at")
        if self.finished_at is not None:
            lower_bound = self.started_at or self.created_at
            if self.finished_at < lower_bound:
                raise ValueError("finished_at precedes the Analysis lifecycle")
        return self


class AnalysisRecord(_AnalysisLifecycle):
    """Strict internal record persisted beneath its owning Workspace."""

    result_payload: dict[str, JsonData] | None = None
    artifact_references: list[AnalysisArtifactRecord] = Field(default_factory=list)
    output_node_id: uuid.UUID | None = None

    @model_validator(mode="after")
    def validate_result(self) -> "AnalysisRecord":
        succeeded = self.state is AnalysisState.SUCCEEDED
        if succeeded != (self.result_payload is not None):
            raise ValueError("Only a successful Analysis has a Result")
        if not succeeded and self.artifact_references:
            raise ValueError("Only a successful Analysis owns published Artifacts")
        names = [artifact.name for artifact in self.artifact_references]
        paths = [artifact.relative_path for artifact in self.artifact_references]
        if len(names) != len(set(names)) or len(paths) != len(set(paths)):
            raise ValueError("Analysis Artifact references must be unique")
        return self

    def _transition(self, **changes: object) -> "AnalysisRecord":
        payload = self.model_dump()
        payload.update(changes)
        payload["revision"] = self.revision + 1
        return AnalysisRecord.model_validate(payload)

    def start(self, timestamp: datetime) -> "AnalysisRecord":
        if self.state is not AnalysisState.QUEUED:
            raise ValueError("Only a queued Analysis can start")
        return self._transition(
            state=AnalysisState.RUNNING,
            started_at=timestamp,
        )

    def cancel_queued(self, timestamp: datetime) -> "AnalysisRecord":
        if self.state is not AnalysisState.QUEUED:
            raise ValueError("Only a queued Analysis can be cancelled immediately")
        return self._transition(
            state=AnalysisState.CANCELLED,
            cancellation_requested_at=timestamp,
            finished_at=timestamp,
        )

    def request_running_cancellation(self, timestamp: datetime) -> "AnalysisRecord":
        if self.state is not AnalysisState.RUNNING:
            raise ValueError("Only a running Analysis can request cancellation")
        if self.cancellation_requested_at is not None:
            return self
        return self._transition(cancellation_requested_at=timestamp)

    def confirm_cancelled(
        self,
        timestamp: datetime,
        *,
        progress: Progress,
    ) -> "AnalysisRecord":
        if (
            self.state is not AnalysisState.RUNNING
            or self.cancellation_requested_at is None
        ):
            raise ValueError("Cancellation confirmation requires a pending request")
        return self._transition(
            state=AnalysisState.CANCELLED,
            progress=progress,
            finished_at=timestamp,
        )

    def fail(
        self,
        timestamp: datetime,
        *,
        failure: Failure,
        progress: Progress,
    ) -> "AnalysisRecord":
        if self.state not in {AnalysisState.QUEUED, AnalysisState.RUNNING}:
            raise ValueError("Only a non-terminal Analysis can fail")
        return self._transition(
            state=AnalysisState.FAILED,
            progress=progress,
            error=failure,
            finished_at=timestamp,
        )

    def succeed(
        self,
        timestamp: datetime,
        *,
        result_payload: dict[str, JsonData],
        artifact_references: list[AnalysisArtifactRecord] | None = None,
        output_node_id: uuid.UUID | None = None,
    ) -> "AnalysisRecord":
        if self.state is not AnalysisState.RUNNING:
            raise ValueError("Only a running Analysis can succeed")
        return self._transition(
            state=AnalysisState.SUCCEEDED,
            progress=Progress(fraction=1.0, message="Complete"),
            result_payload=result_payload,
            artifact_references=artifact_references or [],
            output_node_id=output_node_id,
            finished_at=timestamp,
        )

    @classmethod
    def create(
        cls,
        request: AnalysisRequest,
        *,
        timestamp: datetime,
        parent_analysis_id: uuid.UUID | None = None,
        analysis_id: uuid.UUID | None = None,
        output_node_id: uuid.UUID | None = None,
    ) -> "AnalysisRecord":
        return cls(
            id=analysis_id or uuid.uuid4(),
            parent_analysis_id=parent_analysis_id,
            request=request,
            state=AnalysisState.QUEUED,
            progress=Progress(fraction=0.0, message="Queued"),
            cancellation_requested_at=None,
            error=None,
            created_at=timestamp,
            started_at=None,
            finished_at=None,
            revision=1,
            result_payload=None,
            artifact_references=[],
            output_node_id=output_node_id,
        )


class Analysis(_AnalysisLifecycle):
    """Exact valid public Analysis representation."""

    integrity: AnalysisIntegrity


class CorruptAnalysis(_StrictModel):
    """Minimal collection item for a root record that cannot be parsed."""

    type: Literal["corrupt_analysis"] = "corrupt_analysis"
    id: uuid.UUID
    tab_id: uuid.UUID
    code: Literal["analysis_corrupt"] = "analysis_corrupt"


def public_analysis(
    record: AnalysisRecord,
    *,
    integrity: AnalysisIntegrity,
    progress: Progress | None = None,
) -> Analysis:
    payload = record.model_dump(
        exclude={
            "result_payload",
            "artifact_references",
            "output_node_id",
        }
    )
    payload["progress"] = progress or record.progress
    payload["integrity"] = integrity
    return Analysis.model_validate(payload)


__all__ = [
    "Analysis",
    "AnalysisArtifactRecord",
    "AnalysisIntegrity",
    "AnalysisRecord",
    "AnalysisRequest",
    "AnalysisState",
    "AnalysisSubmission",
    "AnnotationAnalysisRequest",
    "AnnotationAnalysisSubmission",
    "ChildAnalysisRequest",
    "ConcordanceAnalysisRequest",
    "ConcordanceDetachmentAnalysisRequest",
    "ConcordanceDispersionDetachmentAnalysisRequest",
    "CorruptAnalysis",
    "Failure",
    "InvalidAnalysisIntegrity",
    "Progress",
    "QuotationAnalysisRequest",
    "QuotationEngineSelection",
    "QuotationEngineType",
    "QuotationDetachmentAnalysisRequest",
    "RootAnalysisRequest",
    "SequentialAnalysisRequest",
    "TokenFrequencyAnalysisRequest",
    "TopicModelingAnalysisRequest",
    "ValidAnalysisIntegrity",
    "analysis_input_ids",
    "persisted_submission",
    "public_analysis",
]
