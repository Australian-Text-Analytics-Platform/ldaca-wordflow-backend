"""Strict stored, query, and public Result models for Workspace Analyses."""

from __future__ import annotations

import uuid
from typing import Annotated, Generic, Literal, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field

from ..domain.workspace import NodeProvenance
from ..shared.json_data import JsonData
from .names import NodeName
from .tables import CompleteTableResource, PagedTableResource
from .tokenization import TokenizationMetadata


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _PagedQuery(_StrictModel):
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=500)
    sort_by: str | None = None
    descending: bool = False


class TopicModelingResultQuery(_PagedQuery):
    kind: Literal["topic_modeling"] = "topic_modeling"
    topic_ids: list[int] | None = None


class ConcordanceResultQuery(_PagedQuery):
    kind: Literal["concordance"] = "concordance"
    node_id: uuid.UUID | None = None


class QuotationResultQuery(_PagedQuery):
    kind: Literal["quotation"] = "quotation"
    context_length: int = Field(default=20, ge=0, le=1000)


AnalysisResultQuery = Annotated[
    TopicModelingResultQuery | ConcordanceResultQuery | QuotationResultQuery,
    Field(discriminator="kind"),
]


class ArtifactResource(_StrictModel):
    name: str = Field(min_length=1, max_length=500)
    media_type: str | None = Field(default=None, max_length=200)
    url: str = Field(min_length=1)


class StoredArtifactIdentity(_StrictModel):
    name: str = Field(min_length=1, max_length=500)
    media_type: str | None = Field(default=None, max_length=200)


class ResultPagination(_StrictModel):
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    total_rows: int = Field(ge=0)
    total_pages: int = Field(ge=0)


class SourcePagePagination(_StrictModel):
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    total_source_rows: int = Field(ge=0)
    total_source_pages: int = Field(ge=0)
    result_count: int = Field(ge=0)
    has_next: bool
    has_prev: bool


class ResultSorting(_StrictModel):
    sort_by: str | None = None
    descending: bool


class ResultColumnMetadata(_StrictModel):
    concordance_columns: list[str] = Field(default_factory=list)
    quotation_columns: list[str] = Field(default_factory=list)
    metadata_columns: list[str] = Field(default_factory=list)
    all_columns: list[str]


ArtifactValueT = TypeVar("ArtifactValueT")
PrivateArtifactPath = Annotated[str, Field(min_length=1)]


class CompleteTableIdentity(_StrictModel, Generic[ArtifactValueT]):
    table_id: str = Field(min_length=1, max_length=200)
    artifact: ArtifactValueT


class PagedTableIdentity(_StrictModel, Generic[ArtifactValueT]):
    table_id: str = Field(min_length=1, max_length=200)
    artifact: ArtifactValueT


class _TokenNodeTable(_StrictModel, Generic[ArtifactValueT]):
    node_id: uuid.UUID
    node_name: NodeName
    table: CompleteTableIdentity[ArtifactValueT]


class _TokenTables(_StrictModel, Generic[ArtifactValueT]):
    version: Literal[1]
    nodes: list[_TokenNodeTable[ArtifactValueT]]
    statistics: CompleteTableIdentity[ArtifactValueT] | None = None


class TokenAnalysisParameters(_StrictModel):
    node_ids: list[uuid.UUID]
    node_columns: dict[uuid.UUID, str]
    token_limit: int = Field(ge=1)
    server_limit: int = Field(ge=1)
    stop_words: list[str]
    node_tokenizer_models: dict[uuid.UUID, str]


class TokenResultMetadata(_StrictModel):
    token_limit: int = Field(ge=1)
    server_limit: int = Field(ge=1)
    stop_words: list[str]
    node_tokenizer_models: dict[uuid.UUID, str]
    node_display_names: dict[uuid.UUID, str]


class _TokenFrequencyBody(_StrictModel):
    token_limit: int = Field(ge=1)
    analysis_params: TokenAnalysisParameters
    metadata: TokenResultMetadata
    stop_words: list[str]


class TokenFrequencyWorkerResult(_TokenFrequencyBody):
    state: Literal["successful"]
    message: str
    tables: _TokenTables[PrivateArtifactPath]


class TokenFrequencyStoredResult(_TokenFrequencyBody):
    tables: _TokenTables[StoredArtifactIdentity]


class _TokenNodeTableResource(_StrictModel):
    node_id: uuid.UUID
    node_name: NodeName
    table: CompleteTableResource


class _TokenTableResources(_StrictModel):
    version: Literal[1]
    nodes: list[_TokenNodeTableResource]
    statistics: CompleteTableResource | None = None


class TokenFrequencyResult(_TokenFrequencyBody):
    kind: Literal["token_frequency"] = "token_frequency"
    tables: _TokenTableResources


class TopicItem(_StrictModel):
    id: int
    label: str
    representative_words: list[str]
    size: list[int]
    total_size: int = Field(ge=0)
    x: float
    y: float


class _TopicNodeArtifact(_StrictModel, Generic[ArtifactValueT]):
    node_id: uuid.UUID
    node_name: NodeName
    text_column: str
    original_columns: list[str]
    assignments: PagedTableIdentity[ArtifactValueT]


class _TopicArtifacts(_StrictModel, Generic[ArtifactValueT]):
    version: Literal[1]
    topic_meanings_parquet_path: ArtifactValueT
    nodes: list[_TopicNodeArtifact[ArtifactValueT]]


class _TopicNodeArtifactResource(_StrictModel):
    node_id: uuid.UUID
    node_name: NodeName
    text_column: str
    original_columns: list[str]
    assignments: PagedTableResource


class _TopicArtifactsResource(_StrictModel):
    version: Literal[1]
    topic_meanings_parquet_path: ArtifactResource
    nodes: list[_TopicNodeArtifactResource]


class TopicStageTiming(_StrictModel):
    stage: str
    elapsed_ms: float = Field(ge=0)


class TopicMetadata(_StrictModel):
    embeddings_from_ctfidf: bool | None = None
    total_topics_incl_outlier: int | None = Field(default=None, ge=0)
    native: bool | None = None
    engine: str | None = None
    embedding_model: str | None = None
    embedding_backend: str | None = None
    min_topic_size: int | None = Field(default=None, ge=1)
    representative_words_count: int | None = Field(default=None, ge=1)
    random_state: int | None = None
    vectorizer_model: str | None = None
    n_chunks: int | None = Field(default=None, ge=0)
    corpus_sizes_before_sample: list[int] | None = None
    corpus_sizes_after_sample: list[int] | None = None
    stage_timings_ms: list[TopicStageTiming] | None = None
    node_names: list[str] = Field(default_factory=list)


class _TopicModelingBody(_StrictModel):
    topics: list[TopicItem]
    corpus_sizes: list[int]
    per_corpus_topic_counts: list[dict[int, int]] | None = None
    meta: TopicMetadata


class TopicModelingWorkerResult(_TopicModelingBody):
    artifacts: _TopicArtifacts[PrivateArtifactPath]


class TopicModelingStoredResult(_TopicModelingBody):
    artifacts: _TopicArtifacts[StoredArtifactIdentity]


class TopicModelingResult(_TopicModelingBody):
    kind: Literal["topic_modeling"] = "topic_modeling"
    artifacts: _TopicArtifactsResource
    pagination: ResultPagination
    query: TopicModelingResultQuery


class ConcordancePage(_StrictModel):
    data: list[list[dict[str, JsonData]]]
    columns: list[str]
    metadata: ResultColumnMetadata
    pagination: SourcePagePagination
    sorting: ResultSorting


class ConcordanceAnalysisParameters(_StrictModel):
    node_ids: list[uuid.UUID]
    node_columns: dict[uuid.UUID, str]
    search_word: str = Field(min_length=1)
    num_left_tokens: int = Field(default=10, ge=0, le=1000)
    num_right_tokens: int = Field(default=10, ge=0, le=1000)
    regex: bool = False
    whole_word: bool = False
    case_sensitive: bool = False
    search_mode: Literal["regex", "tokens"] = "regex"
    page: int = Field(default=1, ge=1)
    label_to_node_map: dict[str, uuid.UUID] = Field(default_factory=dict)


class ConcordanceWorkerResult(_StrictModel):
    state: Literal["successful"]
    message: str
    data: dict[uuid.UUID, ConcordancePage]
    analysis_params: ConcordanceAnalysisParameters
    combinable: bool


class ConcordanceStoredResult(_StrictModel):
    data: dict[uuid.UUID, ConcordancePage]
    analysis_params: ConcordanceAnalysisParameters
    combinable: bool


class ConcordanceResult(ConcordanceStoredResult):
    kind: Literal["concordance"] = "concordance"
    query: ConcordanceResultQuery


class QuotationWorkerResult(_StrictModel):
    data: list[list[dict[str, JsonData]]]
    columns: list[str]
    metadata: ResultColumnMetadata
    pagination: SourcePagePagination
    sorting: ResultSorting


class QuotationStoredResult(_StrictModel):
    data: list[list[dict[str, JsonData]]]
    columns: list[str]
    metadata: ResultColumnMetadata
    pagination: SourcePagePagination
    sorting: ResultSorting


class QuotationResult(QuotationStoredResult):
    kind: Literal["quotation"] = "quotation"
    query: QuotationResultQuery


class SequentialWorkerResult(_StrictModel):
    state: Literal["successful"]
    table: CompleteTableIdentity[PrivateArtifactPath]


class SequentialStoredResult(_StrictModel):
    table: CompleteTableIdentity[StoredArtifactIdentity]


class SequentialResult(SequentialStoredResult):
    kind: Literal["sequential"] = "sequential"
    table: CompleteTableResource


class DetachedDataBlockMetadata(_StrictModel):
    """Exact portable Data Block metadata accepted from a child process."""

    id: uuid.UUID
    name: NodeName
    provenance: NodeProvenance
    document: str | None = Field(default=None, max_length=500)
    color: str | None = Field(default=None, max_length=100)
    tokenization: dict[str, TokenizationMetadata] = Field(default_factory=dict)


class _DetachmentWorkerData(_StrictModel):
    data_block: DetachedDataBlockMetadata
    parquet_path: PrivateArtifactPath
    output_columns: list[str]
    record_count: int = Field(ge=0)


class DetachmentWorkerResult(_StrictModel):
    state: Literal["successful"]
    result: _DetachmentWorkerData
    message: str


class ConcordanceDetachmentWorkerResult(DetachmentWorkerResult):
    pass


class ConcordanceDispersionDetachmentWorkerResult(DetachmentWorkerResult):
    pass


class QuotationDetachmentWorkerResult(DetachmentWorkerResult):
    pass


class AnnotationWorkerResult(DetachmentWorkerResult):
    pass


class DetachmentStoredResult(_StrictModel):
    output_node_id: uuid.UUID
    output_columns: list[str]
    record_count: int = Field(ge=0)


class ConcordanceDetachmentResult(DetachmentStoredResult):
    kind: Literal["concordance_detachment"] = "concordance_detachment"


class ConcordanceDispersionDetachmentResult(DetachmentStoredResult):
    kind: Literal["concordance_dispersion_detachment"] = (
        "concordance_dispersion_detachment"
    )


class QuotationDetachmentResult(DetachmentStoredResult):
    kind: Literal["quotation_detachment"] = "quotation_detachment"


class AnnotationStoredResult(DetachmentStoredResult):
    annotation_column: str = Field(min_length=1, max_length=500)


class AnnotationResult(AnnotationStoredResult):
    kind: Literal["annotation"] = "annotation"


AnalysisResult = Annotated[
    TokenFrequencyResult
    | TopicModelingResult
    | ConcordanceResult
    | QuotationResult
    | SequentialResult
    | AnnotationResult
    | ConcordanceDetachmentResult
    | ConcordanceDispersionDetachmentResult
    | QuotationDetachmentResult,
    Field(discriminator="kind"),
]

ANALYSIS_WORKER_RESULT_MODELS: dict[str, type[BaseModel]] = {
    "token_frequency": TokenFrequencyWorkerResult,
    "topic_modeling": TopicModelingWorkerResult,
    "concordance": ConcordanceWorkerResult,
    "quotation": QuotationWorkerResult,
    "sequential": SequentialWorkerResult,
    "annotation": AnnotationWorkerResult,
    "concordance_detachment": ConcordanceDetachmentWorkerResult,
    "concordance_dispersion_detachment": (ConcordanceDispersionDetachmentWorkerResult),
    "quotation_detachment": QuotationDetachmentWorkerResult,
}

ANALYSIS_STORED_RESULT_MODELS: dict[str, type[BaseModel]] = {
    "token_frequency": TokenFrequencyStoredResult,
    "topic_modeling": TopicModelingStoredResult,
    "concordance": ConcordanceStoredResult,
    "quotation": QuotationStoredResult,
    "sequential": SequentialStoredResult,
    "annotation": AnnotationStoredResult,
    "concordance_detachment": DetachmentStoredResult,
    "concordance_dispersion_detachment": DetachmentStoredResult,
    "quotation_detachment": DetachmentStoredResult,
}


def stored_result_payload(kind: str, result: BaseModel) -> dict[str, JsonData]:
    """Remove execution-only status fields before persistence."""

    excluded = {
        "token_frequency": {"state", "message"},
        "concordance": {"state", "message"},
        "sequential": {"state"},
    }.get(kind, set())
    return cast(
        dict[str, JsonData],
        result.model_dump(mode="json", exclude=excluded),
    )


__all__ = [
    "ANALYSIS_STORED_RESULT_MODELS",
    "ANALYSIS_WORKER_RESULT_MODELS",
    "AnalysisResult",
    "AnalysisResultQuery",
    "AnnotationResult",
    "AnnotationStoredResult",
    "AnnotationWorkerResult",
    "ArtifactResource",
    "ConcordanceResult",
    "ConcordanceResultQuery",
    "ConcordanceStoredResult",
    "ConcordanceWorkerResult",
    "CompleteTableIdentity",
    "ConcordanceDetachmentResult",
    "ConcordanceDetachmentWorkerResult",
    "ConcordanceDispersionDetachmentResult",
    "ConcordanceDispersionDetachmentWorkerResult",
    "DetachedDataBlockMetadata",
    "DetachmentStoredResult",
    "DetachmentWorkerResult",
    "PrivateArtifactPath",
    "QuotationResult",
    "QuotationResultQuery",
    "QuotationStoredResult",
    "QuotationWorkerResult",
    "QuotationDetachmentResult",
    "QuotationDetachmentWorkerResult",
    "ResultPagination",
    "SequentialResult",
    "SequentialStoredResult",
    "SequentialWorkerResult",
    "StoredArtifactIdentity",
    "TokenFrequencyResult",
    "TokenFrequencyStoredResult",
    "TokenFrequencyWorkerResult",
    "TopicModelingResult",
    "TopicModelingResultQuery",
    "TopicModelingStoredResult",
    "TopicModelingWorkerResult",
    "stored_result_payload",
]
