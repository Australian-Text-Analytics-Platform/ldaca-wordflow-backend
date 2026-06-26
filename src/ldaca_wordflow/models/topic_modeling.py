"""Topic modeling request and response models.

Split from models/__init__.py.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .analysis_common import (
    AnalysisSorting,
    AnalysisTaskMetadata,
    AnalysisTaskState,
    DetachNodeOption,
)

# =============================================================================
# RESPONSE MODELS
# =============================================================================


class TopicModelingRequest(BaseModel):
    """Request schema used by API routes and generated clients for topic modeling request.

    Used by:
    - analysis task helpers, backend API routes, backend request/response models, backend
      tests because they need a stable JSON contract shared by route handlers, generated
      clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    node_ids: list[str]  # 1 or 2 node IDs
    node_columns: dict[str, str]  # Maps node_id -> column_name
    # HDBSCAN minimum cluster size: the smallest group of chunks that counts as a
    # topic. The number of topics is whatever HDBSCAN yields for it (the only
    # native topic-count control; there is no post-fit merge to a target count).
    min_topic_size: int | None = 10
    random_seed: int | None = 42
    representative_words_count: int | None = 5
    # Sampling: one entry per corpus in node_ids order. None = no sampling for that corpus.
    sample_fractions: list[float | None] | None = None

    # Pydantic v2 model config
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "node_ids": ["node1", "node2"],
                "node_columns": {"node1": "text", "node2": "content"},
                "min_topic_size": 10,
                "random_seed": 42,
                "representative_words_count": 5,
                "sample_fractions": [0.2, 0.5],
            }
        }
    )


class TopicModelingTopic(BaseModel):
    """API schema used by routes and generated clients for topic modeling topic.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    id: int
    label: str
    representative_words: list[str] = Field(default_factory=list)
    size: list[int]  # per-corpus sizes aligned to request.node_ids order
    total_size: int
    x: float
    y: float


class TopicModelingData(BaseModel):
    """Data payload schema embedded in API responses for topic modeling data.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    topics: list[TopicModelingTopic]
    corpus_sizes: list[int]
    per_corpus_topic_counts: list[dict[int, int]] | None = None
    meta: AnalysisTaskMetadata | None = None


class TopicModelingResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for topic modeling response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: AnalysisTaskState
    message: str
    data: TopicModelingData | None = None
    metadata: AnalysisTaskMetadata | None = None


class TopicMeaningOverrideItem(BaseModel):
    """One topic's representative-words override for detach.

    Lets the frontend ship exactly what the user sees — post-fit
    "Words per topic" slice, post-fit stopword filter — instead of
    forcing the meanings parquet (written at fit time) into the
    detached node.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    topic_id: int
    words: list[str]


class TopicModelingDetachRequest(BaseModel):
    """Request payload for detaching topic assignments from cached topic-modeling output.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    node_ids: list[str] | None = None
    selected_columns: dict[str, list[str]] = Field(default_factory=dict)
    new_node_names: dict[str, str] | None = None
    topic_column_name: str | None = "TOPIC_topic"
    topic_ids: list[int] | None = None
    topic_meanings_override: list[TopicMeaningOverrideItem] | None = None


class TopicModelingDetachOptionsResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for topic modeling detach options
    response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: AnalysisTaskState
    message: str
    data: dict[str, list[DetachNodeOption]] | None = None
    metadata: AnalysisTaskMetadata | None = None


class TopicModelingDetachedNode(BaseModel):
    """API schema used by routes and generated clients for topic modeling detached node.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    source_node_id: str
    new_node_id: str
    topic_meanings_node_id: str | None = None


class TopicModelingDetachData(BaseModel):
    """Data payload schema embedded in API responses for topic modeling detach data.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    detached_nodes: list[TopicModelingDetachedNode] = Field(default_factory=list)


class TopicModelingDetachResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for topic modeling detach response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: AnalysisTaskState
    message: str
    data: TopicModelingDetachData | None = None
    metadata: AnalysisTaskMetadata | None = None


# Concordance response models
