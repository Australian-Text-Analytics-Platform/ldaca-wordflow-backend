"""Topic modeling request and response models.

Split from models/__init__.py.
"""

from __future__ import annotations
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field
from .analysis_common import AnalysisSorting, AnalysisTaskMetadata, AnalysisTaskState, DetachNodeOption

class TopicModelingEmbeddingCacheMeasurement(BaseModel):
    """API schema used by routes and generated clients for topic modeling embedding cache measurement.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    bytes: int
    files: int



class TopicModelingEmbeddingCacheSizeResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for topic modeling embedding cache size
    response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: Literal["successful"]
    data: TopicModelingEmbeddingCacheMeasurement



class TopicModelingEmbeddingCacheClearData(BaseModel):
    """Data payload schema embedded in API responses for topic modeling embedding cache clear data.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    bytes_freed: int
    files_removed: int
    measured_before: TopicModelingEmbeddingCacheMeasurement



class TopicModelingEmbeddingCacheClearResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for topic modeling embedding cache
    clear response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: Literal["successful"]
    message: str
    data: TopicModelingEmbeddingCacheClearData


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

    node_ids: List[str]  # 1 or 2 node IDs
    node_columns: Dict[str, str]  # Maps node_id -> column_name
    # HDBSCAN minimum cluster size: the smallest group of chunks that counts as a
    # topic. The number of topics is whatever HDBSCAN yields for it (the only
    # native topic-count control; there is no post-fit merge to a target count).
    min_topic_size: Optional[int] = 10
    random_seed: Optional[int] = 42
    representative_words_count: Optional[int] = 5
    # Sampling: one entry per corpus in node_ids order. None = no sampling for that corpus.
    sample_fractions: Optional[List[Optional[float]]] = None

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
    representative_words: List[str] = Field(default_factory=list)
    size: List[int]  # per-corpus sizes aligned to request.node_ids order
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

    topics: List[TopicModelingTopic]
    corpus_sizes: List[int]
    per_corpus_topic_counts: Optional[List[Dict[int, int]]] = None
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
    data: Optional[TopicModelingData] = None
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
    words: List[str]



class TopicModelingDetachRequest(BaseModel):
    """Request payload for detaching topic assignments from cached topic-modeling output.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    node_ids: Optional[List[str]] = None
    selected_columns: Dict[str, List[str]] = Field(default_factory=dict)
    new_node_names: Optional[Dict[str, str]] = None
    topic_column_name: Optional[str] = "TOPIC_topic"
    topic_ids: Optional[List[int]] = None
    topic_meanings_override: Optional[List[TopicMeaningOverrideItem]] = None



TopicModelingDetachNodeOption = DetachNodeOption  # shared base, kept for backwards compat


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
    data: Dict[str, List[TopicModelingDetachNodeOption]] | None = None
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

