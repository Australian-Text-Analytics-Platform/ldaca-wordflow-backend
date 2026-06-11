"""Topic-modeling analysis request schema module.

Used by:
- topic-modeling routes and worker task request validation because they need a backend
  boundary that validates inputs before delegating to workspace or worker state.
Why:
- Keeps topic-modeling specific input contract centralized.

Flow: normalize inputs, delegate to the owning backend state or service boundary, and
    return serialized values or existing domain errors to callers.
"""

from pydantic import Field

from ..models import BaseAnalysisRequest


class TopicModelingRequest(BaseAnalysisRequest):
    """Request model for topic-modeling analysis.

    Used by:
    - topic-modeling run/update endpoints because they need a backend boundary that
      validates inputs before delegating to workspace or worker state.
    Why:
    - Validates node selection and clustering configuration inputs.

    Flow: normalize inputs, delegate to the owning backend state or service boundary, and
        return serialized values or existing domain errors to callers.
    """

    node_ids: list[str] = Field(..., description="List of node IDs to analyze")
    node_columns: dict[str, str] | None = Field(
        None, description="Map of node_id to column name"
    )
    min_topic_size: int = Field(
        10,
        description=(
            "HDBSCAN minimum cluster size: the smallest group of chunks that "
            "counts as a topic. The number of topics is whatever HDBSCAN yields "
            "for this value (the only native topic-count control)."
        ),
    )
    random_seed: int = Field(
        42, description="Random seed used for reproducible topic-modeling runs"
    )
    representative_words_count: int = Field(
        5, description="Number of representative words to keep per topic"
    )
    sample_fractions: list[float | None] | None = Field(
        None,
        description=(
            "One sampling fraction (0 < f ≤ 1) per corpus in node_ids order. "
            "None for a corpus means no sampling. Sampling uses random_seed."
        ),
    )
