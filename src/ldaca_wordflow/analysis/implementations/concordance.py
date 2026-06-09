"""Concordance analysis request/result schema module.

Used by:
- concordance analysis API routes and task persistence payloads because they need a
  backend boundary that validates inputs before delegating to workspace or worker state.
Why:
- Keeps concordance-specific request/result contracts separated from route logic.

Flow: normalize inputs, delegate to the owning backend state or service boundary, and
    return serialized values or existing domain errors to callers.
"""

from typing import Any, Literal

from pydantic import Field

from ..models import BaseAnalysisRequest
from ..results import BaseAnalysisResult


class ConcordanceRequest(BaseAnalysisRequest):
    """Request payload schema for concordance runs.

    Used by:
    - concordance route/task creation flows because they need a backend boundary that
      validates inputs before delegating to workspace or worker state.
    Why:
    - Validates all search/context/pagination-related analysis inputs.

    Flow: normalize inputs, delegate to the owning backend state or service boundary, and
        return serialized values or existing domain errors to callers.
    """

    node_ids: list[str]
    node_columns: dict[str, str] | None = None
    search_word: str
    num_left_tokens: int = 50
    num_right_tokens: int = 50
    regex: bool = False
    whole_word: bool = False
    case_sensitive: bool = False
    # Engine selector: "regex" walks raw text (default, preserves partial-word
    # patterns); "tokens" walks the active node's tokenization column for
    # exact-token matches with N-actual-token context. Persisted on the task
    # so hydration replays the same engine.
    search_mode: Literal["regex", "tokens"] = "regex"
    # node_id -> parquet path holding flattened occurrence rows.
    # Populated when a materialize background task completes for that node.
    materialized_paths: dict[str, str] | None = None


class ConcordanceResult(BaseAnalysisResult):
    """Serializable concordance result wrapper.

    Used by:
    - analysis task result serialization helpers because analysis flows need per-user task
      state to survive across route calls and worker result persistence.
    Why:
    - Provides a consistent JSON output shape for concordance payloads.

    Flow: normalize inputs, delegate to the owning backend state or service boundary, and
        return serialized values or existing domain errors to callers.
    """

    def __init__(self, results: list[dict[str, Any]], total_hits: int):
        """Initialize ConcordanceResult state used by concordance analysis adapters.

        Called by:
        - `ConcordanceResult` construction in backend services and tests because tests need the
          same observable contract that production routes and workers rely on.

        Flow: normalize inputs, delegate to the owning backend state or service boundary, and
            return serialized values or existing domain errors to callers.
        """

        self.results = results
        self.total_hits = total_hits

    def to_json(self, **kwargs: Any) -> dict[str, Any]:
        """Return JSON-serializable concordance payload.

        Called by:
        - `ConcordanceResult` instances owned by backend services, routes, and tests because
          they need a backend boundary that validates inputs before delegating to workspace or
          worker state.

        Flow: normalize inputs, delegate to the owning backend state or service boundary, and
            return serialized values or existing domain errors to callers.
        """
        return {
            "results": self.results,
            "total_hits": self.total_hits,
        }
