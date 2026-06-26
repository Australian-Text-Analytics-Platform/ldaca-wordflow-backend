"""Token frequency and statistics models.

Split from models/__init__.py.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from .analysis_common import AnalysisSorting, AnalysisTaskMetadata, AnalysisTaskState


class StopWordsPayload(BaseModel):
    """API schema used by routes and generated clients for stop words payload.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    stop_words: list[str]


class TokenFrequencyRequest(BaseModel):
    """Request schema used by API routes and generated clients for token frequency request.

    Used by:
    - analysis task helpers, backend API routes, backend request/response models, backend
      tests because they need a stable JSON contract shared by route handlers, generated
      clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    node_ids: list[str]  # 1 or 2 node IDs
    node_columns: dict[str, str]  # Maps node_id -> column_name
    stop_words: list[str] | None = None
    token_limit: int | None = None
    tokenizer_model: str | None = None
    node_tokenizer_models: dict[str, str] | None = None
    # Pydantic v2 model config
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "node_ids": ["node1", "node2"],
                "node_columns": {"node1": "text_column", "node2": "content_column"},
                "stop_words": ["the", "and", "or"],
                "token_limit": 50,
                "node_tokenizer_models": {
                    "node1": "native:plain_words_en",
                    "node2": "huggingface:bert-base-uncased",
                },
            }
        },
    )


class TokenFrequencyPreferenceUpdateRequest(BaseModel):
    """Request schema used by API routes and generated clients for token frequency preference update request.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    token_limit: int | None = None
    stop_words: list[str] | None = None


class TokenFrequencyData(BaseModel):
    """Data payload schema embedded in API responses for token frequency data.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    token: str
    frequency: int


class TokenStatisticsData(BaseModel):
    """Token-level comparative statistics.

    Numeric statistics use a JSON-safe union to preserve semantic distinctions:
    - finite number -> float
    - positive infinity -> "+Inf"
    - negative infinity -> "-Inf"
    - missing/undefined -> None

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    token: str
    freq_reference: int  # OR - observed frequency in reference corpus
    freq_study: int  # OS - observed frequency in study corpus
    expected_reference: float | str | None  # Expected frequency in reference corpus
    expected_study: float | str | None  # Expected frequency in study corpus
    reference_total: int  # Total tokens in reference corpus
    study_total: int  # Total tokens in study corpus
    percent_reference: float | str | None  # %R - percentage in reference corpus
    percent_study: float | str | None  # %S - percentage in study corpus
    percent_diff: float | str | None  # %DIFF - percentage difference
    log_likelihood_llv: float | str | None  # LL - log likelihood G2 statistic
    bayes_factor_bic: float | str | None  # Bayes - Bayes factor (BIC)
    effect_size_ell: float | str | None  # ELL - effect size for log likelihood
    relative_risk: float | str | None = None  # RRisk - relative risk ratio
    log_ratio: float | str | None = None  # LogRatio - log of relative frequencies
    odds_ratio: float | str | None = None  # OddsRatio - odds ratio
    significance: str  # Significance level indicator


class TokenFrequencyNodeResult(BaseModel):
    """API schema used by routes and generated clients for token frequency node result.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    data: list[TokenFrequencyData]
    columns: list[str] = ["token", "frequency"]
    # Optional metadata (e.g., server-side truncation info)
    metadata: AnalysisTaskMetadata | None = None


class TokenFrequencyResponse(BaseModel):
    """Unified response model for token frequency analysis.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: AnalysisTaskState | None = None
    message: str | None = None
    data: dict[str, TokenFrequencyNodeResult] | None = None  # Maps node_name -> { data: [...], columns: [...] }
    statistics: list[TokenStatisticsData] | None = None  # Statistical measures (only when comparing 2 nodes)
    token_limit: int | None = None
    analysis_params: dict[str, Any] | None = None
    metadata: AnalysisTaskMetadata | None = None
    stop_words: list[str] | None = None


# =============================================================================
# AI ANNOTATION MODELS
# =============================================================================
