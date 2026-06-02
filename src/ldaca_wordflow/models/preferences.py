"""User preference models persisted as JSON on disk.

Used by:
- FastAPI request/response validation, generated OpenAPI clients, and backend tests
  because they need a stable JSON contract shared by route handlers, generated clients,
  and tests.

Flow: validate incoming API fields, apply defaults or validators, and serialize route
    responses in the shape expected by frontend clients and tests.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .quotation import QuotationEngineConfig, QuotationEngineType

DEFAULT_HIDDEN_VIEWS: list[str] = ["ai-annotator"]

VALID_VIEWS: set[str] = {
    "data-loader",
    "filter",
    "token-frequency",
    "concordance",
    "analysis",
    "topic-modeling",
    "quotation",
    "ai-annotator",
    "export",
}

ALWAYS_VISIBLE_VIEWS: set[str] = {"data-loader"}


class QuotationPreferences(BaseModel):
    """Preference schema persisted by preference routes for quotation preferences.

    Used by:
    - backend request/response models, backend tests because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    engine: QuotationEngineConfig = Field(
        default_factory=lambda: QuotationEngineConfig(type=QuotationEngineType.LOCAL)
    )
    last_remote_url: str = ""

    model_config = ConfigDict(extra="forbid")


class UserPreferences(BaseModel):
    """Preference schema persisted by preference routes for user preferences.

    Used by:
    - backend API routes, backend request/response models, backend tests, core workspace and
      worker services because they need a stable JSON contract shared by route handlers,
      generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    hidden_views: list[str] = Field(default_factory=lambda: list(DEFAULT_HIDDEN_VIEWS))
    favorite_workspaces: list[str] = Field(default_factory=list)
    quotation: QuotationPreferences = Field(default_factory=QuotationPreferences)
    default_tokenizer_model: str | None = None
    ldaca_oni_api_token: str | None = None

    model_config = ConfigDict(extra="forbid")

    def validated(self) -> UserPreferences:
        """Return a copy with invalid view names stripped and always-visible views unhidden.

        Called by:
        - `UserPreferences` instances owned by backend services, routes, and tests because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: validate incoming API fields, apply defaults or validators, and serialize route
            responses in the shape expected by frontend clients and tests.
        """
        cleaned_hidden = [
            v
            for v in self.hidden_views
            if v in VALID_VIEWS and v not in ALWAYS_VISIBLE_VIEWS
        ]
        return self.model_copy(update={"hidden_views": cleaned_hidden})


class UserPreferencesUpdate(BaseModel):
    """Partial update payload — only provided fields are merged.

    Used by:
    - backend API routes, backend request/response models, backend tests, core workspace and
      worker services because they need a stable JSON contract shared by route handlers,
      generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    hidden_views: list[str] | None = None
    favorite_workspaces: list[str] | None = None
    quotation: QuotationPreferences | None = None
    default_tokenizer_model: str | None = None
    ldaca_oni_api_token: str | None = None

    model_config = ConfigDict(extra="forbid")
