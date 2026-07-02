"""User preference models shared by JSON API routes and TOML disk persistence.

Used by:
- FastAPI request/response validation, generated OpenAPI clients, and backend tests
  because they need a stable JSON contract shared by route handlers, generated clients,
  and tests.

Flow: validate incoming API fields, apply defaults or validators, and serialize route
    responses in the shape expected by frontend clients and tests.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_HIDDEN_VIEWS: list[str] = []

VALID_VIEWS: set[str] = {
    "data-loader",
    "filter",
    "token-frequency",
    "concordance",
    "analysis",
    "topic-modeling",
    "quotation",
    "annotation",
    "export",
}

ALWAYS_VISIBLE_VIEWS: set[str] = {"data-loader"}


class AnnotationAiCustomProvider(BaseModel):
    """A user-defined OpenAI-compatible AI provider for annotation.

    Used by:
    - `AnnotationAiPreferences` (and therefore `UserPreferences`) because the Annotation
      tab lets users register custom providers (name + base URL) that persist to the
      TOML preferences file and reappear in the provider dropdown.

    Flow: validate the provider id/name/base_url, then serialize alongside the rest of
        the preferences payload for disk persistence and the JSON API contract.
    """

    id: str
    name: str
    base_url: str

    model_config = ConfigDict(extra="forbid")


class AnnotationAiPreferences(BaseModel):
    """Persisted Annotation-tab AI settings: provider API keys and custom providers.

    Used by:
    - `UserPreferences` because the Annotation AI panel persists per-provider API keys
      (keyed by provider id) and any user-defined custom providers so they survive
      reloads and sync across the frontend preferences store.

    Flow: hold the api_keys map and custom_providers list, defaulting both to empty so
        older preference files (lacking this section) still validate cleanly.
    """

    api_keys: dict[str, str] = Field(default_factory=dict)
    custom_providers: list[AnnotationAiCustomProvider] = Field(default_factory=list)

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
    default_tokenizer_model: str | None = None
    ldaca_oni_api_token: str | None = None
    analysis_multi_tab_enabled: bool = False
    annotation_ai: AnnotationAiPreferences = Field(
        default_factory=AnnotationAiPreferences
    )

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
    default_tokenizer_model: str | None = None
    ldaca_oni_api_token: str | None = None
    analysis_multi_tab_enabled: bool | None = None
    annotation_ai: AnnotationAiPreferences | None = None

    model_config = ConfigDict(extra="forbid")
