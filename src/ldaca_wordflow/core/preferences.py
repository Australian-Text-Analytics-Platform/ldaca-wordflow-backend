"""Load and save user preferences as a TOML file on disk.

Used by:
- Backend API routes, worker tasks, workspace services, and backend tests because they
  need a backend boundary that validates inputs before delegating to workspace or worker
  state.

Flow: normalize inputs, delegate to the owning backend state or service boundary, and
    return serialized values or existing domain errors to callers.
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path

import tomli_w

from ..core.utils import get_user_data_folder
from ..models.preferences import UserPreferences, UserPreferencesUpdate

logger = logging.getLogger(__name__)

PREFERENCES_FILENAME = "preferences.toml"


def _preferences_path(user_id: str) -> Path:
    """Support user preference persistence with a preferences path helper.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need a
      backend boundary that validates inputs before delegating to workspace or worker state.

    Flow: normalize inputs, delegate to the owning backend state or service boundary, and
        return serialized values or existing domain errors to callers.
    """

    user_root = get_user_data_folder(user_id).parent
    return user_root / PREFERENCES_FILENAME


def load_preferences(user_id: str) -> UserPreferences:
    """Read preferences from disk, returning validated defaults when the file is
    missing or malformed.

    Used by:
    - backend API routes, backend tests because they need a backend boundary that validates
      inputs before delegating to workspace or worker state.

    Flow: normalize inputs, delegate to the owning backend state or service boundary, and
        return serialized values or existing domain errors to callers.
    """
    path = _preferences_path(user_id)
    if not path.exists():
        return UserPreferences().validated()

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        prefs = UserPreferences.model_validate(raw)
        return prefs.validated()
    except (tomllib.TOMLDecodeError, ValueError):
        logger.warning("Corrupt preferences at %s – returning defaults", path)
        return UserPreferences().validated()


def save_preferences(user_id: str, prefs: UserPreferences) -> UserPreferences:
    """Validate and atomically write preferences to disk. Returns the saved copy.

    Used by:
    - backend API routes, backend tests because they need a backend boundary that validates
      inputs before delegating to workspace or worker state.

    Flow: normalize inputs, delegate to the owning backend state or service boundary, and
        return serialized values or existing domain errors to callers.
    """
    clean = prefs.validated()
    path = _preferences_path(user_id)
    tmp = path.with_suffix(".tmp")
    payload = clean.model_dump(mode="json", exclude_none=True)
    tmp.write_text(tomli_w.dumps(payload), encoding="utf-8")
    tmp.replace(path)
    return clean


def merge_preferences(
    current: UserPreferences, update: UserPreferencesUpdate
) -> UserPreferences:
    """Apply a partial update on top of current preferences.

    Used by:
    - backend API routes, backend tests because they need a backend boundary that validates
      inputs before delegating to workspace or worker state.

    Flow: normalize inputs, delegate to the owning backend state or service boundary, and
        return serialized values or existing domain errors to callers.
    """
    overrides: dict[str, object] = {}
    if update.hidden_views is not None:
        overrides["hidden_views"] = update.hidden_views
    if update.favorite_workspaces is not None:
        overrides["favorite_workspaces"] = update.favorite_workspaces
    if update.default_tokenizer_model is not None:
        overrides["default_tokenizer_model"] = update.default_tokenizer_model
    if "ldaca_oni_api_token" in update.model_fields_set:
        overrides["ldaca_oni_api_token"] = update.ldaca_oni_api_token
    if update.analysis_multi_tab_enabled is not None:
        overrides["analysis_multi_tab_enabled"] = update.analysis_multi_tab_enabled
    merged = current.model_copy(update=overrides)
    return merged.validated()
