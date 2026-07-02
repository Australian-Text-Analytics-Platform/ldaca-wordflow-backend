"""Tests for user preferences TOML load / save / merge logic."""

from pathlib import Path
from unittest.mock import patch

import pytest
import tomli_w
from pydantic import ValidationError
from ldaca_wordflow.core.preferences import (
    load_preferences,
    merge_preferences,
    save_preferences,
)
from ldaca_wordflow.models.preferences import (
    DEFAULT_HIDDEN_VIEWS,
    AnnotationAiCustomProvider,
    AnnotationAiPreferences,
    UserPreferences,
    UserPreferencesUpdate,
)


@pytest.fixture()
def user_data_dir(tmp_path: Path):
    """Patch get_user_data_folder to return tmp_path/user_data.

    _preferences_path uses .parent on that, so preferences.toml lands in tmp_path.
    The fixture yields tmp_path (the user root) so tests write files there.
    """
    data_dir = tmp_path / "user_data"
    data_dir.mkdir()
    with patch(
        "ldaca_wordflow.core.preferences.get_user_data_folder", return_value=data_dir
    ):
        yield tmp_path


class TestLoadPreferences:
    def test_returns_defaults_when_file_missing(self, user_data_dir: Path):
        prefs = load_preferences("test-user")
        assert prefs.hidden_views == DEFAULT_HIDDEN_VIEWS
        assert prefs.favorite_workspaces == []
        assert prefs.analysis_multi_tab_enabled is False

    def test_loads_from_disk(self, user_data_dir: Path):
        payload = {
            "hidden_views": ["export"],
            "favorite_workspaces": ["ws-1"],
            "analysis_multi_tab_enabled": True,
        }
        (user_data_dir / "preferences.toml").write_text(tomli_w.dumps(payload))
        prefs = load_preferences("test-user")
        assert prefs.hidden_views == ["export"]
        assert prefs.favorite_workspaces == ["ws-1"]
        assert prefs.analysis_multi_tab_enabled is True

    def test_returns_defaults_on_corrupt_toml(self, user_data_dir: Path):
        (user_data_dir / "preferences.toml").write_text("bad = [toml")
        prefs = load_preferences("test-user")
        assert prefs == UserPreferences().validated()

    def test_strips_invalid_view_names(self, user_data_dir: Path):
        payload = {"hidden_views": ["export", "nonexistent-view"]}
        (user_data_dir / "preferences.toml").write_text(tomli_w.dumps(payload))
        prefs = load_preferences("test-user")
        assert "nonexistent-view" not in prefs.hidden_views
        assert "export" in prefs.hidden_views

    def test_cannot_hide_data_loader(self, user_data_dir: Path):
        payload = {"hidden_views": ["data-loader", "export"]}
        (user_data_dir / "preferences.toml").write_text(tomli_w.dumps(payload))
        prefs = load_preferences("test-user")
        assert "data-loader" not in prefs.hidden_views
        assert "export" in prefs.hidden_views


class TestSavePreferences:
    def test_round_trip(self, user_data_dir: Path):
        prefs = UserPreferences(
            hidden_views=["quotation", "export"],
            favorite_workspaces=["ws-2"],
        )
        saved = save_preferences("test-user", prefs)
        assert saved.hidden_views == ["quotation", "export"]

        loaded = load_preferences("test-user")
        assert loaded == saved

    def test_validated_on_save(self, user_data_dir: Path):
        prefs = UserPreferences(hidden_views=["data-loader"])
        saved = save_preferences("test-user", prefs)
        assert "data-loader" not in saved.hidden_views

    def test_save_omits_quotation_task_parameters(self, user_data_dir: Path):
        save_preferences("test-user", UserPreferences())
        saved_text = (user_data_dir / "preferences.toml").read_text()
        assert "quotation" not in saved_text

    def test_all_default_preferences_write_empty_file(self, user_data_dir: Path):
        # VS Code-style sparse persistence: a preferences object equal to the
        # model defaults records nothing, so the file is effectively empty.
        save_preferences("test-user", UserPreferences())
        saved_text = (user_data_dir / "preferences.toml").read_text().strip()
        assert saved_text == ""

    def test_save_omits_fields_set_to_their_default(self, user_data_dir: Path):
        # Explicitly setting a field to its default value still omits it from the
        # file (the value is indistinguishable from never having changed it).
        prefs = UserPreferences(
            analysis_multi_tab_enabled=False,  # default
            favorite_workspaces=["ws-1"],  # non-default
        )
        save_preferences("test-user", prefs)
        saved_text = (user_data_dir / "preferences.toml").read_text()
        assert "analysis_multi_tab_enabled" not in saved_text
        assert "ws-1" in saved_text

    def test_save_omits_empty_annotation_ai_section(self, user_data_dir: Path):
        # An empty annotation_ai section equals its default and must not appear.
        save_preferences(
            "test-user",
            UserPreferences(annotation_ai=AnnotationAiPreferences()),
        )
        saved_text = (user_data_dir / "preferences.toml").read_text()
        assert "annotation_ai" not in saved_text


class TestMergePreferences:
    def test_partial_update_preserves_other_fields(self):
        current = UserPreferences(
            hidden_views=["quotation"],
            favorite_workspaces=["ws-1"],
            analysis_multi_tab_enabled=False,
        )
        update = UserPreferencesUpdate(
            hidden_views=["export"],
            analysis_multi_tab_enabled=True,
        )
        merged = merge_preferences(current, update)
        assert merged.hidden_views == ["export"]
        assert merged.favorite_workspaces == ["ws-1"]
        assert merged.analysis_multi_tab_enabled is True

    def test_partial_update_can_disable_multi_tab_ui(self):
        current = UserPreferences(analysis_multi_tab_enabled=True)
        update = UserPreferencesUpdate(analysis_multi_tab_enabled=False)
        merged = merge_preferences(current, update)
        assert merged.analysis_multi_tab_enabled is False

    def test_null_multi_tab_update_is_noop(self):
        current = UserPreferences(analysis_multi_tab_enabled=True)
        update = UserPreferencesUpdate(analysis_multi_tab_enabled=None)
        merged = merge_preferences(current, update)
        assert merged.analysis_multi_tab_enabled is True

    def test_empty_update_is_noop(self):
        current = UserPreferences(hidden_views=["quotation"])
        update = UserPreferencesUpdate()
        merged = merge_preferences(current, update)
        assert merged == current.validated()

    def test_rejects_quotation_task_parameters(self):
        with pytest.raises(ValidationError):
            UserPreferencesUpdate.model_validate(
                {"quotation": {"last_remote_url": "http://example.com"}}
            )


class TestAnnotationAiPreferences:
    """Round-trip and merge coverage for the annotation_ai preference section."""

    def test_defaults_to_empty(self):
        prefs = UserPreferences()
        assert prefs.annotation_ai is not None
        assert prefs.annotation_ai.api_keys == {}
        assert prefs.annotation_ai.custom_providers == []

    def test_round_trip_persists_keys_and_providers(self, user_data_dir: Path):
        prefs = UserPreferences(
            annotation_ai=AnnotationAiPreferences(
                api_keys={"openai": "sk-123", "custom:abc": "k"},
                custom_providers=[
                    AnnotationAiCustomProvider(
                        id="custom:abc",
                        name="My LLM",
                        base_url="https://llm.example/v1",
                    )
                ],
            )
        )
        saved = save_preferences("test-user", prefs)
        loaded = load_preferences("test-user")
        assert loaded == saved
        assert loaded.annotation_ai is not None
        assert loaded.annotation_ai.api_keys == {"openai": "sk-123", "custom:abc": "k"}
        assert loaded.annotation_ai.custom_providers[0].name == "My LLM"

    def test_load_defaults_when_section_absent(self, user_data_dir: Path):
        (user_data_dir / "preferences.toml").write_text(
            tomli_w.dumps({"hidden_views": ["export"]})
        )
        prefs = load_preferences("test-user")
        assert prefs.annotation_ai is not None
        assert prefs.annotation_ai.api_keys == {}
        assert prefs.annotation_ai.custom_providers == []

    def test_merge_replaces_annotation_ai(self):
        current = UserPreferences(
            annotation_ai=AnnotationAiPreferences(api_keys={"openai": "old"})
        )
        update = UserPreferencesUpdate(
            annotation_ai=AnnotationAiPreferences(
                api_keys={"google": "new"},
                custom_providers=[
                    AnnotationAiCustomProvider(
                        id="custom:x", name="X", base_url="https://x/v1"
                    )
                ],
            )
        )
        merged = merge_preferences(current, update)
        assert merged.annotation_ai is not None
        assert merged.annotation_ai.api_keys == {"google": "new"}
        assert merged.annotation_ai.custom_providers[0].id == "custom:x"

    def test_merge_omitting_annotation_ai_keeps_current(self):
        current = UserPreferences(
            annotation_ai=AnnotationAiPreferences(api_keys={"openai": "keep"})
        )
        update = UserPreferencesUpdate(hidden_views=["filter"])
        merged = merge_preferences(current, update)
        assert merged.annotation_ai is not None
        assert merged.annotation_ai.api_keys == {"openai": "keep"}

    def test_rejects_unknown_custom_provider_field(self):
        with pytest.raises(ValidationError):
            AnnotationAiCustomProvider.model_validate(
                {"id": "custom:x", "name": "X", "base_url": "u", "secret": "no"}
            )
