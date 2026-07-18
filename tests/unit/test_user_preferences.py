from __future__ import annotations

from pathlib import Path

import anyio
import pytest
import rtoml

from ldaca_wordflow.infrastructure.storage.layout import (
    user_preferences_path,
    user_provider_credentials_path,
)
from ldaca_wordflow.models.provider_credentials import ProviderCredentialPatch
from ldaca_wordflow.models.user_preferences import (
    PREFERENCES_SCHEMA_VERSION,
    UserPreferencesPatch,
)
from ldaca_wordflow.services.provider_credentials import ProviderCredentialStore
from ldaca_wordflow.services.user_preferences import UserPreferenceStore
from ldaca_wordflow.settings import Settings
from ldaca_wordflow.shared.errors import (
    ProviderCredentialMissingError,
    ProviderCredentialsCorruptError,
    UserPreferencesCorruptError,
)


def _stores(
    tmp_path: Path,
) -> tuple[
    UserPreferenceStore,
    ProviderCredentialStore,
    Settings,
]:
    settings = Settings(data_root=tmp_path, multi_user=False)
    preferences = UserPreferenceStore(
        settings,
        io_limiter=anyio.CapacityLimiter(2),
    )
    credentials = ProviderCredentialStore(settings, preferences)
    return preferences, credentials, settings


@pytest.mark.anyio
async def test_missing_preferences_return_schema_versioned_defaults(tmp_path: Path) -> None:
    preferences, _credentials, settings = _stores(tmp_path)

    result = await preferences.get("root")

    assert result.model_dump() == {
        "hidden_views": [],
        "favorite_workspaces": [],
        "default_tokenizer_model": None,
        "analysis_multi_tab_enabled": False,
        "contextual_hints_enabled": True,
    }
    stored = rtoml.loads(
        user_preferences_path(settings, "root").read_text(encoding="utf-8")
    )
    assert stored["schema_version"] == PREFERENCES_SCHEMA_VERSION


@pytest.mark.anyio
async def test_patch_changes_only_explicit_fields_and_accepts_explicit_null(
    tmp_path: Path,
) -> None:
    preferences, _credentials, _settings = _stores(tmp_path)
    await preferences.update(
        "root",
        UserPreferencesPatch(
            hidden_views=["quotation", "quotation", " "],
            default_tokenizer_model=" model ",
        ),
    )

    result = await preferences.update(
        "root",
        UserPreferencesPatch(default_tokenizer_model=None),
    )

    assert result.hidden_views == ["quotation"]
    assert result.default_tokenizer_model is None
    assert result.contextual_hints_enabled is True


@pytest.mark.anyio
async def test_credential_updates_never_touch_sanitized_preferences(
    tmp_path: Path,
) -> None:
    preferences, credentials, settings = _stores(tmp_path)
    await preferences.get("root")

    await credentials.update(
        "root",
        ProviderCredentialPatch(openai_api_key="top-secret"),
    )

    assert "top-secret" not in user_preferences_path(
        settings, "root"
    ).read_text(encoding="utf-8")
    assert "top-secret" in user_provider_credentials_path(
        settings, "root"
    ).read_text(encoding="utf-8")


@pytest.mark.anyio
async def test_preferences_and_credentials_are_isolated_by_user(tmp_path: Path) -> None:
    preferences, credentials, _settings = _stores(tmp_path)

    await preferences.update(
        "user-a",
        UserPreferencesPatch(favorite_workspaces=["workspace-a"]),
    )
    await credentials.update(
        "user-a",
        ProviderCredentialPatch(openai_api_key="user-a-secret"),
    )

    user_b_preferences = await preferences.get("user-b")
    user_b_credentials = await credentials.summary("user-b")

    assert user_b_preferences.favorite_workspaces == []
    assert user_b_credentials.annotation.openai is False
    with pytest.raises(ProviderCredentialMissingError):
        await credentials.annotation_credential("user-b", "openai")


@pytest.mark.anyio
async def test_corrupt_canonical_preference_file_fails_visibly(tmp_path: Path) -> None:
    preferences, _credentials, settings = _stores(tmp_path)
    path = user_preferences_path(settings, "root")
    path.parent.mkdir(parents=True)
    path.write_text("invalid = [", encoding="utf-8")

    with pytest.raises(UserPreferencesCorruptError):
        await preferences.get("root")


@pytest.mark.anyio
async def test_unversioned_preference_file_is_rejected(tmp_path: Path) -> None:
    preferences, _credentials, settings = _stores(tmp_path)
    path = user_preferences_path(settings, "root")
    path.parent.mkdir(parents=True)
    path.write_text(rtoml.dumps({"contextual_hints_enabled": False}), encoding="utf-8")

    with pytest.raises(UserPreferencesCorruptError):
        await preferences.get("root")


@pytest.mark.anyio
async def test_corrupt_canonical_credential_file_fails_visibly(tmp_path: Path) -> None:
    preferences, _credentials, settings = _stores(tmp_path)
    path = user_provider_credentials_path(settings, "root")
    path.parent.mkdir(parents=True)
    path.write_text("invalid = [", encoding="utf-8")

    with pytest.raises(ProviderCredentialsCorruptError):
        await preferences.get("root")


@pytest.mark.anyio
async def test_symlinked_preference_file_is_rejected(tmp_path: Path) -> None:
    preferences, _credentials, settings = _stores(tmp_path)
    path = user_preferences_path(settings, "root")
    path.parent.mkdir(parents=True)
    target = tmp_path / "outside.toml"
    target.write_text("", encoding="utf-8")
    path.symlink_to(target)

    with pytest.raises(UserPreferencesCorruptError):
        await preferences.get("root")


def test_preferences_api_reads_and_patches_current_user(files_test_client) -> None:
    initial = files_test_client.get("/api/preferences")
    assert initial.status_code == 200
    assert initial.json()["contextual_hints_enabled"] is True

    updated = files_test_client.patch(
        "/api/preferences",
        json={
            "favorite_workspaces": ["workspace-a"],
            "contextual_hints_enabled": False,
        },
    )

    assert updated.status_code == 200
    assert updated.json()["favorite_workspaces"] == ["workspace-a"]
    assert updated.json()["contextual_hints_enabled"] is False
    assert updated.json()["analysis_multi_tab_enabled"] is False
