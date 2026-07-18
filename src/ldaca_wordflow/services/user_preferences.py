"""Per-user account preferences and provider-credential persistence."""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path

import anyio
import rtoml
from anyio.to_thread import run_sync as run_sync_in_worker_thread
from pydantic import ValidationError

from ..infrastructure.storage.durable_fs import atomic_output_path
from ..infrastructure.storage.layout import (
    user_preferences_path,
    user_provider_credentials_path,
)
from ..models.provider_credentials import StoredProviderCredentials
from ..models.user_preferences import (
    PREFERENCES_SCHEMA_VERSION,
    StoredUserPreferences,
    UserPreferences,
    UserPreferencesPatch,
)
from ..settings import Settings
from ..shared.errors import (
    ProviderCredentialsCorruptError,
    UserPreferencesCorruptError,
)

logger = logging.getLogger(__name__)

class UserPreferenceStore:
    """Own strict preference and credential files under one per-user lock."""

    def __init__(
        self,
        settings: Settings,
        *,
        io_limiter: anyio.CapacityLimiter,
    ) -> None:
        self._settings = settings
        self._io_limiter = io_limiter
        self._locks: dict[str, anyio.Lock] = {}
        self._locks_guard = anyio.Lock()

    async def get(self, user_id: str) -> UserPreferences:
        async with await self._user_lock(user_id):
            preferences, _credentials = await self._load(user_id)
        return UserPreferences.model_validate(
            preferences.model_dump(exclude={"schema_version"})
        )

    async def update(
        self,
        user_id: str,
        patch: UserPreferencesPatch,
    ) -> UserPreferences:
        async with await self._user_lock(user_id):
            preferences, _credentials = await self._load(user_id)
            values = preferences.model_dump()
            for field in patch.model_fields_set:
                values[field] = getattr(patch, field)
            updated = StoredUserPreferences.model_validate(values)
            await self._run_io(
                _write_preferences,
                user_preferences_path(self._settings, user_id),
                updated,
            )
        return UserPreferences.model_validate(
            updated.model_dump(exclude={"schema_version"})
        )

    async def credentials(self, user_id: str) -> StoredProviderCredentials:
        async with await self._user_lock(user_id):
            _preferences, credentials = await self._load(user_id)
        return credentials

    async def update_credentials(
        self,
        user_id: str,
        transform: Callable[[StoredProviderCredentials], StoredProviderCredentials],
    ) -> StoredProviderCredentials:
        async with await self._user_lock(user_id):
            _preferences, credentials = await self._load(user_id)
            updated = transform(credentials)
            await self._run_io(
                _write_credentials,
                user_provider_credentials_path(self._settings, user_id),
                updated,
            )
        return updated

    async def clear_credentials(self, user_id: str) -> None:
        async with await self._user_lock(user_id):
            await self._load(user_id)
            await self._run_io(
                _write_credentials,
                user_provider_credentials_path(self._settings, user_id),
                StoredProviderCredentials(),
            )

    async def _load(
        self,
        user_id: str,
    ) -> tuple[StoredUserPreferences, StoredProviderCredentials]:
        preference_path = user_preferences_path(self._settings, user_id)
        credential_path = user_provider_credentials_path(self._settings, user_id)
        try:
            result = await self._run_io(
                _load_files,
                preference_path,
                credential_path,
            )
        except _InvalidPreferences as exc:
            logger.warning("Invalid user preferences for user %s", user_id)
            raise UserPreferencesCorruptError() from exc
        except _InvalidCredentials as exc:
            logger.warning("Invalid provider credentials for user %s", user_id)
            raise ProviderCredentialsCorruptError() from exc
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("User preference reader returned an invalid value")
        preferences, credentials = result
        if not isinstance(preferences, StoredUserPreferences) or not isinstance(
            credentials, StoredProviderCredentials
        ):
            raise TypeError("User preference reader returned an invalid value")
        return preferences, credentials

    async def _user_lock(self, user_id: str) -> anyio.Lock:
        async with self._locks_guard:
            return self._locks.setdefault(user_id, anyio.Lock())

    async def _run_io(self, function: Callable[..., object], *args: object) -> object:
        return await run_sync_in_worker_thread(
            partial(function, *args),
            abandon_on_cancel=False,
            limiter=self._io_limiter,
        )


class _InvalidPreferences(ValueError):
    pass


class _InvalidCredentials(ValueError):
    pass


def _read_toml(path: Path, error_type: type[ValueError]) -> dict[str, object] | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise error_type("Stored file must be a regular file")
    try:
        raw = rtoml.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise error_type("Stored TOML is invalid") from exc
    if not isinstance(raw, dict):
        raise error_type("Stored TOML must contain a table")
    return raw


def _load_files(
    preference_path: Path,
    credential_path: Path,
) -> tuple[StoredUserPreferences, StoredProviderCredentials]:
    preference_raw = _read_toml(preference_path, _InvalidPreferences)
    credential_raw = _read_toml(credential_path, _InvalidCredentials)

    try:
        credentials = (
            StoredProviderCredentials.model_validate(credential_raw)
            if credential_raw is not None
            else StoredProviderCredentials()
        )
    except ValidationError as exc:
        raise _InvalidCredentials("Provider credential schema is invalid") from exc

    if preference_raw is None:
        preferences = StoredUserPreferences()
        _write_preferences(preference_path, preferences)
        return preferences, credentials

    if preference_raw.get("schema_version") != PREFERENCES_SCHEMA_VERSION:
        raise _InvalidPreferences("Preference schema version is unsupported")
    try:
        preferences = StoredUserPreferences.model_validate(preference_raw)
    except ValidationError as exc:
        raise _InvalidPreferences("Preference schema is invalid") from exc
    return preferences, credentials


def _write_preferences(path: Path, preferences: StoredUserPreferences) -> None:
    payload = preferences.model_dump(mode="json")
    _write_private_toml(path, payload)


def _write_credentials(path: Path, credentials: StoredProviderCredentials) -> None:
    payload: dict[str, object] = {
        "annotation": {
            provider: secret.get_secret_value()
            for provider in ("openai", "openrouter", "anthropic", "google")
            if (secret := getattr(credentials.annotation, provider)) is not None
        },
        "data_portal": (
            {"api_token": credentials.data_portal.api_token.get_secret_value()}
            if credentials.data_portal.api_token is not None
            else {}
        ),
    }
    _write_private_toml(path, payload)


def _write_private_toml(path: Path, payload: dict[str, object]) -> None:
    with atomic_output_path(path) as temporary:
        temporary.chmod(0o600)
        temporary.write_text(rtoml.dumps(payload), encoding="utf-8")
        temporary.chmod(0o600)
    path.chmod(0o600)


__all__ = ["UserPreferenceStore"]
