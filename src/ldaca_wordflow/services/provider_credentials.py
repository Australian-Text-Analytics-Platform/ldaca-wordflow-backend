"""Mode-aware provider credential persistence and runtime resolution."""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path

import anyio
import rtoml
from anyio.to_thread import run_sync as run_sync_in_worker_thread
from pydantic import SecretStr, ValidationError

from ..infrastructure.storage.durable_fs import atomic_output_path
from ..infrastructure.storage.layout import user_provider_credentials_path
from ..models.provider_credentials import (
    AnnotationProvider,
    ProviderCredentialPatch,
    ProviderCredentialSummary,
    StoredProviderCredentials,
)
from ..settings import Settings
from ..shared.errors import (
    AccessDeniedError,
    InvalidInputError,
    ProviderCredentialMissingError,
    ProviderCredentialsCorruptError,
)
from .sessions import SINGLE_USER

logger = logging.getLogger(__name__)


class ProviderCredentialStore:
    """Persist local credentials or resolve hosted credentials per request."""

    def __init__(
        self,
        settings: Settings,
        *,
        io_limiter: anyio.CapacityLimiter,
    ) -> None:
        self._settings = settings
        self._io_limiter = io_limiter
        self._lock = anyio.Lock()

    async def summary(self) -> ProviderCredentialSummary:
        if self._settings.multi_user:
            return ProviderCredentialSummary(
                storage="browser",
                annotation=None,
                data_portal={
                    "user_configured": None,
                    "deployment_configured": self._deployment_credential() is not None,
                },
            )
        async with self._lock:
            stored = await self._load()
        return self._summary(stored)

    async def update(
        self,
        patch: ProviderCredentialPatch,
    ) -> ProviderCredentialSummary:
        self._require_backend_storage()
        async with self._lock:
            stored = await self._load()
            updated = self._apply_patch(stored, patch)
            await self._run_io(
                _write_credentials,
                self._path(),
                updated,
            )
        return self._summary(updated)

    async def clear(self) -> None:
        self._require_backend_storage()
        async with self._lock:
            await self._load()
            await self._run_io(
                _write_credentials,
                self._path(),
                StoredProviderCredentials(),
            )

    async def annotation_credential(
        self,
        provider: AnnotationProvider,
        *,
        supplied: SecretStr | str | None = None,
    ) -> str:
        if self._settings.multi_user:
            credential = _secret_value(supplied)
        else:
            self._reject_supplied(supplied)
            async with self._lock:
                stored = await self._load()
            credential = _secret_value(getattr(stored.annotation, provider))
        if credential is None:
            raise ProviderCredentialMissingError(
                f"No credential is configured for {provider}"
            )
        return credential

    async def data_portal_credential(
        self,
        *,
        supplied: SecretStr | str | None = None,
    ) -> str | None:
        if self._settings.multi_user:
            return _secret_value(supplied) or self._deployment_credential()
        self._reject_supplied(supplied)
        async with self._lock:
            stored = await self._load()
        return (
            _secret_value(stored.data_portal.api_token)
            or self._deployment_credential()
        )

    async def _load(self) -> StoredProviderCredentials:
        try:
            result = await self._run_io(_load_credentials, self._path())
        except _InvalidCredentials as exc:
            logger.warning("Invalid provider credentials for single-user root")
            raise ProviderCredentialsCorruptError() from exc
        if not isinstance(result, StoredProviderCredentials):
            raise TypeError("Provider credential reader returned an invalid value")
        return result

    def _path(self) -> Path:
        return user_provider_credentials_path(self._settings, SINGLE_USER.id)

    def _deployment_credential(self) -> str | None:
        return _secret_value(self._settings.ldaca_oni_api_token)

    def _require_backend_storage(self) -> None:
        if self._settings.multi_user:
            raise AccessDeniedError(
                "Provider credentials are owned by the browser in multi-user mode"
            )

    def _reject_supplied(self, supplied: SecretStr | str | None) -> None:
        if supplied is not None:
            raise InvalidInputError(
                "Request credentials are not accepted in single-user mode"
            )

    @staticmethod
    def _apply_patch(
        stored: StoredProviderCredentials,
        patch: ProviderCredentialPatch,
    ) -> StoredProviderCredentials:
        values = stored.model_dump()
        annotation = values["annotation"]
        portal = values["data_portal"]
        for field, provider in (
            ("openai_api_key", "openai"),
            ("openrouter_api_key", "openrouter"),
            ("anthropic_api_key", "anthropic"),
            ("google_api_key", "google"),
        ):
            if field in patch.model_fields_set:
                annotation[provider] = getattr(patch, field)
        if "data_portal_api_token" in patch.model_fields_set:
            portal["api_token"] = patch.data_portal_api_token
        return StoredProviderCredentials.model_validate(values)

    def _summary(
        self,
        stored: StoredProviderCredentials,
    ) -> ProviderCredentialSummary:
        return ProviderCredentialSummary(
            storage="backend",
            annotation={
                "openai": stored.annotation.openai is not None,
                "openrouter": stored.annotation.openrouter is not None,
                "anthropic": stored.annotation.anthropic is not None,
                "google": stored.annotation.google is not None,
            },
            data_portal={
                "user_configured": stored.data_portal.api_token is not None,
                "deployment_configured": self._deployment_credential() is not None,
            },
        )

    async def _run_io(self, function: Callable[..., object], *args: object) -> object:
        return await run_sync_in_worker_thread(
            partial(function, *args),
            abandon_on_cancel=False,
            limiter=self._io_limiter,
        )


class _InvalidCredentials(ValueError):
    pass


def _secret_value(value: SecretStr | str | None) -> str | None:
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    return value


def _load_credentials(path: Path) -> StoredProviderCredentials:
    if not path.exists():
        return StoredProviderCredentials()
    if path.is_symlink() or not path.is_file():
        raise _InvalidCredentials("Stored file must be a regular file")
    try:
        raw = rtoml.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _InvalidCredentials("Stored TOML is invalid") from exc
    if not isinstance(raw, dict):
        raise _InvalidCredentials("Stored TOML must contain a table")
    try:
        return StoredProviderCredentials.model_validate(raw)
    except ValidationError as exc:
        raise _InvalidCredentials("Provider credential schema is invalid") from exc


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
    with atomic_output_path(path) as temporary:
        temporary.chmod(0o600)
        temporary.write_text(rtoml.dumps(payload), encoding="utf-8")
        temporary.chmod(0o600)
    path.chmod(0o600)


__all__ = ["ProviderCredentialStore"]
