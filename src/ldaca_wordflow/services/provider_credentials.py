"""Write-only provider credential API and runtime resolution."""

from __future__ import annotations

from ..models.provider_credentials import (
    AnnotationProvider,
    ProviderCredentialPatch,
    ProviderCredentialSummary,
    StoredProviderCredentials,
)
from ..settings import Settings
from ..shared.errors import ProviderCredentialMissingError
from .user_preferences import UserPreferenceStore


class ProviderCredentialStore:
    """Expose credential operations over the shared per-user persistence owner."""

    def __init__(
        self,
        settings: Settings,
        preferences: UserPreferenceStore,
    ) -> None:
        self._settings = settings
        self._preferences = preferences

    async def summary(self, user_id: str) -> ProviderCredentialSummary:
        return self._summary(await self._preferences.credentials(user_id))

    async def update(
        self,
        user_id: str,
        patch: ProviderCredentialPatch,
    ) -> ProviderCredentialSummary:
        updated = await self._preferences.update_credentials(
            user_id,
            lambda stored: self._apply_patch(stored, patch),
        )
        return self._summary(updated)

    async def clear(self, user_id: str) -> None:
        await self._preferences.clear_credentials(user_id)

    async def annotation_credential(
        self,
        user_id: str,
        provider: AnnotationProvider,
    ) -> str:
        stored = await self._preferences.credentials(user_id)
        credential = getattr(stored.annotation, provider)
        if credential is None:
            raise ProviderCredentialMissingError(
                f"No credential is configured for {provider}"
            )
        return credential.get_secret_value()

    async def data_portal_credential(self, user_id: str) -> str | None:
        stored = await self._preferences.credentials(user_id)
        user_token = stored.data_portal.api_token
        if user_token is not None:
            return user_token.get_secret_value()
        deployment_token = self._settings.ldaca_oni_api_token
        return deployment_token.get_secret_value() if deployment_token else None

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
            annotation={
                "openai": stored.annotation.openai is not None,
                "openrouter": stored.annotation.openrouter is not None,
                "anthropic": stored.annotation.anthropic is not None,
                "google": stored.annotation.google is not None,
            },
            data_portal={
                "user_configured": stored.data_portal.api_token is not None,
                "deployment_configured": self._settings.ldaca_oni_api_token is not None,
            },
        )


__all__ = ["ProviderCredentialStore"]
