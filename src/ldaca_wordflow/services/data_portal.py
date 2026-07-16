"""LDaCA Data Portal reads and process-isolated import execution."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TypeVar

import anyio
import httpx
from anyio.to_thread import run_sync as run_sync_in_worker_thread
from pydantic import SecretStr

from ..domain import (
    DataPortalUserFileImportRequest,
    DataPortalUserFileImportResult,
)
from ..shared.errors import (
    BadGatewayError,
    InternalServiceError,
    InvalidInputError,
    ResourceTooLargeError,
)
from ..infrastructure.providers.oni import OniClient, extract_ldaca_identifier
from ..models.data_sources import (
    DataPortalImportSubmitRequest,
    DataPortalRecord,
    DataPortalSearchRequest,
    DataPortalSearchResource,
)
from ..settings import Settings
from ..workers.data_portal import data_portal_import_process
from ..infrastructure.storage.layout import validate_display_name
from .user_files import UserFileStore
from .user_file_import_execution_types import UserFileImportKey
from .user_file_import_executor import UserFileImportProcessExecutor

T = TypeVar("T")
ProgressReporter = Callable[[object], Awaitable[None]]
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DataPortalImportExecution:
    function: Callable[..., object]
    kwargs: Mapping[str, object]
    storage_roots: tuple[str, ...]
    cache_path: Path


class DataPortalService:
    """Own the configured portal boundary without workspace or API dependencies."""

    def __init__(
        self,
        settings: Settings,
        files: UserFileStore,
        *,
        limiter: anyio.CapacityLimiter,
    ) -> None:
        self._settings = settings
        self._files = files
        self._limiter = limiter
        self._http_client = httpx.AsyncClient(
            base_url=settings.ldaca_oni_api_base_url.rstrip("/"),
            timeout=settings.ldaca_oni_timeout,
            follow_redirects=True,
        )

    async def close(self) -> None:
        """Close the runtime-owned portal connection pool."""

        await self._http_client.aclose()

    async def search(
        self, request: DataPortalSearchRequest
    ) -> DataPortalSearchResource:
        """Run a normalized portal search with one-based API pagination."""

        client = self._client(_secret(request.api_token))
        try:
            records, total = await client.search(
                method=request.method,
                query=request.query,
                limit=request.page_size,
                offset=(request.page - 1) * request.page_size,
            )
        except (httpx.HTTPError, ValueError) as exc:
            raise BadGatewayError("Data Portal search failed") from exc
        return DataPortalSearchResource(
            page=request.page,
            page_size=request.page_size,
            total=total,
            items=[DataPortalRecord.model_validate(record) for record in records],
        )

    async def featured(
        self,
        api_token: SecretStr | None,
    ) -> DataPortalSearchResource:
        """Read configured featured collections outside all workspace gates."""

        client = self._client(_secret(api_token))
        try:
            records = await client.featured_collections(
                list(self._settings.ldaca_oni_featured_collection_ids)
            )
        except (httpx.HTTPError, ValueError) as exc:
            raise BadGatewayError("Data Portal featured collections failed") from exc
        items = [DataPortalRecord.model_validate(record) for record in records]
        return DataPortalSearchResource(
            page=1,
            page_size=max(1, len(items)),
            total=len(items),
            items=items,
        )

    async def prepare_import(
        self,
        user_id: str,
        import_id: str,
        request: DataPortalImportSubmitRequest,
    ) -> tuple[DataPortalUserFileImportRequest, DataPortalImportExecution]:
        """Normalize one request and build its process-private invocation."""

        identifier = extract_ldaca_identifier(request.identifier)
        if identifier is None:
            raise InvalidInputError(
                "Data Portal import requires an ARCP identifier or portal URL"
            )
        if request.name is not None:
            valid, reason = validate_display_name(request.name)
            if not valid:
                raise InvalidInputError(f"Invalid import name: {reason}")
        staging = await self._files.prepare_import_staging(user_id, import_id)
        cache = (
            self._settings.get_data_root()
            / ".user-file-import-work"
            / import_id
        )
        try:
            name = request.name.strip() if request.name else None
            return (
                DataPortalUserFileImportRequest(
                    identifier=identifier,
                    name=name,
                ),
                DataPortalImportExecution(
                    function=data_portal_import_process,
                    kwargs={
                        "identifier": identifier,
                        "requested_name": name,
                        "api_base_url": self._settings.ldaca_oni_api_base_url,
                        "api_token": _secret(request.api_token)
                        or _secret(self._settings.ldaca_oni_api_token),
                        "timeout": self._settings.ldaca_oni_timeout,
                        "download_concurrency": (
                            self._settings.ldaca_oni_download_concurrency
                        ),
                        "staging_dir": str(staging),
                        "cache_dir": str(cache),
                        "max_output_bytes": (
                            self._settings.max_user_file_import_bytes
                        ),
                    },
                    storage_roots=(str(staging), str(cache)),
                    cache_path=cache,
                ),
            )
        except BaseException:
            with anyio.CancelScope(shield=True):
                try:
                    await self._files.cleanup_import_staging(user_id, import_id)
                finally:
                    await self._run_io(_remove_tree_if_present, cache)
            raise

    async def execute_import(
        self,
        key: UserFileImportKey,
        execution: DataPortalImportExecution,
        executor: UserFileImportProcessExecutor,
        report_progress: ProgressReporter,
    ) -> DataPortalUserFileImportResult:
        result = await executor.execute(
            key,
            execution.function,
            execution.kwargs,
            report_progress,
            storage_roots=execution.storage_roots,
            max_storage_bytes=self._settings.max_user_file_import_bytes,
            max_storage_files=self._settings.max_user_file_import_files,
        )
        if not isinstance(result, dict):
            raise InternalServiceError("Portal import worker returned invalid output")
        validated = DataPortalUserFileImportResult.model_validate(result)
        if validated.bytes_written > self._settings.max_user_file_import_bytes:
            raise ResourceTooLargeError(
                "Data Portal import exceeds the import storage limit"
            )
        return validated

    async def publish_import(
        self,
        user_id: str,
        import_id: str,
        result: DataPortalUserFileImportResult,
    ) -> None:
        installed = await self._files.install_import_staging(
            user_id,
            import_id,
            result.destination_path,
        )
        if installed != result.destination_path:
            raise RuntimeError("Portal import installed at an unexpected path")

    async def cleanup_import(
        self,
        user_id: str,
        import_id: str,
        execution: DataPortalImportExecution | None,
    ) -> None:
        cache = (
            execution.cache_path
            if execution is not None
            else self._settings.get_data_root()
            / ".user-file-import-work"
            / import_id
        )
        try:
            await self._files.cleanup_import_staging(user_id, import_id)
        finally:
            await self._run_io(_remove_tree_if_present, cache)

    async def reconcile_transient_storage(self, active_import_ids: set[str]) -> None:
        await self._run_io(
            _reconcile_import_work,
            self._settings.get_data_root() / ".user-file-import-work",
            active_import_ids,
        )

    def _client(self, token: str | None) -> OniClient:
        return OniClient(
            self._http_client,
            token=token or _secret(self._settings.ldaca_oni_api_token),
        )

    async def _run_io(
        self,
        function: Callable[..., T],
        *args: object,
    ) -> T:
        return await run_sync_in_worker_thread(
            partial(function, *args),
            abandon_on_cancel=False,
            limiter=self._limiter,
        )


def _secret(value: SecretStr | None) -> str | None:
    if value is None:
        return None
    resolved = value.get_secret_value().strip()
    return resolved or None


def _remove_tree_if_present(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return


def _reconcile_import_work(root: Path, active_import_ids: set[str]) -> None:
    if not root.is_dir() or root.is_symlink():
        return
    for child in root.iterdir():
        if child.is_symlink() or child.name in active_import_ids:
            continue
        try:
            if child.is_dir():
                shutil.rmtree(child)
            elif child.is_file():
                child.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            logger.exception(
                "Could not reconcile Data Portal import cache path=%s",
                child,
            )


__all__ = ["DataPortalImportExecution", "DataPortalService"]
