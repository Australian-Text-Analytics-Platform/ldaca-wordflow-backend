"""Stateless annotation preview and provider-discovery application service."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TypeVar

import anyio
import polars as pl
from anyio.to_thread import run_sync as run_sync_in_worker_thread

from ..infrastructure.providers.annotation_ai import (
    AnnotationAiError,
    AnnotationClassOption,
    InferenceConfig,
    annotate_batch,
    list_models,
    resolve_provider_wire,
)
from ..shared.errors import (
    BadGatewayError,
    InvalidInputError,
    NodeNotFoundError,
)
from ..models.annotations import (
    AnnotationConfig,
    AnnotationModelsResource,
    AnnotationPreviewLabel,
    AnnotationPreviewRequest,
    AnnotationPreviewResource,
    AnnotationProvider,
)
from .workspace import WorkspaceService
from .provider_credentials import ProviderCredentialStore

T = TypeVar("T")


class AnnotationService:
    """Own non-durable annotation behavior outside FastAPI routers."""

    def __init__(
        self,
        workspaces: WorkspaceService,
        *,
        limiter: anyio.CapacityLimiter,
        credentials: ProviderCredentialStore,
    ) -> None:
        self._workspaces = workspaces
        self._limiter = limiter
        self._credentials = credentials

    async def models(
        self,
        user_id: str,
        provider: AnnotationProvider,
    ) -> AnnotationModelsResource:
        """List models for a fixed built-in provider without accepting an SSRF URL."""

        api_key = await self._credentials.annotation_credential(user_id, provider)
        try:
            discovered = await list_models(provider, api_key)
        except AnnotationAiError as exc:
            raise BadGatewayError("Annotation provider request failed") from exc
        return AnnotationModelsResource(provider=provider, models=discovered)

    async def preview(
        self,
        *,
        user_id: str,
        workspace_id: str,
        node_id: str,
        request: AnnotationPreviewRequest,
    ) -> AnnotationPreviewResource:
        """Snapshot one source page, release the gate, and classify it once."""

        async with self._workspaces.submission_context(
            user_id,
            workspace_id,
        ) as lease:
            node = lease.workspace.nodes.get(node_id)
            if node is None:
                raise NodeNotFoundError("Node not found")
            texts, total_rows, start = await self._run_io(
                _read_preview_page,
                node.data,
                request.text_column,
                request.page,
                request.page_size,
            )
        api_key = await self._credentials.annotation_credential(user_id, request.provider)
        try:
            labels = await annotate_batch(
                resolve_provider_wire(request.provider),
                request.model,
                api_key,
                request.instruction,
                _class_options(request),
                texts,
                _inference_config(request),
            )
        except AnnotationAiError as exc:
            raise BadGatewayError("Annotation provider request failed") from exc
        return AnnotationPreviewResource(
            node_id=node_id,
            page=request.page,
            page_size=request.page_size,
            total_rows=total_rows,
            labels=[
                AnnotationPreviewLabel(row_index=start + offset, label=label)
                for offset, label in enumerate(labels)
            ],
        )

    async def _run_io(
        self,
        function: Callable[..., T],
        *args: object,
    ) -> T:
        """Run blocking Polars/filesystem work without abandoning a writer."""

        return await run_sync_in_worker_thread(
            partial(function, *args),
            abandon_on_cancel=False,
            limiter=self._limiter,
        )


def _class_options(config: AnnotationConfig) -> list[AnnotationClassOption]:
    return [
        AnnotationClassOption(name=item.name, description=item.description)
        for item in config.classes
    ]


def _inference_config(config: AnnotationConfig) -> InferenceConfig:
    return InferenceConfig(
        temperature=config.temperature,
        reasoning_enabled=config.reasoning_enabled,
        reasoning_effort=config.reasoning_effort,
    )


def _read_preview_page(
    lazyframe: pl.LazyFrame,
    text_column: str,
    page: int,
    page_size: int,
) -> tuple[list[str], int, int]:
    schema = lazyframe.collect_schema()
    if text_column not in schema:
        raise InvalidInputError("Annotation text column does not exist")
    start = (page - 1) * page_size
    count_frame = lazyframe.select(pl.len()).collect()
    total = int(count_frame.item())
    page_frame = (
        lazyframe.select(
            pl.col(text_column)
            .cast(pl.String, strict=False)
            .fill_null("")
            .alias("text")
        )
        .slice(start, page_size)
        .collect()
    )
    values = page_frame["text"].to_list()
    return [str(value) for value in values], total, start


__all__ = ["AnnotationService"]
