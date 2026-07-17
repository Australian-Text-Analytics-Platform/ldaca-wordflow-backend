"""Typed, side-effect-free Result projections for Workspace-owned Analyses."""

from __future__ import annotations

import json
import math
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TypeVar, cast

import anyio
import polars as pl
from anyio.to_thread import run_sync as run_sync_in_worker_thread
from pydantic import BaseModel, ValidationError

from ..analysis.concordance_core import compute_node_concordance_page
from ..analysis.quotation_core import compute_quotation_page
from ..analysis.token_cache import tokens_cache_path
from ..analysis.generated_columns import TOPIC_DISTRIBUTION_COLUMN
from ..domain.workspace import (
    AnalysisArtifactRecord,
    AnalysisRecord,
    ConcordanceAnalysisRequest,
    QuotationAnalysisRequest,
    analysis_input_ids,
)
from ..infrastructure.providers.quotation_client import QuotationProviderClient
from ..models.analysis_results import (
    ANALYSIS_STORED_RESULT_MODELS,
    AnalysisResultQuery,
    ConcordanceResultQuery,
    ConcordanceStoredResult,
    DetachmentStoredResult,
    QuotationResultQuery,
    QuotationStoredResult,
    SequentialStoredResult,
    PagedTableIdentity,
    StoredArtifactIdentity,
    TokenFrequencyStoredResult,
    TopicModelingResultQuery,
    TopicModelingStoredResult,
)
from ..settings import Settings
from ..shared.errors import (
    AnalysisCorruptError,
    AnalysisKindMismatchError,
    ArtifactGoneError,
    InvalidInputError,
    NodeNotFoundError,
)
from ..shared.json_data import JsonData
from ..shared.table_transport import (
    IpcTablePage,
    encode_ipc_stream,
    encode_schema_stream,
    topic_distribution_dtype,
)
from ..shared.topic_types import topic_count_from_storage_dtype
from ..workers.input_snapshots import (
    create_worker_input_snapshot,
    load_snapshot_node,
)
from .analyses import AnalysisService
from .analysis_artifacts import AnalysisArtifactService
from .analysis_preparation import resolve_analysis_quotation_engine
from .response_snapshots import ResponseSnapshot
from .storage_admission import StorageAdmissionService, StorageReservation
from .workspace import WorkspaceLease

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ResultMaterialization:
    """One strict stored Result plus its requested public projection payload."""

    payload: dict[str, JsonData]
    stored: BaseModel


@dataclass(slots=True)
class _QueryInputSnapshot:
    path: Path
    root: Path
    reservation: StorageReservation


class AnalysisResultService:
    """Materialize successful Results without retaining execution input copies."""

    def __init__(
        self,
        analyses: AnalysisService,
        artifacts: AnalysisArtifactService,
        storage_admission: StorageAdmissionService,
        settings: Settings,
        quotation_client: QuotationProviderClient,
        *,
        query_root: Path,
        cache_root: Callable[[str], Path],
        limiter: anyio.CapacityLimiter,
    ) -> None:
        self._analyses = analyses
        self._artifacts = artifacts
        self._storage_admission = storage_admission
        self._settings = settings
        self._quotation_client = quotation_client
        self._query_root = query_root
        self._cache_root = cache_root
        self._limiter = limiter

    async def reconcile(self) -> None:
        """Remove query snapshots abandoned by this deployment's prior process."""

        await self._run_sync(_remove_query_root, self._query_root)

    async def artifact_response_snapshot(
        self,
        user_id: str,
        workspace_id: str,
        analysis_id: str,
        artifact_name: str,
    ) -> tuple[ResponseSnapshot, AnalysisArtifactRecord]:
        """Create a download snapshot without retaining the Workspace gate."""

        async with self._analyses.successful_record_context(
            user_id,
            workspace_id,
            analysis_id,
            allow_closing=True,
        ) as (lease, record):
            return await self._artifacts.response_snapshot(
                lease,
                record,
                artifact_name,
            )

    async def table_response_snapshot(
        self,
        user_id: str,
        workspace_id: str,
        analysis_id: str,
        table_id: str,
    ) -> ResponseSnapshot:
        """Snapshot one declared complete Result table by semantic identity."""

        async with self._analyses.successful_record_context(
            user_id,
            workspace_id,
            analysis_id,
            allow_closing=True,
        ) as (lease, record):
            stored_model = ANALYSIS_STORED_RESULT_MODELS.get(record.request.kind)
            if stored_model is None or record.result_payload is None:
                raise AnalysisCorruptError("Analysis data is corrupt")
            try:
                stored = stored_model.model_validate(record.result_payload)
            except ValidationError as exc:
                raise AnalysisCorruptError("Analysis data is corrupt") from exc
            artifact = None
            if isinstance(stored, TokenFrequencyStoredResult):
                candidates = [item.table for item in stored.tables.nodes]
                if stored.tables.statistics is not None:
                    candidates.append(stored.tables.statistics)
                artifact = next(
                    (item.artifact for item in candidates if item.table_id == table_id),
                    None,
                )
            elif isinstance(stored, SequentialStoredResult):
                if stored.table.table_id == table_id:
                    artifact = stored.table.artifact
            if artifact is None:
                raise ArtifactGoneError("Analysis Result table is unavailable")
            snapshot, _reference = await self._artifacts.response_snapshot(
                lease,
                record,
                artifact.name,
            )
            return snapshot

    async def paged_table_page(
        self,
        user_id: str,
        workspace_id: str,
        analysis_id: str,
        table_id: str,
        *,
        page: int,
        page_size: int,
        sort_by: str | None,
        descending: bool,
    ) -> IpcTablePage:
        """Materialize one independent Arrow page from a declared Result table."""

        snapshot = await self._paged_table_snapshot(
            user_id, workspace_id, analysis_id, table_id
        )
        try:
            return await self._run_sync(
                _paged_artifact_page,
                snapshot.path,
                page,
                page_size,
                sort_by,
                descending,
            )
        finally:
            with anyio.CancelScope(shield=True):
                await snapshot.cleanup()

    async def paged_table_schema(
        self,
        user_id: str,
        workspace_id: str,
        analysis_id: str,
        table_id: str,
    ) -> bytes:
        """Return a zero-row Arrow stream for a declared paged Result table."""

        snapshot = await self._paged_table_snapshot(
            user_id, workspace_id, analysis_id, table_id
        )
        try:
            return await self._run_sync(_paged_artifact_schema, snapshot.path)
        finally:
            with anyio.CancelScope(shield=True):
                await snapshot.cleanup()

    async def _paged_table_snapshot(
        self,
        user_id: str,
        workspace_id: str,
        analysis_id: str,
        table_id: str,
    ) -> ResponseSnapshot:
        async with self._analyses.successful_record_context(
            user_id,
            workspace_id,
            analysis_id,
            allow_closing=True,
        ) as (lease, record):
            stored_model = ANALYSIS_STORED_RESULT_MODELS.get(record.request.kind)
            if stored_model is None or record.result_payload is None:
                raise AnalysisCorruptError("Analysis data is corrupt")
            try:
                stored = stored_model.model_validate(record.result_payload)
            except ValidationError as exc:
                raise AnalysisCorruptError("Analysis data is corrupt") from exc
            artifact = _paged_table_artifact(stored, table_id)
            snapshot, _reference = await self._artifacts.response_snapshot(
                lease,
                record,
                artifact.name,
            )
            return snapshot

    async def query(
        self,
        user_id: str,
        workspace_id: str,
        analysis_id: str,
        query: AnalysisResultQuery | None,
        *,
        allow_closing: bool,
    ) -> ResultMaterialization:
        """Return one typed projection after releasing the Workspace gate."""

        input_snapshot: _QueryInputSnapshot | None = None
        try:
            async with self._analyses.successful_record_context(
                user_id,
                workspace_id,
                analysis_id,
                allow_closing=allow_closing,
            ) as (lease, record):
                kind = record.request.kind
                stored_model = ANALYSIS_STORED_RESULT_MODELS.get(kind)
                if stored_model is None or record.result_payload is None:
                    raise AnalysisCorruptError("Analysis data is corrupt")
                try:
                    stored = stored_model.model_validate(record.result_payload)
                except ValidationError as exc:
                    raise AnalysisCorruptError("Analysis data is corrupt") from exc
                if isinstance(stored, DetachmentStoredResult):
                    if query is not None:
                        raise AnalysisKindMismatchError(
                            "Child Analysis Results do not accept queries"
                        )
                    await self._artifacts.ensure_available(lease, record)
                    payload = cast(
                        dict[str, JsonData],
                        stored.model_dump(mode="json"),
                    )
                    payload["kind"] = kind
                    return ResultMaterialization(payload=payload, stored=stored)
                if isinstance(
                    stored,
                    TokenFrequencyStoredResult | SequentialStoredResult,
                ):
                    if query is not None:
                        raise AnalysisKindMismatchError(
                            "Complete Analysis Results do not accept queries"
                        )
                    await self._artifacts.ensure_available(lease, record)
                    payload = cast(
                        dict[str, JsonData],
                        stored.model_dump(mode="json"),
                    )
                    payload["kind"] = kind
                    return ResultMaterialization(payload=payload, stored=stored)
                effective_query = query or _default_query(kind)
                if effective_query.kind != kind:
                    raise AnalysisKindMismatchError(
                        "Result query kind does not match the Analysis"
                    )
                request = record.request.model_copy(deep=True)
                await self._artifacts.ensure_available(lease, record)

                if isinstance(
                    effective_query,
                    ConcordanceResultQuery | QuotationResultQuery,
                ):
                    input_snapshot = await self._create_query_snapshot(lease, record)

            if isinstance(effective_query, TopicModelingResultQuery):
                return ResultMaterialization(
                    payload=await self._run_sync(
                        _query_topics,
                        TopicModelingStoredResult.model_validate(stored),
                        effective_query,
                    ),
                    stored=stored,
                )
            if isinstance(effective_query, ConcordanceResultQuery) and isinstance(
                request,
                ConcordanceAnalysisRequest,
            ):
                if input_snapshot is None:
                    raise RuntimeError("Concordance query input was not prepared")
                return ResultMaterialization(
                    payload=await self._run_sync(
                        _query_concordance_snapshot,
                        input_snapshot.path,
                        request,
                        effective_query,
                        str(tokens_cache_path(self._cache_root(user_id))),
                        ConcordanceStoredResult.model_validate(stored),
                    ),
                    stored=stored,
                )
            if isinstance(effective_query, QuotationResultQuery) and isinstance(
                request,
                QuotationAnalysisRequest,
            ):
                if input_snapshot is None:
                    raise RuntimeError("Quotation query input was not prepared")
                snapshot_node = await self._run_sync(
                    load_snapshot_node,
                    input_snapshot.path,
                    str(request.node_id),
                )
                page = await compute_quotation_page(
                    snapshot_node.to_node(),
                    request.column,
                    resolve_analysis_quotation_engine(request, self._settings),
                    page=effective_query.page,
                    page_size=effective_query.page_size,
                    sort_by=effective_query.sort_by,
                    descending=effective_query.descending,
                    quotation_service_max_batch_size=(
                        self._settings.quotation_service_max_batch_size
                    ),
                    quotation_service_timeout=(
                        self._settings.quotation_service_timeout
                    ),
                    extract_remote_fn=self._quotation_client.extract,
                    run_blocking=self._run_sync,
                )
                payload = cast(
                    dict[str, JsonData],
                    QuotationStoredResult.model_validate(page).model_dump(mode="json"),
                )
                payload["kind"] = "quotation"
                payload["query"] = cast(
                    JsonData,
                    effective_query.model_dump(mode="json"),
                )
                return ResultMaterialization(payload=payload, stored=stored)
            raise AnalysisKindMismatchError(
                "Result query kind does not match the Analysis"
            )
        finally:
            with anyio.CancelScope(shield=True):
                if input_snapshot is not None:
                    await self._cleanup_query_snapshot(input_snapshot)

    async def _create_query_snapshot(
        self,
        lease: WorkspaceLease,
        record: AnalysisRecord,
    ) -> _QueryInputSnapshot:
        reservation = await self._storage_admission.acquire_transient(
            self._settings.max_analysis_storage_bytes
        )
        root = self._query_root / f"query-{uuid.uuid4()}"
        snapshot = root / "input"
        try:
            await self._run_sync(
                partial(
                    create_worker_input_snapshot,
                    workspace_id=lease.workspace.id,
                    node_ids=[str(item) for item in _request_node_ids(record)],
                    workspace=lease.workspace,
                    workspace_data_dir=lease.path / "data",
                    snapshot_dir=snapshot,
                    max_snapshot_bytes=self._settings.max_analysis_storage_bytes,
                )
            )
            return _QueryInputSnapshot(snapshot, root, reservation)
        except BaseException:
            with anyio.CancelScope(shield=True):
                await self._run_sync(_remove_query_root, root)
                await reservation.release()
            raise

    async def _cleanup_query_snapshot(self, snapshot: _QueryInputSnapshot) -> None:
        try:
            await self._run_sync(_remove_query_root, snapshot.root)
        finally:
            await snapshot.reservation.release()

    async def _run_sync(
        self,
        function: Callable[..., T],
        *args: object,
    ) -> T:
        return await run_sync_in_worker_thread(
            partial(function, *args),
            abandon_on_cancel=False,
            limiter=self._limiter,
        )


def _default_query(kind: str) -> AnalysisResultQuery:
    if kind == "topic_modeling":
        return TopicModelingResultQuery()
    if kind == "concordance":
        return ConcordanceResultQuery()
    if kind == "quotation":
        return QuotationResultQuery()
    raise AnalysisCorruptError("Analysis data is corrupt")


def _request_node_ids(record: AnalysisRecord) -> tuple[uuid.UUID, ...]:
    return analysis_input_ids(record.request)


def _paged_table_artifact(
    stored: BaseModel,
    table_id: str,
) -> StoredArtifactIdentity:
    if isinstance(stored, TopicModelingStoredResult):
        table = next(
            (
                node.assignments
                for node in stored.artifacts.nodes
                if node.assignments.table_id == table_id
            ),
            None,
        )
        if isinstance(table, PagedTableIdentity):
            return table.artifact
    raise ArtifactGoneError("Analysis Result table is unavailable")


def _paged_artifact_lazyframe(path: Path) -> pl.LazyFrame:
    return pl.scan_parquet(path)


def _paged_artifact_page(
    path: Path,
    page: int,
    page_size: int,
    sort_by: str | None,
    descending: bool,
) -> IpcTablePage:
    if page < 1 or page_size < 1:
        raise InvalidInputError("Page and page size must be positive")
    lazyframe = _paged_artifact_lazyframe(path)
    schema = lazyframe.collect_schema()
    topic_count = _topic_distribution_topic_count(schema)
    if sort_by is not None:
        if sort_by not in schema:
            raise InvalidInputError("Table sort column not found")
        lazyframe = lazyframe.sort(sort_by, descending=descending)
    frame = lazyframe.slice((page - 1) * page_size, page_size + 1).collect()
    has_next = len(frame) > page_size
    frame = _apply_topic_extension(frame.head(page_size), topic_count)
    return IpcTablePage(content=encode_ipc_stream(frame), has_next=has_next)


def _paged_artifact_schema(path: Path) -> bytes:
    schema = _paged_artifact_lazyframe(path).collect_schema()
    if TOPIC_DISTRIBUTION_COLUMN in schema:
        schema[TOPIC_DISTRIBUTION_COLUMN] = topic_distribution_dtype(
            _topic_distribution_topic_count(schema)
        )
    return encode_schema_stream(schema)


def _topic_distribution_topic_count(schema: pl.Schema) -> int:
    if TOPIC_DISTRIBUTION_COLUMN not in schema:
        return 0
    try:
        return topic_count_from_storage_dtype(schema[TOPIC_DISTRIBUTION_COLUMN])
    except ValueError as exc:
        raise AnalysisCorruptError(
            "Analysis Topic Distribution artifact has an invalid schema"
        ) from exc


def _apply_topic_extension(frame: pl.DataFrame, topic_count: int) -> pl.DataFrame:
    if TOPIC_DISTRIBUTION_COLUMN not in frame:
        return frame
    return frame.with_columns(
        pl.Series(
            TOPIC_DISTRIBUTION_COLUMN,
            frame[TOPIC_DISTRIBUTION_COLUMN].to_list(),
            dtype=topic_distribution_dtype(topic_count),
        )
    )


def _sort_and_page(
    rows: list[dict[str, JsonData]],
    *,
    page: int,
    page_size: int,
    sort_by: str | None,
    descending: bool,
    columns: set[str],
) -> tuple[list[dict[str, JsonData]], dict[str, JsonData]]:
    if sort_by is not None:
        if sort_by not in columns:
            raise InvalidInputError("Result sort column not found")
        present = [row for row in rows if row.get(sort_by) is not None]
        missing = [row for row in rows if row.get(sort_by) is None]
        present.sort(
            key=lambda row: _json_sort_key(row[sort_by]),
            reverse=descending,
        )
        rows = [*present, *missing]
    total = len(rows)
    start = (page - 1) * page_size
    return rows[start : start + page_size], {
        "page": page,
        "page_size": page_size,
        "total_rows": total,
        "total_pages": math.ceil(total / page_size) if total else 0,
    }


def _json_sort_key(value: JsonData) -> tuple[int, int | float | str]:
    """Order one non-null JSON scalar without stringifying numeric values."""

    if isinstance(value, bool):
        return 0, int(value)
    if isinstance(value, int | float):
        return 1, value
    if isinstance(value, str):
        return 2, value
    return 3, json.dumps(value, sort_keys=True, separators=(",", ":"))


def _query_topics(
    stored: TopicModelingStoredResult,
    query: TopicModelingResultQuery,
) -> dict[str, JsonData]:
    payload = cast(dict[str, JsonData], stored.model_dump(mode="json"))
    rows = [
        cast(dict[str, JsonData], topic.model_dump(mode="json"))
        for topic in stored.topics
    ]
    if query.topic_ids is not None:
        selected = set(query.topic_ids)
        rows = [row for row in rows if row.get("id") in selected]
    columns = (
        set(rows[0])
        if rows
        else set(type(stored.topics[0]).model_fields)
        if stored.topics
        else set()
    )
    page_rows, pagination = _sort_and_page(
        rows,
        page=query.page,
        page_size=query.page_size,
        sort_by=query.sort_by,
        descending=query.descending,
        columns=columns,
    )
    payload["kind"] = "topic_modeling"
    payload["topics"] = cast(JsonData, page_rows)
    payload["pagination"] = pagination
    payload["query"] = cast(JsonData, query.model_dump(mode="json"))
    return payload


def _query_concordance_snapshot(
    snapshot_dir: Path,
    request: ConcordanceAnalysisRequest,
    query: ConcordanceResultQuery,
    token_cache: str,
    stored: ConcordanceStoredResult,
) -> dict[str, JsonData]:
    node_ids = [query.node_id] if query.node_id is not None else request.node_ids
    if any(node_id not in request.node_ids for node_id in node_ids):
        raise NodeNotFoundError("Analysis Result Data Block not found")
    request_payload = request.model_dump(mode="json", exclude={"kind"})
    data: dict[str, JsonData] = {}
    for node_id in node_ids:
        snapshot = load_snapshot_node(snapshot_dir, str(node_id))
        node = snapshot.to_node()
        column = request.node_columns[node_id]
        data[str(node_id)] = cast(
            JsonData,
            compute_node_concordance_page(
                {
                    "lf": snapshot.data,
                    "column": column,
                    "label": snapshot.name,
                    "tokenization_column": node.find_tokenization_column(column),
                    "node": node,
                    "token_cache_path": token_cache,
                },
                request_payload,
                page=query.page,
                page_size=query.page_size,
                sort_by=query.sort_by,
                descending=query.descending,
            ),
        )
    payload = cast(dict[str, JsonData], stored.model_dump(mode="json"))
    payload["kind"] = "concordance"
    payload["data"] = data
    payload["query"] = cast(JsonData, query.model_dump(mode="json"))
    return payload


def _remove_query_root(root: Path) -> None:
    try:
        if root.is_dir() and not root.is_symlink():
            shutil.rmtree(root)
        elif root.exists() or root.is_symlink():
            root.unlink()
    except FileNotFoundError:
        return


__all__ = ["AnalysisResultService", "ResultMaterialization"]
