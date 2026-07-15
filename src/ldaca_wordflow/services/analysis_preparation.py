"""Prepare immutable process inputs for Workspace-owned root Analyses."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from functools import partial
from pathlib import Path

import anyio
from anyio.to_thread import run_sync as run_sync_in_worker_thread

from ..analysis.generated_columns import (
    CONC_EXTRACTION_COLUMN,
    DETACHABLE_CONCORDANCE_COLUMNS,
    QUOTE_COLUMN_NAMES,
    QUOTE_EXTRACTION_COLUMN,
)
from ..analysis.request_normalization import sanitize_stop_words
from ..analysis.token_cache import tokens_cache_path
from ..domain.workspace import (
    AnalysisRecord,
    AnnotationAnalysisRequest,
    ConcordanceDetachmentAnalysisRequest,
    ConcordanceDispersionDetachmentAnalysisRequest,
    ConcordanceAnalysisRequest,
    Node,
    QuotationAnalysisRequest,
    QuotationDetachmentAnalysisRequest,
    SequentialAnalysisRequest,
    TokenFrequencyAnalysisRequest,
    TopicModelingAnalysisRequest,
    Workspace,
)
from ..infrastructure.storage.embedding_cache import embeddings_cache_path
from ..models.quotation import QuotationEngineType, ResolvedQuotationEngine
from ..settings import Settings
from ..shared.errors import InvalidInputError
from ..workers.entrypoints import (
    annotation_process,
    concordance_process,
    concordance_detachment_process,
    concordance_dispersion_detachment_process,
    quotation_process,
    quotation_detachment_process,
    sequential_process,
    token_frequency_process,
    topic_modeling_process,
)
from ..workers.input_snapshots import create_worker_input_snapshot
from .analysis_execution_types import AnalysisInvocation
from .workspace import WorkspaceLease


class AnalysisExecutionPreparer:
    """Build one process invocation while the Workspace gate is held."""

    def __init__(
        self,
        settings: Settings,
        *,
        limiter: anyio.CapacityLimiter,
        cache_root: Callable[[str], Path],
    ) -> None:
        self._settings = settings
        self._limiter = limiter
        self._cache_root = cache_root

    async def prepare(
        self,
        lease: WorkspaceLease,
        record: AnalysisRecord,
        credential: str | None,
        *,
        user_id: str,
    ) -> AnalysisInvocation:
        """Snapshot selected Data Blocks and return only private immutable inputs."""

        analysis_id = str(record.id)
        workspace = lease.workspace
        node_ids = _request_node_ids(record)
        analysis_dir = lease.path / "analyses" / analysis_id
        execution_dir = analysis_dir / ".execution"
        snapshot_dir = execution_dir / "input"
        artifact_dir = execution_dir / "output"
        try:
            await run_sync_in_worker_thread(
                partial(
                    create_worker_input_snapshot,
                    workspace_id=workspace.id,
                    node_ids=node_ids,
                    workspace=workspace,
                    workspace_data_dir=lease.path / "data",
                    snapshot_dir=snapshot_dir,
                    max_snapshot_bytes=self._settings.max_analysis_storage_bytes,
                ),
                abandon_on_cancel=False,
                limiter=self._limiter,
            )
            return self._invocation(
                record,
                user_id=user_id,
                workspace_id=workspace.id,
                workspace=workspace,
                snapshot_dir=snapshot_dir,
                artifact_dir=artifact_dir,
                credential=credential,
            )
        except BaseException:
            with anyio.CancelScope(shield=True):
                await run_sync_in_worker_thread(
                    _remove_execution_staging,
                    execution_dir,
                    abandon_on_cancel=False,
                    limiter=self._limiter,
                )
            raise

    def _invocation(
        self,
        record: AnalysisRecord,
        *,
        user_id: str,
        workspace_id: str,
        workspace: Workspace,
        snapshot_dir: Path,
        artifact_dir: Path,
        credential: str | None,
    ) -> AnalysisInvocation:
        request = record.request
        nodes = workspace.nodes
        common: dict[str, object] = {
            "user_id": user_id,
            "workspace_id": workspace_id,
            "input_snapshot_dir": str(snapshot_dir),
        }

        def owned(
            function: Callable[..., object],
            kwargs: dict[str, object],
        ) -> AnalysisInvocation:
            return AnalysisInvocation(
                function=function,
                kwargs=kwargs,
                storage_roots=(str(snapshot_dir.parent),),
                max_storage_bytes=self._settings.max_analysis_storage_bytes,
                max_storage_files=self._settings.max_analysis_storage_files,
            )

        if isinstance(request, AnnotationAnalysisRequest):
            if not credential:
                raise InvalidInputError("Annotation credential is unavailable")
            return owned(
                annotation_process,
                {
                    "input_snapshot_dir": str(snapshot_dir),
                    "output_dir": str(artifact_dir),
                    "request_payload": request.model_dump(mode="json"),
                    "api_key": credential,
                },
            )

        if isinstance(request, TokenFrequencyAnalysisRequest):
            node_ids = [str(node_id) for node_id in request.node_ids]
            node_columns = {
                str(node_id): column for node_id, column in request.node_columns.items()
            }
            tokenizer_models = _resolve_tokenizer_models(
                nodes,
                node_ids=node_ids,
                node_columns=node_columns,
                requested={
                    str(node_id): model
                    for node_id, model in request.node_tokenizer_models.items()
                },
            )
            return owned(
                token_frequency_process,
                {
                    **common,
                    "node_ids": node_ids,
                    "node_columns": node_columns,
                    "artifact_dir": str(artifact_dir),
                    "artifact_prefix": "token_frequency",
                    "token_limit": request.token_limit,
                    "stop_words": sanitize_stop_words(request.stop_words),
                    "node_tokenizer_models": tokenizer_models,
                    "token_cache_path": str(
                        tokens_cache_path(self._cache_root(user_id))
                    ),
                },
            )

        if isinstance(request, TopicModelingAnalysisRequest):
            node_ids = [str(node_id) for node_id in request.node_ids]
            node_columns = {
                str(node_id): column for node_id, column in request.node_columns.items()
            }
            return owned(
                topic_modeling_process,
                {
                    **common,
                    "node_infos": [
                        {"node_id": node_id, "text_column": node_columns[node_id]}
                        for node_id in node_ids
                    ],
                    "artifact_dir": str(artifact_dir),
                    "artifact_prefix": "topic_modeling",
                    "min_topic_size": request.min_topic_size,
                    "random_seed": request.random_seed,
                    "representative_words_count": request.representative_words_count,
                    "sample_fractions": request.sample_fractions,
                    "embedding_cache_path": str(
                        embeddings_cache_path(self._cache_root(user_id))
                    ),
                },
            )

        if isinstance(request, ConcordanceAnalysisRequest):
            payload = request.model_dump(mode="json", exclude={"kind"})
            return owned(
                concordance_process,
                {
                    **common,
                    "request_payload": payload,
                    "token_cache_path": str(
                        tokens_cache_path(self._cache_root(user_id))
                    ),
                },
            )

        if isinstance(request, QuotationAnalysisRequest):
            payload = request.model_dump(mode="json", exclude={"kind", "node_id"})
            payload["engine"] = resolve_analysis_quotation_engine(
                request,
                self._settings,
            ).model_dump(mode="json")
            payload["context_length"] = 20
            return owned(
                quotation_process,
                {
                    **common,
                    "node_id": str(request.node_id),
                    "request_payload": payload,
                    "quotation_service_max_batch_size": self._settings.quotation_service_max_batch_size,
                    "quotation_service_timeout": self._settings.quotation_service_timeout,
                },
            )

        if isinstance(request, SequentialAnalysisRequest):
            return owned(
                sequential_process,
                {
                    **common,
                    "node_id": str(request.node_id),
                    "request_payload": request.model_dump(
                        mode="json",
                        exclude={"kind", "node_id"},
                    ),
                },
            )

        parent = (
            workspace.analyses.get(str(record.parent_analysis_id))
            if record.parent_analysis_id is not None
            else None
        )
        parent_request = parent.request if parent is not None else None
        if isinstance(request, ConcordanceDetachmentAnalysisRequest) and isinstance(
            parent_request,
            ConcordanceAnalysisRequest,
        ):
            source = parent_request
            column = source.node_columns[request.node_id]
            generated = [
                item
                for item in request.selected_columns
                if item in DETACHABLE_CONCORDANCE_COLUMNS
            ]
            return owned(
                concordance_detachment_process,
                {
                    "workspace_dir": str(artifact_dir),
                    "input_snapshot_dir": str(snapshot_dir),
                    "parent_node_id": str(request.node_id),
                    "document_column": column,
                    "search_word": source.search_word,
                    "num_left_tokens": source.num_left_tokens,
                    "num_right_tokens": source.num_right_tokens,
                    "regex": source.regex,
                    "whole_word": source.whole_word,
                    "case_sensitive": source.case_sensitive,
                    "search_mode": source.search_mode,
                    "new_node_name": request.name
                    or f"Concordance {str(request.node_id)[:8]}",
                    "include_document_column": column in request.selected_columns,
                    "include_extraction": (
                        CONC_EXTRACTION_COLUMN in request.selected_columns
                    ),
                    "selected_generated_columns": generated,
                    "extra_column_names": _metadata_columns(
                        request.selected_columns,
                        column,
                        {*DETACHABLE_CONCORDANCE_COLUMNS, CONC_EXTRACTION_COLUMN},
                    ),
                    "token_cache_path": str(
                        tokens_cache_path(self._cache_root(user_id))
                    ),
                },
            )

        if isinstance(
            request,
            ConcordanceDispersionDetachmentAnalysisRequest,
        ) and isinstance(
            parent_request,
            ConcordanceAnalysisRequest,
        ):
            source = parent_request
            column = source.node_columns[request.node_id]
            return owned(
                concordance_dispersion_detachment_process,
                {
                    "workspace_dir": str(artifact_dir),
                    "input_snapshot_dir": str(snapshot_dir),
                    "parent_node_id": str(request.node_id),
                    "document_column": column,
                    "search_word": source.search_word,
                    "num_left_tokens": source.num_left_tokens,
                    "num_right_tokens": source.num_right_tokens,
                    "regex": source.regex,
                    "whole_word": source.whole_word,
                    "case_sensitive": source.case_sensitive,
                    "search_mode": source.search_mode,
                    "new_node_name": request.name
                    or f"Concordance dispersion {str(request.node_id)[:8]}",
                    "include_document_column": column in request.selected_columns,
                    "extra_column_names": _metadata_columns(
                        request.selected_columns,
                        column,
                        {*DETACHABLE_CONCORDANCE_COLUMNS, CONC_EXTRACTION_COLUMN},
                    ),
                    "selected_bins": request.selected_bins,
                    "total_bins": request.total_bins,
                    "selected_matched_texts": request.selected_matched_texts,
                    "match_case_insensitive": request.match_case_insensitive,
                    "token_cache_path": str(
                        tokens_cache_path(self._cache_root(user_id))
                    ),
                },
            )

        if isinstance(request, QuotationDetachmentAnalysisRequest) and isinstance(
            parent_request,
            QuotationAnalysisRequest,
        ):
            source = parent_request
            generated = [
                item for item in request.selected_columns if item in QUOTE_COLUMN_NAMES
            ]
            return owned(
                quotation_detachment_process,
                {
                    "workspace_dir": str(artifact_dir),
                    "input_snapshot_dir": str(snapshot_dir),
                    "parent_node_id": str(request.node_id),
                    "document_column": source.column,
                    "engine": resolve_analysis_quotation_engine(
                        source,
                        self._settings,
                    ).model_dump(mode="json"),
                    "quotation_service_max_batch_size": (
                        self._settings.quotation_service_max_batch_size
                    ),
                    "quotation_service_timeout": (
                        self._settings.quotation_service_timeout
                    ),
                    "new_node_name": request.name
                    or f"Quotations {str(request.node_id)[:8]}",
                    "include_document_column": (
                        source.column in request.selected_columns
                    ),
                    "include_extraction": (
                        QUOTE_EXTRACTION_COLUMN in request.selected_columns
                    ),
                    "selected_generated_columns": generated,
                    "extra_column_names": _metadata_columns(
                        request.selected_columns,
                        source.column,
                        {*QUOTE_COLUMN_NAMES, QUOTE_EXTRACTION_COLUMN},
                    ),
                },
            )
        raise InvalidInputError("Analysis kind has no process implementation")


def _request_node_ids(record: AnalysisRecord) -> list[str]:
    request = record.request
    if isinstance(
        request,
        (
            TokenFrequencyAnalysisRequest,
            TopicModelingAnalysisRequest,
            ConcordanceAnalysisRequest,
        ),
    ):
        return [str(node_id) for node_id in request.node_ids]
    return [str(request.node_id)]


def _resolve_tokenizer_models(
    nodes: dict[str, Node],
    *,
    node_ids: list[str],
    node_columns: dict[str, str],
    requested: dict[str, str],
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for node_id in node_ids:
        node = nodes[node_id]
        column = node_columns[node_id]
        token_column = node.find_tokenization_column(column)
        if token_column is None:
            model = requested.get(node_id, "").strip()
            if not model:
                raise InvalidInputError(
                    "Raw-text Data Blocks require a tokenizer model"
                )
            resolved[node_id] = model
            continue
        model = node.tokenization[column]["model"].strip()
        requested_model = requested.get(node_id)
        if not model or (
            requested_model is not None and requested_model.strip() != model
        ):
            raise InvalidInputError("Tokenizer metadata does not match the request")
        resolved[node_id] = model
    return resolved


def _metadata_columns(
    selected: list[str],
    document_column: str,
    generated: set[str],
) -> list[str]:
    return [
        column
        for column in selected
        if column != document_column and column not in generated
    ]


def _remove_execution_staging(path: Path) -> None:
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()
    except FileNotFoundError:
        return


def resolve_analysis_quotation_engine(
    request: QuotationAnalysisRequest,
    settings: Settings,
) -> ResolvedQuotationEngine:
    if request.engine.type.value == "local":
        return ResolvedQuotationEngine()
    endpoint = next(
        (
            engine.url
            for engine in settings.quotation_remote_engines
            if engine.id == request.engine.engine_id
        ),
        None,
    )
    if endpoint is None:
        raise InvalidInputError("Quotation engine is not configured")
    return ResolvedQuotationEngine(type=QuotationEngineType.REMOTE, url=endpoint)


__all__ = ["AnalysisExecutionPreparer", "resolve_analysis_quotation_engine"]
