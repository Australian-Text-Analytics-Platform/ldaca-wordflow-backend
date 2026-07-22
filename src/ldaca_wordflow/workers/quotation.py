"""Process-worker implementations for quotation Analysis execution.

Used by:
- canonical Analysis execution and backend tests that exercise quotation
  computation from immutable inputs.

Flow: normalize source text, run local or remote quotation extraction, preserve
    source-row mappings, and return the owning Analysis result or child Data
    Block payload.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from ..analysis.generated_columns import (
    QUOTE_COLUMN_NAMES,
    QUOTE_EXTRACTION_COLUMN,
    QUOTE_QUOTE_COLUMN,
)
from ..domain.workspace import (
    DerivationInput,
    DerivationProvenance,
    QuotationDetachmentDerivation,
    node_reference,
)
from .utils import process_entrypoint

logger = logging.getLogger(__name__)


def _collect_quotation_source_from_snapshot(
    *,
    input_snapshot_dir: str,
    node_id: str,
    document_column: str,
    extra_column_names: list[str] | None,
    include_all_metadata: bool = False,
) -> tuple[list[str], dict[str, list] | None, dict[str, Any] | None]:
    """Collect quotation source rows inside the worker process.

    Used by:
    - root and child quotation Analysis workers using immutable LazyFrame-plan
      snapshots instead of full Python corpora.

    Flow: load the snapshotted node plan, select the document and requested
    metadata columns, filter blank documents, and return aligned lists for the
    existing quotation extraction builder.
    """

    import polars as pl

    from .input_snapshots import load_snapshot_node

    snapshot_node = load_snapshot_node(input_snapshot_dir, node_id)
    node_data = snapshot_node.data
    schema_names = list(node_data.collect_schema().names())
    if include_all_metadata:
        metadata_columns = [
            column
            for column in schema_names
            if column != document_column
        ]
    else:
        metadata_columns = list(extra_column_names or [])

    corpus_df = (
        node_data.select(
            [pl.col(document_column)] + [pl.col(c) for c in metadata_columns]
        )
        .filter(
            pl.col(document_column)
            .cast(pl.Utf8, strict=False)
            .str.strip_chars()
            .str.len_chars()
            .fill_null(0)
            > 0
        )
        .collect()
    )
    node_corpus = [
        str(value) if value is not None else ""
        for value in corpus_df.get_column(document_column).to_list()
    ]
    if not metadata_columns:
        return node_corpus, None, None

    extra_columns_data: dict[str, list] = {}
    extra_columns_dtypes: dict[str, Any] = {}
    for column in metadata_columns:
        series = corpus_df.get_column(column)
        extra_columns_data[column] = series.to_list()
        extra_columns_dtypes[column] = series.dtype
    return node_corpus, extra_columns_data, extra_columns_dtypes


@process_entrypoint
def run_quotation_analysis(
    user_id: str,
    workspace_id: str,
    input_snapshot_dir: str,
    node_id: str,
    request_payload: dict[str, Any],
    quotation_service_max_batch_size: int,
    quotation_service_timeout: float,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Execute the primary quotation analysis in a worker process.

    Used by:
    - canonical quotation Analysis execution because submission must not run
      quotation extraction or page collection on the event loop.

    Flow: load the snapshotted node plan, reuse
    the quotation page builder, and return the persisted Analysis result payload.
    """

    try:
        if progress_callback:
            progress_callback(0.1, "Loading quotation input...")

        import asyncio

        from ..analysis.quotation_core import compute_quotation_page
        from ..infrastructure.providers.quotation_client import (
            QuotationProviderClient,
            QuotationServiceError,
        )
        from ..models.quotation import QuotationEngineType, ResolvedQuotationEngine
        from .input_snapshots import load_snapshot_node

        snapshot_node = load_snapshot_node(input_snapshot_dir, node_id)
        node = snapshot_node.to_node()
        engine_payload = request_payload.get("engine") or {}
        engine = ResolvedQuotationEngine.model_validate(engine_payload)

        async def run() -> dict[str, Any]:
            async def run_inline(function, *args):
                return function(*args)

            client = (
                QuotationProviderClient(
                    default_timeout=quotation_service_timeout,
                )
                if engine.type is QuotationEngineType.REMOTE
                else None
            )

            async def extract_remote(*args, **kwargs):
                if client is None:
                    raise QuotationServiceError(
                        "Remote extraction requested for a local Analysis"
                    )
                return await client.extract(*args, **kwargs)

            try:
                return await compute_quotation_page(
                    node,
                    str(request_payload["column"]),
                    engine,
                    page=int(request_payload.get("page") or 1),
                    page_size=request_payload.get("page_size"),
                    sort_by=request_payload.get("sort_by"),
                    descending=bool(request_payload.get("descending", False)),
                    quotation_service_max_batch_size=quotation_service_max_batch_size,
                    quotation_service_timeout=quotation_service_timeout,
                    extract_remote_fn=extract_remote,
                    run_blocking=run_inline,
                )
            finally:
                if client is not None:
                    await client.close()

        page_payload = asyncio.run(run())
        return page_payload
    except Exception:
        logger.exception(
            "Quotation Analysis failed for user=%s workspace=%s node=%s",
            user_id,
            workspace_id,
            node_id,
        )
        raise


def _build_quotation_occurrence_dataframe(
    node_corpus: list[str],
    document_column: str,
    include_document_column: bool,
    extra_columns_data: dict[str, list] | None,
    extra_columns_dtypes: dict[str, Any] | None = None,
):
    """Extract quotation occurrences from a corpus. Returns (df, output_columns).

    Called by:
    - quotation detachment child Analyses that recompute a complete published
      Data Block from their immutable request and snapshot.

    Flow: normalize source text, run local or remote quotation extraction, preserve
        source-row mappings, and return flat occurrence rows.
    """
    import polars as pl

    from ldaca_wordflow.analysis.quotation_core import (
        flatten_grouped_quotation_dataframe,
        quotation_groups_via_quote_extractor,
    )

    corpus = [str(v) if v is not None else "" for v in node_corpus]
    non_empty_mask = [bool(value.strip()) for value in corpus]
    filtered_corpus = [value for value, keep in zip(corpus, non_empty_mask) if keep]

    source_column_name = "__quotation_source__"
    data: dict[str, list] = {source_column_name: filtered_corpus}
    # `QUOTE_extraction` is the per-quote-row copy of the raw source document
    # text — exposed under a canonical name so callers (table view, detach)
    # can refer to it without needing to know the user's source column name.
    # Carry it through extraction so each quote retains its source text. The
    # child worker omits it from the published Data Block when not requested.
    data[QUOTE_EXTRACTION_COLUMN] = filtered_corpus
    selected_columns: list[str] = [QUOTE_EXTRACTION_COLUMN]
    output_columns: list[str] = [QUOTE_EXTRACTION_COLUMN]

    if include_document_column:
        data[document_column] = filtered_corpus
        selected_columns.append(document_column)
        output_columns.append(document_column)

    if extra_columns_data:
        for col_name, col_values in extra_columns_data.items():
            filtered_values = [
                value for value, keep in zip(col_values, non_empty_mask) if keep
            ]
            data[col_name] = filtered_values
            selected_columns.append(col_name)
            output_columns.append(col_name)

    input_df = pl.DataFrame(data)
    if extra_columns_dtypes:
        cast_exprs = [
            pl.col(col).cast(dtype)
            for col, dtype in extra_columns_dtypes.items()
            if col in input_df.columns and input_df.schema[col] != dtype
        ]
        if cast_exprs:
            input_df = input_df.with_columns(cast_exprs)
    quote_df = quotation_groups_via_quote_extractor(input_df, source_column_name)
    quote_df = flatten_grouped_quotation_dataframe(quote_df)
    generated_columns = [
        column_name
        for column_name in QUOTE_COLUMN_NAMES
        if column_name in quote_df.columns
    ]
    quote_df = quote_df.select(selected_columns + generated_columns)

    if QUOTE_QUOTE_COLUMN in quote_df.columns:
        quote_df = quote_df.filter(pl.col(QUOTE_QUOTE_COLUMN).is_not_null())

    return quote_df, output_columns + generated_columns


@process_entrypoint
def run_quotation_detachment(
    workspace_dir: str,
    input_snapshot_dir: str,
    parent_node_id: str,
    document_column: str,
    engine: dict[str, Any],
    quotation_service_max_batch_size: int,
    quotation_service_timeout: float,
    new_node_name: str,
    include_document_column: bool = False,
    include_extraction: bool = False,
    selected_generated_columns: list[str] | None = None,
    extra_column_names: list[str] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Recompute a complete local or remote quotation selection."""
    try:
        if progress_callback:
            progress_callback(0.02, "Loading quotation extractor...")

        import asyncio
        import os

        import polars as pl

        from ..analysis.quotation_core import (
            compute_quote_dataframe,
            flatten_grouped_quotation_dataframe,
        )
        from ..infrastructure.providers.quotation_client import (
            QuotationProviderClient,
            QuotationServiceError,
        )
        from ..infrastructure.storage.node_store import write_detached_frame
        from ..models.quotation import QuotationEngineType, ResolvedQuotationEngine
        from .input_snapshots import load_snapshot_node

        logger.info("[Worker %d] Starting quotation detachment", os.getpid())

        if progress_callback:
            progress_callback(0.2, "Preparing text data...")
        node_corpus, extra_columns_data, extra_columns_dtypes = (
            _collect_quotation_source_from_snapshot(
                input_snapshot_dir=input_snapshot_dir,
                node_id=parent_node_id,
                document_column=document_column,
                extra_column_names=extra_column_names,
            )
        )
        if progress_callback:
            progress_callback(0.6, "Extracting quotations...")

        input_data: dict[str, list] = {
            document_column: node_corpus,
            QUOTE_EXTRACTION_COLUMN: node_corpus,
        }
        if extra_columns_data:
            input_data.update(extra_columns_data)
        base_df = pl.DataFrame(input_data)
        if extra_columns_dtypes:
            base_df = base_df.with_columns(
                [
                    pl.col(column).cast(dtype)
                    for column, dtype in extra_columns_dtypes.items()
                ]
            )
        snapshot_node = load_snapshot_node(input_snapshot_dir, parent_node_id).to_node()
        engine_config = ResolvedQuotationEngine.model_validate(engine)

        async def extract() -> pl.DataFrame:
            async def run_inline(function, *args):
                return function(*args)

            client = (
                QuotationProviderClient(default_timeout=quotation_service_timeout)
                if engine_config.type is QuotationEngineType.REMOTE
                else None
            )

            async def extract_remote(*args, **kwargs):
                if client is None:
                    raise QuotationServiceError(
                        "Remote extraction requested for a local Analysis"
                    )
                return await client.extract(*args, **kwargs)

            try:
                grouped = await compute_quote_dataframe(
                    snapshot_node,
                    base_df,
                    document_column,
                    engine_config,
                    use_base_only=True,
                    extract_remote_fn=extract_remote,
                    run_blocking=run_inline,
                    quotation_service_max_batch_size=(quotation_service_max_batch_size),
                    quotation_service_timeout=quotation_service_timeout,
                )
                return flatten_grouped_quotation_dataframe(grouped)
            finally:
                if client is not None:
                    await client.close()

        quote_df = asyncio.run(extract())
        output_columns = list(quote_df.columns)

        # Final projection honoring the user's column choice. Generated quote
        # columns are kept only when ticked; QUOTE_extraction stays opt-in; the
        # document column and metadata columns pass through.
        generated_set = set(QUOTE_COLUMN_NAMES)
        wanted_generated = set(selected_generated_columns or [])
        keep_columns: list[str] = []
        for col in output_columns:
            if col in generated_set:
                if col in wanted_generated:
                    keep_columns.append(col)
            elif col == QUOTE_EXTRACTION_COLUMN:
                if include_extraction:
                    keep_columns.append(col)
            elif col == document_column:
                if include_document_column:
                    keep_columns.append(col)
            else:
                keep_columns.append(col)
        if keep_columns and keep_columns != output_columns:
            quote_df = quote_df.select(keep_columns)
            output_columns = keep_columns

        if progress_callback:
            progress_callback(0.82, "Serializing detached data block...")

        node_payload = write_detached_frame(
            quote_df,
            base_dir=workspace_dir,
            name=new_node_name,
            provenance=DerivationProvenance(
                operation=QuotationDetachmentDerivation(),
                inputs=[
                    DerivationInput(
                        role="source",
                        value=node_reference(parent_node_id),
                    )
                ],
            ),
            document=document_column,
        )

        if progress_callback:
            progress_callback(0.95, "Publishing quotation Data Block...")

        logger.info(
            "[Worker %d] Quotation detachment completed successfully", os.getpid()
        )

        return {
            "state": "successful",
            "result": {
                **node_payload,
                "output_columns": output_columns,
                "record_count": int(quote_df.height),
            },
            "message": "Quotation detach completed successfully",
        }
    except Exception:
        logger.exception("Quotation detachment failed")
        raise
