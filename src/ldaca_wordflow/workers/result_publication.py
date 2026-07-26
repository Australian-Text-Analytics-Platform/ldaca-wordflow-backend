"""Publish selected columns from immutable Run All Result artifacts."""

from __future__ import annotations

import logging
from typing import Any, Callable

from .utils import process_entrypoint

logger = logging.getLogger(__name__)


@process_entrypoint
def run_result_publication(
    *,
    artifact_dir: str,
    request_payload: dict[str, Any],
    result_paths: dict[str, str],
    document_columns: dict[str, str],
    source_colors: dict[str, str | None],
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Create private output files for one atomic Result Publication."""

    try:
        import polars as pl

        from ..domain.workspace import (
            ConcordanceResultPublicationAnalysisRequest,
            ConcordanceResultPublicationDerivation,
            DerivationInput,
            DerivationProvenance,
            QuotationResultPublicationAnalysisRequest,
            QuotationResultPublicationDerivation,
            node_reference,
        )
        from ..infrastructure.storage.node_store import write_published_frame

        kind = request_payload.get("kind")
        if kind == "concordance_result_publication":
            request = ConcordanceResultPublicationAnalysisRequest.model_validate(
                request_payload
            )
            selections = request.sources
            operation = ConcordanceResultPublicationDerivation()
        elif kind == "quotation_result_publication":
            request = QuotationResultPublicationAnalysisRequest.model_validate(
                request_payload
            )
            selections = [request.source]
            operation = QuotationResultPublicationDerivation()
        else:
            raise ValueError("Result Publication kind is unsupported")

        outputs: list[dict[str, Any]] = []
        for index, selection in enumerate(selections):
            source_id = str(selection.source_node_id)
            path = result_paths.get(source_id)
            document_column = document_columns.get(source_id)
            if path is None or document_column is None:
                raise ValueError("Result Publication source artifact is unavailable")
            if document_column not in selection.selected_columns:
                raise ValueError("Result Publication requires the document column")
            frame = pl.scan_parquet(path)
            schema = frame.collect_schema()
            if any(column not in schema for column in selection.selected_columns):
                raise ValueError("Result Publication column is unavailable")
            if progress_callback:
                progress_callback(
                    0.1 + (0.65 * index / max(len(selections), 1)),
                    f"Preparing {selection.new_node_name}",
                )
            selected = frame.select(selection.selected_columns).collect(
                engine="streaming"
            )
            node_payload = write_published_frame(
                selected,
                base_dir=artifact_dir,
                name=selection.new_node_name,
                provenance=DerivationProvenance(
                    operation=operation,
                    inputs=[
                        DerivationInput(
                            role="source",
                            value=node_reference(source_id),
                        )
                    ],
                ),
                document=document_column,
                color=source_colors.get(source_id),
            )
            outputs.append(
                {
                    "source_node_id": source_id,
                    "data": {
                        **node_payload,
                        "output_columns": selection.selected_columns,
                        "record_count": selected.height,
                    },
                }
            )

        if progress_callback:
            progress_callback(0.95, "Saving Result Publication...")
        return {
            "state": "successful",
            "outputs": outputs,
            "message": "Result Publication completed successfully",
        }
    except Exception:
        logger.exception("Result Publication failed")
        raise


__all__ = ["run_result_publication"]
