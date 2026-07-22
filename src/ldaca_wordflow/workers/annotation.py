"""Process-isolated full-column Annotation Analysis implementation."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import polars as pl

from ..domain.workspace import (
    AnnotationAnalysisRequest,
    AnnotationDerivation,
    DerivationInput,
    DerivationProvenance,
    node_reference,
)
from ..infrastructure.providers.annotation_ai import (
    AnnotationClassOption,
    InferenceConfig,
    annotate_all,
    resolve_provider_wire,
)
from ..infrastructure.storage.node_store import write_detached_frame
from .input_snapshots import load_snapshot_node
from .utils import process_entrypoint

logger = logging.getLogger(__name__)


@process_entrypoint
def run_annotation_analysis(
    *,
    input_snapshot_dir: str,
    output_dir: str,
    request_payload: dict[str, Any],
    api_key: str | None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Classify one immutable Data Block snapshot and publish one private output."""

    try:
        request = AnnotationAnalysisRequest.model_validate(request_payload)
        source = load_snapshot_node(input_snapshot_dir, str(request.node_id))
        schema = source.data.collect_schema()
        if request.text_column not in schema:
            raise ValueError("Annotation text column does not exist")
        if request.annotation_column in schema:
            raise ValueError("Annotation column already exists")

        if progress_callback:
            progress_callback(0.05, "Reading annotation input")
        frame = source.data.collect(engine="streaming")
        texts = [
            str(value) if value is not None else ""
            for value in frame.get_column(request.text_column).to_list()
        ]

        if progress_callback:
            progress_callback(0.1, "Classifying rows")
        labels = asyncio.run(
            annotate_all(
                resolve_provider_wire(
                    request.provider,
                    request.provider_base_url,
                ),
                request.model,
                api_key,
                request.instruction,
                [
                    AnnotationClassOption(
                        name=item.name,
                        description=item.description,
                    )
                    for item in request.classes
                ],
                texts,
                config=InferenceConfig(
                    temperature=request.temperature,
                    reasoning_enabled=request.reasoning_enabled,
                    reasoning_effort=request.reasoning_effort,
                ),
            )
        )
        if len(labels) != frame.height:
            raise ValueError("Annotation provider returned a misaligned result")
        result = frame.with_columns(
            pl.Series(name=request.annotation_column, values=labels)
        )

        if progress_callback:
            progress_callback(0.85, "Serializing annotated Data Block")
        payload = write_detached_frame(
            result,
            base_dir=output_dir,
            name=request.output_node_name,
            provenance=DerivationProvenance(
                operation=AnnotationDerivation(
                    annotation_column=request.annotation_column,
                    provider=request.provider,
                    model=request.model,
                ),
                inputs=[
                    DerivationInput(
                        role="source",
                        value=node_reference(str(request.node_id)),
                    )
                ],
            ),
            document=source.document,
            color=source.color,
        )
        if progress_callback:
            progress_callback(0.95, "Publishing annotated Data Block")
        return {
            "state": "successful",
            "result": {
                **payload,
                "output_columns": list(result.columns),
                "record_count": result.height,
            },
            "message": "Annotation completed successfully",
        }
    except Exception:
        logger.exception("Annotation Analysis failed")
        raise


__all__ = ["run_annotation_analysis"]
