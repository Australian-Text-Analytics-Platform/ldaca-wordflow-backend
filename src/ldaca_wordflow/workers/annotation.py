"""Process-isolated full-column Annotation Analysis implementation."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl

from ..domain.workspace import AnnotationAnalysisRequest
from ..infrastructure.providers.annotation_ai import (
    AnnotationClassOption,
    AnnotationExample,
    InferenceConfig,
    annotate_all,
    resolve_provider_wire,
)
from ..infrastructure.storage.durable_fs import atomic_output_path
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
        if request.annotation_column not in schema:
            raise ValueError("Annotation column does not exist")
        if (
            request.correction_column is not None
            and request.correction_column not in schema
        ):
            raise ValueError("Annotation correction column does not exist")

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
                examples=_load_examples(request, input_snapshot_dir),
            )
        )
        if len(labels) != frame.height:
            raise ValueError("Annotation provider returned a misaligned result")
        if request.correction_column is not None:
            corrections = frame.get_column(request.correction_column).to_list()
            labels = [
                (
                    str(correction).strip()
                    if correction is not None and str(correction).strip()
                    else predicted
                )
                for predicted, correction in zip(labels, corrections, strict=True)
            ]
        result = frame.with_columns(
            pl.Series(name=request.annotation_column, values=labels)
        )

        if progress_callback:
            progress_callback(0.85, "Serializing annotated Data Block")
        relative_path = "annotation-run-all.parquet"
        with atomic_output_path(Path(output_dir) / relative_path) as temporary:
            result.write_parquet(temporary)
        if progress_callback:
            progress_callback(0.95, "Publishing annotated Data Block")
        return {
            "state": "successful",
            "result": {
                "parquet_path": relative_path,
                "output_columns": list(result.columns),
                "record_count": result.height,
            },
            "message": "Annotation completed successfully",
        }
    except Exception:
        logger.exception("Annotation Analysis failed")
        raise


def _load_examples(
    request: AnnotationAnalysisRequest,
    input_snapshot_dir: str,
) -> list[AnnotationExample]:
    if request.example_node_id is None:
        return []
    assert request.example_text_column is not None
    assert request.example_annotation_column is not None
    example = load_snapshot_node(input_snapshot_dir, str(request.example_node_id))
    frame = example.data.select(
        request.example_text_column,
        request.example_annotation_column,
    ).collect(engine="streaming")
    pairs: list[AnnotationExample] = []
    for text, label in frame.iter_rows():
        normalized_text = str(text).strip() if text is not None else ""
        normalized_label = str(label).strip() if label is not None else ""
        if normalized_text and normalized_label:
            pairs.append(
                AnnotationExample(text=normalized_text, label=normalized_label)
            )
    return pairs


__all__ = ["run_annotation_analysis"]
