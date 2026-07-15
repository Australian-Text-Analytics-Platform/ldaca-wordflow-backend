"""Process-worker implementation for topic-modeling analysis.

Used by:
- canonical Analysis execution and backend tests that exercise topic-modeling
  computation from immutable inputs.

Flow: load workspace corpora, choose sampling and embedding settings, build topic
    payloads, and return private artifacts to the Analysis service.

The implementation is split across several sub-modules:
- ``topic_types`` — internal frozen dataclasses
- ``topic_pipeline`` — corpus sampling, c-TF-IDF vectorizer/stopword
  selection, and the Rust-pipeline runner
- ``topic_result`` — result payload building and exact reduction

Embedding, dimensionality reduction, clustering, and c-TF-IDF labeling all run
inside the ``polars_text`` Rust extension; there is no Python BERTopic or
SentenceTransformer dependency anymore.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable

from .topic_pipeline import (
    _resolve_top_n_words,
    _resolve_vectorizer_model,
    _run_rust_topic_modeling,
    _sample_corpora_for_topic_modeling,
    _stopwords_for_lang,
)
from .topic_result import (
    _build_empty_topic_payload,
    _build_topic_result_payload,
)
from .topic_types import _PreparedTopicPayload
from .utils import process_entrypoint

# Default ONNX embedder used by the Rust ORT pipeline when no override is given.
# Recorded in result metadata so the API/frontend can report which model was
# used; the actual download/loading is handled inside ``polars_text``.
_DEFAULT_EMBEDDER_MODEL = "onnx-community/all-MiniLM-L6-v2-ONNX"

logger = logging.getLogger(__name__)

__all__ = [
    "run_topic_modeling_analysis",
]


# ---------------------------------------------------------------------------
# Main pipeline stages
# ---------------------------------------------------------------------------


def _load_corpora_from_snapshot(
    input_snapshot_dir: str,
    node_payloads: list[dict[str, Any]],
) -> list[list[str]]:
    """Return raw document lists from Analysis-owned node plan snapshots.

    Called by:
    - ``_prepare_payload`` when the Analysis invocation contains immutable
      snapshot references instead of eagerly collected corpora.

    Flow: load each snapshotted LazyFrame, enrich ``node_payloads`` with display
    names and source schema metadata, and collect the selected text column inside
    the worker process.
    """

    import polars as pl

    from .input_snapshots import load_snapshot_node

    raw_corpora: list[list[str]] = []
    for node_info in node_payloads:
        node_id = str(node_info.get("node_id") or "")
        text_column = str(node_info.get("text_column") or "")
        if not node_id or not text_column:
            raise ValueError(
                "Topic modeling requires node_id and text_column for each node"
            )

        snapshot_node = load_snapshot_node(input_snapshot_dir, node_id)
        node_info.setdefault("node_name", snapshot_node.name)
        node_info.setdefault(
            "original_columns",
            list(snapshot_node.data.collect_schema().names()),
        )
        selected = snapshot_node.data.select(
            pl.col(text_column).alias("__doc_col__")
        ).collect()
        raw_corpora.append(
            [
                str(value) if value is not None else ""
                for value in selected["__doc_col__"].to_list()
            ]
        )
    return raw_corpora


def _prepare_payload(
    *,
    node_infos: list[dict[str, Any]],
    artifact_dir: str,
    corpora: list[list[str]] | None,
    input_snapshot_dir: str | None,
    progress_callback: Callable[[float, str], None] | None,
) -> _PreparedTopicPayload:
    """Prepare payload data consumed by topic-modeling worker pipeline.

    Called by:
    - ``run_topic_modeling_analysis`` (this module).

    Flow: load workspace corpora, choose sampling and embedding settings, build
        topic payloads, and return artifacts to the Analysis service.
    """
    artifact_root = Path(artifact_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)

    if corpora is None:
        if input_snapshot_dir is not None:
            if progress_callback:
                progress_callback(
                    0.03, "Loading source documents from Analysis snapshot..."
                )
            corpora = _load_corpora_from_snapshot(input_snapshot_dir, node_infos)
        else:
            raise ValueError("Topic modeling requires corpora or input_snapshot_dir")

    if len(corpora) != len(node_infos):
        raise ValueError(
            "Topic modeling payload mismatch: corpora and node_infos lengths differ"
        )

    if progress_callback:
        progress_callback(0.05, "Preparing topic modeling payload...")

    node_names = [
        str(info.get("node_name") or info.get("node_id") or "node")
        for info in node_infos
    ]
    return _PreparedTopicPayload(
        artifact_root=artifact_root,
        corpora=corpora,
        node_names=node_names,
    )


def _compute_topic_payload(
    *,
    embedding_cache_path: str,
    node_infos: list[dict[str, Any]],
    corpora: list[list[str]],
    artifact_root: Path,
    artifact_prefix: str,
    random_seed: int,
    representative_words_count: int,
    progress_callback: Callable[[float, str], None] | None,
    sample_fractions: list[float | None] | None,
    min_topic_size: int,
) -> dict[str, Any]:
    """Run the full topic-modeling pipeline: sample, run Rust, build the payload.

    Called by:
    - ``run_topic_modeling_analysis`` (this module).

    Flow: sample each corpus, pick the c-TF-IDF vectorizer/stopwords from the
    document script mix, call the Rust pipeline (chunk -> ORT embed -> PaCMAP
    -> HDBSCAN -> c-TF-IDF, plus optional merge for target/exact modes), and turn
    its JSON result into the wire payload. For ``exact`` mode it also persists a
    JSON re-aggregation context so the slider can request a different count later.
    """
    sampled = _sample_corpora_for_topic_modeling(
        corpora=corpora,
        sample_fractions=sample_fractions,
        random_seed=random_seed,
    )
    if not sampled.all_docs:
        return _build_empty_topic_payload(
            sampled=sampled,
            node_infos=node_infos,
            artifact_root=artifact_root,
            artifact_prefix=artifact_prefix,
        )

    if any(size == 0 for size in sampled.corpus_sizes):
        raise ValueError("All corpora must contain at least one document.")

    random_state = int(random_seed)
    max_representative_words = max(1, int(representative_words_count))
    min_cluster_size = max(2, int(min_topic_size))

    vectorizer_model, stopwords_lang = _resolve_vectorizer_model(sampled.all_docs)

    logger.info(
        "[Worker %d] Running Rust topic-modeling pipeline (%d docs, min_cluster_size=%d)",
        os.getpid(),
        len(sampled.all_docs),
        min_cluster_size,
    )
    if progress_callback:
        progress_callback(0.1, "Embedding and clustering documents...")

    rust_result = _run_rust_topic_modeling(
        all_docs=sampled.all_docs,
        seed=random_state,
        top_k=_resolve_top_n_words(representative_words_count),
        min_cluster_size=min_cluster_size,
        vectorizer_model=vectorizer_model,
        stopwords=_stopwords_for_lang(stopwords_lang),
        embedder_model=_DEFAULT_EMBEDDER_MODEL,
        embedding_cache=embedding_cache_path,
    )

    if progress_callback:
        progress_callback(0.85, "Assembling topic results...")

    payload = _build_topic_result_payload(
        rust_result=rust_result,
        node_infos=node_infos,
        corpus_sizes=sampled.corpus_sizes,
        active_corpora_indices=sampled.active_corpora_indices,
        max_representative_words=max_representative_words,
        artifact_prefix=artifact_prefix,
        artifact_root=artifact_root,
    )
    payload_meta = payload["meta"]
    payload_meta.update(
        {
            "native": True,
            "engine": "rust",
            "embedding_model": _DEFAULT_EMBEDDER_MODEL,
            "embedding_backend": "ort",
            "min_topic_size": min_cluster_size,
            "representative_words_count": max_representative_words,
            "random_state": random_state,
            "vectorizer_model": vectorizer_model,
            "n_chunks": int(rust_result.get("n_chunks") or 0),
            **(
                {
                    "corpus_sizes_before_sample": sampled.corpus_sizes_before_sample,
                    "corpus_sizes_after_sample": sampled.corpus_sizes,
                }
                if sample_fractions is not None
                else {}
            ),
        }
    )
    stage_timings = rust_result.get("stage_timings_ms")
    if isinstance(stage_timings, list):
        payload_meta["stage_timings_ms"] = stage_timings
    payload["meta"] = payload_meta
    return payload


def _compute_topic_modeling(
    workspace_id: str,
    node_infos: list[dict[str, Any]],
    artifact_dir: str,
    artifact_prefix: str,
    embedding_cache_path: str,
    min_topic_size: int = 10,
    input_snapshot_dir: str | None = None,
    corpora: list[list[str]] | None = None,
    random_seed: int = 42,
    representative_words_count: int = 5,
    progress_callback: Callable[[float, str], None] | None = None,
    sample_fractions: list[float | None] | None = None,
) -> dict[str, Any]:
    """Execute topic modeling in a worker process.

    Used by:
    - canonical topic-modeling Analysis execution, which owns submission,
      progress, cancellation, and artifact cleanup.
        Why:
        - Runs the Rust ``polars_text`` topic-modeling pipeline (ORT embeddings
            + PaCMAP + HDBSCAN + c-TF-IDF) out-of-process and returns an artifact
            manifest (Parquet outputs) for main-process lazy retrieval/finalization.

    ``min_topic_size`` is the HDBSCAN minimum cluster size (the only native
    topic-count control); the topic count is whatever emerges. The Rust pipeline
    manages its own in-process embedder and DuckDB embedding cache.

    Flow: load workspace corpora, sample, run the Rust pipeline, build topic
        payloads, and return artifacts to the Analysis service.
    """
    try:
        if progress_callback:
            progress_callback(
                0.01,
                "Loading topic modeling resources. First runs may download model files...",
            )

        logger.info(
            "[Worker %d] Starting topic-modeling Analysis for workspace %s",
            os.getpid(),
            workspace_id,
        )

        prepared_payload = _prepare_payload(
            node_infos=node_infos,
            artifact_dir=artifact_dir,
            corpora=corpora,
            input_snapshot_dir=input_snapshot_dir,
            progress_callback=progress_callback,
        )

        if progress_callback:
            progress_callback(0.07, "Loading embedding model...")

        topic_payload = _compute_topic_payload(
            embedding_cache_path=embedding_cache_path,
            node_infos=node_infos,
            corpora=prepared_payload.corpora,
            artifact_root=prepared_payload.artifact_root,
            artifact_prefix=artifact_prefix,
            random_seed=random_seed,
            representative_words_count=representative_words_count,
            progress_callback=progress_callback,
            sample_fractions=sample_fractions,
            min_topic_size=min_topic_size,
        )

        if progress_callback:
            progress_callback(0.9, "Writing topic-modeling results...")

        result = {
            "topics": topic_payload["topics"],
            "corpus_sizes": topic_payload["corpus_sizes"],
            "per_corpus_topic_counts": topic_payload["per_corpus_topic_counts"],
            "artifacts": topic_payload["artifacts"],
            "meta": {
                **topic_payload["meta"],
                "node_names": prepared_payload.node_names,
            },
        }

        logger.info("[Worker %d] Topic modeling completed successfully", os.getpid())
        return result

    except Exception as e:
        logger.error("[Worker %d] Topic modeling failed: %s", os.getpid(), e)
        raise


@process_entrypoint
def run_topic_modeling_analysis(
    *,
    user_id: str,
    workspace_id: str,
    node_infos: list[dict[str, Any]],
    artifact_dir: str,
    artifact_prefix: str,
    input_snapshot_dir: str,
    embedding_cache_path: str,
    min_topic_size: int = 10,
    random_seed: int = 42,
    representative_words_count: int = 5,
    progress_callback: Callable[[float, str], None] | None = None,
    sample_fractions: list[float | None] | None = None,
) -> dict[str, Any]:
    """Run the canonical snapshot-only topic-modeling process contract."""

    return _compute_topic_modeling(
        workspace_id=workspace_id,
        node_infos=node_infos,
        artifact_dir=artifact_dir,
        artifact_prefix=artifact_prefix,
        min_topic_size=min_topic_size,
        input_snapshot_dir=input_snapshot_dir,
        corpora=None,
        random_seed=random_seed,
        representative_words_count=representative_words_count,
        progress_callback=progress_callback,
        sample_fractions=sample_fractions,
        embedding_cache_path=embedding_cache_path,
    )
