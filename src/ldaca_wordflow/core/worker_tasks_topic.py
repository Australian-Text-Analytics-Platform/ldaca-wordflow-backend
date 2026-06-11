"""Topic modeling worker task implementation.

Used by:
- Backend API routes, worker tasks, workspace services, and backend tests because they
  need a backend boundary that validates inputs before delegating to workspace or worker
  state.

Flow: load workspace corpora, choose sampling and embedding settings, reuse embedding
    caches when possible, build topic payloads, and report artifacts back to the task
    manager.

The implementation is split across several sub-modules:
- ``worker_tasks_topic_types`` — internal frozen dataclasses
- ``worker_tasks_topic_pipeline`` — corpus sampling, c-TF-IDF vectorizer/stopword
  selection, and the Rust-pipeline runner
- ``worker_tasks_topic_result`` — result payload building and exact reduction

Embedding, dimensionality reduction, clustering, and c-TF-IDF labeling all run
inside the ``polars_text`` Rust extension; there is no Python BERTopic or
SentenceTransformer dependency anymore.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, cast

from .worker_utils import worker_task

from .worker_tasks_topic_types import _PreparedTopicPayload
from .worker_tasks_topic_pipeline import (
    _resolve_top_n_words,
    _resolve_vectorizer_model,
    _run_rust_topic_modeling,
    _sample_corpora_for_topic_modeling,
    _stopwords_for_lang,
)
from .worker_tasks_topic_result import (
    _build_empty_topic_payload,
    _build_topic_result_payload,
)

# Default candle embedder used by the Rust pipeline when no override is given.
# Recorded in result metadata so the API/frontend can report which model was
# used; the actual download/caching is handled inside ``polars_text``.
_DEFAULT_EMBEDDER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

logger = logging.getLogger(__name__)

__all__ = [
    "run_topic_modeling_task",
]


# ---------------------------------------------------------------------------
# Main pipeline stages
# ---------------------------------------------------------------------------


def _load_corpora_from_workspace(
    target_workspace_dir: str, node_payloads: list[dict[str, Any]], user_id: str
) -> list[list[str]]:
    """Return the raw document list for each requested node.

    Called by:
    - ``_prepare_payload`` (this module) when corpora are not provided directly.

    The Rust pipeline tokenizes for c-TF-IDF itself (lindera for CJK, a plain
    word splitter for the rest), so this no longer hydrates the pre-tokenized
    "vectorizer corpora" the BERTopic path needed -- it just selects the text
    column from each workspace node and stringifies it.
    """
    import polars as pl

    from docworkspace import Workspace

    workspace = Workspace.load(Path(target_workspace_dir))
    raw_corpora: list[list[str]] = []

    for node_info in node_payloads:
        node_id = str(node_info.get("node_id") or "")
        text_column = str(node_info.get("text_column") or "")
        if not node_id or not text_column:
            raise ValueError(
                "Topic modeling requires node_id and text_column for each node"
            )

        try:
            node = workspace.nodes[node_id]
        except KeyError as exc:
            raise ValueError(
                f"Topic modeling node {node_id} is missing from workspace"
            ) from exc

        selected = cast(
            pl.DataFrame,
            node.data.select(pl.col(text_column).alias("__doc_col__")).collect(),
        )
        raw_corpora.append(
            [
                str(value) if value is not None else ""
                for value in selected["__doc_col__"].to_list()
            ]
        )

    return raw_corpora


def _prepare_payload(
    *,
    user_id: str,
    node_infos: list[dict[str, Any]],
    artifact_dir: str,
    corpora: list[list[str]] | None,
    workspace_dir: str | None,
    progress_callback: Callable[[float, str], None] | None,
) -> _PreparedTopicPayload:
    """Prepare payload data consumed by topic-modeling worker pipeline.

    Called by:
    - ``run_topic_modeling_task`` (this module).

    Flow: load workspace corpora, choose sampling and embedding settings, reuse embedding
        caches when possible, build topic payloads, and report artifacts back to the task
        manager.
    """
    artifact_root = Path(artifact_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)

    if corpora is None:
        if workspace_dir is None:
            raise ValueError(
                "Topic modeling requires corpora or a workspace_dir to load them"
            )
        if progress_callback:
            progress_callback(0.03, "Loading source documents from workspace...")
        corpora = _load_corpora_from_workspace(workspace_dir, node_infos, user_id)

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
    - ``run_topic_modeling_task`` (this module).

    Flow: sample each corpus, pick the c-TF-IDF vectorizer/stopwords from the
    document script mix, call the Rust pipeline (chunk -> candle embed -> PaCMAP
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

    corpus_indices = [
        corpus_idx
        for corpus_idx, size in enumerate(sampled.corpus_sizes)
        for _ in range(size)
    ]
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
        corpus_indices=corpus_indices,
        seed=random_state,
        top_k=_resolve_top_n_words(representative_words_count),
        min_cluster_size=min_cluster_size,
        vectorizer_model=vectorizer_model,
        stopwords=_stopwords_for_lang(stopwords_lang),
        embedder_model=_DEFAULT_EMBEDDER_MODEL,
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
    payload_meta = payload.get("meta")
    if not isinstance(payload_meta, dict):
        payload_meta = {}
    payload_meta.update(
        {
            "native": True,
            "engine": "rust",
            "embedding_model": _DEFAULT_EMBEDDER_MODEL,
            "embedding_backend": "candle",
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
    payload["meta"] = payload_meta
    return payload


@worker_task
def run_topic_modeling_task(
    configure_worker_environment,
    user_id: str,
    workspace_id: str,
    node_infos: list[dict[str, Any]],
    artifact_dir: str,
    artifact_prefix: str,
    min_topic_size: int = 10,
    workspace_dir: str | None = None,
    corpora: list[list[str]] | None = None,
    random_seed: int = 42,
    representative_words_count: int = 5,
    progress_callback: Callable[[float, str], None] | None = None,
    embedding_cache_dir: str | None = None,
    sample_fractions: list[float | None] | None = None,
) -> dict[str, Any]:
    """Execute topic modeling in a worker process.

    Used by:
    - ``core.worker.topic_modeling_task`` because background jobs need one lifecycle owner for
      submission, progress, cancellation, and artifact cleanup.
    - ``TASK_REGISTRY["topic_modeling"]`` because background jobs need one lifecycle owner for
      submission, progress, cancellation, and artifact cleanup.
        Why:
        - Runs the Rust ``polars_text`` topic-modeling pipeline (candle embeddings
            + PaCMAP + HDBSCAN + c-TF-IDF) out-of-process and returns an artifact
            manifest (Parquet outputs) for main-process lazy retrieval/finalization.

    ``min_topic_size`` is the HDBSCAN minimum cluster size (the only native
    topic-count control); the topic count is whatever emerges. ``embedding_cache_dir``
    is retained for call-site compatibility but unused: the Rust pipeline manages
    its own in-process embedder, so there is no Python-side embedding cache.

    Flow: load workspace corpora, sample, run the Rust pipeline, build topic
        payloads, and report artifacts back to the task manager.
    """
    configure_worker_environment()

    try:
        if progress_callback:
            progress_callback(
                0.01,
                "Loading topic modeling resources. First runs may download model files...",
            )

        logger.info(
            "[Worker %d] Starting topic modeling task for workspace %s",
            os.getpid(),
            workspace_id,
        )

        prepared_payload = _prepare_payload(
            user_id=user_id,
            node_infos=node_infos,
            artifact_dir=artifact_dir,
            corpora=corpora,
            workspace_dir=workspace_dir,
            progress_callback=progress_callback,
        )

        if progress_callback:
            progress_callback(0.07, "Loading embedding model...")

        topic_payload = _compute_topic_payload(
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
            "per_corpus_topic_counts": topic_payload.get("per_corpus_topic_counts"),
            "artifacts": topic_payload.get("artifacts", {"version": 1, "nodes": []}),
            "meta": {
                **topic_payload.get("meta", {}),
                "node_names": prepared_payload.node_names,
            },
        }

        if progress_callback:
            progress_callback(1.0, "Topic modeling completed")

        logger.info("[Worker %d] Topic modeling completed successfully", os.getpid())
        return result

    except Exception as e:
        logger.error("[Worker %d] Topic modeling failed: %s", os.getpid(), e)
        if progress_callback:
            progress_callback(-1, f"Failed: {str(e)}")
        raise
