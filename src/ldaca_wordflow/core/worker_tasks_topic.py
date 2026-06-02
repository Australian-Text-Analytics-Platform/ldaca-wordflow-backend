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
- ``worker_tasks_topic_embedding`` — embedder selection, caching, and encoding
- ``worker_tasks_topic_pipeline`` — corpus sampling and BERTopic pipeline building
- ``worker_tasks_topic_result`` — result payload building and exact reduction
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np

from ..api.workspaces.analyses.generated_columns import (
    TOPIC_COLUMN,
    TOPIC_MEANING_COLUMN,
)
from .worker_utils import worker_task

from .worker_tasks_topic_types import _PreparedTopicPayload, _SampledTopicCorpora
from .worker_tasks_topic_embedding import (
    _EMBEDDER_CACHE,
    _EMBEDDING_CHUNK_SIZE,
    _TOPIC_EMBEDDER_REPO_ID,
    _TOPIC_EMBEDDER_REVISION,
    _embed_documents,
    _embedder_cache_label,
    _encode_embeddings_in_chunks,
    _get_embedder,
)
from .worker_tasks_topic_pipeline import (
    _build_classic_pipeline,
    _build_label_vectorizer,
    _compute_min_topic_size,
    _resolve_top_n_words,
    _run_classic_pipeline,
    _sample_corpora_for_topic_modeling,
    _sample_corpus,
)
from .worker_tasks_topic_result import (
    _build_empty_topic_payload,
    _build_topic_result_payload,
    _count_non_outlier_topics,
    _persist_exact_reduction_artifact,
    _resolve_exact_reduce_topics_target,
    reaggregate_exact_topic_modeling_result,
)

logger = logging.getLogger(__name__)

__all__ = [
    "run_topic_modeling_task",
    "reaggregate_exact_topic_modeling_result",
]


# ---------------------------------------------------------------------------
# Main pipeline stages
# ---------------------------------------------------------------------------


def _load_corpora_from_workspace(
    target_workspace_dir: str, node_payloads: list[dict[str, Any]], user_id: str
) -> tuple[list[list[str]], list[list[str] | None], list[str | None]]:
    """Return raw docs, optional tokenized docs, and token columns per node.

    Called by:
    - ``_prepare_payload`` (this module) when corpora are not provided directly.

    Flow: load workspace corpora, choose sampling and embedding settings, reuse embedding
        caches when possible, build topic payloads, and report artifacts back to the task
        manager.
    """
    import polars as pl

    from docworkspace import Workspace

    from .tokens_cache import hydrate_tokenization_lazyframe

    workspace = Workspace.load(Path(target_workspace_dir))
    raw_corpora: list[list[str]] = []
    vectorizer_corpora: list[list[str] | None] = []
    tokens_columns: list[str | None] = []

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

        tokens_column = node.find_tokenization_column(text_column)
        if tokens_column is None:
            vectorizer_corpora.append(None)
            tokens_columns.append(None)
            continue

        node_data = hydrate_tokenization_lazyframe(
            node=node,
            source_column=text_column,
            user_id=user_id,
        )

        tokens_selected = cast(
            pl.DataFrame,
            node_data.select(
                pl.col(tokens_column)
                .list.eval(pl.element().struct.field("token"))
                .alias("__tokens_col__")
            ).collect(),
        )
        joined: list[str] = []
        for tokens in tokens_selected["__tokens_col__"].to_list():
            if tokens is None:
                joined.append("")
                continue
            joined.append(
                " ".join(
                    str(token) for token in tokens if token is not None and str(token)
                )
            )
        vectorizer_corpora.append(joined)
        tokens_columns.append(tokens_column)

    return raw_corpora, vectorizer_corpora, tokens_columns


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
        corpora, vectorizer_corpora, tokens_columns_per_node = (
            _load_corpora_from_workspace(workspace_dir, node_infos, user_id)
        )
    else:
        vectorizer_corpora = [None] * len(corpora)
        tokens_columns_per_node = [None] * len(corpora)

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
        vectorizer_corpora=vectorizer_corpora,
        tokens_columns_per_node=tokens_columns_per_node,
        node_names=node_names,
    )


def _compute_topic_payload(
    *,
    node_infos: list[dict[str, Any]],
    corpora: list[list[str]],
    vectorizer_corpora: list[list[str] | None],
    artifact_root: Path,
    artifact_prefix: str,
    random_seed: int,
    representative_words_count: int,
    progress_callback: Callable[[float, str], None] | None,
    embedding_cache_dir: str | None,
    sample_fractions: list[float | None] | None,
    topic_size_mode: str | None,
    topic_size_value: int | None,
) -> dict[str, Any]:
    """Run the full topic-modeling pipeline: sample, embed, fit, and build
    the result payload.

    Called by:
    - ``run_topic_modeling_task`` (this module).

    Flow: load workspace corpora, choose sampling and embedding settings, reuse embedding
        caches when possible, build topic payloads, and report artifacts back to the task
        manager.
    """
    sampled = _sample_corpora_for_topic_modeling(
        corpora=corpora,
        vectorizer_corpora=vectorizer_corpora,
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
    random.seed(random_state)
    np.random.seed(random_state)

    effective_min_topic_size = _compute_min_topic_size(
        len(sampled.all_docs), topic_size_mode or "target", topic_size_value or 25
    )
    logger.info(
        "[Worker %d] Running classic BERTopic pipeline (%d docs)",
        os.getpid(),
        len(sampled.all_docs),
    )

    embedded = _embed_documents(
        all_docs=sampled.all_docs,
        embedding_cache_dir=embedding_cache_dir,
        progress_callback=progress_callback,
        progress_start=0.08,
        progress_end=0.63,
    )

    top_n_words = _resolve_top_n_words(representative_words_count)
    pipeline_run = _run_classic_pipeline(
        all_docs_for_vectorizer=sampled.all_docs_for_vectorizer,
        all_embeddings=embedded.all_embeddings,
        effective_min_topic_size=effective_min_topic_size,
        random_state=random_state,
        embedder=embedded.embedder,
        top_n_words=top_n_words,
        progress_callback=progress_callback,
        progress_fraction=0.65,
    )

    topic_model = pipeline_run.topic_model
    assigned_topics = pipeline_run.assigned_topics
    raw_total_topics = None
    exact_reduction_artifact_path: str | None = None
    if (topic_size_mode or "target") == "exact" and topic_size_value:
        raw_total_topics = _count_non_outlier_topics(topic_model)
        exact_reduction_artifact_path = str(
            artifact_root / f"{artifact_prefix}_exact_reduction.pkl"
        )
        _persist_exact_reduction_artifact(
            exact_reduction_artifact_path,
            topic_model=topic_model,
            all_docs=sampled.all_docs_for_vectorizer,
            corpus_sizes=sampled.corpus_sizes,
            active_corpora_indices=sampled.active_corpora_indices,
        )
        if progress_callback:
            progress_callback(0.65, f"Reducing topics to exactly {topic_size_value}...")
        topic_model.reduce_topics(
            sampled.all_docs_for_vectorizer,
            nr_topics=_resolve_exact_reduce_topics_target(
                topic_model, int(topic_size_value)
            ),
        )
        assigned_topics = list(topic_model.topics_)

    payload = _build_topic_result_payload(
        topic_model=topic_model,
        node_infos=node_infos,
        all_docs=sampled.all_docs_for_vectorizer,
        corpus_sizes=sampled.corpus_sizes,
        active_corpora_indices=sampled.active_corpora_indices,
        max_representative_words=max_representative_words,
        random_state=random_state,
        assigned_topics=assigned_topics,
        artifact_prefix=artifact_prefix,
        artifact_root=artifact_root,
    )
    payload_meta = payload.get("meta")
    if not isinstance(payload_meta, dict):
        payload_meta = {}
    payload_meta.update(
        {
            "native": True,
            "engine": "bertopic",
            "embedding_model": embedded.embedding_model_name,
            "embedding_backend": embedded.embedding_backend,
            "min_topic_size": effective_min_topic_size,
            "topic_size_mode": topic_size_mode or "target",
            "topic_size_value": topic_size_value,
            "representative_words_count": max_representative_words,
            "random_state": random_state,
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
    if raw_total_topics is not None:
        payload_meta["raw_total_topics"] = raw_total_topics
    payload["meta"] = payload_meta
    payload_artifacts = payload.get("artifacts")
    if isinstance(payload_artifacts, dict) and exact_reduction_artifact_path:
        payload_artifacts["exact_reduction_artifact_path"] = (
            exact_reduction_artifact_path
        )
        payload_artifacts["version"] = 2
    return payload


@worker_task
def run_topic_modeling_task(
    configure_worker_environment,
    user_id: str,
    workspace_id: str,
    node_infos: list[dict[str, Any]],
    artifact_dir: str,
    artifact_prefix: str,
    min_topic_size: int = 5,
    workspace_dir: str | None = None,
    corpora: list[list[str]] | None = None,
    random_seed: int = 42,
    representative_words_count: int = 5,
    progress_callback: Callable[[float, str], None] | None = None,
    embedding_cache_dir: str | None = None,
    sample_fractions: list[float | None] | None = None,
    topic_size_mode: str | None = "target",
    topic_size_value: int | None = 25,
) -> dict[str, Any]:
    """Execute topic modeling in a worker process.

    Used by:
    - ``core.worker.topic_modeling_task`` because background jobs need one lifecycle owner for
      submission, progress, cancellation, and artifact cleanup.
    - ``TASK_REGISTRY["topic_modeling"]`` because background jobs need one lifecycle owner for
      submission, progress, cancellation, and artifact cleanup.
        Why:
        - Runs BERTopic embedding/modeling out-of-process and returns an artifact
            manifest (Parquet outputs) for main-process lazy retrieval/finalization.

    Flow: load workspace corpora, choose sampling and embedding settings, reuse embedding
        caches when possible, build topic payloads, and report artifacts back to the task
        manager.
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
            vectorizer_corpora=prepared_payload.vectorizer_corpora,
            artifact_root=prepared_payload.artifact_root,
            artifact_prefix=artifact_prefix,
            random_seed=random_seed,
            representative_words_count=representative_words_count,
            progress_callback=progress_callback,
            embedding_cache_dir=embedding_cache_dir,
            sample_fractions=sample_fractions,
            topic_size_mode=topic_size_mode,
            topic_size_value=topic_size_value,
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
