"""Result building and exact-reduction helpers for the topic-modeling worker.

Covers persisting / loading exact-reduction checkpoints, building the
JSON-wire payload with per-node parquet artifacts, empty-payload fallback,
language resolution metadata, and the public ``reaggregate_exact_topic_modeling_result``
entry point used by analysis routes.

Used by:
- ``_compute_topic_payload`` in ``worker_tasks_topic`` for normal and exact
  result building.
- ``reaggregate_exact_topic_modeling_result`` (this module) is called directly
  from the topic-modeling API routes and tests.
"""

from __future__ import annotations

import logging
import os
import pickle
import re
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

from ..api.workspaces.analyses.generated_columns import (
    TOPIC_COLUMN,
    TOPIC_MEANING_COLUMN,
)
from .worker_tasks_topic_types import _SampledTopicCorpora


def _make_reagg_path(old_path: Path) -> Path:
    """Return a fresh unique path beside ``old_path`` for a re-aggregation rewrite.

    Re-aggregation must not overwrite ``old_path`` because previously-detached
    workspace nodes hold lazy ``scan_parquet(old_path)`` references. We keep
    the original directory and the original "base" stem (stripping any prior
    ``_r<hex>`` suffix from earlier re-aggregations to keep names compact) and
    append a fresh short hex suffix.

    Called by:
    - ``_build_topic_result_payload`` (this module) during re-aggregation.

    Flow: load workspace corpora, choose sampling and embedding settings, reuse embedding
        caches when possible, build topic payloads, and report artifacts back to the task
        manager.
    """
    base_stem = re.sub(r"(_r[0-9a-f]+)+$", "", old_path.stem)
    return old_path.parent / f"{base_stem}_r{uuid4().hex[:8]}{old_path.suffix}"


def _persist_exact_reduction_artifact(
    artifact_path: str,
    *,
    topic_model: Any,
    all_docs: list[str],
    corpus_sizes: list[int],
    active_corpora_indices: list[list[int]],
) -> None:
    """Save a serialized snapshot of the fitted topic model plus corpus
    metadata so an exact topic-count reduction can resume from this
    checkpoint without re-fitting.

    Called by:
    - ``_compute_topic_payload`` in ``worker_tasks_topic`` when
      ``topic_size_mode == "exact"``.

    Flow: load workspace corpora, choose sampling and embedding settings, reuse embedding
        caches when possible, build topic payloads, and report artifacts back to the task
        manager.
    """
    artifact_payload: dict[str, Any] = {
        "all_docs": all_docs,
        "corpus_sizes": corpus_sizes,
        "active_corpora_indices": active_corpora_indices,
    }
    if hasattr(topic_model, "save"):
        model_path = f"{artifact_path}.bertopic"
        try:
            topic_model.save(
                model_path,
                serialization="pickle",
                save_embedding_model=False,
            )
        except TypeError:
            topic_model.save(model_path, serialization="pickle")
        artifact_payload["model_path"] = model_path
    else:
        artifact_payload["topic_model"] = topic_model
    with open(artifact_path, "wb") as artifact_file:
        pickle.dump(artifact_payload, artifact_file)


def _load_exact_reduction_artifact(artifact_path: str) -> dict[str, Any]:
    """Load exact reduction artifact data for topic-modeling worker pipeline.

    Called by:
    - ``reaggregate_exact_topic_modeling_result`` (this module).

    Flow: load workspace corpora, choose sampling and embedding settings, reuse embedding
        caches when possible, build topic payloads, and report artifacts back to the task
        manager.
    """
    with open(artifact_path, "rb") as artifact_file:
        loaded = pickle.load(artifact_file)
    if not isinstance(loaded, dict):
        raise ValueError("Exact topic reduction artifact is invalid")
    model_path = loaded.get("model_path")
    if isinstance(model_path, str) and model_path:
        from bertopic import BERTopic

        loaded["topic_model"] = BERTopic.load(model_path)
    return loaded


def _count_non_outlier_topics(topic_model: Any) -> int:
    """Count topics that are not the outlier topic (-1).

    Called by:
    - ``_compute_topic_payload`` in ``worker_tasks_topic`` to report
      ``raw_total_topics``.
    - ``_resolve_exact_reduce_topics_target`` (this module).
    - ``reaggregate_exact_topic_modeling_result`` (this module).

    Flow: load workspace corpora, choose sampling and embedding settings, reuse embedding
        caches when possible, build topic payloads, and report artifacts back to the task
        manager.
    """
    topic_freq_pd = topic_model.get_topic_freq()
    if topic_freq_pd is None or "Topic" not in topic_freq_pd:
        return 0
    return sum(
        1
        for topic_id in topic_freq_pd["Topic"].tolist()
        if isinstance(topic_id, (int, np.integer)) and int(topic_id) != -1
    )


def _has_outlier_topic(topic_model: Any) -> bool:
    """Check whether the fitted topic model includes an outlier topic (-1).

    Called by:
    - ``_resolve_exact_reduce_topics_target`` (this module).

    Flow: load workspace corpora, choose sampling and embedding settings, reuse embedding
        caches when possible, build topic payloads, and report artifacts back to the task
        manager.
    """
    topic_freq_pd = topic_model.get_topic_freq()
    if topic_freq_pd is None or "Topic" not in topic_freq_pd:
        return False
    return any(
        isinstance(topic_id, (int, np.integer)) and int(topic_id) == -1
        for topic_id in topic_freq_pd["Topic"].tolist()
    )


def _resolve_exact_reduce_topics_target(
    topic_model: Any, requested_topic_count: int
) -> int:
    """Map a requested non-outlier topic count to BERTopic's reduction target.

    BERTopic retains the outlier topic (``-1``) in ``nr_topics`` accounting, but the
    UI and the rest of this worker expose only non-outlier topics. When an
    outlier bucket exists, ask BERTopic for one extra topic so the visible topic
    count matches the user's exact selection.

    Called by:
    - ``_compute_topic_payload`` in ``worker_tasks_topic``.
    - ``reaggregate_exact_topic_modeling_result`` (this module).

    Flow: load workspace corpora, choose sampling and embedding settings, reuse embedding
        caches when possible, build topic payloads, and report artifacts back to the task
        manager.
    """
    has_outlier_topic = _has_outlier_topic(topic_model)
    current_total = _count_non_outlier_topics(topic_model) + int(has_outlier_topic)
    requested_total = int(requested_topic_count) + int(has_outlier_topic)
    return min(current_total, requested_total)


def _build_topic_result_payload(
    *,
    topic_model: Any,
    node_infos: list[dict[str, Any]],
    all_docs: list[str],
    corpus_sizes: list[int],
    active_corpora_indices: list[list[int]],
    max_representative_words: int,
    random_state: int,
    assigned_topics: list[int] | None = None,
    artifact_prefix: str | None = None,
    artifact_root: Any | None = None,
    existing_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the full topic-result wire payload: per-node parquet artifacts,
    topic coordinates, labels, and a topic-meanings parquet.

    When ``existing_artifacts`` is supplied (re-aggregation path), writes new
    parquet files alongside the old ones (never overwrites) and records
    superseded paths for later cleanup.

    Called by:
    - ``_compute_topic_payload`` in ``worker_tasks_topic`` for the initial run.
    - ``reaggregate_exact_topic_modeling_result`` (this module) for the
      re-aggregation path.

    Flow: load workspace corpora, choose sampling and embedding settings, reuse embedding
        caches when possible, build topic payloads, and report artifacts back to the task
        manager.
    """
    from bertopic._utils import select_topic_representation

    # Tracks parquet paths that this call is replacing during re-aggregation.
    # Previously-detached workspace nodes still hold lazy references to the
    # superseded paths, so we keep the old files on disk and only record them
    # in the manifest. The existing task-cleanup helper walks the manifest
    # tree to delete every ``*_path`` / ``*_parquet_path`` key it finds, so listing
    # them here under ``superseded_artifacts`` lets cleanup reclaim the disk
    # whenever the task is finally cleared.
    newly_superseded: list[dict[str, str]] = []
    if existing_artifacts is None:
        if artifact_prefix is None or artifact_root is None:
            raise ValueError("artifact_prefix and artifact_root are required")
        artifact_root_path = Path(artifact_root)
        topic_meanings_path = (
            artifact_root_path / f"{artifact_prefix}_topic_meanings.parquet"
        )
        existing_node_artifacts: list[dict[str, Any]] = []
    else:
        old_topic_meanings_path = Path(
            str(existing_artifacts.get("topic_meanings_parquet_path") or "")
        )
        if not str(old_topic_meanings_path):
            raise ValueError("Topic meanings artifact path is missing")
        # Re-aggregation must not overwrite the meanings parquet — the previous
        # detach's ``topic_meanings`` node scans it lazily.
        topic_meanings_path = _make_reagg_path(old_topic_meanings_path)
        newly_superseded.append(
            {"topic_meanings_parquet_path": str(old_topic_meanings_path)}
        )
        node_payloads = existing_artifacts.get("nodes")
        if not isinstance(node_payloads, list):
            raise ValueError("Topic assignment artifacts are missing")
        existing_node_artifacts = [
            payload for payload in node_payloads if isinstance(payload, dict)
        ]
        if len(existing_node_artifacts) != len(node_infos):
            raise ValueError("Topic assignment artifacts do not match the request")

    if assigned_topics is None:
        assigned_topics = list(getattr(topic_model, "topics_", []))
    if len(assigned_topics) != len(all_docs):
        raise ValueError("Topic assignments do not match the document count")

    assignments: list[list[int]] = []
    node_artifacts: list[dict[str, Any]] = []
    offset = 0
    for idx, size in enumerate(corpus_sizes):
        end = offset + size
        corpus_topics = assigned_topics[offset:end]
        normalized_topics = [
            int(topic_id) if isinstance(topic_id, (int, np.integer)) else -1
            for topic_id in corpus_topics
        ]
        assignments.append(normalized_topics)

        if existing_artifacts is None:
            node_id = str(node_infos[idx]["node_id"])
            node_name = str(node_infos[idx].get("node_name") or node_id)
            text_column = str(node_infos[idx].get("text_column") or "")
            original_columns = list(node_infos[idx].get("original_columns") or [])
            assignments_path = (
                artifact_root_path
                / f"{artifact_prefix}_topic_assignments_{node_id}.parquet"
            )
        else:
            existing_node = existing_node_artifacts[idx]
            node_id = str(existing_node.get("node_id") or node_infos[idx]["node_id"])
            node_name = str(
                existing_node.get("node_name")
                or node_infos[idx].get("node_name")
                or node_id
            )
            text_column = str(
                existing_node.get("text_column")
                or node_infos[idx].get("text_column")
                or ""
            )
            original_columns = list(
                existing_node.get("original_columns")
                or node_infos[idx].get("original_columns")
                or []
            )
            old_assignments_path = Path(
                str(existing_node.get("assignments_parquet_path") or "")
            )
            if not str(old_assignments_path):
                raise ValueError("Topic assignment artifact path is missing")
            # Re-aggregation must not overwrite the assignments parquet — the
            # previous detach's lazy plan still scans it.
            assignments_path = _make_reagg_path(old_assignments_path)
            newly_superseded.append(
                {"assignments_parquet_path": str(old_assignments_path)}
            )

        pl.DataFrame(
            {
                "__row_nr__": active_corpora_indices[idx],
                TOPIC_COLUMN: normalized_topics,
            }
        ).with_columns(
            [
                pl.col("__row_nr__").cast(pl.Int64),
                pl.col(TOPIC_COLUMN).cast(pl.Int64),
            ]
        ).lazy().sink_parquet(assignments_path)
        node_artifacts.append(
            {
                "node_id": node_id,
                "node_name": node_name,
                "text_column": text_column,
                "original_columns": original_columns,
                "assignments_parquet_path": str(assignments_path),
            }
        )
        offset = end

    per_corpus_topic_counts: list[dict[int, int]] = []
    for corpus_topics in assignments:
        counts: dict[int, int] = {}
        for topic_id in corpus_topics:
            counts[topic_id] = counts.get(topic_id, 0) + 1
        per_corpus_topic_counts.append(counts)

    topic_freq_pd = topic_model.get_topic_freq()
    topic_freq = (
        cast(pl.DataFrame, pl.from_pandas(topic_freq_pd))
        if topic_freq_pd is not None
        else pl.DataFrame(schema={"Topic": pl.Int64})
    )

    topic_ids: list[int] = []
    if "Topic" in topic_freq.columns and not topic_freq.is_empty():
        topic_series = topic_freq.get_column("Topic")
        topic_ids = [
            int(topic_id)
            for topic_id in topic_series.to_list()
            if isinstance(topic_id, (int, np.integer)) and int(topic_id) != -1
        ]

    topics_by_id = cast(dict[int, list[tuple[str, float]]], topic_model.get_topics())
    # Wire-payload slice — matches BERTopic's actual ``top_n_words`` at fit
    # time, so the frontend can scale "Words per topic" up to this cap
    # post-fit without a re-run. Keeping ``representative_words`` narrower
    # than the fit's capacity would make the slider's upper range a lie:
    # the user would dial it up and see no new words because there'd be
    # nothing to reveal.
    from .worker_tasks_topic_pipeline import _resolve_top_n_words

    payload_words_cap = _resolve_top_n_words(max_representative_words)
    payload_representative_words_by_topic: list[list[str]] = []
    # Meaning-column slice — respects the user's chosen display count so
    # the "Add to Workspace" parquet matches what they saw. The
    # server-side label fallback uses the same narrow slice for the same
    # reason: if the frontend can't build its own label, we want a label
    # that reflects the user's intent, not the full top_n_words buffer.
    meaning_words_by_topic: list[list[str]] = []
    labels: list[str] = []
    for topic_id in topic_ids:
        words = topics_by_id.get(topic_id, [])
        payload_words = [
            word
            for word, _score in words[:payload_words_cap]
            if isinstance(word, str) and word
        ]
        payload_representative_words_by_topic.append(payload_words)
        meaning_words = payload_words[:max_representative_words]
        meaning_words_by_topic.append(meaning_words)
        labels.append(
            " | ".join(meaning_words) if meaning_words else f"Topic {topic_id}"
        )

    all_topics_sorted = sorted(topics_by_id.keys())
    indices = (
        np.array([all_topics_sorted.index(topic_id) for topic_id in topic_ids])
        if topic_ids
        else np.array([])
    )

    embeddings, c_tfidf_used = select_topic_representation(
        topic_model.c_tf_idf_,
        topic_model.topic_embeddings_,
        output_ndarray=True,
    )
    if len(indices) > 0:
        embeddings = embeddings[indices]
    else:
        embeddings = np.zeros((0, 2))

    if embeddings.shape[0] == 0:
        coords = embeddings
    elif embeddings.shape[0] == 1:
        coords = np.array([[0.0, 0.0]])
    elif embeddings.shape[0] <= 15:
        from sklearn.decomposition import PCA

        comps = min(2, embeddings.shape[1])
        projected = PCA(n_components=comps, random_state=random_state).fit_transform(
            embeddings
        )
        if comps == 1:
            coords = np.column_stack([projected[:, 0], np.zeros_like(projected[:, 0])])
        else:
            coords = projected
    else:
        try:
            from umap import UMAP

            n_samples = embeddings.shape[0]
            n_neighbors = max(2, min(15, n_samples - 2))
            if c_tfidf_used:
                from sklearn.preprocessing import MinMaxScaler

                normalized = MinMaxScaler().fit_transform(embeddings)
                coords = UMAP(
                    n_neighbors=n_neighbors,
                    n_components=2,
                    metric="hellinger",
                    random_state=random_state,
                ).fit_transform(normalized)
            else:
                coords = UMAP(
                    n_neighbors=n_neighbors,
                    n_components=2,
                    metric="cosine",
                    random_state=random_state,
                ).fit_transform(embeddings)
        except (
            ImportError,
            ModuleNotFoundError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as umap_error:
            logger.warning(
                "[Worker %d] UMAP failed: %s. Falling back to PCA.",
                os.getpid(),
                umap_error,
            )
            from sklearn.decomposition import PCA

            comps = min(2, embeddings.shape[1])
            projected = PCA(
                n_components=comps, random_state=random_state
            ).fit_transform(embeddings)
            if comps == 1:
                coords = np.column_stack(
                    [projected[:, 0], np.zeros_like(projected[:, 0])]
                )
            else:
                coords = projected

    topic_payloads = []
    for i, topic_id in enumerate(topic_ids):
        per_sizes = [
            per_corpus_topic_counts[j].get(topic_id, 0)
            for j in range(len(per_corpus_topic_counts))
        ]
        topic_payloads.append(
            {
                "id": topic_id,
                "label": labels[i] if i < len(labels) else f"Topic {topic_id}",
                "representative_words": payload_representative_words_by_topic[i]
                if i < len(payload_representative_words_by_topic)
                else [],
                "size": per_sizes,
                "total_size": int(sum(per_sizes)),
                "x": float(coords[i, 0]) if i < len(coords) else 0.0,
                "y": float(coords[i, 1]) if i < len(coords) else 0.0,
            }
        )

    pl.DataFrame(
        {
            TOPIC_COLUMN: topic_ids,
            TOPIC_MEANING_COLUMN: meaning_words_by_topic,
        },
        schema={
            TOPIC_COLUMN: pl.Int64,
            TOPIC_MEANING_COLUMN: pl.List(pl.String),
        },
    ).lazy().sink_parquet(topic_meanings_path)

    artifacts: dict[str, Any] = {
        "version": 1,
        "topic_meanings_parquet_path": str(topic_meanings_path),
        "nodes": node_artifacts,
    }
    if existing_artifacts is not None:
        exact_artifact_path = existing_artifacts.get("exact_reduction_artifact_path")
        if isinstance(exact_artifact_path, str) and exact_artifact_path:
            artifacts["exact_reduction_artifact_path"] = exact_artifact_path
            artifacts["version"] = max(2, int(existing_artifacts.get("version") or 1))
        # Carry forward paths superseded by earlier re-aggregations so they
        # remain in the manifest tree and are reclaimed by task cleanup.
        prior_superseded = existing_artifacts.get("superseded_artifacts")
        carried = (
            [item for item in prior_superseded if isinstance(item, dict)]
            if isinstance(prior_superseded, list)
            else []
        )
        combined = carried + newly_superseded
        if combined:
            artifacts["superseded_artifacts"] = combined

    return {
        "topics": topic_payloads,
        "corpus_sizes": corpus_sizes,
        "per_corpus_topic_counts": per_corpus_topic_counts,
        "artifacts": artifacts,
        "meta": {
            "embeddings_from_ctfidf": bool(c_tfidf_used),
            "total_topics_incl_outlier": int(topic_freq.height),
        },
    }


def reaggregate_exact_topic_modeling_result(
    *,
    artifact_path: str,
    existing_artifacts: dict[str, Any],
    node_infos: list[dict[str, Any]],
    topic_size_value: int,
    representative_words_count: int,
    random_seed: int,
) -> dict[str, Any]:
    """Re-aggregate a previously-fit topic model to an exact topic count.

    Used by:
    - Backend API routes, backend tests, core workspace and worker services because they
      need a backend boundary that validates inputs before delegating to workspace or worker
      state.

    Flow: load workspace corpora, choose sampling and embedding settings, reuse embedding
        caches when possible, build topic payloads, and report artifacts back to the task
        manager.
    """
    stored = _load_exact_reduction_artifact(artifact_path)
    topic_model = stored.get("topic_model")
    all_docs = stored.get("all_docs")
    corpus_sizes = stored.get("corpus_sizes")
    active_corpora_indices = stored.get("active_corpora_indices")

    if topic_model is None or not isinstance(all_docs, list):
        raise ValueError("Exact topic reduction artifact is incomplete")
    if not isinstance(corpus_sizes, list) or not isinstance(
        active_corpora_indices, list
    ):
        raise ValueError("Exact topic reduction artifact is missing corpus metadata")

    raw_total_topics = _count_non_outlier_topics(topic_model)
    requested_topic_count = int(topic_size_value)
    if raw_total_topics < 2:
        raise ValueError("Exact topic reduction requires at least two raw topics")
    if requested_topic_count < 2 or requested_topic_count > raw_total_topics:
        raise ValueError(f"Exact topic count must be between 2 and {raw_total_topics}")

    topic_model.reduce_topics(
        all_docs,
        nr_topics=_resolve_exact_reduce_topics_target(
            topic_model, requested_topic_count
        ),
    )
    payload = _build_topic_result_payload(
        topic_model=topic_model,
        node_infos=node_infos,
        all_docs=all_docs,
        corpus_sizes=[int(size) for size in corpus_sizes],
        active_corpora_indices=[
            list(map(int, indices)) for indices in active_corpora_indices
        ],
        max_representative_words=max(1, int(representative_words_count)),
        random_state=int(random_seed),
        existing_artifacts=existing_artifacts,
    )
    payload_meta = payload.get("meta")
    if not isinstance(payload_meta, dict):
        payload_meta = {}
    payload_meta["raw_total_topics"] = raw_total_topics
    payload["meta"] = payload_meta
    return payload


def _build_empty_topic_payload(
    *,
    sampled: _SampledTopicCorpora,
    node_infos: list[dict[str, Any]],
    artifact_root: Path,
    artifact_prefix: str,
) -> dict[str, Any]:
    """Build a valid but empty topic result when there are no documents to
    model (e.g. all sampled corpora reduced to zero documents).

    Called by:
    - ``_compute_topic_payload`` in ``worker_tasks_topic``.

    Flow: load workspace corpora, choose sampling and embedding settings, reuse embedding
        caches when possible, build topic payloads, and report artifacts back to the task
        manager.
    """
    topic_meanings_path = artifact_root / f"{artifact_prefix}_topic_meanings.parquet"
    pl.DataFrame(
        schema={
            TOPIC_COLUMN: pl.Int64,
            TOPIC_MEANING_COLUMN: pl.List(pl.String),
        }
    ).lazy().sink_parquet(topic_meanings_path)

    node_artifacts: list[dict[str, Any]] = []
    for index, _corpus in enumerate(sampled.active_corpora):
        node_id = str(node_infos[index]["node_id"])
        node_name = str(node_infos[index].get("node_name") or node_id)
        text_column = str(node_infos[index].get("text_column") or "")
        original_columns = list(node_infos[index].get("original_columns") or [])
        assignments_path = (
            artifact_root / f"{artifact_prefix}_topic_assignments_{node_id}.parquet"
        )
        pl.DataFrame(
            {
                "__row_nr__": sampled.active_corpora_indices[index],
                TOPIC_COLUMN: [],
            }
        ).with_columns(
            [
                pl.col("__row_nr__").cast(pl.Int64),
                pl.col(TOPIC_COLUMN).cast(pl.Int64),
            ]
        ).lazy().sink_parquet(assignments_path)
        node_artifacts.append(
            {
                "node_id": node_id,
                "node_name": node_name,
                "text_column": text_column,
                "original_columns": original_columns,
                "assignments_parquet_path": str(assignments_path),
            }
        )

    return {
        "topics": [],
        "corpus_sizes": sampled.corpus_sizes,
        "per_corpus_topic_counts": [],
        "artifacts": {
            "version": 1,
            "topic_meanings_parquet_path": str(topic_meanings_path),
            "nodes": node_artifacts,
        },
        "meta": {},
    }
