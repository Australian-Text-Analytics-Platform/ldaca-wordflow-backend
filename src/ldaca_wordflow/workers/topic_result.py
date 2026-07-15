"""Result construction for the topic-modeling worker.

Consumes the dict result of the Rust ``polars-text`` topic-modeling expression
and turns it into the wire payload the API and frontend expect: per-node
``__row_nr__`` -> ``TOPIC_topic`` assignment parquet files, a shared
``TOPIC_topic`` -> ``TOPIC_topic_meaning`` meanings parquet, per-corpus topic
counts, and topic bubble-chart records
(``id``/``label``/``representative_words``/``size``/``total_size``/``x``/``y``).

Used by:
- ``_compute_topic_payload`` in ``topic_modeling`` for result building.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..analysis.generated_columns import (
    TOPIC_COLUMN,
    TOPIC_DISTRIBUTION_COLUMN,
    TOPIC_MEANING_COLUMN,
)
from .topic_pipeline import (
    _resolve_top_n_words,
)
from .topic_types import _SampledTopicCorpora

logger = logging.getLogger(__name__)


def _dominant_topics_by_doc_index(
    documents: list[dict[str, Any]], total_docs: int
) -> list[int]:
    """Flatten Rust ``documents[]`` into a per-doc dominant-topic list.

    Rust returns one record per input document carrying ``doc_index`` (its
    position in the flattened ``all_docs`` order) and ``dominant_topic`` (a topic
    id, or ``-1`` for an all-outlier document). We index by ``doc_index`` so the
    result lines up positionally with ``all_docs`` regardless of record order.

    Called by:
    - ``_build_topic_result_payload`` (this module).
    """
    dominant = [-1] * total_docs
    for doc in documents:
        try:
            doc_index = int(doc["doc_index"])
        except KeyError, TypeError, ValueError:
            continue
        if 0 <= doc_index < total_docs:
            raw = doc.get("dominant_topic", -1)
            dominant[doc_index] = int(raw) if isinstance(raw, (int, np.integer)) else -1
    return dominant


def _distribution_by_doc_index(
    documents: list[dict[str, Any]], total_docs: int
) -> list[list[dict[str, Any]]]:
    """Flatten Rust ``documents[]`` into per-doc topic-distribution lists.

    Mirrors :func:`_dominant_topics_by_doc_index` but extracts the soft
    ``topic_distribution`` (``[{topic_id, proportion}, ...]``) so it can be
    written into the assignment parquet for the detach-time distribution filter.
    Documents missing a distribution (or out of range) get an empty list, which
    the filter treats as proportion 0 for every topic.

    Called by:
    - ``_build_topic_result_payload`` (this module).
    """
    distributions: list[list[dict[str, Any]]] = [[] for _ in range(total_docs)]
    for doc in documents:
        try:
            doc_index = int(doc["doc_index"])
        except KeyError, TypeError, ValueError:
            continue
        if not (0 <= doc_index < total_docs):
            continue
        entries = doc.get("topic_distribution") or []
        normalized: list[dict[str, Any]] = []
        for entry in entries:
            try:
                normalized.append(
                    {
                        "topic_id": int(entry["topic_id"]),
                        "proportion": float(entry["proportion"]),
                    }
                )
            except KeyError, TypeError, ValueError:
                continue
        distributions[doc_index] = normalized
    return distributions


def _build_topic_result_payload(
    *,
    rust_result: dict[str, Any],
    node_infos: list[dict[str, Any]],
    corpus_sizes: list[int],
    active_corpora_indices: list[list[int]],
    max_representative_words: int,
    artifact_prefix: str,
    artifact_root: Any,
) -> dict[str, Any]:
    """Turn the Rust pipeline result dict into the topic-result wire payload.

    Writes one ``__row_nr__`` -> ``TOPIC_topic`` assignment parquet per node and a
    shared meanings parquet, computes per-corpus topic counts (hard dominant-topic
    counts), and assembles topic bubble-chart records using Rust-provided PaCMAP
    ``x``/``y`` coordinates (no Python-side PCA/UMAP projection needed).

    Called by:
    - ``_compute_topic_payload`` in ``topic_modeling`` for the run.
    """
    documents = list(rust_result.get("documents") or [])
    rust_topics = list(rust_result.get("topics") or [])

    if artifact_prefix is None or artifact_root is None:
        raise ValueError("artifact_prefix and artifact_root are required")
    artifact_root_path = Path(artifact_root)
    topic_meanings_path = (
        artifact_root_path / f"{artifact_prefix}_topic_meanings.parquet"
    )

    total_docs = sum(int(size) for size in corpus_sizes)
    dominant_by_index = _dominant_topics_by_doc_index(documents, total_docs)
    distribution_by_index = _distribution_by_doc_index(documents, total_docs)
    # Polars dtype for the persisted per-row distribution column.
    distribution_dtype = pl.List(
        pl.Struct({"topic_id": pl.Int64, "proportion": pl.Float64})
    )

    assignments: list[list[int]] = []
    node_artifacts: list[dict[str, Any]] = []
    offset = 0
    for idx, size in enumerate(corpus_sizes):
        end = offset + size
        normalized_topics = [
            int(topic_id) for topic_id in dominant_by_index[offset:end]
        ]
        assignments.append(normalized_topics)
        corpus_distribution = distribution_by_index[offset:end]

        node_id = str(node_infos[idx]["node_id"])
        node_name = str(node_infos[idx].get("node_name") or node_id)
        text_column = str(node_infos[idx].get("text_column") or "")
        original_columns = list(node_infos[idx].get("original_columns") or [])
        assignments_path = (
            artifact_root_path
            / f"{artifact_prefix}_topic_assignments_{node_id}.parquet"
        )

        pl.DataFrame(
            {
                "__row_nr__": active_corpora_indices[idx],
                TOPIC_COLUMN: normalized_topics,
                TOPIC_DISTRIBUTION_COLUMN: pl.Series(
                    TOPIC_DISTRIBUTION_COLUMN,
                    corpus_distribution,
                    dtype=distribution_dtype,
                ),
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

    payload_words_cap = _resolve_top_n_words(max_representative_words)
    topic_ids: list[int] = []
    payload_representative_words_by_topic: list[list[str]] = []
    meaning_words_by_topic: list[list[str]] = []
    labels: list[str] = []
    coords_by_topic: dict[int, tuple[float, float]] = {}
    for topic in rust_topics:
        try:
            topic_id = int(topic["id"])
        except KeyError, TypeError, ValueError:
            continue
        if topic_id < 0:
            continue
        topic_ids.append(topic_id)
        words = [
            word
            for word in topic.get("representative_words") or []
            if isinstance(word, str) and word
        ]
        payload_words = words[:payload_words_cap]
        payload_representative_words_by_topic.append(payload_words)
        # Respect the requested display count so the detached Data Block matches
        # the visible result. The default label uses the same narrow slice.
        meaning_words = payload_words[:max_representative_words]
        meaning_words_by_topic.append(meaning_words)
        labels.append(
            " | ".join(meaning_words) if meaning_words else f"Topic {topic_id}"
        )
        coords_by_topic[topic_id] = (
            float(topic.get("x") or 0.0),
            float(topic.get("y") or 0.0),
        )

    topic_payloads = []
    for i, topic_id in enumerate(topic_ids):
        per_sizes = [
            per_corpus_topic_counts[j].get(topic_id, 0)
            for j in range(len(per_corpus_topic_counts))
        ]
        x, y = coords_by_topic.get(topic_id, (0.0, 0.0))
        topic_payloads.append(
            {
                "id": topic_id,
                "label": labels[i] if i < len(labels) else f"Topic {topic_id}",
                "representative_words": payload_representative_words_by_topic[i]
                if i < len(payload_representative_words_by_topic)
                else [],
                "size": per_sizes,
                "total_size": int(sum(per_sizes)),
                "x": x,
                "y": y,
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

    has_outlier = any(topic_id == -1 for corpus in assignments for topic_id in corpus)
    n_topics = int(rust_result.get("n_topics") or len(topic_ids))

    return {
        "topics": topic_payloads,
        "corpus_sizes": corpus_sizes,
        "per_corpus_topic_counts": per_corpus_topic_counts,
        "artifacts": artifacts,
        "meta": {
            "embeddings_from_ctfidf": False,
            "total_topics_incl_outlier": n_topics + (1 if has_outlier else 0),
        },
    }


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
    - ``_compute_topic_payload`` in ``topic_modeling``.
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
