"""Pipeline building helpers for the topic-modeling worker.

Encapsulates BERTopic pipeline construction (vectorizer, UMAP, HDBSCAN),
corpus sampling, and parameter resolution so the top-level orchestrator
stays focused on coordination.

Used by:
- ``_compute_topic_payload`` in ``worker_tasks_topic`` delegates sampling and
  pipeline execution to functions in this module.
- Tests that verify stopword handling, label vectorizer config, and topic size
  arithmetic import directly from here.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any, Callable, cast

import polars as pl

logger = logging.getLogger(__name__)

from .worker_tasks_topic_types import _SampledTopicCorpora, _TopicPipelineRun


def _sample_corpus(
    docs: list[str], fraction: float, seed: int
) -> tuple[list[str], list[int]]:
    """Return a reproducible random sample of docs and their original indices.

    Uses the same Polars expression as the preprocessing slice tool
    (``pl.int_range(...).sample(fraction=..., seed=...)``) so identical
    ``(seed, fraction)`` parameters select identical rows across tools.
    Operates on an in-memory integer Series; no parquet artifact is created.

    Called by:
    - ``_sample_corpora_for_topic_modeling`` (this module) for each corpus.
    - Tests that verify deterministic sampling across seeds and fractions.

    Flow: load workspace corpora, choose sampling and embedding settings, reuse embedding
        caches when possible, build topic payloads, and report artifacts back to the task
        manager.
    """
    if fraction >= 1.0:
        return docs, list(range(len(docs)))
    indices = (
        pl.int_range(len(docs), eager=True)
        .sample(fraction=fraction, seed=seed)
        .sort()
        .to_list()
    )
    if not indices:
        # Polars floors fraction*N, so a tiny corpus with very small fraction
        # can yield zero rows. Topic modelling needs at least one document.
        return [docs[0]], [0]
    return [docs[i] for i in indices], indices


def _sample_corpora_for_topic_modeling(
    *,
    corpora: list[list[str]],
    vectorizer_corpora: list[list[str] | None],
    sample_fractions: list[float | None] | None,
    random_seed: int,
) -> _SampledTopicCorpora:
    """Sample each corpus according to ``sample_fractions`` and flatten into
    a single document list for pipeline ingestion.

    Called by:
    - ``_compute_topic_payload`` in ``worker_tasks_topic``.

    Flow: load workspace corpora, choose sampling and embedding settings, reuse embedding
        caches when possible, build topic payloads, and report artifacts back to the task
        manager.
    """
    corpus_sizes_before_sample = [len(corpus) for corpus in corpora]
    active_corpora: list[list[str]] = []
    active_corpora_indices: list[list[int]] = []
    active_vectorizer_corpora: list[list[str] | None] = []

    if sample_fractions is not None:
        for index, corpus in enumerate(corpora):
            fraction = (
                sample_fractions[index] if index < len(sample_fractions) else None
            )
            if fraction is not None and 0.0 < fraction < 1.0:
                sampled_docs, sampled_indices = _sample_corpus(
                    corpus, fraction, random_seed + index
                )
                active_corpora.append(sampled_docs)
                active_corpora_indices.append(sampled_indices)
                vectorizer_corpus = vectorizer_corpora[index]
                active_vectorizer_corpora.append(
                    [vectorizer_corpus[row_index] for row_index in sampled_indices]
                    if vectorizer_corpus is not None
                    else None
                )
            else:
                active_corpora.append(corpus)
                active_corpora_indices.append(list(range(len(corpus))))
                active_vectorizer_corpora.append(vectorizer_corpora[index])
    else:
        active_corpora = list(corpora)
        active_corpora_indices = [list(range(len(corpus))) for corpus in corpora]
        active_vectorizer_corpora = list(vectorizer_corpora)

    all_docs = [doc for corpus in active_corpora for doc in corpus]
    all_docs_for_vectorizer: list[str] = []
    any_pretokenised = False
    for index, raw_corpus in enumerate(active_corpora):
        vectorizer_corpus = active_vectorizer_corpora[index]
        if vectorizer_corpus is not None:
            any_pretokenised = True
            all_docs_for_vectorizer.extend(vectorizer_corpus)
        else:
            all_docs_for_vectorizer.extend(raw_corpus)

    return _SampledTopicCorpora(
        corpus_sizes_before_sample=corpus_sizes_before_sample,
        active_corpora=active_corpora,
        active_corpora_indices=active_corpora_indices,
        active_vectorizer_corpora=active_vectorizer_corpora,
        all_docs=all_docs,
        all_docs_for_vectorizer=all_docs_for_vectorizer,
        any_pretokenised=any_pretokenised,
        corpus_sizes=[len(corpus) for corpus in active_corpora],
    )


def _compute_min_topic_size(
    n_eff: int,
    topic_size_mode: str,
    topic_size_value: int,
) -> int:
    """Derive BERTopic min_topic_size from the chosen sizing mode.

    Args:
        n_eff: Effective document count (post-sample total across all corpora).
        topic_size_mode: "target", "min", or "exact".
        topic_size_value: The user-supplied numeric value for the chosen mode.

    Called by:
    - ``_compute_topic_payload`` in ``worker_tasks_topic``.
    - Tests that verify the arithmetic for each mode.

    Flow: load workspace corpora, choose sampling and embedding settings, reuse embedding
        caches when possible, build topic payloads, and report artifacts back to the task
        manager.
    """
    if topic_size_mode == "min":
        return max(2, int(topic_size_value))
    if topic_size_mode == "exact":
        # Start from the target-mode heuristic, then reduce it so BERTopic is
        # more likely to produce enough raw topics before exact post-fit merging.
        target_min_topic_size = max(2, n_eff // (int(topic_size_value) * 10))
        return max(5, int(target_min_topic_size * 0.75))
    # "target" (default)
    return max(2, n_eff // (int(topic_size_value) * 10))


def _build_label_vectorizer(*, online: bool = False) -> Any:
    """Return the CountVectorizer (or OnlineCountVectorizer) used for the
    label/c-TF-IDF stage with sklearn's built-in English stoplist.

    Called by:
    - ``_build_classic_pipeline`` (this module).

    Flow: load workspace corpora, choose sampling and embedding settings, reuse embedding
        caches when possible, build topic payloads, and report artifacts back to the task
        manager.
    """
    if online:
        from bertopic.vectorizers import OnlineCountVectorizer

        return OnlineCountVectorizer(stop_words="english", decay=0.01)
    from sklearn.feature_extraction.text import CountVectorizer

    return CountVectorizer(stop_words="english")


def _resolve_top_n_words(representative_words_count: int | None) -> int:
    """Pick BERTopic's ``top_n_words`` from the user-requested display cap.

    BERTopic's default is 10. When the user picks "Words per topic = 35"
    and toggles on the frontend stopword filter, c-TF-IDF would compute
    only 10 raw words, the filter would drop 5--9 of them as CJK function
    words (的/是/了/...), and the user would see 1--3 --- even though they
    asked for 35.

    We pre-compute a generous headroom so the post-filter slice still has
    enough material:

    - At least 50 candidates, so even a tiny request like 5 has a healthy
      buffer for the stopword filter.
    - Otherwise 2× the requested cap.

    Performance impact is negligible --- c-TF-IDF already produces a ranked
    vocabulary per topic; ``top_n_words`` just decides where to truncate.

    Called by:
    - ``_build_classic_pipeline`` (this module).
    - ``_compute_topic_payload`` in ``worker_tasks_topic`` uses it for the
      display cap in result payloads.

    Flow: load workspace corpora, choose sampling and embedding settings, reuse embedding
        caches when possible, build topic payloads, and report artifacts back to the task
        manager.
    """
    requested = int(representative_words_count or 0)
    return max(50, requested * 2) if requested > 0 else 50


def _build_classic_pipeline(
    min_topic_size: int,
    random_state: int,
    embedder: Any,
    top_n_words: int = 50,
) -> Any:
    """Build a standard BERTopic pipeline with UMAP + HDBSCAN.

    Called by:
    - ``_run_classic_pipeline`` (this module).

    Flow: load workspace corpora, choose sampling and embedding settings, reuse embedding
        caches when possible, build topic payloads, and report artifacts back to the task
        manager.
    """
    from bertopic import BERTopic
    from umap import UMAP

    return BERTopic(
        verbose=False,
        min_topic_size=min_topic_size,
        embedding_model=embedder,
        umap_model=UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=random_state,
        ),
        vectorizer_model=_build_label_vectorizer(online=False),
        top_n_words=top_n_words,
    )


def _run_classic_pipeline(
    *,
    all_docs_for_vectorizer: list[str],
    all_embeddings: Any,
    effective_min_topic_size: int,
    random_state: int,
    embedder: Any,
    top_n_words: int,
    progress_callback: Callable[[float, str], None] | None,
    progress_fraction: float,
) -> _TopicPipelineRun:
    """Build the classic BERTopic pipeline, fit it, and return the model and
    topic assignments wrapped in ``_TopicPipelineRun``.

    Called by:
    - ``_compute_topic_payload`` in ``worker_tasks_topic``.

    Flow: load workspace corpora, choose sampling and embedding settings, reuse embedding
        caches when possible, build topic payloads, and report artifacts back to the task
        manager.
    """
    if progress_callback:
        progress_callback(
            progress_fraction, "Running classic BERTopic pipeline (UMAP + HDBSCAN)..."
        )
    topic_model = _build_classic_pipeline(
        effective_min_topic_size,
        random_state,
        embedder,
        top_n_words=top_n_words,
    )
    assigned_topics, _ = topic_model.fit_transform(
        all_docs_for_vectorizer, all_embeddings
    )
    return _TopicPipelineRun(
        topic_model=topic_model, assigned_topics=list(assigned_topics)
    )
