"""Embedding helpers for the topic-modeling worker pipeline.

Provides embedder selection, per-process caching, chunked encoding, and an
on-disk embedding cache layer.  Kept in a separate module so the embedder
routing constants and ``_get_embedder`` are importable by tests and prefetch
code without pulling in the full BERTopic pipeline.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

logger = logging.getLogger(__name__)

from .worker_tasks_topic_types import _EmbeddedTopicDocuments

_EMBEDDER_CACHE: dict[tuple[str, str], Any] = {}
_EMBEDDING_CHUNK_SIZE = 512

# Pinned MiniLM-L6 embedder for topic modeling. ``scripts/check_model_updates.py``
# is the release-time deliberate bump point.
_TOPIC_EMBEDDER_REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"
_TOPIC_EMBEDDER_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


def _embedder_cache_label(repo_id: str, revision: str | None) -> str:
    """Format the embedder identifier used for on-disk cache keying so the
    same revision string format as before lands in the cache filename.

    Called by:
    - ``_embed_documents`` (this module) to build the ``cache_model_id`` arg
      for ``_embed_with_cache``.

    Flow: load workspace corpora, choose sampling and embedding settings, reuse embedding
        caches when possible, build topic payloads, and report artifacts back to the task
        manager.
    """
    suffix = revision[:8] if revision else "latest"
    return f"{repo_id}@{suffix}"


def _get_embedder(model_id: str, revision: str | None = None):
    """Get or create a cached native SentenceTransformer per worker process.

    ``revision`` is ``None`` for the multilingual embedder until it gets
    pinned at release time; the cache key uses the empty string as a stable
    sentinel so the per-process cache still works.

    Called by:
    - ``_embed_documents`` (this module).
    - Tests that patch ``sentence_transformers.SentenceTransformer`` to verify
      model selection without downloading weights.

    Flow: load workspace corpora, choose sampling and embedding settings, reuse embedding
        caches when possible, build topic payloads, and report artifacts back to the task
        manager.
    """
    cache_key = (model_id, revision or "")
    embedder = _EMBEDDER_CACHE.get(cache_key)
    if embedder is not None:
        return embedder

    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(model_id, revision=revision)

    _EMBEDDER_CACHE[cache_key] = embedder
    return embedder


def _embedder_provider_id(embedder: Any) -> str:
    """Return a stable provider label for native SentenceTransformer caching.

    Called by:
    - ``_embed_with_cache`` (this module) for DuckDB cache partitioning.
    - ``_embed_documents`` (this module) for result metadata.

    Flow: load workspace corpora, choose sampling and embedding settings, reuse embedding
        caches when possible, build topic payloads, and report artifacts back to the task
        manager.
    """
    device = getattr(embedder, "device", None) or getattr(
        embedder, "_target_device", None
    )
    if device is None:
        return "sentence-transformers"
    return f"sentence-transformers:{device}"


def _encode_embeddings_in_chunks(
    embedder: Any,
    docs: list[str],
    *,
    chunk_size: int = _EMBEDDING_CHUNK_SIZE,
    progress_callback: Callable[[float, str], None] | None = None,
    progress_start: float = 0.08,
    progress_end: float = 0.88,
    docs_offset: int = 0,
    total_docs_for_display: int = 0,
    report_every: int = 10,
):
    """Encode documents in fixed-size chunks, reporting progress periodically.

    Called by:
    - ``_embed_with_cache`` (this module) to encode the missing subset.

    Flow: load workspace corpora, choose sampling and embedding settings, reuse embedding
        caches when possible, build topic payloads, and report artifacts back to the task
        manager.
    """
    effective_chunk_size = max(1, int(chunk_size or 0))
    chunk_embeddings: list[Any] = []
    n_chunks = max(1, (len(docs) + effective_chunk_size - 1) // effective_chunk_size)
    total_display = total_docs_for_display or len(docs)

    for chunk_idx, start in enumerate(range(0, len(docs), effective_chunk_size)):
        chunk = docs[start : start + effective_chunk_size]
        chunk_embeddings.append(embedder.encode(chunk, show_progress_bar=False))

        if progress_callback and (chunk_idx + 1) % report_every == 0:
            done_docs = docs_offset + min(start + effective_chunk_size, len(docs))
            cb_frac = progress_start + ((chunk_idx + 1) / n_chunks) * (
                progress_end - progress_start
            )
            pct = int(done_docs / total_display * 100) if total_display > 0 else 0
            progress_callback(
                cb_frac,
                f"Embedding documents... ({done_docs:,} / {total_display:,},  {pct}%)",
            )

    if len(chunk_embeddings) == 1:
        return chunk_embeddings[0]

    import numpy as np

    normalized_chunks = [np.asarray(chunk) for chunk in chunk_embeddings]
    return np.concatenate(normalized_chunks, axis=0)


def _embed_with_cache(
    embedder: Any,
    docs: list[str],
    cache_dir: str | None,
    progress_callback: Callable[[float, str], None] | None,
    progress_start: float = 0.08,
    progress_end: float = 0.88,
    cache_model_id: str | None = None,
) -> Any:
    """Encode docs using the embedder, reading from / writing to disk cache.

    When cache_dir is None the cache is bypassed and all docs are encoded
    directly (preserves the previous behaviour for callers that don't
    pass a cache dir).

    ``cache_model_id`` keys the on-disk cache so two different embedders
    (e.g. EN MiniLM-L6 vs multilingual MiniLM-L12) don't collide on a
    shared cache directory. Defaults to the English embedder label for
    callers that haven't been migrated yet.

    Called by:
    - ``_embed_documents`` (this module).

    Flow: load workspace corpora, choose sampling and embedding settings, reuse embedding
        caches when possible, build topic payloads, and report artifacts back to the task
        manager.
    """
    import numpy as np

    if cache_dir is None:
        if progress_callback:
            progress_callback(progress_start, f"Embedding {len(docs):,} documents...")
        return _encode_embeddings_in_chunks(
            embedder,
            docs,
            progress_callback=progress_callback,
            progress_start=progress_start,
            progress_end=progress_end,
            total_docs_for_display=len(docs),
        )

    from pathlib import Path

    from .embedding_cache import EmbeddingCache

    cache = EmbeddingCache(
        cache_dir=Path(cache_dir),
        # Include revision in the cache key so a bumped embedder version
        # writes to a fresh cache file rather than reusing stale embeddings.
        model_id=cache_model_id
        or _embedder_cache_label(_TOPIC_EMBEDDER_REPO_ID, _TOPIC_EMBEDDER_REVISION),
        provider_id=_embedder_provider_id(embedder),
    )

    cached_embeds, missing_idx = cache.lookup(docs)

    n_cached = len(docs) - len(missing_idx)
    logger.info(
        "[Worker %d] embedding cache: %d/%d hits, %d misses",
        os.getpid(),
        n_cached,
        len(docs),
        len(missing_idx),
    )

    if not missing_idx:
        if progress_callback:
            progress_callback(
                progress_end, f"All {len(docs):,} embeddings loaded from cache."
            )
        return cached_embeds

    if progress_callback:
        progress_callback(
            progress_start,
            f"Embedding {len(missing_idx):,} new documents "
            f"({n_cached:,} loaded from cache)...",
        )

    missed_docs = [docs[i] for i in missing_idx]
    new_embeds = _encode_embeddings_in_chunks(
        embedder,
        missed_docs,
        progress_callback=progress_callback,
        progress_start=progress_start,
        progress_end=progress_end,
        docs_offset=n_cached,
        total_docs_for_display=len(docs),
    )

    cache.store(missed_docs, new_embeds)

    # Reassemble: fill newly computed embeddings into the pre-allocated array.
    dim = new_embeds.shape[1]
    if cached_embeds.shape[1] != dim:
        # First ever run — cached_embeds was zero-width placeholder
        result = np.zeros((len(docs), dim), dtype=np.float32)
        for idx, emb_row in zip(missing_idx, new_embeds):
            result[idx] = emb_row
        # Rows NOT in missing_idx were already cached — re-fetch after store
        # since cached_embeds had wrong width on first run.
        hit_idx = [i for i in range(len(docs)) if i not in set(missing_idx)]
        if hit_idx:
            cached_embeds2, _ = cache.lookup(docs)
            for i in hit_idx:
                result[i] = cached_embeds2[i]
    else:
        result = cached_embeds.copy()
        for slot, emb_row in zip(missing_idx, new_embeds):
            result[slot] = emb_row

    if progress_callback:
        progress_callback(progress_end, "Embedding complete.")

    return result


def _embed_documents(
    *,
    all_docs: list[str],
    embedding_cache_dir: str | None,
    progress_callback: Callable[[float, str], None] | None,
    progress_start: float,
    progress_end: float,
) -> _EmbeddedTopicDocuments:
    """Select and invoke the embedder for ``all_docs``, returning embeddings
    and embedder metadata wrapped in ``_EmbeddedTopicDocuments``.

    Called by:
    - ``_compute_topic_payload`` in ``worker_tasks_topic``.

    Flow: load workspace corpora, choose sampling and embedding settings, reuse embedding
        caches when possible, build topic payloads, and report artifacts back to the task
        manager.
    """
    embedder_repo_id, embedder_revision = (
        _TOPIC_EMBEDDER_REPO_ID,
        _TOPIC_EMBEDDER_REVISION,
    )
    embedder = _get_embedder(embedder_repo_id, embedder_revision)
    embedding_backend = _embedder_provider_id(embedder)
    return _EmbeddedTopicDocuments(
        embedder=embedder,
        all_embeddings=_embed_with_cache(
            embedder,
            all_docs,
            embedding_cache_dir,
            progress_callback,
            progress_start=progress_start,
            progress_end=progress_end,
            cache_model_id=_embedder_cache_label(embedder_repo_id, embedder_revision),
        ),
        embedding_model_name=embedder_repo_id.split("/")[-1],
        embedding_backend=embedding_backend,
    )
