"""Background prefetch of heavy ML models at startup.

Runs downloads in a daemon thread so the server starts accepting requests
immediately while models are fetched in the background. The first user who
touches the relevant feature (quotation extraction, topic modelling) gets
to skip the cold-download wait if the prefetch has already finished.

Used by:
- Backend API routes, worker tasks, workspace services, and backend tests because they
  need a backend boundary that validates inputs before delegating to workspace or worker
  state.

Flow: normalize inputs, delegate to the owning backend state or service boundary, and
    return serialized values or existing domain errors to callers.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

# Default ONNX embedder repo the Rust topic-modeling pipeline loads when no
# override is supplied. Mirrors ``_DEFAULT_EMBEDDER_MODEL`` in
# ``worker_tasks_topic``; kept as a literal here to avoid importing the heavy
# worker module just to prefetch.
_TOPIC_EMBEDDER_REPO_ID = "onnx-community/all-MiniLM-L6-v2-ONNX"


def _prefetch_spacy_model() -> None:
    """Download the spaCy model used by quotation extraction if not cached.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need a
      backend boundary that validates inputs before delegating to workspace or worker state.

    Flow: normalize inputs, delegate to the owning backend state or service boundary, and
        return serialized values or existing domain errors to callers.
    """
    try:
        from .quotation_extractor import (
            _download_spacy_model_to_cache,
            _get_cached_spacy_model_dir,
        )

        cached_dir = _get_cached_spacy_model_dir()
        if (cached_dir / "config.cfg").exists():
            logger.info("[prefetch] spaCy model already cached at %s", cached_dir)
            return

        logger.info("[prefetch] Downloading spaCy model in background...")
        _download_spacy_model_to_cache()
        logger.info("[prefetch] spaCy model download complete")
    except Exception:
        logger.warning("[prefetch] spaCy model prefetch failed", exc_info=True)


def _prefetch_topic_embedder() -> None:
    """Download the ONNX topic embedder used by the Rust pipeline if not cached.

    Delegates to ``polars_text._internal.prefetch_embedder``, which downloads the
    ONNX weights + tokenizer for the default model into the
    shared Hugging Face cache so the first ``run_topic_modeling`` call skips the
    cold-load. Best-effort: failures (e.g. offline with no cache) are logged and
    swallowed because prefetch is an optimization, not a hard dependency.

    Called by:
    - ``_run_all_prefetches`` (this module) on the background daemon thread.
    """
    try:
        from polars_text._internal import prefetch_embedder

        logger.info(
            "[prefetch] Downloading ORT embedder %s...",
            _TOPIC_EMBEDDER_REPO_ID,
        )
        prefetch_embedder(_TOPIC_EMBEDDER_REPO_ID)
        logger.info("[prefetch] ORT embedder ready")
    except Exception:
        logger.warning("[prefetch] ORT embedder prefetch failed", exc_info=True)


def _run_all_prefetches() -> None:
    """Run each prefetch sequentially in the daemon thread.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need a
      backend boundary that validates inputs before delegating to workspace or worker state.

    Flow: normalize inputs, delegate to the owning backend state or service boundary, and
        return serialized values or existing domain errors to callers.
    """
    _prefetch_spacy_model()
    _prefetch_topic_embedder()


def start_model_prefetch() -> None:
    """Kick off background model downloads in a daemon thread.

    Used by:
    - FastAPI application startup, backend tests because they need a backend boundary that
      validates inputs before delegating to workspace or worker state.

    Flow: normalize inputs, delegate to the owning backend state or service boundary, and
        return serialized values or existing domain errors to callers.
    """
    thread = threading.Thread(
        target=_run_all_prefetches,
        name="model-prefetch",
        daemon=True,
    )
    thread.start()
