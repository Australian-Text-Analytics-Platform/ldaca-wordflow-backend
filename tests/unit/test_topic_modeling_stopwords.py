"""Label-stage CountVectorizer stop_words coverage.

BERTopic's clustering stage works on document embeddings (no stopwords
involved), but the per-topic label stage uses a CountVectorizer to pick
"topic-distinguishing" terms via c-TF-IDF. The label vectorizer uses
sklearn's built-in English stoplist.
"""

from __future__ import annotations

import pytest

from ldaca_wordflow.core.worker_tasks_topic_pipeline import (
    _build_classic_pipeline,
    _build_label_vectorizer,
    _resolve_top_n_words,
)

bertopic = pytest.importorskip("bertopic")


def _make_dummy_embedder():
    class _Embedder:
        def encode(self, *_args, **_kwargs):
            raise AssertionError(
                "encode() must not run at pipeline-build time — only the "
                "vectorizer routing matters for this test"
            )

    return _Embedder()


def _vectorizer(topic_model):
    """Pull the actual vectorizer instance off a BERTopic model."""
    return getattr(topic_model, "vectorizer_model", None) or getattr(
        topic_model, "_vectorizer_model", None
    )


# ---------------------------------------------------------------------------
# Classic-pipeline + label-vectorizer coverage.
# ---------------------------------------------------------------------------


def test_label_vectorizer_english_uses_sklearn_english_stoplist() -> None:
    vec = _build_label_vectorizer()
    assert vec.stop_words == "english"


def test_classic_pipeline_uses_english_stoplist() -> None:
    model = _build_classic_pipeline(
        min_topic_size=10,
        random_state=42,
        embedder=_make_dummy_embedder(),
    )
    vec = _vectorizer(model)
    assert vec is not None
    assert vec.stop_words == "english"


# ---------------------------------------------------------------------------
# top_n_words plumbing — guards against the silent-cap regression where
# "Words per topic = 35" only produced 10 candidates because BERTopic's
# top_n_words default was never overridden. With the frontend stopword
# filter on, that left ~1–3 visible words per topic on a CJK run.
# ---------------------------------------------------------------------------


def test_resolve_top_n_words_floors_at_fifty() -> None:
    assert _resolve_top_n_words(0) == 50
    assert _resolve_top_n_words(None) == 50
    assert _resolve_top_n_words(5) == 50
    assert _resolve_top_n_words(20) == 50


def test_resolve_top_n_words_scales_with_user_cap() -> None:
    # 2× headroom so the post-fit stopword filter has enough material
    # to still produce a meaningful slice at the user's chosen cap.
    assert _resolve_top_n_words(30) == 60
    assert _resolve_top_n_words(35) == 70
    assert _resolve_top_n_words(100) == 200


def test_classic_pipeline_forwards_top_n_words_to_bertopic() -> None:
    """Direct ``top_n_words`` attribute check on the built model — guarantees
    the kwarg actually reaches BERTopic, not just our helper signature."""
    model = _build_classic_pipeline(
        min_topic_size=10,
        random_state=42,
        embedder=_make_dummy_embedder(),
        top_n_words=70,
    )
    assert getattr(model, "top_n_words", None) == 70

