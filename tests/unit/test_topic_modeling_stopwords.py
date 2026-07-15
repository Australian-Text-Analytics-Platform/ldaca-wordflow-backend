"""Vectorizer/stopword routing and top_n_words headroom for the Rust pipeline.

The Rust topic-modeling pipeline tokenizes the topic text itself for c-TF-IDF,
so the Python side only decides *which* segmenter the pipeline should use
(plain English words vs a lindera CJK dictionary) and supplies the matching
stopword list. These tests pin that script-detection routing and the
``top_n_words`` headroom arithmetic that protects the frontend stopword filter
from leaving too few visible words per topic.
"""

from __future__ import annotations

from ldaca_wordflow.workers.topic_pipeline import (
    _LINDERA_JA_VECTORIZER,
    _LINDERA_KO_VECTORIZER,
    _LINDERA_ZH_VECTORIZER,
    _PLAIN_WORDS_EN_VECTORIZER,
    _resolve_top_n_words,
    _resolve_vectorizer_model,
    _stopwords_for_lang,
)


# ---------------------------------------------------------------------------
# Vectorizer/stopword routing by document script.
# ---------------------------------------------------------------------------


def test_english_corpus_routes_to_plain_words_with_english_stopwords() -> None:
    docs = [
        "The market rallied today as investors bought shares.",
        "Rain is expected across the region this weekend.",
    ]
    vectorizer, lang = _resolve_vectorizer_model(docs)
    assert vectorizer == _PLAIN_WORDS_EN_VECTORIZER
    assert lang == "en"


def test_chinese_corpus_routes_to_lindera_zh_without_stopwords() -> None:
    docs = ["市场今天上涨投资者纷纷买入股票", "本周末预计全区都会下雨"]
    vectorizer, lang = _resolve_vectorizer_model(docs)
    assert vectorizer == _LINDERA_ZH_VECTORIZER
    assert lang is None


def test_japanese_corpus_routes_to_lindera_ja() -> None:
    docs = ["今日は市場が上昇しました。", "週末は雨が降るでしょう。"]
    vectorizer, lang = _resolve_vectorizer_model(docs)
    assert vectorizer == _LINDERA_JA_VECTORIZER
    assert lang is None


def test_korean_corpus_routes_to_lindera_ko() -> None:
    docs = ["오늘 시장이 상승했습니다", "주말에 비가 올 것입니다"]
    vectorizer, lang = _resolve_vectorizer_model(docs)
    assert vectorizer == _LINDERA_KO_VECTORIZER
    assert lang is None


def test_empty_corpus_defaults_to_english_plain_words() -> None:
    vectorizer, lang = _resolve_vectorizer_model([])
    assert vectorizer == _PLAIN_WORDS_EN_VECTORIZER
    assert lang == "en"


# ---------------------------------------------------------------------------
# Stopword list selection passed to the Rust pipeline.
# ---------------------------------------------------------------------------


def test_english_stopwords_use_sklearn_stoplist() -> None:
    stopwords = _stopwords_for_lang("en")
    assert stopwords  # non-empty
    assert "the" in stopwords
    assert "and" in stopwords
    # Returned sorted for determinism.
    assert stopwords == sorted(stopwords)


def test_cjk_and_unknown_languages_return_no_stopwords() -> None:
    # lindera handles CJK segmentation; we don't ship a CJK stoplist.
    assert _stopwords_for_lang(None) == []
    assert _stopwords_for_lang("zh") == []


# ---------------------------------------------------------------------------
# top_n_words plumbing — guards against the silent-cap regression where
# "Words per topic = 35" only produced 10 candidates because the c-TF-IDF
# stage was never asked for headroom. With the frontend stopword filter on,
# that left ~1–3 visible words per topic on a CJK run.
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
