"""Deterministic unit tests for the Rust-backed topic-modeling worker.

The heavy lifting (chunking, ORT embeddings, PaCMAP, HDBSCAN, c-TF-IDF) lives
in the ``polars-text`` Rust extension and is exercised by the hand-run
experiment harness, not here -- its output is non-deterministic. These tests
cover only the deterministic Python glue:

- corpus sampling and the c-TF-IDF vectorizer/stopword heuristics
  (``worker_tasks_topic_pipeline``),
- the reconstruction of the result dict from the ``.text.topic_modeling``
  expression (``_run_rust_topic_modeling``), with the expression itself faked,
- the payload/parquet assembly and meta in the orchestrator
  (``run_topic_modeling_task``) and the exact-count re-aggregation path, with
  ``_run_rust_topic_modeling`` faked to a canned result.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest
from ldaca_wordflow.core import (
    worker_tasks_topic,
    worker_tasks_topic_pipeline,
)
from ldaca_wordflow.core.worker_tasks_topic_pipeline import (
    _sample_corpus,
)

_STAGE_TIMINGS = [
    {"stage": "embedding", "elapsed_ms": 12.5},
    {"stage": "total", "elapsed_ms": 15.0},
]


# ---------------------------------------------------------------------------
# Sampling helpers (pure, deterministic)
# ---------------------------------------------------------------------------


def test_sample_corpus_reduces_length_and_is_reproducible():
    docs = [f"doc {i}" for i in range(100)]
    sampled_docs, sampled_idx = _sample_corpus(docs, 0.5, seed=42)
    assert len(sampled_docs) == 50
    assert len(sampled_idx) == 50
    # Same seed reproduces the exact sample.
    docs2, idx2 = _sample_corpus(docs, 0.5, seed=42)
    assert docs2 == sampled_docs
    assert idx2 == sampled_idx
    # A different seed selects a different sample.
    docs3, _ = _sample_corpus(docs, 0.5, seed=99)
    assert docs3 != sampled_docs


def test_sample_corpus_indices_are_original_sorted_positions():
    docs = [f"doc {i}" for i in range(20)]
    sampled_docs, sampled_idx = _sample_corpus(docs, 0.5, seed=7)
    for doc, idx in zip(sampled_docs, sampled_idx):
        assert doc == docs[idx]
    assert sampled_idx == sorted(sampled_idx)


def test_sample_corpus_fraction_at_or_above_one_returns_original():
    docs = ["a", "b", "c"]
    result_docs, result_idx = _sample_corpus(docs, 1.0, seed=42)
    assert result_docs is docs
    assert result_idx == [0, 1, 2]
    result_docs2, _ = _sample_corpus(docs, 2.0, seed=42)
    assert result_docs2 is docs


def test_sample_corpus_min_k_is_one():
    docs = ["only"]
    result_docs, result_idx = _sample_corpus(docs, 0.01, seed=42)
    assert len(result_docs) == 1
    assert len(result_idx) == 1


# ---------------------------------------------------------------------------
# _run_rust_topic_modeling: reconstruct the result dict from the expression
# ---------------------------------------------------------------------------


def _fake_topic_modeling_expr_factory(
    *,
    dominant: list[int],
    words: list[list[str]],
    xs: list[float],
    ys: list[float],
    n_chunks: int,
    distribution: list[list[dict[str, Any]]] | None = None,
    stage_timings: list[dict[str, Any]] | None = None,
):
    """Build a fake ``.text.topic_modeling`` method returning a canned struct.

    The real expression returns one struct per input document with the per-topic
    metadata replicated onto each row under its dominant topic; the fake mirrors
    that exact shape from literal Series so ``_run_rust_topic_modeling``'s
    reconstruction can be tested offline without running the Rust pipeline.
    """

    n = len(dominant)
    # Default each row's distribution to a single entry at its dominant topic
    # (proportion 1.0); outliers (-1) get an empty distribution.
    dist = distribution
    if dist is None:
        dist = [
            ([{"topic_id": int(t), "proportion": 1.0}] if t >= 0 else [])
            for t in dominant
        ]
    timings = stage_timings if stage_timings is not None else _STAGE_TIMINGS

    def _fake(self, **_kwargs):  # noqa: ANN001 - mirrors namespace method shape
        return pl.struct(
            pl.Series("dominant_topic", dominant, dtype=pl.Int32),
            pl.Series(
                "topic_distribution",
                dist,
                dtype=pl.List(
                    pl.Struct({"topic_id": pl.Int32, "proportion": pl.Float32})
                ),
            ),
            pl.Series("representative_words", words, dtype=pl.List(pl.String)),
            pl.Series("x", xs, dtype=pl.Float32),
            pl.Series("y", ys, dtype=pl.Float32),
            pl.Series("n_topics", [0] * n, dtype=pl.UInt32),
            pl.Series("n_chunks", [n_chunks] * n, dtype=pl.UInt32),
            pl.Series(
                "stage_timings_ms",
                [timings] * n,
                dtype=pl.List(
                    pl.Struct({"stage": pl.String, "elapsed_ms": pl.Float64})
                ),
            ),
        )

    return _fake


def test_run_rust_topic_modeling_reconstructs_result_dict(monkeypatch):
    from polars_text.namespace import TextNamespace

    monkeypatch.setattr(
        TextNamespace,
        "topic_modeling",
        _fake_topic_modeling_expr_factory(
            dominant=[0, 0, 1, -1],
            words=[["alpha", "beta"], ["alpha", "beta"], ["gamma"], []],
            xs=[1.0, 1.0, 2.0, 0.0],
            ys=[3.0, 3.0, 4.0, 0.0],
            n_chunks=5,
            distribution=[
                [
                    {"topic_id": 0, "proportion": 0.9},
                    {"topic_id": 1, "proportion": 0.1},
                ],
                [{"topic_id": 0, "proportion": 1.0}],
                [{"topic_id": 1, "proportion": 1.0}],
                [],
            ],
        ),
    )

    result = worker_tasks_topic_pipeline._run_rust_topic_modeling(
        all_docs=["d0", "d1", "d2", "d3"],
        seed=42,
        top_k=50,
        min_cluster_size=10,
        vectorizer_model="native:plain_words_en",
        stopwords=["the"],
        embedder_model="fake-model",
    )

    # Documents carry both the dominant topic and the soft topic_distribution.
    # The distribution is padded so every non-negative topic id (here 0 and 1)
    # appears in every document, with 0.0 where the doc has no presence; this
    # powers the Filter-tab tmdist function and the dataview bars. The outlier
    # document (-1) has no non-negative dominant topics of its own but still
    # gets the full padded key set.
    assert result["documents"] == [
        {
            "doc_index": 0,
            "dominant_topic": 0,
            "topic_distribution": [
                {"topic_id": 0, "proportion": pytest.approx(0.9)},
                {"topic_id": 1, "proportion": pytest.approx(0.1)},
            ],
        },
        {
            "doc_index": 1,
            "dominant_topic": 0,
            "topic_distribution": [
                {"topic_id": 0, "proportion": pytest.approx(1.0)},
                {"topic_id": 1, "proportion": pytest.approx(0.0)},
            ],
        },
        {
            "doc_index": 2,
            "dominant_topic": 1,
            "topic_distribution": [
                {"topic_id": 0, "proportion": pytest.approx(0.0)},
                {"topic_id": 1, "proportion": pytest.approx(1.0)},
            ],
        },
        {
            "doc_index": 3,
            "dominant_topic": -1,
            "topic_distribution": [
                {"topic_id": 0, "proportion": pytest.approx(0.0)},
                {"topic_id": 1, "proportion": pytest.approx(0.0)},
            ],
        },
    ]
    # Outlier topic (-1) is excluded; topics are sorted by id and carry the
    # replicated representative words and coordinates.
    assert result["topics"] == [
        {"id": 0, "representative_words": ["alpha", "beta"], "x": 1.0, "y": 3.0},
        {"id": 1, "representative_words": ["gamma"], "x": 2.0, "y": 4.0},
    ]
    # n_topics is the count of topics with at least one dominant document, not
    # the (zeroed) replicated field.
    assert result["n_topics"] == 2
    assert result["n_chunks"] == 5
    assert result["stage_timings_ms"] == _STAGE_TIMINGS


# ---------------------------------------------------------------------------
# Orchestrator + payload assembly (with _run_rust_topic_modeling faked)
# ---------------------------------------------------------------------------


def _canned_rust_result(
    *,
    documents: list[dict[str, Any]],
    topics: list[dict[str, Any]],
    n_chunks: int = 7,
    stage_timings_ms: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "documents": documents,
        "topics": topics,
        "n_topics": len(topics),
        "n_chunks": n_chunks,
        "stage_timings_ms": stage_timings_ms or _STAGE_TIMINGS,
    }


def _node_info(node_id: str = "node-1") -> dict[str, Any]:
    return {
        "node_id": node_id,
        "node_name": f"Node {node_id}",
        "text_column": "document",
        "original_columns": ["document"],
    }


def test_run_topic_modeling_task_writes_parquet_and_meaning_lists(
    tmp_path, monkeypatch
):
    progress: list[tuple[float, str]] = []

    seen_run_kwargs: dict[str, Any] = {}

    def fake_run(**kwargs):
        seen_run_kwargs.update(kwargs)
        return _canned_rust_result(
            documents=[
                {
                    "doc_index": 0,
                    "dominant_topic": 0,
                    "topic_distribution": [{"topic_id": 0, "proportion": 1.0}],
                },
                {
                    "doc_index": 1,
                    "dominant_topic": 0,
                    "topic_distribution": [
                        {"topic_id": 0, "proportion": 0.7},
                        {"topic_id": 1, "proportion": 0.3},
                    ],
                },
            ],
            topics=[
                {
                    "id": 0,
                    "representative_words": ["alpha", "beta", "gamma"],
                    "x": 1.5,
                    "y": -2.0,
                }
            ],
        )

    monkeypatch.setattr(worker_tasks_topic, "_run_rust_topic_modeling", fake_run)
    embedding_cache_path = tmp_path / "embeddings.duckdb"
    monkeypatch.setattr(
        worker_tasks_topic,
        "embeddings_cache_path",
        lambda _user_id: embedding_cache_path,
    )

    result = worker_tasks_topic.run_topic_modeling_task(
        configure_worker_environment=lambda: None,
        user_id="u",
        workspace_id="w",
        corpora=[["doc one", "doc two"]],
        node_infos=[_node_info()],
        artifact_dir=str(tmp_path),
        artifact_prefix="tm_test",
        representative_words_count=3,
        progress_callback=lambda p, m: progress.append((p, m)),
    )

    assignments = pl.read_parquet(tmp_path / "tm_test_topic_assignments_node-1.parquet")
    meanings = pl.read_parquet(tmp_path / "tm_test_topic_meanings.parquet")

    # The assignment parquet now carries the per-row soft distribution column
    # used by the detach-time distribution filter.
    assert assignments.columns == [
        "__row_nr__",
        "TOPIC_topic",
        "TOPIC_topic_distribution",
    ]
    assert assignments.schema["TOPIC_topic"] == pl.Int64
    assert assignments["TOPIC_topic"].to_list() == [0, 0]
    assert assignments.schema["TOPIC_topic_distribution"] == pl.List(
        pl.Struct({"topic_id": pl.Int64, "proportion": pl.Float64})
    )
    assert assignments["TOPIC_topic_distribution"].to_list() == [
        [{"topic_id": 0, "proportion": 1.0}],
        [{"topic_id": 0, "proportion": 0.7}, {"topic_id": 1, "proportion": 0.3}],
    ]
    assert meanings.schema["TOPIC_topic_meaning"] == pl.List(pl.String)
    assert meanings.to_dicts() == [
        {"TOPIC_topic": 0, "TOPIC_topic_meaning": ["alpha", "beta", "gamma"]}
    ]

    topic = result["topics"][0]
    assert topic["representative_words"] == ["alpha", "beta", "gamma"]
    assert topic["label"] == "alpha | beta | gamma"
    assert topic["x"] == pytest.approx(1.5)
    assert topic["y"] == pytest.approx(-2.0)
    assert topic["size"] == [2]

    assert result["meta"]["engine"] == "rust"
    assert result["meta"]["embedding_backend"] == "ort"
    assert seen_run_kwargs["embedding_cache"] == embedding_cache_path
    assert result["meta"]["n_chunks"] == 7
    assert result["meta"]["stage_timings_ms"] == _STAGE_TIMINGS
    assert progress[0][1].startswith("Loading topic modeling")
    assert progress[-1] == (1.0, "Topic modeling completed")


def test_run_topic_modeling_task_payload_caps_words_but_keeps_headroom(
    tmp_path, monkeypatch
):
    """The payload carries up to the headroom cap; the meaning column respects
    the user's small display count."""

    many_words = [f"w{i}" for i in range(60)]

    def fake_run(**_kwargs):
        return _canned_rust_result(
            documents=[{"doc_index": 0, "dominant_topic": 0}],
            topics=[{"id": 0, "representative_words": many_words, "x": 0.0, "y": 0.0}],
        )

    monkeypatch.setattr(worker_tasks_topic, "_run_rust_topic_modeling", fake_run)

    result = worker_tasks_topic.run_topic_modeling_task(
        configure_worker_environment=lambda: None,
        user_id="u",
        workspace_id="w",
        corpora=[["only doc"]],
        node_infos=[_node_info()],
        artifact_dir=str(tmp_path),
        artifact_prefix="tm_cap",
        representative_words_count=5,
    )

    # Payload keeps the generous headroom (max(50, 2*5) = 50).
    assert result["topics"][0]["representative_words"] == many_words[:50]
    # The meaning parquet respects the requested display count.
    meanings = pl.read_parquet(tmp_path / "tm_cap_topic_meanings.parquet")
    assert meanings.to_dicts()[0]["TOPIC_topic_meaning"] == many_words[:5]


def test_run_topic_modeling_task_sampling_records_before_after_sizes(
    tmp_path, monkeypatch
):
    seen_docs: dict[str, int] = {}

    def fake_run(*, all_docs, **_kwargs):
        seen_docs["count"] = len(all_docs)
        documents = [
            {"doc_index": i, "dominant_topic": 0} for i in range(len(all_docs))
        ]
        return _canned_rust_result(
            documents=documents,
            topics=[{"id": 0, "representative_words": ["x"], "x": 0.0, "y": 0.0}],
        )

    monkeypatch.setattr(worker_tasks_topic, "_run_rust_topic_modeling", fake_run)

    corpus = [f"doc {i}" for i in range(20)]
    result = worker_tasks_topic.run_topic_modeling_task(
        configure_worker_environment=lambda: None,
        user_id="u",
        workspace_id="w",
        corpora=[corpus],
        node_infos=[_node_info("n1")],
        artifact_dir=str(tmp_path),
        artifact_prefix="tm_sample",
        sample_fractions=[0.5],
    )

    assert seen_docs["count"] == 10
    assert result["meta"]["corpus_sizes_before_sample"] == [20]
    assert result["meta"]["corpus_sizes_after_sample"] == [10]


def test_run_topic_modeling_task_passes_min_topic_size_as_cluster_size(
    tmp_path, monkeypatch
):
    """``min_topic_size`` is forwarded to the expression as ``min_cluster_size``
    (the only native topic-count control); there is no post-fit merge."""
    captured_kwargs: dict[str, Any] = {}

    def fake_run(**kwargs):
        captured_kwargs.update(kwargs)
        return _canned_rust_result(
            documents=[
                {"doc_index": 0, "dominant_topic": 0},
                {"doc_index": 1, "dominant_topic": 1},
            ],
            topics=[
                {"id": 0, "representative_words": ["a"], "x": 0.0, "y": 0.0},
                {"id": 1, "representative_words": ["b"], "x": 1.0, "y": 1.0},
            ],
        )

    monkeypatch.setattr(worker_tasks_topic, "_run_rust_topic_modeling", fake_run)

    result = worker_tasks_topic.run_topic_modeling_task(
        configure_worker_environment=lambda: None,
        user_id="u",
        workspace_id="w",
        corpora=[["doc one", "doc two"]],
        node_infos=[_node_info("n1")],
        artifact_dir=str(tmp_path),
        artifact_prefix="tm_min",
        min_topic_size=15,
    )

    assert captured_kwargs["min_cluster_size"] == 15
    assert "topic_size_mode" not in captured_kwargs
    assert "topic_size_value" not in captured_kwargs
    assert result["meta"]["min_topic_size"] == 15
    # No exact re-aggregation context is persisted; manifest stays at version 1.
    assert not (tmp_path / "tm_min_exact_reduction.json").exists()
    assert result["artifacts"]["version"] == 1


def test_run_topic_modeling_task_loads_corpora_from_workspace(tmp_path, monkeypatch):
    captured: dict[str, Any] = {}

    def fake_run(*, all_docs, **_kwargs):
        captured["docs"] = list(all_docs)
        return _canned_rust_result(
            documents=[
                {"doc_index": i, "dominant_topic": 0} for i in range(len(all_docs))
            ],
            topics=[{"id": 0, "representative_words": ["w"], "x": 0.0, "y": 0.0}],
        )

    def fake_load(workspace_dir, node_payloads, user_id):
        captured["workspace_dir"] = workspace_dir
        return [["loaded one", "loaded two", "loaded three"]]

    monkeypatch.setattr(worker_tasks_topic, "_run_rust_topic_modeling", fake_run)
    monkeypatch.setattr(worker_tasks_topic, "_load_corpora_from_workspace", fake_load)

    worker_tasks_topic.run_topic_modeling_task(
        configure_worker_environment=lambda: None,
        user_id="u",
        workspace_id="w",
        corpora=None,
        workspace_dir=str(tmp_path / "ws"),
        node_infos=[_node_info("n1")],
        artifact_dir=str(tmp_path),
        artifact_prefix="tm_ws",
    )

    assert captured["docs"] == ["loaded one", "loaded two", "loaded three"]
    assert captured["workspace_dir"] == str(tmp_path / "ws")
