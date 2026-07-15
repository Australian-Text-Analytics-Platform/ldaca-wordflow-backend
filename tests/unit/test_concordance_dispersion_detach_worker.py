from pathlib import Path
import polars as pl
from ldaca_wordflow.workers.concordance import (
    _aggregate_hits_per_document,
    run_concordance_dispersion_detachment,
)


def _build_hits_df() -> pl.DataFrame:
    """Mirrors `pt.concordance` output shape: token-bounded contexts with no
    surrounding whitespace, plus character indices that bound the matched
    span exclusively (no spaces)."""
    return pl.DataFrame(
        {
            "document": [
                "Hello world this is a test document",
                "Another doc with some content here",
                "Hello world this is a test document",
                "Different doc entirely",
            ],
            "CONC_left_context": ["", "Another", "Hello", "Different"],
            "CONC_matched_text": ["Hello", "doc", "world", "doc"],
            "CONC_right_context": ["world this", "with some", "this is", "entirely"],
            "CONC_start_idx": [0, 8, 6, 10],
            "CONC_end_idx": [5, 11, 11, 13],
            "CONC_l1": ["", "Another", "Hello", "Different"],
            "CONC_r1": ["world", "with", "this", "entirely"],
            "CONC_l1_freq": [1, 1, 1, 1],
            "CONC_r1_freq": [1, 1, 1, 1],
        }
    )


def test_aggregate_hits_per_document_groups_and_renders_extract():
    df = _build_hits_df()

    agg, output_columns = _aggregate_hits_per_document(
        df,
        document_column="document",
        selected_bins=None,
        total_bins=None,
    )

    assert output_columns == [
        "document",
        "CONC_extraction",
        "CONC_matched_text",
        "CONC_l1",
        "CONC_r1",
        "CONC_l1_freq",
        "CONC_r1_freq",
    ]
    # 3 unique documents — the two "Hello world..." rows collapse.
    assert agg.height == 3
    assert agg.schema["CONC_extraction"] == pl.Utf8
    assert agg.schema["CONC_matched_text"] == pl.List(pl.Utf8)
    assert agg.schema["CONC_l1_freq"] == pl.List(pl.Int64)

    extracted = {
        row["document"]: row["CONC_extraction"] for row in agg.iter_rows(named=True)
    }
    # Each line is `- <left_context> <matched> <right_context>` — the full
    # KWIC window read directly from the source document so spacing matches
    # the original. Two-hit document picks up two windows, in order. The
    # `- ` prefix mirrors Markdown bullet syntax for portability.
    assert (
        extracted["Hello world this is a test document"]
        == "- Hello world this\n- Hello world this is"
    )
    assert extracted["Another doc with some content here"] == "- Another doc with some"


def test_aggregate_hits_per_document_filters_by_selected_bins():
    df = _build_hits_df()

    # total_bins=2 splits each document in half. start_idx=0 → bin 0,
    # start_idx=6 in a 35-char doc → ~0.17 → bin 0, start_idx=8 in 34 →
    # ~0.24 → bin 0, start_idx=10 in 22 → ~0.45 → bin 0. All hits in bin 0.
    agg_first_half, _ = _aggregate_hits_per_document(
        df,
        document_column="document",
        selected_bins=[0],
        total_bins=2,
    )
    assert agg_first_half.height == 3  # all docs survive

    # Second half should drop everything.
    agg_second_half, _ = _aggregate_hits_per_document(
        df,
        document_column="document",
        selected_bins=[1],
        total_bins=2,
    )
    assert agg_second_half.height == 0


def test_aggregate_hits_per_document_filters_by_selected_matched_texts():
    df = _build_hits_df()

    # Only "Hello" and "doc". The two "world" hits should drop. The
    # "Different doc entirely" row contains a "doc" hit so it survives.
    agg, _ = _aggregate_hits_per_document(
        df,
        document_column="document",
        selected_bins=None,
        total_bins=None,
        selected_matched_texts=["Hello", "doc"],
    )
    matched_per_doc = {
        row["document"]: row["CONC_matched_text"] for row in agg.iter_rows(named=True)
    }
    # The "Hello world this is a test document" row had two hits ("Hello"
    # and "world"); only "Hello" survives.
    assert matched_per_doc["Hello world this is a test document"] == ["Hello"]
    assert matched_per_doc["Another doc with some content here"] == ["doc"]
    assert matched_per_doc["Different doc entirely"] == ["doc"]


def test_aggregate_hits_per_document_case_insensitive_legend_filter():
    df = _build_hits_df()

    # Pass lowercase "hello" plus case_insensitive=True — the column has
    # "Hello" (uppercase H) but should still match.
    agg, _ = _aggregate_hits_per_document(
        df,
        document_column="document",
        selected_bins=None,
        total_bins=None,
        selected_matched_texts=["hello"],
        match_case_insensitive=True,
    )
    # Only the "Hello world..." document survives.
    assert agg.height == 1
    row = next(iter(agg.iter_rows(named=True)))
    assert row["document"] == "Hello world this is a test document"
    assert row["CONC_matched_text"] == ["Hello"]


def test_aggregate_hits_per_document_empty_selected_matched_texts_is_zero_rows():
    """All legend items hidden → zero-row result (no aggregation)."""
    df = _build_hits_df()

    agg, _ = _aggregate_hits_per_document(
        df,
        document_column="document",
        selected_bins=None,
        total_bins=None,
        selected_matched_texts=[],
    )
    assert agg.height == 0


def test_aggregate_hits_per_document_strips_internal_newlines_from_extracts():
    """Embedded newlines in the raw document slice must not split a bullet.

    The KWIC extract is a raw character slice of the document, so if the
    left/right context spans a line break the extract carries the literal
    "\\n". Joining bulleted extracts with newlines would then render the
    continuation lines without a leading "- ", breaking the dot-point shape.
    Internal whitespace runs collapse to a single space per element.
    """
    df = pl.DataFrame(
        {
            "document": ["Line one\nLine two has match here"],
            "CONC_left_context": ["Line one\nLine two has"],
            "CONC_matched_text": ["match"],
            "CONC_right_context": ["here"],
            "CONC_start_idx": [22],
            "CONC_end_idx": [27],
            "CONC_l1": ["has"],
            "CONC_r1": ["here"],
            "CONC_l1_freq": [1],
            "CONC_r1_freq": [1],
        }
    )

    agg, _ = _aggregate_hits_per_document(
        df,
        document_column="document",
        selected_bins=None,
        total_bins=None,
    )

    row = next(iter(agg.iter_rows(named=True)))
    assert "\n" not in row["CONC_extraction"]
    assert row["CONC_extraction"].startswith("- ")


def test_run_concordance_dispersion_detachment_writes_node_payload(
    tmp_path, worker_snapshot
):
    progress_updates: list[tuple[float, str]] = []

    result = run_concordance_dispersion_detachment(
        workspace_dir=str(tmp_path),
        input_snapshot_dir=str(
            worker_snapshot(
                node_id="11111111-1111-4111-8111-111111111111",
                columns={"document": ["alpha beta gamma alpha", "beta gamma alpha"]},
            )
        ),
        parent_node_id="11111111-1111-4111-8111-111111111111",
        document_column="document",
        search_word="alpha",
        num_left_tokens=1,
        num_right_tokens=1,
        regex=False,
        whole_word=False,
        case_sensitive=False,
        new_node_name="alpha_conc_aggregated",
        progress_callback=lambda progress, message: progress_updates.append(
            (progress, message)
        ),
    )

    assert result["state"] == "successful", result
    payload = result["result"]
    assert payload["parquet_path"].startswith("data/")

    data_file = tmp_path / Path(payload["parquet_path"])
    assert data_file.exists()

    df = pl.read_parquet(data_file)
    assert df.height >= 1
    # Per-document output shape — CONC_extraction is a string, others are
    # List<T>. The per-hit start/end indices and L/R contexts are dropped.
    assert df.schema["CONC_extraction"] == pl.Utf8
    assert df.schema["CONC_matched_text"] == pl.List(pl.Utf8)
    assert "CONC_left_context" not in df.columns
    assert "CONC_right_context" not in df.columns
    assert "CONC_start_idx" not in df.columns
    assert "CONC_end_idx" not in df.columns

    assert progress_updates[-1] == (0.95, "Publishing dispersion Data Block...")
