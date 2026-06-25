import polars as pl
import pytest
from ldaca_wordflow.api.workspaces.analyses.concordance_core import (
    _serialize_materialized_rows,
    build_concordance_search_pattern,
    compute_concordance_page,
    concordance_non_empty_expr,
    normalize_saved_request,
    sanitize_request_for_storage,
)


@pytest.mark.parametrize(
    "raw_request",
    [
        {
            "node_ids": ["node-1"],
            "node_columns": {"node-1": "text"},
            "search_word": "example",
            "page": 3,
            "page_size": 25,
            "descending": False,
            "pagination": {"page": 3},
            "regex": False,
            "case_sensitive": None,
        }
    ],
)
def test_sanitize_request_excludes_pagination_keys(raw_request):
    sanitized = sanitize_request_for_storage(raw_request)

    assert sanitized == {
        "node_ids": ["node-1"],
        "node_columns": {"node-1": "text"},
        "search_word": "example",
        "regex": False,
    }
    for excluded in (
        "page",
        "page_size",
        "sort_by",
        "descending",
        "pagination",
    ):
        assert excluded not in sanitized


def test_normalize_saved_request_rejects_legacy_shape():
    # We no longer support legacy request formats. Requests must include
    # `node_ids` and `node_columns`.
    raw_request = {
        "node_id": "node-legacy",
        "column": "text",
        "search_word": "alpha",
    }

    assert normalize_saved_request(raw_request) is None


def test_filter_concordance_rows_removes_blank_entries():
    df = pl.DataFrame(
        {
            "CONC_matched_text": ["alpha", None, "   ", ""],
            "CONC_left_context": ["", "", "", ""],
            "CONC_right_context": ["", "context", "\t", None],
        }
    )

    filtered = df.filter(concordance_non_empty_expr())

    assert filtered.height == 2


def test_build_concordance_search_pattern_wraps_whole_word_literals():
    pattern, use_regex = build_concordance_search_pattern(
        "alpha.beta",
        regex=False,
        whole_word=True,
    )

    assert pattern == r"\b(?:alpha\.beta)\b"
    assert use_regex is True


def test_compute_concordance_page_groups_matches_by_source_row():
    request = {
        "search_word": "alpha",
        "num_left_tokens": 2,
        "num_right_tokens": 2,
        "regex": False,
        "case_sensitive": False,
    }
    source = pl.DataFrame(
        {
            "text": ["alpha beta alpha", "gamma alpha"],
            "speaker": ["A", "B"],
        }
    ).lazy()

    result = compute_concordance_page(
        source,
        "text",
        request,
        page=1,
        page_size=1,
        sort_by=None,
        descending=False,
        node_label="node-a",
    )

    assert result["pagination"]["page_size"] == 1
    assert len(result["data"]) == 1

    grouped_row = result["data"][0]
    assert isinstance(grouped_row, list)
    assert len(grouped_row) == 2
    assert all(hit["speaker"] == "A" for hit in grouped_row)
    assert all(hit["__source_node"] == "node-a" for hit in grouped_row)
    assert [hit["CONC_matched_text"] for hit in grouped_row] == ["alpha", "alpha"]


def test_compute_concordance_page_whole_word_ignores_partial_matches():
    request = {
        "search_word": "alpha",
        "num_left_tokens": 2,
        "num_right_tokens": 2,
        "regex": False,
        "case_sensitive": False,
        "whole_word": True,
    }
    source = pl.DataFrame(
        {
            "text": ["alphabet soup", "alpha beta"],
            "speaker": ["A", "B"],
        }
    ).lazy()

    result = compute_concordance_page(
        source,
        "text",
        request,
        page=1,
        page_size=5,
        sort_by=None,
        descending=False,
        node_label="node-a",
    )

    assert len(result["data"]) == 1
    assert result["data"][0][0]["speaker"] == "B"
    assert result["data"][0][0]["CONC_matched_text"] == "alpha"


def test_serialize_materialized_rows_groups_by_document_for_dispersion():
    """The materialise worker writes one parquet row per hit but the
    dispersion view needs one *group* per document so each horizontal
    bar carries every hit from that document. Verify consecutive same-
    document hits collapse into one group.
    """
    df = pl.DataFrame(
        {
            "context": [
                "doc-A",  # 3 hits
                "doc-A",
                "doc-A",
                "doc-B",  # 1 hit
                "doc-C",  # 2 hits
                "doc-C",
            ],
            "CONC_matched_text": ["wa", "wo", "ga", "no", "to", "de"],
            "CONC_start_idx": [10, 30, 50, 5, 100, 200],
        }
    )

    grouped, columns = _serialize_materialized_rows(
        df, node_label="jp_corpus", document_column="context"
    )

    assert "__source_node" in columns
    assert [len(g) for g in grouped] == [3, 1, 2]
    assert all(hit["__source_node"] == "jp_corpus" for g in grouped for hit in g)
    # Every hit in a group must share the same document value.
    assert all(len({hit["context"] for hit in g}) == 1 for g in grouped), (
        "consecutive same-document rows must end up in one group"
    )


def test_serialize_materialized_rows_falls_back_to_singleton_groups_without_document_column():
    """When the caller doesn't provide ``document_column`` (legacy parquets
    or unknown column), keep the pre-fix shape: one singleton group per
    hit, so the table view still works."""
    df = pl.DataFrame(
        {
            "CONC_matched_text": ["a", "b", "c"],
            "CONC_start_idx": [0, 1, 2],
        }
    )

    grouped, _ = _serialize_materialized_rows(df, node_label="n")
    assert [len(g) for g in grouped] == [1, 1, 1]
