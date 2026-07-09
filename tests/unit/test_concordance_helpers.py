from types import SimpleNamespace

import polars as pl
import pytest
from ldaca_wordflow.api.workspaces.analyses import concordance_core
from ldaca_wordflow.api.workspaces.analyses.concordance_core import (
    _serialize_materialized_rows,
    build_concordance_search_pattern,
    compute_concordance_page,
    concordance_non_empty_expr,
    normalize_saved_request,
    sanitize_request_for_storage,
)
from ldaca_wordflow.core.exceptions import InvalidInputError


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


def test_resolve_node_sources_uses_explicit_workspace_resolver(monkeypatch):
    source = pl.DataFrame({"text": ["alpha"]}).lazy()
    node = SimpleNamespace(name="Node A", data=source)
    workspace = SimpleNamespace(nodes={"node-1": node})
    resolver_calls: list[tuple[str, str]] = []

    def require_workspace(user_id: str, workspace_id: str):
        resolver_calls.append((user_id, workspace_id))
        return workspace

    class HiddenCurrentWorkspace:
        def get_current_workspace_id(self, _user_id: str):
            raise AssertionError("hidden current workspace id should not be read")

        def get_current_workspace(self, _user_id: str):
            raise AssertionError("hidden current workspace should not be read")

        def set_current_workspace(self, _user_id: str, _workspace_id: str):
            raise AssertionError("hidden current workspace should not be mutated")

    monkeypatch.setattr(concordance_core, "require_workspace", require_workspace, raising=False)
    monkeypatch.setattr(
        concordance_core,
        "workspace_manager",
        HiddenCurrentWorkspace(),
        raising=False,
    )

    sources, label_to_node_map, node_labels, error = concordance_core.resolve_node_sources(
        "user-1",
        "workspace-1",
        {"node_ids": ["node-1"], "node_columns": {"node-1": "text"}},
    )

    assert error is None
    assert resolver_calls == [("user-1", "workspace-1")]
    assert sources["node-1"]["lf"] is source
    assert sources["node-1"]["column"] == "text"
    assert label_to_node_map == {"Node A": "node-1"}
    assert node_labels == {"node-1": "Node A"}


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


def test_serialize_materialized_rows_can_hide_document_column_from_display():
    """The document column stays available for grouping even when hidden."""
    df = pl.DataFrame(
        {
            "context": ["doc-A", "doc-A", "doc-B"],
            "CONC_matched_text": ["a", "b", "c"],
            "CONC_start_idx": [0, 1, 2],
        }
    )

    grouped, columns = _serialize_materialized_rows(
        df,
        node_label="n",
        document_column="context",
        include_document_column=False,
    )

    assert "context" not in columns
    assert [len(g) for g in grouped] == [2, 1]
    assert all("context" not in hit for group in grouped for hit in group)


def test_serialize_materialized_rows_requires_document_column():
    df = pl.DataFrame(
        {
            "CONC_matched_text": ["a", "b", "c"],
            "CONC_start_idx": [0, 1, 2],
        }
    )

    with pytest.raises(InvalidInputError):
        _serialize_materialized_rows(df, node_label="n", document_column="context")
