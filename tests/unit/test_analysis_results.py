"""Typed Analysis Result projection tests."""

from io import BytesIO
from typing import Literal

import polars as pl
import pytest
from pydantic import ValidationError

from ldaca_wordflow.analysis.generated_columns import TOPIC_DISTRIBUTION_COLUMN
from ldaca_wordflow.models.analysis_results import (
    ConcordanceDocumentProjectionQuery,
    QuotationResultQuery,
)
from ldaca_wordflow.shared.topic_types import (
    topic_distribution_dtype,
    topic_distribution_storage_dtype,
)
from ldaca_wordflow.services.analysis_results import (
    _concordance_density,
    _concordance_document_projection_page,
    _paged_artifact_page,
    _paged_artifact_schema,
    _projected_artifact_page,
    _projected_artifact_schema,
    _sort_and_page,
)
from ldaca_wordflow.shared.errors import AnalysisCorruptError, InvalidInputError
from ldaca_wordflow.shared.json_data import JsonData


def test_quotation_result_query_contains_only_page_and_sort_controls() -> None:
    with pytest.raises(ValidationError):
        QuotationResultQuery.model_validate(
            {"kind": "quotation", "context_length": 12}
        )


@pytest.mark.parametrize(
    ("descending", "expected"),
    [
        (False, [2, 10, None]),
        (True, [10, 2, None]),
    ],
)
def test_result_sort_preserves_numeric_order_and_keeps_nulls_last(
    descending: bool,
    expected: list[int | None],
) -> None:
    rows: list[dict[str, JsonData]] = [
        {"value": 10},
        {"value": None},
        {"value": 2},
    ]

    page, _pagination = _sort_and_page(
        rows,
        page=1,
        page_size=10,
        sort_by="value",
        descending=descending,
        columns={"value"},
    )

    assert [row["value"] for row in page] == expected


def test_result_sort_rejects_unknown_columns() -> None:
    with pytest.raises(InvalidInputError):
        _sort_and_page(
            [{"value": 1}],
            page=1,
            page_size=10,
            sort_by="missing",
            descending=False,
            columns={"value"},
        )


def test_topic_assignment_pages_use_ipc_and_preserve_semantic_storage(
    tmp_path,
) -> None:
    path = tmp_path / "assignments.parquet"
    pl.DataFrame(
        {
            "__row_nr__": [0, 1],
            TOPIC_DISTRIBUTION_COLUMN: pl.Series(
                [
                    [
                        {"topic_id": -1, "proportion": 0.25},
                        {"topic_id": 0, "proportion": 0.75},
                    ],
                    [
                        {"topic_id": -1, "proportion": 0.0},
                        {"topic_id": 0, "proportion": 1.0},
                    ],
                ],
                dtype=topic_distribution_storage_dtype(1),
            ),
        }
    ).write_parquet(path)

    page = _paged_artifact_page(path, 1, 1, None, False)
    frame = pl.read_ipc_stream(BytesIO(page.content))
    schema = pl.read_ipc_stream(BytesIO(_paged_artifact_schema(path)))

    assert page.has_next is True
    assert frame[TOPIC_DISTRIBUTION_COLUMN].to_list() == [
        [
            {"topic_id": -1, "proportion": 0.25},
            {"topic_id": 0, "proportion": 0.75},
        ]
    ]
    assert schema.height == 0
    assert schema.schema[TOPIC_DISTRIBUTION_COLUMN] == topic_distribution_dtype(1)


def test_variable_list_topic_assignment_artifact_is_rejected(tmp_path) -> None:
    path = tmp_path / "legacy-assignments.parquet"
    pl.DataFrame(
        {
            TOPIC_DISTRIBUTION_COLUMN: pl.Series(
                [[{"topic_id": 0, "proportion": 1.0}]],
                dtype=pl.List(
                    pl.Struct({"topic_id": pl.Int64, "proportion": pl.Float64})
                ),
            )
        }
    ).write_parquet(path)

    with pytest.raises(AnalysisCorruptError, match="invalid schema"):
        _paged_artifact_schema(path)


def test_concordance_result_supports_document_and_match_pages(tmp_path) -> None:
    path = tmp_path / "concordance.parquet"
    pl.DataFrame(
        {
            "__wordflow_source_row_id": [2, 7],
            "text": ["alpha beta alpha", "beta alpha"],
            "group": ["a", "b"],
            "concordance": [
                [
                    {"CONC_matched_text": "alpha", "CONC_start_idx": 0},
                    {"CONC_matched_text": "alpha", "CONC_start_idx": 11},
                ],
                [{"CONC_matched_text": "alpha", "CONC_start_idx": 5}],
            ],
        }
    ).write_parquet(path)

    documents = pl.read_ipc_stream(
        BytesIO(
            _projected_artifact_page(
                path,
                "concordance_run_all",
                "documents",
                "text",
                ["group"],
                1,
                1,
                None,
                False,
            ).content
        )
    )
    matches = pl.read_ipc_stream(
        BytesIO(
            _projected_artifact_page(
                path,
                "concordance_run_all",
                "matches",
                "text",
                ["group"],
                1,
                2,
                None,
                False,
            ).content
        )
    )
    match_schema = pl.read_ipc_stream(
        BytesIO(
            _projected_artifact_schema(
                path, "concordance_run_all", "matches"
            )
        )
    )

    assert documents.height == 1
    assert documents["concordance"].list.len().to_list() == [2]
    assert matches["CONC_start_idx"].to_list() == [0, 11]
    assert "concordance" not in match_schema.columns


def test_quotation_result_supports_document_and_match_pages(tmp_path) -> None:
    path = tmp_path / "quotation.parquet"
    pl.DataFrame(
        {
            "__wordflow_source_row_id": [3],
            "text": ["Alice said hello and goodbye."],
            "group": ["a"],
            "quotation": [[
                {"quote": "hello", "quote_row_idx": 0},
                {"quote": "goodbye", "quote_row_idx": 1},
            ]],
        }
    ).write_parquet(path)

    documents = pl.read_ipc_stream(
        BytesIO(
            _projected_artifact_page(
                path,
                "quotation_run_all",
                "documents",
                "text",
                ["group"],
                1,
                10,
                None,
                False,
            ).content
        )
    )
    matches = pl.read_ipc_stream(
        BytesIO(
            _projected_artifact_page(
                path,
                "quotation_run_all",
                "matches",
                "text",
                ["group"],
                1,
                10,
                None,
                False,
            ).content
        )
    )

    assert documents["quotation"].list.len().to_list() == [2]
    assert matches["QUOTE_quote"].to_list() == ["hello", "goodbye"]
    assert matches["QUOTE_quote_row_idx"].to_list() == [0, 1]


def test_projected_result_rejects_generated_sort_columns(tmp_path) -> None:
    path = tmp_path / "concordance.parquet"
    pl.DataFrame(
        {
            "__wordflow_source_row_id": [0],
            "text": ["alpha"],
            "concordance": [
                [{"CONC_matched_text": "alpha", "CONC_start_idx": 0}]
            ],
        }
    ).write_parquet(path)

    with pytest.raises(InvalidInputError, match="sort column"):
        _projected_artifact_page(
            path,
            "concordance_run_all",
            "matches",
            "text",
            [],
            1,
            10,
            "CONC_start_idx",
            False,
        )


def test_concordance_density_uses_all_documents_and_exact_match_text(tmp_path) -> None:
    path = tmp_path / "concordance.parquet"
    pl.DataFrame(
        {
            "text": ["Alpha beta alpha", "alpha beta"],
            "concordance": [
                [
                    {"CONC_matched_text": "Alpha", "CONC_start_idx": 0},
                    {"CONC_matched_text": "alpha", "CONC_start_idx": 11},
                ],
                [{"CONC_matched_text": "alpha", "CONC_start_idx": 0}],
            ],
        }
    ).write_parquet(path)

    result = _concordance_density(path, "text")

    assert result.document_count == 2
    assert result.match_count == 3
    assert [item.label for item in result.series] == ["Alpha", "alpha"]
    assert sum(result.series[0].counts) == 1
    assert sum(result.series[1].counts) == 2


def test_concordance_document_projection_filters_before_count_and_paging(
    tmp_path,
) -> None:
    path = tmp_path / "concordance.parquet"
    pl.DataFrame(
        {
            "__wordflow_source_row_id": [2, 7, 8],
            "text": ["Alpha beta alpha", "Alpha beta alpha", "beta only"],
            "group": ["first", "duplicate text", "none"],
            "concordance": [
                [
                    {
                        "CONC_matched_text": "Alpha",
                        "CONC_start_idx": 0,
                        "CONC_extraction": " Alpha ",
                    },
                    {
                        "CONC_matched_text": "alpha",
                        "CONC_start_idx": 11,
                        "CONC_extraction": "alpha",
                    },
                ],
                [
                    {
                        "CONC_matched_text": "alpha",
                        "CONC_start_idx": 11,
                        "CONC_extraction": "alpha",
                    }
                ],
                [
                    {
                        "CONC_matched_text": "beta",
                        "CONC_start_idx": 0,
                        "CONC_extraction": "beta",
                    }
                ],
            ],
        }
    ).write_parquet(path)

    result = _concordance_document_projection_page(
        path,
        "text",
        ["group"],
        ConcordanceDocumentProjectionQuery(
            page=1,
            page_size=1,
            excluded_matched_texts=["Alpha", "beta"],
            bin_count=4,
            selected_bins=[2],
        ),
    )
    frame = pl.read_ipc_stream(BytesIO(result.content))

    assert result.total_rows == 2
    assert result.has_next is True
    assert frame["__wordflow_source_row_id"].to_list() == [2]
    assert frame["group"].to_list() == ["first"]
    assert frame["concordance"].list.len().to_list() == [1]
    assert frame["concordance"].to_list()[0][0]["CONC_matched_text"] == "alpha"

    sorted_result = _concordance_document_projection_page(
        path,
        "text",
        ["group"],
        ConcordanceDocumentProjectionQuery(
            page=1,
            page_size=10,
            sort_by="group",
            excluded_matched_texts=["Alpha", "beta"],
        ),
    )
    sorted_frame = pl.read_ipc_stream(BytesIO(sorted_result.content))
    assert sorted_frame["__wordflow_source_row_id"].to_list() == [7, 2]


@pytest.mark.parametrize("bin_count", [4, 5, 10, 20, 25, 50, 100])
def test_concordance_document_projection_assigns_every_bin_boundary(
    tmp_path,
    bin_count: Literal[4, 5, 10, 20, 25, 50, 100],
) -> None:
    path = tmp_path / f"concordance-{bin_count}.parquet"
    document_length = 1000
    boundary_starts = [index * document_length // bin_count for index in range(bin_count)]
    matches = [
        {
            "CONC_matched_text": f"term-{index}",
            "CONC_start_idx": start,
            "CONC_extraction": f"term-{index}",
        }
        for index, start in enumerate(boundary_starts)
    ]
    matches.append(
        {
            "CONC_matched_text": "last-character",
            "CONC_start_idx": document_length - 1,
            "CONC_extraction": "last-character",
        }
    )
    pl.DataFrame(
        {
            "__wordflow_source_row_id": [1],
            "text": ["x" * document_length],
            "concordance": [matches],
        }
    ).write_parquet(path)

    for selected_bin in range(bin_count):
        result = _concordance_document_projection_page(
            path,
            "text",
            [],
            ConcordanceDocumentProjectionQuery(
                bin_count=bin_count,
                selected_bins=[selected_bin],
            ),
        )
        frame = pl.read_ipc_stream(BytesIO(result.content))
        filtered_matches = frame["concordance"].to_list()[0]
        assert filtered_matches[0]["CONC_start_idx"] == boundary_starts[selected_bin]
        assert all(
            min(match["CONC_start_idx"] * bin_count // document_length, bin_count - 1)
            == selected_bin
            for match in filtered_matches
        )
        if selected_bin == bin_count - 1:
            assert filtered_matches[-1]["CONC_start_idx"] == document_length - 1


@pytest.mark.parametrize(
    "payload",
    [
        {"excluded_matched_texts": ["alpha", "alpha"]},
        {"bin_count": 4},
        {"selected_bins": [0]},
        {"bin_count": 4, "selected_bins": [4]},
    ],
)
def test_concordance_document_projection_query_rejects_invalid_filters(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ConcordanceDocumentProjectionQuery.model_validate(payload)
