"""Typed Analysis Result projection tests."""

from io import BytesIO

import polars as pl
import pytest
from pydantic import ValidationError

from ldaca_wordflow.analysis.generated_columns import TOPIC_DISTRIBUTION_COLUMN
from ldaca_wordflow.models.analysis_results import QuotationResultQuery
from ldaca_wordflow.shared.topic_types import (
    topic_distribution_dtype,
    topic_distribution_storage_dtype,
)
from ldaca_wordflow.services.analysis_results import (
    _paged_artifact_page,
    _paged_artifact_schema,
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
