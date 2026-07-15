"""Typed Analysis Result projection tests."""

import pytest

from ldaca_wordflow.services.analysis_results import _sort_and_page
from ldaca_wordflow.shared.errors import InvalidInputError
from ldaca_wordflow.shared.json_data import JsonData


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
