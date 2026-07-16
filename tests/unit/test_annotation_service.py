"""Stateless Annotation service projection tests."""

import polars as pl
import pytest

from ldaca_wordflow.services.annotations import _read_preview_page
from ldaca_wordflow.shared.errors import InvalidInputError


def test_preview_page_uses_one_based_paging_and_zero_based_row_offsets() -> None:
    texts, total, start = _read_preview_page(
        pl.DataFrame({"text": ["first", "second"]}).lazy(),
        "text",
        2,
        1,
    )

    assert texts == ["second"]
    assert total == 2
    assert start == 1


def test_preview_page_rejects_a_missing_text_column() -> None:
    with pytest.raises(InvalidInputError):
        _read_preview_page(
            pl.DataFrame({"other": ["first"]}).lazy(),
            "text",
            1,
            20,
        )
