"""Contract tests for the canonical Data Block source loader."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from ldaca_wordflow.infrastructure.storage.data_loading import (
    DataFileLoadError,
    detect_file_type,
    load_data_file,
)


def test_json_lines_extensions_share_one_ingestion_path(tmp_path: Path) -> None:
    for extension in ("jsonl", "ndjson"):
        path = tmp_path / f"records.{extension}"
        path.write_text('{"text":"first"}\n{"text":"second"}\n', encoding="utf-8")

        assert detect_file_type(path.name) == "jsonl"
        loaded = load_data_file(path)
        assert isinstance(loaded, pl.LazyFrame)
        assert loaded.collect().to_dicts() == [
            {"text": "first"},
            {"text": "second"},
        ]


def test_text_ingestion_rejects_invalid_utf8_instead_of_replacing_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.txt"
    path.write_bytes(b"valid\xffinvalid")

    with pytest.raises(DataFileLoadError):
        load_data_file(path)
