"""
Tests for unified file preview endpoint
"""

from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import polars as pl


def _write_stub_xlsx(path: Path) -> None:
    """Write a structurally valid ZIP container for mocked Excel-reader tests."""

    with ZipFile(path, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"/>',
        )


def test_csv_preview_supported_types_and_preview(files_test_client, tmp_path):
    """Test CSV file preview with pagination"""
    # Arrange: create CSV in user data
    user_root = tmp_path / "users" / "root" / "files"
    csv_path = user_root / "sample.csv"
    pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).write_csv(csv_path)

    # Act
    resp = files_test_client.post(
        "/api/user-files/preview",
        json={"path": "sample.csv", "page": 1, "page_size": 2},
    )

    # Assert
    assert resp.status_code == 200
    data = resp.json()
    assert data["file_type"] == "csv"
    assert "LazyFrame" in data["supported_types"]
    assert data["columns"] == ["a", "b"]
    assert len(data["rows"]) == 2
    assert data["total_rows"] == 3
    assert data["total_pages"] == 2


def test_generic_zip_preview_is_rejected(files_test_client, tmp_path):
    """Executable/unbounded archive contents are not a user-data format."""

    user_root = tmp_path / "users" / "root" / "files"
    zip_path = user_root / "archive.zip"
    from zipfile import ZipFile

    with ZipFile(zip_path, "w") as zf:
        zf.writestr("a.txt", "hello")
        zf.writestr("b.txt", "world")

    resp = files_test_client.post(
        "/api/user-files/preview",
        json={"path": "archive.zip", "page": 1, "page_size": 10},
    )

    assert resp.status_code == 400
    assert resp.json()["code"] == "invalid_input"


def test_text_preview_returns_single_cell(files_test_client, tmp_path):
    """Plain text files should produce a 1x1 preview table."""

    user_root = tmp_path / "users" / "root" / "files"
    text_path = user_root / "example.txt"
    text_path.write_text("Plain text document.", encoding="utf-8")

    resp = files_test_client.post(
        "/api/user-files/preview",
        json={"path": "example.txt", "page": 1, "page_size": 5},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["file_type"] == "text"
    assert payload["columns"] == ["text"]
    assert payload["rows"] == [{"text": "Plain text document."}]
    assert payload["total_rows"] == 1


def test_excel_preview_returns_sheet_names_for_selector(files_test_client, tmp_path):
    """Excel preview should expose sheet names so Add File can offer a selector."""

    user_root = tmp_path / "users" / "root" / "files"
    excel_path = user_root / "with_sheet_names.xlsx"
    _write_stub_xlsx(excel_path)

    base_df = pl.DataFrame({"col_a": [1, 2], "col_b": ["x", "y"]})

    def fake_read_excel(file_path, sheet_id=None, sheet_name=None):
        if sheet_name is not None:
            return base_df
        if sheet_id == 0:
            return base_df
        if sheet_id is None:
            return base_df
        raise AssertionError("Unexpected read_excel call signature")

    class FakeReader:
        sheet_names = ["Sheet1", "Sheet2"]

    class FakeFastExcel:
        @staticmethod
        def read_excel(_source):
            return FakeReader()

    with (
        patch("ldaca_wordflow.services.file_preview.fastexcel", FakeFastExcel),
        patch(
            "ldaca_wordflow.services.file_preview.pl.read_excel",
            side_effect=fake_read_excel,
        ),
    ):
        resp = files_test_client.post(
            "/api/user-files/preview",
            json={"path": "with_sheet_names.xlsx", "page": 1, "page_size": 1},
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["file_type"] == "excel"
    assert payload["sheet_names"] == ["Sheet1", "Sheet2"]
    assert payload["selected_sheet"] == "Sheet1"
