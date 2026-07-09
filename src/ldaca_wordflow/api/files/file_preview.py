"""File preview helpers for heterogeneous user data files.

Used by:
- ``api.files.preview.unified_file_preview`` because the route should only
  resolve authentication and delegate format-specific preview work.
- file-preview tests because they need a focused seam for Excel sheet handling
  and preview pagination.

Flow:
- Validate the requested path against the user's data folder.
- Detect the file type and normalize preview pagination.
- Dispatch to the appropriate Polars/text/ZIP/Excel reader.
- Return the typed ``FilePreviewResponse`` consumed by the frontend preview UI.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, cast

import polars as pl

try:
    import fastexcel
except Exception:  # pragma: no cover - optional import hardening
    fastexcel: Any | None = None

from ...core.exceptions import AccessDeniedError, FileNotFoundError, InvalidInputError
from ...core.utils import (
    detect_file_type,
    read_text_file,
    read_zip_file,
    validate_file_path,
)
from ...models import FilePreviewRequest, FilePreviewResponse

logger = logging.getLogger(__name__)


def _lazy_scan(file_path, file_type: str) -> pl.LazyFrame:
    """Return a Polars LazyFrame for previewable tabular files.

    Used by:
    - ``build_file_preview_response`` for CSV, TSV, Parquet, JSONL/NDJSON, JSON,
      Excel fallback, and unknown file types.

    Why:
    - Keeps preview path memory-efficient for large tabular files while still
      providing eager fallbacks for formats without stable scanners.
    """
    ft = (file_type or "").lower()
    if ft == "csv":
        return pl.scan_csv(file_path)
    if ft == "tsv":
        return pl.scan_csv(file_path, separator="\t")
    if ft == "parquet":
        return pl.scan_parquet(file_path)
    if ft in ("jsonl", "ndjson"):
        scan_ndjson: Any = getattr(pl, "scan_ndjson", None)
        if callable(scan_ndjson):
            try:
                lf = scan_ndjson(file_path)
                if isinstance(lf, pl.LazyFrame):
                    return lf
            except Exception as exc:
                logger.debug("scan_ndjson failed for %s: %s", file_path, exc)
        return pl.read_ndjson(file_path).lazy()
    if ft == "json":
        return pl.read_json(file_path).lazy()
    if ft == "excel":
        try:
            df = pl.read_excel(file_path, sheet_id=0)
            return df.lazy()
        except Exception:
            return pl.DataFrame().lazy()
    return pl.DataFrame().lazy()


def _get_supported_types_by_extension(file_type: str) -> list[str]:
    """Return supported backend data representations by file extension.

    Used by:
    - ``build_file_preview_response`` to expose frontend capability hints.

    Why:
    - Keeps supported-type metadata deterministic and separate from route
      response assembly.
    """
    ft = (file_type or "").lower()
    mapping: dict[str, list[str]] = {
        "csv": ["LazyFrame"],
        "tsv": ["LazyFrame"],
        "jsonl": ["LazyFrame"],
        "ndjson": ["LazyFrame"],
        "json": ["LazyFrame"],
        "parquet": ["LazyFrame"],
        "excel": ["LazyFrame"],
        "text": ["LazyFrame"],
        "zip": ["LazyFrame"],
        "unknown": [],
    }
    return mapping.get(ft, [])


def _read_excel_sheet(file_path: Path, sheet_name: str) -> pl.DataFrame:
    """Read one Excel sheet as an eager Polars DataFrame.

    Used by:
    - ``_preview_excel`` when the request or workbook metadata selects a sheet.

    Why:
    - Isolates sheet-level reads so the preview dispatcher can stay focused on
      pagination and response shaping.
    """
    result = pl.read_excel(file_path, sheet_name=sheet_name)
    return _coerce_excel_result_to_dataframe(result, preferred_sheet=sheet_name)


def _coerce_excel_result_to_dataframe(
    result: Any,
    preferred_sheet: str | None = None,
) -> pl.DataFrame:
    """Normalize Polars Excel reads into one DataFrame.

    Used by:
    - ``_read_excel_sheet`` and ``_preview_excel``.

    Why:
    - Depending on Polars version/options, ``read_excel`` may return either a
      DataFrame or a dict[str, DataFrame]. Preview rendering expects a
      DataFrame, so this helper removes return-shape ambiguity.
    """
    if isinstance(result, pl.DataFrame):
        return result

    if isinstance(result, dict):
        if (
            preferred_sheet
            and preferred_sheet in result
            and isinstance(result[preferred_sheet], pl.DataFrame)
        ):
            return result[preferred_sheet]

        for value in result.values():
            if isinstance(value, pl.DataFrame):
                return value

    raise TypeError(f"Unexpected Excel read result type: {type(result)!r}")


def _list_excel_sheet_names(file_path: Path) -> list[str]:
    """Return workbook sheet names in a Polars-version-tolerant way.

    Used by:
    - ``_preview_excel`` so the frontend can offer a sheet selector.

    Flow:
    - Try fastexcel metadata when available.
    - Fall back to Polars workbook reads.
    - Fall back to parsing ``xl/workbook.xml`` from the XLSX container.
    """
    if fastexcel is not None:
        try:
            reader = fastexcel.read_excel(str(file_path))
            names = getattr(reader, "sheet_names", None)
            if names:
                return [str(name) for name in names]
        except Exception:
            pass

    workbook = pl.read_excel(file_path, sheet_id=None)
    if isinstance(workbook, dict):
        return [str(name) for name in workbook.keys()]
    if isinstance(workbook, pl.DataFrame):
        pass

    keys = getattr(workbook, "keys", None)
    if callable(keys):
        try:
            return [str(name) for name in keys()]
        except Exception:
            pass

    try:
        with zipfile.ZipFile(file_path) as zf:
            with zf.open("xl/workbook.xml") as workbook_xml:
                root = ET.parse(workbook_xml).getroot()
                names: list[str] = []
                for sheet in root.iter():
                    tag = sheet.tag.rsplit("}", 1)[-1]
                    if tag != "sheet":
                        continue
                    name = sheet.attrib.get("name")
                    if name:
                        names.append(str(name))
                if names:
                    return names
    except Exception:
        pass

    return []


def _rows_from_dataframe(df: pl.DataFrame) -> tuple[list[str], list[dict[str, Any]]]:
    """Serialize a preview DataFrame into columns plus row dictionaries.

    Used by:
    - all preview branches because the response shape is identical once a
      reader has produced a Polars DataFrame.
    """
    return list(df.columns), df.fill_null("None").to_dicts()


def _preview_excel(
    file_path: Path,
    req: FilePreviewRequest,
    *,
    offset: int,
    page_size: int,
) -> tuple[list[str], list[dict[str, Any]], int, list[str], str | None]:
    """Build preview data for one Excel workbook.

    Used by:
    - ``build_file_preview_response`` for ``file_type == "excel"``.

    Flow: list sheets, choose the requested or first available sheet, read that
        sheet or the first workbook sheet, and return the requested page slice.
    """
    sheet_names = _list_excel_sheet_names(file_path)
    payload = req.payload or {}
    requested_sheet = payload.get("sheet_name")
    selected_sheet = requested_sheet or (sheet_names[0] if sheet_names else None)

    if selected_sheet is not None:
        base_df = _read_excel_sheet(file_path, selected_sheet)
    else:
        base_df = _coerce_excel_result_to_dataframe(pl.read_excel(file_path, sheet_id=0))

    total_rows = int(base_df.height)
    df = base_df.slice(offset, page_size)
    columns, preview = _rows_from_dataframe(df)
    return columns, preview, total_rows, sheet_names, selected_sheet


def _preview_eager_dataframe(
    df: pl.DataFrame,
    *,
    offset: int,
    page_size: int,
) -> tuple[list[str], list[dict[str, Any]], int]:
    """Slice and serialize an eagerly loaded file preview DataFrame.

    Used by:
    - ZIP and plain text preview branches where the reader already returns a
      small eager DataFrame.
    """
    total_rows = int(df.height)
    if offset or page_size:
        df = df.slice(offset, page_size)
    columns, preview = _rows_from_dataframe(df)
    return columns, preview, total_rows


def build_file_preview_response(
    req: FilePreviewRequest,
    *,
    data_folder: Path,
) -> FilePreviewResponse:
    """Build a typed file preview response for a user data folder request.

    Used by:
    - ``unified_file_preview`` after authentication resolves the user's data
      folder.

    Flow:
    - Validate the requested filename stays inside ``data_folder``.
    - Detect the file type and normalize pagination.
    - Dispatch to Excel, ZIP, text, or lazy tabular preview readers.
    - Return the stable ``FilePreviewResponse`` expected by generated clients.
    """
    file_path = data_folder / req.filename

    if not validate_file_path(file_path, data_folder):
        raise AccessDeniedError("Access denied: file outside allowed directory")
    if not file_path.exists():
        raise FileNotFoundError(f"File {req.filename} not found")

    file_type = detect_file_type(file_path.name)
    supported_types = _get_supported_types_by_extension(file_type)
    page = max(0, int(req.page))
    page_size = max(1, min(500, int(req.page_size)))
    offset = page * page_size
    sheet_names: list[str] | None = None
    selected_sheet: str | None = None

    try:
        if file_type == "excel":
            columns, preview, total_rows, sheet_names, selected_sheet = _preview_excel(
                file_path,
                req,
                offset=offset,
                page_size=page_size,
            )
        elif file_type == "zip":
            columns, preview, total_rows = _preview_eager_dataframe(
                read_zip_file(file_path),
                offset=offset,
                page_size=page_size,
            )
        elif file_type == "text":
            columns, preview, total_rows = _preview_eager_dataframe(
                read_text_file(file_path),
                offset=offset,
                page_size=page_size,
            )
        else:
            df = cast(
                pl.DataFrame,
                _lazy_scan(file_path, file_type).slice(offset, page_size).collect(),
            )
            columns, preview = _rows_from_dataframe(df)
            total_rows = 0
    except Exception as exc:
        raise InvalidInputError(f"Error generating preview: {str(exc)}") from exc

    return FilePreviewResponse(
        filename=req.filename,
        file_type=file_type,
        supported_types=supported_types,
        columns=columns,
        preview=preview,
        total_rows=total_rows,
        sheet_names=sheet_names,
        selected_sheet=selected_sheet,
    )
