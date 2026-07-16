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

from functools import partial
from pathlib import Path
from typing import Any

import anyio
import fastexcel
import polars as pl
from anyio.to_thread import run_sync as run_sync_in_worker_thread

from ..shared.errors import FileNotFoundError, InvalidInputError, ResourceTooLargeError
from ..infrastructure.storage.data_loading import (
    detect_file_type,
    read_text_file,
    validate_spreadsheet_container,
)
from .user_files import UserFileStore
from ..models.files import FilePreviewRequest, FilePreviewResource


class FileReadService:
    """Own bounded preview, metadata, and UTF-8 reads after safe resolution."""

    def __init__(
        self,
        file_store: UserFileStore,
        *,
        limiter: anyio.CapacityLimiter,
        max_preview_bytes: int,
        max_text_bytes: int,
    ) -> None:
        self._file_store = file_store
        self._limiter = limiter
        self._max_preview_bytes = max_preview_bytes
        self._max_text_bytes = max_text_bytes

    async def preview(
        self, user_id: str, request: FilePreviewRequest
    ) -> FilePreviewResource:
        async with self._file_store.read_path(user_id, request.path) as path:
            if (await self._stat(path)).st_size > self._max_preview_bytes:
                raise ResourceTooLargeError("File is too large to preview")
            return await run_sync_in_worker_thread(
                partial(build_file_preview_response, request, file_path=path),
                abandon_on_cancel=False,
                limiter=self._limiter,
            )

    async def read_text(self, user_id: str, relative_path: str) -> tuple[str, str]:
        async with self._file_store.read_path(user_id, relative_path) as path:
            if (await self._stat(path)).st_size > self._max_text_bytes:
                raise ResourceTooLargeError("File is too large for a text response")
            try:
                content = await run_sync_in_worker_thread(
                    partial(path.read_text, encoding="utf-8"),
                    abandon_on_cancel=False,
                    limiter=self._limiter,
                )
            except UnicodeDecodeError as exc:
                raise InvalidInputError("File is not valid UTF-8 text") from exc
            media_type = (
                "text/markdown" if path.suffix.lower() == ".md" else "text/plain"
            )
            return content, media_type

    async def _stat(self, path: Path):
        return await run_sync_in_worker_thread(
            path.stat,
            abandon_on_cancel=False,
            limiter=self._limiter,
        )


def _lazy_scan(file_path, file_type: str) -> pl.LazyFrame:
    """Return a Polars LazyFrame for previewable tabular files.

    Used by:
    - ``build_file_preview_response`` for CSV, TSV, Parquet, JSONL/NDJSON, and
      JSON files.

    Why:
    - Keeps preview paths lazy where Polars provides a scanner.
    """
    ft = file_type.lower()
    if ft == "csv":
        return pl.scan_csv(file_path)
    if ft == "tsv":
        return pl.scan_csv(file_path, separator="\t")
    if ft == "parquet":
        return pl.scan_parquet(file_path)
    if ft == "jsonl":
        return pl.scan_ndjson(file_path)
    if ft == "json":
        return pl.read_json(file_path).lazy()
    raise InvalidInputError(f"File type is not previewable: {file_type}")


def _get_supported_types_by_extension(file_type: str) -> list[str]:
    """Return supported backend data representations by file extension.

    Used by:
    - ``build_file_preview_response`` to expose frontend capability hints.

    Why:
    - Keeps supported-type metadata deterministic and separate from route
      response assembly.
    """
    ft = file_type.lower()
    mapping: dict[str, list[str]] = {
        "csv": ["LazyFrame"],
        "tsv": ["LazyFrame"],
        "jsonl": ["LazyFrame"],
        "json": ["LazyFrame"],
        "parquet": ["LazyFrame"],
        "excel": ["LazyFrame"],
        "text": ["LazyFrame"],
        "unknown": [],
    }
    return mapping[ft]


def _read_excel_sheet(file_path: Path, sheet_name: str) -> pl.DataFrame:
    """Read one Excel sheet as an eager Polars DataFrame.

    Used by:
    - ``_preview_excel`` when the request or workbook metadata selects a sheet.

    Why:
    - Isolates sheet-level reads so the preview dispatcher can stay focused on
      pagination and response shaping.
    """
    result = pl.read_excel(file_path, sheet_name=sheet_name)
    if not isinstance(result, pl.DataFrame):
        raise TypeError("A single Excel sheet did not produce a DataFrame")
    return result


def _list_excel_sheet_names(file_path: Path) -> list[str]:
    """Return workbook sheet names from the required Excel engine.

    Used by:
    - ``_preview_excel`` so the frontend can offer a sheet selector.

    The backend pins ``fastexcel`` as a runtime dependency, so a missing or
    malformed workbook fails the preview instead of selecting an alternate
    parser with different semantics.
    """
    return [str(name) for name in fastexcel.read_excel(file_path).sheet_names]


def _rows_from_dataframe(df: pl.DataFrame) -> tuple[list[str], list[dict[str, Any]]]:
    """Serialize a preview DataFrame into columns plus row dictionaries.

    Used by:
    - all preview branches because the response shape is identical once a
      reader has produced a Polars DataFrame.
    """
    return list(df.columns), df.to_dicts()


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
    if not sheet_names:
        raise InvalidInputError("Excel workbook contains no sheets")
    requested_sheet = req.sheet_name
    if requested_sheet is not None and requested_sheet not in sheet_names:
        raise InvalidInputError("Excel sheet not found")
    selected_sheet = requested_sheet or sheet_names[0]
    base_df = _read_excel_sheet(file_path, selected_sheet)

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
    - plain text preview where the reader already returns a small eager
      DataFrame.
    """
    total_rows = int(df.height)
    df = df.slice(offset, page_size)
    columns, preview = _rows_from_dataframe(df)
    return columns, preview, total_rows


def build_file_preview_response(
    req: FilePreviewRequest,
    *,
    file_path: Path,
) -> FilePreviewResource:
    """Build a typed preview for a path already resolved by ``UserFileStore``.

    Used by:
    - ``unified_file_preview`` after authentication resolves the user's data
      folder.

    Flow:
    - Detect the file type and normalize pagination.
    - Dispatch to Excel, text, or lazy tabular preview readers.
    - Return the stable ``FilePreviewResponse`` expected by generated clients.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File {req.path} not found")

    file_type = detect_file_type(file_path.name)
    supported_types = _get_supported_types_by_extension(file_type)
    page = req.page
    page_size = req.page_size
    offset = (page - 1) * page_size
    sheet_names: list[str] | None = None
    selected_sheet: str | None = None

    try:
        if file_type == "excel":
            validate_spreadsheet_container(file_path)
            columns, preview, total_rows, sheet_names, selected_sheet = _preview_excel(
                file_path,
                req,
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
            if file_type == "unknown":
                raise InvalidInputError("File type is not previewable")
            lazyframe = _lazy_scan(file_path, file_type)
            total_frame = lazyframe.select(pl.len()).collect()
            total_rows = int(total_frame.item())
            df = lazyframe.slice(offset, page_size).collect()
            columns, preview = _rows_from_dataframe(df)
    except InvalidInputError:
        raise
    except (
        OSError,
        UnicodeError,
        ValueError,
        fastexcel.FastExcelError,
        pl.exceptions.PolarsError,
    ) as exc:
        raise InvalidInputError("File preview could not be generated") from exc

    total_pages = (total_rows + page_size - 1) // page_size if total_rows else 0
    return FilePreviewResource(
        path=req.path,
        file_type=file_type,
        supported_types=supported_types,
        columns=columns,
        rows=preview,
        page=page,
        page_size=page_size,
        total_rows=total_rows,
        total_pages=total_pages,
        sheet_names=sheet_names,
        selected_sheet=selected_sheet,
    )
