"""Workspace node export service.

Used by:
- ``base.export_nodes`` because the route should resolve HTTP identity and
  delegate export-format validation, artifact writing, ZIP packaging, response
  sizing, and cleanup here.

Flow:
- Validate requested export format and node ids.
- Write one artifact per selected node into a temporary export directory.
- Return a single file response for one node or a ZIP response for multiple
  nodes, with background cleanup attached.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import polars as pl
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from docworkspace import Node

from ...core.exceptions import AppError, InternalServiceError, InvalidInputError
from .schema_filter import project_visible


logger = logging.getLogger(__name__)

EXPORT_FORMAT_SPECS: dict[str, dict[str, str | None]] = {
    "csv": {
        "extension": "csv",
        "media_type": "text/csv; charset=utf-8",
        "sink_method": "sink_csv",
    },
    "json": {
        "extension": "json",
        "media_type": "application/json",
        "sink_method": None,
    },
    "parquet": {
        "extension": "parquet",
        "media_type": "application/octet-stream",
        "sink_method": "sink_parquet",
    },
    "ipc": {
        "extension": "arrow",
        "media_type": "application/vnd.apache.arrow.file",
        "sink_method": "sink_ipc",
    },
    "ndjson": {
        "extension": "ndjson",
        "media_type": "application/x-ndjson",
        "sink_method": "sink_ndjson",
    },
    "xlsx": {
        "extension": "xlsx",
        "media_type": (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
        "sink_method": None,
    },
}


def _stringify_value_for_csv(value: object) -> str | None:
    """Return a CSV-safe string for values Polars would otherwise serialize oddly."""

    if isinstance(value, pl.Series):
        return str(value.to_list())
    return None if value is None else str(value)


def _stringify_lazyframe_for_csv(data: pl.LazyFrame) -> pl.LazyFrame:
    """Convert every column to string for CSV exports only."""

    return data.select(
        pl.col(column_name)
        .map_elements(_stringify_value_for_csv, return_dtype=pl.String)
        .alias(column_name)
        for column_name in data.collect_schema().names()
    )


def _prepare_dataframe_for_excel(data: pl.DataFrame) -> pl.DataFrame:
    """Drop timezone metadata from datetime columns because Excel cannot store it."""

    timezone_aware_datetime_columns = [
        pl.col(column_name).dt.replace_time_zone(None).alias(column_name)
        for column_name, dtype in data.schema.items()
        if dtype.base_type() == pl.Datetime
        and getattr(dtype, "time_zone", None) is not None
    ]
    if not timezone_aware_datetime_columns:
        return data
    return data.with_columns(timezone_aware_datetime_columns)


def _sanitize_export_label(value: str | None, fallback: str) -> str:
    """Return a filesystem/archive safe export label."""

    cleaned = "".join(
        "_" if (ord(ch) < 32 or ch in '<>:"/\\|?*') else ch
        for ch in (value or fallback).strip()
    ).strip()
    return cleaned or fallback


def _allocate_export_path(export_dir: Path, stem: str, extension: str) -> Path:
    """Allocate a non-conflicting export path inside ``export_dir``."""

    candidate = export_dir / f"{stem}.{extension}"
    suffix = 1
    while candidate.exists():
        candidate = export_dir / f"{stem}_{suffix}.{extension}"
        suffix += 1
    return candidate


def _cleanup_export_dir(export_dir: Path) -> None:
    """Best-effort cleanup for the temporary export directory."""

    try:
        shutil.rmtree(export_dir)
    except OSError as exc:
        logger.debug(
            "Best-effort export temp dir cleanup failed for %s: %s", export_dir, exc
        )


def _export_node_artifact(
    node: Node,
    node_id: str,
    export_dir: Path,
    fmt: str,
) -> tuple[str, Path]:
    """Write one node export artifact and return its archive filename/path."""

    data = project_visible(node.data)

    spec = EXPORT_FORMAT_SPECS[fmt]
    stem = _sanitize_export_label(getattr(node, "name", None), node_id)
    archive_name = f"{stem}.{spec['extension']}"
    output_path = _allocate_export_path(export_dir, stem, str(spec["extension"]))
    export_data = _stringify_lazyframe_for_csv(data) if fmt == "csv" else data

    try:
        sink_method_name = spec["sink_method"]
        if sink_method_name is not None:
            sink_method = getattr(export_data, sink_method_name, None)
            if sink_method is None:
                raise InternalServiceError(
                    f"LazyFrame export method '{sink_method_name}' is not "
                    f"available for format '{fmt}'"
                )
            sink_method(output_path)
        elif fmt == "xlsx":
            collected_data = _prepare_dataframe_for_excel(
                cast(pl.DataFrame, export_data.collect())
            )
            sheet_name = (
                "".join("_" if ch in "[]" else ch for ch in stem)[:31] or "Sheet1"
            )
            import xlsxwriter

            workbook = xlsxwriter.Workbook(str(output_path), {"strings_to_urls": False})
            try:
                collected_data.write_excel(
                    workbook=workbook,
                    worksheet=sheet_name,
                    dtype_formats={
                        pl.Date: "yyyy-mm-dd",
                        pl.Datetime: "yyyy-mm-dd hh:mm:ss",
                        pl.Time: "hh:mm:ss",
                    },
                    autofit=True,
                )
            finally:
                workbook.close()
        else:
            collected_data = cast(pl.DataFrame, export_data.collect())
            collected_data.write_json(output_path)
    except AppError:
        raise
    except Exception as exc:
        raise InternalServiceError(
            f"Failed to export node '{node_id}' as {fmt}: {exc}",
        ) from exc
    return archive_name, output_path


def _timestamp_fragment() -> str:
    """Build the timestamp prefix used for multi-node export ZIP filenames."""

    now = datetime.now()
    return (
        f"{now.month:02d}-{now.day:02d}_"
        f"{now.hour:02d}-{now.minute:02d}-{now.second:02d}"
    )


def export_workspace_nodes_response(
    *,
    workspace: Any,
    workspace_id: str,
    node_ids: str,
    export_format: str,
) -> FileResponse:
    """Build the node export ``FileResponse`` for one or more nodes.

    Used by:
    - ``base.export_nodes`` after the route resolves the path workspace.
    """

    fmt = export_format.lower()
    spec = EXPORT_FORMAT_SPECS.get(fmt)
    if spec is None:
        raise InvalidInputError(
            f"Unsupported format '{export_format}'. Supported: "
            f"{sorted(EXPORT_FORMAT_SPECS)}"
        )

    ids = [node_id.strip() for node_id in node_ids.split(",") if node_id.strip()]
    if not ids:
        raise InvalidInputError("No node_ids provided")

    export_dir = Path(tempfile.mkdtemp(prefix=f"workspace_export_{workspace_id}_"))
    cleanup_on_return = True
    try:
        exported = [
            _export_node_artifact(
                node=workspace.nodes[node_id],
                node_id=node_id,
                export_dir=export_dir,
                fmt=fmt,
            )
            for node_id in ids
        ]

        if len(exported) == 1:
            filename, artifact_path = exported[0]
            cleanup_on_return = False
            return FileResponse(
                path=str(artifact_path),
                media_type=str(spec["media_type"]),
                filename=filename,
                background=BackgroundTask(_cleanup_export_dir, export_dir),
            )

        zip_filename = (
            f"{_timestamp_fragment()}_"
            f"{_sanitize_export_label(getattr(workspace, 'name', None), workspace_id)}.zip"
        )
        zip_path = _allocate_export_path(export_dir, "_workspace_export", "zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for archive_name, artifact_path in exported:
                archive.write(artifact_path, arcname=archive_name)
        cleanup_on_return = False
        return FileResponse(
            path=str(zip_path),
            media_type="application/zip",
            filename=zip_filename,
            background=BackgroundTask(_cleanup_export_dir, export_dir),
        )
    finally:
        if cleanup_on_return:
            _cleanup_export_dir(export_dir)
