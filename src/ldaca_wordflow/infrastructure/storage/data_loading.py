"""File-type detection, bounded data loading, and dtype normalization.

``NodeService`` uses this module to turn a validated user-file path into a
Polars frame before staging an immutable source Data Block. File preview uses
the same type vocabulary so preview and ingestion cannot disagree.
"""

import stat
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

import polars as pl


def detect_file_type(filename: str) -> str:
    """Detect file type from extension.

    Used by:
    - backend API routes, backend tests, core workspace and worker services because they
      need a backend boundary that validates inputs before delegating to workspace or worker
      state.
    """
    ext = Path(filename).suffix.lower()
    type_map = {
        ".csv": "csv",
        ".json": "json",
        ".jsonl": "jsonl",
        ".ndjson": "jsonl",
        ".parquet": "parquet",
        ".xlsx": "excel",
        ".xls": "excel",
        ".xlsm": "excel",
        ".xlsb": "excel",
        ".ods": "excel",
        ".txt": "text",
        ".text": "text",
        ".md": "text",
        ".rst": "text",
        ".log": "text",
        ".tsv": "tsv",
    }
    return type_map.get(ext, "unknown")


def load_data_file(
    file_path: Path,
    sheet_name: str | None = None,
) -> pl.LazyFrame | pl.DataFrame:
    """Load one supported user file into a Polars frame."""
    file_type = detect_file_type(file_path.name)

    if file_type == "csv":
        return pl.scan_csv(file_path)
    if file_type == "parquet":
        return pl.scan_parquet(file_path)
    if file_type == "json":
        return pl.read_json(file_path)
    if file_type == "jsonl":
        return pl.scan_ndjson(file_path)
    if file_type == "tsv":
        return pl.scan_csv(file_path, separator="\t")
    if file_type == "excel":
        validate_spreadsheet_container(file_path)
        result = (
            pl.read_excel(file_path, sheet_name=sheet_name)
            if sheet_name is not None
            else pl.read_excel(file_path)
        )
        if not isinstance(result, pl.DataFrame):
            raise RuntimeError("Excel import did not produce one DataFrame")
        return result
    if file_type == "text":
        return read_text_file(file_path)
    raise ValueError(f"Unsupported file type: {file_type}")


def read_text_file(file_path: Path) -> pl.DataFrame:
    """Read a plain text file into a single-column Polars DataFrame."""
    content = file_path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    if not lines:
        return pl.DataFrame({"text": []})
    return pl.DataFrame({"text": lines})


def validate_spreadsheet_container(file_path: Path) -> None:
    """Bound and validate ZIP-based spreadsheet containers before parsing."""

    if file_path.suffix.lower() == ".xls":
        return
    try:
        with zipfile.ZipFile(file_path) as archive:
            members = archive.infolist()
            if len(members) > 5_000:
                raise ValueError("Spreadsheet has too many members")
            expanded_total = 0
            seen: set[str] = set()
            for member in members:
                name = member.filename.rstrip("/")
                posix = PurePosixPath(name)
                windows = PureWindowsPath(name)
                if (
                    not name
                    or "\\" in name
                    or "\x00" in name
                    or posix.is_absolute()
                    or windows.drive
                    or windows.root
                    or any(part in {"", ".", ".."} for part in posix.parts)
                ):
                    raise ValueError("Spreadsheet contains an unsafe member path")
                collision = unicodedata.normalize("NFC", name).casefold()
                if collision in seen:
                    raise ValueError("Spreadsheet contains colliding member names")
                seen.add(collision)
                if member.flag_bits & 0x1:
                    raise ValueError("Encrypted spreadsheets are unsupported")
                unix_mode = (member.external_attr >> 16) & 0xFFFF
                kind = stat.S_IFMT(unix_mode)
                allowed = {0, stat.S_IFDIR} if member.is_dir() else {0, stat.S_IFREG}
                if kind not in allowed:
                    raise ValueError("Spreadsheet contains a link or special file")
                if member.file_size > 64 * 1024 * 1024:
                    raise ValueError("Spreadsheet member is too large")
                expanded_total += member.file_size
                if expanded_total > 256 * 1024 * 1024:
                    raise ValueError("Spreadsheet expands beyond the safe limit")
                if member.file_size:
                    if member.compress_size == 0:
                        raise ValueError("Spreadsheet compression ratio is invalid")
                    if member.file_size / member.compress_size > 200:
                        raise ValueError("Spreadsheet compression ratio is too high")
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ValueError("Spreadsheet container is invalid") from exc


_JS_MAX_SAFE_INTEGER = 2**53 - 1

_CANONICAL_DATETIME = pl.Datetime(time_unit="us", time_zone="UTC")
_INTEGERS_TO_PROMOTE = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
}


def normalize_dtypes(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, list[dict[str, str]]]:
    """Coerce columns to the project's canonical dtype profile.

    Returns the normalized frame plus a per-column change log
    ``[{"column", "from_dtype", "to_dtype", "reason"}, ...]`` so callers can
    surface a consolidated warning to the user. The change log is empty when
    nothing needed casting.

    ``NodeService`` returns the change log with the created Data Block so the
    caller can explain any normalization applied at ingestion.
    """
    if df.width == 0:
        return df, []

    changes: list[dict[str, str]] = []
    casts: list[pl.Expr] = []

    for col, dtype in df.schema.items():
        if isinstance(dtype, pl.Datetime):
            time_unit = dtype.time_unit
            time_zone = dtype.time_zone
            if time_unit == "us" and time_zone == "UTC":
                continue
            expr = pl.col(col)
            reason_parts: list[str] = []
            if time_zone is None:
                expr = expr.dt.replace_time_zone("UTC")
                reason_parts.append("naive datetime assumed UTC")
            elif time_zone != "UTC":
                expr = expr.dt.convert_time_zone("UTC")
                reason_parts.append(f"converted from {time_zone} to UTC")
            if time_unit != "us":
                expr = expr.dt.cast_time_unit("us")
                reason_parts.append(
                    f"precision {time_unit}->us "
                    "(text analytics does not need sub-microsecond resolution)"
                )
            casts.append(expr.alias(col))
            changes.append(
                {
                    "column": col,
                    "from_dtype": str(dtype),
                    "to_dtype": str(_CANONICAL_DATETIME),
                    "reason": "; ".join(reason_parts),
                }
            )
        elif dtype in _INTEGERS_TO_PROMOTE:
            casts.append(pl.col(col).cast(pl.Int64).alias(col))
            kind = "unsigned" if str(dtype).startswith("UInt") else "narrower signed"
            changes.append(
                {
                    "column": col,
                    "from_dtype": str(dtype),
                    "to_dtype": "Int64",
                    "reason": (
                        f"{kind} integer promoted to Int64 so joins/stacks "
                        "across heterogeneous sources align"
                    ),
                }
            )
        elif dtype == pl.Float32:
            casts.append(pl.col(col).cast(pl.Float64).alias(col))
            changes.append(
                {
                    "column": col,
                    "from_dtype": "Float32",
                    "to_dtype": "Float64",
                    "reason": "Float32 widened to Float64 for cross-source alignment",
                }
            )

    if casts:
        df = df.with_columns(casts)
    return df, changes
