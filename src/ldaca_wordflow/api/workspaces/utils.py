"""Shared utility helpers for workspace API modules.

Used by:
- FastAPI workspace routers, frontend workspace features, and backend tests because they need this unit's "Shared utility helpers for workspace API modules" behavior.

Flow:
- Workspace and analysis routes call these helpers for shared persistence and artifact staging.
- Helpers sanitize names, allocate workspace data paths, scan Parquet lazily, and sync task state.
- Callers receive saved workspace paths, staged LazyFrames, or consistent HTTP errors.
"""

import logging
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import polars as pl
from docworkspace.workspace.core import Workspace

from docworkspace import Node

from ...analysis.models import AnalysisStatus
from ...core.exceptions import (
    InternalServiceError,
    InvalidInputError,
    NotFoundError,
    WorkspaceNotFoundError,
)
from ...core.workspace import workspace_manager

logger = logging.getLogger(__name__)

ISO_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d{1,6})?)?(Z|[+\-]\d{2}:?\d{2})$"
)


def _safe_workspace_data_stem(name: str) -> str:
    """Create safe workspace data stem values for workspace file utilities.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Create safe workspace data stem values for workspace file utilities" behavior.
    """

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "data"


def _allocate_workspace_data_path(
    workspace_dir: Path, *, stem: str, suffix: str = ".parquet"
) -> Path:
    """Support workspace file utilities with an allocate workspace data path helper.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Support workspace file utilities with an allocate workspace data path helper" behavior.
    """

    data_dir = workspace_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    candidate = data_dir / f"{stem}{suffix}"
    suffix_index = 1
    while candidate.exists():
        candidate = data_dir / f"{stem}_{suffix_index}{suffix}"
        suffix_index += 1
    return candidate


def _scan_workspace_parquet(parquet_path: Path):
    """Support workspace file utilities with a scan workspace parquet helper.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Support workspace file utilities with a scan workspace parquet helper" behavior.
    """

    absolute_path = Path(parquet_path).resolve()
    try:
        lazy_data: Any = pl.scan_parquet(absolute_path)
    except Exception as exc:
        raise InternalServiceError(f"Failed to reload parquet as LazyFrame: {exc}")
    return lazy_data


def update_workspace(
    user_id: str,
    workspace_id: str,
    workspace: Any,
    *,
    best_effort: bool = False,
) -> Path | None:
    """Persist workspace metadata/path updates through one shared code path.

    Used by:
    - workspace lifecycle, node, and analysis endpoints after they have resolved
      the target workspace from an explicit route path.

    Why:
    - Removes repeated save/update boilerplate from route handlers without
      letting persistence infer the target from mutable current-workspace state.
    """
    try:
        workspace.modified_at = datetime.now().isoformat()
        target_dir = workspace_manager._resolve_workspace_dir(
            user_id=user_id,
            workspace_id=workspace_id,
            workspace_name=workspace.name,
        )
        workspace_manager._attach_workspace_dir(workspace, target_dir)
        workspace.save(target_dir)
        workspace_manager._set_cached_path(user_id, workspace_id, target_dir)
        return target_dir
    except Exception:
        if best_effort:
            return None
        raise


async def ensure_task_synced(
    user_id: str,
    workspace_id: str,
    task_id: str,
    memory_task_manager,
):
    """Sync the in-memory task status with the backend worker task manager.

        If the in-memory task is 'running', this checks the worker
    status and updates the in-memory task if the worker has completed (success/fail).

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

        Used by:
        - analysis task-result endpoints that bridge memory store and worker store because they need this unit's "Sync the in-memory task status with the backend worker task manager" behavior.

        Why:
        - Keeps in-memory task records consistent with worker completion.

        Refactor note:
        - Similar sync logic appears across analysis routes; extraction to a shared
            task-sync service could reduce endpoint duplication.
    """
    task = memory_task_manager.get_task(task_id)
    if not task:
        return None

    # Check against string or Enum to be safe.
    # Pending tasks can already exist in analysis storage while the worker task
    # is actively running, so both states should be sync-eligible.
    is_running = task.status in {
        "running",
        "pending",
        AnalysisStatus.RUNNING,
        AnalysisStatus.PENDING,
    }

    if is_running:
        worker_tm = workspace_manager.get_task_manager(user_id)
        try:
            tm_task = await worker_tm.get_task(task.task_id)
            if tm_task:
                from ...analysis.results import GenericAnalysisResult

                if tm_task.status == "successful":
                    task.complete(GenericAnalysisResult(tm_task.result))
                    memory_task_manager.save_task(task)
                elif tm_task.status == "failed":
                    task.fail(tm_task.error or "Task failed")
                    memory_task_manager.save_task(task)
        except Exception as exc:
            logger.debug(
                "Failed to sync task %s from worker manager: %s",
                task.task_id,
                exc,
            )
    return task


def success(data=None, message: str = "ok", state: str = "successful", **extra):
    """Build a standardized success payload.

    Used by:
    - workspace API handlers returning `{state,message,data}` contracts because they need this unit's "Build a standardized success payload" behavior.

    Why:
    - Keeps response assembly lightweight; serialization is handled by FastAPI.
    """
    payload = {"state": state, "message": message, "data": data}
    if extra:
        payload.update(extra)
    return payload


def running(message: str = "running", metadata: dict | None = None):
    """Shortcut for standardized in-progress response payloads.

    Used by:
    - task-producing endpoints that return pre-completion status because they need this unit's "Shortcut for standardized in-progress response payloads" behavior.

    Why:
    - Aligns `running` responses with the same schema as `success`.
    """
    return success(data=None, message=message, state="running", metadata=metadata or {})


def stage_dataframe_as_lazy(
    data: pl.DataFrame,
    workspace_dir: Path,
    node_name: str,
    document_column: str | None = None,
):
    """Persist a dataframe to parquet under the workspace and reload as LazyFrame.

    This mirrors the lazy serialize/reload pattern used by the base add-node endpoint
    so that detached/derived nodes remain portable and lazy by default.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Used by:
    - backend API routes, backend tests because they need this unit's "Persist a dataframe to parquet under the workspace and reload as LazyFrame" behavior.
    """
    parquet_path = _allocate_workspace_data_path(
        workspace_dir,
        stem=_safe_workspace_data_stem(node_name),
    )

    if not isinstance(data, pl.DataFrame):
        raise InvalidInputError(
            f"Expected Polars DataFrame for staging, got {type(data).__name__}",
        )
    df = data

    try:
        df.write_parquet(parquet_path)
    except Exception as exc:
        raise InternalServiceError(f"Failed to persist parquet for workspace: {exc}")
    return _scan_workspace_parquet(parquet_path)


def stage_parquet_artifact_as_lazy(
    artifact_path: str | Path,
    workspace_dir: Path,
    node_name: str,
) -> tuple[Any, Path]:
    """Copy a temporary parquet artifact into workspace data and reload lazily.

    Background workers write ephemeral parquet artifacts under `data/artifacts`.
    Before attaching a derived node to the workspace, the main process must copy
    that parquet into durable workspace storage so the node survives artifact
    cleanup on unload.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Used by:
    - backend API routes because they need this unit's "Copy a temporary parquet artifact into workspace data and reload lazily" behavior.
    """

    source_path = Path(artifact_path)
    if not source_path.exists() or not source_path.is_file():
        raise NotFoundError(
            f"Artifact parquet not found: {source_path}",
        )
    persisted_path = _allocate_workspace_data_path(
        workspace_dir,
        stem=_safe_workspace_data_stem(node_name or source_path.stem),
    )

    try:
        shutil.copy2(source_path, persisted_path)
    except Exception as exc:
        raise InternalServiceError(
            f"Failed to copy artifact parquet into workspace data: {exc}",
        )
    return _scan_workspace_parquet(persisted_path), persisted_path


def require_workspace(user_id: str, workspace_id: str) -> Workspace:
    """Resolve a workspace by explicit id, loading it when needed.

    Used by:
    - explicit workspace-scoped routes because the target workspace should come
      from the request path instead of the user's hidden current-workspace
      pointer.

    Flow: reject blank ids, reuse the loaded workspace when it already matches,
        otherwise ask the manager to load the requested id and raise the
        standard workspace not-found error when it cannot be resolved.
    """
    if not workspace_id.strip():
        raise WorkspaceNotFoundError("Workspace not found")

    current_workspace_id = workspace_manager.get_current_workspace_id(user_id)
    current_workspace = workspace_manager.get_current_workspace(user_id)
    if current_workspace_id == workspace_id and current_workspace is not None:
        return current_workspace

    if not workspace_manager.set_current_workspace(user_id, workspace_id):
        raise WorkspaceNotFoundError("Workspace not found")

    workspace = workspace_manager.get_current_workspace(user_id)
    if workspace is None:
        raise WorkspaceNotFoundError("Workspace not found")
    return workspace


def _parse_temporal(value: Any) -> Any:
    """Parse ISO-like datetime strings into datetime objects."""
    if isinstance(value, str) and ISO_PATTERN.match(value):
        s = value
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        if re.search(r"([+\-]\d{2})(\d{2})$", s):
            s = re.sub(r"([+\-]\d{2})(\d{2})$", r"\1:\2", s)
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return value
    return value


def _coerce_scalar(value: Any) -> Any:
    """Coerce string scalars into bool/int/float when safe."""
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        try:
            if "." in value:
                return float(value)
            return int(value)
        except Exception:
            return value
    return value


def _serialize_column_scalar(value: Any) -> str | int | float | bool:
    """Serialize a column scalar to a JSON-safe primitive."""
    if isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _make_temporal_literal(value: datetime, column_dtype: Any) -> pl.Expr:
    """Build a polars literal matching the column's datetime dtype."""
    if isinstance(column_dtype, pl.Datetime):
        column_tz = getattr(column_dtype, "time_zone", None)
        if column_tz is None and value.tzinfo is not None:
            value = value.replace(tzinfo=None)
        elif column_tz is not None and value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return pl.lit(value).cast(column_dtype)
    if isinstance(column_dtype, pl.Date):
        return pl.lit(value.date()).cast(column_dtype)
    return pl.lit(value)


def _is_string_list_dtype(dtype: Any) -> bool:
    """Return True when dtype is exactly a list of strings."""
    return dtype == pl.List(pl.String) or dtype == pl.List(pl.Utf8)


def _extract_lazy_schema(lazy_frame: pl.LazyFrame) -> tuple[list[str], dict[str, str]]:
    """Collect LazyFrame schema without collecting row data."""
    schema_dict = dict(lazy_frame.collect_schema().items())
    columns = list(schema_dict.keys())
    dtypes = {col: str(dtype) for col, dtype in schema_dict.items()}
    return columns, dtypes


def _propagated_tokenization(
    parents: Node | list[Node],
    result_lf: pl.LazyFrame,
) -> dict[str, Any]:
    """Return parent tokenization metadata whose source column survived."""
    if isinstance(parents, list):
        sources = parents
    else:
        sources = [parents]
    try:
        result_columns = set(result_lf.collect_schema().names())
    except Exception:
        result_columns = None
    merged: dict[str, Any] = {}
    for parent in sources:
        tokenization = getattr(parent, "tokenization", None)
        if not isinstance(tokenization, dict):
            continue
        for source_column, meta in tokenization.items():
            if not isinstance(source_column, str) or not isinstance(meta, dict):
                continue
            if result_columns is not None and source_column not in result_columns:
                continue
            existing = merged.get(source_column)
            if existing is not None and existing != meta:
                raise ValueError(
                    f"Conflicting tokenization metadata for column {source_column!r}."
                )
            merged[source_column] = meta
    return merged


def _validate_existing_column(node: Node, column_name: str) -> None:
    """Raise 400 if column_name is absent from the node's schema."""
    schema_names = node.data.collect_schema().names()
    if column_name not in schema_names:
        raise InvalidInputError(
            f"Node {node.name!r} has no column {column_name!r}; "
            f"available columns: {sorted(schema_names)}"
        )


def _paginated_lazy_preview(
    lazyframe: pl.LazyFrame,
    page: int,
    page_size: int,
) -> tuple[list[dict[str, Any]], list[str], dict[str, str], Any]:
    """Slice a LazyFrame for paginated preview and return data + metadata."""
    try:
        total_rows_df = cast(
            pl.DataFrame,
            lazyframe.select(pl.len().alias("_len")).collect(),
        )
        total_rows_series = total_rows_df.to_series(0)
        total_rows = int(total_rows_series.item()) if total_rows_series.len() else 0
    except Exception:
        total_rows = 0

    total_pages = math.ceil(total_rows / page_size) if total_rows else 0
    normalized_page = min(max(page, 1), total_pages or 1)
    start_idx = (normalized_page - 1) * page_size if total_rows else 0

    preview_df = cast(
        pl.DataFrame,
        lazyframe.slice(start_idx, page_size).collect(),
    )
    columns = list(preview_df.columns)
    dtypes = {col: str(dtype) for col, dtype in preview_df.schema.items()}
    data_rows = preview_df.to_dicts()

    from ...models import PaginationInfo

    pagination = PaginationInfo(
        page=normalized_page,
        page_size=page_size,
        total_rows=total_rows,
        total_pages=total_pages,
        has_next=normalized_page < total_pages,
        has_prev=normalized_page > 1 and total_rows > 0,
    )
    return data_rows, columns, dtypes, pagination


def _create_and_persist_child_node(
    *,
    workspace: Workspace,
    data: pl.LazyFrame,
    name: str,
    operation: str,
    parents: list[Node],
    user_id: str,
    workspace_id: str,
    tokenization: dict[str, Any] | None = None,
    document: str | None = None,
) -> Node:
    """Create a Node, add it to the workspace, persist, and return it."""
    if tokenization is None:
        tokenization = _propagated_tokenization(parents, data)
    new_node = Node(
        data=data,
        name=name,
        workspace=workspace,
        operation=operation,
        parents=parents,
        tokenization=tokenization,
        document=document,
    )
    workspace.add_node(new_node)
    # Smart insertion: keep the list view and connector arrows tidy by placing the
    # derived node directly below its mother node instead of at the end of the list.
    workspace.place_node_after_parent(new_node)
    update_workspace(user_id, workspace_id, workspace)
    return new_node


__all__ = [
    "success",
    "running",
    "failed",
    "_build_detach_options",
    "_coerce_scalar",
    "_create_and_persist_child_node",
    "_extract_lazy_schema",
    "_is_string_list_dtype",
    "_make_temporal_literal",
    "_paginated_lazy_preview",
    "_parse_temporal",
    "_propagated_tokenization",
    "_serialize_column_scalar",
    "_validate_existing_column",
    "ensure_task_synced",
    "require_workspace",
    "stage_dataframe_as_lazy",
    "stage_parquet_artifact_as_lazy",
    "update_workspace",
]


def _build_detach_options(
    workspace,
    node,
    node_id: str,
    column: str,
    *,
    mandatory_columns: list[str],
    extraction_column: str,
    node_option_class,
    detach_options_response_class,
    message: str,
    schema_filter=None,
    task_metadata=None,
):
    """Build a shared detach-options response for analysis tool endpoints.

    Used by:
    - concordance and quotation detach-options route handlers because they need
      a shared response builder that avoids duplicating column-ordering logic.

    Steps:
    - Collect available schema columns, optionally filtering transient internals.
    - Order columns as: text column first, generated columns, then optional metadata.
    - Build the node option and wrap it in the correct tool-specific response model.
    """
    raw_columns = node.data.collect_schema().names()
    if schema_filter:
        raw_columns = [c for c in raw_columns if schema_filter(c)]

    mandatory_set = set(mandatory_columns)
    optional_columns = [
        col
        for col in [column, extraction_column, *raw_columns]
        if col not in mandatory_set
    ]
    ordered_available_columns = list(
        dict.fromkeys([column, *mandatory_columns, *optional_columns])
    )

    node_option = node_option_class(
        node_id=node_id,
        node_name=getattr(node, "name", None) or node_id,
        text_column=column,
        available_columns=ordered_available_columns,
        # Every column is user-choosable now (default-selected on the client,
        # deselectable). `mandatory_columns` is only used to order the
        # generated columns ahead of optional metadata — nothing is disabled.
        disabled_columns=[],
    )

    return detach_options_response_class(
        state="successful",
        message=message,
        data={"nodes": [node_option]},
        metadata=task_metadata,
    )
