"""Workspace node creation workflow.

Used by:
- ``base.add_node_to_workspace`` because the route should only resolve the
  request boundary while this module owns file loading, dtype normalization,
  node-name derivation, staging, and workspace persistence.

Flow:
- Locate the uploaded/user data file and load it through the shared loader.
- Normalize the data to a Polars DataFrame and apply dtype normalization.
- Derive the display name from the file path, stage parquet data in the
  workspace directory, create the DocWorkspace node, and persist the workspace.
"""

from __future__ import annotations

from typing import Any, cast

import polars as pl

from docworkspace import Node

from ...core.exceptions import InvalidInputError, NotFoundError
from ...core.utils import get_user_data_folder, load_data_file, normalize_dtypes
from ...core.workspace import workspace_manager
from .schema_filter import frontend_node_info
from .utils import require_workspace, stage_dataframe_as_lazy, update_workspace


def _node_name_from_filename(filename: str) -> str:
    """Return the node display name derived from a user data filename."""

    node_name = filename
    for extension in [".csv", ".tsv", ".xlsx", ".json", ".jsonl", ".parquet"]:
        if node_name.endswith(extension):
            node_name = node_name[: -len(extension)]
            break

    normalized = node_name.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    if parts:
        return parts[-1]
    return node_name


def _load_eager_dataframe(
    *,
    user_id: str,
    filename: str,
    sheet_name: str | None,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Load one user data file and normalize it to an eager DataFrame."""

    file_path = get_user_data_folder(user_id) / filename
    if not file_path.exists():
        raise InvalidInputError(f"Data file not found: {filename}")

    data = load_data_file(file_path, sheet_name=sheet_name)
    if isinstance(data, pl.LazyFrame):
        eager_data = cast(pl.DataFrame, data.collect())
    elif isinstance(data, pl.DataFrame):
        eager_data = data
    else:
        raise InvalidInputError(
            f"Expected Polars DataFrame/LazyFrame from loader, got {type(data).__name__}",
        )
    return normalize_dtypes(eager_data)


def create_workspace_node_from_file(
    *,
    user_id: str,
    workspace_id: str,
    filename: str,
    sheet_name: str | None,
    mode: str,
) -> dict[str, Any]:
    """Stage a user data file as a new workspace node and return node metadata."""

    if mode != "LazyFrame":
        raise InvalidInputError(f"Invalid mode '{mode}'. Expected one of ['LazyFrame']")

    eager_data, dtype_changes = _load_eager_dataframe(
        user_id=user_id,
        filename=filename,
        sheet_name=sheet_name,
    )
    workspace_dir = workspace_manager.get_workspace_dir(user_id, workspace_id)
    if workspace_dir is None:
        raise NotFoundError(f"Workspace folder not found for workspace {workspace_id}")

    node_name = _node_name_from_filename(filename)
    lazy_data = stage_dataframe_as_lazy(
        eager_data,
        workspace_dir,
        node_name=node_name,
        document_column=None,
    )

    workspace = require_workspace(user_id, workspace_id)
    node = Node(
        data=lazy_data,
        name=node_name,
        workspace=workspace,
        operation="manual_add",
    )
    workspace.add_node(node)
    update_workspace(user_id, workspace_id, workspace)

    info = frontend_node_info(node)
    if dtype_changes:
        info["dtype_normalization"] = dtype_changes
    return info
