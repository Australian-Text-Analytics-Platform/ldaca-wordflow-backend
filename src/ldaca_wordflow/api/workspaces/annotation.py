"""Annotation workspace helpers.

Used by:
- The frontend Annotation view because it needs a backend-created class
  description table that behaves like any other workspace data block.

Flow:
- Resolve the current user/workspace, create an empty two-column Polars table,
  stage it into workspace storage, add it as a root node, persist the workspace,
  and return normal frontend node metadata.
"""

from __future__ import annotations

from typing import cast

import polars as pl
from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from ...core.auth import get_current_user
from ...core.exceptions import InvalidInputError, NotFoundError
from ...models import WorkspaceNodeInfo
from .schema_filter import frontend_node_info
from .utils import Node, require_current_workspace, stage_dataframe_as_lazy, update_workspace

router = APIRouter(prefix="/workspaces", tags=["annotation"])

CLASS_DESCRIPTION_NODE_BASE_NAME = "Annotation Classes"


class AnnotationClassDescriptionRow(BaseModel):
    """One editable class-description row returned to the Annotation UI.

    Used by:
    - ``get_annotation_class_descriptions`` and
      ``update_annotation_class_descriptions`` because the frontend editor
      should always work with semantic ``class``/``description`` keys even when
      the selected workspace columns have different names.
    """

    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(default="", alias="class")
    description: str = ""


class AnnotationClassDescriptionsPayload(BaseModel):
    """Class-description editor payload for one selected workspace node.

    Used by:
    - Annotation class-description GET/PUT routes to round-trip the selected
      class and description columns plus their editable row values.
    """

    class_column: str = "class"
    description_column: str = "description"
    rows: list[AnnotationClassDescriptionRow] = Field(default_factory=list)


def _unique_class_description_node_name(existing_names: set[str]) -> str:
    """Return a stable display name for a new annotation class table.

    Called by:
    - ``create_annotation_class_descriptions`` because each generated table
      should be easy to distinguish in the workspace node list.

    Flow: prefer the base name, then append an incrementing integer suffix when
    the workspace already has a node with that name.
    """
    if CLASS_DESCRIPTION_NODE_BASE_NAME not in existing_names:
        return CLASS_DESCRIPTION_NODE_BASE_NAME
    suffix = 2
    while f"{CLASS_DESCRIPTION_NODE_BASE_NAME} {suffix}" in existing_names:
        suffix += 1
    return f"{CLASS_DESCRIPTION_NODE_BASE_NAME} {suffix}"


def _validate_class_description_columns(
    schema: dict[str, pl.DataType],
    class_column: str,
    description_column: str,
) -> None:
    """Validate the two columns the Annotation editor is about to display/edit.

    Called by:
    - class-description GET/PUT routes because both need the same column guards
      before collecting or rewriting the node's data.
    """
    if class_column == description_column:
        raise InvalidInputError("Class and description columns must be different")
    missing = [
        column
        for column in (class_column, description_column)
        if column not in schema
    ]
    if missing:
        raise InvalidInputError(f"Column not found: {', '.join(missing)}")


def _class_description_payload_from_node(
    node: Node,
    class_column: str,
    description_column: str,
) -> AnnotationClassDescriptionsPayload:
    """Collect a node's selected class-description columns for the editor.

    Called by:
    - ``get_annotation_class_descriptions`` and the PUT route after mutation so
      both responses use identical serialization and null-to-empty handling.
    """
    schema = dict(node.data.collect_schema().items())
    _validate_class_description_columns(schema, class_column, description_column)
    df = cast(
        pl.DataFrame,
        node.data.select(
            [
                pl.col(class_column).cast(pl.String).fill_null("").alias("class"),
                pl.col(description_column)
                .cast(pl.String)
                .fill_null("")
                .alias("description"),
            ]
        ).collect(),
    )
    rows = [
        AnnotationClassDescriptionRow.model_validate(row)
        for row in df.to_dicts()
    ]
    return AnnotationClassDescriptionsPayload(
        class_column=class_column,
        description_column=description_column,
        rows=rows,
    )


def _updated_class_description_frame(
    existing: pl.DataFrame,
    payload: AnnotationClassDescriptionsPayload,
) -> pl.DataFrame:
    """Return a dataframe with edited class-description rows applied.

    Called by:
    - ``update_annotation_class_descriptions`` because the route needs to
      rewrite only the two selected columns while preserving any extra columns
      when users point the editor at an existing table.
    """
    row_count = len(payload.rows)
    values: dict[str, list[object]] = {}
    for column in existing.columns:
        if column == payload.class_column:
            values[column] = [row.class_name for row in payload.rows]
        elif column == payload.description_column:
            values[column] = [row.description for row in payload.rows]
        else:
            existing_values = existing[column].to_list()
            values[column] = [
                existing_values[index] if index < len(existing_values) else None
                for index in range(row_count)
            ]
    schema = dict(existing.schema)
    schema[payload.class_column] = cast(pl.DataType, pl.String)
    schema[payload.description_column] = cast(pl.DataType, pl.String)
    return pl.DataFrame(values, schema=schema)


@router.post("/annotation/class-descriptions", response_model=WorkspaceNodeInfo)
async def create_annotation_class_descriptions(
    current_user: dict = Depends(get_current_user),
):
    """Create an empty class-description table for the Annotation view.

    Used by:
    - Frontend Annotation view through the generated SDK because users need a
      one-click way to seed the class/description table required by the view.

    Flow:
    - Resolve the active workspace.
    - Build an empty ``class``/``description`` table with string dtypes.
    - Persist it to the workspace data folder and add it as a root node.
    - Save the workspace and return the standard node metadata response.
    """
    user_id = current_user["id"]
    workspace = require_current_workspace(user_id)
    node_name = _unique_class_description_node_name(
        {node.name for node in workspace.nodes.values()}
    )
    data = pl.DataFrame(schema={"class": pl.String, "description": pl.String})
    lazy_data = stage_dataframe_as_lazy(data, workspace.ws_root_dir, node_name=node_name)
    node = Node(
        data=lazy_data,
        name=node_name,
        workspace=workspace,
        operation="annotation_class_descriptions",
    )
    workspace.add_node(node)
    update_workspace(user_id, workspace.id, workspace)
    return frontend_node_info(node)


class AnnotationSetParentRequest(BaseModel):
    """Request to set a class-description node's parent to the source node.

    Used by:
    - Frontend Annotation view when Start is clicked because the class table
      should hang off the annotated source block in the workspace graph.
    """

    parent_node_id: str


@router.put(
    "/annotation/class-descriptions/{node_id}/parent",
    response_model=WorkspaceNodeInfo,
)
async def set_annotation_class_parent(
    node_id: str,
    payload: AnnotationSetParentRequest,
    current_user: dict = Depends(get_current_user),
):
    """Reparent a class-description node under the annotated source node.

    Used by:
    - Frontend Annotation view on Start because making the class table a child
      of the source node keeps the annotation lineage visible in the graph.

    Flow:
    - Resolve workspace, source, and class nodes (404 if missing).
    - Reject self-parenting, then set the class node's only parent to the source
      and re-tidy its list position.
    - Persist and return the standard node metadata response.
    """
    user_id = current_user["id"]
    workspace = require_current_workspace(user_id)
    node = workspace.nodes.get(node_id)
    if node is None:
        raise NotFoundError("Class description node not found")
    parent = workspace.nodes.get(payload.parent_node_id)
    if parent is None:
        raise NotFoundError("Source node not found")
    if payload.parent_node_id == node_id:
        raise InvalidInputError("A node cannot be its own parent")
    node.parents = [parent]
    workspace.place_node_after_parent(node)
    update_workspace(user_id, workspace.id, workspace)
    return frontend_node_info(node)


@router.get(
    "/annotation/class-descriptions/{node_id}",
    response_model=AnnotationClassDescriptionsPayload,
)
async def get_annotation_class_descriptions(
    node_id: str,
    class_column: str = "class",
    description_column: str = "description",
    current_user: dict = Depends(get_current_user),
):
    """Return editable class-description rows for one selected node.

    Used by:
    - Frontend Annotation view because the class editor needs a compact
      two-column row payload independent of the full workspace table UI.
    """
    workspace = require_current_workspace(current_user["id"])
    node = workspace.nodes.get(node_id)
    if node is None:
        raise NotFoundError("Node not found")
    return _class_description_payload_from_node(node, class_column, description_column)


@router.put(
    "/annotation/class-descriptions/{node_id}",
    response_model=AnnotationClassDescriptionsPayload,
)
async def update_annotation_class_descriptions(
    node_id: str,
    payload: AnnotationClassDescriptionsPayload,
    current_user: dict = Depends(get_current_user),
):
    """Persist edited class-description rows back to the selected node.

    Used by:
    - Frontend Annotation view on cell blur/add-row because class labels and
      descriptions should behave like workspace data, not transient UI state.

    Flow:
    - Validate the node and selected columns.
    - Rewrite the selected class/description columns from the submitted rows.
    - Restage the dataframe as workspace data and save the workspace.
    """
    user_id = current_user["id"]
    workspace = require_current_workspace(user_id)
    node = workspace.nodes.get(node_id)
    if node is None:
        raise NotFoundError("Node not found")
    existing = cast(pl.DataFrame, node.data.collect())
    _validate_class_description_columns(
        dict(existing.schema),
        payload.class_column,
        payload.description_column,
    )
    updated = _updated_class_description_frame(existing, payload)
    node.data = stage_dataframe_as_lazy(updated, workspace.ws_root_dir, node_name=node.name)
    update_workspace(user_id, workspace.id, workspace)
    return _class_description_payload_from_node(
        node,
        payload.class_column,
        payload.description_column,
    )
