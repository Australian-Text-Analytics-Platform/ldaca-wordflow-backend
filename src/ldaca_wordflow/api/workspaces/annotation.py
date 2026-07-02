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

from ...core.annotation_ai import (
    DEFAULT_BATCH_SIZE,
    AnnotationAiError,
    AnnotationClassOption,
    InferenceConfig,
    annotate_all,
    annotate_batch,
    list_models,
    resolve_provider_wire,
)
from ...core.annotation_preview_store import preview_store, signature_of
from ...core.auth import get_current_user
from ...core.exceptions import BadGatewayError, InvalidInputError, NotFoundError
from ...models import WorkspaceNodeInfo
from .schema_filter import frontend_node_info
from .utils import (
    Node,
    _create_and_persist_child_node,
    require_current_workspace,
    stage_dataframe_as_lazy,
    update_workspace,
)

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


class AnnotationCreateColumnRequest(BaseModel):
    """Request to add a new empty annotation column to a source node.

    Used by:
    - Frontend Annotation view when Start is pressed in "Start new annotation"
      mode because beginning a fresh pass must materialize the column that the
      results table then fills in, before the view switches into resume mode.
    """

    column_name: str


@router.post(
    "/annotation/source/{node_id}/annotation-column",
    response_model=WorkspaceNodeInfo,
)
async def create_annotation_column(
    node_id: str,
    payload: AnnotationCreateColumnRequest,
    current_user: dict = Depends(get_current_user),
):
    """Add an empty string annotation column to the selected source node.

    Used by:
    - Frontend Annotation view on Start (new-annotation mode) because the source
      block needs a real, persisted column before the per-row class dropdowns can
      record annotations and before the view can switch into resume mode.

    Flow:
    - Resolve the workspace and source node (404 if missing).
    - Reject blank names and names that already exist on the node (400).
    - Append a null-filled string column lazily, then restage the materialized
      frame so the node stays portable and lazy by default.
    - Persist the workspace and return the refreshed node metadata.
    """
    user_id = current_user["id"]
    workspace = require_current_workspace(user_id)
    node = workspace.nodes.get(node_id)
    if node is None:
        raise NotFoundError("Source node not found")
    column_name = payload.column_name.strip()
    if not column_name:
        raise InvalidInputError("Annotation column name is required")
    schema = dict(node.data.collect_schema().items())
    if column_name in schema:
        raise InvalidInputError(f"Column already exists: {column_name}")
    updated = node.data.with_columns(pl.lit(None).cast(pl.String).alias(column_name))
    materialized = cast(pl.DataFrame, updated.collect())
    node.data = stage_dataframe_as_lazy(
        materialized, workspace.ws_root_dir, node_name=node.name
    )
    update_workspace(user_id, workspace.id, workspace)
    return frontend_node_info(node)


class AnnotationSetCellRequest(BaseModel):
    """Request to set one annotation cell value on a source node.

    Used by:
    - Frontend Annotation results table when a reviewer picks a class in a row's
      dropdown, because the chosen label must be written into the annotation
      column cell instead of living only in transient component state.
    """

    column_name: str
    row_index: int
    value: str | None = None


@router.put(
    "/annotation/source/{node_id}/annotation-cell",
    response_model=WorkspaceNodeInfo,
)
async def set_annotation_cell(
    node_id: str,
    payload: AnnotationSetCellRequest,
    current_user: dict = Depends(get_current_user),
):
    """Write a single class label into the source node's annotation column.

    Used by:
    - Frontend Annotation results table because each per-row class dropdown
      persists its selection so annotations survive reloads and feed downstream
      analyses, rather than staying as local component state.

    Flow:
    - Resolve the workspace and source node (404 if missing).
    - Require a non-blank, existing string annotation column (400 otherwise).
    - Bounds-check the absolute row index against the collected frame (400).
    - Rewrite just the target cell (blank/whitespace clears it to null), restage
      the materialized frame, persist the workspace, and return node metadata.
    """
    user_id = current_user["id"]
    workspace = require_current_workspace(user_id)
    node = workspace.nodes.get(node_id)
    if node is None:
        raise NotFoundError("Source node not found")
    column_name = payload.column_name.strip()
    if not column_name:
        raise InvalidInputError("Annotation column name is required")
    existing = cast(pl.DataFrame, node.data.collect())
    schema = dict(existing.schema)
    if column_name not in schema:
        raise InvalidInputError(f"Column not found: {column_name}")
    if schema[column_name] != pl.String:
        raise InvalidInputError(f"Annotation column must be text: {column_name}")
    if payload.row_index < 0 or payload.row_index >= existing.height:
        raise InvalidInputError(f"Row index out of range: {payload.row_index}")
    cell_value = (payload.value or "").strip() or None
    updated = existing.with_columns(
        pl.when(pl.int_range(pl.len()) == payload.row_index)
        .then(pl.lit(cell_value, dtype=pl.String))
        .otherwise(pl.col(column_name))
        .alias(column_name)
    )
    node.data = stage_dataframe_as_lazy(
        updated, workspace.ws_root_dir, node_name=node.name
    )
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


def _annotation_classes_from_node(
    node: Node,
    class_column: str,
    description_column: str,
) -> list[AnnotationClassOption]:
    """Load a class-description node's classes as the AI engine's option list.

    Called by:
    - ``annotate_ai_preview`` and ``annotate_ai_all`` because the valid label set
      must be read server-side from the class node (authoritative) rather than
      trusted from the request. Reuses ``_class_description_payload_from_node`` for
      identical column validation/serialization, then drops blank and duplicate
      class names so the prompt lists each choice once — mirroring the frontend's
      former client-side class de-duplication.
    """
    payload = _class_description_payload_from_node(node, class_column, description_column)
    seen: set[str] = set()
    classes: list[AnnotationClassOption] = []
    for row in payload.rows:
        name = row.class_name.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        classes.append(AnnotationClassOption(name=name, description=row.description))
    return classes


class AnnotationAiModelsRequest(BaseModel):
    """Request to list a provider's available models for the AI model picker.

    Used by:
    - Frontend ModelNameCombobox because model listing now runs server-side (the
      browser no longer calls providers directly); it sends only the provider id,
      an optional custom base URL, and the API key needed to authenticate the
      listing call.
    """

    provider_id: str
    base_url: str | None = None
    api_key: str = ""


class AnnotationAiModelsResponse(BaseModel):
    """Sorted, de-duplicated model-id list returned to the model picker."""

    models: list[str] = Field(default_factory=list)


class AnnotationAiPreviewRequest(BaseModel):
    """Request to AI-classify one page of a source node's texts (no persistence).

    Used by:
    - Frontend AnnotationAiPreviewPanel per visible page because the preview shows
      the model's predictions without writing them, so the panel sends the same
      page/page_size slice it displays and gets back one label per row.
    """

    node_id: str
    text_column: str
    class_node_id: str
    class_column: str = "class"
    description_column: str = "description"
    provider_id: str
    base_url: str | None = None
    api_key: str = ""
    model: str
    instruction: str
    # Target column the preview run is annotating. Stored on the server-side
    # preview session as metadata so detach/annotate-all know where the reused
    # labels belong; it is not part of the cache signature (the model's label for a
    # given text does not depend on the write target).
    annotation_column: str = ""
    # Sampling/reasoning knobs from the "Model Configuration" section; defaults
    # reproduce the prior behaviour (deterministic sampling, no reasoning).
    temperature: float = 0.0
    reasoning_enabled: bool = False
    reasoning_effort: str = "medium"
    page: int = 1
    page_size: int = 20


class AnnotationAiPreviewResponse(BaseModel):
    """One predicted class (or null) per previewed row, aligned to page order."""

    labels: list[str | None] = Field(default_factory=list)


class AnnotationAiPreviewStateRequest(BaseModel):
    """Config identifying which cached preview session to hydrate.

    Used by:
    - Frontend AnnotationAiPreviewPanel on mount because the panel's per-row maps
      (AI labels + manual overrides) live only in memory and are lost when the tab
      unmounts; on remount it re-reads them from the server so the preview looks
      exactly as the user left it. Carries the same prediction-affecting config as
      the preview request so the server can confirm the cached labels still match
      the panel's current provider/model/prompt/classes/knobs before returning them.
    """

    node_id: str
    text_column: str
    class_node_id: str
    class_column: str = "class"
    description_column: str = "description"
    provider_id: str
    base_url: str | None = None
    model: str
    instruction: str
    temperature: float = 0.0
    reasoning_enabled: bool = False
    reasoning_effort: str = "medium"


class AnnotationAiPreviewRowState(BaseModel):
    """One hydrated row: the model label, the user override, and the effective one."""

    row_index: int
    ai: str | None = None
    override: str | None = None
    has_override: bool = False
    effective: str | None = None


class AnnotationAiPreviewStateResponse(BaseModel):
    """Every stored row for the requested session (empty when config changed)."""

    rows: list[AnnotationAiPreviewRowState] = Field(default_factory=list)


class AnnotationAiPreviewOverrideRequest(BaseModel):
    """One manual cell edit to persist onto the node's current preview session.

    Used by:
    - Frontend AnnotationAiPreviewPanel when the user changes a prediction in the
      dropdown, so the choice survives a tab switch and is honoured by
      detach/annotate-all. Only the row and its new label are needed — the edit
      targets the node's single active session, so no config has to travel with it.
      A blank/whitespace ``label`` means the user picked "None" and is stored as an
      explicit null override (which still wins over the model's label).
    """

    node_id: str
    row_index: int
    label: str | None = None


class AnnotationAiPreviewOverrideResponse(BaseModel):
    """Whether the edit was applied (False when no active session exists)."""

    ok: bool = True


class AnnotationAiPreviewClearRequest(BaseModel):
    """Request to drop a node's cached AI preview session.

    Used by:
    - Frontend AnnotationAiPreviewPanel's "Close preview" action. Closing the panel
      is an explicit "I'm done previewing" signal (unlike a tab switch, which only
      unmounts the panel and must keep the cache so it can rehydrate). Only the node
      is needed — the store is keyed per user+workspace+node, so the current
      workspace resolves the rest.
    """

    node_id: str


class AnnotationAiPreviewClearResponse(BaseModel):
    """Acknowledgement that the node's preview session was cleared."""

    ok: bool = True


class AnnotationAiAnnotateAllRequest(BaseModel):
    """Request to AI-classify every row and persist the whole annotation column.

    Used by:
    - Frontend AnnotationAiPreviewPanel's "Annotate All" button because a full run
      classifies the entire source node in concurrent batches and writes the
      results back in one go, unlike the transient per-page preview.
    """

    node_id: str
    text_column: str
    annotation_column: str
    class_node_id: str
    class_column: str = "class"
    description_column: str = "description"
    provider_id: str
    base_url: str | None = None
    api_key: str = ""
    model: str
    instruction: str
    # Sampling/reasoning knobs from the "Model Configuration" section; defaults
    # reproduce the prior behaviour (deterministic sampling, no reasoning).
    temperature: float = 0.0
    reasoning_enabled: bool = False
    reasoning_effort: str = "medium"
    batch_size: int | None = None


class AnnotationAiAnnotateAllResponse(BaseModel):
    """Refreshed node metadata plus how many of the total rows got a label."""

    node: WorkspaceNodeInfo
    labeled_rows: int
    total_rows: int


class AnnotationAiDetachRow(BaseModel):
    """One previewed row to detach: its absolute source-row index and label.

    Used by:
    - AnnotationAiDetachRequest as an optional client-supplied override list. The
      authoritative previewed rows now live in the server-side preview store, so
      the panel no longer needs to send them; this model is kept for backward
      compatibility and lets a caller force specific row/label pairs if provided.
    """

    row_index: int
    label: str | None = None


class AnnotationAiDetachRequest(BaseModel):
    """Request to copy the previewed rows into a new annotated child table.

    Used by:
    - Frontend AnnotationAiPreviewPanel's "Detach Previewed Rows" button. The rows
      the user previewed (across every page they viewed) are read from the
      server-side preview store, so the panel sends only the node + target column.
      ``rows`` is optional: when omitted the server materialises exactly the stored
      previewed rows (with their overrides applied); when supplied it takes
      precedence, preserving the old explicit-list behaviour.
    - The same panel's Detach button *gating*: with ``dry_run`` true the endpoint
      only probes the server session and returns how many rows would be detached
      (``node`` is null, nothing is created). The panel calls this on mount so the
      button reflects the cached preview after a tab switch — the local page map is
      reset on remount, but the server session is the source of truth — and so the
      confirmation dialog can show the exact count.
    """

    node_id: str
    annotation_column: str
    rows: list[AnnotationAiDetachRow] | None = None
    new_node_name: str | None = None
    dry_run: bool = False


class AnnotationAiDetachResponse(BaseModel):
    """Metadata for the newly created child node plus how many rows it holds.

    ``node`` is null for a ``dry_run`` probe (nothing is created); ``detached_rows``
    is then the number of rows that *would* be detached, used only to gate/label the
    frontend Detach button.
    """

    node: WorkspaceNodeInfo | None = None
    detached_rows: int


@router.post("/annotation/ai/models", response_model=AnnotationAiModelsResponse)
async def list_annotation_ai_models(
    payload: AnnotationAiModelsRequest,
    current_user: dict = Depends(get_current_user),
):
    """Proxy a provider's model listing so the browser never calls providers.

    Used by:
    - Frontend ModelNameCombobox to populate the model dropdown. Authentication is
      required (per-user gate) even though no workspace is touched, so an
      unauthenticated caller cannot use the backend as an open model-listing proxy.

    Flow:
    - Delegate to ``core.annotation_ai.list_models`` (native SDK per provider).
    - Translate any provider/SDK failure into a 502 with the provider's message.
    """
    try:
        models = await list_models(payload.provider_id, payload.base_url, payload.api_key)
    except AnnotationAiError as error:
        raise BadGatewayError(str(error)) from error
    return AnnotationAiModelsResponse(models=models)


@router.post("/annotation/ai/preview", response_model=AnnotationAiPreviewResponse)
async def annotate_ai_preview(
    payload: AnnotationAiPreviewRequest,
    current_user: dict = Depends(get_current_user),
):
    """AI-classify one page of texts, caching results in the preview store.

    Used by:
    - Frontend AnnotationAiPreviewPanel because previewing must not mutate the
      workspace; the panel renders the returned labels beside the (unchanged)
      source column so users can eyeball provider/model/prompt/class quality.

    Flow:
    - Resolve the workspace + source node (404) and require the text column (400).
    - Load the authoritative class list from the class-description node (404/400).
    - Slice exactly the requested page of the text column — the same
      ``(page-1)*page_size`` slice ``get_node_data`` uses — so labels line up with
      the rows the panel displays.
    - Sync the server-side preview session for this node + config signature, then
      dispatch a provider batch for ONLY the page rows not already cached (re-viewed
      pages are cache hits and cost nothing). Store the fresh labels and return the
      model labels for the whole page in order; provider errors become a 502.
    """
    user_id = current_user["id"]
    workspace = require_current_workspace(user_id)
    node = workspace.nodes.get(payload.node_id)
    if node is None:
        raise NotFoundError("Source node not found")
    schema = dict(node.data.collect_schema().items())
    if payload.text_column not in schema:
        raise InvalidInputError(f"Column not found: {payload.text_column}")
    class_node = workspace.nodes.get(payload.class_node_id)
    if class_node is None:
        raise NotFoundError("Class description node not found")
    classes = _annotation_classes_from_node(
        class_node, payload.class_column, payload.description_column
    )
    if not classes:
        raise InvalidInputError("No annotation classes defined")
    page = max(1, payload.page)
    page_size = max(1, payload.page_size)
    start_idx = (page - 1) * page_size
    page_df = cast(
        pl.DataFrame,
        node.data.slice(start_idx, page_size)
        .select(pl.col(payload.text_column).cast(pl.String).fill_null(""))
        .collect(),
    )
    texts = page_df[payload.text_column].to_list()
    wire = resolve_provider_wire(payload.provider_id, payload.base_url)
    config = InferenceConfig.from_request(
        temperature=payload.temperature,
        reasoning_enabled=payload.reasoning_enabled,
        reasoning_effort=payload.reasoning_effort,
    )
    # Key the cache by everything that changes a prediction. Syncing resets the
    # session if that signature changed since the last preview, so stale labels for
    # a different provider/model/prompt are never reused.
    signature = signature_of(
        text_column=payload.text_column,
        class_node_id=payload.class_node_id,
        class_column=payload.class_column,
        description_column=payload.description_column,
        provider_id=payload.provider_id,
        base_url=payload.base_url,
        model=payload.model,
        instruction=payload.instruction,
        temperature=config.temperature,
        reasoning_enabled=config.reasoning_enabled,
        reasoning_effort=config.reasoning_effort,
    )
    preview_store.sync(
        user_id,
        workspace.id,
        payload.node_id,
        signature=signature,
        annotation_column=payload.annotation_column,
    )
    page_indices = list(range(start_idx, start_idx + len(texts)))
    cached = preview_store.computed_indices(user_id, workspace.id, payload.node_id, page_indices)
    # Only the page rows we have not classified before hit the provider.
    missing = [(index, text) for index, text in zip(page_indices, texts) if index not in cached]
    if missing:
        try:
            fresh = await annotate_batch(
                wire,
                payload.model,
                payload.api_key,
                payload.instruction,
                classes,
                [text for _, text in missing],
                config,
            )
        except AnnotationAiError as error:
            raise BadGatewayError(str(error)) from error
        preview_store.put_ai_labels(
            user_id,
            workspace.id,
            payload.node_id,
            {index: label for (index, _), label in zip(missing, fresh)},
        )
    labels = preview_store.ai_labels_for_page(
        user_id, workspace.id, payload.node_id, page_indices
    )
    return AnnotationAiPreviewResponse(labels=labels)


@router.post(
    "/annotation/ai/preview/state",
    response_model=AnnotationAiPreviewStateResponse,
)
async def annotate_ai_preview_state(
    payload: AnnotationAiPreviewStateRequest,
    current_user: dict = Depends(get_current_user),
):
    """Return the cached preview rows for a node so the panel can rehydrate.

    Used by:
    - Frontend AnnotationAiPreviewPanel on mount because its per-row AI labels and
      manual overrides are in-memory only; after a tab switch the panel re-reads
      them here so the preview reappears exactly as the user left it.

    Flow:
    - Resolve the workspace + source node (404).
    - Recompute the config signature and ask the store for its rows; if the stored
      session belongs to a different config the store returns nothing, so the panel
      shows an empty preview and re-classifies on demand rather than displaying
      stale labels.
    """
    user_id = current_user["id"]
    workspace = require_current_workspace(user_id)
    node = workspace.nodes.get(payload.node_id)
    if node is None:
        raise NotFoundError("Source node not found")
    config = InferenceConfig.from_request(
        temperature=payload.temperature,
        reasoning_enabled=payload.reasoning_enabled,
        reasoning_effort=payload.reasoning_effort,
    )
    signature = signature_of(
        text_column=payload.text_column,
        class_node_id=payload.class_node_id,
        class_column=payload.class_column,
        description_column=payload.description_column,
        provider_id=payload.provider_id,
        base_url=payload.base_url,
        model=payload.model,
        instruction=payload.instruction,
        temperature=config.temperature,
        reasoning_enabled=config.reasoning_enabled,
        reasoning_effort=config.reasoning_effort,
    )
    rows = preview_store.state(user_id, workspace.id, payload.node_id, signature=signature)
    return AnnotationAiPreviewStateResponse(
        rows=[
            AnnotationAiPreviewRowState(
                row_index=row.row_index,
                ai=row.ai,
                override=row.override,
                has_override=row.has_override,
                effective=row.effective,
            )
            for row in rows
        ]
    )


@router.put(
    "/annotation/ai/preview/override",
    response_model=AnnotationAiPreviewOverrideResponse,
)
async def annotate_ai_preview_override(
    payload: AnnotationAiPreviewOverrideRequest,
    current_user: dict = Depends(get_current_user),
):
    """Persist one manual cell edit onto the node's active preview session.

    Used by:
    - Frontend AnnotationAiPreviewPanel when the user changes a prediction in the
      dropdown, so the edit survives a tab switch (it returns via ``preview/state``)
      and is honoured by detach/annotate-all. A blank label is stored as an explicit
      null override (the user chose "None"), which still wins over the model label.

    Flow:
    - Resolve the workspace + source node (404).
    - Write the override to the current session; ``ok=False`` when there is no active
      session (the edit is stale — an override can only follow a preview).
    """
    user_id = current_user["id"]
    workspace = require_current_workspace(user_id)
    node = workspace.nodes.get(payload.node_id)
    if node is None:
        raise NotFoundError("Source node not found")
    label = (payload.label or "").strip() or None
    applied = preview_store.set_override(
        user_id, workspace.id, payload.node_id, payload.row_index, label
    )
    return AnnotationAiPreviewOverrideResponse(ok=applied)


@router.post(
    "/annotation/ai/preview/clear",
    response_model=AnnotationAiPreviewClearResponse,
)
async def annotate_ai_preview_clear(
    payload: AnnotationAiPreviewClearRequest,
    current_user: dict = Depends(get_current_user),
):
    """Discard the node's cached AI preview session.

    Used by:
    - Frontend AnnotationAiPreviewPanel when the user clicks "Close preview". The
      in-memory preview store keeps a node's predictions across tab switches so the
      panel can rehydrate, but an explicit close means those predictions are no
      longer wanted; dropping the session frees the memory and resets the
      detach/annotate-all counts so a later preview starts clean.

    Flow:
    - Resolve the current workspace (the store is keyed per user+workspace+node).
    - Clear the node's session unconditionally. Clearing is idempotent, so a missing
      session (nothing previewed, or already cleared) is a successful no-op; the node
      itself is not required to still exist, which is why no 404 is raised here.
    """
    user_id = current_user["id"]
    workspace = require_current_workspace(user_id)
    preview_store.clear(user_id, workspace.id, payload.node_id)
    return AnnotationAiPreviewClearResponse(ok=True)


@router.post(
    "/annotation/ai/annotate-all",
    response_model=AnnotationAiAnnotateAllResponse,
)
async def annotate_ai_all(
    payload: AnnotationAiAnnotateAllRequest,
    current_user: dict = Depends(get_current_user),
):
    """AI-classify every row concurrently and write the whole column in one go.

    Used by:
    - Frontend AnnotationAiPreviewPanel's "Annotate All" button because filling a
      whole column one cell at a time is slow; this runs all batches concurrently
      on the server and persists the result as a single column rewrite.

    Flow:
    - Resolve the workspace + source node (404), collect it once, and require an
      existing *string* annotation column (400) so the write target is valid.
    - Load the authoritative class list from the class node (404/400).
    - Reuse any labels already cached in the preview store (rows the user previewed,
      with their manual overrides), and dispatch batches concurrently via
      ``annotate_all`` for ONLY the remaining rows — so a full run never re-spends on
      pages the user already previewed and respects edits made during preview.
    - Overwrite the annotation column with the merged labels (null where the model
      chose no class) and restage/persist the node; clear the cache since the
      predictions now live in the column. Provider errors become a 502 and nothing
      is written.
    """
    user_id = current_user["id"]
    workspace = require_current_workspace(user_id)
    node = workspace.nodes.get(payload.node_id)
    if node is None:
        raise NotFoundError("Source node not found")
    existing = cast(pl.DataFrame, node.data.collect())
    schema = dict(existing.schema)
    if payload.text_column not in schema:
        raise InvalidInputError(f"Column not found: {payload.text_column}")
    annotation_column = payload.annotation_column.strip()
    if not annotation_column:
        raise InvalidInputError("Annotation column name is required")
    if annotation_column not in schema:
        raise InvalidInputError(f"Column not found: {annotation_column}")
    if schema[annotation_column] != pl.String:
        raise InvalidInputError(f"Annotation column must be text: {annotation_column}")
    class_node = workspace.nodes.get(payload.class_node_id)
    if class_node is None:
        raise NotFoundError("Class description node not found")
    classes = _annotation_classes_from_node(
        class_node, payload.class_column, payload.description_column
    )
    if not classes:
        raise InvalidInputError("No annotation classes defined")
    texts = (
        existing.select(pl.col(payload.text_column).cast(pl.String).fill_null(""))
        .to_series()
        .to_list()
    )
    wire = resolve_provider_wire(payload.provider_id, payload.base_url)
    batch_size = payload.batch_size or DEFAULT_BATCH_SIZE
    config = InferenceConfig.from_request(
        temperature=payload.temperature,
        reasoning_enabled=payload.reasoning_enabled,
        reasoning_effort=payload.reasoning_effort,
    )
    # Reuse cached preview labels only when they belong to this exact config; a
    # signature mismatch means the panel previewed with different settings, so those
    # labels must not leak into a run the user configured differently.
    signature = signature_of(
        text_column=payload.text_column,
        class_node_id=payload.class_node_id,
        class_column=payload.class_column,
        description_column=payload.description_column,
        provider_id=payload.provider_id,
        base_url=payload.base_url,
        model=payload.model,
        instruction=payload.instruction,
        temperature=config.temperature,
        reasoning_enabled=config.reasoning_enabled,
        reasoning_effort=config.reasoning_effort,
    )
    cached = preview_store.effective_rows(
        user_id, workspace.id, payload.node_id, signature=signature
    )
    # Send only the rows we have no cached label for; assemble the full column by
    # index afterwards so cached and freshly-computed labels land in row order.
    missing = [(index, text) for index, text in enumerate(texts) if index not in cached]
    try:
        fresh = await annotate_all(
            wire,
            payload.model,
            payload.api_key,
            payload.instruction,
            classes,
            [text for _, text in missing],
            batch_size=batch_size,
            config=config,
        )
    except AnnotationAiError as error:
        raise BadGatewayError(str(error)) from error
    fresh_by_index = {index: label for (index, _), label in zip(missing, fresh)}
    labels = [
        cached[index] if index in cached else fresh_by_index.get(index)
        for index in range(len(texts))
    ]
    updated = existing.with_columns(
        pl.Series(annotation_column, labels, dtype=pl.String)
    )
    node.data = stage_dataframe_as_lazy(updated, workspace.ws_root_dir, node_name=node.name)
    update_workspace(user_id, workspace.id, workspace)
    # The predictions now live in the column, so the transient cache is redundant.
    preview_store.clear(user_id, workspace.id, payload.node_id)
    labeled_rows = sum(1 for label in labels if label)
    return AnnotationAiAnnotateAllResponse(
        node=WorkspaceNodeInfo.model_validate(frontend_node_info(node)),
        labeled_rows=labeled_rows,
        total_rows=existing.height,
    )


@router.post(
    "/annotation/ai/detach-previewed",
    response_model=AnnotationAiDetachResponse,
)
async def detach_ai_previewed_rows(
    payload: AnnotationAiDetachRequest,
    current_user: dict = Depends(get_current_user),
):
    """Copy the already-previewed rows into a new annotated child node.

    Used by:
    - Frontend AnnotationAiPreviewPanel's "Detach Previewed Rows" button because
      AI preview predictions live only in the server-side preview store; detaching
      turns every page the user has already assessed (not just the current one) into
      a persisted child table so those labels survive as a real data block without
      re-running (or persisting) the whole column on the source node.

    Flow:
    - Resolve the workspace + source node (404).
    - Build the row->label map from the node's active preview session (every
      previewed row across all viewed pages, override winning over the model label);
      fall back to any explicit ``rows`` the client sent.
    - When ``dry_run`` is set, return that map's size immediately (``node=None``,
      nothing created). The panel calls this on mount to gate/label its Detach
      button from the authoritative server session, so a tab switch (which wipes the
      panel's local page map) no longer leaves the button wrongly disabled.
    - Otherwise require a non-blank annotation column (400) and at least one row
      (400) — this is what fixes "detach only grabbed one page", since the source of
      truth is now the whole server session rather than the panel's current page.
    - Collect the source once; validate every ``row_index`` is in range (400).
    - Filter the source down to those indices (original order preserved), overwrite
      the annotation column with the aligned labels (blank/whitespace -> null), and
      stage the subset as a lazy frame.
    - Add it as a child of the source via ``_create_and_persist_child_node`` (so the
      lineage shows in the graph) and return the new node metadata + row count.
    """
    user_id = current_user["id"]
    workspace = require_current_workspace(user_id)
    node = workspace.nodes.get(payload.node_id)
    if node is None:
        raise NotFoundError("Source node not found")
    # Prefer an explicit client list when provided (back-compat / forced rows);
    # otherwise materialise the whole active preview session so every viewed page is
    # detached, not just the page currently on screen.
    index_to_label: dict[int, str | None] = {}
    if payload.rows:
        for row in payload.rows:
            index_to_label[row.row_index] = (row.label or "").strip() or None
    else:
        index_to_label = dict(
            preview_store.effective_rows(user_id, workspace.id, payload.node_id)
        )
    # Dry-run probe: report how many rows would detach without materialising a child.
    # Returns 0 (not a 400) for an empty session so the panel can simply disable its
    # button; this is the source of truth the panel gates on across tab switches.
    if payload.dry_run:
        return AnnotationAiDetachResponse(node=None, detached_rows=len(index_to_label))
    annotation_column = payload.annotation_column.strip()
    if not annotation_column:
        raise InvalidInputError("Annotation column name is required")
    existing = cast(pl.DataFrame, node.data.collect())
    total_rows = existing.height
    if not index_to_label:
        raise InvalidInputError("No previewed rows to detach")
    for index in index_to_label:
        if index < 0 or index >= total_rows:
            raise InvalidInputError(f"Row index out of range: {index}")
    sorted_indices = sorted(index_to_label)
    labels_in_order = [index_to_label[index] for index in sorted_indices]
    # `filter` preserves the source's ascending order, so labels sorted by index
    # line up with the selected rows without a join.
    selected = (
        existing.with_row_index("__detach_row_index__")
        .filter(pl.col("__detach_row_index__").is_in(sorted_indices))
        .drop("__detach_row_index__")
        .with_columns(pl.Series(annotation_column, labels_in_order, dtype=pl.String))
    )
    new_node_name = (payload.new_node_name or "").strip() or f"{node.name}_previewed"
    lazy_data = stage_dataframe_as_lazy(
        selected, workspace.ws_root_dir, node_name=new_node_name
    )
    new_node = _create_and_persist_child_node(
        workspace=workspace,
        data=lazy_data,
        name=new_node_name,
        operation=f"annotation_detach_previewed({node.name})",
        parents=[node],
        user_id=user_id,
        workspace_id=workspace.id,
    )
    return AnnotationAiDetachResponse(
        node=WorkspaceNodeInfo.model_validate(frontend_node_info(new_node)),
        detached_rows=selected.height,
    )
