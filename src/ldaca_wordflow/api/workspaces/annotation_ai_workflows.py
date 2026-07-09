"""Annotation AI preview and full-column workflows.

Used by:
- ``annotation.annotate_ai_preview`` and ``annotation.annotate_ai_all`` because
  those routes should only resolve HTTP identity and delegate provider dispatch,
  preview-cache coordination, and workspace mutation here.

Flow:
- Resolve source/class nodes and validate selected columns.
- Convert class-description rows into authoritative model label options.
- Run provider batches for missing labels while reusing preview-session cache.
- For full annotation, rewrite the annotation column, persist the workspace, and
  clear transient preview state.
"""

from __future__ import annotations

from typing import Any, cast

import polars as pl

from ...core.annotation_ai import (
    DEFAULT_BATCH_SIZE,
    AnnotationAiError,
    AnnotationClassOption,
    InferenceConfig,
    annotate_all,
    annotate_batch,
    resolve_provider_wire,
)
from ...core.annotation_preview_store import preview_store, signature_of
from ...core.exceptions import BadGatewayError, InvalidInputError, NotFoundError
from ...models.workspace import WorkspaceNodeInfo
from .schema_filter import frontend_node_info
from .utils import (
    Node,
    _create_and_persist_child_node,
    stage_dataframe_as_lazy,
    update_workspace,
)


def _validate_class_description_columns(
    schema: dict[str, pl.DataType],
    class_column: str,
    description_column: str,
) -> None:
    """Validate class-description columns before prompt option loading."""

    if class_column == description_column:
        raise InvalidInputError("Class and description columns must be different")
    missing = [
        column
        for column in (class_column, description_column)
        if column not in schema
    ]
    if missing:
        raise InvalidInputError(f"Column not found: {', '.join(missing)}")


def _annotation_classes_from_node(
    node: Node,
    class_column: str,
    description_column: str,
) -> list[AnnotationClassOption]:
    """Load unique non-blank class labels from the class-description node.

    Used by:
    - annotation AI preview/all workflows because valid labels must come from the
      workspace node, not from client-submitted prompt text.
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
    seen: set[str] = set()
    classes: list[AnnotationClassOption] = []
    for row in df.to_dicts():
        name = str(row.get("class") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        classes.append(
            AnnotationClassOption(
                name=name,
                description=str(row.get("description") or ""),
            )
        )
    return classes


def _annotation_signature(payload: Any, config: InferenceConfig) -> str:
    """Return the preview-cache signature for one request-shaped payload."""

    return signature_of(
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


async def preview_annotation_ai_page(
    *,
    user_id: str,
    workspace_id: str,
    workspace: Any,
    payload: Any,
) -> dict[str, Any]:
    """Classify one page and update the node's transient preview session.

    Used by:
    - ``annotation.annotate_ai_preview`` after the route resolves the workspace.

    Flow:
    - Validate source/class nodes and selected columns.
    - Slice the requested page and compute only labels missing from the cache.
    - Store fresh labels and return the page labels in row order.
    """

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
    signature = _annotation_signature(payload, config)
    preview_store.sync(
        user_id,
        workspace_id,
        payload.node_id,
        signature=signature,
        annotation_column=payload.annotation_column,
    )
    page_indices = list(range(start_idx, start_idx + len(texts)))
    cached = preview_store.computed_indices(
        user_id, workspace_id, payload.node_id, page_indices
    )
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
            workspace_id,
            payload.node_id,
            {index: label for (index, _), label in zip(missing, fresh)},
        )
    labels = preview_store.ai_labels_for_page(
        user_id, workspace_id, payload.node_id, page_indices
    )
    return {"session_id": payload.node_id, "labels": labels}


async def annotate_all_annotation_ai_rows(
    *,
    user_id: str,
    workspace_id: str,
    workspace: Any,
    node_id: str,
    payload: Any,
) -> dict[str, Any]:
    """Classify every row and persist the resulting annotation column.

    Used by:
    - ``annotation.annotate_ai_all`` after the route resolves the workspace.

    Flow:
    - Validate the source node, text column, existing string annotation column,
      and class-description node.
    - Reuse cached preview labels for the same config and provider-batch only
      missing rows.
    - Stage the updated dataframe, persist the workspace, clear transient cache,
      and return refreshed node metadata.
    """

    node = workspace.nodes.get(node_id)
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
    signature = _annotation_signature(payload, config)
    cached = preview_store.effective_rows(
        user_id, workspace_id, node_id, signature=signature
    )
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
    update_workspace(user_id, workspace_id, workspace)
    preview_store.clear(user_id, workspace_id, node_id)
    labeled_rows = sum(1 for label in labels if label)
    return {
        "node": WorkspaceNodeInfo.model_validate(frontend_node_info(node)),
        "labeled_rows": labeled_rows,
        "total_rows": existing.height,
    }


def detach_previewed_annotation_ai_rows(
    *,
    user_id: str,
    workspace_id: str,
    workspace: Any,
    node_id: str,
    payload: Any,
) -> dict[str, Any]:
    """Materialize the active preview session as a child node.

    Used by:
    - ``annotation.detach_ai_previewed_rows`` because detaching should read the
      server-side preview session, validate row indices, and persist the child
      node as one workflow.
    """

    node = workspace.nodes.get(node_id)
    if node is None:
        raise NotFoundError("Source node not found")

    index_to_label = dict(preview_store.effective_rows(user_id, workspace_id, node_id))
    if payload.dry_run:
        return {"node": None, "detached_rows": len(index_to_label)}

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
        workspace_id=workspace_id,
    )
    return {
        "node": WorkspaceNodeInfo.model_validate(frontend_node_info(new_node)),
        "detached_rows": selected.height,
    }
