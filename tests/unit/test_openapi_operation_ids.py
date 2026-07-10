from fastapi.routing import APIRoute
from ldaca_wordflow.main import app


def _iter_route_names(routes, prefix: str = ""):
    """Yield API routes with fully expanded include prefixes.

    FastAPI 0.138 emits wrapped included routes (`_IncludedRouter`), so this
    helper recursively normalizes route paths before building path->name lookups.
    """
    for route in routes:
        if isinstance(route, APIRoute):
            yield (f"{prefix}{route.path}", route)
        elif route.__class__.__name__ == "_IncludedRouter":
            include_context = route.include_context
            nested_prefix = f"{prefix}{include_context.prefix}"
            for nested_route in include_context.included_router.routes:
                yield from _iter_route_names(
                    [nested_route],  # avoid generic traversal helpers for tiny path
                    nested_prefix,
                )


def test_openapi_operation_ids_use_route_names() -> None:
    schema = app.openapi()
    route_by_path_method = {
        (path, next(iter(route.methods)).lower()): route.name
        for path, route in _iter_route_names(app.router.routes)
        if isinstance(route, APIRoute) and route.include_in_schema and route.methods
    }

    assert (
        schema["paths"]["/api/files/"]["get"]["operationId"]
        == route_by_path_method[("/api/files/", "get")]
    )
    assert (
        schema["paths"]["/api/workspaces/{workspace_id}/nodes/{node_id}/data"]["get"][
            "operationId"
        ]
        == route_by_path_method[
            ("/api/workspaces/{workspace_id:uuid}/nodes/{node_id}/data", "get")
        ]
    )


def test_openapi_route_names_are_unique() -> None:
    route_names = [
        route.name
        for route in app.routes
        if isinstance(route, APIRoute) and route.include_in_schema
    ]

    assert len(route_names) == len(set(route_names))


def test_openapi_uses_single_child_router_tag_per_operation() -> None:
    """Main router includes should not duplicate child router tags in OpenAPI."""
    schema = app.openapi()
    duplicate_tags = []
    for path, path_item in schema["paths"].items():
        for method, operation in path_item.items():
            if not isinstance(operation, dict):
                continue
            tags = operation.get("tags") or []
            if len(tags) != len(set(tags)):
                duplicate_tags.append((method.upper(), path, tags))

    assert duplicate_tags == []


def test_openapi_excludes_replaced_workspace_lifecycle_routes() -> None:
    """Core workspace routes should expose explicit resources, not legacy aliases."""
    paths = app.openapi()["paths"]

    assert "/api/users/me/current-workspace" in paths
    assert "/api/workspaces/{workspace_id}" in paths
    assert "/api/workspaces/{workspace_id}/graph" in paths
    assert "/api/workspaces/{workspace_id}/nodes/order" in paths
    assert "/api/workspaces/{workspace_id}/download" in paths
    assert "/api/workspaces/{workspace_id}/download/tasks/{task_id}/artifact" in paths
    assert "/api/workspaces/{workspace_id}/unload" in paths

    for legacy_path in [
        "/api/workspaces/current",
        "/api/workspaces/delete",
        "/api/workspaces/name",
        "/api/workspaces/description",
        "/api/workspaces/save",
        "/api/workspaces/info",
        "/api/workspaces/graph",
        "/api/workspaces/nodes/order",
        "/api/workspaces/download",
        "/api/workspaces/download/tasks/{task_id}/artifact",
        "/api/workspaces/unload",
    ]:
        assert legacy_path not in paths


def test_openapi_excludes_current_workspace_node_and_analysis_routes() -> None:
    """Workspace-scoped routes should carry workspace_id in the path."""
    paths = app.openapi()["paths"]

    for explicit_path in [
        "/api/workspaces/{workspace_id}/nodes",
        "/api/workspaces/{workspace_id}/nodes/{node_id}/data",
        "/api/workspaces/{workspace_id}/nodes/{node_id}/filter",
        "/api/workspaces/{workspace_id}/nodes/{node_id}/filter/preview",
        "/api/workspaces/{workspace_id}/annotation/class-descriptions",
        "/api/workspaces/{workspace_id}/annotation-ai-preview-sessions",
        "/api/workspaces/{workspace_id}/nodes/{node_id}/sequential-analysis",
        "/api/workspaces/{workspace_id}/topic-modeling",
        "/api/workspaces/{workspace_id}/export",
    ]:
        assert explicit_path in paths

    for legacy_path in [
        "/api/workspaces/export",
        "/api/workspaces/nodes",
        "/api/workspaces/nodes/concat",
        "/api/workspaces/nodes/concat/preview",
        "/api/workspaces/nodes/join",
        "/api/workspaces/nodes/join/preview",
        "/api/workspaces/nodes/{node_id}",
        "/api/workspaces/nodes/{node_id}/cast",
        "/api/workspaces/nodes/{node_id}/clone",
        "/api/workspaces/nodes/{node_id}/color",
        "/api/workspaces/nodes/{node_id}/columns/{column_name}",
        "/api/workspaces/nodes/{node_id}/columns/{column_name}/describe",
        "/api/workspaces/nodes/{node_id}/columns/{column_name}/operations",
        "/api/workspaces/nodes/{node_id}/columns/{column_name}/unique",
        "/api/workspaces/nodes/{node_id}/data",
        "/api/workspaces/nodes/{node_id}/document-column",
        "/api/workspaces/nodes/{node_id}/expression/apply",
        "/api/workspaces/nodes/{node_id}/expression/preview",
        "/api/workspaces/nodes/{node_id}/filter",
        "/api/workspaces/nodes/{node_id}/filter/preview",
        "/api/workspaces/nodes/{node_id}/name",
        "/api/workspaces/nodes/{node_id}/query-plan",
        "/api/workspaces/nodes/{node_id}/redo",
        "/api/workspaces/nodes/{node_id}/replace",
        "/api/workspaces/nodes/{node_id}/replace/preview",
        "/api/workspaces/nodes/{node_id}/shape",
        "/api/workspaces/nodes/{node_id}/slice",
        "/api/workspaces/nodes/{node_id}/slice/preview",
        "/api/workspaces/nodes/{node_id}/tokenization-preference",
        "/api/workspaces/nodes/{node_id}/undo",
        "/api/workspaces/annotation/ai/annotate-all",
        "/api/workspaces/annotation/ai/detach-previewed",
        "/api/workspaces/annotation/ai/models",
        "/api/workspaces/annotation/ai/preview",
        "/api/workspaces/annotation/ai/preview/clear",
        "/api/workspaces/annotation/ai/preview/override",
        "/api/workspaces/annotation/ai/preview/state",
        "/api/workspaces/annotation/class-descriptions",
        "/api/workspaces/annotation/class-descriptions/{node_id}",
        "/api/workspaces/annotation/class-descriptions/{node_id}/parent",
        "/api/workspaces/annotation/source/{node_id}/annotation-cell",
        "/api/workspaces/annotation/source/{node_id}/annotation-column",
        "/api/workspaces/concordance",
        "/api/workspaces/concordance/tasks/{task_id}/bins",
        "/api/workspaces/concordance/tasks/{task_id}/request",
        "/api/workspaces/concordance/tasks/{task_id}/result",
        "/api/workspaces/nodes/{node_id}/concordance/detach",
        "/api/workspaces/nodes/{node_id}/concordance/detach-options",
        "/api/workspaces/nodes/{node_id}/concordance/dispersion-detach",
        "/api/workspaces/nodes/{node_id}/concordance/materialize",
        "/api/workspaces/nodes/{node_id}/quotation",
        "/api/workspaces/nodes/{node_id}/quotation/detach",
        "/api/workspaces/nodes/{node_id}/quotation/detach-options",
        "/api/workspaces/nodes/{node_id}/quotation/materialize",
        "/api/workspaces/nodes/{node_id}/sequential-analysis",
        "/api/workspaces/nodes/{node_id}/sequential-analysis/preview",
        "/api/workspaces/quotation/tasks/{task_id}/request",
        "/api/workspaces/quotation/tasks/{task_id}/result",
        "/api/workspaces/sequential-analysis/tasks/{task_id}/detach",
        "/api/workspaces/sequential-analysis/tasks/{task_id}/request",
        "/api/workspaces/sequential-analysis/tasks/{task_id}/result",
        "/api/workspaces/token-frequencies",
        "/api/workspaces/token-frequencies/tasks/{task_id}/request",
        "/api/workspaces/token-frequencies/tasks/{task_id}/result",
        "/api/workspaces/topic-modeling",
        "/api/workspaces/topic-modeling/tasks/{task_id}/detach",
        "/api/workspaces/topic-modeling/tasks/{task_id}/detach-options",
        "/api/workspaces/topic-modeling/tasks/{task_id}/request",
        "/api/workspaces/topic-modeling/tasks/{task_id}/result",
    ]:
        assert legacy_path not in paths


def test_openapi_uses_annotation_ai_preview_session_resources() -> None:
    """AI preview lifecycle exposes one concrete, generation-safe SDK contract."""
    spec = app.openapi()
    paths = spec["paths"]
    schemas = spec["components"]["schemas"]

    assert (
        "post" in paths["/api/workspaces/{workspace_id}/annotation-ai-preview-sessions"]
    )
    assert (
        "get"
        in paths[
            "/api/workspaces/{workspace_id}/annotation-ai-preview-sessions/{node_id}"
        ]
    )
    assert (
        "delete"
        in paths[
            "/api/workspaces/{workspace_id}/annotation-ai-preview-sessions/{node_id}"
        ]
    )
    assert (
        "patch"
        in paths[
            "/api/workspaces/{workspace_id}/annotation-ai-preview-sessions/{node_id}/rows/{row_index}"
        ]
    )
    assert (
        "post"
        in paths[
            "/api/workspaces/{workspace_id}/annotation-ai-preview-sessions/{node_id}/annotations"
        ]
    )
    assert (
        "post"
        in paths[
            "/api/workspaces/{workspace_id}/annotation-ai-preview-sessions/{node_id}/detachments"
        ]
    )

    preview_response = schemas["AnnotationAiPreviewResponse"]
    assert "session_id" in preview_response["required"]
    state_response = schemas["AnnotationAiPreviewStateResponse"]
    assert {"session_id", "annotation_column", "rows"} <= set(
        state_response["properties"]
    )
    assert {"session_id", "annotation_column"} <= set(state_response["required"])

    session_path = paths[
        "/api/workspaces/{workspace_id}/annotation-ai-preview-sessions/{node_id}"
    ]
    get_query = {
        parameter["name"]: parameter
        for parameter in session_path["get"]["parameters"]
        if parameter["in"] == "query"
    }
    assert get_query["annotation_column"]["required"] is True
    delete_query = {
        parameter["name"]: parameter
        for parameter in session_path["delete"]["parameters"]
        if parameter["in"] == "query"
    }
    assert delete_query["session_id"]["required"] is True

    for schema_name in [
        "AnnotationAiPreviewOverrideRequest",
        "AnnotationAiAnnotateAllRequest",
        "AnnotationAiDetachRequest",
    ]:
        assert "session_id" in schemas[schema_name]["required"]

    for legacy_path in [
        "/api/workspaces/{workspace_id}/annotation/ai/preview",
        "/api/workspaces/{workspace_id}/annotation/ai/preview/state",
        "/api/workspaces/{workspace_id}/annotation/ai/preview/override",
        "/api/workspaces/{workspace_id}/annotation/ai/preview/clear",
        "/api/workspaces/{workspace_id}/annotation/ai/annotate-all",
        "/api/workspaces/{workspace_id}/annotation/ai/detach-previewed",
    ]:
        assert legacy_path not in paths


def test_openapi_uses_body_node_creation_and_resource_task_actions() -> None:
    """Mutating node/task operations should not encode state changes as queries."""

    paths = app.openapi()["paths"]

    create_node = paths["/api/workspaces/{workspace_id}/nodes"]["post"]
    assert "requestBody" in create_node
    create_node_query_params = {
        param["name"]
        for param in create_node.get("parameters", [])
        if param.get("in") == "query"
    }
    assert not {"filename", "sheet_name", "mode"} & create_node_query_params

    assert "delete" in paths["/api/tasks/{task_id}"]
    assert "post" in paths["/api/tasks/{task_id}/cancel"]
    assert "/api/tasks/clear" not in paths
    assert "/api/tasks/cancel" not in paths


def test_openapi_uses_batch_get_for_workspace_node_info() -> None:
    """Body-based node-info reads should use an explicit batch-get resource path."""

    paths = app.openapi()["paths"]

    batch_get = paths["/api/workspaces/{workspace_id}/nodes:batchGet"]["post"]
    assert "requestBody" in batch_get
    assert "put" not in paths["/api/workspaces/{workspace_id}/nodes"]


def test_openapi_uses_shared_analysis_task_read_routes() -> None:
    """Analysis task request/result reads should use the shared task resource."""

    paths = app.openapi()["paths"]

    assert "/api/workspaces/{workspace_id}/analysis-tasks/{task_id}/request" in paths
    assert "/api/workspaces/{workspace_id}/analysis-tasks/{task_id}/result" in paths
    assert (
        "/api/workspaces/{workspace_id}/analysis-tasks/{task_id}/result-query" in paths
    )
    assert (
        "/api/workspaces/{workspace_id}/analysis-tasks/{task_id}/preferences" in paths
    )
    assert "delete" not in paths["/api/workspaces/{workspace_id}/token-frequencies"]

    legacy_read_paths = [
        "/api/workspaces/{workspace_id}/concordance/tasks/{task_id}/request",
        "/api/workspaces/{workspace_id}/concordance/tasks/{task_id}/result",
        "/api/workspaces/{workspace_id}/quotation/tasks/{task_id}/request",
        "/api/workspaces/{workspace_id}/quotation/tasks/{task_id}/result",
        "/api/workspaces/{workspace_id}/sequential-analysis/tasks/{task_id}/request",
        "/api/workspaces/{workspace_id}/sequential-analysis/tasks/{task_id}/result",
        "/api/workspaces/{workspace_id}/token-frequencies/tasks/{task_id}/request",
        "/api/workspaces/{workspace_id}/token-frequencies/tasks/{task_id}/result",
        "/api/workspaces/{workspace_id}/topic-modeling/tasks/{task_id}/request",
        "/api/workspaces/{workspace_id}/topic-modeling/tasks/{task_id}/result",
    ]
    for legacy_path in legacy_read_paths:
        assert "get" not in paths.get(legacy_path, {})

    legacy_result_posts = [
        "/api/workspaces/{workspace_id}/concordance/tasks/{task_id}/result",
        "/api/workspaces/{workspace_id}/quotation/tasks/{task_id}/result",
        "/api/workspaces/{workspace_id}/sequential-analysis/tasks/{task_id}/result",
        "/api/workspaces/{workspace_id}/token-frequencies/tasks/{task_id}/result",
    ]
    for legacy_path in legacy_result_posts:
        assert "post" not in paths.get(legacy_path, {})


def test_openapi_uses_shared_analysis_task_detachment_routes() -> None:
    """Task detach/materialize operations should live under the shared task resource."""

    paths = app.openapi()["paths"]

    assert (
        "/api/workspaces/{workspace_id}/analysis-tasks/{task_id}/detach-options"
        in paths
    )
    assert (
        "/api/workspaces/{workspace_id}/analysis-tasks/{task_id}/detachments" in paths
    )
    assert (
        "/api/workspaces/{workspace_id}/analysis-tasks/{task_id}/dispersion-bins"
        in paths
    )
    assert (
        "/api/workspaces/{workspace_id}/analysis-tasks/{task_id}/dispersion-detachments"
        in paths
    )
    assert (
        "/api/workspaces/{workspace_id}/analysis-tasks/{task_id}/materializations"
        in paths
    )

    legacy_detach_paths = {
        "/api/workspaces/{workspace_id}/concordance/tasks/{task_id}/bins": "get",
        "/api/workspaces/{workspace_id}/nodes/{node_id}/concordance/detach": "post",
        "/api/workspaces/{workspace_id}/nodes/{node_id}/concordance/detach-options": "get",
        "/api/workspaces/{workspace_id}/nodes/{node_id}/concordance/dispersion-detach": "post",
        "/api/workspaces/{workspace_id}/nodes/{node_id}/concordance/materialize": "post",
        "/api/workspaces/{workspace_id}/nodes/{node_id}/quotation/detach": "post",
        "/api/workspaces/{workspace_id}/nodes/{node_id}/quotation/detach-options": "get",
        "/api/workspaces/{workspace_id}/nodes/{node_id}/quotation/materialize": "post",
        "/api/workspaces/{workspace_id}/topic-modeling/tasks/{task_id}/detach-options": "get",
        "/api/workspaces/{workspace_id}/topic-modeling/tasks/{task_id}/detach": "post",
        "/api/workspaces/{workspace_id}/sequential-analysis/tasks/{task_id}/detach": "post",
    }
    for legacy_path, method in legacy_detach_paths.items():
        assert method not in paths.get(legacy_path, {})


def test_openapi_documents_common_app_error_response_model() -> None:
    """App-level domain errors should point generated clients at ErrorResponse."""

    operation = app.openapi()["paths"]["/api/workspaces/{workspace_id}"]["get"]
    not_found = operation["responses"]["404"]
    assert (
        not_found["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/ErrorResponse"
    )
