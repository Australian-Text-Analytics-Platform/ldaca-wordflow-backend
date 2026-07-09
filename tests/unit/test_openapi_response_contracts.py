"""OpenAPI response contract coverage for public, file, and streaming routes.

Used by:
- backend endpoint-design cleanup because generated clients and API docs need
  concrete JSON schemas or explicit non-JSON media metadata for these routes.
"""

from __future__ import annotations

from ldaca_wordflow.main import app


def _json_schema_ref(path: str, method: str) -> str | None:
    operation = app.openapi()["paths"][path][method]
    response = operation["responses"]["200"]
    return response["content"]["application/json"]["schema"].get("$ref")


def test_openapi_documents_json_response_models_for_public_routes() -> None:
    """JSON public/admin routes should expose concrete response schemas."""

    expected_refs = {
        ("/api", "get"): "#/components/schemas/RootResponse",
        ("/api/auth/logout", "post"): "#/components/schemas/MessageResponse",
        ("/api/auth/status", "get"): "#/components/schemas/AuthStatusResponse",
        ("/api/auth/health", "get"): "#/components/schemas/AuthHealthResponse",
        ("/api/admin/users", "get"): "#/components/schemas/AdminUsersResponse",
        ("/api/admin/cleanup", "get"): "#/components/schemas/AdminCleanupResponse",
    }

    for (path, method), expected_ref in expected_refs.items():
        assert _json_schema_ref(path, method) == expected_ref


def test_openapi_documents_redirect_response_status_codes() -> None:
    """OAuth redirect routes should document their redirect status codes."""

    schema = app.openapi()
    expected_status_codes = {
        ("/api/auth/google/callback", "post"): "303",
        ("/api/auth/cilogon/login", "get"): "302",
        ("/api/auth/cilogon/callback", "get"): "303",
    }

    for (path, method), status_code in expected_status_codes.items():
        assert status_code in schema["paths"][path][method]["responses"]


def test_openapi_documents_non_json_file_and_stream_media_types() -> None:
    """File, raw-text, and SSE routes should advertise their response media."""

    schema = app.openapi()
    expected_media_types = {
        ("/api/files/sample-data/readme", "get"): {"text/plain"},
        ("/api/files/raw", "get"): {"text/plain", "text/markdown"},
        ("/api/files/content", "get"): {"application/octet-stream"},
        ("/api/tasks/stream", "get"): {"text/event-stream"},
        (
            "/api/workspaces/{workspace_id}/download/tasks/{task_id}/artifact",
            "get",
        ): {"application/zip"},
        (
            "/api/workspaces/{workspace_id}/export",
            "get",
        ): {
            "application/json",
            "application/octet-stream",
            "application/vnd.apache.arrow.file",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/x-ndjson",
            "application/zip",
            "text/csv",
        },
    }

    for (path, method), media_types in expected_media_types.items():
        response = schema["paths"][path][method]["responses"]["200"]
        documented_media_types = set(response.get("content", {}))
        assert media_types <= documented_media_types


def test_openapi_uses_query_path_file_resources_instead_of_catch_all_routes() -> None:
    """File path operations should use one query-path contract."""

    paths = app.openapi()["paths"]

    assert "/api/files/content" in paths
    assert "/api/files/info" in paths
    assert "delete" in paths["/api/files/"]
    assert "/api/files/{filename}" not in paths
    assert "/api/files/{filename}/info" not in paths

    for path, method in [
        ("/api/files/content", "get"),
        ("/api/files/info", "get"),
        ("/api/files/", "delete"),
    ]:
        query_params = {
            param["name"]
            for param in paths[path][method].get("parameters", [])
            if param.get("in") == "query"
        }
        assert "path" in query_params
