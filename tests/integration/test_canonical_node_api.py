"""End-to-end contract for server-ordered source-node resources."""

from pathlib import Path

from fastapi.testclient import TestClient

from ldaca_wordflow.main import create_app
from ldaca_wordflow.settings import Settings


def test_source_node_resource_and_one_based_rows(tmp_path: Path) -> None:
    settings = Settings(
        data_root=tmp_path,
        multi_user=False,
        session_cookie_secure=False,
        cors_allowed_origins=("http://testserver",),
        trusted_hosts=("testserver",),
    )
    app = create_app(settings, serve_frontend=False)
    with TestClient(app, base_url="http://testserver") as client:
        csrf = client.get("/api/session").json()["csrf_token"]
        unsafe = {
            "Origin": "http://testserver",
            "X-CSRF-Token": csrf,
        }
        uploaded = client.post(
            "/api/user-files/uploads",
            params={"path": "source.csv"},
            content=b"text,count\nhello,1\nworld,2\n",
            headers={**unsafe, "Content-Type": "application/octet-stream"},
        )
        assert uploaded.status_code == 201

        workspace = client.post(
            "/api/workspaces",
            json={"name": "Node test"},
            headers=unsafe,
        )
        assert workspace.status_code == 201
        workspace_id = workspace.json()["id"]
        assert (
            client.put(f"/api/workspaces/{workspace_id}/open", headers=unsafe).status_code
            == 200
        )

        created = client.post(
            f"/api/workspaces/{workspace_id}/nodes",
            json={"kind": "file", "file_path": "source.csv"},
            headers=unsafe,
        )
        assert created.status_code == 201
        node_id = created.json()["id"]
        assert created.headers["location"].endswith(f"/nodes/{node_id}")
        node_revision = created.headers["etag"]

        no_op = client.patch(
            f"/api/workspaces/{workspace_id}/nodes/{node_id}",
            json={"name": created.json()["name"]},
            headers=unsafe,
        )
        assert no_op.status_code == 200
        assert no_op.headers["etag"] == node_revision

        selected_document = client.patch(
            f"/api/workspaces/{workspace_id}/nodes/{node_id}",
            json={"document": "text"},
            headers=unsafe,
        )
        assert selected_document.status_code == 200
        selected_revision = selected_document.headers["etag"]
        assert selected_revision != node_revision

        cleared_document = client.patch(
            f"/api/workspaces/{workspace_id}/nodes/{node_id}",
            json={"document": None},
            headers=unsafe,
        )
        assert cleared_document.status_code == 200
        node_revision = cleared_document.headers["etag"]
        assert cleared_document.json()["document"] is None

        empty_patch = client.patch(
            f"/api/workspaces/{workspace_id}/nodes/{node_id}",
            json={},
            headers=unsafe,
        )
        assert empty_patch.status_code == 422

        rows = client.get(
            f"/api/workspaces/{workspace_id}/nodes/{node_id}/rows",
            params={"page": 1, "page_size": 1},
        )
        assert rows.status_code == 200
        assert rows.json()["total_rows"] == 2
        assert rows.json()["rows"] == [{"text": "hello", "count": 1}]
        assert rows.json()["total_pages"] == 2

        zero = client.get(
            f"/api/workspaces/{workspace_id}/nodes/{node_id}/rows",
            params={"page": 0},
        )
        assert zero.status_code == 422
        assert zero.json()["code"] == "request_validation_failed"

        deleted = client.delete(
            f"/api/workspaces/{workspace_id}/nodes/{node_id}",
            headers=unsafe,
        )
        assert deleted.status_code == 204
        assert deleted.content == b""


def test_derived_nodes_share_one_creation_contract_and_preview_is_read_only(
    tmp_path: Path,
) -> None:
    """Every transformation creates a child; preview never advances revision."""

    settings = Settings(
        data_root=tmp_path,
        multi_user=False,
        session_cookie_secure=False,
        cors_allowed_origins=("http://testserver",),
        trusted_hosts=("testserver",),
    )
    with TestClient(
        create_app(settings, serve_frontend=False),
        base_url="http://testserver",
    ) as client:
        csrf = client.get("/api/session").json()["csrf_token"]
        unsafe = {"Origin": "http://testserver", "X-CSRF-Token": csrf}
        assert (
            client.post(
                "/api/user-files/uploads",
                params={"path": "source.csv"},
                content=b"text,count\na,1\nb,2\nc,3\n",
                headers={**unsafe, "Content-Type": "application/octet-stream"},
            ).status_code
            == 201
        )
        workspace = client.post(
            "/api/workspaces",
            json={"name": "Derivations"},
            headers=unsafe,
        )
        workspace_id = workspace.json()["id"]
        assert (
            client.put(f"/api/workspaces/{workspace_id}/open", headers=unsafe).status_code
            == 200
        )
        source = client.post(
            f"/api/workspaces/{workspace_id}/nodes",
            json={"kind": "file", "file_path": "source.csv"},
            headers=unsafe,
        )
        source_id = source.json()["id"]
        assert source.json()["provenance"] == {"type": "source"}
        assert "operation" not in source.json()

        operation = {
            "kind": "filter",
            "source_node_id": source_id,
            "conditions": [{"column": "count", "operator": "gte", "value": 2}],
        }
        preview = client.post(
            f"/api/workspaces/{workspace_id}/nodes/previews",
            json=operation,
            params={"page": 1, "page_size": 10},
            headers=unsafe,
        )
        assert preview.status_code == 200
        assert preview.headers["etag"] == source.headers["etag"]
        assert preview.json()["rows"] == [
            {"text": "b", "count": 2},
            {"text": "c", "count": 3},
        ]

        derived = client.post(
            f"/api/workspaces/{workspace_id}/nodes",
            json=operation,
            headers=unsafe,
        )
        assert derived.status_code == 201
        assert derived.json()["parent_ids"] == [source_id]
        assert derived.json()["provenance"]["operation"]["kind"] == "filter"
        assert derived.json()["derivation_description"].startswith("filter of ")
        assert "operation" not in derived.json()
        assert derived.headers["etag"] != preview.headers["etag"]
        assert derived.headers["location"].endswith(f"/nodes/{derived.json()['id']}")

        next_command = client.post(
            f"/api/workspaces/{workspace_id}/nodes",
            json={"kind": "clone", "source_node_id": source_id},
            headers=unsafe,
        )
        assert next_command.status_code == 201
        assert next_command.headers["etag"] != derived.headers["etag"]
