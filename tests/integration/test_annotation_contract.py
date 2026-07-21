"""Canonical stateless-preview and Workspace-owned Annotation contracts."""

from __future__ import annotations

import time
from functools import partial
from io import BytesIO
from pathlib import Path

import polars as pl
from anyio.to_thread import run_sync as run_sync_in_worker_thread
from fastapi.testclient import TestClient

from ldaca_wordflow.main import create_app
from ldaca_wordflow.services import annotations as annotation_service_module
from ldaca_wordflow.services import analysis_executor as analysis_executor_module
from ldaca_wordflow.workers import annotation as annotation_worker_module
from ldaca_wordflow.settings import Settings


def _client(tmp_path: Path) -> TestClient:
    return TestClient(
        create_app(
            Settings(
                data_root=tmp_path,
                multi_user=False,
                session_cookie_secure=False,
                cors_allowed_origins=("http://testserver",),
                trusted_hosts=("testserver",),
            ),
            serve_frontend=False,
        ),
        base_url="http://testserver",
    )


def _source(client: TestClient, unsafe: dict[str, str]) -> tuple[str, str, str]:
    uploaded = client.post(
        "/api/user-files/uploads",
        params={"path": "documents.csv"},
        content=b"text\nfirst document\nsecond document\n",
        headers={**unsafe, "Content-Type": "application/octet-stream"},
    )
    assert uploaded.status_code == 201
    workspace = client.post(
        "/api/workspaces",
        json={"name": "Annotations"},
        headers=unsafe,
    )
    workspace_id = workspace.json()["id"]
    assert (
        client.put(f"/api/workspaces/{workspace_id}/open", headers=unsafe).status_code
        == 200
    )
    node = client.post(
        f"/api/workspaces/{workspace_id}/nodes",
        json={"kind": "file", "file_path": "documents.csv"},
        headers=unsafe,
    )
    assert node.status_code == 201
    return workspace_id, node.json()["id"], node.headers["etag"]


def _request() -> dict[str, object]:
    return {
        "text_column": "text",
        "annotation_column": "stance",
        "classes": [
            {"name": "support", "description": "supports the claim"},
            {"name": "critical", "description": "criticises the claim"},
        ],
        "provider": "openai",
        "model": "test-model",
        "instruction": "Classify each document.",
    }


def _configure_credentials(client: TestClient, unsafe: dict[str, str]) -> None:
    response = client.patch(
        "/api/provider-credentials",
        json={"openai_api_key": "provider-secret"},
        headers=unsafe,
    )
    assert response.status_code == 200, response.text


def _wait_analysis(
    client: TestClient,
    workspace_id: str,
    analysis_id: str,
) -> dict[str, object]:
    deadline = time.monotonic() + 10
    while True:
        response = client.get(
            f"/api/workspaces/{workspace_id}/analyses/{analysis_id}"
        )
        assert response.status_code == 200
        payload = response.json()
        if payload["state"] in {"succeeded", "failed", "cancelled"}:
            return payload
        assert time.monotonic() < deadline
        time.sleep(0.02)


def test_preview_is_stateless_and_uses_one_based_paging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: list[str] = []

    async def fake_annotate_batch(
        _wire, _model, _api_key, _instruction, _classes, texts, _config
    ):
        captured.extend(texts)
        return ["support" for _ in texts]

    monkeypatch.setattr(
        annotation_service_module, "annotate_batch", fake_annotate_batch
    )
    with _client(tmp_path) as client:
        csrf = client.get("/api/session").json()["csrf_token"]
        unsafe = {"Origin": "http://testserver", "X-CSRF-Token": csrf}
        _configure_credentials(client, unsafe)
        workspace_id, node_id, _etag = _source(client, unsafe)
        rejected = client.post(
            f"/api/workspaces/{workspace_id}/nodes/{node_id}/annotation-previews",
            json={
                **_request(),
                "api_key": "request-secret",
                "page": 2,
                "page_size": 1,
            },
            headers=unsafe,
        )
        assert rejected.status_code == 400
        assert rejected.json()["code"] == "invalid_input"
        response = client.post(
            f"/api/workspaces/{workspace_id}/nodes/{node_id}/annotation-previews",
            json={**_request(), "page": 2, "page_size": 1},
            headers=unsafe,
        )
        assert response.status_code == 200
        assert captured == ["second document"]
        assert response.json() == {
            "node_id": node_id,
            "page": 2,
            "page_size": 1,
            "total_rows": 2,
            "labels": [{"row_index": 1, "label": "support"}],
        }


def test_full_annotation_is_secret_free_and_completes_as_an_analysis(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def fake_annotate_all(
        _wire,
        _model,
        api_key,
        _instruction,
        _classes,
        texts,
        **_kwargs,
    ):
        assert api_key == "provider-secret"
        return ["support", "critical"][: len(texts)]

    class _ProgressQueue:
        def __init__(self) -> None:
            self.items: list[object] = []

        def put(self, item: object) -> None:
            self.items.append(item)

    async def execute_in_process(_self, _key, invocation, report_progress):
        progress = _ProgressQueue()
        result = await run_sync_in_worker_thread(
            partial(
                invocation.function,
                **dict(invocation.kwargs),
                progress_queue=progress,
            )
        )
        for item in progress.items:
            await report_progress(item)
        return result

    monkeypatch.setattr(annotation_worker_module, "annotate_all", fake_annotate_all)
    monkeypatch.setattr(
        analysis_executor_module.AnalysisProcessExecutor,
        "execute_reserved",
        execute_in_process,
    )
    with _client(tmp_path) as client:
        csrf = client.get("/api/session").json()["csrf_token"]
        unsafe = {"Origin": "http://testserver", "X-CSRF-Token": csrf}
        _configure_credentials(client, unsafe)
        workspace_id, node_id, _etag = _source(client, unsafe)
        tab_id = client.post(
            f"/api/workspaces/{workspace_id}/tabs",
            json={"kind": "annotation", "name": "Document classes"},
            headers=unsafe,
        ).json()["id"]
        rejected = client.post(
            f"/api/workspaces/{workspace_id}/tabs/{tab_id}/analysis",
            json={
                "kind": "annotation",
                "node_id": node_id,
                **_request(),
                "output_node_name": "Classified documents",
                "api_key": "request-secret",
            },
            headers=unsafe,
        )
        assert rejected.status_code == 400
        assert rejected.json()["code"] == "invalid_input"
        accepted = client.post(
            f"/api/workspaces/{workspace_id}/tabs/{tab_id}/analysis",
                json={
                    "kind": "annotation",
                    "node_id": node_id,
                **_request(),
                "output_node_name": "Classified documents",
            },
            headers=unsafe,
        )
        assert accepted.status_code == 201, accepted.text
        analysis_id = accepted.json()["id"]
        analysis = _wait_analysis(client, workspace_id, analysis_id)
        assert analysis["state"] == "succeeded", analysis
        assert "provider-secret" not in str(analysis)

        result = client.get(
            f"/api/workspaces/{workspace_id}/analyses/{analysis_id}/result"
        )
        assert result.status_code == 200, result.text
        assert result.json()["kind"] == "annotation"
        assert len(result.json()["output_node_ids"]) == 1
        derived_id = result.json()["output_node_ids"][0]

        detail = client.get(f"/api/workspaces/{workspace_id}").json()
        nodes = client.get(f"/api/workspaces/{workspace_id}/nodes").json()
        assert any(node["id"] == derived_id for node in nodes)
        rows = client.post(
            f"/api/workspaces/{workspace_id}/sql",
            json={
                "mode": "query",
                "node_ids": [derived_id],
                "sql": f'SELECT * FROM "{derived_id}"',
                "page": 1,
                "page_size": 10,
            },
            headers=unsafe,
        )
        assert rows.status_code == 200
        frame = pl.read_ipc_stream(BytesIO(rows.content))
        assert frame["stance"].to_list() == [
            "support",
            "critical",
        ]
        assert detail["total_nodes"] == 2
        workspace_path = tmp_path / "workspaces" / workspace_id
        assert all(
            b"provider-secret" not in path.read_bytes()
            for path in workspace_path.rglob("*")
            if path.is_file()
        )


def test_model_discovery_rejects_custom_provider_urls_by_construction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def fake_models(provider, api_key):
        assert provider == "openai"
        assert api_key == "provider-secret"
        return ["model-b", "model-a"]

    monkeypatch.setattr(annotation_service_module, "list_models", fake_models)
    with _client(tmp_path) as client:
        csrf = client.get("/api/session").json()["csrf_token"]
        unsafe = {"Origin": "http://testserver", "X-CSRF-Token": csrf}
        _configure_credentials(client, unsafe)
        supplied = client.post(
            "/api/annotation-providers/openai/models",
            json={"api_key": "request-secret"},
            headers=unsafe,
        )
        assert supplied.status_code == 400
        assert supplied.json()["code"] == "invalid_input"
        response = client.post(
            "/api/annotation-providers/openai/models",
            json={},
            headers=unsafe,
        )
        assert response.status_code == 200
        assert response.json() == {
            "provider": "openai",
            "models": ["model-b", "model-a"],
        }
        rejected = client.post(
            "/api/annotation-providers/custom/models",
            json={},
            headers=unsafe,
        )
        assert rejected.status_code == 422
        assert client.get("/api/annotation-providers/openai/models").status_code == 405


def test_multi_user_model_and_preview_credentials_are_request_only(
    multi_user_test_client: TestClient,
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def fake_models(provider, api_key):
        assert provider == "openai"
        assert api_key == "browser-only-secret"
        return ["model-a"]

    async def fake_annotate_batch(
        _wire, _model, api_key, _instruction, _classes, texts, _config
    ):
        assert api_key == "browser-only-secret"
        return ["support" for _ in texts]

    monkeypatch.setattr(annotation_service_module, "list_models", fake_models)
    monkeypatch.setattr(
        annotation_service_module,
        "annotate_batch",
        fake_annotate_batch,
    )
    missing = multi_user_test_client.post(
        "/api/annotation-providers/openai/models",
        json={},
    )
    assert missing.status_code == 409
    assert missing.json()["code"] == "provider_credential_missing"
    models = multi_user_test_client.post(
        "/api/annotation-providers/openai/models",
        json={"api_key": "browser-only-secret"},
    )
    assert models.status_code == 200
    assert models.json()["models"] == ["model-a"]

    workspace_id, node_id, _etag = _source(multi_user_test_client, {})
    preview = multi_user_test_client.post(
        f"/api/workspaces/{workspace_id}/nodes/{node_id}/annotation-previews",
        json={
            **_request(),
            "api_key": "browser-only-secret",
            "page": 1,
            "page_size": 1,
        },
    )
    assert preview.status_code == 200, preview.text
    assert all(
        b"browser-only-secret" not in path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
