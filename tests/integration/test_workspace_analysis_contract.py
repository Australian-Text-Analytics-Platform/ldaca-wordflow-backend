"""End-to-end contract for Workspace-owned Analysis lifecycle resources."""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from ldaca_wordflow.main import create_app
from ldaca_wordflow.settings import Settings


def _wait_analysis(
    client: TestClient,
    workspace_id: str,
    analysis_id: str,
) -> dict[str, object]:
    deadline = time.monotonic() + 15
    while True:
        detail = client.get(f"/api/workspaces/{workspace_id}/analyses/{analysis_id}")
        assert detail.status_code == 200, detail.text
        payload = detail.json()
        if payload["state"] in {"succeeded", "failed", "cancelled"}:
            return payload
        assert time.monotonic() < deadline
        time.sleep(0.05)


def test_sequential_analysis_is_owned_by_its_tab_and_workspace(tmp_path: Path) -> None:
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
        upload = client.post(
            "/api/user-files/uploads",
            params={"path": "timeline.csv"},
            content=b"time,value\n0,1\n1,2\n2,3\n",
            headers={**unsafe, "Content-Type": "application/octet-stream"},
        )
        assert upload.status_code == 201
        workspace = client.post(
            "/api/workspaces",
            json={"name": "Analysis contract"},
            headers=unsafe,
        ).json()
        workspace_id = workspace["id"]
        assert (
            client.put(
                f"/api/workspaces/{workspace_id}/open", headers=unsafe
            ).status_code
            == 200
        )
        node = client.post(
            f"/api/workspaces/{workspace_id}/nodes",
            json={"kind": "file", "file_path": "timeline.csv"},
            headers=unsafe,
        ).json()
        tab = client.post(
            f"/api/workspaces/{workspace_id}/tabs",
            json={"kind": "sequential", "name": "Timeline"},
            headers=unsafe,
        ).json()

        created = client.post(
            f"/api/workspaces/{workspace_id}/tabs/{tab['id']}/analysis",
            json={
                "kind": "sequential",
                "node_id": node["id"],
                "time_column": "time",
                "column_type": "numeric",
                "numeric_interval": 1,
            },
            headers=unsafe,
        )
        assert created.status_code == 201, created.text
        analysis_id = created.json()["id"]
        assert created.json()["state"] == "queued"
        assert created.headers["location"] == (
            f"/api/workspaces/{workspace_id}/analyses/{analysis_id}"
        )

        detail_payload = _wait_analysis(client, workspace_id, analysis_id)
        assert detail_payload["state"] == "succeeded", detail_payload
        assert (
            client.get(
                f"/api/workspaces/{workspace_id}/tabs/{tab['id']}/analysis"
            ).json()
            == detail_payload
        )

        listing = client.get(f"/api/workspaces/{workspace_id}/analyses")
        assert listing.status_code == 200
        assert [item["id"] for item in listing.json()["items"]] == [analysis_id]

        result = client.get(
            f"/api/workspaces/{workspace_id}/analyses/{analysis_id}/result"
        )
        assert result.status_code == 200, result.text
        assert result.json()["kind"] == "sequential"
        queried = client.post(
            f"/api/workspaces/{workspace_id}/analyses/{analysis_id}/result/query",
            json={"kind": "sequential", "page": 1, "page_size": 1},
            headers=unsafe,
        )
        assert queried.status_code == 200, queried.text
        assert len(queried.json()["data"]) == 1
        assert queried.json()["pagination"]["page_size"] == 1

        cleared = client.delete(
            f"/api/workspaces/{workspace_id}/tabs/{tab['id']}/analysis",
            headers=unsafe,
        )
        assert cleared.status_code == 204
        assert cleared.content == b""
        assert (
            client.get(
                f"/api/workspaces/{workspace_id}/tabs/{tab['id']}/analysis"
            ).status_code
            == 404
        )


def test_analysis_artifacts_publish_under_the_analysis_directory(
    tmp_path: Path,
) -> None:
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
                params={"path": "words.csv"},
                content=b"text\nhello world hello\nworld again\n",
                headers={**unsafe, "Content-Type": "application/octet-stream"},
            ).status_code
            == 201
        )
        workspace_id = client.post(
            "/api/workspaces",
            json={"name": "Artifact ownership"},
            headers=unsafe,
        ).json()["id"]
        assert (
            client.put(
                f"/api/workspaces/{workspace_id}/open", headers=unsafe
            ).status_code
            == 200
        )
        node_id = client.post(
            f"/api/workspaces/{workspace_id}/nodes",
            json={"kind": "file", "file_path": "words.csv"},
            headers=unsafe,
        ).json()["id"]
        tab_id = client.post(
            f"/api/workspaces/{workspace_id}/tabs",
            json={"kind": "token_frequency", "name": "Tokens"},
            headers=unsafe,
        ).json()["id"]
        created = client.post(
            f"/api/workspaces/{workspace_id}/tabs/{tab_id}/analysis",
            json={
                "kind": "token_frequency",
                "node_ids": [node_id],
                "node_columns": {node_id: "text"},
                "node_tokenizer_models": {node_id: "native:plain_words_en"},
            },
            headers=unsafe,
        )
        assert created.status_code == 201, created.text
        analysis_id = created.json()["id"]
        terminal = _wait_analysis(client, workspace_id, analysis_id)
        assert terminal["state"] == "succeeded", terminal

        analysis_dir = tmp_path / "workspaces" / workspace_id / "analyses" / analysis_id
        assert not (analysis_dir / ".execution").exists()
        assert any((analysis_dir / "artifacts").iterdir())

        workspace_payload = json.loads(
            (tmp_path / "workspaces" / workspace_id / "workspace.json").read_text()
        )
        reference = next(
            item for item in workspace_payload["analyses"] if item["id"] == analysis_id
        )
        record_text = (
            tmp_path / "workspaces" / workspace_id / reference["record_path"]
        ).read_text()
        assert str(tmp_path) not in record_text
        record = json.loads(record_text)
        assert record["artifact_references"]

        result = client.get(
            f"/api/workspaces/{workspace_id}/analyses/{analysis_id}/result"
        )
        assert result.status_code == 200, result.text
        payload = result.json()
        assert payload["kind"] == "token_frequency"
        artifact_url = payload["artifacts"]["nodes"][0]["token_parquet_path"]["url"]
        assert artifact_url.startswith(
            f"/api/workspaces/{workspace_id}/analyses/{analysis_id}/artifacts/"
        )
        download = client.get(artifact_url)
        assert download.status_code == 200
        assert download.content

        artifact_path = analysis_dir / record["artifact_references"][0]["relative_path"]
        artifact_path.unlink()
        gone = client.get(
            f"/api/workspaces/{workspace_id}/analyses/{analysis_id}/result"
        )
        assert gone.status_code == 410
        assert gone.json()["code"] == "artifact_gone"

        assert (
            client.delete(
                f"/api/workspaces/{workspace_id}/nodes/{node_id}",
                headers=unsafe,
            ).status_code
            == 204
        )
        invalid = client.get(f"/api/workspaces/{workspace_id}/analyses/{analysis_id}")
        assert invalid.status_code == 200
        assert invalid.json()["integrity"] == {
            "status": "invalid",
            "code": "analysis_input_missing",
            "missing_input_ids": [node_id],
        }
        unusable = client.get(
            f"/api/workspaces/{workspace_id}/analyses/{analysis_id}/result"
        )
        assert unusable.status_code == 410
        assert unusable.json()["code"] == "analysis_input_missing"


def test_child_analysis_publishes_an_independent_data_block(tmp_path: Path) -> None:
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
                params={"path": "documents.csv"},
                content=b"text,source\nhello world,A\nhello again,B\n",
                headers={**unsafe, "Content-Type": "application/octet-stream"},
            ).status_code
            == 201
        )
        workspace_id = client.post(
            "/api/workspaces",
            json={"name": "Child Analysis"},
            headers=unsafe,
        ).json()["id"]
        assert (
            client.put(
                f"/api/workspaces/{workspace_id}/open", headers=unsafe
            ).status_code
            == 200
        )
        node_id = client.post(
            f"/api/workspaces/{workspace_id}/nodes",
            json={"kind": "file", "file_path": "documents.csv"},
            headers=unsafe,
        ).json()["id"]
        tab_id = client.post(
            f"/api/workspaces/{workspace_id}/tabs",
            json={"kind": "concordance", "name": "Search"},
            headers=unsafe,
        ).json()["id"]
        root = client.post(
            f"/api/workspaces/{workspace_id}/tabs/{tab_id}/analysis",
            json={
                "kind": "concordance",
                "node_ids": [node_id],
                "node_columns": {node_id: "text"},
                "search_word": "hello",
            },
            headers=unsafe,
        )
        assert root.status_code == 201, root.text
        root_id = root.json()["id"]
        assert _wait_analysis(client, workspace_id, root_id)["state"] == "succeeded"

        created = client.post(
            f"/api/workspaces/{workspace_id}/analyses/{root_id}/children",
            json={
                "kind": "concordance_detachment",
                "node_id": node_id,
                "selected_columns": ["source", "CONC_matched_text"],
                "name": "Hello matches",
            },
            headers=unsafe,
        )
        assert created.status_code == 201, created.text
        child = created.json()
        child_id = child["id"]
        assert child["parent_analysis_id"] == root_id
        assert created.headers["location"] == (
            f"/api/workspaces/{workspace_id}/analyses/{child_id}"
        )
        terminal = _wait_analysis(client, workspace_id, child_id)
        assert terminal["state"] == "succeeded", terminal

        result = client.get(
            f"/api/workspaces/{workspace_id}/analyses/{child_id}/result"
        )
        assert result.status_code == 200, result.text
        payload = result.json()
        assert payload["kind"] == "concordance_detachment"
        assert payload["output_columns"] == ["source", "CONC_matched_text"]
        output_node_id = payload["output_node_id"]
        output = client.get(f"/api/workspaces/{workspace_id}/nodes/{output_node_id}")
        assert output.status_code == 200, output.text
        assert output.json()["name"] == "Hello matches"

        assert (
            client.delete(
                f"/api/workspaces/{workspace_id}/tabs/{tab_id}/analysis",
                headers=unsafe,
            ).status_code
            == 204
        )
        assert (
            client.get(
                f"/api/workspaces/{workspace_id}/nodes/{output_node_id}"
            ).status_code
            == 200
        )
