"""Small shared fixtures for the canonical backend test suite."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from ldaca_wordflow.domain.workspace import Node, Workspace
from ldaca_wordflow.main import create_app
from ldaca_wordflow.settings import Settings
from ldaca_wordflow.workers.input_snapshots import create_worker_input_snapshot


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def files_test_client(tmp_path: Path):
    """Run the real lifespan with an isolated single-user file root and CSRF."""

    app = create_app(
        Settings(
            data_root=tmp_path,
            multi_user=False,
            session_cookie_secure=False,
            cors_allowed_origins=("http://testserver",),
            trusted_hosts=("testserver",),
        ),
        serve_frontend=False,
    )
    with TestClient(app, base_url="http://testserver") as client:
        csrf = client.get("/api/session").json()["csrf_token"]
        client.headers.update(
            {
                "Origin": "http://testserver",
                "X-CSRF-Token": csrf,
            }
        )
        yield client


@pytest.fixture
def worker_snapshot(tmp_path: Path):
    """Create one canonical task-input snapshot from in-memory test columns."""

    def create(*, node_id: str, columns: dict[str, list[object]]) -> Path:
        workspace = Workspace(name="Worker fixture", workspace_id="fixture")
        workspace.add_node(
            Node(
                data=pl.DataFrame(columns).lazy(),
                name="Source",
                id=node_id,
                document="document" if "document" in columns else None,
            )
        )
        return create_worker_input_snapshot(
            workspace_id=workspace.id,
            node_ids=[node_id],
            workspace=workspace,
            workspace_data_dir=tmp_path,
            snapshot_dir=tmp_path / f"snapshot-{node_id}",
            max_snapshot_bytes=1024 * 1024,
        )

    return create
