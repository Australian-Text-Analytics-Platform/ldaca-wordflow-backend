"""Tests for files-root LDaCA background task endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(name="client")
def files_client_alias(files_test_client: TestClient):
    return files_test_client


def test_import_ldaca_starts_background_task_under_user_scope(client: TestClient):
    """Import should return running state and task metadata without blocking."""
    from ldaca_wordflow.api import files as files_api

    mock_tm = MagicMock()
    mock_tm.submit_task = AsyncMock(return_value=MagicMock(id="task-123"))

    with (
        patch.object(
            files_api.workspace_manager,
            "get_current_workspace_id",
            return_value=None,
        ),
        patch.object(
            files_api.workspace_manager,
            "get_task_manager",
            return_value=mock_tm,
        ) as get_task_manager,
    ):
        response = client.post(
            "/api/files/import-ldaca",
            headers={"X-LDACA-API-Token": " portal-token "},
            json={"url": "https://example.org/dataset.zip"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "running"
    assert payload["metadata"]["task_id"] == "task-123"
    get_task_manager.assert_called_once_with("test_user")
    mock_tm.submit_task.assert_awaited_once()
    await_args = mock_tm.submit_task.await_args
    assert await_args is not None
    submit_kwargs = await_args.kwargs
    assert submit_kwargs["task_args"]["api_token"] == "portal-token"


def test_import_ldaca_ignores_current_workspace_for_task_scope(client: TestClient):
    """LDaCA import must remain user-scoped even when a workspace is active."""
    from ldaca_wordflow.api import files as files_api

    mock_tm = MagicMock()
    mock_tm.submit_task = AsyncMock(return_value=MagicMock(id="task-456"))

    with (
        patch.object(
            files_api.workspace_manager,
            "get_current_workspace_id",
            return_value="workspace-should-not-be-used",
        ),
        patch.object(
            files_api.workspace_manager,
            "get_task_manager",
            return_value=mock_tm,
        ) as get_task_manager,
    ):
        response = client.post(
            "/api/files/import-ldaca",
            json={"url": "https://example.org/dataset.zip"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "running"
    assert payload["metadata"]["task_id"] == "task-456"
    get_task_manager.assert_called_once_with("test_user")
    await_args = mock_tm.submit_task.await_args
    assert await_args is not None
    assert await_args.kwargs["workspace_id"] == "global"


def test_list_files_tasks_returns_user_scope_tasks(client: TestClient):
    """Files task listing should be filtered to user scope for the current user."""
    from ldaca_wordflow.api import files as files_api

    mock_user_tm = MagicMock()
    mock_user_tm.list = AsyncMock(
        return_value=[
            {
                "task_id": "task-abc",
                "state": "running",
                "task_type": "ldaca_import",
            }
        ]
    )
    with patch.object(
        files_api.workspace_manager,
        "get_task_manager",
        return_value=mock_user_tm,
    ):
        response = client.get("/api/files/tasks")

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "successful"
    assert payload["data"][0]["task_id"] == "task-abc"
    mock_user_tm.list.assert_awaited_once_with(user_id="test_user")


def test_ldaca_featured_returns_staff_picked_collections(client: TestClient):
    """Featured LDaCA collections should be served through the backend proxy."""
    from ldaca_wordflow.api import files as files_api

    fake_client = MagicMock()
    fake_client.featured_collections = AsyncMock(
        return_value=[
            {
                "id": "arcp://name,hdl10.26180~23961609",
                "crate_id": "arcp://name,hdl10.26180~23961609",
                "title": "A COrpus of Oz Early English (COOEE)",
                "description": "Historical English corpus",
                "types": ["Dataset", "RepositoryCollection"],
                "license": "https://creativecommons.org/licenses/by/4.0/",
                "importable": True,
                "stats": {"documents": 600},
            }
        ]
    )

    with patch.object(
        files_api.OniClient, "from_settings", return_value=fake_client
    ) as from_settings:
        response = client.get("/api/files/ldaca/featured")

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "successful"
    assert payload["data"][0]["title"] == "A COrpus of Oz Early English (COOEE)"
    from_settings.assert_called_once_with(files_api.settings, token=None)
    fake_client.featured_collections.assert_awaited_once()


def test_ldaca_search_proxies_typed_search(client: TestClient):
    """Search requests should preserve the requested method and query."""
    from ldaca_wordflow.api import files as files_api
    from ldaca_wordflow.core.oni_client import OniSearchMethod

    fake_client = MagicMock()
    fake_client.search = AsyncMock(
        return_value=[
            {
                "id": "arcp://name,hdl10.26180~23961609",
                "crate_id": "arcp://name,hdl10.26180~23961609",
                "title": "A COrpus of Oz Early English (COOEE)",
                "description": None,
                "types": ["Dataset"],
                "license": None,
                "importable": True,
                "stats": {},
            }
        ]
    )

    with patch.object(
        files_api.OniClient, "from_settings", return_value=fake_client
    ) as from_settings:
        response = client.post(
            "/api/files/ldaca/search",
            headers={"X-LDACA-API-Token": "portal-token"},
            json={
                "method": "identifier",
                "query": "https://data.ldaca.edu.au/collection?id=arcp%3A%2F%2Fname%2Chdl10.26180~23961609",
                "limit": 10,
                "offset": 0,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "successful"
    assert payload["data"][0]["id"] == "arcp://name,hdl10.26180~23961609"
    from_settings.assert_called_once_with(files_api.settings, token="portal-token")
    fake_client.search.assert_awaited_once_with(
        method=OniSearchMethod.IDENTIFIER,
        query="https://data.ldaca.edu.au/collection?id=arcp%3A%2F%2Fname%2Chdl10.26180~23961609",
        limit=10,
        offset=0,
    )
