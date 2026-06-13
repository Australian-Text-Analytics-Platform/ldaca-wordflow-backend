import asyncio
import time

import polars as pl
import pytest
from ldaca_wordflow.analysis.manager import get_task_manager
from ldaca_wordflow.api.workspaces.analyses.concordance_core import (
    DEFAULT_CONCORDANCE_PAGE_SIZE,
)
from ldaca_wordflow.api.workspaces.analyses.page_size_estimation import (
    DEFAULT_PAGE_SIZE_CANDIDATES,
)
from ldaca_wordflow.core.workspace import workspace_manager

from docworkspace import Node

# When a client sends no page_size, the backend estimates a dense first-page size
# by probing DEFAULT_PAGE_SIZE_CANDIDATES. For small fixtures the estimator falls
# back to the largest candidate since the 10-occurrence target cannot be reached.


def _assert_grouped_result_rows(node_result: dict, *, expected_page_size: int):
    assert node_result["pagination"]["page_size"] == expected_page_size
    assert isinstance(node_result["data"], list)
    assert node_result["data"]

    first_group = node_result["data"][0]
    assert isinstance(first_group, list)
    assert first_group
    assert all(isinstance(hit, dict) for hit in first_group)
    assert all("CONC_matched_text" in hit for hit in first_group)


async def _wait_for_concordance_result(
    client,
    workspace_id: str,
    task_id: str,
    *,
    timeout: float = 20.0,
    poll_interval: float = 0.25,
):
    """Poll the current-result endpoint until concordance data is available."""

    deadline = time.monotonic() + timeout
    last_payload = None

    while time.monotonic() < deadline:
        resp = await client.get(f"/api/workspaces/concordance/tasks/{task_id}/result")
        if resp.status_code != 200:
            await asyncio.sleep(poll_interval)
            continue

        payload = resp.json()
        if payload and payload.get("state") == "successful" and payload.get("data"):
            return payload

        last_payload = payload
        await asyncio.sleep(poll_interval)

    raise AssertionError(
        f"Concordance result not available after {timeout}s (last payload={last_payload})"
    )


def _clear_concordance_state(user_id: str, workspace_id: str):
    task_manager = get_task_manager(user_id)
    task_manager.clear_all()


def _add_node(workspace_id: str, data: pl.LazyFrame, node_name: str):
    workspace = workspace_manager.get_current_workspace("test")
    assert workspace is not None
    node = Node(
        data=data,
        name=node_name,
        workspace=workspace,
        operation="test_setup",
        parents=[],
    )
    workspace.add_node(node)
    return node


@pytest.mark.anyio
async def test_concordance_single_node_roundtrip(authenticated_client, workspace_id):
    """Single-node concordance should store results and expose task request/result endpoints."""
    # Ensure clean state for this workspace/user
    _clear_concordance_state("test", workspace_id)

    df = pl.DataFrame(
        {
            "text": [
                "alpha beta alpha",
                "beta gamma",
                "Alpha beta",  # Mixed case to test case sensitivity flag
            ],
            "speaker": ["A", "B", "C"],
        }
    )
    node = _add_node(workspace_id, df.lazy(), "single_text_node")
    node.document = "text"
    assert node is not None

    request_payload = {
        "node_ids": [node.id],
        "node_columns": {node.id: "text"},
        "search_word": "alpha",
        "num_left_tokens": 2,
        "num_right_tokens": 2,
        "regex": False,
        "case_sensitive": False,
    }

    resp = await authenticated_client.post(
        "/api/workspaces/concordance",
        json=request_payload,
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["state"] == "successful"
    task_id = payload.get("metadata", {}).get("task_id")
    assert task_id

    result_payload = await _wait_for_concordance_result(
        authenticated_client, workspace_id, task_id
    )
    assert result_payload["state"] == "successful"
    assert result_payload.get("combinable") is False
    assert node.id in result_payload["data"]
    node_result = result_payload["data"][node.id]
    assert node_result["metadata"]["concordance_columns"]
    _assert_grouped_result_rows(
        node_result,
        expected_page_size=max(DEFAULT_PAGE_SIZE_CANDIDATES),
    )
    assert result_payload["analysis_params"]["node_ids"] == [node.id]
    assert (
        result_payload["analysis_params"].get(
            "page_size", DEFAULT_CONCORDANCE_PAGE_SIZE
        )
        == DEFAULT_CONCORDANCE_PAGE_SIZE
    )
    assert result_payload["analysis_params"].get("descending", True) is True

    # Current request should surface the persisted request
    current_req = await authenticated_client.get(
        f"/api/workspaces/concordance/tasks/{task_id}/request"
    )
    assert current_req.status_code == 200
    current_req_payload = current_req.json()
    assert current_req_payload["node_ids"] == [node.id]
    assert current_req_payload["node_columns"][node.id] == "text"
    assert "page" not in current_req_payload
    assert "page_size" not in current_req_payload
    assert "sort_by" not in current_req_payload
    assert "descending" not in current_req_payload
    assert "pagination" not in current_req_payload

    task_manager = get_task_manager("test")
    stored_task = task_manager.get_task(task_id)
    stored_request = (
        stored_task.request.model_dump()
        if stored_task and hasattr(stored_task.request, "model_dump")
        else {}
    )
    assert "page" not in stored_request
    assert "page_size" not in stored_request
    assert "pagination" not in stored_request

    task = task_manager.get_task(task_id)
    assert task is not None
    stored_result = task.result.to_json() if task.result else {}
    assert isinstance(stored_result, dict)
    assert stored_result.get("ready") is True

    # Request a smaller page size via POST (non-persistent override)
    current_res_post = await authenticated_client.post(
        f"/api/workspaces/concordance/tasks/{task_id}/result",
        json={"node_id": node.id, "page_size": 1},
    )
    assert current_res_post.status_code == 200
    tailored = current_res_post.json()
    assert tailored["state"] == "successful"
    assert node.id in tailored["data"]
    node_fetch = tailored["data"][node.id]
    _assert_grouped_result_rows(node_fetch, expected_page_size=1)
    assert len(node_fetch["data"]) == 1
    assert len(node_fetch["data"][0]) >= 1
    assert tailored["analysis_params"].get("page_size") == 1

    # Request the second page explicitly using node_id and page_number alias
    page_two = await authenticated_client.post(
        f"/api/workspaces/concordance/tasks/{task_id}/result",
        json={"node_id": node.id, "page_number": 2, "page_size": 1},
    )
    assert page_two.status_code == 200
    page_two_payload = page_two.json()
    assert page_two_payload["state"] == "successful"
    assert page_two_payload["data"][node.id]["pagination"]["page"] == 2

    # GET again should return default pagination (no persisted overrides)
    refreshed_payload = await _wait_for_concordance_result(
        authenticated_client, workspace_id, task_id
    )
    assert refreshed_payload["data"][node.id]["pagination"]["page_size"] == max(
        DEFAULT_PAGE_SIZE_CANDIDATES
    )


@pytest.mark.anyio
async def test_concordance_multi_node_separated(authenticated_client, workspace_id):
    """Two-node concordance returns one independently-paged slice per node.

    The combined comparison view is synthesized client-side, so the backend only
    ever returns per-node keys. This guards that both nodes appear, that a scoped
    ``node_id`` page override re-pages only the targeted node, and that
    ``combinable`` advertises the combine action to the frontend.
    """
    _clear_concordance_state("test", workspace_id)

    df_left = pl.DataFrame(
        {
            "text": ["alpha beta", "beta alpha", "gamma alpha"],
            "speaker": ["L1", "L2", "L3"],
        }
    )
    df_right = pl.DataFrame(
        {
            "text": ["alpha delta", "epsilon alpha", "alpha"],
            "speaker": ["R1", "R2", "R3"],
        }
    )

    left_node = _add_node(workspace_id, df_left.lazy(), "left_docs")
    left_node.document = "text"
    right_node = _add_node(workspace_id, df_right.lazy(), "right_docs")
    right_node.document = "text"

    request_payload = {
        "node_ids": [left_node.id, right_node.id],
        "node_columns": {left_node.id: "text", right_node.id: "text"},
        "search_word": "alpha",
        "num_left_tokens": 2,
        "num_right_tokens": 2,
        "regex": False,
        "case_sensitive": False,
    }

    resp = await authenticated_client.post(
        "/api/workspaces/concordance",
        json=request_payload,
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["state"] == "successful"
    task_id = payload.get("metadata", {}).get("task_id")
    assert task_id

    result_payload = await _wait_for_concordance_result(
        authenticated_client, workspace_id, task_id
    )
    assert result_payload["state"] == "successful"
    assert result_payload.get("combinable") is True
    # Per-node keys only; no backend __COMBINED__ payload.
    assert "__COMBINED__" not in result_payload["data"]
    assert left_node.id in result_payload["data"]
    assert right_node.id in result_payload["data"]
    for node_id in (left_node.id, right_node.id):
        _assert_grouped_result_rows(
            result_payload["data"][node_id],
            expected_page_size=max(DEFAULT_PAGE_SIZE_CANDIDATES),
        )

    # A scoped node_id page override re-pages only that node; the sibling key is
    # absent from the partial response so the frontend keeps its existing slice.
    scoped = await authenticated_client.post(
        f"/api/workspaces/concordance/tasks/{task_id}/result",
        json={"node_id": left_node.id, "page": 2, "page_size": 1},
    )
    assert scoped.status_code == 200
    scoped_payload = scoped.json()
    assert scoped_payload["state"] == "successful"
    assert left_node.id in scoped_payload["data"]
    assert right_node.id not in scoped_payload["data"]
    assert scoped_payload["data"][left_node.id]["pagination"]["page"] == 2
    assert scoped_payload["data"][left_node.id]["pagination"]["page_size"] == 1


@pytest.mark.anyio
async def test_concordance_multi_node_mismatched_columns(
    authenticated_client, workspace_id
):
    """Two nodes with different schemas each return their own per-node columns."""
    _clear_concordance_state("test", workspace_id)

    left_df = pl.DataFrame(
        {
            "text": ["alpha beta", "beta alpha"],
            "speaker": ["L1", "L2"],
            "topic": ["economy", "housing"],
        }
    )
    right_df = pl.DataFrame(
        {
            "text": ["alpha delta", "alpha"],
            "speaker": ["R1", "R2"],
            "word_count": [200, 150],
        }
    )

    left_node = _add_node(workspace_id, left_df.lazy(), "left_docs")
    left_node.document = "text"
    right_node = _add_node(workspace_id, right_df.lazy(), "right_docs")
    right_node.document = "text"

    request_payload = {
        "node_ids": [left_node.id, right_node.id],
        "node_columns": {left_node.id: "text", right_node.id: "text"},
        "search_word": "alpha",
        "num_left_tokens": 2,
        "num_right_tokens": 2,
        "regex": False,
        "case_sensitive": False,
    }

    resp = await authenticated_client.post(
        "/api/workspaces/concordance",
        json=request_payload,
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["state"] == "successful"
    task_id = payload.get("metadata", {}).get("task_id")
    assert task_id

    result_payload = await _wait_for_concordance_result(
        authenticated_client, workspace_id, task_id
    )
    assert result_payload["state"] == "successful"
    assert result_payload.get("combinable") is True
    assert "__COMBINED__" not in result_payload["data"]
    assert left_node.id in result_payload["data"]
    assert right_node.id in result_payload["data"]
    # Each node carries its own metadata columns reflecting its distinct schema.
    left_columns = result_payload["data"][left_node.id]["columns"]
    right_columns = result_payload["data"][right_node.id]["columns"]
    assert "topic" in left_columns
    assert "word_count" in right_columns
