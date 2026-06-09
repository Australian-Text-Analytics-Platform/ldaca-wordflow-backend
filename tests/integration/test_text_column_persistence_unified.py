from types import SimpleNamespace

import polars as pl
import pytest
from ldaca_wordflow.analysis.manager import get_task_manager
from ldaca_wordflow.core.workspace import workspace_manager

from docworkspace import Node


@pytest.fixture(autouse=True)
def _stub_worker_task_manager(monkeypatch):
    class ImmediateTaskManager:
        async def any_running(self, **_kwargs):
            return False

        async def latest_by_type(self, *args, **_kwargs):
            return None

        async def submit_task(self, **_kwargs):
            return SimpleNamespace(id="test-worker-task")

    def fake_get_task_manager(self, _user_id):
        return ImmediateTaskManager()

    monkeypatch.setattr(
        workspace_manager.__class__, "get_task_manager", fake_get_task_manager
    )


@pytest.mark.anyio
async def test_text_column_preference_is_set_by_node_endpoint_not_analyses(
    authenticated_client, workspace_id, monkeypatch
):
    user_id = "test"
    workspace = workspace_manager.get_current_workspace(user_id)
    assert workspace is not None

    node = Node(
        data=pl.DataFrame(
            {
                "text_a": [
                    "alpha from column a",
                    "beta from column a",
                    "gamma from column a",
                ],
                "text_b": [
                    "alpha from column b",
                    "delta from column b",
                    "epsilon from column b",
                ],
            }
        ).lazy(),
        name="dual_text_node",
        workspace=workspace,
        operation="test_setup",
        parents=[],
    )
    workspace.add_node(node)

    assert node is not None

    async def set_document_column(column: str) -> None:
        response = await authenticated_client.put(
            f"/api/workspaces/nodes/{node.id}/document-column",
            json={"document_column": column},
        )
        assert response.status_code == 200, response.text
        refreshed_node = workspace.nodes.get(node.id)
        assert refreshed_node is not None
        assert refreshed_node.document == column

    await set_document_column("text_a")

    token_response = await authenticated_client.post(
        "/api/workspaces/token-frequencies",
        json={
            "node_ids": [node.id],
            "node_columns": {node.id: "text_b"},
            "tokenizer_model": "native:plain_words_en",
        },
    )
    assert token_response.status_code == 200, token_response.text

    refreshed = workspace.nodes.get(node.id)
    assert refreshed is not None
    assert refreshed.document == "text_a"

    await set_document_column("text_b")

    concordance_response = await authenticated_client.post(
        "/api/workspaces/concordance",
        json={
            "node_ids": [node.id],
            "node_columns": {node.id: "text_a"},
            "search_word": "alpha",
            "num_left_tokens": 1,
            "num_right_tokens": 1,
            "regex": False,
            "case_sensitive": False,
        },
    )
    assert concordance_response.status_code == 200, concordance_response.text

    refreshed = workspace.nodes.get(node.id)
    assert refreshed is not None
    assert refreshed.document == "text_b"

    await set_document_column("text_a")

    async def fake_compute_quote_dataframe(
        node,
        base_df,
        column,
        engine,
        *,
        use_base_only=False,
        **_kwargs,
    ):
        grouped_quotes = [[] for _ in range(base_df.height)]
        return base_df.with_columns(pl.Series("quotation", grouped_quotes))

    monkeypatch.setattr(
        "ldaca_wordflow.api.workspaces.analyses.quotation_core.compute_quote_dataframe",
        fake_compute_quote_dataframe,
    )

    quotation_response = await authenticated_client.post(
        f"/api/workspaces/nodes/{node.id}/quotation",
        json={
            "column": "text_b",
        },
    )
    assert quotation_response.status_code == 200, quotation_response.text

    refreshed = workspace.nodes.get(node.id)
    assert refreshed is not None
    assert refreshed.document == "text_a"

    await set_document_column("text_b")

    topic_response = await authenticated_client.post(
        "/api/workspaces/topic-modeling",
        json={
            "node_ids": [node.id],
            "node_columns": {node.id: "text_a"},
            "min_topic_size": 2,
        },
    )
    assert topic_response.status_code == 200, topic_response.text

    refreshed = workspace.nodes.get(node.id)
    assert refreshed is not None
    assert refreshed.document == "text_b"

    payload = topic_response.json()
    task_id = payload.get("metadata", {}).get("task_id")
    assert task_id

    analysis_task = get_task_manager(user_id).get_task(task_id)
    assert analysis_task is not None

    request_data = (
        analysis_task.request.model_dump()
        if hasattr(analysis_task.request, "model_dump")
        else analysis_task.request.dict()
    )
    assert request_data["node_columns"][node.id] == "text_a"
