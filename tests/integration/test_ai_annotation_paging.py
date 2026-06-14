import shutil

import pytest
from ldaca_wordflow.analysis.manager import get_task_manager
from ldaca_wordflow.core.workspace import workspace_manager


@pytest.mark.anyio
async def test_ai_annotation_recomputes_pages_without_persisting_results(
    authenticated_client,
    workspace_id,
    sample_node_id,
    monkeypatch,
):
    calls: list[list[str]] = []

    async def fake_classify_texts(*, texts, text_column_name="text", **_kwargs):
        calls.append(list(texts))
        return [
            {
                "row_index": index,
                text_column_name: text,
                "classification": f"label-{index}",
                "error": None,
            }
            for index, text in enumerate(texts)
        ]

    monkeypatch.setattr(
        "ldaca_wordflow.api.workspaces.analyses.ai_annotation.classify_texts",
        fake_classify_texts,
    )

    request_payload = {
        "node_ids": [sample_node_id],
        "node_columns": {sample_node_id: "document"},
        "classes": [
            {"name": "support", "description": "Supportive tone"},
            {"name": "critical", "description": "Critical tone"},
        ],
        "model": "gpt-4o-mini",
        "page": 1,
        "page_size": 2,
    }

    initial_response = await authenticated_client.post(
        "/api/workspaces/ai-annotation",
        json=request_payload,
    )

    assert initial_response.status_code == 200
    initial_payload = initial_response.json()
    assert initial_payload["state"] == "successful"

    task_id = initial_payload["metadata"]["task_id"]
    task_manager = get_task_manager("test")
    task = task_manager.get_task(task_id)
    assert task is not None

    stored_result = task.result.to_json() if task.result is not None else {}
    assert "node_results" not in stored_result

    page_two_response = await authenticated_client.post(
        f"/api/workspaces/ai-annotation/tasks/{task_id}/result",
        json={"page": 2, "page_size": 2},
    )
    assert page_two_response.status_code == 200

    page_two_repeat_response = await authenticated_client.post(
        f"/api/workspaces/ai-annotation/tasks/{task_id}/result",
        json={"page": 2, "page_size": 2},
    )
    assert page_two_repeat_response.status_code == 200

    assert calls == [
        ["This is a sample document.", "Another sample text for analysis."],
        ["More text content for testing.", "Final sample sentence."],
        ["More text content for testing.", "Final sample sentence."],
    ]

    refreshed_task = task_manager.get_task(task_id)
    assert refreshed_task is not None
    refreshed_result = (
        refreshed_task.result.to_json() if refreshed_task.result is not None else {}
    )
    assert "node_results" not in refreshed_result


@pytest.mark.anyio
async def test_ai_annotation_detach_survives_artifact_cleanup(
    authenticated_client,
    workspace_id,
    sample_node_id,
    monkeypatch,
):
    """Regression: detached AI-annotation node must not depend on the artifacts dir.

    Previously the detach handler wrote `result_df` into `data/artifacts/`
    and pointed the new node's LazyFrame at it. Explicit artifact cleanup could
    then wipe that directory, silently corrupting the detached node — the same
    failure mode that affected topic-modeling detach.
    """

    async def fake_classify_texts(*, texts, text_column_name="text", **_kwargs):
        return [
            {
                "row_index": index,
                text_column_name: text,
                "classification": f"label-{index}",
                "error": None,
            }
            for index, text in enumerate(texts)
        ]

    monkeypatch.setattr(
        "ldaca_wordflow.api.workspaces.analyses.ai_annotation.classify_texts",
        fake_classify_texts,
    )

    detach_response = await authenticated_client.post(
        f"/api/workspaces/nodes/{sample_node_id}/ai-annotation/detach",
        json={
            "column": "document",
            "new_node_name": "annotated_sample",
            "annotation_column": "ai_annotation",
            "classes": [
                {"name": "support", "description": "Supportive tone"},
                {"name": "critical", "description": "Critical tone"},
            ],
            "model": "gpt-4o-mini",
        },
    )
    assert detach_response.status_code == 200, detach_response.text

    # Simulate explicit artifact cleanup: wipe everything under `data/artifacts/`.
    workspace_dir = workspace_manager.get_workspace_dir("test", workspace_id)
    assert workspace_dir is not None
    artifacts_dir = workspace_dir / "data" / "artifacts"
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)

    # Locate the newly added detached node and confirm its data still
    # collects — i.e. the parquet lives under workspace-owned `data/`, not
    # the transient artifacts dir.
    ws = workspace_manager.get_current_workspace("test")
    assert ws is not None
    detached = next(
        node for node in ws.nodes.values() if node.name == "annotated_sample"
    )
    collected = detached.data.collect()
    assert "ai_annotation" in collected.columns
    assert len(collected) > 0


@pytest.mark.anyio
async def test_ai_annotation_save_survives_artifact_cleanup(
    authenticated_client,
    workspace_id,
    sample_node_id,
    monkeypatch,
):
    """Regression: saving AI annotations must not point the node at the artifacts dir.

    `save_ai_annotation` mutates an existing node's `.data` — if it scans
    a transient artifact parquet, later explicit artifact cleanup corrupts
    the user's data. This used to be the case before the fix.
    """

    async def fake_classify_texts(*, texts, text_column_name="text", **_kwargs):
        return [
            {
                "row_index": index,
                text_column_name: text,
                "classification": f"label-{index}",
                "error": None,
            }
            for index, text in enumerate(texts)
        ]

    monkeypatch.setattr(
        "ldaca_wordflow.api.workspaces.analyses.ai_annotation.classify_texts",
        fake_classify_texts,
    )

    detach_response = await authenticated_client.post(
        f"/api/workspaces/nodes/{sample_node_id}/ai-annotation/detach",
        json={
            "column": "document",
            "new_node_name": "annotated_for_save",
            "annotation_column": "ai_annotation",
            "classes": [
                {"name": "support", "description": "Supportive tone"},
                {"name": "critical", "description": "Critical tone"},
            ],
            "model": "gpt-4o-mini",
        },
    )
    assert detach_response.status_code == 200, detach_response.text

    ws = workspace_manager.get_current_workspace("test")
    assert ws is not None
    detached = next(
        node for node in ws.nodes.values() if node.name == "annotated_for_save"
    )
    detached_id = detached.id

    save_response = await authenticated_client.post(
        f"/api/workspaces/nodes/{detached_id}/ai-annotation/save",
        json={
            "annotation_column": "ai_annotation",
            "edits": [{"row_index": 0, "provider": "test", "annotation": "edited"}],
        },
    )
    assert save_response.status_code == 200, save_response.text

    workspace_dir = workspace_manager.get_workspace_dir("test", workspace_id)
    assert workspace_dir is not None
    artifacts_dir = workspace_dir / "data" / "artifacts"
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)

    # The save endpoint reassigns `node.data`; after artifact cleanup the
    # saved value must still be readable.
    ws = workspace_manager.get_current_workspace("test")
    assert ws is not None
    saved_node = ws.nodes[detached_id]
    collected = saved_node.data.collect()
    assert collected["ai_annotation"].to_list()[0] == "edited"
