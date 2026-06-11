from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from ldaca_wordflow.analysis.implementations.topic_modeling import (
    TopicModelingRequest as AnalysisTopicModelingRequest,
)
from ldaca_wordflow.analysis.manager import get_task_manager
from ldaca_wordflow.analysis.models import AnalysisStatus, AnalysisTask
from ldaca_wordflow.analysis.results import GenericAnalysisResult
from ldaca_wordflow.api.workspaces.analyses import (
    topic_modeling as topic_modeling_routes,
)
from ldaca_wordflow.core.workspace import workspace_manager

from docworkspace import Node


@pytest.fixture(autouse=True)
def _stub_worker_task_manager(monkeypatch):
    holder: dict[str, object] = {}

    class ImmediateTaskManager:
        async def any_running(self, **_kwargs):
            return False

        async def latest_by_type(self, *args, **_kwargs):
            return None

        async def submit_task(self, **_kwargs):
            holder["submit_kwargs"] = _kwargs
            return SimpleNamespace(id="topic-worker-task")

    def fake_get_task_manager(self, _user_id):
        return ImmediateTaskManager()

    monkeypatch.setattr(
        workspace_manager.__class__, "get_task_manager", fake_get_task_manager
    )

    return holder


@pytest.mark.asyncio
async def test_clear_topic_modeling_results_success(authenticated_client, workspace_id):
    response = await authenticated_client.delete("/api/workspaces/topic-modeling")

    assert response.status_code == 200
    assert response.json() == {
        "state": "successful",
        "message": "Topic modeling analysis results have been cleared.",
    }


@pytest.mark.asyncio
async def test_topic_modeling_result_returns_payload(
    authenticated_client, workspace_id
):
    user_id = "test"
    task_id = "topic-task-1"

    task_manager = get_task_manager(user_id)
    payload = {
        "topics": [
            {
                "id": 0,
                "label": "topic",
                "representative_words": ["alpha", "beta", "gamma"],
                "size": [1],
                "total_size": 1,
                "x": 0.0,
                "y": 0.0,
            }
        ],
        "corpus_sizes": [1],
        "meta": {"native": True},
    }
    task_manager.save_task(
        AnalysisTask(
            task_id=task_id,
            user_id=user_id,
            workspace_id=workspace_id,
            request=AnalysisTopicModelingRequest(
                node_ids=["node-1"],
                node_columns={"node-1": "document"},
                min_topic_size=5,
                random_seed=42,
                representative_words_count=5,
            ),
            status=AnalysisStatus.COMPLETED,
            result=GenericAnalysisResult(payload),
        )
    )

    response = await authenticated_client.get(
        f"/api/workspaces/topic-modeling/tasks/{task_id}/result"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "successful"
    assert body["data"]["topics"][0]["label"] == "topic"
    assert body["data"]["topics"][0]["representative_words"] == [
        "alpha",
        "beta",
        "gamma",
    ]


@pytest.mark.anyio
async def test_topic_modeling_request_persists_random_seed_and_word_count(
    authenticated_client, workspace_id, _stub_worker_task_manager
):
    user_id = "test"
    workspace = workspace_manager.get_current_workspace(user_id)
    assert workspace is not None

    node = Node(
        data=pl.DataFrame(
            {
                "document": [
                    "alpha beta gamma",
                    "beta gamma delta",
                    "gamma delta epsilon",
                ]
            }
        ).lazy(),
        name="topic_source",
        workspace=workspace,
        operation="test_setup",
        parents=[],
    )
    workspace.add_node(node)

    response = await authenticated_client.post(
        "/api/workspaces/topic-modeling",
        json={
            "node_ids": [node.id],
            "node_columns": {node.id: "document"},
            "min_topic_size": 2,
            "random_seed": 123,
            "representative_words_count": 7,
        },
    )

    assert response.status_code == 200, response.text
    task_id = response.json().get("metadata", {}).get("task_id")
    assert task_id

    analysis_task = get_task_manager(user_id).get_task(task_id)
    assert analysis_task is not None

    request_data = (
        analysis_task.request.model_dump()
        if hasattr(analysis_task.request, "model_dump")
        else analysis_task.request.dict()
    )
    submit_kwargs = _stub_worker_task_manager.get("submit_kwargs")
    assert isinstance(submit_kwargs, dict)
    task_args = submit_kwargs["task_args"]
    assert "corpora" not in task_args
    assert Path(task_args["workspace_dir"]).exists()
    assert task_args["node_infos"][0]["text_column"] == "document"
    assert request_data["random_seed"] == 123
    assert request_data["representative_words_count"] == 7


@pytest.mark.anyio
async def test_topic_modeling_detach_keeps_topic_meaning_only_on_support_node(
    authenticated_client, workspace_id, tmp_path
):
    user_id = "test"
    workspace = workspace_manager.get_current_workspace(user_id)
    assert workspace is not None

    source_node = Node(
        data=pl.DataFrame(
            {
                "document": [
                    "alpha beta",
                    "beta gamma",
                    "gamma delta",
                    "delta epsilon",
                ],
                "source": ["a", "b", "c", "d"],
            }
        ).lazy(),
        name="topic_source",
        workspace=workspace,
        operation="test_setup",
        parents=[],
    )
    workspace.add_node(source_node)

    assignments_path = tmp_path / "assignments.parquet"
    pl.DataFrame(
        {
            "__row_nr__": [1, 3],
            "TOPIC_topic": [0, 1],
        },
        schema={"__row_nr__": pl.Int64, "TOPIC_topic": pl.Int64},
    ).write_parquet(assignments_path)

    meanings_path = tmp_path / "topic_meanings.parquet"
    pl.DataFrame(
        {
            "TOPIC_topic": [0],
            "TOPIC_topic_meaning": [["alpha", "beta", "gamma"]],
        },
        schema={
            "TOPIC_topic": pl.Int64,
            "TOPIC_topic_meaning": pl.List(pl.String),
        },
    ).write_parquet(meanings_path)

    task_id = "completed-topic-task"
    payload = {
        "topics": [
            {
                "id": 0,
                "label": "alpha | beta | gamma",
                "representative_words": ["alpha", "beta", "gamma"],
                "size": [2],
                "total_size": 2,
                "x": 0.0,
                "y": 0.0,
            }
        ],
        "corpus_sizes": [2],
        "artifacts": {
            "version": 1,
            "topic_meanings_parquet_path": str(meanings_path),
            "nodes": [
                {
                    "node_id": source_node.id,
                    "node_name": source_node.name,
                    "text_column": "document",
                    "original_columns": ["document", "source"],
                    "assignments_parquet_path": str(assignments_path),
                }
            ],
        },
    }
    get_task_manager(user_id).save_task(
        AnalysisTask(
            task_id=task_id,
            user_id=user_id,
            workspace_id=workspace_id,
            request=AnalysisTopicModelingRequest(
                node_ids=[source_node.id],
                node_columns={source_node.id: "document"},
                min_topic_size=5,
                random_seed=42,
                representative_words_count=5,
                sample_fractions=[0.5],
            ),
            status=AnalysisStatus.COMPLETED,
            result=GenericAnalysisResult(payload),
        )
    )

    detach_response = await authenticated_client.post(
        f"/api/workspaces/topic-modeling/tasks/{task_id}/detach",
        json={
            "node_ids": [source_node.id],
            # The topic column is now an explicit, default-selected choice.
            "selected_columns": {source_node.id: ["document", "TOPIC_topic"]},
        },
    )

    assert detach_response.status_code == 200, detach_response.text
    detached_node_id = (
        detach_response.json()
        .get("data", {})
        .get("detached_nodes", [{}])[0]
        .get("new_node_id")
    )
    topic_meanings_node_id = (
        detach_response.json()
        .get("data", {})
        .get("detached_nodes", [{}])[0]
        .get("topic_meanings_node_id")
    )
    assert detached_node_id
    assert topic_meanings_node_id

    detached_workspace_node = workspace.nodes[detached_node_id]
    detached_df = detached_workspace_node.data.collect().sort("document")
    assert detached_workspace_node.name == "topic_source_topic_sampled_fr_0_5_rs_42"
    assert detached_df["document"].to_list() == ["beta gamma", "delta epsilon"]
    assert detached_df["TOPIC_topic"].to_list() == [0, 1]

    graph_response = await authenticated_client.get("/api/workspaces/graph")
    assert graph_response.status_code == 200, graph_response.text
    nodes = graph_response.json().get("nodes", [])
    detached_node = next(node for node in nodes if node.get("id") == detached_node_id)
    support_node = next(
        node for node in nodes if node.get("id") == topic_meanings_node_id
    )

    detached_raw_schema = detached_node.get("schema", [])
    detached_schema = (
        {column["name"]: column["js_type"] for column in detached_raw_schema}
        if isinstance(detached_raw_schema, list)
        else detached_raw_schema
    )
    support_raw_schema = support_node.get("schema", [])
    support_schema = (
        {column["name"]: column["js_type"] for column in support_raw_schema}
        if isinstance(support_raw_schema, list)
        else support_raw_schema
    )

    assert "TOPIC_topic_meaning" not in detached_schema
    assert support_schema["TOPIC_topic_meaning"] in {"list_string", "List(String)"}
    assert set(support_schema) == {"TOPIC_topic", "TOPIC_topic_meaning"}


@pytest.mark.anyio
async def test_topic_modeling_detach_survives_artifact_cleanup(
    authenticated_client, workspace_id, tmp_path
):
    """Regression: detached topic nodes must not depend on transient task artifacts.

    Prior to materialising the detach output into a workspace-owned parquet,
    the detached node's LazyFrame still scanned files under
    `data/artifacts/` that get wiped by `clear_previous_completed_analysis_task`
    on the next analysis submit (and by `clear_workspace_artifacts_dir` on
    workspace unload). Wiping those mid-session corrupted every prior detach
    so the next workspace load showed zero nodes.
    """
    user_id = "test"
    workspace = workspace_manager.get_current_workspace(user_id)
    assert workspace is not None

    source_node = Node(
        data=pl.DataFrame(
            {
                "document": [
                    "alpha beta",
                    "beta gamma",
                    "gamma delta",
                    "delta epsilon",
                ],
                "source": ["a", "b", "c", "d"],
            }
        ).lazy(),
        name="topic_source",
        workspace=workspace,
        operation="test_setup",
        parents=[],
    )
    workspace.add_node(source_node)

    assignments_path = tmp_path / "assignments.parquet"
    pl.DataFrame(
        {"__row_nr__": [1, 3], "TOPIC_topic": [0, 1]},
        schema={"__row_nr__": pl.Int64, "TOPIC_topic": pl.Int64},
    ).write_parquet(assignments_path)
    meanings_path = tmp_path / "topic_meanings.parquet"
    pl.DataFrame(
        {"TOPIC_topic": [0, 1], "TOPIC_topic_meaning": [["alpha"], ["delta"]]},
        schema={"TOPIC_topic": pl.Int64, "TOPIC_topic_meaning": pl.List(pl.String)},
    ).write_parquet(meanings_path)

    task_id = "completed-topic-task-survives-cleanup"
    payload = {
        "topics": [
            {
                "id": 0,
                "label": "alpha",
                "representative_words": ["alpha"],
                "size": [1],
                "total_size": 1,
                "x": 0.0,
                "y": 0.0,
            },
            {
                "id": 1,
                "label": "delta",
                "representative_words": ["delta"],
                "size": [1],
                "total_size": 1,
                "x": 0.0,
                "y": 0.0,
            },
        ],
        "corpus_sizes": [2],
        "artifacts": {
            "version": 1,
            "topic_meanings_parquet_path": str(meanings_path),
            "nodes": [
                {
                    "node_id": source_node.id,
                    "node_name": source_node.name,
                    "text_column": "document",
                    "original_columns": ["document", "source"],
                    "assignments_parquet_path": str(assignments_path),
                }
            ],
        },
    }
    get_task_manager(user_id).save_task(
        AnalysisTask(
            task_id=task_id,
            user_id=user_id,
            workspace_id=workspace_id,
            request=AnalysisTopicModelingRequest(
                node_ids=[source_node.id],
                node_columns={source_node.id: "document"},
                min_topic_size=5,
                random_seed=42,
                representative_words_count=5,
                sample_fractions=[0.5],
            ),
            status=AnalysisStatus.COMPLETED,
            result=GenericAnalysisResult(payload),
        )
    )

    detach_response = await authenticated_client.post(
        f"/api/workspaces/topic-modeling/tasks/{task_id}/detach",
        json={
            "node_ids": [source_node.id],
            # The topic column is now an explicit, default-selected choice.
            "selected_columns": {source_node.id: ["document", "TOPIC_topic"]},
        },
    )
    assert detach_response.status_code == 200, detach_response.text
    detached_node_id = detach_response.json()["data"]["detached_nodes"][0][
        "new_node_id"
    ]
    meanings_node_id = detach_response.json()["data"]["detached_nodes"][0][
        "topic_meanings_node_id"
    ]

    # Simulate the artifact cleanup that runs on the next analysis submit
    # / on workspace unload: both transient parquet files vanish.
    assignments_path.unlink()
    meanings_path.unlink()

    # Both detached nodes' data must still be collectible — i.e. the detach
    # output is self-contained in workspace-owned parquet files, not
    # scanning the now-gone artifact paths.
    detached_df = workspace.nodes[detached_node_id].data.collect().sort("document")
    assert detached_df["document"].to_list() == ["beta gamma", "delta epsilon"]
    assert detached_df["TOPIC_topic"].to_list() == [0, 1]

    meanings_df = workspace.nodes[meanings_node_id].data.collect().sort("TOPIC_topic")
    assert meanings_df["TOPIC_topic"].to_list() == [0, 1]


@pytest.mark.anyio
async def test_topic_modeling_detach_with_meanings_override_replaces_meanings(
    authenticated_client, workspace_id, tmp_path
):
    """``topic_meanings_override`` lets the detach mirror what's on screen.

    Regression: without the override path, the meanings node always came
    from the fit-time parquet — so a post-fit "Words per topic" change
    or stopword filter toggle wasn't reflected in the detached node.
    """
    user_id = "test"
    workspace = workspace_manager.get_current_workspace(user_id)
    assert workspace is not None

    source_node = Node(
        data=pl.DataFrame(
            {
                "document": ["alpha beta", "gamma delta"],
                "source": ["a", "b"],
            }
        ).lazy(),
        name="topic_source_override",
        workspace=workspace,
        operation="test_setup",
        parents=[],
    )
    workspace.add_node(source_node)

    assignments_path = tmp_path / "assignments_override.parquet"
    pl.DataFrame(
        {"__row_nr__": [0, 1], "TOPIC_topic": [0, 0]},
        schema={"__row_nr__": pl.Int64, "TOPIC_topic": pl.Int64},
    ).write_parquet(assignments_path)

    # Artifact carries the fit-time words; the override should win.
    meanings_path = tmp_path / "topic_meanings_override.parquet"
    pl.DataFrame(
        {
            "TOPIC_topic": [0],
            "TOPIC_topic_meaning": [["original_one", "original_two", "original_three"]],
        },
        schema={
            "TOPIC_topic": pl.Int64,
            "TOPIC_topic_meaning": pl.List(pl.String),
        },
    ).write_parquet(meanings_path)

    task_id = "completed-topic-task-override"
    payload = {
        "topics": [
            {
                "id": 0,
                "label": "original_one | original_two | original_three",
                "representative_words": [
                    "original_one",
                    "original_two",
                    "original_three",
                ],
                "size": [2],
                "total_size": 2,
                "x": 0.0,
                "y": 0.0,
            }
        ],
        "corpus_sizes": [2],
        "artifacts": {
            "version": 1,
            "topic_meanings_parquet_path": str(meanings_path),
            "nodes": [
                {
                    "node_id": source_node.id,
                    "node_name": source_node.name,
                    "text_column": "document",
                    "original_columns": ["document", "source"],
                    "assignments_parquet_path": str(assignments_path),
                }
            ],
        },
    }
    get_task_manager(user_id).save_task(
        AnalysisTask(
            task_id=task_id,
            user_id=user_id,
            workspace_id=workspace_id,
            request=AnalysisTopicModelingRequest(
                node_ids=[source_node.id],
                node_columns={source_node.id: "document"},
                min_topic_size=5,
                random_seed=42,
                representative_words_count=3,
            ),
            status=AnalysisStatus.COMPLETED,
            result=GenericAnalysisResult(payload),
        )
    )

    detach_response = await authenticated_client.post(
        f"/api/workspaces/topic-modeling/tasks/{task_id}/detach",
        json={
            "node_ids": [source_node.id],
            "selected_columns": {source_node.id: ["document"]},
            "topic_meanings_override": [
                {"topic_id": 0, "words": ["visible_a", "visible_b"]},
            ],
        },
    )

    assert detach_response.status_code == 200, detach_response.text
    topic_meanings_node_id = (
        detach_response.json()
        .get("data", {})
        .get("detached_nodes", [{}])[0]
        .get("topic_meanings_node_id")
    )
    assert topic_meanings_node_id

    meanings_node = workspace.nodes[topic_meanings_node_id]
    meanings_df = meanings_node.data.collect()
    assert meanings_df["TOPIC_topic"].to_list() == [0]
    assert meanings_df["TOPIC_topic_meaning"].to_list() == [["visible_a", "visible_b"]]

    # Artifact parquet is untouched on disk — override writes a fresh file.
    artifact_meanings = pl.read_parquet(meanings_path)
    assert artifact_meanings["TOPIC_topic_meaning"].to_list() == [
        ["original_one", "original_two", "original_three"]
    ]
    override_files = list(
        meanings_path.parent.glob("topic_meanings_override_override_*.parquet")
    )
    assert len(override_files) == 1
