"""
Integration tests for analysis persistence.

Tests the end-to-end flow from API endpoints to file persistence.
"""

from datetime import datetime
from types import SimpleNamespace

import polars as pl
import pytest
from httpx import AsyncClient
from ldaca_wordflow.analysis.manager import get_task_manager
from ldaca_wordflow.analysis.results import GenericAnalysisResult
from ldaca_wordflow.api.workspaces.analyses.token_frequencies import (
    DEFAULT_TOKEN_LIMIT,
    MAX_SERVER_TOKEN_LIMIT,
    SERVER_LIMIT_MULTIPLIER,
)
from ldaca_wordflow.core.worker import token_frequencies_task
from ldaca_wordflow.core.workspace import workspace_manager


# Helper functions
async def post_json(client: AsyncClient, path: str, payload: dict):
    """Helper to POST JSON data and return response."""
    return await client.post(path, json=payload)


async def get_json(client: AsyncClient, path: str):
    """Helper to GET JSON data and return response."""
    return await client.get(path)


def assert_analysis_record_structure(record_dict: dict, expected_task: str):
    """Assert that a record dict has the expected structure."""
    required_keys = {"task", "saved_at", "request", "result"}
    assert set(record_dict.keys()) == required_keys

    assert record_dict["task"] == expected_task
    assert isinstance(record_dict["saved_at"], str)
    assert isinstance(record_dict["request"], dict)
    assert isinstance(record_dict["result"], dict)

    # Validate ISO 8601 timestamp format
    datetime.fromisoformat(record_dict["saved_at"])


def assert_successful_result(result_dict: dict):
    """Assert that a result dict represents a successful analysis."""
    # Contract migrated: 'success': True -> 'state': 'successful'
    assert result_dict.get("state") == "successful"
    # Some analyses return a generic data envelope, while others expose
    # analysis-specific fields at top-level.
    assert "data" in result_dict or "analysis_params" in result_dict


def _simulate_token_frequency_completion(workspace_id: str):
    """Run token frequencies synchronously and persist the result via TaskManager."""

    task_manager = get_task_manager("test")
    token_tasks = [
        task
        for task in task_manager.get_all_tasks()
        if getattr(task, "workspace_id", None) == workspace_id
    ]
    token_tasks.sort(key=lambda task: task.updated_at or task.created_at, reverse=True)
    assert token_tasks
    task = token_tasks[0]
    assert task is not None

    req = task.request.model_dump() if hasattr(task.request, "model_dump") else {}
    workspace = workspace_manager.get_current_workspace("test")
    assert workspace is not None

    node_ids = req.get("node_ids") or []
    node_columns = req.get("node_columns") or {}
    node_corpora: dict[str, list[str]] = {}
    node_display_names: dict[str, str] = {}
    for node_id in node_ids:
        node = workspace.nodes.get(node_id)
        assert node is not None
        node_data = getattr(node, "data", None)
        assert node_data is not None
        column_name = node_columns.get(node_id)
        assert column_name
        docs_df = node_data.select(column_name).collect()
        node_corpora[node_id] = [
            str(v) if v is not None else "" for v in docs_df[column_name].to_list()
        ]
        node_display_names[node_id] = str(getattr(node, "name", None) or node_id)

    artifacts_dir = workspace_manager.ensure_workspace_artifacts_dir(
        "test", workspace_id
    )
    assert artifacts_dir is not None
    worker_result = token_frequencies_task(
        user_id="test",
        workspace_id=workspace_id,
        node_corpora=node_corpora,
        node_display_names=node_display_names,
        artifact_dir=str(artifacts_dir),
        artifact_prefix=f"test_token_freq_{task.task_id}",
        token_limit=req.get("token_limit") or DEFAULT_TOKEN_LIMIT,
        stop_words=req.get("stop_words") or [],
        tokenizer_model=req.get("tokenizer_model") or "native:plain_words_en",
    )

    task.complete(GenericAnalysisResult(worker_result))
    task_manager.save_task(task)


def _list_analysis_records(user_id: str, workspace_id: str, task: str | None = None):
    task_manager = get_task_manager(user_id)
    tasks = [
        t
        for t in task_manager.get_all_tasks()
        if getattr(t, "workspace_id", None) == workspace_id
    ]
    if task:
        tasks = [t for t in tasks if _record_analysis_name(t) == task]
    tasks.sort(key=lambda t: t.updated_at or t.created_at)

    def _to_record(t):
        req = t.request.model_dump() if hasattr(t.request, "model_dump") else t.request
        res = t.result.to_json() if hasattr(t.result, "to_json") else t.result
        return SimpleNamespace(
            task=task,
            task_id=t.task_id,
            saved_at=(t.updated_at or t.created_at).isoformat(),
            request=req,
            result=res,
        )

    return [_to_record(t) for t in tasks]


def _record_analysis_name(task) -> str | None:
    """Return the analysis family name encoded by a persisted task request."""
    request_class = task.request.__class__.__name__
    if request_class == "AnalysisTokenFrequencyRequest":
        return "token_frequencies"
    if request_class == "ConcordanceAnalysisRequest":
        return "concordance_analysis"
    if request_class == "QuotationRequest":
        return "quotation_analysis"
    if request_class == "SequentialAnalysisRequest":
        return "sequential_analysis"
    if request_class == "TopicModelingRequest":
        return "topic_modeling"
    if request_class == "AIAnnotationRequest":
        return "ai_annotation"
    return None


def _task_id_from_response(response) -> str:
    """Extract the explicit task id returned by task-backed submit endpoints."""
    task_id = response.json().get("metadata", {}).get("task_id")
    assert task_id
    return task_id


@pytest.fixture(autouse=True)
def _stub_task_manager(monkeypatch):
    """Avoid spawning real worker processes in tests; mimic immediate task submission."""

    class ImmediateTaskManager:
        def __init__(self):
            self.counter = 0

        async def any_running(self, **_kwargs):  # pragma: no cover
            return False

        async def latest_by_type(self, *args, **_kwargs):  # pragma: no cover
            return None

        async def submit_task(self, **_kwargs):  # pragma: no cover
            self.counter += 1
            return SimpleNamespace(id=f"test-task-{self.counter}")

    immediate_task_manager = ImmediateTaskManager()

    def fake_get_task_manager(self, _user_id):
        return immediate_task_manager

    monkeypatch.setattr(
        workspace_manager.__class__, "get_task_manager", fake_get_task_manager
    )


@pytest.fixture(autouse=True)
def _reset_analysis_task_manager_state():
    task_manager = get_task_manager("test")
    task_manager.clear_all()
    yield
    task_manager.clear_all()


@pytest.mark.anyio
class TestTokenFrequencyPersistence:
    """Test token frequency analysis persistence."""

    async def test_token_frequency_creates_analysis_record(
        self, authenticated_client, workspace_id, tiny_node_id, test_user
    ):
        """Test that token frequency analysis creates a proper analysis record."""
        # Given: A workspace with a text node
        request_payload = {
            "node_ids": [tiny_node_id],
            "node_columns": {tiny_node_id: "document"},
            "tokenizer_model": "native:plain_words_en",
        }

        # When: We call the token frequencies endpoint
        response = await post_json(
            authenticated_client,
            "/api/workspaces/token-frequencies",
            request_payload,
        )

        # Then: The response starts a background task
        assert response.status_code == 200
        result_data = response.json()
        assert result_data.get("state") == "running"
        task_id = result_data.get("metadata", {}).get("task_id")
        assert task_id

        # Simulate worker completion and fetch persisted result
        _simulate_token_frequency_completion(workspace_id)

        result_resp = await get_json(
            authenticated_client,
            f"/api/workspaces/token-frequencies/tasks/{task_id}/result",
        )
        assert result_resp.status_code == 200
        final_result = result_resp.json()
        assert_successful_result(final_result)

        # And: An analysis record was persisted
        analyses = _list_analysis_records(test_user["id"], workspace_id)
        assert len(analyses) == 1

        record = analyses[0]
        assert record.task_id == task_id
        # Check that the core request parameters are preserved
        assert record.request["node_ids"] == request_payload["node_ids"]
        assert record.request["node_columns"] == request_payload["node_columns"]
        expected_limit = DEFAULT_TOKEN_LIMIT
        assert "limit" not in record.request
        assert record.request["token_limit"] == expected_limit
        assert record.request.get("stop_words") == []
        # Result is stored (may be wrapped by task manager); validate via endpoint contract
        assert final_result.get("token_limit") == expected_limit
        assert (
            final_result.get("analysis_params", {}).get("token_limit") == expected_limit
        )
        assert final_result.get("stop_words") == []
        assert final_result.get("metadata", {}).get("stop_words") == []
        assert final_result.get("analysis_params", {}).get("stop_words") == []

        # Validate timestamp
        saved_time = datetime.fromisoformat(record.saved_at)
        assert isinstance(saved_time, datetime)

        # Validate persisted artifact manifest structure
        assert "artifacts" in record.result
        assert isinstance(record.result["artifacts"], dict)
        assert isinstance(record.result["artifacts"].get("nodes"), list)

    async def test_token_frequency_preserves_independent_tab_tasks(
        self, authenticated_client, workspace_id, tiny_node_id, test_user
    ):
        """Token-frequency runs are independent task records, not one current task."""
        # Given: We run token frequency analysis twice from different tabs
        first_request = {
            "node_ids": [tiny_node_id],
            "node_columns": {tiny_node_id: "document"},
            "tokenizer_model": "native:plain_words_en",
        }

        second_request = {
            "node_ids": [tiny_node_id],
            "node_columns": {tiny_node_id: "document"},
            "stop_words": ["alpha", "beta"],
            "tokenizer_model": "native:plain_words_en",
        }

        # When: We call the endpoint twice
        first_resp = await post_json(
            authenticated_client,
            "/api/workspaces/token-frequencies",
            first_request,
        )
        assert first_resp.status_code == 200
        first_task_id = first_resp.json().get("metadata", {}).get("task_id")
        assert first_task_id

        second_resp = await post_json(
            authenticated_client,
            "/api/workspaces/token-frequencies",
            second_request,
        )
        assert second_resp.status_code == 200
        second_task_id = second_resp.json().get("metadata", {}).get("task_id")
        assert second_task_id
        assert second_task_id != first_task_id

        # Then: both analysis records exist independently
        analyses = _list_analysis_records(test_user["id"], workspace_id)
        assert len(analyses) == 2

        by_task_id = {record.task_id: record for record in analyses}
        assert by_task_id[first_task_id].request.get("stop_words") == []
        assert by_task_id[second_task_id].request.get("stop_words") == ["alpha", "beta"]

        # And: the first task remains addressable by explicit id after the second run.
        first_request_resp = await get_json(
            authenticated_client,
            f"/api/workspaces/token-frequencies/tasks/{first_task_id}/request",
        )
        assert first_request_resp.status_code == 200
        assert "tab_id" not in first_request_resp.json()

    async def test_current_result_update_persists_preferences(
        self, authenticated_client, workspace_id, tiny_node_id, test_user
    ):
        """Updating current-result should synchronize presentation preferences."""

        initial_request = {
            "node_ids": [tiny_node_id],
            "node_columns": {tiny_node_id: "document"},
            "stop_words": ["the", "and"],
            "tokenizer_model": "native:plain_words_en",
        }

        initial_response = await post_json(
            authenticated_client,
            "/api/workspaces/token-frequencies",
            initial_request,
        )
        assert initial_response.status_code == 200
        assert initial_response.json().get("state") == "running"
        task_id = _task_id_from_response(initial_response)

        update_payload = {"token_limit": 30, "stop_words": ["alpha", "beta"]}
        update_response = await post_json(
            authenticated_client,
            f"/api/workspaces/token-frequencies/tasks/{task_id}/result",
            update_payload,
        )
        assert update_response.status_code == 200
        update_json = update_response.json()
        assert update_json == {"state": "successful", "message": "saved"}

        _simulate_token_frequency_completion(workspace_id)

        current_result_response = await get_json(
            authenticated_client,
            f"/api/workspaces/token-frequencies/tasks/{task_id}/result",
        )
        assert current_result_response.status_code == 200
        updated_result = current_result_response.json()
        assert updated_result["token_limit"] == 30
        assert updated_result.get("stop_words") == ["alpha", "beta"]
        assert updated_result.get("metadata", {}).get("stop_words") == [
            "alpha",
            "beta",
        ]
        assert updated_result.get("analysis_params", {}).get("stop_words") == [
            "alpha",
            "beta",
        ]
        expected_server_limit = min(
            max(30 * SERVER_LIMIT_MULTIPLIER, DEFAULT_TOKEN_LIMIT),
            MAX_SERVER_TOKEN_LIMIT,
        )
        assert (
            updated_result.get("metadata", {}).get("server_limit")
            == expected_server_limit
        )

        analyses = _list_analysis_records(test_user["id"], workspace_id)
        assert len(analyses) == 1
        record = analyses[0]
        assert record.request["token_limit"] == 30
        assert "limit" not in record.request
        assert record.request.get("stop_words") == ["alpha", "beta"]
        assert "artifacts" in record.result

        clear_response = await post_json(
            authenticated_client,
            f"/api/workspaces/token-frequencies/tasks/{task_id}/result",
            {"stop_words": []},
        )
        assert clear_response.status_code == 200

        clear_json = clear_response.json()
        assert clear_json == {"state": "successful", "message": "saved"}

        analyses = _list_analysis_records(test_user["id"], workspace_id)
        assert len(analyses) == 1
        record = analyses[0]
        assert record.request["token_limit"] == 30  # limit unchanged
        assert record.request.get("stop_words") == []
        current_result = (
            await get_json(
                authenticated_client,
                f"/api/workspaces/token-frequencies/tasks/{task_id}/result",
            )
        ).json()
        assert current_result.get("token_limit") == 30
        assert current_result.get("stop_words") == []


@pytest.mark.anyio
class TestSequentialAnalysisPersistence:
    """Test sequential analysis persistence and presentation preferences."""

    async def _run_sequential_analysis(
        self,
        client: AsyncClient,
        workspace_id: str,
        node_id: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> dict:
        from ldaca_wordflow.api.workspaces.analyses import (
            sequential_analysis as sequential_module,
        )

        request_payload = {
            "time_column": "published_at",
            "group_by_columns": ["category"],
            "frequency": "daily",
            "sort_by_time": True,
        }

        dummy_df = pl.DataFrame(
            {
                "published_at": [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2),
                ],
                "category": ["alpha", "alpha", "beta"],
            }
        )
        dummy_node = SimpleNamespace(data=dummy_df.lazy())
        dummy_workspace = SimpleNamespace(nodes={node_id: dummy_node})

        monkeypatch.setattr(
            sequential_module.workspace_manager,
            "get_current_workspace_id",
            lambda *_args, **_kwargs: workspace_id,
        )
        monkeypatch.setattr(
            sequential_module.workspace_manager,
            "get_current_workspace",
            lambda *_args, **_kwargs: dummy_workspace,
        )

        response = await post_json(
            client,
            f"/api/workspaces/nodes/{node_id}/sequential-analysis",
            request_payload,
        )

        assert response.status_code == 200
        result_data = response.json()
        assert_successful_result(result_data)
        return result_data

    async def test_sequential_analysis_includes_chart_type(
        self,
        authenticated_client,
        workspace_id,
        timeline_node_id,
        test_user,
        monkeypatch,
    ):
        """Sequential analysis responses should include a default chart type."""

        result_data = await self._run_sequential_analysis(
            authenticated_client, workspace_id, timeline_node_id, monkeypatch
        )

        assert result_data.get("chart_type") == "line"

        analyses = _list_analysis_records(test_user["id"], workspace_id)
        assert len(analyses) == 1
        record = analyses[0]
        task_id = result_data.get("metadata", {}).get("task_id")
        assert task_id
        assert record.task_id == task_id
        assert record.result.get("chart_type") == "line"

        current_result_response = await post_json(
            authenticated_client,
            f"/api/workspaces/sequential-analysis/tasks/{task_id}/result",
            {},
        )
        assert current_result_response.status_code == 200
        current_payload = current_result_response.json()
        assert current_payload["data"]["chart_type"] == "line"

    async def test_sequential_analysis_chart_type_update_persists(
        self,
        authenticated_client,
        workspace_id,
        timeline_node_id,
        test_user,
        monkeypatch,
    ):
        """Updating the chart type should persist via current-result endpoint."""

        result_data = await self._run_sequential_analysis(
            authenticated_client, workspace_id, timeline_node_id, monkeypatch
        )

        task_id = result_data.get("metadata", {}).get("task_id")
        assert task_id
        update_response = await post_json(
            authenticated_client,
            f"/api/workspaces/sequential-analysis/tasks/{task_id}/result",
            {"chart_type": "bar"},
        )
        assert update_response.status_code == 200
        update_json = update_response.json()
        assert update_json == {
            "state": "successful",
            "message": "saved",
            "data": {"chart_type": "bar"},
        }

        current_result_response = await post_json(
            authenticated_client,
            f"/api/workspaces/sequential-analysis/tasks/{task_id}/result",
            {},
        )
        assert current_result_response.status_code == 200
        current_payload = current_result_response.json()
        assert current_payload["data"]["chart_type"] == "bar"

        analyses = _list_analysis_records(test_user["id"], workspace_id)
        assert len(analyses) == 1
        record = analyses[0]
        assert record.result.get("chart_type") == "bar"

    async def test_sequential_analysis_rejects_invalid_chart_type(
        self,
        authenticated_client,
        workspace_id,
        timeline_node_id,
        monkeypatch,
    ):
        """Invalid chart types should be rejected with clear feedback."""

        result_data = await self._run_sequential_analysis(
            authenticated_client, workspace_id, timeline_node_id, monkeypatch
        )

        task_id = result_data.get("metadata", {}).get("task_id")
        assert task_id
        invalid_response = await post_json(
            authenticated_client,
            f"/api/workspaces/sequential-analysis/tasks/{task_id}/result",
            {"chart_type": "scatter"},
        )
        assert invalid_response.status_code == 400
        error_payload = invalid_response.json()
        assert "Invalid chart type" in error_payload["detail"]

    async def test_sequential_analysis_numeric_params(
        self,
        authenticated_client,
        workspace_id,
        timeline_node_id,
        monkeypatch,
    ):
        """Numeric sequential analysis should persist origin/interval inputs."""
        captured_kwargs: dict[str, object] = {}

        dummy_df = pl.DataFrame({"score": [0, 5, 10, 15]})
        dummy_node = SimpleNamespace(data=dummy_df.lazy())
        dummy_workspace = SimpleNamespace(nodes={timeline_node_id: dummy_node})

        from ldaca_wordflow.api.workspaces.analyses import (
            sequential_analysis as sequential_module,
        )

        monkeypatch.setattr(
            sequential_module.workspace_manager,
            "get_current_workspace_id",
            lambda *_args, **_kwargs: workspace_id,
        )
        monkeypatch.setattr(
            sequential_module.workspace_manager,
            "get_current_workspace",
            lambda *_args, **_kwargs: dummy_workspace,
        )

        original_run = sequential_module._run_sequential_analysis

        def _capture_run(*_args, **kwargs):
            captured_kwargs.update(kwargs)
            return original_run(*_args, **kwargs)

        monkeypatch.setattr(sequential_module, "_run_sequential_analysis", _capture_run)

        payload = {
            "time_column": "score",
            "column_type": "numeric",
            "numeric_origin": 0,
            "numeric_interval": 5,
            "sort_by_time": True,
        }

        response = await post_json(
            authenticated_client,
            f"/api/workspaces/nodes/{timeline_node_id}/sequential-analysis",
            payload,
        )

        assert response.status_code == 200
        assert captured_kwargs.get("column_type") == "numeric"
        assert captured_kwargs.get("numeric_interval") == 5

    async def test_sequential_analysis_numeric_requires_interval(
        self,
        authenticated_client,
        workspace_id,
        timeline_node_id,
        monkeypatch,
    ):
        """Missing numeric interval inputs should raise a validation error."""

        dummy_df = pl.DataFrame({"score": [0, 5, 10]})
        dummy_node = SimpleNamespace(data=dummy_df.lazy())
        dummy_workspace = SimpleNamespace(nodes={timeline_node_id: dummy_node})

        from ldaca_wordflow.api.workspaces.analyses import (
            sequential_analysis as sequential_module,
        )

        monkeypatch.setattr(
            sequential_module.workspace_manager,
            "get_current_workspace_id",
            lambda *_args, **_kwargs: workspace_id,
        )
        monkeypatch.setattr(
            sequential_module.workspace_manager,
            "get_current_workspace",
            lambda *_args, **_kwargs: dummy_workspace,
        )

        payload = {
            "time_column": "score",
            "column_type": "numeric",
            "sort_by_time": True,
        }

        response = await post_json(
            authenticated_client,
            f"/api/workspaces/nodes/{timeline_node_id}/sequential-analysis",
            payload,
        )

        assert response.status_code == 422
        detail = response.json().get("detail", "")
        if isinstance(detail, list):
            detail_text = " ".join(str(item) for item in detail)
        else:
            detail_text = str(detail)
        assert "interval" in detail_text.lower()

    async def test_sequential_analysis_detach_creates_filtered_node(
        self,
        authenticated_client,
        workspace_id,
        timeline_node_id,
        monkeypatch,
        tmp_path,
    ):
        """Selected sequential periods should detach into a filtered child node."""
        from ldaca_wordflow.api.workspaces.analyses import (
            sequential_analysis as sequential_module,
        )

        source_df = pl.DataFrame(
            {
                "published_at": [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2),
                ],
                "category": ["alpha", "alpha", "beta"],
                "value": [1, 2, 3],
            }
        )
        source_node = SimpleNamespace(
            data=source_df.lazy(),
            name="timeline",
            id=timeline_node_id,
        )

        class DummyWorkspace:
            def __init__(self):
                self.nodes = {timeline_node_id: source_node}
                self.added_nodes = []

            def add_node(self, node):
                self.added_nodes.append(node)
                self.nodes[node.id] = node

            def place_node_after_parent(self, _node):
                return None

        dummy_workspace = DummyWorkspace()

        monkeypatch.setattr(
            sequential_module.workspace_manager,
            "get_current_workspace_id",
            lambda *_args, **_kwargs: workspace_id,
        )
        monkeypatch.setattr(
            sequential_module.workspace_manager,
            "get_current_workspace",
            lambda *_args, **_kwargs: dummy_workspace,
        )
        monkeypatch.setattr(
            sequential_module, "update_workspace", lambda *_args, **_kwargs: None
        )

        class DummyDetachedNode:
            def __init__(self, **kwargs):
                self.id = "detached-node"
                self.data = kwargs["data"]
                self.name = kwargs["name"]
                self.workspace = kwargs.get("workspace")
                self.operation = kwargs.get("operation")
                self.parents = kwargs.get("parents", [])

        monkeypatch.setattr(sequential_module, "Node", DummyDetachedNode)

        result_data = await self._run_sequential_analysis(
            authenticated_client,
            workspace_id,
            timeline_node_id,
            monkeypatch,
        )
        monkeypatch.setattr(
            sequential_module.workspace_manager,
            "get_current_workspace_id",
            lambda *_args, **_kwargs: workspace_id,
        )
        monkeypatch.setattr(
            sequential_module.workspace_manager,
            "get_current_workspace",
            lambda *_args, **_kwargs: dummy_workspace,
        )
        task_id = result_data.get("metadata", {}).get("task_id")
        assert task_id

        selected_period = result_data["data"][0]
        response = await post_json(
            authenticated_client,
            f"/api/workspaces/sequential-analysis/tasks/{task_id}/detach",
            {
                "selected_periods": [
                    {
                        "period_start": selected_period["period_start"],
                        "period_end": selected_period["period_end"],
                    }
                ],
                "new_node_name": "timeline_trend",
            },
        )

        assert response.status_code == 200
        assert response.json() == {
            "new_node_id": "detached-node",
            "new_node_name": "timeline_trend",
        }
        assert len(dummy_workspace.added_nodes) == 1

        detached_df = dummy_workspace.added_nodes[0].data.collect().sort("value")
        assert detached_df["value"].to_list() == [1, 2]

    async def test_sequential_analysis_detach_respects_visible_groups(
        self,
        authenticated_client,
        workspace_id,
        timeline_node_id,
        monkeypatch,
    ):
        """Hidden legend groups should be excluded from detached sequential rows."""
        from ldaca_wordflow.api.workspaces.analyses import (
            sequential_analysis as sequential_module,
        )

        source_df = pl.DataFrame(
            {
                "published_at": [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2),
                ],
                "category": ["alpha", "beta", "alpha"],
                "value": [1, 2, 3],
            }
        )
        source_node = SimpleNamespace(
            data=source_df.lazy(),
            name="timeline",
            id=timeline_node_id,
        )

        class DummyWorkspace:
            def __init__(self):
                self.nodes = {timeline_node_id: source_node}
                self.added_nodes = []

            def add_node(self, node):
                self.added_nodes.append(node)
                self.nodes[node.id] = node

            def place_node_after_parent(self, _node):
                return None

        dummy_workspace = DummyWorkspace()

        monkeypatch.setattr(
            sequential_module.workspace_manager,
            "get_current_workspace_id",
            lambda *_args, **_kwargs: workspace_id,
        )
        monkeypatch.setattr(
            sequential_module.workspace_manager,
            "get_current_workspace",
            lambda *_args, **_kwargs: dummy_workspace,
        )
        monkeypatch.setattr(
            sequential_module, "update_workspace", lambda *_args, **_kwargs: None
        )

        class DummyDetachedNode:
            def __init__(self, **kwargs):
                self.id = "detached-node-groups"
                self.data = kwargs["data"]
                self.name = kwargs["name"]
                self.workspace = kwargs.get("workspace")
                self.operation = kwargs.get("operation")
                self.parents = kwargs.get("parents", [])

        monkeypatch.setattr(sequential_module, "Node", DummyDetachedNode)

        response = await post_json(
            authenticated_client,
            f"/api/workspaces/nodes/{timeline_node_id}/sequential-analysis",
            {
                "time_column": "published_at",
                "group_by_columns": ["category"],
                "frequency": "daily",
                "sort_by_time": True,
            },
        )
        assert response.status_code == 200

        task_id = response.json().get("metadata", {}).get("task_id")
        assert task_id

        detach_response = await post_json(
            authenticated_client,
            f"/api/workspaces/sequential-analysis/tasks/{task_id}/detach",
            {
                "selected_periods": [
                    {
                        "period_start": "2024-01-01T00:00:00",
                        "period_end": "2024-01-01T00:00:00",
                    }
                ],
                "visible_groups": [
                    {
                        "values": {
                            "category": "alpha",
                        }
                    }
                ],
                "new_node_name": "timeline_trend",
            },
        )

        assert detach_response.status_code == 200
        assert len(dummy_workspace.added_nodes) == 1

        detached_df = dummy_workspace.added_nodes[0].data.collect().sort("value")
        assert detached_df["value"].to_list() == [1]


@pytest.mark.anyio
class TestWorkspaceGraphEnrichment:
    """Test workspace graph enrichment with analysis data."""

    async def test_graph_surfaces_tokenizer_models_on_tokenised_node(
        self, authenticated_client, workspace_id, tiny_node_id, test_user
    ):
        """After Tokenise, ``/graph`` must surface ``tokenizer_models`` per-node.

        The frontend reads this light-weight column-to-model map to restore
        tokenizer selectors and tokens-mode availability after tab navigation.
        """
        # Given: a tokenised node. Register tokenization directly on
        # the in-memory node so we don't round-trip through plbin (polars
        # FFI plugin plans don't serialise cleanly across every polars
        # version, and this test only cares about the graph payload shape).
        from ldaca_wordflow.api.workspaces.analyses.generated_columns import (
            tokenization_column_name,
        )

        workspace = workspace_manager.get_current_workspace(test_user["id"])
        assert workspace is not None
        node = workspace.nodes[tiny_node_id]
        tokenization_name = tokenization_column_name(
            "document", "huggingface:bert-base-uncased"
        )
        node.register_tokenization(
            "document",
            {
                "column_name": tokenization_name,
                "model": "huggingface:bert-base-uncased",
                "language": "en",
                "params": {"lowercase": True, "remove_punct": True},
            },
        )

        # When: we read the graph payload.
        response = await get_json(authenticated_client, "/api/workspaces/graph")
        assert response.status_code == 200
        graph_data = response.json()

        # Then: the tokenised node carries `tokenizer_models` metadata.
        node_entry = next(
            (n for n in graph_data["nodes"] if n.get("id") == tiny_node_id), None
        )
        assert node_entry is not None
        assert "tokenizer_models" in node_entry
        tokenizer_models = node_entry["tokenizer_models"]
        assert len(tokenizer_models) == 1
        assert tokenizer_models["document"] == "huggingface:bert-base-uncased"


@pytest.mark.anyio
class TestAnalysisPersistenceEdgeCases:
    """Test edge cases and error conditions."""

    async def test_persistence_survives_analysis_errors(
        self, authenticated_client, workspace_id, test_user
    ):
        """Test that failed analyses don't corrupt the persistence system."""
        # Given: A request that will fail (invalid node)
        invalid_request = {
            "node_ids": ["invalid_node"],
            "node_columns": {"invalid_node": "document"},
        }

        # When: We call the endpoint with invalid data
        with pytest.raises(KeyError):
            await post_json(
                authenticated_client,
                "/api/workspaces/token-frequencies",
                invalid_request,
            )

        # And: No analysis records were created
        analyses = _list_analysis_records(test_user["id"], workspace_id)
        assert len(analyses) == 0

    async def test_multiple_workspaces_isolated(
        self, authenticated_client, test_user, tiny_text_file
    ):
        """Test that analyses in different workspaces are isolated."""
        # Given: Two different workspaces
        ws1_response = await post_json(
            authenticated_client, "/api/workspaces/", {"name": "workspace_1"}
        )
        ws1_id = ws1_response.json()["id"]

        ws2_response = await post_json(
            authenticated_client, "/api/workspaces/", {"name": "workspace_2"}
        )
        ws2_id = ws2_response.json()["id"]

        # Explicitly switch active workspace to ws1 before adding/running ws1 analysis
        switch_ws1 = await authenticated_client.post(
            "/api/workspaces/current", params={"workspace_id": ws1_id}
        )
        assert switch_ws1.status_code == 200

        # Add nodes to both workspaces
        node1_response = await authenticated_client.post(
            "/api/workspaces/nodes", params={"filename": tiny_text_file.name}
        )
        node1_id = node1_response.json()["id"]  # Changed from node_id to id

        # Switch active workspace to ws2 before adding/running ws2 analysis
        switch_ws2 = await authenticated_client.post(
            "/api/workspaces/current", params={"workspace_id": ws2_id}
        )
        assert switch_ws2.status_code == 200

        node2_response = await authenticated_client.post(
            "/api/workspaces/nodes", params={"filename": tiny_text_file.name}
        )
        node2_id = node2_response.json()["id"]  # Changed from node_id to id

        # When: We run analyses in both workspaces
        ws1_payload = {
            "node_ids": [node1_id],
            "node_columns": {node1_id: "document"},
            "stop_words": ["alpha"],
            "tokenizer_model": "native:plain_words_en",
        }

        ws2_payload = {
            "node_ids": [node2_id],
            "node_columns": {node2_id: "document"},
            "stop_words": ["beta"],
            "tokenizer_model": "native:plain_words_en",
        }

        # Switch back to ws1 before submitting ws1 analysis
        switch_ws1_again = await authenticated_client.post(
            "/api/workspaces/current", params={"workspace_id": ws1_id}
        )
        assert switch_ws1_again.status_code == 200

        ws1_response_payload = await post_json(
            authenticated_client,
            "/api/workspaces/token-frequencies",
            ws1_payload,
        )
        assert ws1_response_payload.status_code == 200
        ws1_result = ws1_response_payload.json()
        assert ws1_result.get("state") == "running"
        assert ws1_result.get("metadata", {}).get("task_id")

        # Switch to ws2 before submitting ws2 analysis
        switch_ws2_again = await authenticated_client.post(
            "/api/workspaces/current", params={"workspace_id": ws2_id}
        )
        assert switch_ws2_again.status_code == 200

        ws2_response_payload = await post_json(
            authenticated_client,
            "/api/workspaces/token-frequencies",
            ws2_payload,
        )
        assert ws2_response_payload.status_code == 200
        ws2_result = ws2_response_payload.json()
        assert ws2_result.get("state") == "running"
        assert ws2_result.get("metadata", {}).get("task_id")

        # Then: Under single-active-workspace semantics, only the currently active
        # workspace analysis remains in the per-user in-memory task manager.
        ws1_analyses = _list_analysis_records(test_user["id"], ws1_id)
        ws2_analyses = _list_analysis_records(test_user["id"], ws2_id)

        assert len(ws1_analyses) == 0
        assert len(ws2_analyses) == 1

        # And: The active workspace analysis carries the expected request data
        assert ws2_analyses[0].request["token_limit"] == DEFAULT_TOKEN_LIMIT
        assert ws2_analyses[0].request.get("stop_words") == ["beta"]
