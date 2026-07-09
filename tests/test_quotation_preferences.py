from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest
from ldaca_wordflow.analysis.implementations.quotation import QuotationRequest
from ldaca_wordflow.analysis.manager import get_task_manager
from ldaca_wordflow.analysis.results import GenericAnalysisResult
from ldaca_wordflow.api.workspaces.analyses.generated_columns import (
    QUOTE_COLUMN_NAMES,
)
from ldaca_wordflow.core.workspace import workspace_manager
from ldaca_wordflow.models import QuotationEngineConfig

USER_ID = "test"
WORKSPACE_ID = "00000000-0000-0000-0000-000000000301"
TASK = "quotation"


def _prime_workspace_state():
    """Prime workspace state for TaskManager-backed tests."""
    base_df = pl.DataFrame({"text": ["alpha doc", "beta doc"]}).lazy()

    class DummyWorkspace:
        def __init__(self, df):
            self._df = df
            self.nodes = {
                "node-1": SimpleNamespace(id="node-1", name="node-1", data=self._df)
            }
            self.metadata = {}

        def get_node(self, node_id):
            return self.nodes.get(
                node_id,
                SimpleNamespace(id=node_id, name=node_id, data=self._df),
            )

        def set_metadata(self, key, value):
            self.metadata[key] = value

    dummy_ws = DummyWorkspace(base_df)
    workspace_manager._current[USER_ID] = {
        "wid": WORKSPACE_ID,
        "workspace": dummy_ws,
        "path": None,
    }


def _cleanup_workspace_state():
    workspace_manager._current.pop(USER_ID, None)
    task_manager = get_task_manager(USER_ID)
    task_manager.clear_all()


def _seed_paginated_analysis(rows: list[dict[str, Any]], context_length: int = 15):
    _prime_workspace_state()
    task_manager = get_task_manager(USER_ID)
    request = QuotationRequest(node_id="node-1", column="text")
    task_id = task_manager.create_task(request)
    task = task_manager.get_task(task_id)
    assert task is not None

    result_dict = {
        "data": [[rows[0]]] if rows else [],
        "columns": list(rows[0].keys()) if rows else [],
        "metadata": {
            "quotation_columns": list(rows[0].keys()) if rows else [],
            "metadata_columns": [],
            "all_columns": list(rows[0].keys()) if rows else [],
        },
        "pagination": {
            "page": 1,
            "page_size": 1,
            "total_source_rows": len(rows),
            "total_source_pages": max(1, len(rows)),
            "result_count": 1 if rows else 0,
            "has_next": len(rows) > 1,
            "has_prev": False,
        },
        "sorting": {"sort_by": None, "descending": True},
        "preferences": {"context_length": context_length},
    }

    result = GenericAnalysisResult(result_dict)
    task.complete(result)
    task_manager.save_task(task)
    return task_id


def _fake_compute_for_quotes(quote_rows_by_marker: dict[str, list[dict[str, Any]]]):
    async def fake_compute(
        node,
        base_df,
        column,
        engine,
        *,
        use_base_only=False,
        **_kwargs,
    ):
        grouped_quotes = []
        for text in base_df.get_column(column).to_list():
            grouped_quotes.append(
                next(
                    (
                        quote_rows
                        for marker, quote_rows in quote_rows_by_marker.items()
                        if marker in text
                    ),
                    [],
                )
            )
        return base_df.with_columns(pl.Series("quotation", grouped_quotes))

    return fake_compute


def _assert_quotation_page_payload(payload: dict[str, Any], page: int, quote: str):
    assert payload["pagination"]["page"] == page
    assert payload["metadata"]["quotation_columns"] == list(QUOTE_COLUMN_NAMES)
    assert payload["metadata"]["metadata_columns"] == ["text"]
    assert payload["metadata"]["all_columns"] == ["text", *QUOTE_COLUMN_NAMES]
    assert payload["data"][0][0]["QUOTE_quote"] == quote


@pytest.fixture
def seeded_paginated_quotation():
    rows = [
        {"quote": "alpha"},
        {"quote": "beta"},
    ]
    task_id = _seed_paginated_analysis(rows)
    yield task_id
    _cleanup_workspace_state()


@pytest.fixture
def seeded_quotation_analysis():
    _prime_workspace_state()
    task_manager = get_task_manager(USER_ID)
    request = QuotationRequest(node_id="node-1", column="text")
    task_id = task_manager.create_task(request)
    task = task_manager.get_task(task_id)
    assert task is not None

    result = GenericAnalysisResult(
        {
            "data": [],
            "columns": [],
            "metadata": {
                "quotation_columns": [],
                "metadata_columns": [],
                "all_columns": [],
            },
            "preferences": {"context_length": 15},
        }
    )
    task.complete(result)
    task_manager.save_task(task)

    yield task_id
    _cleanup_workspace_state()


@pytest.mark.asyncio
async def test_update_context_length_persists_preference(
    authenticated_client, seeded_quotation_analysis
):
    task_id = seeded_quotation_analysis
    response = await authenticated_client.patch(
        f"/api/workspaces/{WORKSPACE_ID}/analysis-tasks/{task_id}/preferences",
        json={"context_length": 42},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["data"]["context_length"] == 42

    task_manager = get_task_manager(USER_ID)
    task = task_manager.get_task(task_id)
    assert task is not None
    assert task.result is not None
    assert task.result.data["preferences"]["context_length"] == 42


@pytest.mark.asyncio
async def test_update_context_length_clamps_bounds(
    authenticated_client, seeded_quotation_analysis
):
    task_id = seeded_quotation_analysis

    high_response = await authenticated_client.patch(
        f"/api/workspaces/{WORKSPACE_ID}/analysis-tasks/{task_id}/preferences",
        json={"context_length": 99999},
    )
    assert high_response.status_code == 200
    assert high_response.json()["data"]["context_length"] == 2000

    low_response = await authenticated_client.patch(
        f"/api/workspaces/{WORKSPACE_ID}/analysis-tasks/{task_id}/preferences",
        json={"context_length": -5},
    )
    assert low_response.status_code == 200
    assert low_response.json()["data"]["context_length"] == 0

    task = get_task_manager(USER_ID).get_task(task_id)
    assert task is not None
    assert task.result is not None
    result = task.result.to_json()
    assert isinstance(result, dict)
    assert result["preferences"]["context_length"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "request_kwargs"),
    [
        ("get", {"params": {"page": 2, "page_size": 1}}),
        ("post", {"json": {"page": 2, "page_size": 1}}),
    ],
)
async def test_quotation_current_result_respects_page_params(
    authenticated_client,
    seeded_paginated_quotation,
    monkeypatch,
    method,
    request_kwargs,
):
    task_id = seeded_paginated_quotation
    monkeypatch.setattr(
        "ldaca_wordflow.api.workspaces.analyses.quotation_core.compute_quote_dataframe",
        _fake_compute_for_quotes(
            {"alpha": [{"quote": "alpha"}], "beta": [{"quote": "beta"}]}
        ),
    )
    url = (
        f"/api/workspaces/{WORKSPACE_ID}/analysis-tasks/{task_id}/result"
        if method == "get"
        else f"/api/workspaces/{WORKSPACE_ID}/analysis-tasks/{task_id}/result-query"
    )
    response = await getattr(authenticated_client, method)(url, **request_kwargs)
    assert response.status_code == 200
    _assert_quotation_page_payload(response.json(), page=2, quote="beta")


@pytest.mark.asyncio
async def test_quotation_current_result_returns_all_quotes_for_document_page(
    authenticated_client, seeded_paginated_quotation, monkeypatch
):
    task_id = seeded_paginated_quotation
    # Page size is in documents; all quotes within the selected docs should be returned.

    monkeypatch.setattr(
        "ldaca_wordflow.api.workspaces.analyses.quotation_core.compute_quote_dataframe",
        _fake_compute_for_quotes(
            {
                "alpha": [
                    {"quote": "alpha-1", "quote_row_idx": 0},
                    {"quote": "alpha-2", "quote_row_idx": 1},
                ],
                "beta": [{"quote": "beta-1", "quote_row_idx": 0}],
            }
        ),
    )

    response = await authenticated_client.get(
        f"/api/workspaces/{WORKSPACE_ID}/analysis-tasks/{task_id}/result",
        params={"page": 1, "page_size": 1},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["pagination"]["page"] == 1
    assert payload["pagination"]["page_size"] == 1
    assert payload["metadata"]["quotation_columns"] == list(QUOTE_COLUMN_NAMES)
    assert payload["metadata"]["metadata_columns"] == ["text"]
    assert payload["metadata"]["all_columns"] == ["text", *QUOTE_COLUMN_NAMES]
    assert [row["QUOTE_quote"] for row in payload["data"][0]] == ["alpha-1", "alpha-2"]


@pytest.mark.asyncio
async def test_quotation_endpoint_submits_without_recomputing_on_demand(
    authenticated_client, monkeypatch, seeded_paginated_quotation, tmp_path
):
    recompute_called = False

    async def fake_compute(
        node,
        base_df_slice,
        column,
        engine,
        *,
        use_base_only=False,
        **_kwargs,
    ):
        nonlocal recompute_called
        recompute_called = True
        return await _fake_compute_for_quotes(
            {"alpha": [{"quote": "alpha"}], "beta": [{"quote": "beta"}]}
        )(node, base_df_slice, column, engine, use_base_only=use_base_only, **_kwargs)

    monkeypatch.setattr(
        "ldaca_wordflow.api.workspaces.analyses.quotation_core.compute_quote_dataframe",
        fake_compute,
    )
    monkeypatch.setattr(
        workspace_manager,
        "get_workspace_dir",
        lambda *_args, **_kwargs: tmp_path,
    )

    class ImmediateTaskManager:
        async def submit_task(self, **kwargs):
            return SimpleNamespace(id=kwargs["task_id"])

    monkeypatch.setattr(
        workspace_manager,
        "get_task_manager",
        lambda *_args, **_kwargs: ImmediateTaskManager(),
    )

    response = await authenticated_client.post(
        f"/api/workspaces/{WORKSPACE_ID}/nodes/node-1/quotation",
        json={"column": "text", "page": 2, "page_size": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "running"
    assert payload.get("metadata", {}).get("task_id")
    assert recompute_called is False
