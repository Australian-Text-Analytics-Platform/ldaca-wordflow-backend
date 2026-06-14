import polars as pl
import pytest
from ldaca_wordflow.core.workspace import workspace_manager

from docworkspace import Node


@pytest.mark.anyio
async def test_concordance_detach_starts_task(authenticated_client, workspace_id):
    """Ensure detaching concordance starts a background task."""
    df = pl.DataFrame({"text": ["alpha beta", "beta gamma", "alpha gamma"]})
    workspace = workspace_manager.get_current_workspace("test")
    assert workspace is not None

    node = Node(
        data=df.lazy(),
        name="text_node",
        workspace=workspace,
        operation="test_add",
        parents=[],
    )
    workspace.add_node(node)
    assert node is not None

    # Act: call detach endpoint
    resp = await authenticated_client.post(
        f"/api/workspaces/nodes/{node.id}/concordance/detach",
        json={
            "node_id": node.id,
            "column": "text",
            "search_word": "alpha",
            "num_left_tokens": 2,
            "num_right_tokens": 2,
            "regex": False,
            "case_sensitive": False,
        },
    )

    # Assert
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload.get("state") == "running"
    assert payload.get("metadata", {}).get("task_id")


@pytest.mark.anyio
async def test_concordance_detach_accepts_generated_columns(
    authenticated_client, workspace_id
):
    """Selecting CONC_* generated columns must not be projected off the source.

    Regression: the dialog now default-selects the generated concordance
    columns (CONC_matched_text, CONC_l1, ...). Those columns are produced by
    the detach worker and do not exist in the source node, so the endpoint
    must route them to ``selected_generated_columns`` instead of trying to
    ``select`` them off ``node_data`` (which raised ColumnNotFoundError -> 500).
    """
    df = pl.DataFrame(
        {
            "text": ["alpha beta", "beta gamma", "alpha gamma"],
            "speaker": ["a", "b", "c"],
        }
    )
    workspace = workspace_manager.get_current_workspace("test")
    assert workspace is not None

    node = Node(
        data=df.lazy(),
        name="text_node",
        workspace=workspace,
        operation="test_add",
        parents=[],
    )
    workspace.add_node(node)

    resp = await authenticated_client.post(
        f"/api/workspaces/nodes/{node.id}/concordance/detach",
        json={
            "node_id": node.id,
            "column": "text",
            "search_word": "alpha",
            "num_left_tokens": 2,
            "num_right_tokens": 2,
            "regex": False,
            "case_sensitive": False,
            # Mix of generated CONC_* columns, the extraction column, and a
            # real source metadata column — the shape the dialog now sends.
            "selected_columns": [
                "CONC_left_context",
                "CONC_matched_text",
                "CONC_right_context",
                "CONC_start_idx",
                "CONC_end_idx",
                "CONC_l1",
                "CONC_r1",
                "CONC_extraction",
                "speaker",
            ],
        },
    )

    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload.get("state") == "running"
    assert payload.get("metadata", {}).get("task_id")


@pytest.mark.anyio
async def test_concordance_detach_options_include_mandatory_and_optional_columns(
    authenticated_client, workspace_id
):
    df = pl.DataFrame(
        {
            "text": ["alpha beta", "beta gamma", "alpha gamma"],
            "speaker": ["a", "b", "c"],
        }
    )
    workspace = workspace_manager.get_current_workspace("test")
    assert workspace is not None

    node = Node(
        data=df.lazy(),
        name="text_node",
        workspace=workspace,
        operation="test_add",
        parents=[],
    )
    workspace.add_node(node)

    resp = await authenticated_client.get(
        f"/api/workspaces/nodes/{node.id}/concordance/detach-options",
        params={"column": "text"},
    )

    assert resp.status_code == 200, resp.text
    payload = resp.json()
    node_option = payload["data"]["nodes"][0]

    assert node_option["node_id"] == node.id
    assert node_option["text_column"] == "text"
    assert node_option["available_columns"] == [
        "text",
        "CONC_left_context",
        "CONC_matched_text",
        "CONC_right_context",
        "CONC_start_idx",
        "CONC_end_idx",
        "CONC_l1",
        "CONC_r1",
        "CONC_extraction",
        "speaker",
    ]
    # Every column — including the generated concordance columns — is now
    # user-choosable in the dialog, so nothing is reported as disabled. The
    # generated columns are still ordered ahead of optional metadata.
    assert node_option["disabled_columns"] == []


@pytest.mark.anyio
async def test_concordance_detach_options_ignore_token_metadata(
    authenticated_client, workspace_id
):
    """Token metadata must not create extra detach-picker columns.

    Token specs live in ``Node.tokenization`` and are hydrated only inside analysis
    paths. Registering a token spec should leave detach options scoped to the
    node's physical columns.
    """
    from ldaca_wordflow.api.workspaces.analyses.generated_columns import (
        tokenization_column_name,
    )

    df = pl.DataFrame({"text": ["alpha beta", "beta gamma"]})
    workspace = workspace_manager.get_current_workspace("test")
    assert workspace is not None
    node = Node(
        data=df.lazy(),
        name="text_node",
        workspace=workspace,
        operation="test_add",
        parents=[],
    )
    workspace.add_node(node)

    # Register tokenization directly on the in-memory node so the test
    # doesn't need to round-trip through plbin (the polars FFI plan can't
    # always be deserialized cross-version; see prior FfiPlugin episode).
    tokenization_name = tokenization_column_name(
        "text", "huggingface:bert-base-uncased"
    )
    node.register_tokenization(  # type: ignore[arg-type]
        "text",
        {
            "column_name": tokenization_name,
            "model": "huggingface:bert-base-uncased",
            "language": "en",
            "params": {"lowercase": True, "remove_punct": True},
        },
    )

    resp = await authenticated_client.get(
        f"/api/workspaces/nodes/{node.id}/concordance/detach-options",
        params={"column": "text"},
    )

    assert resp.status_code == 200, resp.text
    node_option = resp.json()["data"]["nodes"][0]
    assert tokenization_name not in node_option["available_columns"]
    assert tokenization_name not in node_option["disabled_columns"]
    assert "text" in node_option["available_columns"]
