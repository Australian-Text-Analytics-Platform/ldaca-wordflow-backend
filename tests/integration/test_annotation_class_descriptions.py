import polars as pl

from ldaca_wordflow.core.workspace import workspace_manager


async def test_create_annotation_class_descriptions_node(
    authenticated_client, workspace_id, test_user
):
    response = await authenticated_client.post(
        "/api/workspaces/annotation/class-descriptions"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "Annotation Classes"
    assert payload["columns"] == ["class", "description"]
    assert payload["schema"] == {"class": "String", "description": "String"}

    workspace = workspace_manager.get_current_workspace(test_user["id"])
    assert workspace is not None
    node = workspace.nodes[payload["id"]]
    collected = node.data.collect()
    assert collected.schema == {"class": pl.String, "description": pl.String}
    assert collected.height == 0


async def test_create_annotation_class_descriptions_uses_unique_names(
    authenticated_client, workspace_id
):
    first = await authenticated_client.post("/api/workspaces/annotation/class-descriptions")
    second = await authenticated_client.post("/api/workspaces/annotation/class-descriptions")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["name"] == "Annotation Classes"
    assert second.json()["name"] == "Annotation Classes 2"


async def test_update_annotation_class_descriptions_rows(
    authenticated_client, workspace_id, test_user
):
    created = await authenticated_client.post(
        "/api/workspaces/annotation/class-descriptions"
    )
    assert created.status_code == 200
    node_id = created.json()["id"]

    payload = {
        "class_column": "class",
        "description_column": "description",
        "rows": [
            {"class": "support", "description": "Supportive stance"},
            {"class": "critical", "description": "Critical stance"},
        ],
    }
    updated = await authenticated_client.put(
        f"/api/workspaces/annotation/class-descriptions/{node_id}",
        json=payload,
    )

    assert updated.status_code == 200
    assert updated.json() == payload

    fetched = await authenticated_client.get(
        f"/api/workspaces/annotation/class-descriptions/{node_id}",
        params={"class_column": "class", "description_column": "description"},
    )
    assert fetched.status_code == 200
    assert fetched.json() == payload

    workspace = workspace_manager.get_current_workspace(test_user["id"])
    assert workspace is not None
    assert workspace.nodes[node_id].data.collect().to_dicts() == [
        {"class": "support", "description": "Supportive stance"},
        {"class": "critical", "description": "Critical stance"},
    ]


async def test_set_annotation_class_parent_links_to_source(
    authenticated_client, workspace_id, test_user
):
    source = await authenticated_client.post("/api/workspaces/annotation/class-descriptions")
    classes = await authenticated_client.post("/api/workspaces/annotation/class-descriptions")
    source_id = source.json()["id"]
    class_id = classes.json()["id"]

    response = await authenticated_client.put(
        f"/api/workspaces/annotation/class-descriptions/{class_id}/parent",
        json={"parent_node_id": source_id},
    )
    assert response.status_code == 200

    workspace = workspace_manager.get_current_workspace(test_user["id"])
    assert workspace is not None
    parent_ids = {
        parent.id if hasattr(parent, "id") else parent
        for parent in workspace.nodes[class_id].parents
    }
    assert source_id in parent_ids


async def test_set_annotation_class_parent_rejects_self_and_missing(
    authenticated_client, workspace_id
):
    created = await authenticated_client.post("/api/workspaces/annotation/class-descriptions")
    node_id = created.json()["id"]

    same = await authenticated_client.put(
        f"/api/workspaces/annotation/class-descriptions/{node_id}/parent",
        json={"parent_node_id": node_id},
    )
    assert same.status_code == 400

    missing = await authenticated_client.put(
        f"/api/workspaces/annotation/class-descriptions/{node_id}/parent",
        json={"parent_node_id": "does-not-exist"},
    )
    assert missing.status_code == 404


async def test_create_annotation_column_adds_empty_string_column(
    authenticated_client, workspace_id, test_user
):
    created = await authenticated_client.post("/api/workspaces/annotation/class-descriptions")
    node_id = created.json()["id"]

    response = await authenticated_client.post(
        f"/api/workspaces/annotation/source/{node_id}/annotation-column",
        json={"column_name": "annotation"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "annotation" in payload["columns"]
    assert payload["schema"]["annotation"] == "String"

    workspace = workspace_manager.get_current_workspace(test_user["id"])
    assert workspace is not None
    collected = workspace.nodes[node_id].data.collect()
    assert "annotation" in collected.columns
    assert collected.schema["annotation"] == pl.String


async def test_create_annotation_column_rejects_duplicate_and_blank(
    authenticated_client, workspace_id
):
    created = await authenticated_client.post("/api/workspaces/annotation/class-descriptions")
    node_id = created.json()["id"]

    duplicate = await authenticated_client.post(
        f"/api/workspaces/annotation/source/{node_id}/annotation-column",
        json={"column_name": "class"},
    )
    assert duplicate.status_code == 400

    blank = await authenticated_client.post(
        f"/api/workspaces/annotation/source/{node_id}/annotation-column",
        json={"column_name": "   "},
    )
    assert blank.status_code == 400


async def test_create_annotation_column_missing_node_returns_404(
    authenticated_client, workspace_id
):
    response = await authenticated_client.post(
        "/api/workspaces/annotation/source/does-not-exist/annotation-column",
        json={"column_name": "annotation"},
    )
    assert response.status_code == 404

async def _make_source_with_rows(authenticated_client):
    """Create a 2-row source node carrying an empty ``annotation`` column.

    Called by the annotation-cell tests because setting a cell needs a node that
    actually has rows; reuses the public class-description + add-column routes so
    the fixture exercises the same wiring the frontend does.
    """
    created = await authenticated_client.post("/api/workspaces/annotation/class-descriptions")
    node_id = created.json()["id"]
    await authenticated_client.put(
        f"/api/workspaces/annotation/class-descriptions/{node_id}",
        json={
            "class_column": "class",
            "description_column": "description",
            "rows": [
                {"class": "a", "description": "first"},
                {"class": "b", "description": "second"},
            ],
        },
    )
    await authenticated_client.post(
        f"/api/workspaces/annotation/source/{node_id}/annotation-column",
        json={"column_name": "annotation"},
    )
    return node_id


async def test_set_annotation_cell_writes_value(
    authenticated_client, workspace_id, test_user
):
    node_id = await _make_source_with_rows(authenticated_client)

    response = await authenticated_client.put(
        f"/api/workspaces/annotation/source/{node_id}/annotation-cell",
        json={"column_name": "annotation", "row_index": 1, "value": "support"},
    )
    assert response.status_code == 200

    workspace = workspace_manager.get_current_workspace(test_user["id"])
    assert workspace is not None
    collected = workspace.nodes[node_id].data.collect()
    assert collected["annotation"].to_list() == [None, "support"]


async def test_set_annotation_cell_clears_to_null_on_blank(
    authenticated_client, workspace_id, test_user
):
    node_id = await _make_source_with_rows(authenticated_client)
    await authenticated_client.put(
        f"/api/workspaces/annotation/source/{node_id}/annotation-cell",
        json={"column_name": "annotation", "row_index": 0, "value": "support"},
    )

    cleared = await authenticated_client.put(
        f"/api/workspaces/annotation/source/{node_id}/annotation-cell",
        json={"column_name": "annotation", "row_index": 0, "value": "  "},
    )
    assert cleared.status_code == 200

    workspace = workspace_manager.get_current_workspace(test_user["id"])
    assert workspace is not None
    collected = workspace.nodes[node_id].data.collect()
    assert collected["annotation"].to_list() == [None, None]


async def test_set_annotation_cell_rejects_bad_column_and_row(
    authenticated_client, workspace_id
):
    node_id = await _make_source_with_rows(authenticated_client)

    missing_column = await authenticated_client.put(
        f"/api/workspaces/annotation/source/{node_id}/annotation-cell",
        json={"column_name": "nope", "row_index": 0, "value": "x"},
    )
    assert missing_column.status_code == 400

    out_of_range = await authenticated_client.put(
        f"/api/workspaces/annotation/source/{node_id}/annotation-cell",
        json={"column_name": "annotation", "row_index": 5, "value": "x"},
    )
    assert out_of_range.status_code == 400


async def test_set_annotation_cell_missing_node_returns_404(
    authenticated_client, workspace_id
):
    response = await authenticated_client.put(
        "/api/workspaces/annotation/source/does-not-exist/annotation-cell",
        json={"column_name": "annotation", "row_index": 0, "value": "x"},
    )
    assert response.status_code == 404
