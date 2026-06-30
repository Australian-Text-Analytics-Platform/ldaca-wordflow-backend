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
