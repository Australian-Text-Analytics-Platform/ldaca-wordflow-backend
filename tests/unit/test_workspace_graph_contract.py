async def test_workspace_graph_returns_lightweight_nodes(
    authenticated_client, workspace_id, tiny_node_id
):
    graph_response = await authenticated_client.get(f"/api/workspaces/{workspace_id}/graph")

    assert graph_response.status_code == 200
    graph_node = next(
        node
        for node in graph_response.json()["nodes"]
        if node["id"] == tiny_node_id
    )

    assert graph_node["id"] == tiny_node_id
    assert graph_node["name"]
    assert "schema" not in graph_node
    assert "columns" not in graph_node
    assert "shape" not in graph_node
    assert "tokenizer_models" not in graph_node
    assert "dtype_normalization" not in graph_node

    info_response = await authenticated_client.post(
        f"/api/workspaces/{workspace_id}/nodes:batchGet",
        json={"nodes": [tiny_node_id]},
    )
    assert info_response.status_code == 200
    node_info = info_response.json()["nodes"][0]
    assert node_info["id"] == tiny_node_id
    assert "schema" in node_info
    assert "columns" in node_info
    assert "shape" in node_info


async def test_workspace_nodes_batch_get_endpoint_batches_metadata(
    authenticated_client, workspace_id, tiny_node_id
):
    empty_response = await authenticated_client.post(
        f"/api/workspaces/{workspace_id}/nodes:batchGet",
        json={"nodes": []},
    )
    assert empty_response.status_code == 200
    assert empty_response.json() == {"nodes": []}

    response = await authenticated_client.post(
        f"/api/workspaces/{workspace_id}/nodes:batchGet",
        json={"nodes": [tiny_node_id, tiny_node_id]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert [node["id"] for node in payload["nodes"]] == [tiny_node_id, tiny_node_id]
    assert all("schema" in node for node in payload["nodes"])
