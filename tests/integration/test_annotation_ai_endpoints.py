"""Endpoint tests for the server-side AI annotation routes.

Covers ``/annotation/ai/models``, ``/annotation/ai/preview``, and
``/annotation/ai/annotate-all``. Every provider/LLM call is monkeypatched at the
router's import site (``annotation.list_models`` / ``annotate_batch`` /
``annotate_all``) so the tests exercise request validation, page slicing, column
writes, and error translation **without any network traffic** — the engine's own
SDK dispatch is verified separately by its unit probes.
"""

import polars as pl

from ldaca_wordflow.api.workspaces import annotation as annotation_module
from ldaca_wordflow.core.annotation_ai import AnnotationAiError, InferenceConfig
from ldaca_wordflow.core.workspace import workspace_manager


async def _make_class_node(authenticated_client) -> str:
    """Create a class-description node carrying two labelled classes.

    Called by the preview/annotate-all tests because the routes read the valid
    label set from an authoritative class node; reuses the public
    class-descriptions routes so the fixture matches real frontend wiring.
    """
    created = await authenticated_client.post(
        "/api/workspaces/annotation/class-descriptions"
    )
    class_id = created.json()["id"]
    await authenticated_client.put(
        f"/api/workspaces/annotation/class-descriptions/{class_id}",
        json={
            "class_column": "class",
            "description_column": "description",
            "rows": [
                {"class": "support", "description": "Supportive stance"},
                {"class": "critical", "description": "Critical stance"},
            ],
        },
    )
    return class_id


async def test_annotate_ai_models_returns_sorted_models(
    authenticated_client, workspace_id, monkeypatch
):
    async def fake_list_models(provider_id, base_url, api_key):
        assert provider_id == "openrouter"
        assert api_key == "sk-test"
        return ["z-model", "a-model"]

    monkeypatch.setattr(annotation_module, "list_models", fake_list_models)

    response = await authenticated_client.post(
        "/api/workspaces/annotation/ai/models",
        json={"provider_id": "openrouter", "base_url": None, "api_key": "sk-test"},
    )
    assert response.status_code == 200
    assert response.json()["models"] == ["z-model", "a-model"]


async def test_annotate_ai_models_provider_error_returns_502(
    authenticated_client, workspace_id, monkeypatch
):
    async def fake_list_models(provider_id, base_url, api_key):
        raise AnnotationAiError("invalid api key")

    monkeypatch.setattr(annotation_module, "list_models", fake_list_models)

    response = await authenticated_client.post(
        "/api/workspaces/annotation/ai/models",
        json={"provider_id": "openai", "api_key": "bad"},
    )
    assert response.status_code == 502
    assert "invalid api key" in response.json()["detail"]


async def test_annotate_ai_preview_returns_labels_for_page(
    authenticated_client, workspace_id, sample_node_id, monkeypatch
):
    class_id = await _make_class_node(authenticated_client)
    captured: dict[str, object] = {}

    async def fake_annotate_batch(wire, model, api_key, instruction, classes, texts, config=None):
        captured["texts"] = list(texts)
        captured["classes"] = [option.name for option in classes]
        captured["instruction"] = instruction
        captured["config"] = config
        return [f"L{index}" for index in range(len(texts))]

    monkeypatch.setattr(annotation_module, "annotate_batch", fake_annotate_batch)

    response = await authenticated_client.post(
        "/api/workspaces/annotation/ai/preview",
        json={
            "node_id": sample_node_id,
            "text_column": "document",
            "class_node_id": class_id,
            "provider_id": "openrouter",
            "model": "some/model",
            "api_key": "sk-test",
            "instruction": "Classify the stance.",
            "page": 1,
            "page_size": 20,
        },
    )
    assert response.status_code == 200
    # sample.csv has 4 document rows; page 1 (size 20) returns them all in order.
    assert response.json()["labels"] == ["L0", "L1", "L2", "L3"]
    assert captured["texts"] == [
        "This is a sample document.",
        "Another sample text for analysis.",
        "More text content for testing.",
        "Final sample sentence.",
    ]
    assert captured["classes"] == ["support", "critical"]
    assert captured["instruction"] == "Classify the stance."
    # No inference fields were sent, so the engine gets the safe defaults:
    # deterministic sampling with reasoning off.
    config = captured["config"]
    assert isinstance(config, InferenceConfig)
    assert config.temperature == 0.0
    assert config.reasoning_enabled is False


async def test_annotate_ai_preview_forwards_inference_config(
    authenticated_client, workspace_id, sample_node_id, monkeypatch
):
    class_id = await _make_class_node(authenticated_client)
    captured: dict[str, object] = {}

    async def fake_annotate_batch(wire, model, api_key, instruction, classes, texts, config=None):
        captured["config"] = config
        return [None for _ in texts]

    monkeypatch.setattr(annotation_module, "annotate_batch", fake_annotate_batch)

    response = await authenticated_client.post(
        "/api/workspaces/annotation/ai/preview",
        json={
            "node_id": sample_node_id,
            "text_column": "document",
            "class_node_id": class_id,
            "provider_id": "openrouter",
            "model": "some/model",
            "instruction": "x",
            "temperature": 0.7,
            "reasoning_enabled": True,
            "reasoning_effort": "high",
        },
    )
    assert response.status_code == 200
    config = captured["config"]
    assert isinstance(config, InferenceConfig)
    assert config.temperature == 0.7
    assert config.reasoning_enabled is True
    assert config.reasoning_effort == "high"


async def test_annotate_ai_preview_clamps_and_normalizes_inference_config(
    authenticated_client, workspace_id, sample_node_id, monkeypatch
):
    class_id = await _make_class_node(authenticated_client)
    captured: dict[str, object] = {}

    async def fake_annotate_batch(wire, model, api_key, instruction, classes, texts, config=None):
        captured["config"] = config
        return [None for _ in texts]

    monkeypatch.setattr(annotation_module, "annotate_batch", fake_annotate_batch)

    response = await authenticated_client.post(
        "/api/workspaces/annotation/ai/preview",
        json={
            "node_id": sample_node_id,
            "text_column": "document",
            "class_node_id": class_id,
            "provider_id": "openrouter",
            "model": "some/model",
            "instruction": "x",
            # Out-of-range temperature is clamped; an unknown effort falls back.
            "temperature": 9.0,
            "reasoning_enabled": True,
            "reasoning_effort": "bogus",
        },
    )
    assert response.status_code == 200
    config = captured["config"]
    assert isinstance(config, InferenceConfig)
    assert config.temperature == 2.0
    assert config.reasoning_effort == "medium"


async def test_annotate_ai_preview_slices_requested_page(
    authenticated_client, workspace_id, sample_node_id, monkeypatch
):
    class_id = await _make_class_node(authenticated_client)
    captured: dict[str, object] = {}

    async def fake_annotate_batch(wire, model, api_key, instruction, classes, texts, config=None):
        captured["texts"] = list(texts)
        return [None for _ in texts]

    monkeypatch.setattr(annotation_module, "annotate_batch", fake_annotate_batch)

    response = await authenticated_client.post(
        "/api/workspaces/annotation/ai/preview",
        json={
            "node_id": sample_node_id,
            "text_column": "document",
            "class_node_id": class_id,
            "provider_id": "openrouter",
            "model": "some/model",
            "instruction": "x",
            "page": 2,
            "page_size": 2,
        },
    )
    assert response.status_code == 200
    # Page 2 at size 2 is the second half of the 4-row source.
    assert captured["texts"] == [
        "More text content for testing.",
        "Final sample sentence.",
    ]
    assert response.json()["labels"] == [None, None]


async def test_annotate_ai_preview_does_not_mutate_source(
    authenticated_client, workspace_id, sample_node_id, test_user, monkeypatch
):
    class_id = await _make_class_node(authenticated_client)

    async def fake_annotate_batch(wire, model, api_key, instruction, classes, texts, config=None):
        return ["support" for _ in texts]

    monkeypatch.setattr(annotation_module, "annotate_batch", fake_annotate_batch)

    await authenticated_client.post(
        "/api/workspaces/annotation/ai/preview",
        json={
            "node_id": sample_node_id,
            "text_column": "document",
            "class_node_id": class_id,
            "provider_id": "openrouter",
            "model": "some/model",
            "instruction": "x",
        },
    )

    workspace = workspace_manager.get_current_workspace(test_user["id"])
    assert workspace is not None
    columns = workspace.nodes[sample_node_id].data.collect_schema().names()
    assert columns == ["document"]


async def test_annotate_ai_preview_missing_text_column_returns_400(
    authenticated_client, workspace_id, sample_node_id
):
    class_id = await _make_class_node(authenticated_client)
    response = await authenticated_client.post(
        "/api/workspaces/annotation/ai/preview",
        json={
            "node_id": sample_node_id,
            "text_column": "nope",
            "class_node_id": class_id,
            "provider_id": "openrouter",
            "model": "some/model",
            "instruction": "x",
        },
    )
    assert response.status_code == 400


async def test_annotate_ai_preview_missing_node_returns_404(
    authenticated_client, workspace_id
):
    class_id = await _make_class_node(authenticated_client)
    response = await authenticated_client.post(
        "/api/workspaces/annotation/ai/preview",
        json={
            "node_id": "does-not-exist",
            "text_column": "document",
            "class_node_id": class_id,
            "provider_id": "openrouter",
            "model": "some/model",
            "instruction": "x",
        },
    )
    assert response.status_code == 404


async def test_annotate_ai_preview_no_classes_returns_400(
    authenticated_client, workspace_id, sample_node_id
):
    empty_class = await authenticated_client.post(
        "/api/workspaces/annotation/class-descriptions"
    )
    class_id = empty_class.json()["id"]
    response = await authenticated_client.post(
        "/api/workspaces/annotation/ai/preview",
        json={
            "node_id": sample_node_id,
            "text_column": "document",
            "class_node_id": class_id,
            "provider_id": "openrouter",
            "model": "some/model",
            "instruction": "x",
        },
    )
    assert response.status_code == 400


async def test_annotate_ai_preview_provider_error_returns_502(
    authenticated_client, workspace_id, sample_node_id, monkeypatch
):
    class_id = await _make_class_node(authenticated_client)

    async def fake_annotate_batch(*args, **kwargs):
        raise AnnotationAiError("rate limited")

    monkeypatch.setattr(annotation_module, "annotate_batch", fake_annotate_batch)

    response = await authenticated_client.post(
        "/api/workspaces/annotation/ai/preview",
        json={
            "node_id": sample_node_id,
            "text_column": "document",
            "class_node_id": class_id,
            "provider_id": "openrouter",
            "model": "some/model",
            "instruction": "x",
        },
    )
    assert response.status_code == 502
    assert "rate limited" in response.json()["detail"]


async def _add_annotation_column(authenticated_client, node_id: str, column: str) -> None:
    """Attach an empty string annotation column to a source node.

    Called by the annotate-all tests because that route only rewrites an existing
    string column; this reuses the public add-column route to create the write
    target the way the frontend's manual/AI Start flow does.
    """
    response = await authenticated_client.post(
        f"/api/workspaces/annotation/source/{node_id}/annotation-column",
        json={"column_name": column},
    )
    assert response.status_code == 200


async def test_annotate_ai_annotate_all_writes_column(
    authenticated_client, workspace_id, sample_node_id, test_user, monkeypatch
):
    class_id = await _make_class_node(authenticated_client)
    await _add_annotation_column(authenticated_client, sample_node_id, "annotation")
    captured: dict[str, object] = {}

    async def fake_annotate_all(
        wire, model, api_key, instruction, classes, texts, batch_size, config=None
    ):
        captured["texts"] = list(texts)
        captured["count"] = len(texts)
        captured["batch_size"] = batch_size
        captured["config"] = config
        return ["support", None, "critical", None]

    monkeypatch.setattr(annotation_module, "annotate_all", fake_annotate_all)

    response = await authenticated_client.post(
        "/api/workspaces/annotation/ai/annotate-all",
        json={
            "node_id": sample_node_id,
            "text_column": "document",
            "annotation_column": "annotation",
            "class_node_id": class_id,
            "provider_id": "openrouter",
            "model": "some/model",
            "instruction": "x",
            "temperature": 0.3,
            "reasoning_enabled": True,
            "reasoning_effort": "low",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["labeled_rows"] == 2
    assert payload["total_rows"] == 4
    assert payload["node"]["id"] == sample_node_id

    workspace = workspace_manager.get_current_workspace(test_user["id"])
    assert workspace is not None
    collected = workspace.nodes[sample_node_id].data.collect()
    assert collected["annotation"].to_list() == ["support", None, "critical", None]
    assert collected.schema["annotation"] == pl.String
    assert captured["count"] == 4
    # The whole-column run forwards the same inference config as preview.
    config = captured["config"]
    assert isinstance(config, InferenceConfig)
    assert config.temperature == 0.3
    assert config.reasoning_enabled is True
    assert config.reasoning_effort == "low"


async def test_annotate_ai_annotate_all_missing_column_returns_400(
    authenticated_client, workspace_id, sample_node_id, monkeypatch
):
    class_id = await _make_class_node(authenticated_client)

    async def fake_annotate_all(*args, **kwargs):
        raise AssertionError("provider must not be called on validation failure")

    monkeypatch.setattr(annotation_module, "annotate_all", fake_annotate_all)

    response = await authenticated_client.post(
        "/api/workspaces/annotation/ai/annotate-all",
        json={
            "node_id": sample_node_id,
            "text_column": "document",
            "annotation_column": "not_there",
            "class_node_id": class_id,
            "provider_id": "openrouter",
            "model": "some/model",
            "instruction": "x",
        },
    )
    assert response.status_code == 400


async def test_annotate_ai_annotate_all_provider_error_leaves_column(
    authenticated_client, workspace_id, sample_node_id, test_user, monkeypatch
):
    class_id = await _make_class_node(authenticated_client)
    await _add_annotation_column(authenticated_client, sample_node_id, "annotation")

    async def fake_annotate_all(*args, **kwargs):
        raise AnnotationAiError("provider exploded")

    monkeypatch.setattr(annotation_module, "annotate_all", fake_annotate_all)

    response = await authenticated_client.post(
        "/api/workspaces/annotation/ai/annotate-all",
        json={
            "node_id": sample_node_id,
            "text_column": "document",
            "annotation_column": "annotation",
            "class_node_id": class_id,
            "provider_id": "openrouter",
            "model": "some/model",
            "instruction": "x",
        },
    )
    assert response.status_code == 502

    workspace = workspace_manager.get_current_workspace(test_user["id"])
    assert workspace is not None
    collected = workspace.nodes[sample_node_id].data.collect()
    # Nothing written: the column stays all-null after a provider failure.
    assert collected["annotation"].to_list() == [None, None, None, None]


async def test_detach_previewed_rows_creates_annotated_child_node(
    authenticated_client, workspace_id, sample_node_id, test_user
):
    """Detaching materializes exactly the previewed rows + labels as a child node.

    No provider is monkeypatched because detach never calls the LLM: the panel
    already holds the predictions and posts them back, so this is a pure
    workspace mutation (row subset + column write + child-node creation).
    """
    await _add_annotation_column(authenticated_client, sample_node_id, "annotation")

    response = await authenticated_client.post(
        "/api/workspaces/annotation/ai/detach-previewed",
        json={
            "node_id": sample_node_id,
            "annotation_column": "annotation",
            "rows": [
                {"row_index": 0, "label": "support"},
                {"row_index": 1, "label": "  "},
                {"row_index": 2, "label": "critical"},
            ],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["detached_rows"] == 3
    new_id = payload["node"]["id"]
    assert new_id != sample_node_id

    workspace = workspace_manager.get_current_workspace(test_user["id"])
    assert workspace is not None
    child = workspace.nodes[new_id]
    # The detached child hangs off the source so the lineage shows in the graph.
    assert [parent.id for parent in child.parents] == [sample_node_id]
    collected = child.data.collect()
    assert collected.height == 3
    assert collected["document"].to_list() == [
        "This is a sample document.",
        "Another sample text for analysis.",
        "More text content for testing.",
    ]
    # Blank/whitespace labels coerce to null; real labels are written verbatim.
    assert collected["annotation"].to_list() == ["support", None, "critical"]
    # The source node is untouched — detach copies rows, it does not move them.
    source = workspace.nodes[sample_node_id].data.collect()
    assert source.height == 4
    assert source["annotation"].to_list() == [None, None, None, None]


async def test_detach_previewed_rows_out_of_range_returns_400(
    authenticated_client, workspace_id, sample_node_id
):
    await _add_annotation_column(authenticated_client, sample_node_id, "annotation")

    response = await authenticated_client.post(
        "/api/workspaces/annotation/ai/detach-previewed",
        json={
            "node_id": sample_node_id,
            "annotation_column": "annotation",
            "rows": [{"row_index": 99, "label": "support"}],
        },
    )
    assert response.status_code == 400


async def test_detach_previewed_rows_empty_returns_400(
    authenticated_client, workspace_id, sample_node_id
):
    response = await authenticated_client.post(
        "/api/workspaces/annotation/ai/detach-previewed",
        json={
            "node_id": sample_node_id,
            "annotation_column": "annotation",
            "rows": [],
        },
    )
    assert response.status_code == 400


async def test_detach_previewed_rows_missing_node_returns_404(
    authenticated_client, workspace_id
):
    response = await authenticated_client.post(
        "/api/workspaces/annotation/ai/detach-previewed",
        json={
            "node_id": "does-not-exist",
            "annotation_column": "annotation",
            "rows": [{"row_index": 0, "label": "support"}],
        },
    )
    assert response.status_code == 404


def _preview_body(node_id: str, class_id: str, **overrides) -> dict:
    """Build a minimal AI-preview request body for the cache tests.

    Called by the caching/hydration tests below so preview, state, annotate-all,
    and detach all send the *same* prediction-affecting config — otherwise their
    signatures would differ and the server would (correctly) refuse to reuse the
    cache, defeating the very behaviour under test.
    """
    body = {
        "node_id": node_id,
        "text_column": "document",
        "class_node_id": class_id,
        "provider_id": "openrouter",
        "model": "some/model",
        "instruction": "Classify the stance.",
    }
    body.update(overrides)
    return body


async def test_annotate_ai_preview_caches_page_and_reuses(
    authenticated_client, workspace_id, sample_node_id, monkeypatch
):
    """A repeated preview of the same page is served from the store, not the LLM."""
    class_id = await _make_class_node(authenticated_client)
    calls = {"count": 0}

    async def fake_annotate_batch(wire, model, api_key, instruction, classes, texts, config=None):
        calls["count"] += 1
        return [f"L{index}" for index in range(len(texts))]

    monkeypatch.setattr(annotation_module, "annotate_batch", fake_annotate_batch)

    body = _preview_body(sample_node_id, class_id, page=1, page_size=20)
    first = await authenticated_client.post(
        "/api/workspaces/annotation/ai/preview", json=body
    )
    second = await authenticated_client.post(
        "/api/workspaces/annotation/ai/preview", json=body
    )
    assert first.status_code == 200
    assert second.status_code == 200
    # Same labels both times, but the provider was hit only for the first call.
    assert first.json()["labels"] == ["L0", "L1", "L2", "L3"]
    assert second.json()["labels"] == ["L0", "L1", "L2", "L3"]
    assert calls["count"] == 1


async def test_annotate_ai_preview_state_hydrates_stored_rows(
    authenticated_client, workspace_id, sample_node_id, monkeypatch
):
    """After a preview, ``preview/state`` returns the cached rows for rehydration."""
    class_id = await _make_class_node(authenticated_client)

    async def fake_annotate_batch(wire, model, api_key, instruction, classes, texts, config=None):
        return ["support", None, "critical", "support"][: len(texts)]

    monkeypatch.setattr(annotation_module, "annotate_batch", fake_annotate_batch)

    await authenticated_client.post(
        "/api/workspaces/annotation/ai/preview",
        json=_preview_body(sample_node_id, class_id, page=1, page_size=20),
    )
    state = await authenticated_client.post(
        "/api/workspaces/annotation/ai/preview/state",
        json=_preview_body(sample_node_id, class_id),
    )
    assert state.status_code == 200
    rows = state.json()["rows"]
    assert [row["row_index"] for row in rows] == [0, 1, 2, 3]
    assert [row["ai"] for row in rows] == ["support", None, "critical", "support"]
    # Nothing was overridden yet, so effective mirrors the model labels.
    assert all(row["has_override"] is False for row in rows)
    assert [row["effective"] for row in rows] == ["support", None, "critical", "support"]


async def test_annotate_ai_preview_state_mismatched_config_returns_empty(
    authenticated_client, workspace_id, sample_node_id, monkeypatch
):
    """State for a different model must not surface labels from another config."""
    class_id = await _make_class_node(authenticated_client)

    async def fake_annotate_batch(wire, model, api_key, instruction, classes, texts, config=None):
        return ["support" for _ in texts]

    monkeypatch.setattr(annotation_module, "annotate_batch", fake_annotate_batch)

    await authenticated_client.post(
        "/api/workspaces/annotation/ai/preview",
        json=_preview_body(sample_node_id, class_id, page=1, page_size=20),
    )
    state = await authenticated_client.post(
        "/api/workspaces/annotation/ai/preview/state",
        json=_preview_body(sample_node_id, class_id, model="different/model"),
    )
    assert state.status_code == 200
    assert state.json()["rows"] == []


async def test_annotate_ai_preview_override_persists_and_hydrates(
    authenticated_client, workspace_id, sample_node_id, monkeypatch
):
    """A manual override survives via the store and wins in the state payload."""
    class_id = await _make_class_node(authenticated_client)

    async def fake_annotate_batch(wire, model, api_key, instruction, classes, texts, config=None):
        return ["support" for _ in texts]

    monkeypatch.setattr(annotation_module, "annotate_batch", fake_annotate_batch)

    await authenticated_client.post(
        "/api/workspaces/annotation/ai/preview",
        json=_preview_body(sample_node_id, class_id, page=1, page_size=20),
    )
    override = await authenticated_client.put(
        "/api/workspaces/annotation/ai/preview/override",
        json={"node_id": sample_node_id, "row_index": 0, "label": "critical"},
    )
    assert override.status_code == 200
    assert override.json()["ok"] is True

    state = await authenticated_client.post(
        "/api/workspaces/annotation/ai/preview/state",
        json=_preview_body(sample_node_id, class_id),
    )
    row0 = next(row for row in state.json()["rows"] if row["row_index"] == 0)
    # The AI layer keeps the model's "support"; the effective label is the edit.
    assert row0["ai"] == "support"
    assert row0["override"] == "critical"
    assert row0["has_override"] is True
    assert row0["effective"] == "critical"


async def test_annotate_ai_preview_override_without_session_reports_not_ok(
    authenticated_client, workspace_id, sample_node_id
):
    """Overriding a node that was never previewed is stale and returns ok=False."""
    response = await authenticated_client.put(
        "/api/workspaces/annotation/ai/preview/override",
        json={"node_id": sample_node_id, "row_index": 0, "label": "support"},
    )
    assert response.status_code == 200
    assert response.json()["ok"] is False


async def test_detach_previewed_rows_uses_store_across_pages(
    authenticated_client, workspace_id, sample_node_id, test_user, monkeypatch
):
    """Detach materialises every previewed page from the store, not just the last.

    This is the regression test for "Detach only grabbed one page": the panel used
    to send the browser's current page, but detach now reads the whole server-side
    session, so viewing two pages and detaching without a ``rows`` list yields all
    four rows.
    """
    class_id = await _make_class_node(authenticated_client)
    await _add_annotation_column(authenticated_client, sample_node_id, "annotation")

    async def fake_annotate_batch(wire, model, api_key, instruction, classes, texts, config=None):
        # Return a stable label per text so both pages get distinct predictions.
        return [f"c-{text[:4]}" for text in texts]

    monkeypatch.setattr(annotation_module, "annotate_batch", fake_annotate_batch)

    # View page 1 and page 2 (two rows each) so all four rows land in the store.
    for page in (1, 2):
        preview = await authenticated_client.post(
            "/api/workspaces/annotation/ai/preview",
            json=_preview_body(
                sample_node_id, class_id, annotation_column="annotation", page=page, page_size=2
            ),
        )
        assert preview.status_code == 200

    # Detach WITHOUT a rows list — the server rebuilds them from the session.
    detach = await authenticated_client.post(
        "/api/workspaces/annotation/ai/detach-previewed",
        json={"node_id": sample_node_id, "annotation_column": "annotation"},
    )
    assert detach.status_code == 200
    assert detach.json()["detached_rows"] == 4

    workspace = workspace_manager.get_current_workspace(test_user["id"])
    assert workspace is not None
    child = workspace.nodes[detach.json()["node"]["id"]]
    collected = child.data.collect()
    assert collected.height == 4
    assert collected["annotation"].to_list() == [
        "c-This",
        "c-Anot",
        "c-More",
        "c-Fina",
    ]


async def test_detach_previewed_rows_dry_run_probes_store_without_materialising(
    authenticated_client, workspace_id, sample_node_id, test_user, monkeypatch
):
    """A ``dry_run`` detach reports the session size but creates nothing.

    The panel calls this on mount to gate its Detach button from the authoritative
    server session (its own per-page map is wiped on a tab-switch remount), so the
    probe must return the full count without adding a child node.
    """
    class_id = await _make_class_node(authenticated_client)
    await _add_annotation_column(authenticated_client, sample_node_id, "annotation")

    async def fake_annotate_batch(wire, model, api_key, instruction, classes, texts, config=None):
        return [f"c-{text[:4]}" for text in texts]

    monkeypatch.setattr(annotation_module, "annotate_batch", fake_annotate_batch)

    for page in (1, 2):
        preview = await authenticated_client.post(
            "/api/workspaces/annotation/ai/preview",
            json=_preview_body(
                sample_node_id, class_id, annotation_column="annotation", page=page, page_size=2
            ),
        )
        assert preview.status_code == 200

    workspace = workspace_manager.get_current_workspace(test_user["id"])
    assert workspace is not None
    nodes_before = set(workspace.nodes)

    probe = await authenticated_client.post(
        "/api/workspaces/annotation/ai/detach-previewed",
        json={
            "node_id": sample_node_id,
            "annotation_column": "annotation",
            "dry_run": True,
        },
    )
    assert probe.status_code == 200
    body = probe.json()
    # All four previewed rows are counted, but nothing is created.
    assert body["detached_rows"] == 4
    assert body["node"] is None
    assert set(workspace.nodes) == nodes_before


async def test_detach_previewed_rows_dry_run_returns_zero_for_empty_session(
    authenticated_client, workspace_id, sample_node_id
):
    """A ``dry_run`` probe on a node with no preview session reports zero, not 400.

    That lets the panel simply disable its Detach button on an empty/cleared session
    instead of treating the probe as an error.
    """
    response = await authenticated_client.post(
        "/api/workspaces/annotation/ai/detach-previewed",
        json={
            "node_id": sample_node_id,
            "annotation_column": "annotation",
            "dry_run": True,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["detached_rows"] == 0
    assert body["node"] is None


async def test_annotate_ai_annotate_all_reuses_cached_labels(
    authenticated_client, workspace_id, sample_node_id, test_user, monkeypatch
):
    """Annotate All reuses previewed labels and only sends the uncached remainder."""
    class_id = await _make_class_node(authenticated_client)
    await _add_annotation_column(authenticated_client, sample_node_id, "annotation")

    async def fake_annotate_batch(wire, model, api_key, instruction, classes, texts, config=None):
        # Only the first two rows are previewed (page 1, size 2).
        return ["support", "critical"][: len(texts)]

    async def fake_annotate_all(
        wire, model, api_key, instruction, classes, texts, batch_size, config=None
    ):
        # The two previewed rows are cached, so annotate-all only sees the rest.
        assert list(texts) == [
            "More text content for testing.",
            "Final sample sentence.",
        ]
        return ["support", None]

    monkeypatch.setattr(annotation_module, "annotate_batch", fake_annotate_batch)
    monkeypatch.setattr(annotation_module, "annotate_all", fake_annotate_all)

    await authenticated_client.post(
        "/api/workspaces/annotation/ai/preview",
        json=_preview_body(
            sample_node_id, class_id, annotation_column="annotation", page=1, page_size=2
        ),
    )
    response = await authenticated_client.post(
        "/api/workspaces/annotation/ai/annotate-all",
        json={
            "node_id": sample_node_id,
            "text_column": "document",
            "annotation_column": "annotation",
            "class_node_id": class_id,
            "provider_id": "openrouter",
            "model": "some/model",
            "instruction": "Classify the stance.",
        },
    )
    assert response.status_code == 200
    workspace = workspace_manager.get_current_workspace(test_user["id"])
    assert workspace is not None
    collected = workspace.nodes[sample_node_id].data.collect()
    # Rows 0-1 come from the preview cache; rows 2-3 from the annotate-all call.
    assert collected["annotation"].to_list() == ["support", "critical", "support", None]


async def test_annotate_ai_preview_clear_drops_cached_session(
    authenticated_client, workspace_id, sample_node_id, monkeypatch
):
    """Clearing a node's session empties its rehydration state and detach count.

    Mirrors the panel's "Close preview": after previewing a page, ``preview/clear``
    drops the server session so ``preview/state`` returns no rows and a detach
    ``dry_run`` counts zero — proving a later preview starts from a clean slate.
    """
    class_id = await _make_class_node(authenticated_client)

    async def fake_annotate_batch(wire, model, api_key, instruction, classes, texts, config=None):
        return ["support" for _ in texts]

    monkeypatch.setattr(annotation_module, "annotate_batch", fake_annotate_batch)

    # Prime the store with one previewed page.
    preview = await authenticated_client.post(
        "/api/workspaces/annotation/ai/preview",
        json=_preview_body(sample_node_id, class_id, page=1, page_size=20),
    )
    assert preview.status_code == 200
    primed = await authenticated_client.post(
        "/api/workspaces/annotation/ai/preview/state",
        json=_preview_body(sample_node_id, class_id),
    )
    assert primed.json()["rows"], "precondition: session should hold rows before clear"

    cleared = await authenticated_client.post(
        "/api/workspaces/annotation/ai/preview/clear",
        json={"node_id": sample_node_id},
    )
    assert cleared.status_code == 200
    assert cleared.json()["ok"] is True

    # State no longer hydrates anything, and a detach probe finds nothing to detach.
    state = await authenticated_client.post(
        "/api/workspaces/annotation/ai/preview/state",
        json=_preview_body(sample_node_id, class_id),
    )
    assert state.json()["rows"] == []
    probe = await authenticated_client.post(
        "/api/workspaces/annotation/ai/detach-previewed",
        json={"node_id": sample_node_id, "annotation_column": "annotation", "dry_run": True},
    )
    assert probe.json()["detached_rows"] == 0


async def test_annotate_ai_preview_clear_without_session_is_noop(
    authenticated_client, workspace_id, sample_node_id
):
    """Clearing a node that was never previewed succeeds (idempotent no-op)."""
    response = await authenticated_client.post(
        "/api/workspaces/annotation/ai/preview/clear",
        json={"node_id": sample_node_id},
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True
