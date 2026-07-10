"""Unit tests for the generation-safe AI annotation preview store.

These tests exercise the store without HTTP, pinning the ownership contract used
by preview, hydration, overrides, clear, detach, and annotate-all. In particular,
every operation after ``sync`` must name the opaque generation it observed so a
late request cannot read, mutate, materialise, or delete a replacement session.
"""

import pytest

from ldaca_wordflow.core.annotation_preview_store import (
    AnnotationPreviewStore,
    signature_of,
)
from ldaca_wordflow.core.exceptions import (
    AnnotationPreviewSessionBusyError,
    AnnotationPreviewSessionConflictError,
    InvalidInputError,
)

_IDENTITY = ("user-1", "ws-1", "node-1")
_COLUMN = "annotation"


def _sig(
    *,
    text_column: str = "document",
    class_node_id: str = "class-1",
    class_column: str = "class",
    description_column: str = "description",
    provider_id: str = "openrouter",
    base_url: str | None = None,
    model: str = "some/model",
    instruction: str = "Classify the stance.",
    temperature: float = 0.0,
    reasoning_enabled: bool = False,
    reasoning_effort: str = "medium",
    class_options: tuple[tuple[str, str], ...] = (
        ("support", "supports the claim"),
        ("critical", "criticises the claim"),
    ),
    source_revision: str = "source-revision-1",
) -> str:
    """Build a signature while keeping each argument visible to the type checker."""

    return signature_of(
        text_column=text_column,
        class_node_id=class_node_id,
        class_column=class_column,
        description_column=description_column,
        provider_id=provider_id,
        base_url=base_url,
        model=model,
        instruction=instruction,
        temperature=temperature,
        reasoning_enabled=reasoning_enabled,
        reasoning_effort=reasoning_effort,
        class_options=class_options,
        source_revision=source_revision,
    )


def _new_session(store: AnnotationPreviewStore, *, column: str = _COLUMN) -> str:
    """Create the standard test generation and return its opaque id."""

    return store.sync(*_IDENTITY, signature=_sig(), annotation_column=column)


def _effective(
    store: AnnotationPreviewStore,
    session_id: str,
    *,
    identity: tuple[str, str, str] = _IDENTITY,
    signature: str | None = None,
    column: str = _COLUMN,
) -> dict[int, str | None]:
    """Read materialisable rows through the same guarded contract as workflows."""

    return store.effective_rows(
        *identity,
        session_id,
        signature=signature,
        annotation_column=column,
    )


def test_signature_changes_only_for_prediction_fields() -> None:
    """Prediction inputs change the hash; write target and page are not inputs."""

    base = _sig()
    assert _sig(model="other/model") != base
    assert _sig(instruction="different") != base
    assert _sig(temperature=0.7) != base
    assert _sig(reasoning_enabled=True) != base
    assert _sig(provider_id="openai") != base
    assert _sig(source_revision="source-revision-2") != base
    assert (
        _sig(
            class_options=(
                ("support", "strongly supports the claim"),
                ("critical", "criticises the claim"),
            )
        )
        != base
    )
    assert _sig() == base


def test_sync_reuses_exact_identity_and_replaces_changed_signature() -> None:
    """Paging reuses one id, while a prediction-config change gets a fresh id."""

    store = AnnotationPreviewStore()
    first = _new_session(store)
    store.put_ai_labels(*_IDENTITY, first, {0: "support"})

    assert _new_session(store) == first
    assert store.computed_indices(*_IDENTITY, first, [0, 1]) == {0}

    second = store.sync(
        *_IDENTITY,
        signature=_sig(model="other/model"),
        annotation_column=_COLUMN,
    )
    assert second != first
    assert store.computed_indices(*_IDENTITY, second, [0, 1]) == set()
    with pytest.raises(AnnotationPreviewSessionConflictError):
        store.put_ai_labels(*_IDENTITY, first, {1: "late"})


def test_target_column_change_creates_empty_generation_without_overrides() -> None:
    """A new output target resets both model labels and target-specific edits."""

    store = AnnotationPreviewStore()
    first = _new_session(store)
    store.put_ai_labels(*_IDENTITY, first, {0: "support"})
    store.set_override(*_IDENTITY, first, 0, "critical")

    second = store.sync(
        *_IDENTITY,
        signature=_sig(),
        annotation_column="other_annotation",
    )
    assert second != first
    assert (
        _effective(
            store,
            second,
            column="other_annotation",
        )
        == {}
    )
    with pytest.raises(AnnotationPreviewSessionConflictError):
        store.set_override(*_IDENTITY, first, 0, "support")


def test_late_provider_completion_cannot_write_replacement_session() -> None:
    """A provider response carrying an old id cannot corrupt the new generation."""

    store = AnnotationPreviewStore()
    old = _new_session(store)
    current = store.sync(
        *_IDENTITY,
        signature=_sig(model="new-model"),
        annotation_column=_COLUMN,
    )
    store.put_ai_labels(*_IDENTITY, current, {0: "current"})

    with pytest.raises(AnnotationPreviewSessionConflictError):
        store.put_ai_labels(*_IDENTITY, old, {0: "late-old"})
    assert store.ai_labels_for_page(*_IDENTITY, current, [0]) == ["current"]


def test_computed_none_is_cached_and_reads_require_expected_id() -> None:
    """A null model result remains a cache hit and all page reads are guarded."""

    store = AnnotationPreviewStore()
    session_id = _new_session(store)
    store.put_ai_labels(*_IDENTITY, session_id, {0: "support", 1: None})

    assert store.computed_indices(*_IDENTITY, session_id, [0, 1, 2]) == {0, 1}
    assert store.ai_labels_for_page(*_IDENTITY, session_id, [0, 1, 2]) == [
        "support",
        None,
        None,
    ]
    with pytest.raises(AnnotationPreviewSessionConflictError):
        store.ai_labels_for_page(*_IDENTITY, "not-current", [0])


def test_override_layers_over_ai_and_survives_same_generation_relabel() -> None:
    """The edit wins without contaminating the raw model-label layer."""

    store = AnnotationPreviewStore()
    session_id = _new_session(store)
    store.put_ai_labels(*_IDENTITY, session_id, {0: "support"})
    store.set_override(*_IDENTITY, session_id, 0, "critical")

    assert store.ai_labels_for_page(*_IDENTITY, session_id, [0]) == ["support"]
    assert _effective(store, session_id) == {0: "critical"}
    store.put_ai_labels(*_IDENTITY, session_id, {0: "support-new"})
    assert _effective(store, session_id) == {0: "critical"}


def test_explicit_none_override_beats_model_label() -> None:
    """Choosing None is distinct from having no override."""

    store = AnnotationPreviewStore()
    session_id = _new_session(store)
    store.put_ai_labels(*_IDENTITY, session_id, {0: "support"})
    store.set_override(*_IDENTITY, session_id, 0, None)
    assert _effective(store, session_id) == {0: None}


def test_missing_or_stale_override_is_a_semantic_conflict() -> None:
    """No override call silently attaches to an absent/current-by-node session."""

    store = AnnotationPreviewStore()
    with pytest.raises(AnnotationPreviewSessionConflictError):
        store.set_override(*_IDENTITY, "missing", 0, "support")

    old = _new_session(store)
    store.sync(
        *_IDENTITY,
        signature=_sig(model="replacement"),
        annotation_column=_COLUMN,
    )
    with pytest.raises(AnnotationPreviewSessionConflictError):
        store.set_override(*_IDENTITY, old, 0, "support")


def test_override_rejects_unpreviewed_row() -> None:
    """An expected id cannot inject arbitrary rows into later materialisation."""

    store = AnnotationPreviewStore()
    session_id = _new_session(store)
    with pytest.raises(InvalidInputError):
        store.set_override(*_IDENTITY, session_id, 99, "support")
    assert _effective(store, session_id) == {}


def test_state_returns_exact_session_metadata_and_sorted_rows() -> None:
    """Hydration identifies the matching id/column and preserves row layers."""

    store = AnnotationPreviewStore()
    session_id = _new_session(store)
    store.put_ai_labels(*_IDENTITY, session_id, {2: "critical", 0: "support"})
    store.set_override(*_IDENTITY, session_id, 0, "support-edited")

    state = store.state(
        *_IDENTITY,
        signature=_sig(),
        annotation_column=_COLUMN,
    )
    assert state is not None
    assert state.session_id == session_id
    assert state.annotation_column == _COLUMN
    assert [row.row_index for row in state.rows] == [0, 2]
    assert state.rows[0].ai == "support"
    assert state.rows[0].override == "support-edited"
    assert state.rows[0].effective == "support-edited"


def test_state_rejects_prediction_or_target_mismatch() -> None:
    """Neither another model nor another output column can hydrate this session."""

    store = AnnotationPreviewStore()
    _new_session(store)
    assert (
        store.state(
            *_IDENTITY,
            signature=_sig(model="other"),
            annotation_column=_COLUMN,
        )
        is None
    )
    assert (
        store.state(
            *_IDENTITY,
            signature=_sig(),
            annotation_column="other_annotation",
        )
        is None
    )


def test_materialisation_validates_id_signature_and_target() -> None:
    """Annotate-all/detach cannot turn a mismatched session into persisted data."""

    store = AnnotationPreviewStore()
    session_id = _new_session(store)
    store.put_ai_labels(*_IDENTITY, session_id, {0: "support", 1: None})
    assert _effective(store, session_id, signature=_sig()) == {
        0: "support",
        1: None,
    }
    assert _effective(store, session_id, signature=None) == {0: "support", 1: None}

    with pytest.raises(AnnotationPreviewSessionConflictError):
        _effective(store, "old", signature=_sig())
    with pytest.raises(AnnotationPreviewSessionConflictError):
        _effective(store, session_id, signature=_sig(model="other"))
    with pytest.raises(AnnotationPreviewSessionConflictError):
        _effective(store, session_id, column="other_annotation")


@pytest.mark.parametrize(
    "other_identity",
    [
        ("user-2", "ws-1", "node-1"),
        ("user-1", "ws-2", "node-1"),
        ("user-1", "ws-1", "node-2"),
    ],
)
def test_session_ids_are_isolated_by_user_workspace_and_node(
    other_identity: tuple[str, str, str],
) -> None:
    """An id never grants access through another ownership key."""

    store = AnnotationPreviewStore()
    session_id = _new_session(store)
    store.put_ai_labels(*_IDENTITY, session_id, {0: "support"})

    with pytest.raises(AnnotationPreviewSessionConflictError):
        _effective(store, session_id, identity=other_identity)


def test_stale_clear_cannot_delete_replacement_generation() -> None:
    """A delayed close names its generation and leaves a newer preview intact."""

    store = AnnotationPreviewStore()
    old = _new_session(store)
    current = store.sync(
        *_IDENTITY,
        signature=_sig(model="replacement"),
        annotation_column=_COLUMN,
    )
    store.put_ai_labels(*_IDENTITY, current, {0: "current"})

    with pytest.raises(AnnotationPreviewSessionConflictError):
        store.clear(*_IDENTITY, old)
    assert store.ai_labels_for_page(*_IDENTITY, current, [0]) == ["current"]

    store.clear(*_IDENTITY, current)
    with pytest.raises(AnnotationPreviewSessionConflictError):
        store.clear(*_IDENTITY, current)


def test_materialization_claim_seals_snapshot_against_active_session_work() -> None:
    """A claimed snapshot cannot race page writes, edits, sync, clear, or claims."""

    store = AnnotationPreviewStore()
    session_id = _new_session(store)
    store.put_ai_labels(*_IDENTITY, session_id, {0: "support"})
    snapshot = store.claim_effective_rows(
        *_IDENTITY,
        session_id,
        signature=_sig(),
        annotation_column=_COLUMN,
    )
    assert snapshot == {0: "support"}

    guarded_calls = [
        lambda: _new_session(store),
        lambda: store.sync(
            *_IDENTITY,
            signature=_sig(model="replacement"),
            annotation_column=_COLUMN,
        ),
        lambda: store.computed_indices(*_IDENTITY, session_id, [0]),
        lambda: store.put_ai_labels(*_IDENTITY, session_id, {0: "late"}),
        lambda: store.ai_labels_for_page(*_IDENTITY, session_id, [0]),
        lambda: store.set_override(*_IDENTITY, session_id, 0, "critical"),
        lambda: store.clear(*_IDENTITY, session_id),
        lambda: store.claim_effective_rows(
            *_IDENTITY,
            session_id,
            signature=_sig(),
            annotation_column=_COLUMN,
        ),
    ]
    for guarded_call in guarded_calls:
        with pytest.raises(AnnotationPreviewSessionBusyError):
            guarded_call()

    # Hydration and dry-run-style effective-row reads are observational only.
    state = store.state(
        *_IDENTITY,
        signature=_sig(),
        annotation_column=_COLUMN,
    )
    assert state is not None
    assert state.session_id == session_id
    assert _effective(store, session_id, signature=None) == snapshot


def test_failed_materialization_release_reopens_same_generation() -> None:
    """Provider/staging failure releases ownership so preview work can continue."""

    store = AnnotationPreviewStore()
    session_id = _new_session(store)
    store.claim_effective_rows(
        *_IDENTITY,
        session_id,
        signature=_sig(),
        annotation_column=_COLUMN,
    )
    store.release_materialization(*_IDENTITY, session_id)

    store.put_ai_labels(*_IDENTITY, session_id, {0: "after-failure"})
    assert store.ai_labels_for_page(*_IDENTITY, session_id, [0]) == ["after-failure"]
    with pytest.raises(AnnotationPreviewSessionConflictError):
        store.release_materialization(*_IDENTITY, session_id)


def test_successful_materialization_consumes_claimed_generation() -> None:
    """Annotate-all completion deletes exactly the generation it claimed."""

    store = AnnotationPreviewStore()
    session_id = _new_session(store)
    store.claim_effective_rows(
        *_IDENTITY,
        session_id,
        signature=_sig(),
        annotation_column=_COLUMN,
    )
    store.complete_materialization(*_IDENTITY, session_id)

    with pytest.raises(AnnotationPreviewSessionConflictError):
        store.put_ai_labels(*_IDENTITY, session_id, {0: "late"})
