"""Unit tests for the in-memory AI annotation preview store.

These exercise :mod:`ldaca_wordflow.core.annotation_preview_store` directly (no
HTTP), pinning the behaviours the endpoints rely on: the config signature scoping,
session reset on config change, the AI-vs-override layering, and the effective-row
selection that drives detach/annotate-all. A fresh :class:`AnnotationPreviewStore`
is built per test so the module singleton's process-lifetime state never leaks in.
"""

from ldaca_wordflow.core.annotation_preview_store import (
    AnnotationPreviewStore,
    signature_of,
)

_IDENTITY = ("user-1", "ws-1", "node-1")


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
) -> str:
    """Build a signature from a shared base config, overriding one field at a time.

    A typed wrapper (rather than ``**dict`` unpacking) so the type checker can see
    each argument's concrete type; every test tweaks a single knob to assert what
    does and does not change the signature.
    """
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
    )


def test_signature_changes_only_for_prediction_fields():
    """Prediction-affecting fields change the signature; write-target/page do not."""
    base = _sig()
    assert _sig(model="other/model") != base
    assert _sig(instruction="different") != base
    assert _sig(temperature=0.7) != base
    assert _sig(reasoning_enabled=True) != base
    assert _sig(provider_id="openai") != base
    # annotation_column and pagination are not signature inputs at all: the same
    # config always hashes identically regardless of where the labels get written.
    assert _sig() == base


def test_sync_resets_rows_when_signature_changes():
    """A config change wipes the node's rows so stale predictions never resurface."""
    store = AnnotationPreviewStore()
    sig_a = _sig()
    store.sync(*_IDENTITY, signature=sig_a, annotation_column="annotation")
    store.put_ai_labels(*_IDENTITY, {0: "support", 1: "critical"})
    assert store.computed_indices(*_IDENTITY, [0, 1]) == {0, 1}

    # Same signature: rows are preserved so paging accumulates.
    store.sync(*_IDENTITY, signature=sig_a, annotation_column="annotation")
    assert store.computed_indices(*_IDENTITY, [0, 1]) == {0, 1}

    # New signature (different model): the session resets to empty.
    store.sync(*_IDENTITY, signature=_sig(model="other"), annotation_column="annotation")
    assert store.computed_indices(*_IDENTITY, [0, 1]) == set()


def test_computed_flag_marks_none_labels_as_cached():
    """A model result of ``None`` (no class) is still a cache hit, not a gap."""
    store = AnnotationPreviewStore()
    store.sync(*_IDENTITY, signature=_sig(), annotation_column="annotation")
    store.put_ai_labels(*_IDENTITY, {0: "support", 1: None})
    # Both rows were classified, so neither should be re-sent to the provider.
    assert store.computed_indices(*_IDENTITY, [0, 1, 2]) == {0, 1}
    assert store.ai_labels_for_page(*_IDENTITY, [0, 1, 2]) == ["support", None, None]


def test_override_wins_over_ai_label_and_survives_relabel():
    """An override beats the model label and a later relabel keeps the override."""
    store = AnnotationPreviewStore()
    store.sync(*_IDENTITY, signature=_sig(), annotation_column="annotation")
    store.put_ai_labels(*_IDENTITY, {0: "support"})
    assert store.set_override(*_IDENTITY, 0, "critical") is True

    # The AI layer still reports the model's raw label...
    assert store.ai_labels_for_page(*_IDENTITY, [0]) == ["support"]
    # ...but the effective label the column/detach uses is the override.
    assert store.effective_rows(*_IDENTITY) == {0: "critical"}

    # Re-running the model for that row updates ``ai`` but must not clobber the edit.
    store.put_ai_labels(*_IDENTITY, {0: "support"})
    assert store.effective_rows(*_IDENTITY) == {0: "critical"}


def test_explicit_none_override_beats_model_label():
    """Choosing "None" is an explicit override, distinct from having no override."""
    store = AnnotationPreviewStore()
    store.sync(*_IDENTITY, signature=_sig(), annotation_column="annotation")
    store.put_ai_labels(*_IDENTITY, {0: "support"})
    assert store.set_override(*_IDENTITY, 0, None) is True
    # effective is the explicit null, not the model's "support".
    assert store.effective_rows(*_IDENTITY) == {0: None}


def test_set_override_without_session_reports_stale():
    """An override with no active session is stale and returns False."""
    store = AnnotationPreviewStore()
    assert store.set_override(*_IDENTITY, 0, "support") is False


def test_state_returns_rows_only_on_signature_match():
    """Hydration returns rows for the matching config and nothing otherwise."""
    store = AnnotationPreviewStore()
    sig = _sig()
    store.sync(*_IDENTITY, signature=sig, annotation_column="annotation")
    store.put_ai_labels(*_IDENTITY, {2: "critical", 0: "support"})
    store.set_override(*_IDENTITY, 0, "support-edited")

    rows = store.state(*_IDENTITY, signature=sig)
    # Rows come back in ascending index order for direct seeding.
    assert [row.row_index for row in rows] == [0, 2]
    first = rows[0]
    assert first.ai == "support"
    assert first.override == "support-edited"
    assert first.has_override is True
    assert first.effective == "support-edited"

    # A different config must not surface these labels.
    assert store.state(*_IDENTITY, signature=_sig(model="other")) == []


def test_effective_rows_signature_guard_for_annotate_all():
    """annotate-all reuse only trusts the cache when the config still matches."""
    store = AnnotationPreviewStore()
    sig = _sig()
    store.sync(*_IDENTITY, signature=sig, annotation_column="annotation")
    store.put_ai_labels(*_IDENTITY, {0: "support", 1: None})

    # Matching signature: both computed rows are reused (including the None class).
    assert store.effective_rows(*_IDENTITY, signature=sig) == {0: "support", 1: None}
    # Mismatched signature: nothing is reused, so annotate-all recomputes.
    assert store.effective_rows(*_IDENTITY, signature=_sig(model="other")) == {}
    # No signature (detach): the current session is used as-is.
    assert store.effective_rows(*_IDENTITY) == {0: "support", 1: None}


def test_clear_drops_session():
    """Clearing forgets the node's session (used after annotate-all writes)."""
    store = AnnotationPreviewStore()
    store.sync(*_IDENTITY, signature=_sig(), annotation_column="annotation")
    store.put_ai_labels(*_IDENTITY, {0: "support"})
    store.clear(*_IDENTITY)
    assert store.effective_rows(*_IDENTITY) == {}
    assert store.state(*_IDENTITY, signature=_sig()) == []
