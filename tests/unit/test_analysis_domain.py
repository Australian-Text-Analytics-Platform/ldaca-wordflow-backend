"""Strict Workspace-owned Analysis model and lifecycle invariants."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import TypeAdapter, ValidationError

from ldaca_wordflow.domain.workspace import (
    AnalysisQuerySnapshotRecord,
    AnalysisRecord,
    AnalysisRequest,
    AnalysisState,
    AnalysisKind,
    AnnotationAnalysisRequest,
    AnnotationAnalysisSubmission,
    ConcordanceAnalysisRequest,
    ConcordanceDetachmentAnalysisRequest,
    Failure,
    Progress,
    Tab,
    TokenFrequencyAnalysisRequest,
    ValidAnalysisIntegrity,
    Workspace,
    TopicModelingDetachmentAnalysisRequest,
    persisted_submission,
    public_analysis,
)


def _concordance() -> ConcordanceAnalysisRequest:
    node_id = uuid.uuid4()
    return ConcordanceAnalysisRequest(
        node_ids=[node_id],
        node_columns={node_id: "text"},
        search_word="word",
    )


def test_analysis_request_union_is_strict_and_discriminated() -> None:
    request = _concordance()
    restored = TypeAdapter(AnalysisRequest).validate_python(
        request.model_dump(mode="json")
    )

    assert restored == request
    with pytest.raises(ValidationError):
        TypeAdapter(AnalysisRequest).validate_python(
            {**request.model_dump(mode="json"), "unknown": True}
        )


def test_tokenizer_mappings_follow_each_analysis_mode_contract() -> None:
    first = uuid.uuid4()
    second = uuid.uuid4()

    with pytest.raises(ValidationError, match="exactly match"):
        TokenFrequencyAnalysisRequest(
            node_ids=[first, second],
            node_columns={first: "text", second: "body"},
            node_tokenizer_models={first: "native:plain_words_en"},
        )

    text_request = ConcordanceAnalysisRequest(
        node_ids=[first, second],
        node_columns={first: "text", second: "body"},
        node_tokenizer_models={first: "native:plain_words_en"},
        search_word="word",
    )
    assert text_request.node_tokenizer_models == {
        first: "native:plain_words_en"
    }

    with pytest.raises(ValidationError, match="Tokens mode"):
        ConcordanceAnalysisRequest(
            node_ids=[first, second],
            node_columns={first: "text", second: "body"},
            node_tokenizer_models={first: "native:plain_words_en"},
            search_word="word",
            search_mode="tokens",
        )

    tokens_request = ConcordanceAnalysisRequest(
        node_ids=[first, second],
        node_columns={first: "text", second: "body"},
        node_tokenizer_models={
            first: "native:plain_words_en",
            second: "lindera:jieba",
        },
        search_word="word",
        search_mode="tokens",
    )
    assert set(tokens_request.node_tokenizer_models) == {first, second}


def test_topic_modeling_detachment_request_preserves_ordered_sources() -> None:
    first = uuid.uuid4()
    second = uuid.uuid4()
    request = TopicModelingDetachmentAnalysisRequest(
        node_ids=[first, second],
        selected_columns={first: ["text"], second: []},
        new_node_names={first: "First topics", second: "Second topics"},
        topic_ids=[3, 1],
        topic_meanings_override=[
            {"topic_id": 3, "words": ["one", "two"]},
            {"topic_id": 1, "words": ["three"]},
        ],
    )

    restored = TypeAdapter(AnalysisRequest).validate_python(
        request.model_dump(mode="json")
    )
    assert restored == request
    assert tuple(restored.node_ids) == (first, second)
    assert restored.selected_columns[second] == []

    with pytest.raises(ValidationError, match="unique"):
        TopicModelingDetachmentAnalysisRequest(
            node_ids=[first, first],
            selected_columns={first: ["text"]},
            new_node_names={first: "Topics"},
        )
    with pytest.raises(ValidationError, match="align"):
        TopicModelingDetachmentAnalysisRequest(
            node_ids=[first, second],
            selected_columns={first: ["text"]},
            new_node_names={first: "Topics", second: "Other topics"},
        )
    with pytest.raises(ValidationError):
        ConcordanceAnalysisRequest(
            node_ids=request.node_ids,
            node_columns={},
            search_word="word",
        )


def test_annotation_submission_strips_transient_secret_before_persistence() -> None:
    submission = AnnotationAnalysisSubmission(
        node_id=uuid.uuid4(),
        text_column="text",
        annotation_column="class",
        classes=[{"name": "Relevant", "description": ""}],
        provider="openai",
        model="model",
        instruction="Classify the text",
        output_node_name="Annotated",
        api_key="transient-secret",
    )

    persisted = persisted_submission(submission)

    assert isinstance(persisted, AnnotationAnalysisRequest)
    assert persisted.kind == "annotation"
    assert "api_key" not in persisted.model_dump(mode="json")
    assert "transient-secret" not in repr(submission)


@pytest.mark.parametrize(
    "progress",
    [
        {"fraction": -0.1, "message": "Bad"},
        {"fraction": 1.1, "message": "Bad"},
        {"fraction": float("nan"), "message": "Bad"},
        {"fraction": float("inf"), "message": "Bad"},
        {"fraction": "0.5", "message": "Bad"},
        {"fraction": 0.5, "message": 123},
        {"fraction": 0.5, "message": "bad\u0000message"},
        {"fraction": 0.5, "message": "x" * 501},
    ],
)
def test_progress_rejects_normalization_and_unsafe_values(
    progress: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        Progress.model_validate(progress)


def test_analysis_lifecycle_and_public_shape_are_exact() -> None:
    created_at = datetime.now(UTC)
    record = AnalysisRecord.create(_concordance(), timestamp=created_at)
    public = public_analysis(record, integrity=ValidAnalysisIntegrity())

    assert set(public.model_dump()) == {
        "id",
        "parent_analysis_id",
        "request",
        "state",
        "progress",
        "cancellation_requested_at",
        "error",
        "integrity",
        "created_at",
        "started_at",
        "finished_at",
        "revision",
        "output_node_ids",
    }
    assert record.state is AnalysisState.QUEUED
    assert record.progress == Progress(fraction=0.0, message="Queued")
    assert record.output_node_ids == []
    assert public.output_node_ids == []

    payload = record.model_dump()
    payload.update(
        state="succeeded",
        started_at=created_at + timedelta(seconds=1),
        finished_at=created_at + timedelta(seconds=2),
        progress={"fraction": 0.9, "message": "Done"},
        result_payload={"ok": True},
    )
    with pytest.raises(ValidationError):
        AnalysisRecord.model_validate(payload)

    payload.update(progress={"fraction": 1.0, "message": "Done"})
    succeeded = AnalysisRecord.model_validate(payload)
    assert succeeded.state is AnalysisState.SUCCEEDED


def test_failed_and_cancelled_lifecycle_fields_are_not_interchangeable() -> None:
    record = AnalysisRecord.create(_concordance(), timestamp=datetime.now(UTC))
    payload = record.model_dump()
    payload.update(
        state="failed",
        finished_at=record.created_at,
        error=Failure(code="analysis_execution_failed", message="Analysis failed"),
    )
    assert AnalysisRecord.model_validate(payload).state is AnalysisState.FAILED

    payload.update(state="cancelled", error=None)
    with pytest.raises(ValidationError):
        AnalysisRecord.model_validate(payload)
    payload["cancellation_requested_at"] = record.created_at
    assert AnalysisRecord.model_validate(payload).state is AnalysisState.CANCELLED


def test_workspace_enforces_one_level_analysis_ownership_and_reservations() -> None:
    workspace = Workspace(name="analyses")
    node_id = uuid.uuid4()
    request = ConcordanceAnalysisRequest(
        node_ids=[node_id],
        node_columns={node_id: "text"},
        search_word="word",
    )
    root = workspace.add_analysis(
        AnalysisRecord.create(request, timestamp=datetime.now(UTC))
    )
    child = workspace.add_analysis(
        AnalysisRecord.create(
            ConcordanceDetachmentAnalysisRequest(
                node_id=node_id,
                selected_columns=["left", "match", "right"],
            ),
            timestamp=datetime.now(UTC),
            parent_analysis_id=root.id,
        )
    )

    assert workspace.analysis_children(str(root.id)) == [child]
    assert workspace.reserved_node_ids() == {str(node_id)}

    grandchild = AnalysisRecord.create(
        ConcordanceDetachmentAnalysisRequest(
            node_id=node_id,
            selected_columns=["match"],
        ),
        timestamp=datetime.now(UTC),
        parent_analysis_id=child.id,
    )
    with pytest.raises(ValueError, match="root parent"):
        workspace.add_analysis(grandchild)


def test_workspace_separates_live_visibility_from_detached_reservations() -> None:
    workspace = Workspace(name="analyses")
    node_id = uuid.uuid4()
    root = workspace.add_analysis(
        AnalysisRecord.create(
            ConcordanceAnalysisRequest(
                node_ids=[node_id],
                node_columns={node_id: "text"},
                search_word="word",
            ),
            timestamp=datetime.now(UTC),
        )
    )
    tab = Tab.create(
        kind=AnalysisKind.CONCORDANCE,
        name="Concordance",
        timestamp=datetime.now(UTC),
    )
    tab.analysis_id = root.id
    workspace.add_tab(tab)

    assert workspace.live_analysis_ids() == {str(root.id)}
    assert workspace.analysis_tab_id(str(root.id)) == str(tab.id)

    tab.analysis_id = None

    assert workspace.live_analysis_ids() == set()
    assert workspace.analysis_tab_id(str(root.id)) is None
    assert workspace.reserved_node_ids() == {str(node_id)}


def test_analysis_transition_methods_preserve_request_and_advance_revision() -> None:
    created_at = datetime.now(UTC)
    record = AnalysisRecord.create(_concordance(), timestamp=created_at)

    running = record.start(created_at + timedelta(seconds=1))
    requested = running.request_running_cancellation(
        created_at + timedelta(seconds=2)
    )
    repeated = requested.request_running_cancellation(
        created_at + timedelta(seconds=3)
    )
    cancelled = requested.confirm_cancelled(
        created_at + timedelta(seconds=4),
        progress=Progress(fraction=0.5, message="Stopping"),
    )

    assert running.revision == 2
    assert requested.revision == 3
    assert repeated is requested
    assert cancelled.revision == 4
    assert cancelled.request == record.request
    assert cancelled.state is AnalysisState.CANCELLED
    assert cancelled.progress.fraction == 0.5


def test_queued_cancellation_and_interrupted_failure_have_exact_timestamps() -> None:
    created_at = datetime.now(UTC)
    record = AnalysisRecord.create(_concordance(), timestamp=created_at)
    cancelled_at = created_at + timedelta(seconds=1)

    cancelled = record.cancel_queued(cancelled_at)
    failed = record.fail(
        cancelled_at,
        failure=Failure(code="analysis_interrupted", message="Analysis interrupted"),
        progress=record.progress,
    )

    assert cancelled.cancellation_requested_at == cancelled_at
    assert cancelled.finished_at == cancelled_at
    assert cancelled.started_at is None
    assert failed.state is AnalysisState.FAILED
    assert failed.started_at is None


def test_success_is_one_atomic_validated_transition() -> None:
    created_at = datetime.now(UTC)
    record = AnalysisRecord.create(_concordance(), timestamp=created_at)
    running = record.start(created_at + timedelta(seconds=1))

    succeeded = running.succeed(
        created_at + timedelta(seconds=2),
        result_payload={"kind": "concordance"},
    )

    assert succeeded.state is AnalysisState.SUCCEEDED
    assert succeeded.progress.fraction == 1.0
    assert succeeded.result_payload == {"kind": "concordance"}
    assert succeeded.output_node_ids == []
    with pytest.raises(ValueError, match="running"):
        record.succeed(created_at, result_payload={"kind": "concordance"})


def test_success_records_query_snapshot_as_a_private_explicit_dependency() -> None:
    created_at = datetime.now(UTC)
    running = AnalysisRecord.create(
        _concordance(), timestamp=created_at
    ).start(created_at + timedelta(seconds=1))
    query_snapshot = AnalysisQuerySnapshotRecord(
        relative_path=f"analyses/{running.id}/query-input"
    )

    succeeded = running.succeed(
        created_at + timedelta(seconds=2),
        result_payload={"kind": "concordance"},
        query_snapshot=query_snapshot,
    )

    assert succeeded.query_snapshot == query_snapshot
    assert "query_snapshot" not in public_analysis(
        succeeded, integrity=ValidAnalysisIntegrity()
    ).model_dump()

    invalid = running.model_dump()
    invalid["query_snapshot"] = query_snapshot.model_dump()
    with pytest.raises(ValidationError, match="successful"):
        AnalysisRecord.model_validate(invalid)


def test_analysis_output_node_ids_are_required_unique_and_strictly_plural() -> None:
    created_at = datetime.now(UTC)
    first = uuid.uuid4()
    second = uuid.uuid4()
    succeeded = AnalysisRecord.create(
        _concordance(), timestamp=created_at
    ).start(created_at + timedelta(seconds=1)).succeed(
        created_at + timedelta(seconds=2),
        result_payload={"kind": "concordance"},
        output_node_ids=[first, second],
    )

    assert succeeded.output_node_ids == [first, second]
    assert public_analysis(
        succeeded, integrity=ValidAnalysisIntegrity()
    ).output_node_ids == [first, second]

    missing = succeeded.model_dump(exclude={"output_node_ids"})
    with pytest.raises(ValidationError):
        AnalysisRecord.model_validate(missing)

    singular = succeeded.model_dump(exclude={"output_node_ids"})
    singular["output_node_id"] = first
    with pytest.raises(ValidationError):
        AnalysisRecord.model_validate(singular)

    duplicate = succeeded.model_dump()
    duplicate["output_node_ids"] = [first, first]
    with pytest.raises(ValidationError, match="unique"):
        AnalysisRecord.model_validate(duplicate)
