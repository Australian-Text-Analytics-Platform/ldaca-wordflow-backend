"""Shared annotation value-object contract tests."""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from ldaca_wordflow.domain import AnnotationClass as DomainAnnotationClass
from ldaca_wordflow.domain.workspace import (
    AnnotationAnalysisRequest,
    AnnotationAnalysisSubmission,
    persisted_submission,
)
from ldaca_wordflow.models.annotations import (
    AnnotationModelsRequest,
)


def _classes() -> list[DomainAnnotationClass]:
    return [
        DomainAnnotationClass(name="Relevant", description="Keep"),
        DomainAnnotationClass(name="relevant", description="Duplicate"),
    ]


def test_annotation_class_is_one_shared_strict_value_contract() -> None:
    annotation_class = DomainAnnotationClass(name="  Relevant  ")
    assert annotation_class.name == "Relevant"
    assert annotation_class.description == ""

    with pytest.raises(ValidationError):
        DomainAnnotationClass.model_validate({"name": "Relevant", "unknown": True})
    with pytest.raises(ValidationError):
        setattr(annotation_class, "name", "Changed")


def test_analysis_request_rejects_duplicate_class_names() -> None:
    with pytest.raises(ValidationError, match="class names must be unique"):
        AnnotationAnalysisRequest(
            node_id=uuid.uuid4(),
            text_column="text",
            annotation_column="class",
            class_node_id=uuid.uuid4(),
            class_column="class",
            description_column="description",
            classes=_classes(),
            provider_configuration_id=uuid.uuid4(),
            provider="openai",
            model="model",
            instruction="Classify the text",
        )


def test_annotation_submission_persists_only_the_safe_provider_snapshot() -> None:
    configuration_id = uuid.UUID("8edb7484-4b45-4834-bf67-ef113a834fb9")
    submission = AnnotationAnalysisSubmission(
        node_id=uuid.UUID("830961ae-6712-4cd9-872c-258f5255177f"),
        text_column="text",
        annotation_column="class",
        class_node_id=uuid.uuid4(),
        class_column="class",
        description_column="description",
        classes=[DomainAnnotationClass(name="Relevant")],
        provider_configuration_id=configuration_id,
        provider="custom",
        provider_base_url="http://localhost:8080/v1/",
        model="local-model",
        instruction="Classify the text",
        api_key="request-secret",
    )

    persisted = persisted_submission(submission)

    assert isinstance(persisted, AnnotationAnalysisRequest)
    assert persisted.provider_configuration_id == configuration_id
    assert persisted.provider == "custom"
    assert persisted.provider_base_url == "http://localhost:8080/v1"
    assert "request-secret" not in persisted.model_dump_json()


def test_analysis_and_discovery_share_the_safe_provider_snapshot_contract() -> None:
    configuration_id = uuid.UUID("1fe3bfd6-cb0f-4108-a954-24cad5deae20")
    analysis = AnnotationAnalysisRequest(
        node_id=uuid.uuid4(),
        text_column="text",
        annotation_column="class",
        class_node_id=uuid.uuid4(),
        class_column="class",
        description_column="description",
        classes=[DomainAnnotationClass(name="Relevant")],
        provider_configuration_id=configuration_id,
        provider="openrouter",
        model="model",
        instruction="Classify the text",
    )
    discovery = AnnotationModelsRequest(
        provider_configuration_id=configuration_id,
        provider="custom",
        provider_base_url="http://localhost:8080/v1/",
    )

    assert analysis.provider_configuration_id == configuration_id
    assert discovery.provider_base_url == "http://localhost:8080/v1"
    assert discovery.api_key is None
