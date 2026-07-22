"""Shared annotation value-object contract tests."""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from ldaca_wordflow.domain import AnnotationClass as DomainAnnotationClass
from ldaca_wordflow.domain.workspace import AnnotationAnalysisRequest
from ldaca_wordflow.models.annotations import (
    AnnotationClass as PreviewAnnotationClass,
    AnnotationConfig,
)


def _classes() -> list[DomainAnnotationClass]:
    return [
        DomainAnnotationClass(name="Relevant", description="Keep"),
        DomainAnnotationClass(name="relevant", description="Duplicate"),
    ]


def test_annotation_class_is_one_shared_strict_value_contract() -> None:
    assert PreviewAnnotationClass is DomainAnnotationClass

    annotation_class = DomainAnnotationClass(name="  Relevant  ")
    assert annotation_class.name == "Relevant"
    assert annotation_class.description == ""

    with pytest.raises(ValidationError):
        DomainAnnotationClass.model_validate({"name": "Relevant", "unknown": True})
    with pytest.raises(ValidationError):
        setattr(annotation_class, "name", "Changed")


def test_preview_and_analysis_requests_reject_duplicate_class_names() -> None:
    with pytest.raises(ValidationError, match="class names must be unique"):
        AnnotationConfig(
            text_column="text",
            annotation_column="class",
            classes=_classes(),
            provider="openai",
            model="model",
            instruction="Classify the text",
        )

    with pytest.raises(ValidationError, match="class names must be unique"):
        AnnotationAnalysisRequest(
            node_id=uuid.uuid4(),
            text_column="text",
            annotation_column="class",
            classes=_classes(),
            provider="openai",
            model="model",
            instruction="Classify the text",
            output_node_name="Annotated",
        )
