"""Typed filesystem-artifact projections for persisted Analysis results."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from ..models.analysis_results import (
    TokenFrequencyWorkerResult,
    TopicModelingWorkerResult,
)

ArtifactPath = tuple[str | int, ...]
ArtifactProjection = tuple[ArtifactPath, str]
ArtifactProjector = Callable[[BaseModel], list[ArtifactProjection]]


def no_artifacts(_result: BaseModel) -> list[ArtifactProjection]:
    """Declare that an Analysis kind has no filesystem paths in its result."""

    return []


def token_frequency_artifacts(result: BaseModel) -> list[ArtifactProjection]:
    value = TokenFrequencyWorkerResult.model_validate(result)
    projected: list[ArtifactProjection] = [
        (
            ("artifacts", "nodes", index, "token_parquet_path"),
            node.token_parquet_path,
        )
        for index, node in enumerate(value.artifacts.nodes)
    ]
    if value.artifacts.statistics_parquet_path is not None:
        projected.append(
            (
                ("artifacts", "statistics_parquet_path"),
                value.artifacts.statistics_parquet_path,
            )
        )
    projected.extend(
        (
            ("artifacts", "input_token_streams", index, "token_stream_parquet_path"),
            stream.token_stream_parquet_path,
        )
        for index, stream in enumerate(value.artifacts.input_token_streams)
    )
    return projected


def topic_modeling_artifacts(result: BaseModel) -> list[ArtifactProjection]:
    value = TopicModelingWorkerResult.model_validate(result)
    return [
        (
            ("artifacts", "topic_meanings_parquet_path"),
            value.artifacts.topic_meanings_parquet_path,
        ),
        *[
            (
                ("artifacts", "nodes", index, "assignments_parquet_path"),
                node.assignments_parquet_path,
            )
            for index, node in enumerate(value.artifacts.nodes)
        ],
    ]


ANALYSIS_ARTIFACT_PROJECTORS: dict[str, ArtifactProjector] = {
    "token_frequency": token_frequency_artifacts,
    "topic_modeling": topic_modeling_artifacts,
    "concordance": no_artifacts,
    "quotation": no_artifacts,
    "sequential": no_artifacts,
}


__all__ = [
    "ANALYSIS_ARTIFACT_PROJECTORS",
    "ArtifactProjection",
    "ArtifactProjector",
]
