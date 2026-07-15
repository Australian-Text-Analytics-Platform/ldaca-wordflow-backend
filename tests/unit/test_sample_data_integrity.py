"""Sample imports must honor catalogue size and digest declarations."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from ldaca_wordflow.shared.errors import BadGatewayError
from ldaca_wordflow.models.data_sources import (
    SampleCatalogueResource,
    SampleCollection,
    SampleFile,
)
from ldaca_wordflow.services.sample_data import _copy_verified


def test_bundled_copy_rejects_a_manifest_size_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    destination = tmp_path / "destination.csv"
    source.write_bytes(b"abc")

    with pytest.raises(BadGatewayError, match="integrity"):
        _copy_verified(
            source,
            destination,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            2,
        )

    assert not destination.exists()


def test_collection_manifest_requires_exact_total_and_unique_paths() -> None:
    file = SampleFile(path="sample/data.csv", size=3, sha256="0" * 64)

    with pytest.raises(ValidationError, match="total_size_bytes"):
        SampleCollection(
            id="sample",
            name="Sample",
            total_size_bytes=4,
            files=[file],
        )
    with pytest.raises(ValidationError, match="distinct"):
        SampleCollection(
            id="sample",
            name="Sample",
            total_size_bytes=6,
            files=[file, file],
        )
    with pytest.raises(ValidationError, match="distinct"):
        SampleCollection(
            id="sample",
            name="Sample",
            total_size_bytes=6,
            files=[
                file,
                SampleFile(path="data.csv", size=3, sha256="1" * 64),
            ],
        )


def test_catalogue_requires_unique_portable_collection_ids() -> None:
    collection = SampleCollection(
        id="sample",
        name="Sample",
        total_size_bytes=0,
        files=[],
    )
    with pytest.raises(ValidationError, match="collection IDs"):
        SampleCatalogueResource(
            schema_version=1,
            collections=[
                collection,
                collection.model_copy(update={"id": "SAMPLE"}),
            ],
        )
    for collection_id in (".", "..", "CON", "name."):
        with pytest.raises(ValidationError, match="not portable"):
            SampleCollection(
                id=collection_id,
                name="Invalid",
                total_size_bytes=0,
                files=[],
            )


def test_catalogue_rejects_unknown_schema_versions() -> None:
    with pytest.raises(ValidationError):
        SampleCatalogueResource.model_validate(
            {"schema_version": 2, "collections": []}
        )
