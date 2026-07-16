"""Strict Data Portal document selection and materialization."""

from pathlib import Path

import pytest

from ldaca_wordflow.workers.data_portal import (
    _content_size,
    _select_text_documents,
    _write_documents,
)


def test_select_text_documents_prefers_plain_text_derivatives() -> None:
    metadata = {
        "@graph": [
            {
                "@id": "arcp://name,example/work/1",
                "@type": "CreativeWork",
                "name": "Document 1",
                "dateCreated": "1788",
            },
            {
                "@id": "https://data.ldaca.edu.au/api/stream?path=data%2F1.txt",
                "@type": ["File"],
                "name": "Document 1 with codes",
                "encodingFormat": ["text/plain"],
                "contentSize": "20",
                "ldac:annotationOf": {"@id": "arcp://name,example/work/1"},
            },
            {
                "@id": (
                    "https://data.ldaca.edu.au/api/stream?"
                    "path=data%2F1-plain.txt"
                ),
                "@type": ["File"],
                "name": "Document 1 plain",
                "encodingFormat": ["text/plain"],
                "contentSize": "18",
                "ldac:annotationOf": {"@id": "arcp://name,example/work/1"},
            },
        ]
    }

    assert _select_text_documents(metadata) == [
        {
            "file_id": (
                "https://data.ldaca.edu.au/api/stream?"
                "path=data%2F1-plain.txt"
            ),
            "path": "data/1-plain.txt",
            "name": "Document 1 plain",
            "encoding_format": "text/plain",
            "content_size": 18,
            "annotation_of": "arcp://name,example/work/1",
            "work_name": "Document 1",
            "date_created": "1788",
        }
    ]


def test_document_materialization_requires_every_downloaded_text(
    tmp_path: Path,
) -> None:
    with pytest.raises(KeyError, match="missing.txt"):
        _write_documents(
            [{"path": "missing.txt"}],
            {},
            tmp_path / "documents.parquet",
        )


@pytest.mark.parametrize("value", ["invalid", "-1"])
def test_invalid_declared_content_sizes_are_not_trusted(value: str) -> None:
    assert _content_size(value) is None
