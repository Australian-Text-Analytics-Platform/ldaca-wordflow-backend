"""Canonical retained sample User File Import contract."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from fastapi.testclient import TestClient

from ldaca_wordflow.main import create_app
from ldaca_wordflow.models.data_sources import (
    SampleCatalogueResource,
    SampleCollection,
    SampleFile,
)
from ldaca_wordflow.services.sample_data import SampleDataService
from ldaca_wordflow.settings import Settings


def test_sample_collection_is_atomically_installed_by_a_retained_import(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundled = tmp_path / "bundled"
    source = bundled / "example" / "documents.csv"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"text\nhello\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    catalogue = SampleCatalogueResource(
        schema_version=1,
        collections=[
            SampleCollection(
                id="example",
                name="Example",
                bundled=True,
                total_size_bytes=source.stat().st_size,
                files=[
                    SampleFile(
                        path="example/documents.csv",
                        size=source.stat().st_size,
                        sha256=digest,
                    )
                ],
            )
        ],
    )

    async def fake_catalogue(_self: SampleDataService) -> SampleCatalogueResource:
        _self._bundled_root = bundled
        return catalogue

    monkeypatch.setattr(SampleDataService, "_fetch_catalogue", fake_catalogue)
    settings = Settings(
        data_root=tmp_path / "data-root",
        multi_user=False,
        session_cookie_secure=False,
        cors_allowed_origins=("http://testserver",),
        trusted_hosts=("testserver",),
    )
    with TestClient(
        create_app(settings, serve_frontend=False),
        base_url="http://testserver",
    ) as client:
        csrf = client.get("/api/session").json()["csrf_token"]
        unsafe = {"Origin": "http://testserver", "X-CSRF-Token": csrf}
        listed = client.get("/api/sample-collections")
        assert listed.status_code == 200
        assert listed.json()["collections"][0]["installed"] is False

        accepted = client.post(
            "/api/sample-collections/example/imports",
            headers=unsafe,
        )
        assert accepted.status_code == 202
        assert accepted.headers["location"] == (
            f"/api/user-file-imports/{accepted.json()['id']}"
        )
        deadline = time.monotonic() + 10
        while True:
            resource = client.get(accepted.headers["location"]).json()
            if resource["state"] in {"succeeded", "failed", "cancelled"}:
                break
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert resource["state"] == "succeeded", resource
        assert resource["request"] == {
            "kind": "sample",
            "collection_id": "example",
        }

        content = client.get(
            "/api/user-files/content",
            params={"path": "sample_data/example/documents.csv"},
        )
        assert content.status_code == 200
        assert content.content == source.read_bytes()
        relisted = client.get("/api/sample-collections")
        assert relisted.json()["collections"][0]["installed"] is True

        duplicate = client.post(
            "/api/sample-collections/example/imports",
            headers=unsafe,
        )
        assert duplicate.status_code == 409
