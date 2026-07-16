"""Process-isolated LDaCA Data Portal import entrypoint.

The worker receives only explicit immutable settings, a transient portal token,
and private import-owned paths. It never imports FastAPI, runtime state, or
global settings, and it writes only inside the staging/cache directories
prepared by the owning service.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
import tempfile
from contextlib import chdir
from collections.abc import Callable
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import polars as pl
import httpx

from ..infrastructure.providers.tabular_config import load_tabular_config
from ..infrastructure.providers.oni import OniClient, jsonld_value
from ..shared.portable_names import portable_name_error
from .utils import process_entrypoint


@process_entrypoint
def data_portal_import_process(
    *,
    progress_queue: Queue[Any],
    identifier: str,
    requested_name: str | None,
    api_base_url: str,
    api_token: str | None,
    timeout: float,
    download_concurrency: int,
    staging_dir: str,
    cache_dir: str,
    max_output_bytes: int,
) -> dict[str, object]:
    """Tabulate/download one portal object into a private completed directory."""

    report = cast(Callable[[dict[str, object]], None], progress_queue.put)
    staging = Path(staging_dir).resolve(strict=True)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=False)
    report({"fraction": 0.05, "message": "Fetching Data Portal metadata"})
    metadata, documents, texts = asyncio.run(
        _fetch_portal_content(
            identifier=identifier,
            api_base_url=api_base_url,
            api_token=api_token,
            timeout=timeout,
            download_concurrency=download_concurrency,
            report=report,
            max_download_bytes=max_output_bytes,
        )
    )
    corpus_name = requested_name or _metadata_name(metadata, identifier)
    folder_name = _safe_name(corpus_name)
    filename = f"{folder_name}.parquet"
    destination = staging / filename

    with chdir(cache), tempfile.TemporaryDirectory(dir=cache) as raw_working:
        working = Path(raw_working)
        if documents:
            _write_documents(documents, texts, destination)
        else:
            report({"fraction": 0.25, "message": "Tabulating RO-Crate metadata"})
            _tabulate_metadata(identifier, metadata, working, destination)

    if destination.stat().st_size > max_output_bytes:
        raise ValueError("Data Portal import exceeds its storage budget")

    readme = staging / "README.md"
    readme.write_text(
        f"# {corpus_name}\n\nSource: {identifier}\n",
        encoding="utf-8",
    )
    total_bytes = destination.stat().st_size + readme.stat().st_size
    if total_bytes > max_output_bytes:
        raise ValueError("Data Portal import exceeds its storage limit")
    report({"fraction": 0.95, "message": "Portal import is ready to publish"})
    return {
        "kind": "data_portal",
        "destination_path": f"LDaCA/{folder_name}",
        "file_count": 2,
        "bytes_written": total_bytes,
    }


async def _fetch_portal_content(
    *,
    identifier: str,
    api_base_url: str,
    api_token: str | None,
    timeout: float,
    download_concurrency: int,
    report: Callable[[dict[str, object]], None],
    max_download_bytes: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Fetch metadata and document text through one worker-owned HTTP pool."""

    async with httpx.AsyncClient(
        base_url=api_base_url.rstrip("/"),
        timeout=timeout,
        follow_redirects=True,
    ) as http_client:
        client = OniClient(
            http_client,
            token=api_token,
            max_json_bytes=min(max_download_bytes, 8 * 1024 * 1024),
        )
        metadata = await client.get_metadata(identifier)
        documents = _select_text_documents(metadata)
        texts: dict[str, str] = {}
        if documents:
            declared_sizes = [
                int(document["content_size"])
                for document in documents
                if isinstance(document.get("content_size"), int)
            ]
            if sum(declared_sizes) > max_download_bytes:
                raise ValueError("Data Portal documents exceed the import storage budget")
            report({"fraction": 0.25, "message": "Downloading portal documents"})
            texts = await client.download_object_texts(
                identifier,
                [str(document["path"]) for document in documents],
                concurrency=download_concurrency,
                max_total_bytes=max_download_bytes,
                max_document_bytes=min(max_download_bytes, 16 * 1024 * 1024),
            )
        return metadata, documents, texts


def _safe_name(value: str) -> str:
    """Return one deterministic, collision-resistant portable storage name."""

    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    stem = (normalized or "ldaca_import")[:180]
    candidate = f"{stem}-{digest}"
    if portable_name_error(candidate, exact=True) is not None:
        raise RuntimeError("Portal storage-name sanitizer produced an invalid name")
    return candidate


def _metadata_name(metadata: dict[str, Any], fallback: str) -> str:
    entities = metadata.get("@graph", [])
    root = next(
        (
            entity
            for entity in entities
            if isinstance(entity, dict) and entity.get("@id") in {"./", fallback}
        ),
        None,
    )
    name = jsonld_value(root.get("name")) if isinstance(root, dict) else None
    if name is None:
        name = jsonld_value(metadata.get("name"))
    if isinstance(name, list):
        name = next((item for item in name if item), None)
    return str(name or fallback)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _first_string(value: Any) -> str | None:
    normalized = jsonld_value(value)
    if isinstance(normalized, list):
        normalized = next((item for item in normalized if item), None)
    return str(normalized) if normalized is not None else None


def _reference_id(value: Any) -> str | None:
    normalized = _first_string(value)
    return normalized if normalized else None


def _content_size(value: Any) -> int | None:
    normalized = _first_string(value)
    if normalized is None:
        return None
    try:
        size = int(normalized)
    except ValueError:
        return None
    return size if size >= 0 else None


def _file_path(file_id: str) -> str | None:
    parsed = urlparse(file_id)
    query_path = parse_qs(parsed.query).get("path")
    if query_path:
        return query_path[0]
    return file_id.removeprefix("./") if not parsed.scheme else None


def _select_text_documents(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    entities = {
        entity.get("@id"): entity
        for entity in metadata.get("@graph", [])
        if isinstance(entity, dict) and entity.get("@id")
    }
    selected: dict[str, dict[str, Any]] = {}
    for entity in entities.values():
        file_id = entity.get("@id")
        if not isinstance(file_id, str):
            continue
        types = {str(item) for item in _as_list(jsonld_value(entity.get("@type")))}
        encodings = {
            str(item).casefold()
            for item in _as_list(jsonld_value(entity.get("encodingFormat")))
        }
        if "File" not in types or "text/plain" not in encodings:
            continue
        path = _file_path(file_id)
        if not path:
            continue
        annotation_of = _reference_id(
            entity.get("ldac:annotationOf") or entity.get("annotationOf")
        )
        work = entities.get(annotation_of) if annotation_of else None
        key = annotation_of or path.removesuffix("-plain.txt").removesuffix(".txt")
        candidate = {
            "file_id": file_id,
            "path": path,
            "name": _first_string(entity.get("name")) or path,
            "encoding_format": "text/plain",
            "content_size": _content_size(entity.get("contentSize")),
            "annotation_of": annotation_of,
            "work_name": _first_string(work.get("name"))
            if isinstance(work, dict)
            else None,
            "date_created": _first_string(work.get("dateCreated"))
            if isinstance(work, dict)
            else None,
        }
        current = selected.get(key)
        if current is None or (
            "-plain." in path and "-plain." not in str(current["path"])
        ):
            selected[key] = candidate
    return sorted(selected.values(), key=lambda item: str(item["path"]))


def _write_documents(
    documents: list[dict[str, Any]],
    texts: dict[str, str],
    destination: Path,
) -> None:
    rows = [
        {**document, "text": texts[str(document["path"])]}
        for document in documents
    ]
    pl.DataFrame(rows).write_parquet(destination)


def _tabulate_metadata(
    identifier: str,
    metadata: dict[str, Any],
    working: Path,
    destination: Path,
) -> None:
    from .._vendor.rocrate_tabular.tabulator import ROCrateTabulator

    crate = working / "crate"
    crate.mkdir()
    (crate / "ro-crate-metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    config = load_tabular_config(identifier)
    config_path = working / "tabular-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    database = working / "rocrate.sqlite"
    tabulator = ROCrateTabulator()
    try:
        tabulator.load_config(str(config_path))
        tabulator.crate_to_db(crate, database)
        for table_name in config.get("tables", {}):
            tabulator.entity_table(table_name)
    finally:
        tabulator.close()
    tables = config.get("tables", {})
    table_name = next(
        (
            name
            for name in ("RepositoryObject", "CreativeWork", "File")
            if name in tables
        ),
        next(iter(tables), "property"),
    )
    quoted = '"' + table_name.replace('"', '""') + '"'
    with sqlite3.connect(database) as connection:
        frame = pl.read_database(f"SELECT * FROM {quoted}", connection)
    frame.write_parquet(destination)


__all__ = ["data_portal_import_process"]
