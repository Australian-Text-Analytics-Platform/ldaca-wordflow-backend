"""Bounded, transactional workspace ZIP import service tests."""

from __future__ import annotations

import io
import json
import os
import stat
import struct
import unicodedata
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import anyio
import polars as pl
import pytest
from ldaca_wordflow.domain.workspace import Node, Workspace
from ldaca_wordflow.infrastructure.storage.workspace_access import (
    read_workspace_owner,
    write_workspace_owner,
)
from ldaca_wordflow.infrastructure.storage.workspace_store import WorkspaceStore

from ldaca_wordflow.shared.errors import (
    InvalidWorkspaceArchiveError,
    ResourceTooLargeError,
    UploadTooLargeError,
)
from ldaca_wordflow.services.workspace_archives import (
    WorkspaceArchiveLimits,
    WorkspaceArchiveService,
    WorkspaceArchiveStorage,
    _create_workspace_export,
)

from ._storage import unlimited_storage_admission
from ldaca_wordflow.services.response_snapshots import ResponseSnapshotService
from ldaca_wordflow.services.storage_admission import StorageReservation


class ByteSource:
    def __init__(self, content: bytes) -> None:
        self._content = content
        self._offset = 0

    async def read(self, size: int) -> bytes:
        chunk = self._content[self._offset : self._offset + size]
        self._offset += len(chunk)
        await anyio.sleep(0)
        return chunk


class BlockingSource:
    def __init__(self) -> None:
        self.blocked = anyio.Event()
        self._sent = False

    async def read(self, size: int) -> bytes:
        if not self._sent:
            self._sent = True
            return b"PK-partial"
        self.blocked.set()
        await anyio.sleep_forever()
        raise AssertionError("unreachable")


def test_export_rejects_single_node_before_crossing_hard_byte_limit(
    tmp_path: Path,
) -> None:
    """A single Parquet writer cannot overshoot the admitted export ceiling."""

    workspace = Workspace(name="bounded")
    workspace.add_node(
        Node(
            id=str(uuid.uuid4()),
            name="large",
            data=pl.DataFrame({"text": ["x" * 4096]}).lazy(),
        )
    )
    target = tmp_path / "response.zip"

    with pytest.raises(ResourceTooLargeError, match="Workspace export"):
        _create_workspace_export(workspace, target, 64)

    assert not target.exists()
    assert not list(tmp_path.glob(".workspace-export-stage-*"))


class FakeWorkspaceStorage:
    """Minimal injected storage contract with filesystem-backed discovery."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def workspace_staging_root(self) -> Path:
        path = self.root / ".staging"
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def install_staged_archive(
        self,
        user_id: str,
        staging: Path,
        workspace_name: str,
        reservation: StorageReservation,
    ) -> dict[str, Any]:
        """Mirror the canonical UUID-directory final-install boundary."""

        workspace_id = str(uuid.uuid4())
        (staging / ".wordflow-safe-materialized-import").unlink()
        destination = self.root / workspace_id
        metadata_path = staging / "workspace.json"
        metadata = json.loads(metadata_path.read_text())
        workspace = metadata["workspace_metadata"]
        workspace["id"] = workspace_id
        workspace["name"] = workspace_name
        workspace["revision"] = 1
        workspace["created_at"] = "2026-01-01T00:00:00+00:00"
        workspace["modified_at"] = workspace["created_at"]
        metadata_path.write_text(json.dumps(metadata))
        write_workspace_owner(staging, user_id)
        await reservation.recheck_path(staging)
        os.replace(staging, destination)
        return {
            "id": workspace_id,
            "name": workspace_name,
            "description": "",
            "created_at": workspace["created_at"],
            "modified_at": workspace["modified_at"],
            "total_nodes": 0,
            "root_nodes": 0,
            "leaf_nodes": 0,
            "revision": 1,
        }


def _archive(
    entries: list[tuple[str | zipfile.ZipInfo, bytes]],
    *,
    compression: int = zipfile.ZIP_DEFLATED,
) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=compression) as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return stream.getvalue()


def _valid_archive(
    *,
    workspace_id: str | None = None,
    name: str = "Imported",
    tabs: list[dict[str, Any]] | None = None,
) -> bytes:
    node_id = str(uuid.uuid4())
    manifest = {
        "format": "wordflow-materialized-workspace",
        "version": 3,
        "workspace": {
            "id": workspace_id or str(uuid.uuid4()),
            "name": name,
            "description": "",
            "created_at": None,
            "modified_at": None,
        },
        "nodes": [
            {
                "id": node_id,
                "name": "Values",
                "provenance": {"type": "source"},
                "document": None,
                "color": None,
                "tokenization": {},
                "data_file": f"data/{node_id}.parquet",
            }
        ],
        "tabs": tabs or [],
    }
    parquet = io.BytesIO()
    pl.DataFrame({"value": ["hello"]}).write_parquet(parquet)
    return _archive(
        [
            ("workspace/workspace.json", json.dumps(manifest).encode()),
            (f"workspace/data/{node_id}.parquet", parquet.getvalue()),
        ]
    )


def _set_encrypted_flag(content: bytes) -> bytes:
    modified = bytearray(content)
    for signature, flag_offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        position = 0
        while (position := modified.find(signature, position)) != -1:
            flags = struct.unpack_from("<H", modified, position + flag_offset)[0]
            struct.pack_into("<H", modified, position + flag_offset, flags | 1)
            position += 4
    return bytes(modified)


def _corrupt_first_member(content: bytes) -> bytes:
    modified = bytearray(content)
    local = modified.index(b"PK\x03\x04")
    filename_length, extra_length = struct.unpack_from("<HH", modified, local + 26)
    payload = local + 30 + filename_length + extra_length
    modified[payload] ^= 0x01
    return bytes(modified)


def _service(
    storage: FakeWorkspaceStorage,
    *,
    limits: WorkspaceArchiveLimits | None = None,
) -> WorkspaceArchiveService:
    limiter = anyio.CapacityLimiter(4)
    admission = unlimited_storage_admission(storage.root, limiter=limiter)
    return WorkspaceArchiveService(
        cast(WorkspaceArchiveStorage, storage),
        workspace_store=WorkspaceStore(
            max_nodes=10_000,
            max_snapshot_bytes=1024 * 1024 * 1024,
        ),
        response_snapshots=ResponseSnapshotService(
            storage.root / ".response-snapshots",
            admission,
            max_snapshot_bytes=1024 * 1024 * 1024,
            max_concurrent_snapshots=2,
            limiter=limiter,
        ),
        storage_admission=admission,
        max_export_bytes=1024 * 1024 * 1024,
        limits=limits or WorkspaceArchiveLimits(),
        limiter=limiter,
        upload_chunk_size=16,
    )


async def test_valid_archive_is_staged_then_atomically_installed(
    tmp_path: Path,
) -> None:
    storage = FakeWorkspaceStorage(tmp_path)
    workspace_id = str(uuid.uuid4())

    summary = await _service(storage).import_upload(
        "alice",
        "workspace.zip",
        ByteSource(_valid_archive(workspace_id=workspace_id)),
    )

    assert summary["id"] != workspace_id
    assert summary["name"] == "Imported"
    assert summary["revision"] == 1
    assert summary["created_at"] == summary["modified_at"]
    installed = tmp_path / cast(str, summary["id"])
    parquet_files = list((installed / "data").glob("*.parquet"))
    assert len(parquet_files) == 1
    assert pl.read_parquet(parquet_files[0]).to_dicts() == [{"value": "hello"}]
    assert read_workspace_owner(installed) == "alice"
    assert list((tmp_path / ".staging").iterdir()) == []


async def test_archive_rejects_duplicate_root_analysis_tab_references(
    tmp_path: Path,
) -> None:
    storage = FakeWorkspaceStorage(tmp_path)
    timestamp = datetime.now(UTC).isoformat()
    analysis_id = str(uuid.uuid4())
    tabs = [
        {
            "id": str(uuid.uuid4()),
            "kind": "concordance",
            "name": name,
            "analysis_id": analysis_id,
            "created_at": timestamp,
            "modified_at": timestamp,
            "revision": 1,
        }
        for name in ("First", "Second")
    ]

    with pytest.raises(InvalidWorkspaceArchiveError):
        await _service(storage).import_upload(
            "alice",
            "workspace.zip",
            ByteSource(_valid_archive(tabs=tabs)),
        )

    assert list((tmp_path / ".staging").iterdir()) == []


async def test_cancelled_archive_upload_cleans_seekable_temp(tmp_path: Path) -> None:
    storage = FakeWorkspaceStorage(tmp_path)
    source = BlockingSource()

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(
            _service(storage).import_upload,
            "alice",
            "workspace.zip",
            source,
        )
        await source.blocked.wait()
        tasks.cancel_scope.cancel()

    assert list((tmp_path / ".staging").iterdir()) == []


async def test_archive_upload_enforces_actual_compressed_byte_limit(
    tmp_path: Path,
) -> None:
    storage = FakeWorkspaceStorage(tmp_path)
    archive = _valid_archive()
    limits = WorkspaceArchiveLimits(max_archive_bytes=len(archive) - 1)

    with pytest.raises(UploadTooLargeError) as captured:
        await _service(storage, limits=limits).import_upload(
            "alice", "workspace.zip", ByteSource(archive)
        )

    assert captured.value.code == "upload_too_large"
    assert list((tmp_path / ".staging").iterdir()) == []


@pytest.mark.parametrize(
    "entry_name",
    [
        "../workspace.json",
        "/absolute/workspace.json",
        r"C:\\workspace\\workspace.json",
        r"\\\\server\\share\\workspace.json",
        "workspace\\workspace.json",
    ],
)
async def test_archive_rejects_traversal_and_platform_absolute_paths(
    tmp_path: Path,
    entry_name: str,
) -> None:
    storage = FakeWorkspaceStorage(tmp_path)
    archive = _archive([(entry_name, b"{}")])

    with pytest.raises(InvalidWorkspaceArchiveError):
        await _service(storage).import_upload(
            "alice", "workspace.zip", ByteSource(archive)
        )

    assert list((tmp_path / ".staging").iterdir()) == []


@pytest.mark.parametrize("kind", ["symlink", "special"])
async def test_archive_rejects_symlink_and_special_members(
    tmp_path: Path,
    kind: str,
) -> None:
    storage = FakeWorkspaceStorage(tmp_path)
    info = zipfile.ZipInfo("workspace/payload")
    info.create_system = 3
    file_kind = stat.S_IFLNK if kind == "symlink" else stat.S_IFIFO
    info.external_attr = (file_kind | 0o644) << 16
    archive = _archive(
        [
            (
                "workspace/workspace.json",
                json.dumps(
                    {"workspace_metadata": {"id": str(uuid.uuid4()), "name": "Bad"}}
                ).encode(),
            ),
            (info, b"target"),
        ]
    )

    with pytest.raises(InvalidWorkspaceArchiveError):
        await _service(storage).import_upload(
            "alice", "workspace.zip", ByteSource(archive)
        )


@pytest.mark.parametrize(
    "colliding_names",
    [
        ("workspace/Data.txt", "workspace/data.TXT"),
        (
            "workspace/" + unicodedata.normalize("NFC", "cafe\u0301") + ".txt",
            "workspace/" + unicodedata.normalize("NFD", "café") + ".txt",
        ),
    ],
)
async def test_archive_rejects_case_and_unicode_normalized_collisions(
    tmp_path: Path,
    colliding_names: tuple[str, str],
) -> None:
    storage = FakeWorkspaceStorage(tmp_path)
    archive = _archive(
        [
            (
                "workspace/workspace.json",
                json.dumps(
                    {"workspace_metadata": {"id": str(uuid.uuid4()), "name": "Bad"}}
                ).encode(),
            ),
            (colliding_names[0], b"one"),
            (colliding_names[1], b"two"),
        ]
    )

    with pytest.raises(InvalidWorkspaceArchiveError):
        await _service(storage).import_upload(
            "alice", "workspace.zip", ByteSource(archive)
        )


async def test_archive_rejects_encryption_and_bad_crc(tmp_path: Path) -> None:
    storage = FakeWorkspaceStorage(tmp_path)
    encrypted = _set_encrypted_flag(_valid_archive())
    corrupt = _corrupt_first_member(
        _archive(
            [
                (
                    "workspace/workspace.json",
                    json.dumps(
                        {
                            "workspace_metadata": {
                                "id": str(uuid.uuid4()),
                                "name": "Corrupt",
                            }
                        }
                    ).encode(),
                )
            ],
            compression=zipfile.ZIP_STORED,
        )
    )

    for content in (encrypted, corrupt):
        with pytest.raises(InvalidWorkspaceArchiveError):
            await _service(storage).import_upload(
                "alice", "workspace.zip", ByteSource(content)
            )


@pytest.mark.parametrize(
    ("limits", "archive"),
    [
        (
            WorkspaceArchiveLimits(max_members=1),
            _valid_archive(),
        ),
        (
            WorkspaceArchiveLimits(max_member_bytes=4),
            _valid_archive(),
        ),
        (
            WorkspaceArchiveLimits(max_expanded_bytes=8),
            _valid_archive(),
        ),
        (
            WorkspaceArchiveLimits(max_compression_ratio=2.0),
            _archive(
                [
                    (
                        "workspace.json",
                        json.dumps(
                            {
                                "workspace_metadata": {
                                    "id": str(uuid.uuid4()),
                                    "name": "Bomb",
                                }
                            }
                        ).encode(),
                    ),
                    ("bomb.txt", b"A" * 10_000),
                ]
            ),
        ),
    ],
)
async def test_archive_limits_reject_member_count_size_and_ratio(
    tmp_path: Path,
    limits: WorkspaceArchiveLimits,
    archive: bytes,
) -> None:
    storage = FakeWorkspaceStorage(tmp_path)

    with pytest.raises(InvalidWorkspaceArchiveError):
        await _service(storage, limits=limits).import_upload(
            "alice", "workspace.zip", ByteSource(archive)
        )


async def test_archive_rejects_deployment_access_sidecar(
    tmp_path: Path,
) -> None:
    storage = FakeWorkspaceStorage(tmp_path)
    stream = io.BytesIO(_valid_archive())
    with zipfile.ZipFile(stream, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("workspace/access.json", '{"owner_id":"attacker"}')

    with pytest.raises(InvalidWorkspaceArchiveError):
        await _service(storage).import_upload(
            "alice", "workspace.zip", ByteSource(stream.getvalue())
        )

    assert list((tmp_path / ".staging").iterdir()) == []
