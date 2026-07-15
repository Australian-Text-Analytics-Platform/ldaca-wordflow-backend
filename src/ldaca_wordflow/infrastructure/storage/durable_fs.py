"""Crash-safe same-filesystem publication primitives.

All backend persistence boundaries use these helpers so directory creation,
file flushing, atomic replacement, and parent-directory flushing have one
platform contract.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class AtomicWriteCapacityError(ValueError):
    """Serialized content exceeds a caller-owned persistence budget."""


def fsync_directory(path: Path) -> None:
    """Durably record directory-entry changes where the platform supports it."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def mkdir_durable(path: Path) -> None:
    """Create a directory chain and flush each newly published parent entry."""

    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    path.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):
        fsync_directory(created.parent)


def fsync_file_and_parent(path: Path) -> None:
    """Flush a complete file and the directory entry that names it."""

    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    fsync_directory(path.parent)


@contextmanager
def atomic_output_path(target: Path) -> Iterator[Path]:
    """Yield a same-directory temporary path and publish it on clean exit."""

    mkdir_durable(target.parent)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(raw_path)
    try:
        yield temporary
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        fsync_directory(target.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(
    target: Path,
    payload: Any,
    *,
    max_bytes: int | None = None,
) -> int:
    """Serialize bounded JSON without exposing a truncated destination."""

    content = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if max_bytes is not None and len(content) > max_bytes:
        raise AtomicWriteCapacityError("Serialized JSON exceeds its storage budget")
    with atomic_output_path(target) as temporary:
        temporary.write_bytes(content)
    return len(content)


__all__ = [
    "AtomicWriteCapacityError",
    "atomic_output_path",
    "atomic_write_json",
    "fsync_directory",
    "fsync_file_and_parent",
    "mkdir_durable",
]
