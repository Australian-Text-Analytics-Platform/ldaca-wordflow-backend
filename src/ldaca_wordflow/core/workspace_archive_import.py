"""Workspace archive import service.

Used by:
- ``api.workspaces.lifecycle.upload_workspace_zip`` because the route should
  read the uploaded file and delegate ZIP validation, safe extraction, metadata
  rewriting, directory replacement, and workspace registry refresh here.

Flow:
- Validate the uploaded filename and archive bytes.
- Extract only safe members under the workspace root containing
  ``metadata.json``.
- Preserve a unique incoming workspace id when possible, otherwise allocate a
  fresh id.
- Copy the normalized archive tree into the user's workspace storage and return
  the refreshed workspace summary.
"""

from __future__ import annotations

import io
import json
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from .exceptions import InvalidInputError
from .workspace import workspace_manager


def _safe_member_path(name: str) -> PurePosixPath:
    """Validate and normalize one ZIP member path.

    Used by:
    - ``_extract_workspace_zip`` so uploaded archives cannot write outside the
      temporary extraction root.
    """

    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise InvalidInputError("Invalid zip entry path")
    if any(part in {"", "."} for part in path.parts):
        raise InvalidInputError("Invalid zip entry path")
    return path


def _workspace_root_prefix(safe_paths: list[PurePosixPath]) -> tuple[str, ...]:
    """Find the path prefix whose root contains ``metadata.json``."""

    metadata_candidates = [
        path
        for path in safe_paths
        if path.name == "metadata.json" and "__MACOSX" not in path.parts
    ]
    if not metadata_candidates:
        raise InvalidInputError("ZIP must contain workspace metadata.json")
    metadata_path_in_zip = min(metadata_candidates, key=lambda path: len(path.parts))
    return metadata_path_in_zip.parts[:-1]


def _extract_workspace_zip(file_bytes: bytes, extraction_dir: Path) -> None:
    """Safely extract one workspace ZIP into ``extraction_dir``."""

    with zipfile.ZipFile(io.BytesIO(file_bytes), "r") as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        if not members:
            raise InvalidInputError("ZIP archive is empty")

        safe_paths = [_safe_member_path(member.filename) for member in members]
        root_prefix = _workspace_root_prefix(safe_paths)

        for member, safe_path in zip(members, safe_paths):
            if "__MACOSX" in safe_path.parts:
                continue
            if root_prefix and safe_path.parts[: len(root_prefix)] != root_prefix:
                continue

            relative_parts = safe_path.parts[len(root_prefix) :] if root_prefix else safe_path.parts
            if not relative_parts:
                continue

            relative_path = PurePosixPath(*relative_parts)
            if relative_path.name in {".DS_Store"}:
                continue

            destination = extraction_dir / Path(*relative_path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _workspace_id_for_import(
    metadata: dict[str, Any],
    existing_ids: set[str],
) -> tuple[str, str | None]:
    """Return the imported workspace id and incoming display name."""

    workspace_metadata = metadata.setdefault("workspace_metadata", {})
    incoming_id = workspace_metadata.get("id")
    incoming_name = workspace_metadata.get("name")

    if (
        isinstance(incoming_id, str)
        and incoming_id
        and incoming_id not in existing_ids
    ):
        workspace_id = incoming_id
    else:
        workspace_id = str(uuid.uuid4())

    return workspace_id, incoming_name if isinstance(incoming_name, str) else None


def import_workspace_zip(
    *,
    user_id: str,
    filename: str,
    file_bytes: bytes,
) -> dict[str, Any]:
    """Import one uploaded workspace ZIP and return its workspace summary.

    Used by:
    - ``api.workspaces.lifecycle.upload_workspace_zip`` after FastAPI provides
      the authenticated user and raw upload bytes.

    Flow:
    - Reject non-ZIP filenames and empty uploads.
    - Extract the archive into a temporary normalized workspace root.
    - Rewrite ``metadata.json`` with the final id/name and replace the target
      workspace directory.
    - Refresh workspace discovery and return the imported workspace summary.
    """

    if not filename.lower().endswith(".zip"):
        raise InvalidInputError("Only .zip files are supported")
    if not file_bytes:
        raise InvalidInputError("Uploaded file is empty")

    existing_ids = {
        str(item.get("id"))
        for item in workspace_manager.list_user_workspaces_summaries(user_id)
        if item.get("id")
    }

    try:
        with tempfile.TemporaryDirectory(prefix="workspace_zip_") as temp_dir:
            extraction_dir = Path(tempfile.mkdtemp(prefix="extracted_", dir=temp_dir))
            _extract_workspace_zip(file_bytes, extraction_dir)

            metadata_file = extraction_dir / "metadata.json"
            if not metadata_file.exists():
                raise InvalidInputError(
                    "ZIP missing required metadata.json at workspace root",
                )
            with metadata_file.open("r", encoding="utf-8") as f:
                metadata = json.load(f)

            workspace_id, incoming_name = _workspace_id_for_import(
                metadata,
                existing_ids,
            )
            workspace_name = (
                incoming_name.strip()
                if incoming_name is not None and incoming_name.strip()
                else filename.rsplit(".zip", 1)[0]
            )

            workspace_metadata = metadata.setdefault("workspace_metadata", {})
            workspace_metadata["id"] = workspace_id
            workspace_metadata["name"] = workspace_name
            with metadata_file.open("w", encoding="utf-8") as f:
                json.dump(metadata, f)

            target_dir = workspace_manager._resolve_workspace_dir(
                user_id=user_id,
                workspace_id=workspace_id,
                workspace_name=workspace_name,
            )
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            shutil.copytree(extraction_dir, target_dir)

            workspace_manager._refresh_user_workspace_paths(user_id)
    except zipfile.BadZipFile as exc:
        raise InvalidInputError(f"Invalid ZIP file: {exc}") from exc

    return next(
        (
            item
            for item in workspace_manager.list_user_workspaces_summaries(user_id)
            if item.get("id") == workspace_id
        ),
        {
            "id": workspace_id,
            "name": workspace_name,
        },
    )
