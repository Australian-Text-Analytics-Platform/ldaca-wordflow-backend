"""File CRUD endpoints and tree/folder helpers.

Used by:
- FastAPI router aggregation in ``__init__.py``.

Flow:
- Routes validate user paths, delegate filesystem operations to core utils,
  and return file-tree or response payloads.
"""

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from ...core.auth import get_current_user
from ...core.exceptions import AccessDeniedError, FileNotFoundError, InternalServiceError, InvalidInputError, NotFoundError, ResourceConflictError
from ...core.utils import (
    detect_file_type,
    get_user_data_folder,
    validate_file_path,
    validate_workspace_name,
)
from ...models import (
    CreateFolderRequest,
    CreateFolderResponse,
    FileTreeNodeResponse,
    FileUploadResponse,
    MessageResponse,
    MoveFileRequest,
)

router = APIRouter()
logger = logging.getLogger(__name__)

README_FILENAME = "README.md"


def _relative_path_for_api(path: Path) -> str:
    """Normalize relative paths for API responses using forward slashes.

    Called by:
    - ``_build_file_tree``, ``create_folder``, ``move_file``.
    """
    return path.as_posix()


def _visible_entry_names(names: list[str]) -> list[str]:
    """Return sorted non-hidden directory entries.

    Called by:
    - ``_build_file_tree``.
    """
    return sorted(name for name in names if not name.startswith("."))


def _build_file_tree(data_folder: Path) -> list[dict[str, Any]]:
    """Build a nested file tree rooted at the user's data directory.

    Steps:
    - Walk the data folder top-down, filtering out hidden entries.
    - Accumulate directory and file nodes keyed by relative path.

    Called by:
    - ``get_user_files``.
    """
    root_children: list[dict[str, Any]] = []
    directory_nodes: dict[str, dict[str, Any]] = {"": {"children": root_children}}

    for current_dir, dirnames, filenames in data_folder.walk(top_down=True):
        dirnames[:] = _visible_entry_names(dirnames)
        visible_filenames = _visible_entry_names(filenames)

        relative_dir = current_dir.relative_to(data_folder)
        relative_dir_str = (
            "" if relative_dir == Path(".") else _relative_path_for_api(relative_dir)
        )
        current_children = directory_nodes[relative_dir_str]["children"]

        for dirname in dirnames:
            directory_path = (
                Path(relative_dir_str, dirname) if relative_dir_str else Path(dirname)
            )
            directory_rel = _relative_path_for_api(directory_path)
            directory_node = {
                "name": dirname,
                "path": directory_rel,
                "type": "directory",
                "children": [],
            }
            current_children.append(directory_node)
            directory_nodes[directory_rel] = directory_node

        for filename in visible_filenames:
            file_rel_path = (
                Path(relative_dir_str, filename) if relative_dir_str else Path(filename)
            )
            absolute_file_path = current_dir / filename
            current_children.append(
                {
                    "name": filename,
                    "path": _relative_path_for_api(file_rel_path),
                    "type": "file",
                    "size": absolute_file_path.stat().st_size,
                }
            )

    return root_children


def _resolve_user_file_path(relative_path: str, data_folder: Path) -> Path:
    """Resolve and validate an API-supplied path inside the user's data folder.

    Steps:
    - Join the relative path to the data folder root.
    - Validate that the resolved path stays inside the allowed directory.

    Called by:
    - ``create_folder``, ``move_file``, and ``get_raw_file`` (in
      ``preview.py``).
    """
    file_path = data_folder / relative_path
    if not validate_file_path(file_path, data_folder):
        raise InvalidInputError("Invalid file path")
    return file_path


def _delete_parent_folder_if_redundant(file_path: Path, data_folder: Path) -> None:
    """Delete the file's parent folder when it becomes empty or README-only.

    Steps:
    - Check whether the parent is a valid subdirectory of the data folder.
    - Remove it if empty, or if it only contains a README.md file.

    Used by:
    - ``delete_file`` and ``move_file``.

    Why:
    - Imported datasets often live in their own wrapper folder with a data file
      plus ``README.md``. When the data file is deleted, removing the now-empty
      wrapper avoids leaving behind dead folders in the file browser.
    """
    parent = file_path.parent
    if parent == data_folder or not parent.exists() or not parent.is_dir():
        return

    if not validate_file_path(parent, data_folder):
        return

    remaining_entries = list(parent.iterdir())
    if not remaining_entries:
        parent.rmdir()
        return

    if len(remaining_entries) != 1:
        return

    remaining_entry = remaining_entries[0]
    if (
        remaining_entry.is_file()
        and remaining_entry.name.lower() == README_FILENAME.lower()
    ):
        remaining_entry.unlink()
        parent.rmdir()


@router.get(
    "/", response_model=list[FileTreeNodeResponse], response_model_exclude_none=True
)
async def get_user_files(current_user: dict = Depends(get_current_user)):
    """List user-visible files as a nested tree.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the
      owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - frontend file browser panel.

    Why:
    - The frontend consumes the directory structure directly instead of
      reconstructing it from a flat file listing.
    """
    user_id = current_user["id"]
    data_folder = get_user_data_folder(user_id)
    return _build_file_tree(data_folder)


@router.post("/folders", response_model=CreateFolderResponse)
async def create_folder(
    request: CreateFolderRequest,
    current_user: dict = Depends(get_current_user),
):
    """Create a folder inside the user's data directory.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the
      owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI POST /folders route.
    """
    user_id = current_user["id"]
    data_folder = get_user_data_folder(user_id)

    is_valid, reason = validate_workspace_name(request.name)
    if not is_valid:
        raise InvalidInputError(f"Invalid folder name: {reason}")
    parent_path = request.parent_path.strip()
    parent_folder = (
        data_folder
        if not parent_path
        else _resolve_user_file_path(parent_path, data_folder)
    )

    if not parent_folder.exists() or not parent_folder.is_dir():
        raise NotFoundError(f"Folder {parent_path or '.'} not found")
    folder_path = parent_folder / request.name.strip()
    if not validate_file_path(folder_path, data_folder):
        raise InvalidInputError("Invalid file path")
    if folder_path.exists():
        raise InvalidInputError(f"Folder {request.name.strip()} already exists")
    try:
        folder_path.mkdir(parents=False, exist_ok=False)
    except OSError as exc:
        raise InternalServiceError(f"Failed to create folder: {str(exc)}") from exc
    relative_path = folder_path.relative_to(data_folder)
    return {
        "message": "Folder created",
        "path": _relative_path_for_api(relative_path),
    }


@router.post("/move", response_model=CreateFolderResponse)
async def move_file(
    request: MoveFileRequest,
    current_user: dict = Depends(get_current_user),
):
    """Move a file into another directory inside the user's data folder.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the
      owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI POST /move route.
    """
    user_id = current_user["id"]
    data_folder = get_user_data_folder(user_id)

    source_path = _resolve_user_file_path(request.source_path, data_folder)
    target_directory = (
        data_folder
        if not request.target_directory_path.strip()
        else _resolve_user_file_path(request.target_directory_path, data_folder)
    )

    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"File {request.source_path} not found")
    if not target_directory.exists() or not target_directory.is_dir():
        raise NotFoundError(f"Folder {request.target_directory_path} not found",)
    destination_path = target_directory / source_path.name
    if not validate_file_path(destination_path, data_folder):
        raise InvalidInputError("Invalid file path")
    if destination_path.exists():
        raise InvalidInputError(f"File {destination_path.name} already exists in destination",)
    original_source_path = source_path
    try:
        source_path.rename(destination_path)
        _delete_parent_folder_if_redundant(original_source_path, data_folder)
    except OSError as exc:
        raise InternalServiceError(f"Failed to move file: {str(exc)}") from exc
    return {
        "message": "File moved",
        "path": _relative_path_for_api(destination_path.relative_to(data_folder)),
    }


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile, current_user: dict = Depends(get_current_user)):
    """Upload file to user's data folder.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the
      owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI POST /upload route.
    """
    user_id = current_user["id"]
    data_folder = get_user_data_folder(user_id)

    if not file.filename:
        raise InvalidInputError("No filename provided")
    file_path = data_folder / file.filename

    if file_path.exists():
        raise ResourceConflictError(f"File {file.filename} already exists")
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    file_type = detect_file_type(file.filename)

    return {
        "filename": file.filename,
        "size": len(content),
        "upload_time": str(file_path.stat().st_ctime),
        "file_type": file_type,
        "preview_available": file_type in ["csv", "json", "parquet"],
    }


@router.delete("/{filename:path}", response_model=MessageResponse)
async def delete_file(filename: str, current_user: dict = Depends(get_current_user)):
    """Delete user's file.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the
      owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI DELETE /{filename:path} route.
    """
    user_id = current_user["id"]
    data_folder = get_user_data_folder(user_id)
    file_path = data_folder / filename

    if not validate_file_path(file_path, data_folder):
        raise AccessDeniedError("Access denied: file outside allowed directory")
    if not file_path.exists():
        raise FileNotFoundError(f"File {filename} not found")
    if file_path.is_dir():
        import shutil
        shutil.rmtree(file_path)
    else:
        file_path.unlink()
    _delete_parent_folder_if_redundant(file_path, data_folder)
    return {"message": f"File {filename} deleted successfully"}
