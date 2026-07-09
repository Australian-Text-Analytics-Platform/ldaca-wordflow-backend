"""File preview, file info, raw file, and download endpoints.

Used by:
- FastAPI router aggregation in ``__init__.py``.

Flow:
- ``unified_file_preview`` delegates format-specific preview work.
- ``get_file_info``, ``get_raw_file``, and ``download_file`` serve data directly.
"""

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response, StreamingResponse

from ...core.auth import get_current_user
from ...core.exceptions import (
    AccessDeniedError,
    FileNotFoundError,
    InternalServiceError,
    InvalidInputError,
)
from ...core.utils import (
    detect_file_type,
    get_user_data_folder,
    validate_file_path,
)
from ...models import FileInfoResponse, FilePreviewRequest, FilePreviewResponse
from .crud import _resolve_user_file_path
from .file_preview import build_file_preview_response

router = APIRouter()
TEXT_RESPONSE_SCHEMA = {"schema": {"type": "string"}}
BINARY_RESPONSE_SCHEMA = {"schema": {"type": "string", "format": "binary"}}
RAW_FILE_RESPONSES = {
    200: {
        "description": "Raw UTF-8 file content.",
        "content": {
            "text/plain": TEXT_RESPONSE_SCHEMA,
            "text/markdown": TEXT_RESPONSE_SCHEMA,
        },
    }
}
DOWNLOAD_FILE_RESPONSES = {
    200: {
        "description": "Binary file download.",
        "content": {"application/octet-stream": BINARY_RESPONSE_SCHEMA},
    }
}


# ── Preview ────────────────────────────────────────────────────────────────


@router.post("/preview", response_model=FilePreviewResponse)
async def unified_file_preview(
    req: FilePreviewRequest, current_user: dict = Depends(get_current_user)
):
    """Unified file preview endpoint.

    - Returns supported types based on extension.
    - Provides preview data (first few rows or page slice).
    - For Excel files, returns sheet_names and supports selecting sheet via
      payload.sheet_name.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the
      owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - frontend file preview modal/table.

    Why:
    - Provides one format-aware preview API for heterogeneous file types.
    """
    user_id = current_user["id"]
    data_folder = get_user_data_folder(user_id)
    return build_file_preview_response(req, data_folder=data_folder)


# ── File info ──────────────────────────────────────────────────────────────


@router.get("/info", response_model=FileInfoResponse)
async def get_file_info(
    path: str = Query(..., description="Path relative to the user's data directory"),
    current_user: dict = Depends(get_current_user),
):
    """Get detailed file information.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the
      owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - frontend file detail consumers because path-bearing file reads should use
      one query-parameter contract instead of catch-all routes.
    """
    user_id = current_user["id"]
    data_folder = get_user_data_folder(user_id)
    file_path = data_folder / path

    if not validate_file_path(file_path, data_folder):
        raise AccessDeniedError("Access denied: file outside allowed directory")
    if not file_path.exists():
        raise FileNotFoundError(f"File {path} not found")
    stat = file_path.stat()
    file_type = detect_file_type(path)

    return {
        "filename": path,
        "size_Byte": stat.st_size,
        "created_at": stat.st_ctime,
        "modified_at": stat.st_mtime,
        "file_type": file_type,
    }


# ── Raw file ───────────────────────────────────────────────────────────────


@router.get(
    "/raw",
    response_class=Response,
    responses=RAW_FILE_RESPONSES,
)
async def get_raw_file(
    path: str = Query(..., description="Path relative to the user's data directory"),
    current_user: dict = Depends(get_current_user),
):
    """Return raw UTF-8 text content for a file inside the user's data folder.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the
      owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI GET /raw route.
    """
    user_id = current_user["id"]
    data_folder = get_user_data_folder(user_id)
    file_path = _resolve_user_file_path(path, data_folder)

    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"File {path} not found")
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidInputError("File is not valid UTF-8 text") from exc
    except OSError as exc:
        raise InternalServiceError(f"Error reading file: {str(exc)}") from exc
    media_type = "text/markdown" if file_path.suffix.lower() == ".md" else "text/plain"
    return Response(content=content, media_type=media_type)


# ── Download (catch-all, must be last) ─────────────────────────────────────


@router.get(
    "/content",
    response_class=StreamingResponse,
    responses=DOWNLOAD_FILE_RESPONSES,
)
async def download_file(
    path: str = Query(..., description="Path relative to the user's data directory"),
    current_user: dict = Depends(get_current_user),
):
    """Download user's file.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the
      owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - frontend file download actions because path-bearing file reads should use
      one query-parameter contract instead of catch-all routes.
    """
    user_id = current_user["id"]
    data_folder = get_user_data_folder(user_id)
    file_path = data_folder / path

    if not validate_file_path(file_path, data_folder):
        raise AccessDeniedError("Access denied: file outside allowed directory")
    if not file_path.exists():
        raise FileNotFoundError(f"File {path} not found")
    def iterfile():
        """Stream file content in chunks for download.

        Called by:
        - The ``download_file`` local workflow.
        """

        with open(file_path, mode="rb") as file_like:
            yield from file_like

    download_filename = file_path.name

    return StreamingResponse(
        iterfile(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={download_filename}"},
    )
