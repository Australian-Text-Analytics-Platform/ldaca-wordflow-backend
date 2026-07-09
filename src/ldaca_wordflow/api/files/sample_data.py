"""Sample data catalogue, import, and status helpers.

Used by:
- FastAPI router aggregation in ``__init__.py``.

Flow:
- Catalogue fetches a remote JSON catalogue, computes per-collection status
  via ``_compute_collection_status``, and returns typed responses.
- Import delegates to ``import_sample_data_for_user`` and (optionally)
  schedules a background download for non-bundled remote datasets.
- README proxies individual ``.md`` files from the remote sample-data repo.
"""

import hashlib
import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, Query, Response

from ...core.auth import get_current_user
from ...core.exceptions import AppError, BadGatewayError, InvalidInputError, NotFoundError
from ...core.utils import (
    download_remote_sample_data,
    get_user_data_folder,
    import_sample_data_for_user,
)
from ...models import (
    ImportSampleDataRequest,
    ImportSampleDataResponse,
    SampleDataCatalogueResponse,
    SampleDataCollection,
    SampleDataFileEntry,
)

router = APIRouter()
logger = logging.getLogger(__name__)
README_RESPONSE = {
    200: {
        "description": "README markdown text from the sample-data catalogue.",
        "content": {"text/plain": {"schema": {"type": "string"}}},
    }
}

SampleDataCollectionStatus = Literal[
    "bundled", "downloaded", "partial", "not_downloaded"
]


def _compute_collection_status(
    col: dict, target_dir: Path
) -> SampleDataCollectionStatus:
    """Return status string for a single catalogue collection.

    Steps:
    - Compare local files against catalogue entries.
    - SHA256-verify each file that exists; count present matches.
    - Return the appropriate status based on the ratio of matched files.

    Called by:
    - ``get_sample_data_catalogue``.
    """
    files = col.get("files", [])
    if not files:
        return "not_downloaded"
    present = 0
    for entry in files:
        dest = target_dir / Path(entry.get("path", ""))
        if dest.exists():
            digest = hashlib.sha256(dest.read_bytes()).hexdigest()
            if digest == entry.get("sha256", ""):
                present += 1
    if present == len(files):
        return "bundled" if col.get("bundled") else "downloaded"
    if present > 0:
        return "partial"
    return "not_downloaded"


@router.get("/sample-data/catalogue", response_model=SampleDataCatalogueResponse)
async def get_sample_data_catalogue(
    current_user: dict = Depends(get_current_user),
):
    """Return the sample data catalogue augmented with per-collection status.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the
      owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI GET /sample-data/catalogue route.
    """
    import httpx

    from ...settings import settings

    remote_base = (settings.sample_data_remote_url or "").rstrip("/")
    if not remote_base:
        raise AppError("Sample data remote URL not configured.")
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{remote_base}/catalogue.json")
            resp.raise_for_status()
            catalogue = resp.json()
    except Exception:
        raise BadGatewayError("Could not fetch sample data catalogue.")
    user_id = current_user["id"]
    target_dir = get_user_data_folder(user_id) / "sample_data"

    collections: list[SampleDataCollection] = []
    for col in catalogue.get("collections", []):
        status = _compute_collection_status(col, target_dir)
        collections.append(
            SampleDataCollection(
                id=col["id"],
                name=col["name"],
                description=col["description"],
                language=col["language"],
                bundled=col["bundled"],
                total_size_bytes=col["total_size_bytes"],
                recommended_for=col["recommended_for"],
                files=[SampleDataFileEntry(**f) for f in col["files"]],
                status=status,
            )
        )

    return SampleDataCatalogueResponse(
        schema_version=catalogue.get("schema_version", 1),
        collections=collections,
    )


@router.get(
    "/sample-data/readme",
    response_class=Response,
    responses=README_RESPONSE,
)
async def get_sample_data_readme(
    path: str = Query(
        ..., description="Relative path of the README inside the sample data repo"
    ),
    current_user: dict = Depends(get_current_user),
):
    """Proxy a README.md from the remote sample data repository.

    Only .md files are permitted; path traversal is rejected.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the
      owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI GET /sample-data/readme route.
    """
    from pathlib import PurePosixPath

    import httpx

    from ...settings import settings

    try:
        safe = PurePosixPath(path)
    except Exception:
        raise InvalidInputError("Invalid path.")
    if safe.suffix.lower() != ".md" or any(part == ".." for part in safe.parts):
        raise InvalidInputError("Only .md files are permitted.")
    remote_base = (settings.sample_data_remote_url or "").rstrip("/")
    if not remote_base:
        raise AppError("Sample data remote URL not configured.")
    url = f"{remote_base}/{safe}"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return Response(content=resp.text, media_type="text/plain; charset=utf-8")
    except Exception:
        raise BadGatewayError("Could not fetch README.")
@router.post("/import-sample-data", response_model=ImportSampleDataResponse)
async def import_sample_data(
    background_tasks: BackgroundTasks,
    request: ImportSampleDataRequest = ImportSampleDataRequest(),
    current_user: dict = Depends(get_current_user),
):
    """Import (or re-import) sample data for the current user on demand.

    Copies bundled datasets (SCL, ADO/twitter) immediately, then downloads any
    missing or updated remote datasets if SAMPLE_DATA_REMOTE_URL is configured.

    If request.collection_ids is non-empty, only those collections are
    downloaded remotely. An empty list means "download all non-bundled
    collections" (default).

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the
      owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI POST /import-sample-data route.
    """
    from ...settings import settings

    user_id = current_user["id"]
    try:
        summary = import_sample_data_for_user(user_id)
    except FileNotFoundError as e:
        raise NotFoundError(str(e)) from e
    remote_url = (settings.sample_data_remote_url or "").strip()
    collection_ids: list[str] | None = (
        request.collection_ids if request.collection_ids else None
    )
    if remote_url:
        background_tasks.add_task(download_remote_sample_data, user_id, collection_ids)

    has_remote = bool(remote_url)
    return {
        "status": "ok",
        "removed_existing": summary["removed_existing"],
        "file_count": summary["file_count"],
        "bytes_copied": summary["bytes_copied"],
        "sample_dir": summary["sample_dir"],
        "remote_download_started": has_remote,
        "message": (
            "Sample data imported. Larger datasets are downloading in the background "
            "and will appear in the file browser shortly."
            if has_remote
            else "Sample data imported successfully."
        ),
    }
