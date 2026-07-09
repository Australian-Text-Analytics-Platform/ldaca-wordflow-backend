"""File management package — aggregate router.

Used by:
- FastAPI router registration in main.py.

Flow:
- ``main.py`` imports ``from .api.files import router as files_router`` and
  mounts it at ``/api``. The top-level ``router`` here carries the ``/files``
  prefix so every sub-router inherits the correct path nesting.
- Each sub-module defines its own ``router = APIRouter()`` with the routes it
  owns. The aggregate router includes them here so route ownership is explicit
  while public file URLs stay under one ``/api/files`` resource tree.
"""

from pathlib import Path

from fastapi import APIRouter

from ...core.auth import get_current_user
from ...core.oni_client import OniClient
from ...core.workspace import workspace_manager
from ...settings import settings
from . import crud, ldaca, preview, sample_data, tasks
from .crud import _build_file_tree

router = APIRouter(prefix="/files", tags=["files"])
router.include_router(ldaca.router)
router.include_router(crud.router)
router.include_router(sample_data.router)
router.include_router(tasks.router)
router.include_router(preview.router)


def validate_file_path(file_path: Path, user_folder: Path) -> bool:
    """Validate that file path is within user's allowed directory.

    Sole consumer of this helper is the files API router — keeping it here
    avoids an extra import hop.

    Used by:
    - file-management route handlers in this module.
    """
    try:
        file_path.resolve().relative_to(user_folder.resolve())
        return True
    except ValueError:
        return False
