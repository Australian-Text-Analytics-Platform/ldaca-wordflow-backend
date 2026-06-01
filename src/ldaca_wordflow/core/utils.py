"""Core utility re-exports.

This module re-exports domain-specific functions from their proper modules
so that existing import paths (``from ldaca_wordflow.core.utils import ...``)
continue to work.  The ``settings`` import is kept here so that tests that
patch ``ldaca_wordflow.core.utils.settings`` via ``monkeypatch.setattr``
still function correctly — they modify the shared singleton object in-place.
"""

from ..settings import settings  # noqa: F401 — needed for monkeypatch.setattr

from .data_loading import (
    detect_file_type,
    load_data_file,
    normalize_dtypes,
    read_text_file,
    read_zip_file,
)
from .sample_data import (
    download_remote_sample_data,
    import_sample_data_for_user,
)
from .serialization import stringify_unsafe_integers
from .user_folders import (
    allocate_workspace_folder,
    ensure_display_folder_name,
    get_user_cache_folder,
    get_user_data_folder,
    get_user_workspace_folder,
    setup_user_folders,
    validate_file_path,
    validate_workspace_name,
)
