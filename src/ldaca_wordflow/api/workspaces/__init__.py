"""Unified workspaces API package.

Exports a single FastAPI `router` that combines core workspace endpoints
(`base.py`) and modular analysis endpoints under `analyses/`.

Used by:
- `main.py` router registration because they need this unit's "Unified workspaces API package" behavior.

Why:
- Keeps workspace endpoint composition centralized.

Flow:
- Create the package-level workspace router consumed by `main.py`.
- Include lifecycle, node, base, analysis, and tabs subrouters in one place.
- Re-export the workspace manager so existing tests can patch the historical path.
"""

from fastapi import APIRouter

from ...core.workspace import (
    workspace_manager,
)
from . import (
    annotation,
    base,
    lifecycle,
    nodes_concat,
    nodes_crud,
    nodes_expression,
    nodes_filter,
    nodes_join,
    nodes_replace,
    nodes_slice,
    tabs,
)
from .analyses import (
    concordance,
    quotation,
    sequential_analysis,
    token_frequencies,
    topic_modeling,
)

router = APIRouter()
router.include_router(lifecycle.router)
router.include_router(nodes_filter.router)
router.include_router(nodes_slice.router)
router.include_router(nodes_replace.router)
router.include_router(nodes_concat.router)
router.include_router(nodes_join.router)
router.include_router(nodes_expression.router)
router.include_router(nodes_crud.router)
router.include_router(base.router)
router.include_router(annotation.router)
router.include_router(token_frequencies.router)
router.include_router(sequential_analysis.router)
router.include_router(quotation.router)
router.include_router(concordance.router)
router.include_router(topic_modeling.router)
router.include_router(tabs.router)

__all__ = ["router", "workspace_manager"]
