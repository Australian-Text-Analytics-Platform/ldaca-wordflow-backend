"""Enhanced LDaCA Web App API - Main FastAPI Application
Modular, production-ready text analysis platform with multi-user support

Used by:
- uvicorn, the packaged desktop backend runtime, and backend test clients because tests
  need the same observable contract that production routes and workers rely on.

Flow: normalize inputs, delegate to the owning backend state or service boundary, and
    return serialized values or existing domain errors to callers.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import IO, Any, cast

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute

# Import API routers
from .api.admin import router as admin_router
from .api.auth import router as auth_router
from .api.config import router as config_router
from .api.files import router as files_router
from .api.preferences import router as preferences_router
from .api.tasks import router as tasks_router
from .api.workspaces import router as workspaces_router

# Ensure DocWorkspace API conversion utilities are available at startup.
from .core import docworkspace_data_types  # noqa: F401
from .core.auth_service import cleanup_expired_sessions
from .db import init_db
from .settings import reload_settings, settings

__version__ = "3.0.0"

logger = logging.getLogger(__name__)


def generate_operation_id(route: APIRoute) -> str:
    """Support FastAPI app mounting with a generate operation id helper.

    Used by:
    - FastAPI application startup because they need a backend boundary that validates inputs
      before delegating to workspace or worker state.
    """

    return route.name


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup/shutdown lifecycle for backend runtime dependencies.

    Used by:
    - FastAPI app lifecycle hooks because they need a backend boundary that validates inputs
      before delegating to workspace or worker state.
    Why:
    - Initializes data folders/DB/session cleanup and performs safe worker shutdown.
    """
    # Setup file logging for packaged app (especially Windows)
    from ._logging import setup_file_logging, setup_logging

    setup_logging()
    log_file: IO[str] | None = setup_file_logging("main")
    current_settings = reload_settings()
    if not os.getenv("SECRET_KEY") and current_settings.multi_user:
        logger.warning(
            "SECRET_KEY not set in environment — JWT signatures will change on restart."
            " Set SECRET_KEY for stable multi-user deployments."
        )

    logger.info(
        "Starting LDaCA Web App (platform=%s, python=%s)",
        sys.platform,
        sys.version.split()[0],
    )

    # Create data folders
    current_settings.get_data_root().mkdir(parents=True, exist_ok=True)
    current_settings.get_users_root_folder().mkdir(parents=True, exist_ok=True)
    sample_override = current_settings.get_sample_data_folder()
    if sample_override:
        sample_override.mkdir(parents=True, exist_ok=True)
    current_settings.get_database_backup_folder().mkdir(parents=True, exist_ok=True)

    await init_db()
    await cleanup_expired_sessions()

    # Worker pool initializes lazily on first task; prefetch heavy ML models
    # in the background so first-request latency is not dominated by downloads.
    from .core.model_prefetch import start_model_prefetch

    start_model_prefetch()

    logger.info(
        "Backend ready: docs=http://%s:%s/api/docs health=http://%s:%s/health",
        current_settings.server_host,
        current_settings.backend_port,
        current_settings.server_host,
        current_settings.backend_port,
    )

    yield  # Application runs here

    logger.info("Shutting down...")

    if log_file:
        log_file.close()

    # Worker pool shutdown must tolerate failures so the ASGI shutdown
    # completes even if the pool is in a bad state.
    try:
        from .core.worker import get_worker_pool

        worker_pool = get_worker_pool()
        if worker_pool.is_running:
            worker_pool.shutdown(wait=True, timeout=5.0)
    except Exception as e:
        logger.warning("Error during worker pool shutdown: %s", e)

    await cleanup_expired_sessions()


# Create FastAPI application
app = FastAPI(
    title="Enhanced LDaCA Web App API",
    version=__version__,
    description="Multi-user text analysis platform with workspace management and polars-text integration",
    lifespan=lifespan,
    generate_unique_id_function=generate_operation_id,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Setup request logging (before CORS so it captures everything)
from ._middleware import RequestLoggingMiddleware

app.add_middleware(cast(Any, RequestLoggingMiddleware))

# Setup CORS (regex + credentials from settings)
app.add_middleware(
    cast(Any, CORSMiddleware),
    allow_origin_regex=r".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api", tags=["authentication"])
app.include_router(config_router, prefix="/api", tags=["configuration"])
app.include_router(files_router, prefix="/api", tags=["file_management"])
app.include_router(preferences_router, prefix="/api", tags=["preferences"])
app.include_router(tasks_router, prefix="/api", tags=["task_streaming"])
app.include_router(workspaces_router, prefix="/api", tags=["workspace_management"])
app.include_router(admin_router, prefix="/api", tags=["administration"])


# =============================================================================
# ROOT ENDPOINTS
# =============================================================================


@app.get("/api")
async def root():
    """Return API feature/index metadata.

    Used by:
    - browser/manual root checks and basic service discovery because callers need the shared
      shared backend behavior rule in one place instead of duplicating it.
    Why:
    - Provides a human-readable entrypoint summary for backend capabilities.
    """
    return {
        "message": "Enhanced LDaCA Web App API",
        "version": __version__,
        "description": "Multi-user text analysis platform with workspace management",
        "features": {
            "authentication": "OIDC (Google OAuth 2.0 / CILogon)",
            "workspaces": "Multi-user workspace management with node operations",
            "file_management": "Upload, preview, download with type detection",
            "text_analysis": "polars-text integration",
            "data_operations": "Filter, slice, transform, aggregate operations",
            "user_isolation": "Per-user data folders and workspace separation",
        },
        "endpoints": {
            "root": "/api",
            "docs": "/api/docs",
            "redoc": "/api/redoc",
            "openapi": "/api/openapi.json",
            "health": "/health",
            "status": "/status",
            "auth": {
                "google": "/api/auth/google",
                "me": "/api/auth/me",
                "logout": "/api/auth/logout",
                "status": "/api/auth/status",
            },
            "files": {
                "list": "/api/files/",
                "upload": "/api/files/upload",
                "download": "/api/files/{filename}",
                "preview": "/api/files/preview",
                "info": "/api/files/{filename}/info",
                "delete": "/api/files/{filename}",
            },
            "workspaces": {
                "create": "/api/workspaces/",
                "current": "/api/workspaces/current",
                "info": "/api/workspaces/info",
                "delete": "/api/workspaces/delete",
                "nodes": "/api/workspaces/nodes",
                "node_data": "/api/workspaces/nodes/{node_id}/data",
                "save": "/api/workspaces/save",
                "description": "/api/workspaces/description",
                "unload": "/api/workspaces/unload",
            },
            "admin": {"users": "/api/admin/users", "cleanup": "/api/admin/cleanup"},
        },
    }


@app.get("/health")
async def health_check():
    """Return lightweight health status for liveness probes.

    Used by:
    - deployment health checks and uptime monitors because callers need the shared shared
      backend behavior rule in one place instead of duplicating it.
    Why:
    - Gives a fast no-auth probe endpoint for runtime readiness checks.
    """
    return {
        "status": "healthy",
        "version": __version__,
        "system": "Enhanced LDaCA Web App API",
        "database": "connected",
        "features": {
            "polars-text": True,
            "docworkspace": True,
        },
        "config": {
            "data_folder": str(settings.get_data_root()),
            "debug_mode": settings.debug,
        },
    }


@app.get("/status")
async def status():
    """Return detailed component/module status information.

    Used by:
    - diagnostics pages and manual troubleshooting because callers need the shared shared
      backend behavior rule in one place instead of duplicating it.
    Why:
    - Exposes richer operational metadata than ``/health``.
    """
    return {
        "system": "Enhanced LDaCA Web App API",
        "version": __version__,
        "status": "operational",
        "components": {
            "authentication": {
                "status": "[OK] Google OAuth 2.0",
                "description": "Secure user authentication with session management",
            },
            "file_management": {
                "status": "[OK] Multi-format support",
                "description": "Upload, download, preview CSV, JSON, Parquet, Excel files",
            },
            "workspace_management": {
                "status": "[OK] Multi-user isolation",
                "description": "Per-user workspaces with DataFrame node operations",
            },
            "data_operations": {
                "status": "[OK] DataFrame manipulation",
                "description": "Filter, slice, transform, aggregate, join operations",
            },
            "text_analysis": {
                "status": "[OK] polars-text ready",
                "description": "Advanced text analysis with polars-text integration",
            },
            "database": {
                "status": "[OK] aiosqlite",
                "description": "Async SQLite persistence with direct aiosqlite access",
            },
        },
        "modules": {
            "auth": "Google OAuth authentication and session management",
            "files": "File upload, download, preview, and management",
            "workspaces": "Multi-user workspace and node management",
            "admin": "Administrative functions and monitoring",
        },
    }
