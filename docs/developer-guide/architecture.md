# Backend Architecture

The backend is a FastAPI application published as the `ldaca-wordflow` Python
package. It serves the REST API, optional bundled frontend assets, workspace
state, user-scoped files, authentication, and long-running analysis tasks.

## Runtime Responsibilities

The backend owns:

- application startup and shutdown,
- runtime settings and data-root resolution,
- SQLite user/session state,
- workspace discovery, loading, saving, and deletion,
- user file browsing and imports,
- workspace node operations over Polars LazyFrames,
- analysis task submission and result storage,
- process-pool worker execution,
- task and workspace events over SSE.

Routers under `src/ldaca_wordflow/api/` are the HTTP boundary. They should
stay thin: validate request shape, resolve the current user, call a core or
analysis helper, and return a predictable payload.

## Main Components

`main.py` creates the FastAPI app, configures CORS, initializes folders and the
database, starts model/cache maintenance, registers routers, and optionally
mounts the built frontend from package resources.

`settings.py` is the runtime configuration source. It derives paths from
`DATA_ROOT` and supports `reload_settings()` for in-process updates.

`db.py` manages async SQLite state for users and sessions via `aiosqlite`. Multi-user mode
uses access tokens stored in the database; single-user mode synthesizes a root
user.

`core/workspace.py` provides the `WorkspaceManager`, the backend's user-scoped
bridge around `docworkspace.Workspace`.

`analysis/` and `core/worker_task_manager.py` split task concerns: analysis
tasks are user-visible result records, while worker tasks are process-pool jobs
that can report progress and emit events.

## Persistence Model

Backend application data defaults to `~/Documents/ldaca` unless `DATA_ROOT`
overrides it. Each user has a data folder containing uploaded files, saved
workspaces, caches, and generated artifacts.

A saved workspace is a directory with `metadata.json` and `data/` files. The
metadata describes workspace and node relationships; `.plbin` files hold
serialized Polars LazyFrame plans. Data files that LazyFrame scans depend on
live beside those plans when possible.

## Request Flow

1. Authentication resolves a user through `get_current_user`.
2. A router loads or validates request inputs.
3. Workspace routes resolve the user's active workspace through
   `workspace_manager`.
4. Short operations run inline and save the workspace.
5. Long operations submit a worker task and return a `task_id`.
6. Worker completion updates analysis state or materializes nodes, then emits
   task/workspace events over `/api/tasks/stream`.

The backend guide pages document each of those parts in more detail.
