# API Patterns

## Router Shape

Backend routers should be thin. The normal structure is:

1. import Pydantic request/response models from the owning domain module under
   `ldaca_wordflow.models` (`models.nodes`, `models.files`,
   `models.workspace`, or the analysis-specific modules);
2. resolve the current user with `Depends(get_current_user)`;
3. validate route-level identifiers and simple request invariants;
4. call a workspace, core, analysis, or worker helper;
5. return a stable JSON payload.

Avoid putting large business workflows directly inside endpoint functions when
the logic can live in `core/` or `analysis/`.

FastAPI operation IDs are generated from route function names so the frontend
hey-api SDK has stable, readable function names. Name new endpoint functions as
the frontend should import them, and avoid duplicate route names.

## Workspace Scoping

Workspace-scoped routes should put `workspace_id` in the path when the caller
already knows the target workspace. Prefer routes such as
`/workspaces/{workspace_id}`, `/workspaces/{workspace_id}/graph`, and
`/workspaces/{workspace_id}/nodes/{node_id}/data` over relying on the user's
hidden current-workspace pointer. Node CRUD, node transforms, annotation,
analysis, and export routes are all workspace-scoped and should live under
`/workspaces/{workspace_id:uuid}/...`. Route handlers must accept the
`workspace_id` path parameter and resolve the target workspace from that value,
typically with `require_workspace(user_id, str(workspace_id))`; do not recover
the target from `workspace_manager.get_current_workspace_id()` or
`workspace_manager.get_current_workspace()`, and do not hide this resolution in a
router-level preload dependency. Current workspace selection is user-scoped UI
session state at `GET/PUT /users/me/current-workspace`; do not use it as a
workspace-scoped data target. Workspace download and unload also take
`workspace_id` in the path, so row-level manager actions do not accidentally
target whatever workspace is currently loaded. Declare explicit workspace path
parameters with the UUID converter (`/{workspace_id:uuid}`) so static collection
or inventory subroutes such as `/upload` and `/tokenizer-models` remain
reachable.

## Authentication

Protected routes should use:

```python
current_user: dict = Depends(get_current_user)
```

Do not bypass this dependency in new routes. Single-user mode and multi-user
token validation are already handled there.

Public bootstrap endpoints should expose only data needed before login. Runtime
configuration follows this split: `GET /runtime-config` is read-only and public,
while mutable server settings live under admin routes such as
`PATCH /admin/config` and must pass the admin allowlist check.

## Response Conventions

Routes that return JSON should declare a concrete `response_model`. The frontend
OpenAPI client is generated from these models, so leaving a route as a bare
`dict` or `Any` response forces downstream `unknown` types and handwritten
adapter casts. Keep dynamic row payloads as `dict[str, Any]`, but type the
envelope, pagination, sorting, metadata, task state, and operation result fields.
Routes that intentionally return non-JSON data should keep `Response`,
`StreamingResponse`, or `FileResponse`, but declare `responses={...}` metadata
with the expected media type and a short schema/description. This applies to raw
text, binary downloads, ZIP artifacts, exports, redirects, and SSE streams.

Application errors should raise `AppError` subclasses from
`core.exceptions`. FastAPI serializes those through the shared
`ErrorResponse` envelope:

```json
{
  "error": "workspace_not_found",
  "message": "Workspace not found",
  "details": null
}
```

The app-level OpenAPI metadata documents common `400`, `401`, `403`, `404`,
`409`, `410`, `500`, and `502` responses with this envelope. Do not import or
raise raw FastAPI `HTTPException` in application modules; add or reuse a
semantic `AppError` subclass instead. `tests/unit/test_app_error_usage.py`
guards this rule and intentionally allows raw `HTTPException` only inside
`core.exceptions`, where the shared `AppError` base class is defined.

Many task and operation routes return a state envelope:

```json
{
  "state": "successful",
  "data": {},
  "message": "..."
}
```

Long-running operations should return a `task_id`. Clients should cancel or
clear by `task_id`, not by task type. Task type is for grouping and display.
For tabbed analysis views, the backend does not own a global "current" task:
the tab sidecar maps each `tab_id` to its task id, and request/result endpoints
must remain addressable by that explicit id even after sibling tabs run. Keep
`tab_id` out of analysis request/result payloads; it is frontend sidecar state.

Node/table responses use the shared API models in `core/api_models.py` where
possible. Column schema entries include both the Polars dtype string and a
frontend-oriented `js_type`.

## Polars Laziness

Keep Polars plans lazy through transformations. Collect only at clear
boundaries:

- data preview and paginated API response serialization,
- artifact writing,
- format export where the target format requires eager data,
- final worker outputs that cannot stay lazy.

This matters because workspace nodes store LazyFrame plans and because large
corpora should not be fully materialized during routine graph edits.

## Path Safety

Any endpoint that accepts a path must resolve it under the user's allowed data
root or workspace root. Do not trust client-provided relative paths. ZIP import
and export code should validate entries before writing them.

## Adding A Worker Task

To add a long-running job:

1. implement a picklable worker wrapper in `core/worker.py` or a dedicated
   `worker_tasks_*.py` module;
2. call environment configuration before heavy imports;
3. register the task type in `TASK_REGISTRY`;
4. submit through `workspace_manager.get_task_manager(user_id)`;
5. update completion handling if the worker creates nodes, stores analysis
   results, or emits materialization events;
6. add frontend handling only through task stream events and query invalidation,
   not polling loops.

## Adding A Workspace Operation

For node/workspace operations, prefer existing helpers in
`api/workspaces/utils.py`, `schema_filter.py`, and `core/polars_operations.py`.
Preserve tokenization metadata when a transformation keeps the tokenized source
column. Drop or invalidate tokenization metadata when the source user column is
removed or renamed.
