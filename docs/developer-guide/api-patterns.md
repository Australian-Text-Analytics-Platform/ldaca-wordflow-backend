# API Patterns

## Router Shape

Backend routers should be thin. The normal structure is:

1. declare Pydantic request/response models near the route or in `core/`;
2. resolve the current user with `Depends(get_current_user)`;
3. validate route-level identifiers and simple request invariants;
4. call a workspace, core, analysis, or worker helper;
5. return a stable JSON payload.

Avoid putting large business workflows directly inside endpoint functions when
the logic can live in `core/` or `analysis/`.

FastAPI operation IDs are generated from route function names so the frontend
hey-api SDK has stable, readable function names. Name new endpoint functions as
the frontend should import them, and avoid duplicate route names.

## Authentication

Protected routes should use:

```python
current_user: dict = Depends(get_current_user)
```

Do not bypass this dependency in new routes. Single-user mode and multi-user
token validation are already handled there.

## Response Conventions

Routes that return JSON should declare a concrete `response_model`. The frontend
OpenAPI client is generated from these models, so leaving a route as a bare
`dict` or `Any` response forces downstream `unknown` types and handwritten
adapter casts. Keep dynamic row payloads as `dict[str, Any]`, but type the
envelope, pagination, sorting, metadata, task state, and operation result fields.

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
