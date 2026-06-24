# Workspace And Files

## Workspace Manager

`core/workspace.py` contains `WorkspaceManager`. It keeps one active in-memory
`docworkspace.Workspace` per user while allowing many saved workspace folders
on disk.

The manager is responsible for:

- resolving each user's workspace root,
- listing persisted workspace summaries,
- allocating display-name-safe workspace folders,
- loading a selected workspace into memory,
- rebasing serialized LazyFrame scan paths before load,
- saving and unloading the current workspace,
- deleting workspace folders,
- clearing workspace-specific analysis caches and task state.

The key invariant is that API code should not create its own global workspace
state. It should resolve the user's workspace through `workspace_manager`.

## File APIs

`api/files.py` exposes the user's data folder:

- tree listing,
- folder creation,
- upload, move, delete, and raw download,
- preview for CSV, JSON, Parquet, IPC, Excel, and text-like data,
- sample-data import,
- LDaCA RO-Crate import through a background worker task.

LDaCA import is backed by the LDaCA Data Portal Oni API. The files router
exposes `/files/ldaca/featured` for staff-picked collections,
`/files/ldaca/search` for keyword and identifier portal searches, and
`/files/import-ldaca` to submit the selected ARCP identifier as a background
task. Search responses include collection and file-format metadata so the
frontend can filter returned results without exposing Oni credentials. API keys
or bearer tokens stay in backend settings; the frontend never calls Oni
directly.

The import worker fetches RO-Crate metadata through Oni, loads the backend-owned
`ldaca_tabular_configs` resource, tabulates metadata with the vendored
`rocrate-tabular` package, and writes the selected table to
`LDaCA/<corpus>/<corpus>.parquet` in the user's data folder.

The file router validates paths against the user's data root to avoid path
traversal. Data preview prefers lazy Polars scans where possible and collects
only for preview serialization.

## Workspace Lifecycle Routes

`api/workspaces/lifecycle.py` handles workspace CRUD and active-workspace
selection:

- list, create, delete, rename, unload, and set current workspace,
- workspace graph and node summaries,
- workspace save/download as a zip,
- workspace zip upload/import,
- workspace description and metadata.

ZIP import and export treat paths carefully: entries are validated with
`PurePosixPath`, and workspace source paths are rebased after the final folder
location is known.

## Node Operations

`api/workspaces/nodes.py` owns most row/column transformations:

- node data paging, shape, unique values, describe, query plan,
- delete, rename, clone,
- filter and filter preview,
- slice/sample and preview,
- join and concat preview/apply,
- find/replace preview/apply,
- column operations,
- constrained Polars expression preview/apply.

Node operations should preserve laziness. Collection belongs at API response
serialization, preview limits, artifact writing, or other explicit I/O
boundaries.

The Polars expression endpoint validates code with
`core/polars_expr_validator.py`, executes in a restricted environment, and
allows only the supported transformation contexts.

## Schema Filtering

`api/workspaces/schema_filter.py` is the frontend-facing schema filter.
Tokenisation specs are tracked in `Node.tokenization` keyed by source column,
with hydrated column names such as `tokenization.text.lindera:jieba`. The node stores
the source column, selected model, language, and cache metadata for cached
tokens. Frontend node selectors update `Node.document` via
`PUT /api/workspaces/nodes/{node_id}/document-column`; tokenizer preferences are
registered via `PUT /api/workspaces/nodes/{node_id}/tokenization-preference`.
Source-node visualization colours are durable `Node.color` metadata updated via
`POST /api/workspaces/nodes/{node_id}/color`, separate from analysis request
payloads.
Analysis submit paths should read these fields, not mutate them. Analysis paths
resolve the per-user DuckDB cache path and call
`pl.col(...).text.tokenize(..., cache=path)` to hydrate temporary token
structs.
Normal table/schema responses preserve the physical node schema and expose
structured `tokenization` metadata where the UI needs it.

Use this shared projection for frontend node metadata instead of filtering or
fabricating token columns in individual routers.

## Export

`api/workspaces/base.py` still owns export endpoints and some legacy column
routes. Export supports CSV, JSON, Parquet, IPC, NDJSON, and XLSX. Lazy sinks
are used where available; JSON and Excel collect at the serialization boundary.
