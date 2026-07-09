# Workspace And Files

## Workspace Manager

`core/workspace.py` contains `WorkspaceManager`. It keeps one active in-memory
`docworkspace.Workspace` per user while allowing many saved workspace folders
on disk.

The manager keeps the user-facing selected workspace id separate from the
resident workspace object. `GET/PUT /api/users/me/current-workspace` owns that
selection preference. Explicit workspace-scoped routes load the requested
`workspace_id` as the resident object when needed, but they must not rewrite the
selection preference as a side effect.

The manager is responsible for:

- resolving each user's workspace root,
- listing persisted workspace summaries,
- allocating display-name-safe workspace folders,
- loading a requested workspace into memory,
- rebasing serialized LazyFrame scan paths before load,
- saving and unloading the resident workspace,
- deleting workspace folders,
- clearing workspace-specific analysis caches and task state.

The key invariant is that API code should not create its own global workspace
state. It should resolve the user's workspace through `workspace_manager`.

## File APIs

`api/files.py` exposes the user's data folder:

- tree listing,
- folder creation,
- upload, move, delete, raw text reads, and binary download,
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

Single-path file operations use a query parameter instead of catch-all path
routes: `GET /api/files/raw?path=...`, `GET /api/files/content?path=...`,
`GET /api/files/info?path=...`, and `DELETE /api/files/?path=...`. Keep new
static file endpoints out of catch-all path routing; use request bodies only
when an operation naturally has multiple path fields, such as move.

## Workspace Lifecycle Routes

`api/workspaces/lifecycle.py` handles workspace CRUD and workspace selection:

- list, create, delete, rename, unload, and set current workspace,
- lightweight workspace graph summaries,
- workspace save/download as a zip,
- workspace zip upload/import,
- workspace description and metadata.

Workspace-scoped lifecycle actions name their target in the URL. Save, download,
download-artifact retrieval, and unload use
`/api/workspaces/{workspace_id}/...`; only collection operations such as list,
create, and ZIP import stay directly under `/api/workspaces/`.
Unloading an existing workspace is idempotent when it is already not resident;
if the unloaded id is the user's selected workspace, the selection is cleared.

`GET /api/workspaces/{workspace_id}/graph` intentionally returns only graph/topology and
display/action state: node ids, names, parent/child ids, document column,
colour, and undo/redo flags. It must not collect or duplicate schema, columns,
shape, tokenizer models, or dtype-normalization metadata. Consumers that need
full node metadata use `POST /api/workspaces/{workspace_id}/nodes:batchGet` with body
`{"nodes": ["node-id"]}` or multiple ids as the source of truth.

ZIP import and export treat paths carefully: entries are validated with
`PurePosixPath`, and workspace source paths are rebased after the final folder
location is known.

## Node Operations

The workspace node routers own most row/column transformations:

- node info, node data paging, shape, unique values, describe, query plan,
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

Column casting is owned by `core/node_casting.py`. The cast route resolves the
workspace/node and persists the resulting `LazyFrame`; the service builds and
sample-validates the Polars cast expression so dtype conversion logic stays
testable outside the HTTP router.

The Polars expression endpoint validates code with
`core/polars_expr_validator.py`, executes in a restricted environment, and
allows only the supported transformation contexts.

## Schema Filtering

`api/workspaces/schema_filter.py` is the frontend-facing schema filter.
Tokenisation specs are tracked in `Node.tokenization` keyed by source column,
with hydrated column names such as `tokenization.text.lindera:jieba`. The node stores
the source column, selected model, language, and cache metadata for cached
tokens. Frontend node selectors update `Node.document` via
`PUT /api/workspaces/{workspace_id}/nodes/{node_id}/document-column`; tokenizer preferences are
registered via `PUT /api/workspaces/{workspace_id}/nodes/{node_id}/tokenization-preference`.
Source-node visualization colours are durable `Node.color` metadata updated via
`POST /api/workspaces/{workspace_id}/nodes/{node_id}/color`, separate from analysis request
payloads.
Analysis submit paths should read these fields, not mutate them. Analysis paths
resolve the per-user DuckDB cache path and call
`pl.col(...).text.tokenize(..., cache=path)` to hydrate temporary token
structs.
Node-info/table/schema responses preserve the physical node schema and expose
structured `tokenization` metadata where the UI needs it.

Use this shared projection for frontend node metadata instead of filtering or
fabricating token columns in individual routers.

## Export

`api/workspaces/base.py` still owns export endpoints and column-mutation
routes. Export supports CSV, JSON, Parquet, IPC, NDJSON, and XLSX. Lazy sinks
are used where available; JSON and Excel collect at the serialization boundary.
