# Analysis And Workers

## Two Task Layers

The backend has two related but separate task concepts.

`analysis/manager.py` stores user-visible analysis tasks. These records track
which feature ran, what request produced the result, the terminal result or
error, and parent/child relationships for follow-up tasks such as
materialization.

`core/worker_task_manager.py` manages process-pool futures. It starts workers,
captures progress, tracks worker pids, cancels running work, emits SSE events,
and applies side effects when a worker returns.

This split lets the UI show one coherent task center while the backend keeps
worker execution details out of analysis result models.

Task lifecycle ownership is backend-authoritative. Clearing a parent analysis
task must recursively clear registered child analysis tasks and worker tasks,
remove any related analysis caches, and emit `task_removed` for every removed
record. Frontend task lists should reconcile from `/api/tasks/clear`,
`tasks_snapshot`, and stream events rather than guessing related task ids.

Tabbed analysis tasks are equal, independent records. The backend must not treat
one task as the current task for an analysis type, and submitting a task in one
tab must not delete or replace another tab's task. The frontend persists the
tab-to-task relationship in `tabs.json` (`tab_id -> task_id`) and should fetch
request/result payloads by explicit `task_id`. Analysis request/result APIs do
not accept frontend `tab_id`; tab identity stays in the frontend tab sidecar.
The same sidecar also stores tab-owned node selectors: legacy `inputs` for the
default source selector and `input_sets` for additional named selectors.

## Worker Registry

`core/worker.py` defines the worker registry. Every process worker goes through
`_configure_worker_environment()` before importing heavy dependencies. That
setup disables tokenizer parallelism and configures Numba threading so worker
processes do not oversubscribe CPU cores.

Registered worker tasks include:

- LDaCA import,
- workspace ZIP download,
- token frequencies,
- concordance detach, dispersion detach, and materialization,
- quotation detach and materialization,
- topic modeling.

The LDaCA import worker uses `core/oni_client.py` to retrieve RO-Crate metadata
from the LDaCA Oni API, then invokes the vendored `rocrate-tabular` converter.
Keep new LDaCA import work in that Oni plus `rocrate-tabular` path.

Worker functions should be picklable, import heavy modules inside the worker
body, report progress through the provided queue, and write large outputs to
artifacts instead of returning huge payloads.

## Worker Completion Side Effects

`WorkerTaskManager` monitors each future. On completion it may:

- store an analysis result in the analysis task manager,
- add a materialized node to the active workspace,
- save workspace metadata,
- emit `task_changed`, `workspace_updated`, or `analysis_materialized`,
- record failed or cancelled state.

Detach tasks generally create new workspace nodes from worker-produced
artifacts. Materialize tasks update an existing analysis task request/result so
future paging can read from a persisted artifact.

When a worker is a child of an existing analysis task, pass or register the
parent task id and call the analysis task manager's child-link helper after
submission. This keeps later clear operations recursive even when the child is a
worker-only task.

## SSE Task Stream

`api/tasks.py` exposes `/api/tasks/stream`. The stream sends:

- an initial `tasks_snapshot`,
- `task_changed` events for progress and terminal states,
- `workspace_updated` when worker side effects change the graph,
- `analysis_materialized` when a paged analysis result has been persisted,
- heartbeat events to keep the connection alive.

Native `EventSource` cannot send an `Authorization` header, so the endpoint
also accepts `?token=...` and adapts it to the normal auth dependency.

## Analysis Modules

The analysis routes live under `api/workspaces/analyses/`.

- Token frequencies submit independent worker jobs, store result artifacts,
  support explicit task request/result endpoints, and expose update/clear flows.
- Concordance supports regex and token modes, result paging, dispersion bins,
  detach, dispersion detach, and materialization.
- Quotation can use a local extractor or remote quotation service, then pages,
  detaches, or materializes quote results.
- Sequential analysis runs synchronously over lazy Polars expressions for time
  and group buckets, with selected-period detach.
- Topic modeling runs entirely through the Rust `polars-text` pipeline. The
  worker builds a single-column frame of sampled documents and calls
  `pl.col(...).text.topic_modeling(...)`, which chunks text by paragraph,
  sentence, and token length; embeds chunks with an in-process ONNX Runtime
  model; reduces with PaCMAP; clusters with HDBSCAN; and labels topics with
  c-TF-IDF. The expression returns one struct per document (dominant topic,
  per-document topic distribution, replicated topic keywords/coordinates, and
  global counts); the worker rolls these up into the topic/document payload.
  Embedding runs in-process in Rust with a separate `embeddings.duckdb`
  content-hash cache, so there is no Python-side embedding cache for topic
  modeling.
- Annotation currently exposes workspace helper routes for setup/editing data:
  `POST /api/workspaces/annotation/class-descriptions` creates an empty
  `class`/`description` data block, while
  `GET /api/workspaces/annotation/class-descriptions/{node_id}` and
  `PUT /api/workspaces/annotation/class-descriptions/{node_id}` round-trip
  the selected class/description columns for the frontend editor.
  `POST /api/workspaces/annotation/source/{node_id}/annotation-column` adds an
  empty string annotation column to a source node, and
  `PUT /api/workspaces/annotation/source/{node_id}/annotation-cell` writes one
  cell of that column (`column_name`, absolute `row_index`, nullable `value`)
  using a Polars `pl.when(int_range == row_index)` rewrite restaged via
  `stage_dataframe_as_lazy`; blank/whitespace values are stored as null.
  AI-assisted annotation runs entirely server-side under
  `POST /api/workspaces/annotation/ai/{models,preview,preview/state,preview/clear,annotate-all,detach-previewed}`
  plus `PUT .../annotation/ai/preview/override`,
  backed by the provider-dispatch engine in `core/annotation_ai.py`. The browser
  never calls a model provider: it posts the provider id, optional custom base URL,
  API key, model, and instruction to these routes, and the engine dispatches the
  provider's native async SDK (`AsyncOpenAI` for openai-style + custom endpoints,
  `AsyncAnthropic`, and the `google-genai` async client). The OpenAI-style chat
  call always passes `stream=False` explicitly: some OpenAI-compatible servers —
  notably Apple's on-device `fm serve` — stream a Server-Sent-Events body whenever
  the request omits `stream`, which the SDK's non-streaming path then hands back as
  a bare `str` (`'str' object has no attribute 'choices'`); pinning the flag forces
  one JSON completion everywhere and is a no-op for the hosted providers. Each
  request carries an
  `InferenceConfig` (from `temperature`/`reasoning_enabled`/`reasoning_effort`
  fields on the preview and annotate-all bodies; `InferenceConfig.from_request`
  clamps temperature to `[0, 2]` and normalizes the effort to `low`/`medium`/`high`)
  that the engine maps onto each provider natively: OpenAI gets a `reasoning_effort`
  string (and temperature is omitted when reasoning is on, since reasoning models
  reject non-default temperatures); Anthropic gets a `thinking` budget with
  `max_tokens` raised by `ANSWER_TOKEN_HEADROOM` (temperature omitted); Google gets
  a `ThinkingConfig(thinking_budget=…)` while temperature is always sent. With
  reasoning off the engine simply forwards the temperature and no thinking config.
  Preview labels are **cached and persisted** in an in-memory preview store
  (`core/annotation_preview_store.py`, a module-level singleton keyed by
  user + workspace + node, guarded by a lock). Each node's session records its
  previewed rows under a *signature* — a stable hash of the prediction-affecting
  config only (text column, class node/columns, provider, base URL, model,
  instruction, temperature, and the reasoning knobs; **not** the annotation column
  or page) — so a signature change resets that node's rows. The store is
  process-lifetime: it survives across requests and tab switches but is cleared on
  backend restart. `/models` enumerates a
  provider's model ids (sorted, de-duplicated). Custom endpoints are listed the
  same way — through the OpenAI SDK's `/models` route against their base URL, since
  many local servers (Apple `fm serve`, Ollama, LM Studio, vLLM) expose it — except
  a custom provider with no base URL returns `[]` so the SDK never falls back to
  api.openai.com; an endpoint that simply lacks `/models` surfaces its SDK error as
  a 502 while the picker's free-text entry still works. `/preview`
  classifies one page of the source node but first serves any rows already stored
  under the current signature and calls the provider (`annotate_batch`) only for
  the uncached rows, then persists the new labels and returns `{"labels":[...]}`
  for the page. `/preview/state` returns the whole stored session for that
  signature (`{ai, override, effective, has_override}` per row) so the frontend can
  rehydrate the preview panel on remount, and `/preview/override` persists a single
  cell edit (`row_index`, nullable `label`) so manual overrides survive a tab
  switch. `/preview/clear` drops a node's session outright (`{node_id}` →
  `preview_store.clear`, idempotent — a missing session is a successful no-op, no
  404). This is the key asymmetry the frontend relies on: a tab switch only unmounts
  the panel and **keeps** the cache so it can rehydrate, whereas clicking "Close
  preview" is an explicit "done previewing" that calls `/preview/clear`, so the next
  Preview re-classifies from scratch and no stale detach/annotate-all count lingers.
  `/annotate-all` reuses the store's cached labels for already-previewed
  rows, fans only the remainder out over concurrent batches (`asyncio.gather` under
  a semaphore, order preserved), overwrites the whole String annotation column,
  restages via `update_workspace`, and clears the node's preview session.
  `/detach-previewed` takes no LLM path at all: it reads the node's **entire**
  preview session from the store (effective label = override ?? AI, across every
  previewed page — the client no longer sends the rows, which fixes the earlier
  "detach only grabs the current page" bug), copies just those rows into a new
  **child node** (via `_create_and_persist_child_node`) with the labels written
  into the annotation column (blanks → null), and leaves the source untouched.
  The same route also accepts `dry_run: true`, in which case it returns just
  `{node: null, detached_rows: N}` — the session's row count — without touching the
  workspace or 404-ing on an empty session (it returns `0`). The frontend uses this
  probe both to enable the Detach button and to fill the confirmation dialog's row
  count after a tab switch, since the browser has no local copy of the previewed
  set once the panel remounts.
  Provider
  failures surface as `BadGatewayError` (`502`). Runnable manual annotation
  processing is still being redesigned separately from the removed
  implementation.
- Standalone tokenization routes are not exposed. Tokenizer model inventory is
  available from `GET /api/workspaces/tokenizer-models`; it is sourced from
  `polars-text` and carries model IDs, display labels, and supported ISO 639-1
  language codes without backend recommendation policy. Node selectors persist
  document columns through `PUT /api/workspaces/nodes/{node_id}/document-column`.
  Token frequency and concordance store per-column preferences through
  `PUT /api/workspaces/nodes/{node_id}/tokenization-preference`, then derive the
  model from `Node.tokenization` when submitting worker jobs. Token-frequency
  requests still include `node_tokenizer_models` as task/snapshot settings and
  as a short-lived fallback while a preference write is in flight.
  `Node.tokenization` metadata is hydrated from the per-user DuckDB token cache
  when an existing workspace node carries cached token specs.

Shared helpers in `cleanup.py`, `current_tasks.py`, `generated_columns.py`,
`page_size_estimation.py`, and core cache modules keep route code smaller.

## Cache Ownership

Analysis artifacts are user, workspace, and task scoped. Transient files belong
under `<workspace>/data/artifacts`, and the task that produces an artifact owns
its cleanup. Materialized cache filenames include the producing task id, and
task request/result payloads should record any additional artifact paths under
path-like keys such as `*_path`, `*_parquet_path`, or `*_paths`.

Task cleanup is centralized in `core/task_artifacts.py` and is invoked by the
analysis and worker task managers. Feature routes should clear task records,
not manually unlink artifact files. Cleanup only deletes files inside
`data/artifacts`; Add to Workspace copies or writes durable node data into the
top-level `data` directory, where ownership transfers from the task to the
workspace node.

Workspace unload and workspace switching must snapshot reloadable analysis task
records, then evict the in-memory analysis and worker task records for the
inactive workspace. The unload lifecycle preserves `data/artifacts` so persisted
tabs can rehydrate task results after the workspace is loaded again. Artifact
files are reclaimed by explicit task cleanup or workspace deletion; do not wire
workspace unload to whole-artifact-directory cleanup.

Tokenisation and embeddings are performance caches rather than workspace
artifacts. Token specs live in `Node.tokenization`; Wordflow resolves a per-user
`tokens.duckdb` path, then delegates cache population and reuse to
`pl.col(...).text.tokenize(..., cache=path)`. The resulting token structs are
attached to temporary LazyFrames by token-mode concordance and token
frequencies. Topic modeling resolves a per-user `embeddings.duckdb` path and
passes it into the Rust `polars-text` embedding stage. The token and embedding
caches are intentionally separate files. Missing cache files are recreated from
schema on first use.
