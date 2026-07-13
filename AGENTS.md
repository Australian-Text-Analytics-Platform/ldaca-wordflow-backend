# Backend operating guide

These rules extend the root `AGENTS.md` for the `ldaca-wordflow` FastAPI
package. Read [the backend architecture index](../docs/architecture/backend/overview.md)
before changing an unfamiliar backend boundary.

## Structure and ownership

- `main.py` constructs the side-effect-free FastAPI application.
- `runtime.py` owns lifespan construction, startup, shutdown, and stateful
  services.
- `api/` declares routes, resolves dependencies, and shapes HTTP responses.
- `services/` owns application use cases and runtime coordination.
- `domain/` owns business state and invariants without FastAPI dependencies.
- `analysis/` owns framework-independent analysis behavior.
- `workers/` contains picklable executor entrypoints.
- `infrastructure/` implements storage, database, provider, and process edges.
- `shared/` contains dependency-light cross-layer types and errors.

Keep imports directed inward: domain, services, analysis, workers, and
infrastructure must not import `ldaca_wordflow.api`.

## FastAPI and runtime rules

- Keep routers thin. Validate transport shapes, then delegate behavior to the
  owning service or analysis helper.
- Resolve protected requests through `get_current_session` or
  `get_current_user`; do not recreate authentication in a route.
- Resolve runtime-owned services through dependencies. Do not add module-level
  mutable state or construct stateful resources during import or OpenAPI export.
- Stateful resources start and stop inside FastAPI lifespan. Preserve the
  startup and reverse-order shutdown contracts documented in
  [runtime architecture](../docs/architecture/backend/runtime.md).
- `WorkspaceService` is the only workspace mutation and persistence boundary.
  `TaskService` is the only public task state machine.
- Domain exceptions stay independent of FastAPI and are translated at the API
  edge.

## Data and workers

- Preserve lazy Polars plans. Avoid `collect()` except at I/O, artifact, or
  final response boundaries.
- Snapshot immutable task inputs under the workspace gate, then release it
  before process execution, provider calls, or streaming.
- Worker entrypoints call `configure_worker_environment()` before importing
  heavy dependencies. Large results belong in task-owned artifacts, not process
  return payloads.
- Completion handlers run on the application event loop and must be idempotent.

## Commands

Run from `backend/`:

```sh
uv sync
uv run pytest -q
uvx ty check
uv run python scripts/export_openapi.py --output <path>
```

After an API schema change, export OpenAPI through the established generation
workflow and regenerate the frontend client; never edit generated client files
by hand.

## Done

Run `uvx ty check` and `uv run pytest -q`. Update the applicable backend
architecture, domain, API reference, settings reference, runbook, or ADR when
the change makes it incomplete or inaccurate.
