# LDaCA Wordflow Backend (archived)

This standalone source repository is archived. The backend now lives in the
[`backend/` directory of the LDaCA Wordflow monorepo](https://github.com/Australian-Text-Analytics-Platform/ldaca-wordflow/tree/main/backend),
which is authoritative for source, CI, issues, and releases.

Existing commits, tags, and releases remain here as historical records. Open
new issues and pull requests in the
[Wordflow monorepo](https://github.com/Australian-Text-Analytics-Platform/ldaca-wordflow).

## Quick start

```bash
# Install and run (backend + frontend on one port)
uvx ldaca-wordflow

# Or run only the backend API
uvx ldaca-wordflow --backend

# Custom port
uvx ldaca-wordflow --port 9000
```

## Notebook and BinderHub launch

Use the asynchronous launcher when Python must retain control of the event loop,
such as a BinderHub notebook cell:

```python
from ldaca_wordflow import start_async_server

server = await start_async_server()
# The compiled web application is now ready through jupyter-server-proxy.

await server.close()
```

`start_async_server()` returns only after ASGI lifespan startup succeeds. Its
caller-owned handle supports waiting and bounded graceful shutdown. Under
JupyterHub, the launcher derives the proxy `root_path` from
`JUPYTERHUB_SERVICE_PREFIX`.

## Development

- Backend source: [`backend/`](https://github.com/Australian-Text-Analytics-Platform/ldaca-wordflow/tree/main/backend)
- Backend architecture: [`docs/architecture/backend/overview.md`](https://github.com/Australian-Text-Analytics-Platform/ldaca-wordflow/blob/main/docs/architecture/backend/overview.md)
- HTTP endpoint inventory: [`docs/reference/backend-api.md`](https://github.com/Australian-Text-Analytics-Platform/ldaca-wordflow/blob/main/docs/reference/backend-api.md)
- Settings reference: [`docs/reference/backend-settings.md`](https://github.com/Australian-Text-Analytics-Platform/ldaca-wordflow/blob/main/docs/reference/backend-settings.md)
- Lint: `uv run ruff check .`
- Type check: `uv run ty check`
- Tests: `uv run pytest -q`
- OpenAPI export: `uv run python scripts/export_openapi.py --output <path>`
