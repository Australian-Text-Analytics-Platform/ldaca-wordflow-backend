# LDaCA Wordflow Backend

FastAPI service for LDaCA Wordflow. The package can run the API by itself or
serve the bundled production frontend from the same process.

## Quick start

```bash
# Install and run (backend + frontend on one port)
uvx ldaca-wordflow

# Or run only the backend API
uvx ldaca-wordflow --backend

# Custom port
uvx ldaca-wordflow --port 9000
```

## Development

- Backend architecture: [`../docs/architecture/backend/overview.md`](../docs/architecture/backend/overview.md)
- HTTP endpoint inventory: [`../docs/reference/backend-api.md`](../docs/reference/backend-api.md)
- Settings reference: [`../docs/reference/backend-settings.md`](../docs/reference/backend-settings.md)
- Type check: `uvx ty check`
- Tests: `uv run pytest -q`
- OpenAPI export: `uv run python scripts/export_openapi.py --output <path>`
