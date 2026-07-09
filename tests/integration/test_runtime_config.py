"""Runtime configuration endpoint contract tests.

Used by:
- backend API contract tests because runtime bootstrap data is intentionally
  public while mutable process configuration must stay behind the admin
  boundary.

Flow:
- Assert the public read-only runtime endpoint exists and the legacy mutable
  config route is gone.
- Exercise admin config authorization in multi-user mode.
- Confirm the process-local data-root update returns the reloaded value.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import httpx


async def test_runtime_config_is_public_and_legacy_config_routes_are_removed(test_client):
    """Public bootstrap config is read-only; legacy `/api/config` is absent."""

    response = await test_client.get("/api/runtime-config")
    assert response.status_code == 200
    assert response.json() == {
        "multi_user_mode": False,
        "google_client_id": "",
    }

    assert (await test_client.get("/api/config/")).status_code == 404
    assert (
        await test_client.post("/api/config/", json={"data_root": "/tmp/legacy"})
    ).status_code == 404


async def test_openapi_exposes_runtime_and_admin_config_without_legacy_route(test_client):
    """Generated clients should see the clean config routes only."""

    from ldaca_wordflow.main import app

    app.openapi_schema = None
    response = await test_client.get("/api/openapi.json")
    assert response.status_code == 200

    paths = response.json()["paths"]
    assert "/api/runtime-config" in paths
    assert "/api/admin/config" in paths
    assert "/api/config/" not in paths


async def test_admin_config_requires_authentication_in_multi_user_mode(settings_override):
    """Unauthenticated multi-user callers cannot mutate runtime config."""

    from ldaca_wordflow.main import app

    settings_override.multi_user = True
    settings_override.get_admin_emails.return_value = {"admin@example.com"}

    with (
        patch("ldaca_wordflow.settings.settings", settings_override),
        patch("ldaca_wordflow.core.auth.settings", settings_override),
    ):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.patch(
                "/api/admin/config", json={"data_root": "/tmp/next"}
            )

    assert response.status_code == 401


async def test_admin_config_requires_admin_allowlist_in_multi_user_mode(settings_override):
    """Authenticated non-admin users are rejected by the admin config route."""

    from ldaca_wordflow.core.auth import get_current_user
    from ldaca_wordflow.main import app

    settings_override.multi_user = True
    settings_override.get_admin_emails.return_value = {"admin@example.com"}

    def fake_user():
        return {"id": "u-1", "email": "user@example.com", "name": "User"}

    app.dependency_overrides[get_current_user] = fake_user
    try:
        with patch("ldaca_wordflow.settings.settings", settings_override):
            transport = httpx.ASGITransport(app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.patch(
                    "/api/admin/config", json={"data_root": "/tmp/next"}
                )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 403
    payload = response.json()
    assert payload["error"] == "access_denied"
    assert payload["message"] == "Admin access required"


async def test_admin_config_updates_process_data_root(tmp_path):
    """Single-user admin config updates reload the process-local data root."""

    from ldaca_wordflow.main import app
    from ldaca_wordflow.settings import reload_settings

    original_data_root = os.environ.get("DATA_ROOT")
    original_multi_user = os.environ.get("MULTI_USER")
    try:
        os.environ["DATA_ROOT"] = str(tmp_path / "original")
        os.environ["MULTI_USER"] = "false"
        reload_settings()

        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.patch(
                "/api/admin/config", json={"data_root": str(tmp_path / "next")}
            )

        assert response.status_code == 200
        assert response.json()["data_root"] == str(tmp_path / "next")
    finally:
        if original_data_root is None:
            os.environ.pop("DATA_ROOT", None)
        else:
            os.environ["DATA_ROOT"] = original_data_root
        if original_multi_user is None:
            os.environ.pop("MULTI_USER", None)
        else:
            os.environ["MULTI_USER"] = original_multi_user
        reload_settings()
