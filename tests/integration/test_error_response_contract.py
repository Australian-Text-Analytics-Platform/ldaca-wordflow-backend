import pytest


@pytest.mark.anyio
async def test_app_errors_use_standard_error_response(authenticated_client):
    """AppError subclasses should serialize through one stable JSON envelope."""

    response = await authenticated_client.get(
        "/api/workspaces/00000000-0000-0000-0000-00000000dead"
    )

    assert response.status_code == 404
    assert response.json() == {
        "error": "workspace_not_found",
        "message": "Workspace not found",
        "details": None,
    }
