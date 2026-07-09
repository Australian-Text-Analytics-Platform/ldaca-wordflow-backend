"""
Comprehensive Authentication Tests
==================================

This file consolidates all authentication testing into a single, non-redundant test suite.
Tests cover both single-user and multi-user modes, API endpoints, and integration scenarios.
"""

from unittest.mock import patch


class TestAuthenticationAPI:
    """Test authentication API endpoints with current configuration"""

    async def test_auth_info_endpoint(self, test_client):
        """Test the main auth info endpoint"""
        response = await test_client.get("/api/auth/")

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        required_fields = [
            "authenticated",
            "requires_authentication",
            "available_auth_methods",
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    async def test_me_endpoint_single_user_mode(self, test_client):
        """Test /me endpoint returns test user in single-user mode"""
        response = await test_client.get("/api/auth/me")

        assert response.status_code == 200
        data = response.json()

        # Should return the configured test user
        assert data["id"] == "test"
        assert data["email"] == "test@localhost"
        assert data["name"] == "Test User"

    async def test_auth_status_endpoint(self, test_client):
        """Test authentication status endpoint"""
        response = await test_client.get("/api/auth/status")

        assert response.status_code == 200
        data = response.json()

        assert data["authenticated"] is True
        assert "user" in data
        assert data["user"]["id"] == "test"

    async def test_logout_endpoint(self, test_client):
        """Test logout endpoint"""
        response = await test_client.post("/api/auth/logout")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    async def test_google_oauth_disabled_single_user(self, test_client):
        """Test Google OAuth is disabled in single-user mode"""
        response = await test_client.post(
            "/api/auth/google", json={"id_token": "test-token"}
        )

        # Should be disabled in single-user mode
        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "invalid_input"
        assert "single-user mode" in data["message"].lower()


class TestAuthenticatedEndpoints:
    """Test endpoints that require authentication"""

    async def test_me_with_authenticated_client(self, authenticated_client):
        """Test /me endpoint with authenticated client fixture"""
        response = await authenticated_client.get("/api/auth/me")

        assert response.status_code == 200
        data = response.json()

        # Should return the test user from fixture
        assert data["id"] == "test"
        assert data["email"] == "test@example.com"
        assert data["name"] == "Test User"

    async def test_workspace_operations_with_auth(self, authenticated_client):
        """Test that workspace operations work with authentication"""
        response = await authenticated_client.get("/api/workspaces/")
        assert response.status_code == 200

        # Should return workspace list
        data = response.json()
        assert isinstance(data, list)


class TestAuthenticationModes:
    """Test authentication business logic for different modes"""

    async def test_get_current_user_single_user_mode(self, test_client):
        """Test get_current_user function in single-user mode"""
        from unittest.mock import MagicMock, patch

        from ldaca_wordflow.core.auth import get_current_user

        # Create a mock settings object with test values
        mock_settings = MagicMock()
        mock_settings.multi_user = False
        mock_settings.single_user_id = "test"
        mock_settings.single_user_name = "Test User"
        mock_settings.single_user_email = "test@localhost"

        # Patch settings in the core.auth module
        with patch("ldaca_wordflow.core.auth.settings", mock_settings):
            # In single-user mode, all calls should return the same user
            result1 = await get_current_user("Bearer test-token")
            result2 = await get_current_user("different-token")
            result3 = await get_current_user(None)

            # All should return test user
            for result in [result1, result2, result3]:
                assert result["id"] == "test"
                assert result["email"] == "test@localhost"
                assert result["name"] == "Test User"

            # Results should be identical
            assert result1 == result2 == result3

    async def test_auth_methods_single_user_mode(self, test_client):
        """Test available auth methods in single-user mode"""
        from ldaca_wordflow.core.auth import get_available_auth_methods

        methods = get_available_auth_methods()

        # Single-user mode should have no auth methods
        assert methods == []


class TestAuthenticationIntegration:
    """Test authentication integration with the rest of the system"""

    async def test_auth_system_consistency(self, test_client):
        """Auth info and /me should agree on authenticated state."""
        auth_info_response = await test_client.get("/api/auth/")
        auth_info = auth_info_response.json()

        me_response = await test_client.get("/api/auth/me")

        if auth_info["authenticated"]:
            assert me_response.status_code == 200
            me_data = me_response.json()
            assert "id" in me_data
            assert "email" in me_data
        else:
            assert me_response.status_code == 401
