"""
Configuration for pytest tests
Provides shared fixtures and setup for all tests
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ldaca_wordflow import db


@pytest.fixture(autouse=True)
def _reset_sniffio_async_context():
    """Stop a leaked async-library marker from breaking sync ``TestClient`` tests.

    Why this exists (free-threaded Python 3.14t only): on the free-threaded
    build a newly started ``threading.Thread`` *inherits* the parent thread's
    :mod:`contextvars` context, whereas the GIL build gives each thread a fresh,
    empty context. anyio's asyncio backend records the running library in
    sniffio's ``current_async_library_cvar`` ContextVar; after an async test
    (e.g. one using the Starlette ``TestClient``/httpx ASGI transport) that
    marker can linger in the main thread's context. A later *synchronous*
    ``TestClient`` request calls ``anyio.start_blocking_portal``, which spawns a
    worker thread to run a fresh event loop -- but on 3.14t that thread inherits
    the lingering ``"asyncio"`` marker, so anyio raises
    ``RuntimeError: Already running asyncio in this thread``. Manifested as ~19
    failures in ``test_files_preview.py`` once the suite runs after the async
    integration tests.

    Flow: before each test, clear sniffio's ContextVar + thread-local marker in
    the main thread so any portal thread spawned during the test inherits a
    clean context. Harmless on the GIL build (the marker is already clean there)
    and on async tests (anyio re-sets the marker inside its own loop context).

    Used by: every test (autouse), guarding the synchronous ``TestClient``
    fixtures (``files_test_client`` and friends) against cross-test async-context
    leakage on free-threaded interpreters.
    """
    try:
        import sniffio._impl as _sniffio
    except Exception:  # pragma: no cover - sniffio always present via anyio
        yield
        return

    _sniffio.thread_local.name = None
    token = _sniffio.current_async_library_cvar.set(None)
    try:
        yield
    finally:
        _sniffio.current_async_library_cvar.reset(token)
        _sniffio.thread_local.name = None


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session", autouse=True)
def _tokens_cache_in_tmpdir(tmp_path_factory):
    """Redirect the per-user tokens cache DB into a tmpdir for the test session.

    Without this fixture, analyses that hydrate tokens would write DuckDB files
    into the developer's real ``~/.../user_cache/tokens.duckdb``.
    """
    from ldaca_wordflow.core import tokens_cache as _tc

    tmp_root = tmp_path_factory.mktemp("tokens-cache")
    original = _tc.tokens_cache_path

    def _redirect(user_id: str) -> Path:
        path = tmp_root / user_id / _tc.TOKENS_CACHE_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    setattr(_tc, "tokens_cache_path", _redirect)
    try:
        yield tmp_root
    finally:
        setattr(_tc, "tokens_cache_path", original)


@pytest.fixture(scope="session", autouse=True)
async def init_test_db():
    """Initialize test database with tables for all tests"""
    # Import after setting up the path

    # Use in-memory database for tests without modifying the global config
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    # Create a test-specific database engine
    test_db_url = "sqlite+aiosqlite:///:memory:"
    test_engine = create_async_engine(test_db_url)
    test_session_maker = async_sessionmaker(test_engine, expire_on_commit=False)

    # Store the original for restoration
    original_engine = getattr(db, "engine", None)
    original_session_maker = getattr(db, "async_session_maker", None)

    # Replace with test versions
    db.engine = test_engine
    db.async_session_maker = test_session_maker

    # Create tables in the test database
    await db.create_db_and_tables()

    yield

    # Cleanup
    await test_engine.dispose()

    # Restore original if they existed
    if original_engine:
        db.engine = original_engine
    if original_session_maker:
        db.async_session_maker = original_session_maker


_SETTINGS_PATCH_TARGETS: list[str] = [
    "ldaca_wordflow.settings.settings",
    "ldaca_wordflow.main.settings",
    "ldaca_wordflow.api.auth.settings",
    "ldaca_wordflow.core.auth.settings",
    "ldaca_wordflow.core.utils.settings",
    "ldaca_wordflow.core.user_folders.settings",
    "ldaca_wordflow.core.sample_data.settings",
]


def _settings_patches(override):
    """Return a list of started patch contexts for all settings import sites.

    Centralises the multi-patch boilerplate so conftest fixtures don't
    duplicate the target list.
    """
    return [patch(target, override) for target in _SETTINGS_PATCH_TARGETS]


@pytest.fixture
def settings_override(tmp_path: Path):
    """Provide MagicMock settings pointing to a temporary data root.

    This isolates tests from the repository filesystem and avoids cleanup needs.
    """
    mock_settings = MagicMock()
    # Path helpers
    mock_settings.get_data_root.return_value = tmp_path
    mock_settings.get_user_data_folder.return_value = tmp_path / "users"
    mock_settings.get_sample_data_folder.return_value = tmp_path / "sample_data"
    mock_settings.get_database_backup_folder.return_value = tmp_path / "backups"
    # Back-compat attributes some code might read
    mock_settings.data_folder = tmp_path
    mock_settings.user_data_folder = "users"
    mock_settings.sample_data_folder = "sample_data"
    mock_settings.allowed_origins = ["http://localhost:3000"]
    # Core config
    mock_settings.cors_allowed_origins = ["http://localhost:3000"]
    mock_settings.cors_allow_credentials = True
    mock_settings.multi_user = False
    mock_settings.single_user_id = "test"
    mock_settings.single_user_name = "Test User"
    mock_settings.single_user_email = "test@localhost"
    mock_settings.google_client_id = ""
    mock_settings.database_url = "sqlite+aiosqlite:///:memory:"
    mock_settings.server_host = "127.0.0.1"
    mock_settings.server_port = 8001
    mock_settings.debug = True
    return mock_settings


@pytest.fixture
async def test_db_session():
    """Provide a test database session"""
    from ldaca_wordflow import db

    async with db.async_session_maker() as session:
        yield session


@pytest.fixture
def temp_data_root(test_user):
    """Ensure a temporary user data root exists for tests that write files.

    This creates the user's data directory used by get_user_data_folder so tests
    that write files without explicitly creating directories will work reliably.
    """
    from ldaca_wordflow.core.utils import get_user_data_folder

    user_data_dir = get_user_data_folder(test_user["id"])
    user_data_dir.mkdir(parents=True, exist_ok=True)
    return user_data_dir.parent


@pytest.fixture
async def authenticated_client(settings_override):
    """Async test client with mocked authentication and isolated temp data root."""
    from datetime import datetime

    import httpx
    from ldaca_wordflow.core.auth import get_current_user
    from ldaca_wordflow.main import app

    mock_user = {
        "id": "test",
        "email": "test@example.com",
        "name": "Test User",
        "picture": "https://example.com/avatar.jpg",
        "created_at": datetime(2024, 1, 1, 0, 0, 0),
        "last_login": datetime(2024, 1, 1, 12, 0, 0),
        "is_active": True,
        "is_verified": True,
    }

    def mock_get_current_user():
        return mock_user

    patches = [
        *_settings_patches(settings_override),
        patch("ldaca_wordflow.db.init_db"),
        patch("ldaca_wordflow.core.auth_service.cleanup_expired_sessions"),
    ]

    for p in patches:
        p.start()

    app.dependency_overrides[get_current_user] = mock_get_current_user

    try:
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        for p in patches:
            p.stop()


@pytest.fixture
async def test_client(settings_override):
    """Provide an async test client without authentication (single-user mode)."""
    import httpx
    from ldaca_wordflow.main import app

    patches = [
        *_settings_patches(settings_override),
        patch("ldaca_wordflow.db.init_db"),
        patch("ldaca_wordflow.core.auth_service.cleanup_expired_sessions"),
    ]

    for p in patches:
        p.start()

    try:
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            yield client
    finally:
        for p in patches:
            p.stop()


@pytest.fixture
def files_test_client(tmp_path: Path):
    """Sync TestClient for files-root routes with isolated settings and auth."""
    from fastapi.testclient import TestClient

    with (
        patch("ldaca_wordflow.main.settings") as mock_settings,
        patch("ldaca_wordflow.main.init_db"),
        patch("ldaca_wordflow.main.cleanup_expired_sessions"),
        patch("ldaca_wordflow.core.utils.settings") as mock_utils_settings,
        patch("ldaca_wordflow.core.user_folders.settings") as mock_user_folders_settings,
        patch("ldaca_wordflow.core.sample_data.settings") as mock_sample_data_settings,
    ):
        mock_settings.debug = False
        mock_settings.cors_allow_origin_regex = r"http://localhost(:\d+)?"
        mock_settings.cors_allow_credentials = True
        mock_settings.multi_user = True
        mock_settings.get_data_root.return_value = tmp_path
        mock_settings.get_user_data_folder.return_value = tmp_path / "users"
        mock_settings.get_sample_data_folder.return_value = tmp_path / "sample_data"
        mock_settings.get_database_backup_folder.return_value = tmp_path / "backups"
        mock_settings.user_data_folder = "users"

        mock_utils_settings.get_data_root.return_value = tmp_path
        mock_utils_settings.user_data_folder = "users"
        mock_utils_settings.multi_user = True

        mock_user_folders_settings.get_data_root.return_value = tmp_path
        mock_user_folders_settings.user_data_folder = "users"
        mock_user_folders_settings.multi_user = True

        mock_sample_data_settings.get_data_root.return_value = tmp_path
        mock_sample_data_settings.user_data_folder = "users"
        mock_sample_data_settings.multi_user = True
        mock_sample_data_settings.get_sample_data_folder.return_value = tmp_path / "sample_data"

        (tmp_path / "users").mkdir(parents=True, exist_ok=True)
        (tmp_path / "sample_data").mkdir(parents=True, exist_ok=True)
        (tmp_path / "backups").mkdir(parents=True, exist_ok=True)
        (tmp_path / "users" / "user_test_user" / "user_data").mkdir(
            parents=True, exist_ok=True
        )

        app = __import__("ldaca_wordflow.main", fromlist=["app"]).app

        def fake_user():
            return {"id": "test_user"}

        from ldaca_wordflow.api import files as files_api

        app.dependency_overrides[files_api.get_current_user] = fake_user

        try:
            yield TestClient(app)
        finally:
            app.dependency_overrides.clear()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_settings():
    """Mock the config module with test configuration"""
    with patch("ldaca_wordflow.settings.settings") as mock_config:
        # Core settings
        mock_config.database_url = "sqlite+aiosqlite:///:memory:"
        mock_config.user_data_folder = "./test_data"
        mock_config.sample_data_folder = "./test_data/sample_data"
        mock_config.server_host = "127.0.0.1"
        mock_config.server_port = 8000
        mock_config.debug = True
        mock_config.cors_allowed_origins = ["http://localhost:3000"]
        mock_config.cors_allow_credentials = True
        mock_config.multi_user = False
        mock_config.single_user_id = "test"
        mock_config.single_user_name = "Test User"
        mock_config.single_user_email = "test@localhost"
        mock_config.google_client_id = ""  # Empty for single-user mode
        mock_config.token_expire_hours = 1
        mock_config.secret_key = "test-secret-key"
        mock_config.log_level = "DEBUG"
        mock_config.log_file = "./test_logs/test.log"

        # Backward compatibility properties
        mock_config.data_folder = Path("./test_data")
        mock_config.allowed_origins = ["http://localhost:3000"]

        # Path methods
        mock_config.get_user_data_folder.return_value = Path("./test_data")
        mock_config.get_sample_data_folder.return_value = Path(
            "./test_data/sample_data"
        )
        mock_config.get_database_backup_folder.return_value = Path(
            "./test_data/backups"
        )

        yield mock_config


@pytest.fixture
def sample_csv_file(temp_dir):
    """Create a sample CSV file for testing"""
    csv_content = """name,age,city
Alice,25,New York
Bob,30,London
Charlie,35,Tokyo"""

    csv_file = temp_dir / "sample.csv"
    csv_file.write_text(csv_content)
    return csv_file


@pytest.fixture
def sample_json_file(temp_dir):
    """Create a sample JSON file for testing"""
    json_content = """[
    {"name": "Alice", "age": 25, "city": "New York"},
    {"name": "Bob", "age": 30, "city": "London"},
    {"name": "Charlie", "age": 35, "city": "Tokyo"}
]"""

    json_file = temp_dir / "sample.json"
    json_file.write_text(json_content)
    return json_file


@pytest.fixture
def test_user():
    """Provide consistent test user data for analysis tests."""
    return {
        "id": "test",
        "email": "test@example.com",
        "name": "Test User",
        "picture": "https://example.com/avatar.jpg",
        "is_active": True,
        "is_verified": True,
    }


@pytest.fixture
async def workspace_id(authenticated_client):
    """Create a test workspace and ensure it's deleted after the test."""
    response = await authenticated_client.post(
        "/api/workspaces/",
        json={"name": "test_workspace", "description": "Test workspace for analysis"},
    )
    assert response.status_code == 200
    workspace_id = response.json()["id"]

    try:
        yield workspace_id
    finally:
        cleanup_response = await authenticated_client.delete(
            f"/api/workspaces/{workspace_id}"
        )
        if cleanup_response.status_code not in (200, 404):
            raise AssertionError(
                f"Failed to delete test workspace {workspace_id}: "
                f"status={cleanup_response.status_code} body={cleanup_response.text}"
            )


@pytest.fixture
def tiny_text_file(test_user):
    """Create a tiny CSV file for testing."""
    import csv

    from ldaca_wordflow.core.utils import get_user_data_folder

    user_data_dir = get_user_data_folder(test_user["id"])
    user_data_dir.mkdir(parents=True, exist_ok=True)

    # Create a tiny CSV file
    tiny_file = user_data_dir / "tiny.csv"
    with open(tiny_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["document"])
        writer.writerow(["Hello world."])
        writer.writerow(["Another sentence."])

    return tiny_file


@pytest.fixture
def sample_text_file(test_user):
    """Create a sample CSV file for testing."""
    import csv

    from ldaca_wordflow.core.utils import get_user_data_folder

    user_data_dir = get_user_data_folder(test_user["id"])
    user_data_dir.mkdir(parents=True, exist_ok=True)

    # Create a sample CSV file
    sample_file = user_data_dir / "sample.csv"
    with open(sample_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["document"])
        writer.writerow(["This is a sample document."])
        writer.writerow(["Another sample text for analysis."])
        writer.writerow(["More text content for testing."])
        writer.writerow(["Final sample sentence."])

    return sample_file


@pytest.fixture
def timeline_csv_file(test_user):
    """Create a CSV file with timestamped records for frequency analysis tests."""
    import csv

    from ldaca_wordflow.core.utils import get_user_data_folder

    user_data_dir = get_user_data_folder(test_user["id"])
    user_data_dir.mkdir(parents=True, exist_ok=True)

    timeline_file = user_data_dir / "timeline.csv"
    with open(timeline_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["document", "published_at", "category"])
        writer.writerow(["Entry one", "2024-01-01T08:15:00Z", "alpha"])
        writer.writerow(["Entry two", "2024-01-02T09:00:00Z", "beta"])
        writer.writerow(["Entry three", "2024-01-02T11:30:00Z", "alpha"])
        writer.writerow(["Entry four", "2024-01-03T14:45:00Z", "beta"])
        writer.writerow(["Entry five", "2024-01-03T16:00:00Z", "gamma"])

    return timeline_file


@pytest.fixture
async def tiny_node_id(authenticated_client, workspace_id, tiny_text_file):
    """Add a tiny node to the workspace and return its ID."""
    response = await authenticated_client.post(
        "/api/workspaces/nodes",
        params={"filename": tiny_text_file.name},
    )
    assert response.status_code == 200
    result = response.json()
    # The API returns 'id', not 'node_id'
    return result["id"]


@pytest.fixture
async def sample_node_id(authenticated_client, workspace_id, sample_text_file):
    """Add a sample node to the workspace and return its ID."""
    response = await authenticated_client.post(
        "/api/workspaces/nodes",
        params={"filename": sample_text_file.name},
    )
    assert response.status_code == 200
    result = response.json()
    # The API returns 'id', not 'node_id'
    return result["id"]


@pytest.fixture
async def timeline_node_id(authenticated_client, workspace_id, timeline_csv_file):
    """Add a timeline-friendly node to the workspace and return its ID."""
    response = await authenticated_client.post(
        "/api/workspaces/nodes",
        params={"filename": timeline_csv_file.name},
    )
    assert response.status_code == 200
    result = response.json()
    return result["id"]
