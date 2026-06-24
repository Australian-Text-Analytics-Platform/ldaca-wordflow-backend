"""Data structures for auth persistence rows.

Used by:
- backend migration paths while the project transitions from SQLAlchemy to
  direct SQLite access, and by tests that import row-shaped objects for
  lightweight typing.

Flow:
- Keep a stable `User`/`UserSession` shape so call sites that previously
  expected ORM-style attributes can keep the same semantic contract.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class User:
    """User record shape persisted in `users`.

    Attributes mirror the fields returned by `auth_service` dictionaries.
    """

    id: str
    email: str
    name: str
    picture: str | None = None
    google_id: str | None = None
    user_folder_path: str | None = None
    created_at: datetime | None = None
    last_login: datetime | None = None
    is_active: bool = True
    is_superuser: bool = False
    is_verified: bool = False


@dataclass(slots=True)
class UserSession:
    """User session record shape persisted in `user_sessions`."""

    user_id: str
    access_token: str
    refresh_token: str | None = None
    expires_at: datetime | None = None
    created_at: datetime | None = None
    id: int | None = None
