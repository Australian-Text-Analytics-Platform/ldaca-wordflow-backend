"""Authentication and user models.

Split from models/__init__.py.
"""

from __future__ import annotations

from pydantic import BaseModel

class User(BaseModel):
    """API schema used by routes and generated clients for user.

    Used by:
    - backend API routes, backend package imports, backend request/response models, backend
      tests because they need a stable JSON contract shared by route handlers, generated
      clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    id: str
    email: str
    name: str
    picture: str | None = None
    is_active: bool | None = None
    is_verified: bool | None = None
    created_at: str | None = None
    last_login: str | None = None



class AuthMethod(BaseModel):
    """API schema used by routes and generated clients for auth method.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    name: str  # "google", "github", etc. (changed from 'type' to match frontend)
    display_name: str
    enabled: bool



class AuthInfoResponse(BaseModel):
    """Main auth info response - tells frontend everything it needs to know

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    authenticated: bool
    user: User | None = None
    available_auth_methods: list[AuthMethod] = []
    requires_authentication: bool
    data_folder: str | None = None



class GoogleIn(BaseModel):
    """API schema used by routes and generated clients for google in.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    id_token: str



class GoogleOut(BaseModel):
    """API schema used by routes and generated clients for google out.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    access_token: str
    refresh_token: str
    expires_in: int
    scope: str
    token_type: str
    user: User  # Updated to use User instead of UserInfo



class UserResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for user response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    id: str  # UUID string, not integer
    email: str
    name: str
    picture: str | None = None  # Made optional
    is_active: bool
    is_verified: bool
    created_at: str  # Will be converted from datetime
    last_login: str  # Will be converted from datetime


# =============================================================================
# USER MANAGEMENT MODELS
# =============================================================================


# =============================================================================
# FILE MANAGEMENT MODELS
# =============================================================================



