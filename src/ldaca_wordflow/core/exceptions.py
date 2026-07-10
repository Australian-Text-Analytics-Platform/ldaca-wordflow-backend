"""App exception hierarchy.

Used by:
- Route handlers, worker tasks, and backend services.

Why:
- Standardises error -> HTTP response mapping in one place instead of ~500
  scattered ``raise HTTPException(status_code=..., detail=...)`` calls.

Flow:
- Module defines a base ``AppError`` (inheriting ``HTTPException``) with a
  ``status_code`` class attribute and subclasses for each semantic error
  category (not found, forbidden, conflict, validation, internal, etc.).
- Route handlers raise ``WorkspaceNotFoundError("my-workspace")`` instead of
  ``HTTPException(status_code=404, detail="...")``.
- ``main.app_error_handler`` converts all ``AppError`` subclasses to the
  standard ``ErrorResponse`` JSON envelope.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException


class AppError(HTTPException):
    """Base for all application-level exceptions that map to HTTP responses.

    Subclasses set ``status_code`` as a class attribute.
    """

    status_code: int = 500

    def __init__(
        self, detail: Any = None, *, headers: dict[str, str] | None = None
    ) -> None:
        super().__init__(status_code=self.status_code, detail=detail, headers=headers)


# ── 400 Bad Request ──────────────────────────────────────────────────────────


class InvalidInputError(AppError):
    status_code = 400


class UnsupportedOperationError(AppError):
    status_code = 400


# ── 401 Unauthorised ─────────────────────────────────────────────────────────


class UnauthenticatedError(AppError):
    status_code = 401


# ── 403 Forbidden ────────────────────────────────────────────────────────────


class AccessDeniedError(AppError):
    status_code = 403


class AdminAccessRequiredError(AppError):
    status_code = 403


# ── 404 Not Found ────────────────────────────────────────────────────────────


class NotFoundError(AppError):
    status_code = 404


class WorkspaceNotFoundError(AppError):
    status_code = 404


class NodeNotFoundError(AppError):
    status_code = 404


class TaskNotFoundError(AppError):
    status_code = 404


class FileNotFoundError(AppError):  # noqa: A001 (shadows builtin on purpose)
    status_code = 404


class NoActiveWorkspaceError(AppError):
    status_code = 404


# ── 409 Conflict ─────────────────────────────────────────────────────────────


class ResourceConflictError(AppError):
    status_code = 409


class AnnotationPreviewSessionConflictError(ResourceConflictError):
    """The caller's annotation preview generation is no longer current.

    Raised by:
    - ``AnnotationPreviewStore`` whenever a mutating or materialising operation
      presents a missing or superseded opaque session id. The distinct error
      code lets the Annotation UI discard stale work instead of presenting an
      ordinary validation failure or retrying against a newer session.
    """


class AnnotationPreviewSessionBusyError(ResourceConflictError):
    """The current annotation preview generation is being materialised.

    Raised by:
    - ``AnnotationPreviewStore`` when preview, override, clear, replacement, or
      another materialisation attempts to change a claimed generation. Unlike a
      stale-session conflict, this generation can become current and mutable
      again if the owner fails and releases its claim, so clients must retain its
      id and retry explicit cleanup before opening a replacement.
    """


# ── 410 Gone ─────────────────────────────────────────────────────────────────


class ResourceGoneError(AppError):
    status_code = 410


# ── 422 Unprocessable Entity ─────────────────────────────────────────────────


class ValidationError(AppError):
    status_code = 422


# ── 500 Internal Server Error ────────────────────────────────────────────────


class InternalServiceError(AppError):
    status_code = 500


# ── 502 Bad Gateway ──────────────────────────────────────────────────────────


class BadGatewayError(AppError):
    status_code = 502
