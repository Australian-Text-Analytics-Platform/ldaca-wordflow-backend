"""Remote quotation engine client utilities.

Used by:
- Backend API routes, worker tasks, workspace services, and backend tests because they
  need a backend boundary that validates inputs before delegating to workspace or worker
  state.

Flow: normalize inputs, delegate to the owning backend state or service boundary, and
    return serialized values or existing domain errors to callers.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ...models.quotation import QuotationEngineConfig, QuotationEngineType
from ...settings import settings

logger = logging.getLogger(__name__)

__all__ = [
    "QuotationServiceError",
    "normalise_engine_base_url",
    "extract_remote_quotations",
]


class QuotationServiceError(RuntimeError):
    """Raised when remote quotation extraction fails.

    Used by:
    - quotation core/worker paths calling remote extraction because background jobs need one
      lifecycle owner for submission, progress, cancellation, and artifact cleanup.
    Why:
    - Provides a domain-specific error type for consistent API translation.

    Flow: normalize inputs, delegate to the owning backend state or service boundary, and
        return serialized values or existing domain errors to callers.
    """


def normalise_engine_base_url(raw_url: str) -> str:
    """Normalise user-provided engine URL to the `/api/v1/quotation` root.

    Accepts roots that may already include `/api/v1/quotation` or `/extract` suffixes
    and trims redundant segments. Trailing slashes are stripped for consistency.

    Used by:
    - `extract_remote_quotations` because startup and shutdown flows need the same
      initialization and cleanup behavior in packaged and local runs.
    Why:
    - Prevents endpoint concatenation bugs from user-supplied base URLs.

    Flow: normalize inputs, delegate to the owning backend state or service boundary, and
        return serialized values or existing domain errors to callers.
    """

    base = (raw_url or "").strip()
    if not base:
        raise ValueError("Quotation engine URL cannot be empty")

    base = base.rstrip("/")

    # Remove terminal endpoint fragments so we always target the API root
    if base.endswith("/extract"):
        base = base[: -len("/extract")]
    if base.endswith("/health"):
        base = base[: -len("/health")]

    if not base.endswith("/api/v1/quotation"):
        base = f"{base}/api/v1/quotation"

    return base.rstrip("/")


async def extract_remote_quotations(
    engine: QuotationEngineConfig,
    documents: dict[str, dict[str, Any]],
    *,
    options: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Call remote quotation extraction API and return decoded JSON payload.

    Used by:
    - `quotation_core.extract_remote_paginated` because startup and shutdown flows need the
      same initialization and cleanup behavior in packaged and local runs.
    Why:
    - Encapsulates transport, timeout, and error normalization for remote engine calls.

    Refactor note:
    - If additional remote analysis engines are added, consider a shared HTTP
        client abstraction to consolidate retry/auth/error policy.

    Flow: normalize inputs, delegate to the owning backend state or service boundary, and
        return serialized values or existing domain errors to callers.
    """

    if engine.type is not QuotationEngineType.REMOTE:
        raise QuotationServiceError(
            "Remote extraction requested with non-remote engine config"
        )
    if not engine.url:
        raise QuotationServiceError("Remote quotation engine URL is required")

    base_url = normalise_engine_base_url(str(engine.url))
    extract_url = f"{base_url}/extract"

    payload: dict[str, Any] = {"documents": documents}
    if options:
        payload["options"] = options

    request_timeout = (
        timeout if timeout is not None else settings.quotation_service_timeout
    )

    try:
        async with httpx.AsyncClient(
            timeout=request_timeout, follow_redirects=True
        ) as client:
            response = await client.post(extract_url, json=payload)
    except httpx.RequestError as exc:
        logger.error("Quotation service unreachable at %s: %s", engine.url, exc)
        raise QuotationServiceError(
            f"Failed to reach quotation service at {engine.url}: {exc}"
        ) from exc

    if response.status_code >= 400:
        detail: Any = response.text
        try:
            body = response.json()
            if isinstance(body, dict):
                if isinstance(body.get("error"), dict):
                    detail = body["error"].get("message", detail)
                elif "message" in body:
                    detail = body["message"]
        except ValueError:
            pass
        logger.error("Quotation service returned %d: %s", response.status_code, detail)
        raise QuotationServiceError(
            f"Quotation service responded with {response.status_code}: {detail}"
        )

    try:
        return response.json()
    except ValueError as exc:  # pragma: no cover - unexpected payload
        raise QuotationServiceError(
            "Quotation service returned non-JSON response"
        ) from exc
