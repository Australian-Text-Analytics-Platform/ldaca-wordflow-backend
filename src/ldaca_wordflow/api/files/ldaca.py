"""LDaCA ONI proxy endpoints.

Used by:
- FastAPI router aggregation in ``__init__.py``.

Flow:
- Route handlers create an OniClient from settings, fetch/strip the
  X-LDACA-API-Token header, normalize results, and return typed responses.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header

from ...core.auth import get_current_user
from ...core.oni_client import OniClient, OniSearchMethod
from ...models.files import OniSearchRequest, OniSearchResponse, OniSearchResult
from ...settings import settings
from ...core.exceptions import InvalidInputError

router = APIRouter()
logger = logging.getLogger(__name__)

LDACA_API_TOKEN_HEADER = "X-LDACA-API-Token"


def _normalise_ldaca_api_token(api_token: str | None) -> str | None:
    """Normalize ldaca api token values before file-management routes uses them.

    Called by:
    - ``_ldaca_oni_client`` and ``import_ldaca_dataset`` (in ``tasks.py``).
    """

    return api_token.strip() if api_token and api_token.strip() else None


def _ldaca_oni_client(api_token: str | None) -> OniClient:
    """Support file-management routes with a ldaca oni client helper.

    Called by:
    - ldaca featured-collections and search route handlers.
    """

    return OniClient.from_settings(
        settings, token=_normalise_ldaca_api_token(api_token)
    )


def _normalise_oni_results(
    records: Sequence[Mapping[str, Any]],
) -> list[OniSearchResult]:
    """Normalize oni results values before file-management routes uses them.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning
      manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or
      response shaping.

    Called by:
    - ``list_ldaca_featured_collections`` and ``search_ldaca_collections``.
    """

    return [
        OniSearchResult.model_validate(
            {
                "collections": [],
                "file_formats": [],
                **record,
            }
        )
        for record in records
    ]


@router.get("/ldaca/featured", response_model=OniSearchResponse)
async def list_ldaca_featured_collections(
    current_user: dict = Depends(get_current_user),
    ldaca_api_token: Annotated[str | None, Header(alias=LDACA_API_TOKEN_HEADER)] = None,
):
    """Return staff-picked LDaCA collections for the import dialog.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the
      owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI GET /ldaca/featured route.
    """
    del current_user
    client = _ldaca_oni_client(ldaca_api_token)
    data = _normalise_oni_results(
        await client.featured_collections(
            settings.get_ldaca_oni_featured_collection_ids()
        )
    )
    return {
        "state": "successful",
        "data": data,
        "message": "LDaCA featured collections loaded",
    }


@router.post("/ldaca/search", response_model=OniSearchResponse)
async def search_ldaca_collections(
    request: OniSearchRequest,
    current_user: dict = Depends(get_current_user),
    ldaca_api_token: Annotated[str | None, Header(alias=LDACA_API_TOKEN_HEADER)] = None,
):
    """Search LDaCA records through the backend ONI proxy.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the
      owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI POST /ldaca/search route.
    """
    del current_user
    try:
        method = OniSearchMethod(request.method)
    except ValueError as exc:
        raise InvalidInputError("Unsupported LDaCA search method") from exc
    client = _ldaca_oni_client(ldaca_api_token)
    data = _normalise_oni_results(
        await client.search(
            method=method,
            query=request.query,
            limit=request.limit,
            offset=request.offset,
        )
    )
    return {
        "state": "successful",
        "data": data,
        "message": "LDaCA search completed",
    }
