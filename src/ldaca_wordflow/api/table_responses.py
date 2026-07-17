"""Thin HTTP shaping for Arrow IPC table responses."""

from __future__ import annotations

from fastapi import Response

from ..shared.table_transport import (
    ARROW_STREAM_MEDIA_TYPE,
    HAS_NEXT_HEADER,
    IpcTablePage,
)

ARROW_STREAM_RESPONSE = {
    200: {
        "description": "Arrow IPC stream",
        "content": {
            ARROW_STREAM_MEDIA_TYPE: {
                "schema": {"type": "string", "format": "binary"}
            }
        },
    }
}


def arrow_stream_response(content: bytes) -> Response:
    return Response(
        content=content,
        media_type=ARROW_STREAM_MEDIA_TYPE,
        headers={"Cache-Control": "no-store"},
    )


def arrow_page_response(page: IpcTablePage) -> Response:
    return Response(
        content=page.content,
        media_type=ARROW_STREAM_MEDIA_TYPE,
        headers={
            "Cache-Control": "no-store",
            HAS_NEXT_HEADER: "true" if page.has_next else "false",
        },
    )


__all__ = ["ARROW_STREAM_RESPONSE", "arrow_page_response", "arrow_stream_response"]
