"""Task Center SSE stream helpers.

Used by:
- ``api.tasks.stream_tasks`` because the route should only resolve stream
  authentication and return a ``StreamingResponse`` while this module owns the
  event generator, heartbeat, error event, and unsubscribe lifecycle.

Flow:
- Subscribe to the per-user worker task manager queue.
- Emit an initial snapshot, then incremental events or heartbeat events.
- On cancellation/error, log and emit an error payload when possible.
- Always unsubscribe the queue before the generator exits.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator

from .workspace import workspace_manager


logger = logging.getLogger(__name__)

TASK_STREAM_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Cache-Control",
}


def _sse_data(payload: dict) -> str:
    """Serialize one payload as a server-sent event frame."""

    return f"data: {json.dumps(payload)}\n\n"


async def task_event_stream(user_id: str) -> AsyncIterator[str]:
    """Yield Task Center server-sent events for one authenticated user.

    Used by:
    - ``api.tasks.stream_tasks`` as the ``StreamingResponse`` body iterator.
    """

    task_manager = workspace_manager.get_task_manager(user_id)
    queue = await task_manager.subscribe(user_id)

    try:
        tasks = await task_manager.list(user_id=user_id)
        yield _sse_data(
            {
                "type": "tasks_snapshot",
                "tasks": [task for task in tasks if isinstance(task, dict)],
                "timestamp": time.time(),
            }
        )

        last_heartbeat = time.time()
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                if time.time() - last_heartbeat > 30:
                    yield _sse_data(
                        {
                            "type": "heartbeat",
                            "timestamp": time.time(),
                        }
                    )
                    last_heartbeat = time.time()
                continue

            if not isinstance(event, dict):
                continue

            yield _sse_data(event)
            last_heartbeat = time.time()

    except asyncio.CancelledError:  # pragma: no cover
        logger.debug("Unified SSE stream cancelled for user %s", user_id)
    except Exception as exc:  # pragma: no cover
        logger.error("Unified SSE stream error: %s", exc)
        yield _sse_data(
            {
                "type": "error",
                "message": str(exc),
                "timestamp": time.time(),
            }
        )
    finally:
        try:
            await task_manager.unsubscribe(user_id, None, queue)
        except Exception as exc:  # pragma: no cover
            logger.error(
                "Error unsubscribing unified stream for user %s: %s", user_id, exc
            )
