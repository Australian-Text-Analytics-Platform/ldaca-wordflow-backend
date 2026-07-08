"""Sequential-analysis worker task implementation.

Submit routes hand this worker a task-owned LazyFrame snapshot and request
payload. The worker performs the Polars aggregation and returns the same result
shape the synchronous route used to produce.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, cast

from .worker_utils import worker_task

logger = logging.getLogger(__name__)


@worker_task
def run_sequential_analysis_task(
    configure_worker_environment,
    user_id: str,
    workspace_id: str,
    input_snapshot_dir: str,
    node_id: str,
    request_payload: dict[str, Any],
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Execute sequential analysis inside a worker process.

    Used by:
    - ``core.worker.sequential_analysis_task`` and ``TASK_REGISTRY`` because the
      API submit endpoint must return quickly with a task id.

    Flow: configure the worker environment, load the snapshotted node plan,
    invoke the existing pure-Polars aggregation, and return a JSON-safe payload
    for ``WorkerTaskManager`` to persist on the analysis task.
    """

    configure_worker_environment()
    try:
        if progress_callback:
            progress_callback(0.05, "Loading sequential analysis input...")

        from .worker_input_snapshots import load_snapshot_node
        from ..api.workspaces.analyses.sequential_analysis import (
            DEFAULT_CHART_TYPE,
            _run_sequential_analysis,
        )

        snapshot_node = load_snapshot_node(input_snapshot_dir, node_id)
        if progress_callback:
            progress_callback(0.25, "Running sequential analysis...")

        result_df = _run_sequential_analysis(
            snapshot_node.data,
            time_column=str(request_payload["time_column"]),
            group_by_columns=cast(
                list[str] | None, request_payload.get("group_by_columns")
            ),
            frequency=str(request_payload.get("frequency") or "monthly"),
            sort_by_time=bool(request_payload.get("sort_by_time", True)),
            column_type=str(request_payload.get("column_type") or "datetime"),
            numeric_origin=cast(float | None, request_payload.get("numeric_origin")),
            numeric_interval=cast(float | None, request_payload.get("numeric_interval")),
            custom_interval_value=cast(
                int | None, request_payload.get("custom_interval_value")
            ),
            custom_interval_unit=cast(
                str | None, request_payload.get("custom_interval_unit")
            ),
            case_sensitive=bool(request_payload.get("case_sensitive", True)),
        )

        if progress_callback:
            progress_callback(1.0, "Sequential analysis completed")

        return {
            "state": "successful",
            "data": result_df.to_dicts(),
            "columns": list(result_df.columns),
            "total_records": len(result_df),
            "chart_type": DEFAULT_CHART_TYPE,
        }
    except Exception as exc:
        logger.exception(
            "Sequential analysis task failed for user=%s workspace=%s node=%s",
            user_id,
            workspace_id,
            node_id,
        )
        return {
            "state": "failed",
            "error": str(exc),
            "message": f"Sequential analysis task failed: {exc}",
        }


__all__ = ["run_sequential_analysis_task"]
