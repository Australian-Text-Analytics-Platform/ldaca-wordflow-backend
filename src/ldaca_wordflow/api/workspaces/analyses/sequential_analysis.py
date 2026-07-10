"""Sequential Analysis endpoints extracted from monolithic base module.

Exposes updated paths:
    POST /workspaces/{workspace_id}/nodes/{node_id}/sequential-analysis
    PATCH /workspaces/{workspace_id}/analysis-tasks/{task_id}/preferences

Used by:
- FastAPI workspace analysis routers, frontend analysis features, and backend tests because they need this unit's "Sequential Analysis endpoints extracted from monolithic base module" behavior.

Flow:
- FastAPI mounts these routes through the workspace package router.
- Route handlers validate temporal columns, chart preferences, grouping, and detach requests.
- Helpers submit or read sequential analysis tasks and preserve selected visible-group state.
- Responses return previews, task results, preference updates, or detached-node metadata.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Optional, cast
from uuid import UUID

import polars as pl
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict

from docworkspace import Node

from ....analysis.manager import get_task_manager
from ....analysis.models import AnalysisStatus
from ....analysis.results import GenericAnalysisResult
from ....core.auth import get_current_user
from ....core.exceptions import (
    InternalServiceError,
    InvalidInputError,
    NotFoundError,
    TaskNotFoundError,
)
from ....models.sequential_analysis import (
    SequentialAnalysisPreferenceUpdateRequest,
    SequentialAnalysisPreviewResponse,
    SequentialAnalysisRequest,
    SequentialAnalysisResponse,
)
from ..utils import ensure_task_synced, require_workspace, update_workspace
from .sequential_analysis_submission import submit_sequential_analysis

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id:uuid}",
)


VALID_CHART_TYPES = {"line", "bar", "area"}
DEFAULT_CHART_TYPE = "line"

# Polars duration suffix and strftime format for each custom-interval unit.
# Keys must match the Literal in `SequentialAnalysisRequest.custom_interval_unit`.
_CUSTOM_UNIT_SPEC: dict[str, tuple[str, str]] = {
    "seconds": ("s", "%Y-%m-%d %H:%M:%S"),
    "minutes": ("m", "%Y-%m-%d %H:%M"),
    "hours": ("h", "%Y-%m-%d %H:00"),
    "days": ("d", "%Y-%m-%d"),
    "weeks": ("w", "%Y-%m-%d"),
}


class SelectedPeriod(BaseModel):
    """API schema used by routes and generated clients for selected period.

    Used by:
    - backend API routes because they need this unit's "API schema used by routes and generated clients for selected period" behavior.
    """

    period_start: str | int | float
    period_end: str | int | float


class VisibleGroupSelection(BaseModel):
    """One exact chart-series identity selected for sequential detachment.

    Used by:
    - ``detach_sequential_analysis`` because the chart sends the visible legend
      groups back to the backend so only those source rows are materialised.
      ``None`` is a real group value for nullable columns and compiles to
      ``is_null()`` in ``_build_group_filter_expression``.
    """

    values: dict[str, str | int | float | bool | None]


class SequentialAnalysisDetachRequest(BaseModel):
    """Request schema used by API routes and generated clients for sequential analysis detach request.

    Used by:
    - backend API routes because they need this unit's "Request schema used by API routes and generated clients for sequential analysis detach request" behavior.
    """

    selected_periods: list[SelectedPeriod]
    visible_groups: list[VisibleGroupSelection] | None = None
    new_node_name: str
    model_config = ConfigDict(extra="forbid")


def _coerce_period_bound(value: Any, *, column_type: str, time_dtype: Any) -> Any:
    """Coerce period bound values into the shape expected by sequential-analysis routes.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Coerce period bound values into the shape expected by sequential-analysis routes" behavior.
    """

    if column_type == "numeric":
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise InvalidInputError(
                f"Invalid numeric period bound: {value!r}",
            ) from exc
    if isinstance(value, datetime | date):
        parsed: datetime | date = value
    elif isinstance(value, str):
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise InvalidInputError(
                f"Invalid datetime period bound: {value!r}",
            ) from exc
    else:
        raise InvalidInputError(
            f"Unsupported datetime period bound: {value!r}",
        )
    if time_dtype == pl.Date and isinstance(parsed, datetime):
        return parsed.date()
    return parsed


def _build_group_filter_expression(
    *,
    visible_groups: list[VisibleGroupSelection] | None,
    schema: Any,
    case_sensitive: bool,
) -> pl.Expr | None:
    """Build group filter expression values used by sequential-analysis routes.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Build group filter expression values used by sequential-analysis routes" behavior.
    """

    if not visible_groups:
        return None

    group_expr: pl.Expr | None = None
    for group_selection in visible_groups:
        value_expr: pl.Expr | None = None
        for column_name, raw_value in group_selection.values.items():
            if schema.get(column_name) is None:
                raise InvalidInputError(
                    f"Group column '{column_name}' is not available on the source node",
                )
            column_expr = pl.col(column_name)
            if raw_value is None:
                current_expr = column_expr.is_null()
            elif (
                not case_sensitive
                and isinstance(raw_value, str)
                and schema.get(column_name) in {pl.String, pl.Utf8}
            ):
                current_expr = column_expr.str.to_lowercase() == raw_value.lower()
            else:
                current_expr = column_expr == pl.lit(raw_value)

            value_expr = (
                current_expr if value_expr is None else (value_expr & current_expr)
            )

        if value_expr is None:
            continue
        group_expr = value_expr if group_expr is None else (group_expr | value_expr)

    return group_expr


def _get_path_workspace(user_id: str, workspace_id: UUID) -> tuple[str, Any]:
    """Return the workspace selected by the explicit route path.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Sequential-analysis route handlers that need the workspace selected by
      the URL path.
    """

    workspace_id_str = str(workspace_id)
    ws = require_workspace(user_id, workspace_id_str)
    return workspace_id_str, ws


def _normalize_type_name(value: object | None) -> str | None:
    """Normalize type name values before sequential-analysis routes uses them.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Normalize type name values before sequential-analysis routes uses them" behavior.
    """

    if value is None:
        return None
    text = str(value).lower()
    if any(token in text for token in ("datetime", "timestamp")):
        return "datetime"
    if "date" in text and "update" not in text:
        return "datetime"
    if "time" in text and "interval" not in text:
        return "datetime"
    if "int" in text and "interval" not in text:
        return "integer"
    if any(token in text for token in ("float", "double", "decimal", "numeric")):
        return "float"
    return None


def _column_type_lookup(schema: Any) -> dict[str, str]:
    """Support sequential-analysis routes with a column type lookup helper.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Support sequential-analysis routes with a column type lookup helper" behavior.
    """

    lookup: dict[str, str] = {}
    for name, raw in zip(schema.names(), schema.dtypes()):
        if not isinstance(name, str):
            continue
        normalized = _normalize_type_name(raw)
        if normalized:
            lookup.setdefault(name, normalized)
    return lookup


# ---------------------------------------------------------------------------
# Standalone sequential-analysis logic (ported from docframe)
# ---------------------------------------------------------------------------


def _run_sequential_analysis(
    lf: pl.LazyFrame,
    *,
    time_column: str,
    group_by_columns: list[str] | None = None,
    frequency: str = "monthly",
    sort_by_time: bool = True,
    column_type: str = "datetime",
    numeric_origin: float | None = None,
    numeric_interval: float | None = None,
    custom_interval_value: int | None = None,
    custom_interval_unit: str | None = None,
    case_sensitive: bool = True,
) -> pl.DataFrame:
    """Pure-Polars implementation of sequential analysis.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Used by:
    - `run_sequential_analysis` because they need this unit's "Pure-Polars implementation of sequential analysis" behavior.

    Why:
    - Keeps binning/grouping logic independent from route orchestration.

    Groups records by time period (datetime truncation or numeric binning),
    counts occurrences per group, and returns a DataFrame with aggregated
    results.  No text-processing dependency required.
    """

    normalized_column_type = (column_type or "datetime").lower()
    if normalized_column_type not in {"datetime", "numeric"}:
        raise ValueError(
            "Unsupported column_type. Use 'datetime' or 'numeric' for sequential analysis"
        )

    # Collect to DataFrame for aggregation
    df = cast(pl.DataFrame, lf.collect())

    time_format = ""
    numeric_interval_value: float | None = None
    numeric_origin_value: float | None = None

    if normalized_column_type == "datetime":
        if frequency == "second":
            time_expr = pl.col(time_column).dt.truncate("1s").alias("time_period")
            time_format = "%Y-%m-%d %H:%M:%S"
        elif frequency == "minute":
            time_expr = pl.col(time_column).dt.truncate("1m").alias("time_period")
            time_format = "%Y-%m-%d %H:%M"
        elif frequency == "hourly":
            time_expr = pl.col(time_column).dt.truncate("1h").alias("time_period")
            time_format = "%Y-%m-%d %H:%M"
        elif frequency == "daily":
            time_expr = pl.col(time_column).dt.date().alias("time_period")
            time_format = "%Y-%m-%d"
        elif frequency == "weekly":
            time_expr = (
                pl.col(time_column).dt.truncate("1w").dt.date().alias("time_period")
            )
            time_format = "%Y-W%U"
        elif frequency == "monthly":
            time_expr = (
                pl.col(time_column).dt.truncate("1mo").dt.date().alias("time_period")
            )
            time_format = "%Y-%m"
        elif frequency == "quarterly":
            time_expr = (
                pl.col(time_column).dt.truncate("3mo").dt.date().alias("time_period")
            )
            time_format = "%Y-Q"
        elif frequency == "yearly":
            time_expr = (
                pl.col(time_column).dt.truncate("1y").dt.date().alias("time_period")
            )
            time_format = "%Y"
        elif frequency == "custom":
            if custom_interval_value is None or custom_interval_value <= 0:
                raise ValueError(
                    "custom_interval_value must be a positive integer when frequency='custom'"
                )
            unit_spec = _CUSTOM_UNIT_SPEC.get(custom_interval_unit or "")
            if unit_spec is None:
                raise ValueError(
                    f"Unsupported custom_interval_unit '{custom_interval_unit}'. "
                    f"Use one of: {sorted(_CUSTOM_UNIT_SPEC)}"
                )
            duration_suffix, time_format = unit_spec
            duration = f"{int(custom_interval_value)}{duration_suffix}"
            time_expr = pl.col(time_column).dt.truncate(duration).alias("time_period")
        else:
            time_expr = pl.col(time_column).dt.date().alias("time_period")
            time_format = "%Y-%m-%d"

        df = df.with_columns(time_expr)
    else:
        # Numeric binning
        if numeric_interval is None or numeric_interval <= 0:
            raise ValueError(
                "numeric_interval must be a positive number for numeric sequential analysis"
            )
        numeric_interval_value = float(numeric_interval)
        if numeric_origin is not None:
            numeric_origin_value = float(numeric_origin)
        else:
            origin_series = df.select(
                pl.col(time_column).cast(pl.Float64()).min()
            ).to_series()
            numeric_origin_value = origin_series[0] if len(origin_series) else None
        if numeric_origin_value is None:
            raise ValueError(
                "Unable to determine numeric_origin from the provided data"
            )

        df = df.with_columns(
            pl.col(time_column).cast(pl.Float64()).alias("__numeric_value__"),
        )
        df = df.with_columns(
            (
                (pl.col("__numeric_value__") - pl.lit(numeric_origin_value))
                / pl.lit(numeric_interval_value)
            )
            .floor()
            .cast(pl.Int64)
            .alias("__numeric_bin__"),
        )
        df = df.with_columns(
            (
                pl.lit(numeric_origin_value)
                + pl.col("__numeric_bin__").cast(pl.Float64)
                * pl.lit(numeric_interval_value)
            ).alias("time_period"),
        )

    # Determine grouping columns
    group_cols = ["time_period"] + (group_by_columns or [])

    # Lowercase group-by column values for case-insensitive grouping
    if not case_sensitive and group_by_columns:
        for col_name in group_by_columns:
            if (
                df.schema.get(col_name) == pl.String
                or df.schema.get(col_name) == pl.Utf8
            ):
                df = df.with_columns(pl.col(col_name).str.to_lowercase())

    # Perform aggregation
    result_df = df.group_by(group_cols).agg(
        [
            pl.len().alias("sequential_count"),
            pl.col(time_column).min().alias("period_start"),
            pl.col(time_column).max().alias("period_end"),
        ]
    )

    # Add formatted time period for display
    if normalized_column_type == "datetime":
        if frequency == "weekly":
            result_df = result_df.with_columns(
                pl.col("time_period")
                .dt.strftime("%Y-W%W")
                .alias("time_period_formatted")
            )
        elif frequency == "quarterly":
            result_df = result_df.with_columns(
                [
                    pl.col("time_period").dt.year().alias("__year__"),
                    ((pl.col("time_period").dt.month() - 1).floordiv(3).add(1)).alias(
                        "__quarter__"
                    ),
                ]
            )
            result_df = result_df.with_columns(
                pl.format(
                    "{}-Q{}",
                    pl.col("__year__"),
                    pl.col("__quarter__"),
                ).alias("time_period_formatted")
            ).drop(["__year__", "__quarter__"])
        else:
            result_df = result_df.with_columns(
                pl.col("time_period")
                .dt.strftime(time_format)
                .alias("time_period_formatted")
            )
    else:
        interval_lit = pl.lit(numeric_interval_value)
        result_df = result_df.with_columns(
            [
                pl.col("time_period").round(6).alias("time_period"),
                (pl.col("time_period") + interval_lit).alias("__numeric_period_end__"),
            ]
        )

        def _format_numeric(value: Optional[float]) -> Optional[str]:
            """Format numeric values for sequential-analysis routes responses.

            Called by:
            - The `_run_sequential_analysis` local workflow in this module because they need this unit's "Format numeric values for sequential-analysis routes responses" behavior.
            """

            if value is None:
                return None
            return format(value, ".6g")

        result_df = result_df.with_columns(
            [
                pl.col("time_period")
                .map_elements(_format_numeric, return_dtype=pl.String)
                .alias("__numeric_period_label_start__"),
                pl.col("__numeric_period_end__")
                .map_elements(_format_numeric, return_dtype=pl.String)
                .alias("__numeric_period_label_end__"),
            ]
        )
        result_df = result_df.with_columns(
            pl.format(
                "[{}, {})",
                pl.col("__numeric_period_label_start__"),
                pl.col("__numeric_period_label_end__"),
            ).alias("time_period_formatted")
        ).drop(
            [
                "__numeric_period_end__",
                "__numeric_period_label_start__",
                "__numeric_period_label_end__",
            ]
        )

    # Sort by time if requested
    if sort_by_time:
        sort_cols = ["time_period"] + (group_by_columns or [])
        result_df = result_df.sort(sort_cols)

    return result_df


@router.post(
    "/nodes/{node_id}/sequential-analysis/preview",
    response_model=SequentialAnalysisPreviewResponse,
)
async def preview_sequential_analysis(
    workspace_id: UUID,
    node_id: str,
    request: SequentialAnalysisRequest,
    include_data: bool = Query(
        False,
        description="If true, returns the full aggregated rows in addition to the row count.",
    ),
    current_user: dict = Depends(get_current_user),
):
    """Run the aggregation server-side but skip task registration and
    conflict checks.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
        - frontend preview/count checks because they need row-count feedback without registering
            or replacing the live analysis task.
        - callers that pass ``include_data=true`` because they need the same aggregated rows the
            regular endpoint produces without mutating task state.

    Why a separate endpoint rather than a flag on the existing one:
    the existing endpoint is task-aware (rejects conflicting requests,
    inherits chart_type, writes the result into the task store). All
        of that machinery is wrong for a pure preview query.
    """
    user_id = current_user["id"]
    _workspace_id, ws = _get_path_workspace(user_id, workspace_id)

    try:
        node = ws.nodes[node_id]
        node_data = node.data

        if request.group_by_columns and len(request.group_by_columns) > 3:
            raise InvalidInputError("Maximum 3 group by columns allowed")
        sequential_result = _run_sequential_analysis(
            node_data,
            time_column=request.time_column,
            group_by_columns=request.group_by_columns,
            frequency=request.frequency,
            # Sort is cheap and gives include-data preview callers chronological rows.
            sort_by_time=include_data,
            column_type=request.column_type,
            numeric_origin=request.numeric_origin,
            numeric_interval=request.numeric_interval,
            custom_interval_value=request.custom_interval_value,
            custom_interval_unit=request.custom_interval_unit,
            case_sensitive=request.case_sensitive,
        )

        payload: dict[str, Any] = {
            "state": "successful",
            "total_records": len(sequential_result),
            "columns": list(sequential_result.columns),
        }
        if include_data:
            payload["data"] = sequential_result.to_dicts()
            payload["analysis_params"] = {
                "time_column": request.time_column,
                "group_by_columns": request.group_by_columns or [],
                "frequency": request.frequency,
                "column_type": request.column_type,
                "numeric_origin": request.numeric_origin,
                "numeric_interval": request.numeric_interval,
                "case_sensitive": request.case_sensitive,
            }
        return payload
    except Exception as exc:  # pragma: no cover
        logger.error("Sequential analysis preview error: %s", exc, exc_info=True)
        raise InternalServiceError(f"Internal server error: {exc}")


@router.post(
    "/nodes/{node_id}/sequential-analysis",
    response_model=SequentialAnalysisResponse,
)
async def run_sequential_analysis(
    workspace_id: UUID,
    node_id: str,
    request: SequentialAnalysisRequest,
    current_user: dict = Depends(get_current_user),
):
    """Run sequential analysis for one node and persist/update task payload.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - frontend sequential-analysis run action because they need this unit's "Run sequential analysis for one node and persist/update task payload" behavior.

    Why:
    - Produces aggregated time-series counts and stores them under an explicit task id.
    """
    user_id = current_user["id"]
    workspace_id_str, ws = _get_path_workspace(user_id, workspace_id)
    return await submit_sequential_analysis(
        user_id=user_id,
        workspace_id=workspace_id_str,
        workspace=ws,
        node_id=node_id,
        request=request,
    )


async def sequential_analysis_task_result(
    workspace_id: str,
    task_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return stored result payload for a sequential-analysis task.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - shared analysis-task result route because sequential-analysis results
      need the existing worker-sync and pending-state normalization.
    """
    user_id = current_user["id"]
    task_manager = get_task_manager(user_id)

    task = await ensure_task_synced(user_id, workspace_id, task_id, task_manager)
    if not task:
        raise TaskNotFoundError("Task not found")
    result = task.result
    if result is None:
        return {"state": "pending", "metadata": {"task_id": task_id}}

    return result.to_json()


async def update_sequential_analysis_task_result(
    workspace_id: str,
    task_id: str,
    updates: SequentialAnalysisPreferenceUpdateRequest | None,
    current_user: dict = Depends(get_current_user),
):
    """Persist display-only sequential analysis options on a saved task.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - shared analysis-task preferences route because chart-type changes are
      presentation-only task preferences.

    Why:
    - Avoids recomputation when only chart presentation changes.

    Refactor note:
    - Mirrors preference update behavior in other analyses; a shared
        task-preferences helper could reduce duplication.
    """
    user_id = current_user["id"]
    task_manager = get_task_manager(user_id)
    task = task_manager.get_task(task_id)
    if not task or not task.result:
        raise NotFoundError("No sequential analysis found")
    result_payload = task.result.to_json()
    if not isinstance(result_payload, dict):
        result_payload = {}

    chart_type = result_payload.get("chart_type")
    if not isinstance(chart_type, str) or chart_type not in VALID_CHART_TYPES:
        chart_type = DEFAULT_CHART_TYPE

    if updates is not None and updates.chart_type is not None:
        candidate = updates.chart_type
        if not isinstance(candidate, str) or candidate not in VALID_CHART_TYPES:
            raise InvalidInputError(
                "Invalid chart type. Valid options are: line, bar, area",
            )
        chart_type = candidate

    result_payload["chart_type"] = chart_type

    task.result = GenericAnalysisResult(result_payload)
    task.status = AnalysisStatus.COMPLETED
    task_manager.save_task(task)

    return {
        "state": "successful",
        "message": "saved",
        "data": {"chart_type": chart_type},
    }


async def detach_sequential_analysis_task(
    workspace_id: UUID,
    task_id: str,
    request: SequentialAnalysisDetachRequest,
    current_user: dict = Depends(get_current_user),
):
    """Create a filtered child node from selected sequential-analysis periods.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - shared analysis-task detachment route because selected
      sequential-analysis periods should be detached through the task resource.
    """
    user_id = current_user["id"]
    workspace_id_str, ws = _get_path_workspace(user_id, workspace_id)

    if not request.selected_periods:
        raise InvalidInputError("At least one selected period is required")
    task_manager = get_task_manager(user_id)
    task = task_manager.get_task(task_id)
    if task is None or task.request is None:
        raise NotFoundError("Sequential analysis task not found")
    stored_request = task.request
    node_id = getattr(stored_request, "node_id", None)
    time_column = getattr(stored_request, "time_column", None)
    column_type = getattr(stored_request, "column_type", "datetime") or "datetime"
    case_sensitive = bool(getattr(stored_request, "case_sensitive", True))

    if not isinstance(node_id, str) or not node_id:
        raise InvalidInputError("Sequential analysis task is missing node_id")
    if not isinstance(time_column, str) or not time_column:
        raise InvalidInputError("Sequential analysis task is missing time_column")
    if node_id not in ws.nodes:
        raise NotFoundError("Source node not found")
    source_node = ws.nodes[node_id]
    source_lazy = source_node.data
    schema = source_lazy.collect_schema()
    time_dtype = schema.get(time_column)
    if time_dtype is None:
        raise InvalidInputError(
            f"Time column '{time_column}' is not available on the source node",
        )
    filter_expr: pl.Expr | None = None
    for selected_period in request.selected_periods:
        period_start = _coerce_period_bound(
            selected_period.period_start,
            column_type=column_type,
            time_dtype=time_dtype,
        )
        period_end = _coerce_period_bound(
            selected_period.period_end,
            column_type=column_type,
            time_dtype=time_dtype,
        )
        period_expr = (pl.col(time_column) >= pl.lit(period_start)) & (
            pl.col(time_column) <= pl.lit(period_end)
        )
        filter_expr = (
            period_expr if filter_expr is None else (filter_expr | period_expr)
        )

    if filter_expr is None:
        raise InvalidInputError("No valid period filters were provided")
    group_expr = _build_group_filter_expression(
        visible_groups=request.visible_groups,
        schema=schema,
        case_sensitive=case_sensitive,
    )
    final_filter_expr = (
        filter_expr if group_expr is None else (filter_expr & group_expr)
    )

    filtered_lazy = source_lazy.filter(final_filter_expr)

    new_node = Node(
        data=filtered_lazy,
        name=request.new_node_name,
        workspace=ws,
        operation=f"sequential_analysis_detach({getattr(source_node, 'name', node_id)})",
        parents=[source_node],
    )
    ws.add_node(new_node)
    # Smart insertion: keep the detached node directly below its source node.
    ws.place_node_after_parent(new_node)
    update_workspace(user_id, workspace_id_str, ws, best_effort=True)

    return {
        "new_node_id": getattr(new_node, "id", None),
        "new_node_name": request.new_node_name,
    }
