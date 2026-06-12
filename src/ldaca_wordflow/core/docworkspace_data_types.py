"""DocWorkspace data-type and schema conversion utilities for FastAPI.

Used by:
- Backend API routes, worker tasks, workspace services, and backend tests because they
  need a backend boundary that validates inputs before delegating to workspace or worker
  state.

Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
    cleanup, and return stable workspace metadata to callers.
"""

from dataclasses import dataclass
from typing import Any

import polars as pl

# Import API models
from .api_models import ColumnSchema


@dataclass(frozen=True)
class Annotation:
    """Semantic annotation value stored in annotation-typed columns.

    Used by:
    - backend API routes because they need a backend boundary that validates inputs before
      delegating to workspace or worker state.

    Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
        cleanup, and return stable workspace metadata to callers.
    """

    provider: str
    annotation: str


@dataclass(frozen=True)
class TopicProportion:
    """One ``(topic_id, proportion)`` entry of a topic-distribution value.

    The list-of-these is the logical payload of the ``TMDist`` semantic type
    (see ``TM_DISTRIBUTION_POLARS_DTYPE``).

    Used by:
    - backend helpers and tests that construct/inspect topic-distribution
      values; the wire/storage form is the canonical Polars dtype below.
    """

    topic_id: int
    proportion: float


ANNOTATION_POLARS_DTYPE = pl.List(
    pl.Struct(
        [
            pl.Field("provider", pl.Utf8),
            pl.Field("annotation", pl.Utf8),
        ]
    )
)


# ``TMDist`` — the semantic data type for a per-document topic distribution.
#
# Polars has no user-extensible dtype system (unlike pandas' ``ExtensionDtype``):
# logical types are always backed by a native physical/Arrow type. The idiomatic
# way to model a domain type is therefore a canonical ``Struct``/``List`` physical
# dtype plus a semantic name by convention — exactly how ``annotation`` is handled
# above. ``TMDist`` follows that pattern: it is physically a
# ``List(Struct{topic_id: Int64, proportion: Float64})`` (proportions sum to ~1
# across a document's chunks) and is surfaced to the frontend via the logical
# ``js_type`` string ``"tmdist"`` so the data view can render it as a stacked
# proportion bar instead of raw struct text.
TM_DISTRIBUTION_POLARS_DTYPE = pl.List(
    pl.Struct(
        [
            pl.Field("topic_id", pl.Int64),
            pl.Field("proportion", pl.Float64),
        ]
    )
)


class DocWorkspaceDataTypeUtils:
    """Utilities for DocWorkspace dtype mapping and schema serialization.

    Used by:
    - backend tests, core workspace and worker services because tests need the same
      observable contract that production routes and workers rely on.

    Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
        cleanup, and return stable workspace metadata to callers.
    """

    @staticmethod
    def polars_dtype_to_ldaca_dtype(polars_dtype: pl.DataType) -> str:
        """Convert Polars dtype into LDaCA-controlled dtype categories.

        Called by:
        - `DocWorkspaceDataTypeUtils` instances owned by backend services, routes, and tests
          because they need a backend boundary that validates inputs before delegating to
          workspace or worker state.

        Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
            cleanup, and return stable workspace metadata to callers.
        """
        if polars_dtype == ANNOTATION_POLARS_DTYPE:
            return "annotation"
        if polars_dtype == TM_DISTRIBUTION_POLARS_DTYPE:
            return "tmdist"
        if polars_dtype in (
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        ):
            return "integer"
        if polars_dtype in (pl.Float32, pl.Float64):
            return "float"
        if polars_dtype == pl.Boolean:
            return "boolean"
        if polars_dtype == pl.Categorical:
            return "categorical"
        if polars_dtype in (pl.Utf8, pl.String):
            return "string"
        if polars_dtype in (pl.Date, pl.Datetime, pl.Time):
            return "datetime"
        if polars_dtype == pl.List(pl.String) or polars_dtype == pl.List(pl.Utf8):
            return "list[string]"

        cls_obj = getattr(polars_dtype, "__class__", None)
        cls_name = getattr(cls_obj, "__name__", "") if cls_obj else ""
        type_name = (
            getattr(polars_dtype, "__name__", "")
            if hasattr(polars_dtype, "__name__")
            else ""
        )
        lowered_type = type_name.lower()
        if (
            cls_name == "List"
            or lowered_type == "list"
            or cls_name == "Array"
            or lowered_type == "array"
        ):
            return "unknown"
        if cls_name == "Struct" or lowered_type == "struct":
            return "object"
        return "unknown"

    @staticmethod
    def get_node_schema_json_with_ldaca_dtype(node: Any) -> list[ColumnSchema]:
        """Build JSON-ready column schema with LDaCA dtype mapping.

        Called by:
        - `DocWorkspaceDataTypeUtils` instances owned by backend services, routes, and tests
          because they need a stable JSON contract shared by route handlers, generated clients,
          and tests.

        Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
            cleanup, and return stable workspace metadata to callers.
        """
        data_schema = node.data.collect_schema()
        return [
            ColumnSchema(
                name=col_name,
                dtype=str(polars_type),
                js_type=DocWorkspaceDataTypeUtils.polars_dtype_to_ldaca_dtype(
                    polars_type
                ),
            )
            for col_name, polars_type in data_schema.items()
        ]
