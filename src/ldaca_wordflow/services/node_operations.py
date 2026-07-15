"""Pure builders for canonical immutable node derivations.

Used by ``NodeService`` while it holds a function-scoped workspace lease. The
module contains no FastAPI or persistence code: it resolves source nodes,
builds and validates one lazy Polars plan, and returns a materialized ``Node``
only for committed creation calls. Preview calls reuse the same plan builder
without mutating the workspace graph.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, cast

import polars as pl
from ..domain.workspace import Node, TokenizationMeta, Workspace
from ..domain.workspace.provenance import (
    BinaryExpression,
    CastExpression,
    ColumnExpression,
    ConcatStringExpression,
    DerivationInput,
    DerivationOperation,
    DerivationProvenance,
    ExpressionItem,
    ExpressionSpec,
    FilterCondition,
    LiteralExpression,
    RoundExpression,
    StringExpression,
    UnaryExpression,
    derivation_operation_from_model,
    node_reference,
)

from ..analysis.topic_types import TM_DISTRIBUTION_POLARS_DTYPE
from ..shared.errors import InvalidInputError, NodeNotFoundError
from ..shared.json_data import JsonData
from .node_casting import cast_lazyframe_column
from ..models.node_resources import (
    CastNodeCreateRequest,
    CloneNodeCreateRequest,
    ConcatNodeCreateRequest,
    ExpressionNodeCreateRequest,
    FilterNodeCreateRequest,
    JoinNodeCreateRequest,
    NodeDerivationRequest,
    ReplaceNodeCreateRequest,
    SliceNodeCreateRequest,
)
from ..infrastructure.storage.layout import validate_display_name

_ISO_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d{1,6})?)?(Z|[+\-]\d{2}:?\d{2})$"
)


def build_derived_node(
    workspace: Workspace,
    request: NodeDerivationRequest,
) -> Node:
    """Build, validate, and attach one immutable child node.

    Called by ``NodeService.create_derived`` inside the workspace mutation
    gate. The node is inserted only after the lazy plan and output name have
    passed validation, so failed requests cannot alter graph topology.
    """

    lazyframe, default_name, operation, parents = build_derived_lazyframe(
        workspace,
        request,
    )
    name = (request.name or default_name).strip()
    valid, reason = validate_display_name(name)
    if not valid:
        raise InvalidInputError(f"Invalid node name: {reason}")

    # Schema resolution catches invalid columns, join keys, expressions, and
    # most dtype errors before the graph is changed or persisted.
    try:
        lazyframe.collect_schema()
    except Exception as exc:
        raise InvalidInputError(
            "The node operation does not produce a valid schema"
        ) from exc

    node = Node(
        data=lazyframe,
        name=name,
        parents=parents,
        provenance=DerivationProvenance(
            operation=operation,
            inputs=_provenance_inputs(request, parents),
        ),
        document=_propagated_document(parents, lazyframe),
        tokenization=_propagated_tokenization(parents, lazyframe),
    )
    workspace.add_node(node)
    workspace.place_node_after_parent(node)
    return node


def build_derived_lazyframe(
    workspace: Workspace,
    request: NodeDerivationRequest,
) -> tuple[pl.LazyFrame, str, DerivationOperation, list[Node]]:
    """Return the lazy result, default name, typed operation, and parents."""

    operation = derivation_operation_from_model(request)

    if isinstance(request, CloneNodeCreateRequest):
        source = _node(workspace, request.source_node_id)
        return (
            source.data.clone(),
            f"{source.name}_copy",
            operation,
            [source],
        )

    if isinstance(request, SliceNodeCreateRequest):
        source = _node(workspace, request.source_node_id)
        result, suffix = _slice(source, request)
        return result, f"{source.name}_{suffix}", operation, [source]

    if isinstance(request, FilterNodeCreateRequest):
        source = _node(workspace, request.source_node_id)
        schema = dict(source.data.collect_schema().items())
        predicate = _filter_expression(request, schema)
        return (
            source.data.filter(predicate),
            f"{source.name}_filtered",
            operation,
            [source],
        )

    if isinstance(request, ReplaceNodeCreateRequest):
        source = _node(workspace, request.source_node_id)
        output_column, expression = _replace_expression(source, request)
        return (
            source.data.with_columns(expression),
            f"{source.name}_{output_column}",
            operation,
            [source],
        )

    if isinstance(request, ExpressionNodeCreateRequest):
        source = _node(workspace, request.source_node_id)
        return (
            _apply_expression(source.data, request),
            f"{source.name}_{request.context}",
            operation,
            [source],
        )

    if isinstance(request, ConcatNodeCreateRequest):
        parents = [_node(workspace, node_id) for node_id in request.source_node_ids]
        if len({node.id for node in parents}) != len(parents):
            raise InvalidInputError("Concatenation source nodes must be distinct")
        frames = _aligned_concat_frames(parents)
        result = pl.concat(frames, how="vertical")
        if request.deduplicate:
            result = result.unique(maintain_order=True)
        labels = ", ".join(node.name for node in parents)
        return result, f"Stack({labels})", operation, parents

    if isinstance(request, JoinNodeCreateRequest):
        left = _node(workspace, request.left_node_id)
        right = _node(workspace, request.right_node_id)
        if left.id == right.id:
            raise InvalidInputError("Join source nodes must be distinct")
        if request.how == "cross":
            result = left.data.join(right.data, how="cross")
        else:
            result = left.data.join(
                right.data,
                left_on=cast(str, request.left_on),
                right_on=cast(str, request.right_on),
                how=request.how,
            )
        return (
            result,
            f"{left.name}_join_{right.name}",
            operation,
            [left, right],
        )

    if isinstance(request, CastNodeCreateRequest):
        source = _node(workspace, request.source_node_id)
        result = cast_lazyframe_column(
            source.data,
            column_name=request.column,
            target_type=request.target_type,
            datetime_format=request.datetime_format,
            strict=request.strict,
        )
        return (
            result.lazyframe,
            f"{source.name}_{request.column}_{request.target_type}",
            operation,
            [source],
        )

    raise InvalidInputError("Unsupported node operation")


def _node(workspace: Workspace, node_id: str | uuid.UUID) -> Node:
    node = workspace.nodes.get(str(node_id))
    if node is None:
        raise NodeNotFoundError("Node not found")
    return node


def _slice(
    source: Node,
    request: SliceNodeCreateRequest,
) -> tuple[pl.LazyFrame, str]:
    if request.mode == "shuffle":
        indices = pl.int_range(pl.len()).sample(
            fraction=1.0,
            shuffle=True,
            seed=request.random_seed,
        )
        return source.data.select(pl.all().gather(indices)), "shuffled"
    if request.mode == "random_sample":
        sample_size = cast(float, request.sample_size)
        indices = (
            pl.int_range(pl.len()).sample(
                fraction=sample_size, seed=request.random_seed
            )
            if sample_size < 1
            else pl.int_range(pl.len()).sample(
                n=int(sample_size), seed=request.random_seed
            )
        )
        return source.data.select(pl.all().gather(indices)), "sampled"
    return source.data.slice(request.offset, request.length), "sliced"


def _provenance_inputs(
    request: NodeDerivationRequest,
    parents: list[Node],
) -> list[DerivationInput]:
    if isinstance(request, JoinNodeCreateRequest):
        return [
            DerivationInput(role="left", value=node_reference(parents[0].id)),
            DerivationInput(role="right", value=node_reference(parents[1].id)),
        ]
    if isinstance(request, ConcatNodeCreateRequest):
        return [
            DerivationInput(role="member", value=node_reference(parent.id))
            for parent in parents
        ]
    return [DerivationInput(role="source", value=node_reference(parents[0].id))]


def _filter_expression(
    request: FilterNodeCreateRequest,
    schema: dict[str, pl.DataType],
) -> pl.Expr:
    expressions = [
        _condition_expression(condition, schema) for condition in request.conditions
    ]
    result = expressions[0]
    for expression in expressions[1:]:
        result = result | expression if request.logic == "or" else result & expression
    return result


def _condition_expression(
    condition: FilterCondition,
    schema: dict[str, pl.DataType],
) -> pl.Expr:
    if condition.column not in schema:
        raise InvalidInputError("Filter column is not present on the node")
    column = pl.col(condition.column)
    dtype = schema[condition.column]
    value = condition.value
    operator = condition.operator

    if dtype == TM_DISTRIBUTION_POLARS_DTYPE:
        expression = _topic_distribution_expression(condition)
    elif operator in {"eq", "ne", "gt", "gte", "lt", "lte"}:
        scalar = _coerce_scalar(_parse_temporal(value))
        literal = cast(
            Any,
            _temporal_literal(scalar, dtype)
            if isinstance(scalar, datetime)
            else scalar,
        )
        if operator == "eq":
            expression: pl.Expr = column == literal
        elif operator == "ne":
            expression = column != literal
        elif operator == "gt":
            expression = column > literal
        elif operator == "gte":
            expression = column >= literal
        elif operator == "lt":
            expression = column < literal
        else:
            expression = column <= literal
    elif operator == "in":
        values = value if isinstance(value, list) else [value]
        include_null = any(item is None for item in values)
        normalized = [
            _coerce_scalar(_parse_temporal(item)) for item in values if item is not None
        ]
        if dtype == pl.List(pl.String):
            expression = column.list.eval(
                pl.element().cast(pl.String).is_in([str(item) for item in normalized]),
                parallel=False,
            ).list.any()
        elif normalized:
            expression = column.is_in(normalized)
            if include_null:
                expression = expression | column.is_null()
        else:
            expression = column.is_null() if include_null else pl.lit(False)
    elif operator == "contains":
        pattern = str(value or "")
        if condition.regex:
            expression = column.cast(pl.String).str.contains(
                pattern if condition.case_sensitive else f"(?i){pattern}"
            )
        elif condition.case_sensitive:
            expression = column.cast(pl.String).str.contains(pattern, literal=True)
        else:
            expression = (
                column.cast(pl.String)
                .str.to_lowercase()
                .str.contains(pattern.lower(), literal=True)
            )
    elif operator == "starts_with":
        expression = column.cast(pl.String).str.starts_with(str(value or ""))
    elif operator == "ends_with":
        expression = column.cast(pl.String).str.ends_with(str(value or ""))
    elif operator == "is_null":
        expression = column.is_null()
    elif operator == "is_not_null":
        expression = column.is_not_null()
    elif operator == "between":
        if not isinstance(value, dict):
            raise InvalidInputError("Between filters require start and end values")
        start = _coerce_scalar(_parse_temporal(value.get("start")))
        end = _coerce_scalar(_parse_temporal(value.get("end")))
        if start is None and end is None:
            raise InvalidInputError("Between filters require a start or end value")
        if isinstance(start, datetime):
            start = _temporal_literal(start, dtype)
        if isinstance(end, datetime):
            end = _temporal_literal(end, dtype)
        if start is None:
            expression = column <= cast(Any, end)
        elif end is None:
            expression = column >= cast(Any, start)
        else:
            expression = column.is_between(
                cast(Any, start),
                cast(Any, end),
                closed="both",
            )
    else:  # pragma: no cover - the request literal already constrains this
        raise InvalidInputError("Unsupported filter operator")
    return ~expression if condition.negate else expression


def _topic_distribution_expression(condition: FilterCondition) -> pl.Expr:
    if not isinstance(condition.value, dict):
        raise InvalidInputError(
            "Topic-distribution filters require topic_id and threshold"
        )
    try:
        topic_id = int(cast(Any, condition.value.get("topic_id")))
        threshold = float(cast(Any, condition.value.get("threshold")))
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(
            "Topic-distribution filters require numeric topic_id and threshold"
        ) from exc
    proportion = (
        pl.col(condition.column)
        .list.eval(
            pl.when(pl.element().struct.field("topic_id") == topic_id)
            .then(pl.element().struct.field("proportion"))
            .otherwise(0.0)
        )
        .list.max()
        .fill_null(0.0)
    )
    operators = {
        "eq": proportion == threshold,
        "ne": proportion != threshold,
        "gt": proportion > threshold,
        "gte": proportion >= threshold,
        "lt": proportion < threshold,
        "lte": proportion <= threshold,
    }
    expression = operators.get(condition.operator)
    if expression is None:
        raise InvalidInputError("Unsupported topic-distribution filter operator")
    return expression


def _replace_expression(
    source: Node,
    request: ReplaceNodeCreateRequest,
) -> tuple[str, pl.Expr]:
    if request.source_column not in source.data.collect_schema().names():
        raise InvalidInputError("Replace source column is not present on the node")
    output = re.sub(r"\s+", " ", request.output_column or request.source_column).strip()
    if not output:
        raise InvalidInputError("Output column name cannot be blank")
    column = pl.col(request.source_column).cast(pl.String)
    if request.mode == "extract":
        extracted = column.str.extract_all(request.pattern)
        if request.count == "first":
            extracted = extracted.list.head(request.match_limit or 1)
        expression = (
            pl.when(extracted.list.len() > 0)
            .then(extracted.list.join(request.connector))
            .otherwise(pl.lit(None))
        )
    elif request.count == "all":
        expression = column.str.replace_all(request.pattern, request.replacement)
    else:
        expression = column.str.replace(
            request.pattern,
            request.replacement,
            n=request.match_limit or 1,
        )
    return output[:120], expression.alias(output[:120])


def _compile_expression(
    specification: ExpressionSpec,
    columns: set[str],
) -> pl.Expr:
    """Compile the closed expression DSL without evaluating client code."""

    if isinstance(specification, ColumnExpression):
        if specification.name not in columns:
            raise InvalidInputError("Expression column is not present on the node")
        return pl.col(specification.name)
    if isinstance(specification, LiteralExpression):
        return pl.lit(specification.value)
    if isinstance(specification, BinaryExpression):
        left = _compile_expression(specification.left, columns)
        right = _compile_expression(specification.right, columns)
        operation = specification.op
        if operation == "add":
            return left + right
        if operation == "subtract":
            return left - right
        if operation == "multiply":
            return left * right
        if operation == "divide":
            return left / right
        if operation == "modulo":
            return left % right
        if operation == "eq":
            return left == right
        if operation == "ne":
            return left != right
        if operation == "gt":
            return left > right
        if operation == "gte":
            return left >= right
        if operation == "lt":
            return left < right
        if operation == "lte":
            return left <= right
        if operation == "and":
            return left & right
        if operation == "or":
            return left | right
        if operation == "is_in":
            return left.is_in(
                cast(list[object], specification.right.value)
                if isinstance(specification.right, LiteralExpression)
                and isinstance(specification.right.value, list)
                else right
            )
        return left.fill_null(right)
    if isinstance(specification, UnaryExpression):
        operand = _compile_expression(specification.operand, columns)
        operations = {
            "not": lambda: ~operand,
            "is_null": operand.is_null,
            "is_not_null": operand.is_not_null,
            "abs": operand.abs,
            "lowercase": operand.str.to_lowercase,
            "uppercase": operand.str.to_uppercase,
            "year": operand.dt.year,
            "month": operand.dt.month,
            "day": operand.dt.day,
            "sum": operand.sum,
            "mean": operand.mean,
            "min": operand.min,
            "max": operand.max,
            "count": operand.count,
            "n_unique": operand.n_unique,
        }
        return operations[specification.op]()
    if isinstance(specification, StringExpression):
        operand = _compile_expression(specification.operand, columns).cast(pl.String)
        if specification.op == "contains":
            return operand.str.contains(
                specification.value,
                literal=specification.literal,
            )
        if specification.op == "starts_with":
            return operand.str.starts_with(specification.value)
        return operand.str.ends_with(specification.value)
    if isinstance(specification, CastExpression):
        data_type = {
            "string": pl.String,
            "integer": pl.Int64,
            "float": pl.Float64,
            "boolean": pl.Boolean,
            "datetime": pl.Datetime,
            "date": pl.Date,
        }[specification.dtype]
        return _compile_expression(specification.operand, columns).cast(
            data_type,
            strict=specification.strict,
        )
    if isinstance(specification, RoundExpression):
        return _compile_expression(specification.operand, columns).round(
            specification.decimals
        )
    if isinstance(specification, ConcatStringExpression):
        return pl.concat_str(
            [
                _compile_expression(operand, columns)
                for operand in specification.operands
            ],
            separator=specification.separator,
        )
    raise InvalidInputError("Unsupported expression operation")


def _compile_item(item: ExpressionItem, columns: set[str]) -> pl.Expr:
    expression = _compile_expression(item.expression, columns)
    return expression.alias(item.alias) if item.alias is not None else expression


def _apply_expression(
    lazyframe: pl.LazyFrame,
    request: ExpressionNodeCreateRequest,
) -> pl.LazyFrame:
    columns = set(lazyframe.collect_schema().names())
    if request.context == "filter":
        return lazyframe.filter(_compile_item(request.expressions[0], columns))
    if request.context == "with_columns":
        return lazyframe.with_columns(
            _compile_item(item, columns) for item in request.expressions
        )
    if request.context == "select":
        return lazyframe.select(
            _compile_item(item, columns) for item in request.expressions
        )
    if request.context == "sort":
        pairs = [
            (_compile_item(item, columns), item.descending)
            for item in request.expressions
        ]
        return lazyframe.sort(
            [expression for expression, _ in pairs],
            descending=[descending for _, descending in pairs],
        )
    keys = [_compile_item(item, columns) for item in request.group_by]
    aggregations = [_compile_item(item, columns) for item in request.expressions]
    return lazyframe.group_by(keys).agg(aggregations)


def _aligned_concat_frames(nodes: list[Node]) -> list[pl.LazyFrame]:
    base_schema = nodes[0].data.collect_schema()
    base_names = base_schema.names()
    frames = [nodes[0].data.select(base_names)]
    for node in nodes[1:]:
        schema = node.data.collect_schema()
        missing = sorted(set(base_names) - set(schema.names()))
        extra = sorted(set(schema.names()) - set(base_names))
        mismatched = sorted(
            name
            for name in base_names
            if name in schema and schema[name] != base_schema[name]
        )
        if missing or extra or mismatched:
            raise InvalidInputError(
                "Concatenation sources must have identical columns and data types",
                details=cast(
                    dict[str, JsonData],
                    {
                        "node_id": node.id,
                        "missing": missing,
                        "extra": extra,
                        "mismatched": mismatched,
                    },
                ),
            )
        frames.append(node.data.select(base_names))
    return frames


def _parse_temporal(value: object) -> object:
    if not isinstance(value, str) or _ISO_PATTERN.fullmatch(value) is None:
        return value
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    if re.search(r"([+\-]\d{2})(\d{2})$", normalized):
        normalized = re.sub(r"([+\-]\d{2})(\d{2})$", r"\1:\2", normalized)
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return value


def _coerce_scalar(value: object) -> object:
    if not isinstance(value, str):
        return value
    lowered = value.casefold()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return value


def _temporal_literal(value: datetime, dtype: pl.DataType) -> pl.Expr:
    if isinstance(dtype, pl.Datetime):
        timezone_name = dtype.time_zone
        if timezone_name is None and value.tzinfo is not None:
            value = value.replace(tzinfo=None)
        elif timezone_name is not None and value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return pl.lit(value).cast(dtype)
    if dtype == pl.Date:
        return pl.lit(value.date()).cast(dtype)
    return pl.lit(value)


def _propagated_tokenization(
    parents: list[Node],
    lazyframe: pl.LazyFrame,
) -> dict[str, TokenizationMeta]:
    columns = set(lazyframe.collect_schema().names())
    result: dict[str, TokenizationMeta] = {}
    for parent in parents:
        for source_column, metadata in parent.tokenization.items():
            token_column = metadata.get("column_name")
            if source_column not in columns or token_column not in columns:
                continue
            existing = result.get(source_column)
            if existing is not None and existing != metadata:
                raise InvalidInputError(
                    "Source nodes have conflicting tokenization metadata"
                )
            result[source_column] = cast(TokenizationMeta, dict(metadata))
    return result


def _propagated_document(parents: list[Node], lazyframe: pl.LazyFrame) -> str | None:
    columns = set(lazyframe.collect_schema().names())
    values = {parent.document for parent in parents if parent.document in columns}
    return next(iter(values)) if len(values) == 1 else None


__all__ = ["build_derived_lazyframe", "build_derived_node"]
