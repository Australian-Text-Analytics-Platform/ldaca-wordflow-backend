from fastapi.routing import APIRoute
from ldaca_wordflow.main import app


def _iter_route_names(routes, prefix: str = ""):
    """Yield API routes with fully expanded include prefixes.

    FastAPI 0.138 emits wrapped included routes (`_IncludedRouter`), so this
    helper recursively normalizes route paths before building path->name lookups.
    """
    for route in routes:
        if isinstance(route, APIRoute):
            yield (f"{prefix}{route.path}", route)
        elif route.__class__.__name__ == "_IncludedRouter":
            include_context = route.include_context
            nested_prefix = f"{prefix}{include_context.prefix}"
            for nested_route in include_context.included_router.routes:
                yield from _iter_route_names(
                    [nested_route],  # avoid generic traversal helpers for tiny path
                    nested_prefix,
                )


def test_openapi_operation_ids_use_route_names() -> None:
    schema = app.openapi()
    route_by_path_method = {
        (path, next(iter(route.methods)).lower()): route.name
        for path, route in _iter_route_names(app.router.routes)
        if isinstance(route, APIRoute) and route.include_in_schema and route.methods
    }

    assert (
        schema["paths"]["/api/files/"]["get"]["operationId"]
        == route_by_path_method[("/api/files/", "get")]
    )
    assert (
        schema["paths"]["/api/workspaces/nodes/{node_id}/data"]["get"]["operationId"]
        == route_by_path_method[("/api/workspaces/nodes/{node_id}/data", "get")]
    )


def test_openapi_route_names_are_unique() -> None:
    route_names = [
        route.name
        for route in app.routes
        if isinstance(route, APIRoute) and route.include_in_schema
    ]

    assert len(route_names) == len(set(route_names))
