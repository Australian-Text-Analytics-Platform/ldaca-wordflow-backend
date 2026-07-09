"""Workspace route identity invariants.

These tests guard the endpoint-design rule that workspace-scoped API handlers
should take their target workspace from the URL path, not from the mutable
current-workspace pointer used by the UI selection resource.
"""

from __future__ import annotations

import ast
from pathlib import Path


WORKSPACES_API_DIR = (
    Path(__file__).parents[2] / "src" / "ldaca_wordflow" / "api" / "workspaces"
)
ROUTER_METHODS = {"get", "post", "put", "patch", "delete"}


def _is_router_method_decorator(decorator: ast.expr) -> bool:
    if isinstance(decorator, ast.Call):
        decorator = decorator.func
    return (
        isinstance(decorator, ast.Attribute)
        and decorator.attr in ROUTER_METHODS
        and isinstance(decorator.value, ast.Name)
    )


def _router_prefixes(tree: ast.Module) -> dict[str, str]:
    """Return APIRouter variable prefixes declared in one workspace route module.

    Used by:
    - ``test_workspace_scoped_route_handlers_use_path_workspace_identity`` so
      the invariant can detect handlers mounted on routers whose prefix, rather
      than decorator path, contains ``workspace_id``.
    """

    prefixes: dict[str, str] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not isinstance(statement.value, ast.Call):
            continue
        func = statement.value.func
        if not isinstance(func, ast.Name) or func.id != "APIRouter":
            continue
        prefix = ""
        for keyword in statement.value.keywords:
            if (
                keyword.arg == "prefix"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ):
                prefix = keyword.value.value
                break
        for target in statement.targets:
            if isinstance(target, ast.Name):
                prefixes[target.id] = prefix
    return prefixes


def _workspace_scoped_route(node: ast.AsyncFunctionDef, prefixes: dict[str, str]) -> str | None:
    """Return the full route path when a handler is scoped by workspace id."""

    for decorator in node.decorator_list:
        call = decorator if isinstance(decorator, ast.Call) else None
        func = call.func if call is not None else decorator
        if not _is_router_method_decorator(func):
            continue
        if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name):
            continue
        decorator_path = ""
        if (
            call is not None
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
        ):
            decorator_path = call.args[0].value
        full_path = f"{prefixes.get(func.value.id, '')}{decorator_path}"
        if "{workspace_id" in full_path:
            return full_path
    return None


def _current_workspace_target_helpers(node: ast.AST) -> list[str]:
    """List calls that recover workspace identity from hidden current state.

    Used by:
    - the route-identity invariant to ensure explicit workspace-scoped handlers
      resolve the target workspace from the URL path.
    """

    helper_names: list[str] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name) and func.id in {
            "require_current_workspace",
            "require_current_workspace_id",
        }:
            helper_names.append(func.id)
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "get_current_workspace_id"
            and isinstance(func.value, ast.Name)
            and func.value.id == "workspace_manager"
        ):
            helper_names.append("workspace_manager.get_current_workspace_id")
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "get_current_workspace"
            and isinstance(func.value, ast.Name)
            and func.value.id == "workspace_manager"
        ):
            helper_names.append("workspace_manager.get_current_workspace")
    return helper_names


def _workspace_router_depends_on_path_loader(
    statement: ast.stmt,
) -> tuple[str, str] | None:
    """Return a router name/prefix when it still preloads a workspace path."""

    if not isinstance(statement, ast.Assign):
        return None
    if not isinstance(statement.value, ast.Call):
        return None
    func = statement.value.func
    if not isinstance(func, ast.Name) or func.id != "APIRouter":
        return None

    prefix = ""
    preloads_workspace_path = False
    for keyword in statement.value.keywords:
        if (
            keyword.arg == "prefix"
            and isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, str)
        ):
            prefix = keyword.value.value
        if keyword.arg != "dependencies":
            continue
        for dependency_node in ast.walk(keyword.value):
            if isinstance(dependency_node, ast.Name) and (
                dependency_node.id == "require_workspace_path"
            ):
                preloads_workspace_path = True

    if not preloads_workspace_path or "{workspace_id" not in prefix:
        return None
    router_names = [target.id for target in statement.targets if isinstance(target, ast.Name)]
    return ", ".join(router_names), prefix


def test_workspace_scoped_route_handlers_use_path_workspace_identity() -> None:
    """Route handlers should not recover their target id from hidden state."""

    offenders: list[str] = []
    for path in sorted(WORKSPACES_API_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        prefixes = _router_prefixes(tree)
        for statement in tree.body:
            preload = _workspace_router_depends_on_path_loader(statement)
            if preload is None:
                continue
            router_name, prefix = preload
            offenders.append(
                f"{path.relative_to(WORKSPACES_API_DIR)}:{router_name} "
                f"preloads workspace state for {prefix}"
            )
        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncFunctionDef):
                continue
            route_path = _workspace_scoped_route(node, prefixes)
            if route_path is None:
                continue
            if "{workspace_id:uuid}" not in route_path:
                offenders.append(
                    f"{path.relative_to(WORKSPACES_API_DIR)}:{node.name} "
                    f"does not constrain workspace_id as a UUID for {route_path}"
                )
            arg_names = {arg.arg for arg in node.args.args}
            if "workspace_id" not in arg_names:
                offenders.append(
                    f"{path.relative_to(WORKSPACES_API_DIR)}:{node.name} "
                    f"does not declare workspace_id for {route_path}"
                )

            helper_names = _current_workspace_target_helpers(node)
            if node.name == "start_workspace_download":
                # This route compares the path id to the currently loaded
                # workspace only to decide whether to flush in-memory state
                # before packaging an inactive workspace directory.
                continue
            if helper_names:
                offenders.append(
                    f"{path.relative_to(WORKSPACES_API_DIR)}:{node.name} "
                    f"calls {sorted(set(helper_names))} for {route_path}"
                )

    assert offenders == []
