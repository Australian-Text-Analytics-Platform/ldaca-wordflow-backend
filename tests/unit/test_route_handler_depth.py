"""Static route-depth guardrails for backend router simplification.

Used by:
- backend endpoint-design cleanup because route handlers should remain shallow
  HTTP boundaries and delegate domain workflows to named services.
"""

from __future__ import annotations

import ast
from pathlib import Path


AUTH_API = Path(__file__).parents[2] / "src" / "ldaca_wordflow" / "api" / "auth.py"
WORKSPACE_BASE = (
    Path(__file__).parents[2] / "src" / "ldaca_wordflow" / "api" / "workspaces" / "base.py"
)
WORKSPACE_ANNOTATION = (
    Path(__file__).parents[2]
    / "src"
    / "ldaca_wordflow"
    / "api"
    / "workspaces"
    / "annotation.py"
)
TASKS_API = Path(__file__).parents[2] / "src" / "ldaca_wordflow" / "api" / "tasks.py"
FILES_PREVIEW = (
    Path(__file__).parents[2]
    / "src"
    / "ldaca_wordflow"
    / "api"
    / "files"
    / "preview.py"
)
WORKSPACE_LIFECYCLE = (
    Path(__file__).parents[2]
    / "src"
    / "ldaca_wordflow"
    / "api"
    / "workspaces"
    / "lifecycle.py"
)
TOKEN_FREQUENCIES = (
    Path(__file__).parents[2]
    / "src"
    / "ldaca_wordflow"
    / "api"
    / "workspaces"
    / "analyses"
    / "token_frequencies.py"
)
SEQUENTIAL_ANALYSIS = (
    Path(__file__).parents[2]
    / "src"
    / "ldaca_wordflow"
    / "api"
    / "workspaces"
    / "analyses"
    / "sequential_analysis.py"
)
TOPIC_MODELING = (
    Path(__file__).parents[2]
    / "src"
    / "ldaca_wordflow"
    / "api"
    / "workspaces"
    / "analyses"
    / "topic_modeling.py"
)
CONCORDANCE = (
    Path(__file__).parents[2]
    / "src"
    / "ldaca_wordflow"
    / "api"
    / "workspaces"
    / "analyses"
    / "concordance.py"
)
QUOTATION = (
    Path(__file__).parents[2]
    / "src"
    / "ldaca_wordflow"
    / "api"
    / "workspaces"
    / "analyses"
    / "quotation.py"
)


def _function_line_count(path: Path, function_name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == function_name:
            assert node.end_lineno is not None
            return node.end_lineno - node.lineno + 1
    raise AssertionError(f"{function_name} not found in {path}")


def test_cast_node_route_stays_a_shallow_http_boundary() -> None:
    """``cast_node`` should delegate Polars casting details to a service."""

    assert _function_line_count(WORKSPACE_BASE, "cast_node") <= 80


def test_calculate_token_frequencies_route_stays_a_shallow_http_boundary() -> None:
    """``calculate_token_frequencies`` should delegate task submission details."""

    assert _function_line_count(TOKEN_FREQUENCIES, "calculate_token_frequencies") <= 80


def test_upload_workspace_zip_route_stays_a_shallow_http_boundary() -> None:
    """``upload_workspace_zip`` should delegate archive import details."""

    assert _function_line_count(WORKSPACE_LIFECYCLE, "upload_workspace_zip") <= 80


def test_annotation_ai_preview_route_stays_a_shallow_http_boundary() -> None:
    """``annotate_ai_preview`` should delegate preview workflow details."""

    assert _function_line_count(WORKSPACE_ANNOTATION, "annotate_ai_preview") <= 80


def test_annotation_ai_all_route_stays_a_shallow_http_boundary() -> None:
    """``annotate_ai_all`` should delegate full annotation workflow details."""

    assert _function_line_count(WORKSPACE_ANNOTATION, "annotate_ai_all") <= 80


def test_detach_ai_previewed_rows_route_stays_a_shallow_http_boundary() -> None:
    """``detach_ai_previewed_rows`` should delegate detach workflow details."""

    assert _function_line_count(WORKSPACE_ANNOTATION, "detach_ai_previewed_rows") <= 80


def test_run_sequential_analysis_route_stays_a_shallow_http_boundary() -> None:
    """``run_sequential_analysis`` should delegate task submission details."""

    assert _function_line_count(SEQUENTIAL_ANALYSIS, "run_sequential_analysis") <= 80


def test_run_topic_modeling_route_stays_a_shallow_http_boundary() -> None:
    """``run_topic_modeling`` should delegate task submission details."""

    assert _function_line_count(TOPIC_MODELING, "run_topic_modeling") <= 80


def test_stream_tasks_route_stays_a_shallow_http_boundary() -> None:
    """``stream_tasks`` should delegate SSE event generation details."""

    assert _function_line_count(TASKS_API, "stream_tasks") <= 80


def test_cilogon_callback_route_stays_a_shallow_http_boundary() -> None:
    """``cilogon_callback`` should delegate provider/session workflow details."""

    assert _function_line_count(AUTH_API, "cilogon_callback") <= 80


def test_unified_file_preview_route_stays_a_shallow_http_boundary() -> None:
    """``unified_file_preview`` should delegate file parsing details."""

    assert _function_line_count(FILES_PREVIEW, "unified_file_preview") <= 80


def test_export_nodes_route_stays_a_shallow_http_boundary() -> None:
    """``export_nodes`` should delegate file-building details."""

    assert _function_line_count(WORKSPACE_BASE, "export_nodes") <= 80


def test_add_node_to_workspace_route_stays_a_shallow_http_boundary() -> None:
    """``add_node_to_workspace`` should delegate file staging details."""

    assert _function_line_count(WORKSPACE_BASE, "add_node_to_workspace") <= 80


def test_run_concordance_route_stays_a_shallow_http_boundary() -> None:
    """``run_concordance`` should delegate task submission details."""

    assert _function_line_count(CONCORDANCE, "run_concordance") <= 80


def test_get_quotation_route_stays_a_shallow_http_boundary() -> None:
    """``get_quotation`` should delegate task submission details."""

    assert _function_line_count(QUOTATION, "get_quotation") <= 80
