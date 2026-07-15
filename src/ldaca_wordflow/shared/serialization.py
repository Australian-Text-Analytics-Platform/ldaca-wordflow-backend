"""JSON-safe integer serialization helpers.

Used by:
- backend API routes, core workspace and worker services because they need a backend
  boundary that validates inputs before delegating to workspace or worker state.

Flow: walk row dictionaries (flat or grouped) and convert out-of-range integers to
    strings so JavaScript receivers never lose precision.
"""

from typing import Any, overload

_JS_MAX_SAFE_INTEGER = 2**53 - 1


@overload
def stringify_unsafe_integers(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Type signature used by callers passing flat row payloads.

    Used by:
    - backend API routes, core workspace and worker services because they need a backend
      boundary that validates inputs before delegating to workspace or worker state.
    """
    ...


@overload
def stringify_unsafe_integers(
    data: list[list[dict[str, Any]]],
) -> list[list[dict[str, Any]]]:
    """Type signature used by callers passing grouped row payloads.

    Used by:
    - backend API routes, core workspace and worker services because they need a backend
      boundary that validates inputs before delegating to workspace or worker state.
    """
    ...


def stringify_unsafe_integers(
    data: list[dict[str, Any]] | list[list[dict[str, Any]]],
) -> list[dict[str, Any]] | list[list[dict[str, Any]]]:
    """Convert integers exceeding JavaScript's Number.MAX_SAFE_INTEGER to strings.

    JSON numbers are IEEE 754 doubles in JavaScript, so integers above 2^53-1
    lose precision when parsed by the browser.  Serialising them as strings
    preserves the exact digits for display.

    Accepts both flat (``list[dict]``) and grouped (``list[list[dict]]``)
    row structures.

    Used by:
    - backend API routes, core workspace and worker services because they need a backend
      boundary that validates inputs before delegating to workspace or worker state.
    """
    if not data:
        return data
    result: list[Any] = []
    for item in data:
        if isinstance(item, list):
            result.append(stringify_unsafe_integers(item))
        elif isinstance(item, dict):
            new_row: dict[str, Any] = {}
            for k, v in item.items():
                if isinstance(v, int) and abs(v) > _JS_MAX_SAFE_INTEGER:
                    new_row[k] = str(v)
                else:
                    new_row[k] = v
            result.append(new_row)
        else:
            result.append(item)
    return result
