"""In-memory server-side store for AI annotation preview results.

Used by:
- The ``/annotation/ai/{preview,preview/state,preview/override,annotate-all,
  detach-previewed}`` endpoints in ``api/workspaces/annotation.py``. AI preview
  predictions are non-deterministic and cost real provider spend, so they must be
  remembered on the server instead of recomputed. One shared store lets four flows
  reuse the labels a page already produced:
    * re-viewing a page (cache hit, no second provider call),
    * closing and reopening the tab (the panel re-hydrates from ``preview/state``),
    * ``annotate-all`` (already-previewed rows are reused, only the rest are sent),
    * ``detach-previewed`` (materialises every previewed row, not just the page the
      browser happens to still hold — this is what fixes the "detach only grabbed
      one page" bug).

Why in-memory (deliberate):
- The store lives for the backend process lifetime, mirroring the worker task
  manager. It survives tab switches — the realistic "leave and come back" case —
  but is intentionally cleared on restart to keep the implementation simple and to
  avoid persisting provider output to disk. (A durable sidecar was considered and
  rejected for this iteration.)

Shape:
- Keyed by ``(user_id, workspace_id, node_id)`` -> :class:`PreviewSession`. A
  session records the config ``signature`` its labels belong to, the target
  ``annotation_column`` (metadata for detach/annotate-all), and a per-row map
  ``{row_index: PreviewCell}``. When the prediction-affecting config changes the
  signature changes, so the session's rows are reset because the old predictions
  no longer apply.
- ``effective`` label for a row = the user's ``override`` when one was set, else the
  model's ``ai`` label. That single rule drives what the panel shows on rehydrate,
  what detach materialises, and what annotate-all reuses.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field


def signature_of(
    *,
    text_column: str,
    class_node_id: str,
    class_column: str,
    description_column: str,
    provider_id: str,
    base_url: str | None,
    model: str,
    instruction: str,
    temperature: float,
    reasoning_enabled: bool,
    reasoning_effort: str,
) -> str:
    """Hash the config that determines a row's predicted label.

    Called by the preview and state endpoints (and by annotate-all when deciding
    whether its cache is still valid). Only fields that change the *prediction* go
    in — the text column, class node/columns, provider, base URL, model,
    instruction, and the sampling/reasoning knobs. The target ``annotation_column``
    and pagination are excluded on purpose: they do not affect what the model
    returns for a given text, so keeping them out maximises cache reuse (switching
    the write target or paging never invalidates already-computed labels). A change
    to any included field yields a new signature, which resets the session so stale
    predictions are never shown for a different configuration.
    """
    parts = [
        text_column,
        class_node_id,
        class_column,
        description_column,
        provider_id,
        base_url or "",
        model,
        instruction,
        f"{temperature:.6f}",
        "1" if reasoning_enabled else "0",
        reasoning_effort,
    ]
    joined = "\u0000".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


@dataclass
class PreviewCell:
    """One previewed row's model label and optional user override.

    ``computed`` distinguishes "the model was asked and returned no class"
    (``ai=None, computed=True`` — a genuine cache hit, do not re-ask) from "never
    previewed" (``computed=False``). ``has_override`` similarly distinguishes an
    explicit "None" pick by the user (``override=None, has_override=True``, which
    must win over the model's label) from "no override".
    """

    ai: str | None = None
    computed: bool = False
    override: str | None = None
    has_override: bool = False

    @property
    def effective(self) -> str | None:
        """The label that should actually be applied: override wins when set."""
        return self.override if self.has_override else self.ai


@dataclass
class PreviewSession:
    """All previewed rows for one node under one config signature."""

    signature: str
    annotation_column: str
    rows: dict[int, PreviewCell] = field(default_factory=dict)


@dataclass
class PreviewRowState:
    """One row in the hydration payload returned to the panel on remount."""

    row_index: int
    ai: str | None
    override: str | None
    has_override: bool
    effective: str | None


class AnnotationPreviewStore:
    """Process-wide, lock-guarded map of preview sessions.

    Every public method takes the ``(user_id, workspace_id, node_id)`` identity so
    one user's preview never bleeds into another's. The lock is held only for the
    short dict manipulations, never across an ``await``, so concurrent page
    previews (the panel can fire several while paging) update the same session
    safely without serialising the provider calls themselves.
    """

    def __init__(self) -> None:
        self._sessions: dict[tuple[str, str, str], PreviewSession] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(user_id: str, workspace_id: str, node_id: str) -> tuple[str, str, str]:
        return (user_id, workspace_id, node_id)

    def sync(
        self,
        user_id: str,
        workspace_id: str,
        node_id: str,
        *,
        signature: str,
        annotation_column: str,
    ) -> None:
        """Ensure a session exists for this config; reset its rows if config changed.

        Called at the top of a preview request. If no session exists, or the stored
        one belongs to a different signature (provider/model/prompt/classes/knobs
        changed), a fresh empty session is installed so the previous config's
        predictions cannot leak into the new one. Otherwise the existing rows are
        kept (so paging accumulates) and only the ``annotation_column`` metadata is
        refreshed.
        """
        key = self._key(user_id, workspace_id, node_id)
        with self._lock:
            session = self._sessions.get(key)
            if session is None or session.signature != signature:
                self._sessions[key] = PreviewSession(
                    signature=signature, annotation_column=annotation_column
                )
            else:
                session.annotation_column = annotation_column

    def computed_indices(
        self, user_id: str, workspace_id: str, node_id: str, indices: list[int]
    ) -> set[int]:
        """Return which of ``indices`` already have a model label (cache hits).

        Used by the preview endpoint to decide which page rows still need a provider
        call, and by annotate-all to skip rows it already previewed.
        """
        key = self._key(user_id, workspace_id, node_id)
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                return set()
            return {
                index
                for index in indices
                if index in session.rows and session.rows[index].computed
            }

    def put_ai_labels(
        self,
        user_id: str,
        workspace_id: str,
        node_id: str,
        labels: dict[int, str | None],
    ) -> None:
        """Record freshly computed model labels, preserving any existing override.

        Called after a preview batch returns. Marks each touched row ``computed`` so
        a later view of the same page is a cache hit even when the model chose no
        class (``None``).
        """
        key = self._key(user_id, workspace_id, node_id)
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                return
            for index, label in labels.items():
                cell = session.rows.get(index)
                if cell is None:
                    cell = PreviewCell()
                    session.rows[index] = cell
                cell.ai = label
                cell.computed = True

    def ai_labels_for_page(
        self, user_id: str, workspace_id: str, node_id: str, indices: list[int]
    ) -> list[str | None]:
        """Return the raw *model* labels (not effective) for ``indices``, in order.

        The preview response deliberately returns AI labels rather than effective
        labels: the panel keeps the model's prediction and the user's override as
        separate layers (the override is seeded from ``preview/state``), so mixing
        them here would contaminate the panel's "AI said X" column.
        """
        key = self._key(user_id, workspace_id, node_id)
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                return [None] * len(indices)
            out: list[str | None] = []
            for index in indices:
                cell = session.rows.get(index)
                out.append(cell.ai if (cell is not None and cell.computed) else None)
            return out

    def set_override(
        self, user_id: str, workspace_id: str, node_id: str, row_index: int, label: str | None
    ) -> bool:
        """Persist one manual cell edit onto the current session.

        Called by the override endpoint when the user changes a prediction in the
        dropdown, so the edit survives a tab switch (it comes back via
        ``preview/state``) and is honoured by detach/annotate-all. Operates on the
        node's current session (there is exactly one), so no config needs to travel
        with the edit. Returns False when there is no session to edit — an override
        can only follow a preview, so a missing session means the request is stale.
        """
        key = self._key(user_id, workspace_id, node_id)
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                return False
            cell = session.rows.get(row_index)
            if cell is None:
                cell = PreviewCell()
                session.rows[row_index] = cell
            cell.override = label
            cell.has_override = True
            return True

    def state(
        self,
        user_id: str,
        workspace_id: str,
        node_id: str,
        *,
        signature: str,
    ) -> list[PreviewRowState]:
        """Return every stored row for hydration, but only if the config matches.

        Called by ``preview/state`` when the panel remounts. If the stored session
        belongs to a different signature than the panel's current config, an empty
        list is returned so stale labels for another provider/model/prompt are never
        shown; the panel then previews afresh. Rows are returned in ascending index
        order so the panel can seed its per-position maps directly.
        """
        key = self._key(user_id, workspace_id, node_id)
        with self._lock:
            session = self._sessions.get(key)
            if session is None or session.signature != signature:
                return []
            return [
                PreviewRowState(
                    row_index=index,
                    ai=session.rows[index].ai,
                    override=session.rows[index].override,
                    has_override=session.rows[index].has_override,
                    effective=session.rows[index].effective,
                )
                for index in sorted(session.rows)
            ]

    def effective_rows(
        self,
        user_id: str,
        workspace_id: str,
        node_id: str,
        *,
        signature: str | None = None,
    ) -> dict[int, str | None]:
        """Return ``{row_index: effective_label}`` for every previewed/edited row.

        Used by detach (``signature=None`` — take the node's current session, since
        detach carries no config) and by annotate-all (``signature`` supplied — reuse
        the cache only when it still matches the requested config, otherwise return
        nothing and let annotate-all recompute). A row is included when it was
        computed or explicitly overridden; the value is the override when set, else
        the model label.
        """
        key = self._key(user_id, workspace_id, node_id)
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                return {}
            if signature is not None and session.signature != signature:
                return {}
            return {
                index: cell.effective
                for index, cell in session.rows.items()
                if cell.computed or cell.has_override
            }

    def clear(self, user_id: str, workspace_id: str, node_id: str) -> None:
        """Drop a node's session (e.g. after its column is written by annotate-all)."""
        key = self._key(user_id, workspace_id, node_id)
        with self._lock:
            self._sessions.pop(key, None)


# Module-level singleton shared by every annotation endpoint in this process.
# A single instance is what makes preview/state/override/detach/annotate-all agree
# on the same cached predictions; importing this name everywhere keeps that
# guarantee (there is no per-request construction that could fork the cache).
preview_store = AnnotationPreviewStore()
