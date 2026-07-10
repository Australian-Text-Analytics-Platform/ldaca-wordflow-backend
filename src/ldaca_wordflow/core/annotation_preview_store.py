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
- Keyed by ``(user_id, workspace_id, node_id)`` -> the one current
  :class:`PreviewSession`. Each generation has an opaque ``session_id`` and
  records the config ``signature`` its labels belong to, the target
  ``annotation_column`` (metadata for detach/annotate-all), and a per-row map
  ``{row_index: PreviewCell}``. A prediction-config or target-column change
  installs a fresh empty generation. Every later read/write supplies the expected
  id, so a provider response, override, clear, detach, or annotate-all request from
  a superseded generation cannot affect the current one.
- Materialising actions claim the current generation and copy its effective rows
  under the same lock. While claimed, active preview/session writes are rejected;
  annotate-all consumes the generation on success, while failure and detach release
  it. This seals the snapshot across provider awaits and workspace persistence.
- ``effective`` label for a row = the user's ``override`` when one was set, else the
  model's ``ai`` label. That single rule drives what the panel shows on rehydrate,
  what detach materialises, and what annotate-all reuses.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

from .exceptions import (
    AnnotationPreviewSessionBusyError,
    AnnotationPreviewSessionConflictError,
    InvalidInputError,
)


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
    class_options: Sequence[tuple[str, str]],
    source_revision: str,
) -> str:
    """Hash the config that determines a row's predicted label.

    Called by the preview and state endpoints (and by annotate-all when deciding
    whether its cache is still valid). Only fields that change the *prediction* go
    in — the source node's current ordered-text revision, text column, class
    node/columns and their current ordered label/
    description pairs, provider, base URL, model, instruction, and the
    sampling/reasoning knobs. The target ``annotation_column``
    and pagination are excluded on purpose: they do not affect what the model
    returns for a given text. The target column is still a separate part of the
    session identity and therefore starts a new generation; pagination alone reuses
    the current generation. A change to any included field yields a new signature,
    which also resets the session so stale predictions are never shown for a
    different configuration.
    """
    payload = {
        "text_column": text_column,
        "source_revision": source_revision,
        "class_node_id": class_node_id,
        "class_column": class_column,
        "description_column": description_column,
        "class_options": list(class_options),
        "provider_id": provider_id,
        "base_url": base_url,
        "model": model,
        "instruction": instruction,
        "temperature": f"{temperature:.6f}",
        "reasoning_enabled": reasoning_enabled,
        "reasoning_effort": reasoning_effort,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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

    session_id: str
    signature: str
    annotation_column: str
    rows: dict[int, PreviewCell] = field(default_factory=dict)
    materialization_claimed: bool = False


@dataclass
class PreviewRowState:
    """One row in the hydration payload returned to the panel on remount."""

    row_index: int
    ai: str | None
    override: str | None
    has_override: bool
    effective: str | None


@dataclass
class PreviewSessionState:
    """Current matching session metadata and rows returned for hydration."""

    session_id: str
    annotation_column: str
    rows: list[PreviewRowState]


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
    ) -> str:
        """Return the current generation id, creating one for a changed identity.

        Called at the top of a preview request. If no session exists, or the stored
        one belongs to a different signature or target annotation column, a fresh
        empty session with a new opaque id is installed. Resetting on a target
        change deliberately carries neither model labels nor overrides: this is a
        simple ownership boundary, and it prevents edits made for one output column
        from silently appearing in another. An identical identity reuses the id and
        rows so paging accumulates.
        """
        key = self._key(user_id, workspace_id, node_id)
        with self._lock:
            session = self._sessions.get(key)
            if session is not None:
                self._require_unclaimed(session)
            if (
                session is None
                or session.signature != signature
                or session.annotation_column != annotation_column
            ):
                session = PreviewSession(
                    session_id=str(uuid.uuid4()),
                    signature=signature,
                    annotation_column=annotation_column,
                )
                self._sessions[key] = session
            return session.session_id

    @staticmethod
    def _require_session(
        session: PreviewSession | None,
        expected_session_id: str,
        *,
        signature: str | None = None,
        annotation_column: str | None = None,
    ) -> PreviewSession:
        """Validate a generation while the caller holds the store lock.

        Used by every operation after ``sync``. A single semantic conflict covers
        both an absent session and a superseded id: clients handle both by dropping
        their stale local preview and hydrating/starting the current generation.
        Optional signature and column checks protect materialising operations whose
        request also carries those contracts.
        """

        if (
            session is None
            or session.session_id != expected_session_id
            or (signature is not None and session.signature != signature)
            or (
                annotation_column is not None
                and session.annotation_column != annotation_column
            )
        ):
            raise AnnotationPreviewSessionConflictError(
                "Annotation preview session is missing or no longer current"
            )
        return session

    @staticmethod
    def _require_unclaimed(session: PreviewSession) -> None:
        """Reject active-session work while a materialisation owns its snapshot.

        Called under the store lock by preview sync/read/write, override, clear,
        and claim operations. Annotate-all and non-dry detach seal a generation so
        the exact row snapshot they are about to persist cannot change underneath
        them, including from provider completions already in flight.
        """

        if session.materialization_claimed:
            raise AnnotationPreviewSessionBusyError(
                "Annotation preview session is being materialised"
            )

    def computed_indices(
        self,
        user_id: str,
        workspace_id: str,
        node_id: str,
        expected_session_id: str,
        indices: list[int],
    ) -> set[int]:
        """Return which of ``indices`` already have a model label (cache hits).

        Used by the preview endpoint to decide which page rows still need a provider
        call, and by annotate-all to skip rows it already previewed.
        """
        key = self._key(user_id, workspace_id, node_id)
        with self._lock:
            session = self._require_session(
                self._sessions.get(key), expected_session_id
            )
            self._require_unclaimed(session)
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
        expected_session_id: str,
        labels: dict[int, str | None],
    ) -> None:
        """Record freshly computed model labels, preserving any existing override.

        Called after a preview batch returns. Marks each touched row ``computed`` so
        a later view of the same page is a cache hit even when the model chose no
        class (``None``).
        """
        key = self._key(user_id, workspace_id, node_id)
        with self._lock:
            session = self._require_session(
                self._sessions.get(key), expected_session_id
            )
            self._require_unclaimed(session)
            for index, label in labels.items():
                cell = session.rows.get(index)
                if cell is None:
                    cell = PreviewCell()
                    session.rows[index] = cell
                cell.ai = label
                cell.computed = True

    def ai_labels_for_page(
        self,
        user_id: str,
        workspace_id: str,
        node_id: str,
        expected_session_id: str,
        indices: list[int],
    ) -> list[str | None]:
        """Return the raw *model* labels (not effective) for ``indices``, in order.

        The preview response deliberately returns AI labels rather than effective
        labels: the panel keeps the model's prediction and the user's override as
        separate layers (the override is seeded from ``preview/state``), so mixing
        them here would contaminate the panel's "AI said X" column.
        """
        key = self._key(user_id, workspace_id, node_id)
        with self._lock:
            session = self._require_session(
                self._sessions.get(key), expected_session_id
            )
            self._require_unclaimed(session)
            out: list[str | None] = []
            for index in indices:
                cell = session.rows.get(index)
                out.append(cell.ai if (cell is not None and cell.computed) else None)
            return out

    def set_override(
        self,
        user_id: str,
        workspace_id: str,
        node_id: str,
        expected_session_id: str,
        row_index: int,
        label: str | None,
    ) -> None:
        """Persist one manual cell edit onto the expected session.

        Called by the override endpoint when the user changes a prediction in the
        dropdown, so the edit survives a tab switch (it comes back via
        ``preview/state``) and is honoured by detach/annotate-all. The expected id
        makes a delayed edit from an old panel generation fail with a semantic
        conflict instead of attaching to whichever session happens to be current.
        """
        key = self._key(user_id, workspace_id, node_id)
        with self._lock:
            session = self._require_session(
                self._sessions.get(key), expected_session_id
            )
            self._require_unclaimed(session)
            cell = session.rows.get(row_index)
            if cell is None or not cell.computed:
                raise InvalidInputError(
                    "Cannot override a row that has not been previewed"
                )
            cell.override = label
            cell.has_override = True

    def state(
        self,
        user_id: str,
        workspace_id: str,
        node_id: str,
        *,
        signature: str,
        annotation_column: str,
    ) -> PreviewSessionState | None:
        """Return matching generation metadata and rows for hydration.

        Called by ``preview/state`` when the panel remounts. If the stored session
        belongs to a different prediction signature *or target column*, ``None`` is
        returned so labels and overrides from another exact session identity are
        never shown. Rows are sorted for direct seeding. The matching response also
        carries its opaque id so every subsequent operation can guard ownership.
        """
        key = self._key(user_id, workspace_id, node_id)
        with self._lock:
            session = self._sessions.get(key)
            if (
                session is None
                or session.signature != signature
                or session.annotation_column != annotation_column
            ):
                return None
            return PreviewSessionState(
                session_id=session.session_id,
                annotation_column=session.annotation_column,
                rows=[
                    PreviewRowState(
                        row_index=index,
                        ai=session.rows[index].ai,
                        override=session.rows[index].override,
                        has_override=session.rows[index].has_override,
                        effective=session.rows[index].effective,
                    )
                    for index in sorted(session.rows)
                ],
            )

    def effective_rows(
        self,
        user_id: str,
        workspace_id: str,
        node_id: str,
        expected_session_id: str,
        *,
        signature: str | None,
        annotation_column: str,
    ) -> dict[int, str | None]:
        """Return ``{row_index: effective_label}`` for every previewed/edited row.

        Used by dry-run detach probes, which are observational and therefore do not
        claim the generation. Materialising workflows use ``claim_effective_rows``
        instead. Any supplied-contract mismatch is a conflict, never an empty-cache
        fallback. A row is included when it was computed or explicitly overridden.
        """
        key = self._key(user_id, workspace_id, node_id)
        with self._lock:
            session = self._require_session(
                self._sessions.get(key),
                expected_session_id,
                signature=signature,
                annotation_column=annotation_column,
            )
            return {
                index: cell.effective
                for index, cell in session.rows.items()
                if cell.computed or cell.has_override
            }

    def claim_effective_rows(
        self,
        user_id: str,
        workspace_id: str,
        node_id: str,
        expected_session_id: str,
        *,
        signature: str | None,
        annotation_column: str,
    ) -> dict[int, str | None]:
        """Seal one generation and return its immutable materialisation snapshot.

        Called by annotate-all before its provider await and by non-dry detach
        before building a child node. The lock makes validation, sealing, and row
        copying atomic. While sealed, preview work, overrides, sync/replacement,
        clear, and another claim receive 409 instead of changing the snapshot.
        """

        key = self._key(user_id, workspace_id, node_id)
        with self._lock:
            session = self._require_session(
                self._sessions.get(key),
                expected_session_id,
                signature=signature,
                annotation_column=annotation_column,
            )
            self._require_unclaimed(session)
            session.materialization_claimed = True
            return {
                index: cell.effective
                for index, cell in session.rows.items()
                if cell.computed or cell.has_override
            }

    def release_materialization(
        self,
        user_id: str,
        workspace_id: str,
        node_id: str,
        expected_session_id: str,
    ) -> None:
        """Release a failed or non-consuming materialisation claim.

        Used by annotate-all failure paths and detach's ``finally`` block. A
        mismatched or unclaimed generation is a conflict because it indicates the
        workflow lost ownership rather than a harmless cleanup no-op.
        """

        key = self._key(user_id, workspace_id, node_id)
        with self._lock:
            session = self._require_session(
                self._sessions.get(key), expected_session_id
            )
            if not session.materialization_claimed:
                raise AnnotationPreviewSessionConflictError(
                    "Annotation preview session is not being materialised"
                )
            session.materialization_claimed = False

    def complete_materialization(
        self,
        user_id: str,
        workspace_id: str,
        node_id: str,
        expected_session_id: str,
    ) -> None:
        """Consume a claimed generation after annotate-all persists its column."""

        key = self._key(user_id, workspace_id, node_id)
        with self._lock:
            session = self._require_session(
                self._sessions.get(key), expected_session_id
            )
            if not session.materialization_claimed:
                raise AnnotationPreviewSessionConflictError(
                    "Annotation preview session is not being materialised"
                )
            del self._sessions[key]

    def clear(
        self,
        user_id: str,
        workspace_id: str,
        node_id: str,
        expected_session_id: str,
    ) -> None:
        """Drop only the expected generation.

        Called by explicit close. Annotate-all uses ``complete_materialization`` so
        only its sealed generation is consumed. A stale close must not delete a
        newer or actively materialising generation for the same node.
        """
        key = self._key(user_id, workspace_id, node_id)
        with self._lock:
            session = self._require_session(
                self._sessions.get(key), expected_session_id
            )
            self._require_unclaimed(session)
            del self._sessions[key]


# Module-level singleton shared by every annotation endpoint in this process.
# A single instance is what makes preview/state/override/detach/annotate-all agree
# on the same cached predictions; importing this name everywhere keeps that
# guarantee (there is no per-request construction that could fork the cache).
preview_store = AnnotationPreviewStore()
