"""Workspace aggregate owning node registration, ordering, and lineage."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import polars_text  # noqa: F401  (register text namespace side-effects)

from .node import Node
from .provenance import NodeProvenance, compose_provenance, referenced_node_ids
from .tab import Tab
from .analysis import AnalysisRecord, AnalysisState, analysis_input_ids


class Workspace:
    """Core workspace managing a collection of Nodes and their relationships."""

    def __init__(
        self,
        *,
        name: str | None = None,
        workspace_id: str | None = None,
        created_at: str | None = None,
        modified_at: str | None = None,
    ) -> None:
        now = datetime.now(UTC).isoformat()

        self.id = workspace_id or str(uuid.uuid4())
        self.name = name or f"workspace_{self.id[:8]}"
        self.nodes: dict[str, Node] = {}
        self.tabs: dict[str, Tab] = {}
        self.analyses: dict[str, AnalysisRecord] = {}
        self._corrupt_analysis_records: dict[str, bytes] = {}
        self._children_by_parent: dict[str, list[Node]] = {}
        self.description: str = ""
        self.created_at: str = created_at or now
        self.modified_at: str = modified_at or now

    # Node management -------------------------------------------------

    def add_node(self, node: Node) -> Node:
        if node.id in self.nodes:
            raise ValueError(f"Workspace already contains node {node.id}")
        if any(parent is node for parent in node.parents):
            raise ValueError("A node cannot be its own parent")
        if len({parent.id for parent in node.parents}) != len(node.parents):
            raise ValueError("A node cannot contain duplicate parents")
        if any(
            parent.workspace is not self or self.nodes.get(parent.id) is not parent
            for parent in node.parents
        ):
            raise ValueError("Node parents must already belong to this workspace")
        if referenced_node_ids(node.provenance) != [
            parent.id for parent in node.parents
        ]:
            raise ValueError("Node parents do not match its provenance")
        source_workspace = node.workspace
        if source_workspace is not None and source_workspace is not self:
            raise ValueError("Node already belongs to another workspace")
        self.nodes[node.id] = node
        self._children_by_parent[node.id] = []
        for parent in node.parents:
            self._children_by_parent[parent.id].append(node)
        node.workspace = self
        return node

    def children_of(self, node_id: str) -> list[Node]:
        """Return the aggregate-owned children of one registered node."""

        return list(self._children_by_parent[node_id])

    def reorder_nodes(self, ordered_ids: list[str]) -> list[str]:
        """Apply an exact duplicate-free permutation of all workspace nodes."""

        if len(ordered_ids) != len(set(ordered_ids)) or set(ordered_ids) != set(
            self.nodes
        ):
            raise ValueError("Node order must be an exact duplicate-free permutation")
        self.nodes = {node_id: self.nodes[node_id] for node_id in ordered_ids}
        return list(self.nodes.keys())

    # Tab management --------------------------------------------------

    def add_tab(self, tab: Tab) -> Tab:
        tab_id = str(tab.id)
        if tab_id in self.tabs:
            raise ValueError(f"Workspace already contains tab {tab_id}")
        if tab.analysis_id is not None and any(
            existing.analysis_id == tab.analysis_id for existing in self.tabs.values()
        ):
            raise ValueError("A root analysis may belong to only one tab")
        self.tabs[tab_id] = tab
        return tab

    def remove_tab(self, tab_id: str) -> Tab | None:
        return self.tabs.pop(tab_id, None)

    # Analysis management --------------------------------------------

    def add_analysis(self, analysis: AnalysisRecord) -> AnalysisRecord:
        analysis_id = str(analysis.id)
        if (
            analysis_id in self.analyses
            or analysis_id in self._corrupt_analysis_records
        ):
            raise ValueError(f"Workspace already contains analysis {analysis_id}")
        if analysis.parent_analysis_id is not None:
            parent = self.analyses.get(str(analysis.parent_analysis_id))
            if parent is None or parent.parent_analysis_id is not None:
                raise ValueError("A child Analysis requires an existing root parent")
        self.analyses[analysis_id] = analysis
        return analysis

    def add_corrupt_analysis(self, analysis_id: str, content: bytes) -> None:
        canonical_id = str(uuid.UUID(analysis_id))
        if canonical_id != analysis_id:
            raise ValueError("Corrupt Analysis storage identity is invalid")
        if (
            analysis_id in self.analyses
            or analysis_id in self._corrupt_analysis_records
        ):
            raise ValueError(f"Workspace already contains analysis {analysis_id}")
        self._corrupt_analysis_records[analysis_id] = content

    @property
    def corrupt_analysis_ids(self) -> set[str]:
        return set(self._corrupt_analysis_records)

    def corrupt_analysis_bytes(self, analysis_id: str) -> bytes:
        return self._corrupt_analysis_records[analysis_id]

    def remove_analysis(self, analysis_id: str) -> AnalysisRecord | bytes | None:
        analysis = self.analyses.pop(analysis_id, None)
        if analysis is not None:
            if analysis.parent_analysis_id is None:
                children = [
                    child_id
                    for child_id, child in self.analyses.items()
                    if str(child.parent_analysis_id) == analysis_id
                ]
                for child_id in children:
                    self.analyses.pop(child_id)
            return analysis
        return self._corrupt_analysis_records.pop(analysis_id, None)

    def replace_analysis(self, analysis: AnalysisRecord) -> AnalysisRecord:
        """Replace one valid lifecycle record without changing its identity."""

        analysis_id = str(analysis.id)
        if analysis_id not in self.analyses:
            raise ValueError("Analysis does not exist")
        previous = self.analyses[analysis_id]
        if previous.parent_analysis_id != analysis.parent_analysis_id:
            raise ValueError("Analysis ownership cannot change")
        if previous.request != analysis.request:
            raise ValueError("Analysis request cannot change")
        self.analyses[analysis_id] = analysis
        return analysis

    def analysis_children(self, root_analysis_id: str) -> list[AnalysisRecord]:
        return [
            analysis
            for analysis in self.analyses.values()
            if str(analysis.parent_analysis_id) == root_analysis_id
        ]

    def live_analysis_ids(self) -> set[str]:
        """Return Analyses reachable through the current Tab collection.

        Detached records remain aggregate-owned while cancellation and cleanup
        finish, but they are not addressable through the public Analysis API.
        """

        root_ids = {
            str(tab.analysis_id)
            for tab in self.tabs.values()
            if tab.analysis_id is not None
        }
        child_ids = {
            analysis_id
            for analysis_id, analysis in self.analyses.items()
            if analysis.parent_analysis_id is not None
            and str(analysis.parent_analysis_id) in root_ids
        }
        return root_ids | child_ids

    def analysis_tab_id(self, root_analysis_id: str) -> str | None:
        """Return the sole Tab currently referencing a root Analysis."""

        for tab_id, tab in self.tabs.items():
            if str(tab.analysis_id) == root_analysis_id:
                return tab_id
        return None

    def reserved_node_ids(self) -> set[str]:
        """Derive active shared input reservations from durable Analysis state."""

        reserved: set[str] = set()
        for analysis in self.analyses.values():
            if analysis.state in {AnalysisState.QUEUED, AnalysisState.RUNNING}:
                reserved.update(
                    str(node_id) for node_id in analysis_input_ids(analysis.request)
                )
        return reserved

    def place_node_after_parent(self, node: Node) -> None:
        """Move ``node`` to sit immediately below its first in-workspace parent.

        Used by:
        - Backend node-creation helpers (``_create_and_persist_child_node``,
          clone, detach, and analysis result writers) because a freshly derived
          node should appear right under its mother node in the list view
          instead of being appended to the end.
        Why:
        - ``add_node`` deliberately appends so that loading a saved workspace
          preserves the persisted order; smart insertion is therefore an
          explicit post-creation step rather than ``add_node`` behavior.
        Flow:
        1. Resolve the node and bail out when it is unknown to this workspace.
        2. Find the first parent that also lives in this workspace (root/import
           nodes have none, so they keep their appended position).
        3. Reinsert the node directly after that parent via ``reorder_nodes``.
        """
        if self.nodes.get(node.id) is not node:
            raise ValueError("Node must belong to this workspace")
        if not node.parents:
            return
        parent_id = node.parents[0].id
        order = [nid for nid in self.nodes if nid != node.id]
        idx = order.index(parent_id)
        order.insert(idx + 1, node.id)
        self.reorder_nodes(order)

    def node_removal_affected_ids(self, node_id: str) -> set[str]:
        """Return the Data Blocks whose state deletion would rewrite."""

        if node_id not in self.nodes:
            return set()
        return {node_id, *(child.id for child in self.children_of(node_id))}

    def remove_node(self, node_id: str) -> bool:
        if node_id not in self.nodes:
            return False
        node = self.nodes[node_id]
        child_nodes = self.children_of(node_id)
        survivors = {key: value for key, value in self.nodes.items() if key != node_id}
        replacements: dict[str, tuple[NodeProvenance, list[Node]]] = {}

        # Validate every composed lineage before mutating the aggregate.
        for child in child_nodes:
            provenance = compose_provenance(
                child.provenance,
                removed_node_id=node_id,
                replacement=node.provenance,
            )
            parent_ids = referenced_node_ids(provenance)
            if child.id in parent_ids or any(
                parent_id not in survivors for parent_id in parent_ids
            ):
                raise ValueError("Deleting the node would create invalid provenance")
            replacements[child.id] = (
                provenance,
                [survivors[parent_id] for parent_id in parent_ids],
            )

        for child in child_nodes:
            provenance, parents = replacements[child.id]
            child.provenance = provenance
            child.parents = parents

        node.parents = []
        node.workspace = None
        del self.nodes[node_id]
        self._children_by_parent = {key: [] for key in self.nodes}
        for child in self.nodes.values():
            for parent in child.parents:
                self._children_by_parent[parent.id].append(child)

        # The persisted plan remains part of the previous committed snapshot
        # until ``WorkspaceStore.commit`` atomically publishes metadata without
        # this node. Post-commit reconciliation then removes the old plan.
        return True

    # Dunder ----------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Workspace(id={self.id[:8]}, name='{self.name}', nodes={len(self.nodes)})"
        )


__all__ = ["Workspace"]
