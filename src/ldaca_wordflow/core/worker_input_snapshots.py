"""Task-owned LazyFrame plan snapshots for worker-backed analysis submissions.

Submit routes use this module to hand workers immutable references to the
requested node plans without materialising corpus rows on the FastAPI event
loop. Workers then load the snapshot and perform any expensive schema, collect,
tokenization, or artifact work out-of-process.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from docworkspace import Node

_SNAPSHOT_ROOT_NAME = "task_inputs"
_SNAPSHOT_FILENAME = "snapshot.json"
_SNAPSHOT_DATA_DIR = "data"


@dataclass(frozen=True)
class SnapshotNode:
    """Node plan and metadata loaded from a task input snapshot.

    Used by worker tasks that need node data but must not receive large Python
    lists from submit routes. The dataclass mirrors the small subset of
    ``docworkspace.Node`` state used by analyses while keeping the serialized
    snapshot format independent from live workspace mutation.
    """

    id: str
    name: str
    data: pl.LazyFrame
    document: str | None
    tokenization: dict[str, Any]

    def to_node(self) -> Node:
        """Return a detached ``Node`` facade for helpers expecting node methods.

        Called by:
        - token-frequency and concordance workers because tokenization helpers
          operate on the public ``Node`` API.
        """

        return Node(
            data=self.data,
            name=self.name,
            workspace=None,
            id=self.id,
            document=self.document,
            tokenization=self.tokenization,
        )


def _node_snapshot_payload(
    node: Any, rel_data_path: Path, fallback_node_id: str
) -> dict[str, Any]:
    """Build JSON metadata for one snapshotted node.

    Called by:
    - ``create_worker_input_snapshot`` while serializing selected nodes for a
      worker task.
    """

    parent_ids = []
    for parent in getattr(node, "parents", []) or []:
        parent_ids.append(getattr(parent, "id", str(parent)))
    return {
        "id": str(getattr(node, "id", fallback_node_id)),
        "name": str(
            getattr(node, "name", None) or getattr(node, "id", fallback_node_id)
        ),
        "document": getattr(node, "document", None),
        "operation": getattr(node, "operation", None),
        "color": getattr(node, "color", None),
        "tokenization": {
            str(source): dict(meta)
            for source, meta in (getattr(node, "tokenization", {}) or {}).items()
        },
        "parents": parent_ids,
        "data_path": rel_data_path.as_posix(),
    }


def create_worker_input_snapshot(
    *,
    workspace_id: str,
    task_id: str,
    node_ids: list[str],
    workspace: Any,
    artifact_dir: str | Path,
) -> Path:
    """Persist selected node LazyFrame plans for a worker task.

    Used by:
    - analysis submit routes before ``WorkerTaskManager.submit_task`` because
      routes need a durable, task-owned handoff that contains no collected
      corpus data.

    Flow:
    1. Serialize only the requested nodes' LazyFrame plans under
       ``data/artifacts/task_inputs/{task_id}``.
    2. Write JSON metadata that workers can load without touching the live
       workspace object.
    """

    resolved_artifact_dir = Path(artifact_dir)

    snapshot_dir = resolved_artifact_dir / _SNAPSHOT_ROOT_NAME / task_id
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    data_dir = snapshot_dir / _SNAPSHOT_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    nodes: dict[str, dict[str, Any]] = {}
    for node_id in node_ids:
        node = workspace.nodes.get(node_id)
        if node is None:
            raise KeyError(f"Node {node_id} not found")
        rel_data_path = Path(_SNAPSHOT_DATA_DIR) / f"{node_id}.plbin"
        node.data.serialize(snapshot_dir / rel_data_path, format="binary")
        nodes[node_id] = _node_snapshot_payload(node, rel_data_path, node_id)

    payload = {
        "version": 1,
        "workspace_id": workspace_id,
        "task_id": task_id,
        "nodes": nodes,
    }
    with (snapshot_dir / _SNAPSHOT_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return snapshot_dir


def load_snapshot_node(snapshot_dir: str | Path, node_id: str) -> SnapshotNode:
    """Load one snapshotted node plan for worker-side analysis preparation.

    Used by:
    - worker tasks after submission so all expensive Polars operations happen in
      the process pool rather than on the API event loop.
    """

    root = Path(snapshot_dir)
    with (root / _SNAPSHOT_FILENAME).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    nodes = payload.get("nodes")
    if not isinstance(nodes, Mapping):
        raise ValueError("Task input snapshot is missing node metadata")
    raw_node = nodes.get(node_id)
    if not isinstance(raw_node, Mapping):
        raise KeyError(f"Node {node_id} is missing from task input snapshot")

    data_path = raw_node.get("data_path")
    if not isinstance(data_path, str):
        raise ValueError(f"Snapshot node {node_id} is missing data_path")
    data = pl.LazyFrame.deserialize(root / data_path, format="binary")
    raw_tokenization = raw_node.get("tokenization") or {}
    tokenization = dict(raw_tokenization) if isinstance(raw_tokenization, Mapping) else {}
    document = raw_node.get("document")
    return SnapshotNode(
        id=str(raw_node.get("id") or node_id),
        name=str(raw_node.get("name") or node_id),
        data=data,
        document=str(document) if isinstance(document, str) else None,
        tokenization=tokenization,
    )


__all__ = [
    "SnapshotNode",
    "create_worker_input_snapshot",
    "load_snapshot_node",
]
