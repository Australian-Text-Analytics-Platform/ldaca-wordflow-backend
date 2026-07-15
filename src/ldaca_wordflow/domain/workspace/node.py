"""Lazy Polars node and graph relationships for a workspace aggregate."""

from __future__ import annotations

import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Mapping,
    Sequence,
    TypedDict,
    cast,
)

import polars as pl

from .provenance import NodeProvenance, SourceProvenance, referenced_node_ids

if TYPE_CHECKING:  # pragma: no cover
    from .graph import Workspace


class TokenizationMeta(TypedDict):
    """Metadata for one source column's tokenization spec."""

    column_name: str
    model: str
    language: str | None
    params: dict[str, Any]


class Node:
    """One lazy dataset and its explicit lineage inside a workspace.

    ``Workspace`` owns graph registration and persistence. A node contains no
    undo history and exposes no dataframe-operation facade; application
    services construct derived nodes explicitly so every aggregate mutation is
    visible at the use-case boundary.
    """

    @staticmethod
    def _lazyframe_height(data: pl.LazyFrame) -> int:
        collected = data.select(pl.len()).collect()
        return int(collected.item())

    def __init__(
        self,
        data: pl.LazyFrame,
        name: str,
        parents: Sequence["Node"] = (),
        provenance: NodeProvenance | None = None,
        id: str | None = None,
        document: str | None = None,
        color: str | None = None,
        tokenization: Mapping[str, TokenizationMeta] | None = None,
    ) -> None:
        self.id = id or str(uuid.uuid4())
        self.name = name or f"node_{self.id[:8]}"

        if not isinstance(data, pl.LazyFrame):
            raise TypeError(
                "Node data must be a polars LazyFrame "
                f"(received {type(data).__name__})."
            )
        self._data: pl.LazyFrame = data
        self._document_column: str | None = document
        self.color: str | None = color
        self.tokenization = cast(
            dict[str, TokenizationMeta],
            {k: dict(v) for k, v in tokenization.items()} if tokenization else {},
        )
        self.parents: list[Node] = list(parents)
        # Graph attachment is an explicit Workspace operation; construction
        # itself never mutates or implicitly joins an aggregate.
        self.workspace: Workspace | None = None
        self.provenance: NodeProvenance = provenance or SourceProvenance()
        if referenced_node_ids(self.provenance) != [
            parent.id for parent in self.parents
        ]:
            raise ValueError(
                "Node parents must exactly match its ordered provenance references"
            )

    # Commonly accessed convenience properties (explicit to avoid delegation surprises)
    @property
    def shape(self) -> tuple[int, int]:
        height = self._lazyframe_height(self.data)
        return (height, self.data.collect_schema().len())

    @property
    def data(self) -> pl.LazyFrame:
        return self._data

    @data.setter
    def data(self, value: pl.LazyFrame) -> None:
        if not isinstance(value, pl.LazyFrame):
            raise TypeError(
                "Node data must be a polars LazyFrame "
                f"(received {type(value).__name__})."
            )

        self._data = value

    @property
    def children(self) -> list["Node"]:
        if self.workspace is None:
            return []
        return self.workspace.children_of(self.id)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def document(self) -> str | None:
        return self._document_column

    @document.setter
    def document(self, value: str | None) -> None:
        self._document_column = value

    def register_tokenization(self, source_column: str, meta: TokenizationMeta) -> None:
        """Record tokenization metadata keyed by source column."""
        self.tokenization[source_column] = cast(TokenizationMeta, dict(meta))

    def find_tokenization_column(
        self,
        source_column: str,
        *,
        model: str | None = None,
    ) -> str | None:
        """Return the hydrated token column name for ``source_column``."""
        meta = self.tokenization.get(source_column)
        if meta is None:
            return None
        if model is not None and meta.get("model") != model:
            return None
        return meta["column_name"]

    # Representation --------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Node(id={self.id[:8]}, name='{self.name}', dtype={type(self.data).__name__}, "
            f"parents={len(self.parents)}, children={len(self.children)}, document={self.document})"
        )


__all__ = ["Node", "TokenizationMeta"]
