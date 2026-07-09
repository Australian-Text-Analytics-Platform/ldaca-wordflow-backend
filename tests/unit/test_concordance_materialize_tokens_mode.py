"""Materialize honours search_mode end-to-end.

Before this regression test, ``run_concordance_materialize_task`` only
walked raw text through the regex engine — so a tokens-mode CJK search
that found ~750 live page hits collapsed to ~4 byte-coincidence regex
hits after the user clicked Process All. The fix:

- ``ConcordanceMaterializeRequest.search_mode`` accepted by the route.
- Route extracts the tokenization column when ``search_mode="tokens"``.
- Worker dispatches ``_build_tokens_concordance_occurrence_dataframe``
  which walks tokens for exact-token matches, then the existing
  L1/R1 freq join applies unchanged.

These tests assert column-shape parity with the regex builder so
downstream consumers (paginated reads, detach, dispersion bin fetches)
don't have to branch on the parquet's origin.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

import polars as pl
import pytest
from ldaca_wordflow.api.workspaces.analyses import concordance as concordance_api
from ldaca_wordflow.api.workspaces import utils as workspace_utils
from ldaca_wordflow.api.workspaces.analyses.generated_columns import (
    CONC_END_IDX_COLUMN,
    CONC_EXTRACTION_COLUMN,
    CONC_L1_COLUMN,
    CONC_LEFT_CONTEXT_COLUMN,
    CONC_MATCHED_TEXT_COLUMN,
    CONC_R1_COLUMN,
    CONC_RIGHT_CONTEXT_COLUMN,
    CONC_START_IDX_COLUMN,
    CORE_CONCORDANCE_COLUMNS,
)
from ldaca_wordflow.core.worker_tasks_concordance import (
    _build_concordance_occurrence_dataframe,
    _build_tokens_concordance_occurrence_dataframe,
)
from ldaca_wordflow.models.concordance import ConcordanceMaterializeRequest

from docworkspace import Node

_ZH_TOKENS_DOC_A = [
    {"token": "今天", "start": 0, "end": 2},
    {"token": "天气", "start": 2, "end": 4},
    {"token": "很", "start": 4, "end": 5},
    {"token": "好", "start": 5, "end": 6},
    {"token": "今天", "start": 6, "end": 8},
    {"token": "我们", "start": 8, "end": 10},
    {"token": "出去", "start": 10, "end": 12},
    {"token": "玩", "start": 12, "end": 13},
]
_ZH_DOC_A = "今天天气很好今天我们出去玩"
_ZH_TOKENS_DOC_B = [
    {"token": "晚上", "start": 0, "end": 2},
    {"token": "今天", "start": 2, "end": 4},
    {"token": "见面", "start": 4, "end": 6},
]
_ZH_DOC_B = "晚上今天见面"


def test_tokens_builder_emits_same_core_columns_as_regex_builder() -> None:
    """The tokens-mode builder produces the exact column list the regex
    builder produces, in the same order — minus L1/R1 freq, which the
    caller (``run_concordance_materialize_task``) appends after a
    ``group_by`` join.
    """
    extras = {"doc_id": ["a", "b"]}
    extras_dtypes = {"doc_id": pl.Utf8}

    tokens_df, tokens_cols = _build_tokens_concordance_occurrence_dataframe(
        node_corpus=[_ZH_DOC_A, _ZH_DOC_B],
        node_tokens=[_ZH_TOKENS_DOC_A, _ZH_TOKENS_DOC_B],
        document_column="text",
        search_word="今天",
        num_left_tokens=2,
        num_right_tokens=2,
        case_sensitive=False,
        include_document_column=True,
        extra_columns_data=extras,
        extra_columns_dtypes=extras_dtypes,
    )

    # Expected order: [document_column, *extras, *CORE_CONCORDANCE_COLUMNS,
    # CONC_extraction] — identical to regex builder.
    assert tokens_cols == [
        "text",
        "doc_id",
        *CORE_CONCORDANCE_COLUMNS,
        CONC_EXTRACTION_COLUMN,
    ]
    assert list(tokens_df.columns) == tokens_cols


def test_tokens_builder_finds_exact_token_matches() -> None:
    """Doc A has two 今天 tokens, Doc B has one — so 3 rows total."""
    df, _cols = _build_tokens_concordance_occurrence_dataframe(
        node_corpus=[_ZH_DOC_A, _ZH_DOC_B],
        node_tokens=[_ZH_TOKENS_DOC_A, _ZH_TOKENS_DOC_B],
        document_column="text",
        search_word="今天",
        num_left_tokens=2,
        num_right_tokens=2,
        case_sensitive=False,
        include_document_column=True,
        extra_columns_data=None,
        extra_columns_dtypes=None,
    )
    assert df.height == 3
    assert df.get_column(CONC_MATCHED_TEXT_COLUMN).to_list() == [
        "今天",
        "今天",
        "今天",
    ]


def test_tokens_builder_handles_empty_corpus_gracefully() -> None:
    """No matches → empty df with correct schema so the downstream
    group_by join doesn't choke.
    """
    df, cols = _build_tokens_concordance_occurrence_dataframe(
        node_corpus=[_ZH_DOC_A],
        node_tokens=[_ZH_TOKENS_DOC_A],
        document_column="text",
        search_word="不存在",  # not in any token list
        num_left_tokens=2,
        num_right_tokens=2,
        case_sensitive=False,
        include_document_column=True,
        extra_columns_data=None,
        extra_columns_dtypes=None,
    )
    assert df.height == 0
    assert "text" in df.columns
    for required in (
        CONC_LEFT_CONTEXT_COLUMN,
        CONC_MATCHED_TEXT_COLUMN,
        CONC_RIGHT_CONTEXT_COLUMN,
        CONC_START_IDX_COLUMN,
        CONC_END_IDX_COLUMN,
        CONC_L1_COLUMN,
        CONC_R1_COLUMN,
        CONC_EXTRACTION_COLUMN,
    ):
        assert required in df.columns
    assert cols[-1] == CONC_EXTRACTION_COLUMN


def test_regex_and_tokens_builders_agree_on_english_word_boundary_case() -> None:
    """Sanity check: when tokenisation matches whitespace boundaries
    exactly (i.e. English), tokens-mode and regex-mode whole-word
    searches return the same number of hits. Guards against accidental
    drift between the two builders' counting semantics.
    """
    en_text_a = "the quick brown fox the lazy dog"
    en_text_b = "the cat sat the mat"

    def _tokens(text: str) -> list[dict]:
        out = []
        cursor = 0
        for tok in text.split():
            start = text.find(tok, cursor)
            out.append({"token": tok, "start": start, "end": start + len(tok)})
            cursor = start + len(tok)
        return out

    regex_df, _ = _build_concordance_occurrence_dataframe(
        node_corpus=[en_text_a, en_text_b],
        document_column="text",
        search_word="the",
        num_left_tokens=2,
        num_right_tokens=2,
        regex=False,
        whole_word=True,
        case_sensitive=False,
        include_document_column=True,
        extra_columns_data=None,
        extra_columns_dtypes=None,
    )
    tokens_df, _ = _build_tokens_concordance_occurrence_dataframe(
        node_corpus=[en_text_a, en_text_b],
        node_tokens=[_tokens(en_text_a), _tokens(en_text_b)],
        document_column="text",
        search_word="the",
        num_left_tokens=2,
        num_right_tokens=2,
        case_sensitive=False,
        include_document_column=True,
        extra_columns_data=None,
        extra_columns_dtypes=None,
    )

    # Doc A has 2 "the"s, Doc B has 2 "the"s → 4 total. Both engines
    # must agree on this for English where tokenisation = whitespace
    # splitting.
    assert regex_df.height == tokens_df.height == 4


@pytest.mark.asyncio
async def test_tokens_materialize_route_selects_tokenization_column_once(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenization_column = "tokenization.text.huggingface:bert-base-uncased"
    node = Node(
        data=pl.DataFrame(
            {
                "text": ["hello world", "hello again"],
                "speaker": ["A", "B"],
            }
        ).lazy(),
        name="tokens_probe",
    )
    node.register_tokenization(
        "text",
        {
            "column_name": tokenization_column,
            "model": "huggingface:bert-base-uncased",
            "language": "en",
            "params": {"lowercase": True, "remove_punct": True},
        },
    )

    captured_task_args: dict[str, Any] = {}
    workspace_id = UUID("00000000-0000-0000-0000-000000000001")

    class TaskManager:
        async def submit_task(self, **kwargs: Any):
            captured_task_args.update(cast(dict[str, Any], kwargs["task_args"]))
            return SimpleNamespace(id=kwargs["task_id"])

    class LinkManager:
        def link_child_task(self, _parent_task_id: str, _child_task_id: str) -> None:
            return None

    class Workspace:
        nodes = {node.id: node}

    class WorkspaceManager:
        def get_current_workspace_id(self, _user_id: str) -> str:
            return str(workspace_id)

        def get_current_workspace(self, _user_id: str) -> Workspace:
            return Workspace()

        def load_workspace(self, _user_id: str, _workspace_id: str) -> Workspace:
            return Workspace()

        def get_task_manager(self, _user_id: str) -> TaskManager:
            return TaskManager()

        def get_workspace_dir(self, _user_id: str, _workspace_id: str):
            return tmp_path

        def ensure_workspace_artifacts_dir(self, _user_id: str, _workspace_id: str):
            artifact_dir = tmp_path / "artifacts"
            artifact_dir.mkdir(exist_ok=True)
            return artifact_dir

    workspace_manager = WorkspaceManager()
    monkeypatch.setattr(concordance_api, "workspace_manager", workspace_manager)
    monkeypatch.setattr(workspace_utils, "workspace_manager", workspace_manager)
    monkeypatch.setattr(
        concordance_api, "get_task_manager", lambda _user_id: LinkManager()
    )

    response = await concordance_api.materialize_concordance(
        workspace_id,
        node.id,
        ConcordanceMaterializeRequest(
            node_id=node.id,
            column="text",
            search_word="hello",
            num_left_tokens=1,
            num_right_tokens=1,
            search_mode="tokens",
        ),
        current_user={"id": "user"},
        parent_task_id="parent-task",
    )

    assert response["state"] == "running"
    assert captured_task_args["node_tokens"] is None
    assert captured_task_args["extra_columns_data"] is None
    assert captured_task_args["input_snapshot_dir"]
