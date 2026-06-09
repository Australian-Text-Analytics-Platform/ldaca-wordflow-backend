from pathlib import Path
from typing import cast

import polars as pl
from ldaca_wordflow.core import worker
from ldaca_wordflow.core.worker_tasks_concordance import run_concordance_detach_task


def test_concordance_detach_task_forwards_extra_columns_data(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_concordance_detach_task(
        configure_worker_environment,
        workspace_dir,
        node_corpus,
        parent_node_id,
        document_column,
        search_word,
        num_left_tokens,
        num_right_tokens,
        regex,
        whole_word,
        case_sensitive,
        new_node_name,
        include_document_column=False,
        include_extraction=False,
        selected_generated_columns=None,
        extra_columns_data=None,
        extra_columns_dtypes=None,
        materialized_path=None,
        language=None,
        progress_callback=None,
    ):
        captured["include_document_column"] = include_document_column
        captured["include_extraction"] = include_extraction
        captured["selected_generated_columns"] = selected_generated_columns
        captured["extra_columns_data"] = extra_columns_data
        captured["whole_word"] = whole_word
        return {"state": "successful"}

    monkeypatch.setattr(
        worker, "run_concordance_detach_task", fake_run_concordance_detach_task
    )

    result = worker.concordance_detach_task(
        user_id="user-1",
        workspace_id="ws-1",
        workspace_dir="/tmp/workspace",
        node_corpus=["alpha beta"],
        parent_node_id="node-1",
        document_column="document",
        search_word="alpha",
        num_left_tokens=2,
        num_right_tokens=2,
        regex=False,
        whole_word=True,
        case_sensitive=False,
        new_node_name="node_1_conc",
        include_document_column=True,
        extra_columns_data={"source": ["a"]},
    )

    assert result == {"state": "successful"}
    assert captured["include_document_column"] is True
    assert captured["extra_columns_data"] == {"source": ["a"]}
    assert captured["whole_word"] is True


def test_concordance_detach_task_writes_node_payload_under_workspace_data(tmp_path):
    progress_updates: list[tuple[float, str]] = []

    result = run_concordance_detach_task(
        configure_worker_environment=lambda: None,
        workspace_dir=str(tmp_path),
        node_corpus=["alpha beta", "beta gamma"],
        parent_node_id="parent-1",
        document_column="document",
        search_word="alpha",
        num_left_tokens=1,
        num_right_tokens=1,
        regex=False,
        whole_word=False,
        case_sensitive=False,
        new_node_name="detached_concordance",
        include_document_column=True,
        progress_callback=lambda progress, message: progress_updates.append(
            (
                progress,
                message,
            )
        ),
    )

    assert result["state"] == "successful"
    payload = result["result"]["node_payload"]
    assert payload["data_path"].startswith("data/")
    assert "artifacts" not in payload["data_path"]

    data_file = tmp_path / Path(payload["data_path"])
    assert data_file.exists()

    restored = pl.LazyFrame.deserialize(data_file.open("rb"), format="binary")
    restored_df = cast(pl.DataFrame, restored.collect())
    assert restored_df.height >= 1
    # CONC_extraction is opt-in; the default (`include_extraction=False`)
    # call above must NOT include it.
    assert "CONC_extraction" not in restored_df.columns
    assert progress_updates[0][1].startswith("Loading concordance")
    assert any(
        "Preparing text data" in message for _progress, message in progress_updates
    )
    assert progress_updates[-1] == (1.0, "Concordance detach completed")


def test_concordance_detach_includes_extraction_when_opted_in(tmp_path):
    """When `include_extraction=True`, the per-hit detach output keeps the
    `CONC_extraction` raw-window column. Default keeps the existing
    backward-compatible "exclude" behaviour.
    """
    result = run_concordance_detach_task(
        configure_worker_environment=lambda: None,
        workspace_dir=str(tmp_path),
        node_corpus=["alpha beta gamma", "beta gamma alpha"],
        parent_node_id="parent-1",
        document_column="document",
        search_word="alpha",
        num_left_tokens=1,
        num_right_tokens=1,
        regex=False,
        whole_word=False,
        case_sensitive=False,
        new_node_name="detached_with_extract",
        include_document_column=True,
        include_extraction=True,
    )
    assert result["state"] == "successful"
    payload = result["result"]["node_payload"]
    data_file = tmp_path / Path(payload["data_path"])
    restored_df = cast(
        pl.DataFrame,
        pl.LazyFrame.deserialize(data_file.open("rb"), format="binary").collect(),
    )
    assert "CONC_extraction" in restored_df.columns
    assert restored_df.schema["CONC_extraction"] == pl.Utf8
    # Sanity check the slice matches what dispersion-detach would have
    # produced for the same hits: "alpha beta" for the first row.
    assert restored_df.get_column("CONC_extraction").to_list()[0] == "alpha beta"
