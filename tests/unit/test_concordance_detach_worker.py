from pathlib import Path
import polars as pl
from ldaca_wordflow.analysis.generated_columns import (
    DETACHABLE_CONCORDANCE_COLUMNS,
)
from ldaca_wordflow.workers.concordance import run_concordance_detachment


def test_concordance_detach_task_writes_node_payload_under_workspace_data(
    tmp_path, worker_snapshot
):
    progress_updates: list[tuple[float, str]] = []

    result = run_concordance_detachment(
        workspace_dir=str(tmp_path),
        input_snapshot_dir=str(
            worker_snapshot(
                node_id="11111111-1111-4111-8111-111111111111",
                columns={"document": ["alpha beta", "beta gamma"]},
            )
        ),
        parent_node_id="11111111-1111-4111-8111-111111111111",
        document_column="document",
        search_word="alpha",
        num_left_tokens=1,
        num_right_tokens=1,
        regex=False,
        whole_word=False,
        case_sensitive=False,
        new_node_name="detached_concordance",
        include_document_column=True,
        selected_generated_columns=list(DETACHABLE_CONCORDANCE_COLUMNS),
        progress_callback=lambda progress, message: progress_updates.append(
            (
                progress,
                message,
            )
        ),
    )

    assert result["state"] == "successful"
    payload = result["result"]
    assert payload["parquet_path"].startswith("data/")
    assert "artifacts" not in payload["parquet_path"]

    data_file = tmp_path / Path(payload["parquet_path"])
    assert data_file.exists()

    restored_df = pl.read_parquet(data_file)
    assert restored_df.height >= 1
    # CONC_extraction is opt-in; the default (`include_extraction=False`)
    # call above must NOT include it.
    assert "CONC_extraction" not in restored_df.columns
    assert progress_updates[0][1].startswith("Loading concordance")
    assert any(
        "Preparing text data" in message for _progress, message in progress_updates
    )
    assert progress_updates[-1] == (0.95, "Publishing concordance Data Block...")


def test_concordance_detach_includes_extraction_when_opted_in(
    tmp_path, worker_snapshot
):
    """When `include_extraction=True`, the per-hit detach output keeps the
    `CONC_extraction` raw-window column.
    """
    result = run_concordance_detachment(
        workspace_dir=str(tmp_path),
        input_snapshot_dir=str(
            worker_snapshot(
                node_id="11111111-1111-4111-8111-111111111111",
                columns={"document": ["alpha beta gamma", "beta gamma alpha"]},
            )
        ),
        parent_node_id="11111111-1111-4111-8111-111111111111",
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
        selected_generated_columns=list(DETACHABLE_CONCORDANCE_COLUMNS),
    )
    assert result["state"] == "successful"
    payload = result["result"]
    data_file = tmp_path / Path(payload["parquet_path"])
    restored_df = pl.read_parquet(data_file)
    assert "CONC_extraction" in restored_df.columns
    assert restored_df.schema["CONC_extraction"] == pl.Utf8
    # Sanity check the slice matches what dispersion-detach would have
    # produced for the same hits: "alpha beta" for the first row.
    assert restored_df.get_column("CONC_extraction").to_list()[0] == "alpha beta"
