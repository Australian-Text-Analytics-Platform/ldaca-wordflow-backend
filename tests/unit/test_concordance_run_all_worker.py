import polars as pl
from ldaca_wordflow.analysis.generated_columns import (
    DETACHABLE_CONCORDANCE_COLUMNS,
)
from ldaca_wordflow.workers.concordance import run_concordance_run_all


def test_concordance_run_all_writes_complete_analysis_table_artifact(
    tmp_path, worker_snapshot
):
    progress_updates: list[tuple[float, str]] = []

    result = run_concordance_run_all(
        artifact_dir=str(tmp_path),
        input_snapshot_dir=str(
            worker_snapshot(
                node_id="11111111-1111-4111-8111-111111111111",
                columns={
                    "document": ["alpha beta", "beta gamma"],
                    "metadata": ["A", "B"],
                },
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
        progress_callback=lambda progress, message: progress_updates.append(
            (
                progress,
                message,
            )
        ),
    )

    assert result["state"] == "successful"
    source = result["source"]
    assert source["node_id"] == "11111111-1111-4111-8111-111111111111"
    assert source["document_column"] == "document"
    assert source["metadata_columns"] == ["metadata"]
    assert source["analysis_columns"] == [
        *DETACHABLE_CONCORDANCE_COLUMNS,
        "CONC_extraction",
    ]
    assert source["table"]["table_id"] == "concordance-run-all"
    assert "data_block" not in source

    data_file = tmp_path / source["table"]["artifact"]
    assert data_file.exists()

    restored_df = pl.read_parquet(data_file)
    assert restored_df.height >= 1
    assert restored_df["metadata"].to_list() == ["A"]
    assert set(DETACHABLE_CONCORDANCE_COLUMNS).issubset(restored_df.columns)
    assert "CONC_extraction" in restored_df.columns
    assert "__wordflow_source_row_id" in restored_df.columns
    assert progress_updates[0][1].startswith("Loading concordance")
    assert any(
        "Preparing text data" in message for _progress, message in progress_updates
    )
    assert progress_updates[-1] == (0.95, "Saving concordance Result...")


def test_concordance_run_all_retains_extraction_in_canonical_result(
    tmp_path, worker_snapshot
):
    result = run_concordance_run_all(
        artifact_dir=str(tmp_path),
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
    )
    assert result["state"] == "successful"
    source = result["source"]
    data_file = tmp_path / source["table"]["artifact"]
    restored_df = pl.read_parquet(data_file)
    assert "CONC_extraction" in restored_df.columns
    assert restored_df.schema["CONC_extraction"] == pl.Utf8
    # Sanity check the slice matches what Run All would have
    # produced for the same hits: "alpha beta" for the first row.
    assert restored_df.get_column("CONC_extraction").to_list()[0] == "alpha beta"
