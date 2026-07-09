from pathlib import Path

import polars as pl
from ldaca_wordflow.api.workspaces.analyses.generated_columns import QUOTE_COLUMN_NAMES
from ldaca_wordflow.core.worker_tasks_quotation import run_quotation_detach_task


def test_quotation_detach_task_writes_node_payload_without_internal_source_column(
    tmp_path,
    monkeypatch,
):
    progress_updates: list[tuple[float, str]] = []

    def fake_quotation_groups_via_quote_extractor(
        input_df: pl.DataFrame, source_column: str
    ):
        assert source_column == "__quotation_source__"
        # Mirror the real `quotation_groups_for_dataframe`: it preserves
        # every input column and adds a `quotation` group column. The
        # worker pipeline relies on that contract (e.g. for QUOTE_extraction
        # to flow through), so the mock must match.
        return input_df.with_columns(
            pl.Series(
                "quotation",
                [
                    [
                        {
                            "speaker": "Ada",
                            "speaker_start_idx": 0,
                            "speaker_end_idx": 3,
                            "quote": "Hello",
                            "quote_start_idx": 5,
                            "quote_end_idx": 10,
                            "verb": "said",
                            "verb_start_idx": 11,
                            "verb_end_idx": 15,
                            "quote_type": "direct",
                            "quote_token_count": 1,
                            "is_floating_quote": False,
                            "quote_row_idx": 0,
                        }
                    ]
                ],
            )
        )

    monkeypatch.setattr(
        "ldaca_wordflow.api.workspaces.analyses.quotation_core.quotation_groups_via_quote_extractor",
        fake_quotation_groups_via_quote_extractor,
    )

    result = run_quotation_detach_task(
        configure_worker_environment=lambda: None,
        workspace_dir=str(tmp_path),
        node_corpus=['Ada said "Hello"'],
        parent_node_id="parent-1",
        document_column="document",
        engine_config={},
        new_node_name="detached_quotation",
        include_document_column=True,
        selected_generated_columns=list(QUOTE_COLUMN_NAMES),
        extra_columns_data={"speaker_label": ["narrator"]},
        progress_callback=lambda progress, message: progress_updates.append(
            (
                progress,
                message,
            )
        ),
    )

    assert result["state"] == "successful"
    payload = result["result"]["node_payload"]
    data_file = tmp_path / Path(payload["data_path"])
    assert data_file.exists()

    restored = pl.LazyFrame.deserialize(data_file.open("rb"), format="binary")
    assert restored.collect_schema().names() == [
        "document",
        "speaker_label",
        "QUOTE_speaker",
        "QUOTE_speaker_start_idx",
        "QUOTE_speaker_end_idx",
        "QUOTE_quote",
        "QUOTE_quote_start_idx",
        "QUOTE_quote_end_idx",
        "QUOTE_verb",
        "QUOTE_verb_start_idx",
        "QUOTE_verb_end_idx",
        "QUOTE_quote_type",
        "QUOTE_quote_token_count",
        "QUOTE_is_floating_quote",
        "QUOTE_quote_row_idx",
    ]
    assert "__quotation_source__" not in restored.collect_schema().names()

    assert result["result"]["output_columns"] == [
        "document",
        "speaker_label",
        "QUOTE_speaker",
        "QUOTE_speaker_start_idx",
        "QUOTE_speaker_end_idx",
        "QUOTE_quote",
        "QUOTE_quote_start_idx",
        "QUOTE_quote_end_idx",
        "QUOTE_verb",
        "QUOTE_verb_start_idx",
        "QUOTE_verb_end_idx",
        "QUOTE_quote_type",
        "QUOTE_quote_token_count",
        "QUOTE_is_floating_quote",
        "QUOTE_quote_row_idx",
    ]
    assert progress_updates[0][1].startswith("Loading quotation")
    assert any(
        "Extracting quotations" in message for _progress, message in progress_updates
    )
    assert progress_updates[-1] == (1.0, "Quotation detach completed")
