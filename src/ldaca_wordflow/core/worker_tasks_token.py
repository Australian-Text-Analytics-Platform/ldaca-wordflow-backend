"""Token-frequency worker task implementation.

Separated from `worker.py` to keep the worker module focused and smaller.

Used by:
- Backend API routes, worker tasks, workspace services, and backend tests because they
  need a backend boundary that validates inputs before delegating to workspace or worker
  state.

Flow: resolve tokenization preferences, hydrate or create token columns, aggregate
    frequencies, and persist derived artifacts for result queries.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, cast

from .analysis_helpers import sanitize_stop_words
from .worker_utils import worker_task

logger = logging.getLogger(__name__)

_PLAIN_WORDS_EN_MODEL = "native:plain_words_en"


@worker_task
def run_token_frequencies_task(
    configure_worker_environment,
    user_id: str,
    workspace_id: str,
    node_corpora: dict[str, list[str]],
    node_display_names: dict[str, str],
    artifact_dir: str,
    artifact_prefix: str,
    token_limit: int = 10,
    stop_words: list[str] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    node_token_streams: dict[str, str] | None = None,
    tokenizer_model: str | None = None,
    node_tokenizer_models: dict[str, str] | None = None,
    input_snapshot_dir: str | None = None,
    node_ids: list[str] | None = None,
    node_columns: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Execute token-frequency analysis inside a worker process.

    Used by:
    - `core.worker.token_frequencies_task` because background jobs need one lifecycle owner
      for submission, progress, cancellation, and artifact cleanup.
    - `TASK_REGISTRY["token_frequencies"]` because background jobs need one lifecycle owner
      for submission, progress, cancellation, and artifact cleanup.
    Why:
        - Computes token frequencies off the API thread and writes Parquet artifacts
            for main-process lazy retrieval.

    Refactor note:
    - If wrapper indirection is removed, this function can be imported directly
      into `TASK_REGISTRY`.

    Flow: resolve tokenization preferences, hydrate or create token columns, aggregate
        frequencies, and persist derived artifacts for result queries.
    """
    configure_worker_environment()

    try:
        if progress_callback:
            progress_callback(0.02, "Loading token frequency resources...")

        import polars as pl
        import polars_text as pt

        logger.info("Starting token frequencies task for workspace %s", workspace_id)

        artifact_root = Path(artifact_dir)
        artifact_root.mkdir(parents=True, exist_ok=True)

        if progress_callback:
            progress_callback(0.1, "Validating payload...")

        if progress_callback:
            progress_callback(0.2, "Preparing text data...")

        requested_stop_words = sanitize_stop_words(stop_words)
        effective_limit = int(token_limit) if int(token_limit) > 0 else 25

        DEFAULT_TOKEN_LIMIT = 25
        SERVER_LIMIT_MULTIPLIER = 5
        MAX_SERVER_TOKEN_LIMIT = 5000
        server_limit = min(
            max(effective_limit * SERVER_LIMIT_MULTIPLIER, DEFAULT_TOKEN_LIMIT),
            MAX_SERVER_TOKEN_LIMIT,
        )

        token_streams = dict(node_token_streams or {})
        corpora = dict(node_corpora or {})
        display_names = dict(node_display_names or {})
        fallback_tokenizer_model = (tokenizer_model or "").strip() or None
        requested_node_tokenizer_models = {
            node_id: model.strip()
            for node_id, model in (node_tokenizer_models or {}).items()
            if model and model.strip()
        }

        if input_snapshot_dir is not None:
            if progress_callback:
                progress_callback(0.12, "Preparing token frequency inputs...")
            from .worker_input_snapshots import load_snapshot_node
            from .tokens_cache import hydrate_tokenization_lazyframe

            if not node_ids:
                raise ValueError("Token frequency snapshot input requires node_ids")
            if not node_columns:
                raise ValueError("Token frequency snapshot input requires node_columns")
            for node_id in node_ids:
                snapshot_node = load_snapshot_node(input_snapshot_dir, node_id)
                source_column = node_columns.get(node_id)
                if not source_column:
                    raise ValueError(f"Missing token-frequency column for node {node_id}")
                display_names[node_id] = snapshot_node.name
                node = snapshot_node.to_node()

                def collect_source_corpus() -> list[str]:
                    """Collect raw source text for direct token-frequency counting.

                    Called by:
                    - The snapshot-input preparation branch in this worker
                      because raw text inputs and stateless plain-English
                      tokenizer preferences should use the direct frequency
                      counter instead of materializing temporary token streams.
                    """

                    docs_df = cast(
                        pl.DataFrame,
                        snapshot_node.data.select(
                            pl.col(source_column).alias("__doc_col__")
                        ).collect(),
                    )
                    return [
                        str(value) if value is not None else ""
                        for value in docs_df["__doc_col__"].to_list()
                    ]

                tokenization_col = node.find_tokenization_column(source_column)
                if tokenization_col is not None:
                    tokenization_meta = node.tokenization.get(source_column, {})
                    model = (
                        tokenization_meta.get("model")
                        if isinstance(tokenization_meta, dict)
                        else None
                    )
                    tokenization_model = (
                        model.strip() if isinstance(model, str) and model.strip() else None
                    )
                    if tokenization_model is not None:
                        requested_node_tokenizer_models[node_id] = tokenization_model
                    if tokenization_model == _PLAIN_WORDS_EN_MODEL:
                        # Plain words is stateless and cheap to count directly.
                        # Hydrating the tokenization preference would build
                        # offset structs, touch DuckDB, and spill an exploded
                        # token stream before doing the same frequency count.
                        corpora[node_id] = collect_source_corpus()
                        continue
                    node_data = hydrate_tokenization_lazyframe(
                        node=node,
                        source_column=source_column,
                        user_id=user_id,
                    )
                    stream_path = (
                        artifact_root
                        / f"{artifact_prefix}_tokens_stream_{node_id}.parquet"
                    )
                    (
                        node_data.select(
                            pl.col(tokenization_col)
                            .list.eval(pl.element().struct.field("token"))
                            .explode()
                            .alias("token")
                        )
                        .filter(pl.col("token").is_not_null())
                        .sink_parquet(stream_path)
                    )
                    token_streams[node_id] = str(stream_path)
                else:
                    corpora[node_id] = collect_source_corpus()

        def tokenizer_model_for_node(node_id: str) -> str | None:
            """Support token-frequency worker helpers with a tokenizer model for node helper.

            Called by:
            - The `run_token_frequencies_task` local workflow in this module because background jobs
              need one lifecycle owner for submission, progress, cancellation, and artifact cleanup.

            Flow: resolve tokenization preferences, hydrate or create token columns, aggregate
                frequencies, and persist derived artifacts for result queries.
            """

            return (
                requested_node_tokenizer_models.get(node_id) or fallback_tokenizer_model
            )

        prepared_node_ids = list({**corpora, **token_streams}.keys())
        if not prepared_node_ids:
            raise ValueError("At least one corpus is required")
        if len(prepared_node_ids) > 2:
            raise ValueError("Maximum of 2 corpora can be compared")
        missing_tokenizer_model_node_ids = [
            node_id
            for node_id in corpora
            if tokenizer_model_for_node(node_id) is None
        ]
        if missing_tokenizer_model_node_ids:
            raise ValueError(
                "node_tokenizer_models must include a tokenizer model for raw-text nodes: "
                + ", ".join(missing_tokenizer_model_node_ids)
            )

        for i, node_id in enumerate(prepared_node_ids):
            node_name = display_names.get(node_id) or node_id

            if progress_callback:
                progress_callback(
                    0.2 + 0.3 * (i + 1) / max(len(prepared_node_ids), 1),
                    f"Prepared text data for {node_name}",
                )

        if progress_callback:
            progress_callback(0.6, "Computing token frequencies...")

        frequency_results: dict[str, dict[str, int]] = {}
        node_models_used: dict[str, str] = {}
        stats_df = None
        for node_id in prepared_node_ids:
            if node_id in token_streams:
                # The API endpoint spilled one row per token (post-explode,
                # post-null-filter) to a parquet via
                # ``sink_parquet`` so we count in Polars without
                # round-tripping through Python objects. The endpoint
                # guarantees the column name is ``token``.
                # ``scan_parquet`` + ``group_by`` + ``len`` stays lazy
                # until the final ``collect`` returns a small N×2 frame.
                freq_df = cast(
                    pl.DataFrame,
                    (
                        pl.scan_parquet(token_streams[node_id])
                        .group_by("token")
                        .len()
                        .rename({"len": "frequency"})
                        .with_columns(
                            pl.col("token").cast(pl.Utf8),
                            pl.col("frequency").cast(pl.Int64),
                        )
                        .collect()
                    ),
                )
                frequency_results[node_id] = {
                    str(row["token"]): int(row["frequency"])
                    for row in freq_df.to_dicts()
                }
            else:
                docs = corpora.get(node_id) or []
                series = pl.Series(
                    "document",
                    [str(v) if v is not None else "" for v in docs],
                )
                effective_tokenizer_model = tokenizer_model_for_node(node_id)
                assert effective_tokenizer_model is not None
                node_models_used[node_id] = effective_tokenizer_model
                frequency_results[node_id] = pt.token_frequencies(
                    series,
                    model=effective_tokenizer_model,
                )

        if len(prepared_node_ids) == 2:
            stats_df = pt.token_frequency_stats(
                frequency_results[prepared_node_ids[0]],
                frequency_results[prepared_node_ids[1]],
            )

        if progress_callback:
            progress_callback(0.85, "Writing token-frequency results...")

        node_artifacts: list[dict[str, Any]] = []
        for frame_key, freq_dict in frequency_results.items():
            sorted_tokens = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
            filtered_tokens = [
                (token, freq) for token, freq in sorted_tokens if freq and freq > 0
            ]
            token_rows = [
                {"token": token, "frequency": int(freq)}
                for token, freq in filtered_tokens
            ]
            token_path = (
                artifact_root
                / f"{artifact_prefix}_token_frequencies_{frame_key}.parquet"
            )
            pl.DataFrame(token_rows).with_columns(
                [
                    pl.col("token").cast(pl.Utf8),
                    pl.col("frequency").cast(pl.Int64),
                ]
            ).lazy().sink_parquet(token_path)
            display_name = display_names.get(frame_key, frame_key)
            node_artifacts.append(
                {
                    "node_id": frame_key,
                    "node_name": display_name,
                    "token_parquet_path": str(token_path),
                }
            )

        statistics_path: str | None = None
        if len(prepared_node_ids) == 2 and stats_df is not None:
            stats_path = artifact_root / f"{artifact_prefix}_token_statistics.parquet"
            stats_df.lazy().sink_parquet(stats_path)
            statistics_path = str(stats_path)

        shared_model = None
        if node_models_used:
            unique_models = set(node_models_used.values())
            shared_model = (
                next(iter(unique_models)) if len(unique_models) == 1 else None
            )

        analysis_params_dict = {
            "node_ids": list(prepared_node_ids),
            "node_columns": {},
            "token_limit": effective_limit,
            "server_limit": server_limit,
            "stop_words": requested_stop_words,
            "tokenizer_model": shared_model,
            "node_tokenizer_models": node_models_used,
        }

        result_payload: dict[str, Any] = {
            "state": "successful",
            "message": f"Successfully calculated token frequencies for {len(prepared_node_ids)} node(s)",
            "artifacts": {
                "version": 1,
                "nodes": node_artifacts,
                "statistics_parquet_path": statistics_path,
                "input_token_streams": [
                    {
                        "node_id": node_id,
                        "token_stream_parquet_path": path,
                    }
                    for node_id, path in token_streams.items()
                ],
            },
            "token_limit": effective_limit,
            "analysis_params": analysis_params_dict,
            "metadata": {
                "token_limit": effective_limit,
                "server_limit": server_limit,
                "stop_words": requested_stop_words,
                "tokenizer_model": shared_model,
                "node_tokenizer_models": node_models_used,
                "node_display_names": {**display_names},
            },
            "stop_words": requested_stop_words,
        }

        if progress_callback:
            progress_callback(1.0, "Token frequency analysis completed")

        logger.info("Token frequencies completed successfully")
        return result_payload

    except Exception as e:
        logger.error("Token frequencies failed: %s", e)
        if progress_callback:
            progress_callback(-1, f"Failed: {str(e)}")
        raise
