"""Process-pool worker facade and task registry.

Used by:
- Backend API routes, worker tasks, workspace services, and backend tests because they
  need a backend boundary that validates inputs before delegating to workspace or worker
  state.

Flow: configure the child-process environment, delegate to the registered task
    implementation, translate progress callbacks, and keep pool lifecycle concerns
    outside API routes.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any, Callable

from .worker_tasks_concordance import (
    run_concordance_analysis_task,
    run_concordance_detach_task,
    run_concordance_dispersion_detach_task,
    run_concordance_materialize_task,
)
from .worker_tasks_download import run_workspace_download_task
from .worker_tasks_import import run_ldaca_import_task
from .worker_tasks_quotation import (
    run_quotation_analysis_task,
    run_quotation_detach_task,
    run_quotation_materialize_task,
)
from .worker_tasks_sequential import run_sequential_analysis_task
from .worker_tasks_token import run_token_frequencies_task
from .worker_tasks_topic import run_topic_modeling_task
from .worker_utils import configure_worker_environment

logger = logging.getLogger(__name__)


def _build_progress_callback(
    progress_queue: Any | None,
    progress_callback: Callable[[float, str], None] | None,
) -> Callable[[float, str], None] | None:
    """Build progress callback values used by background worker process orchestration.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need a
      backend boundary that validates inputs before delegating to workspace or worker state.

    Flow: configure the child-process environment, delegate to the registered task
        implementation, translate progress callbacks, and keep pool lifecycle concerns
        outside API routes.
    """

    if progress_queue is None and progress_callback is None:
        return None

    def _cb(progress: float, message: str) -> None:
        """Support background worker process orchestration with a cb helper.

        Called by:
        - The `_build_progress_callback` local workflow in this module because background jobs
          need one lifecycle owner for submission, progress, cancellation, and artifact cleanup.

        Flow: configure the child-process environment, delegate to the registered task
            implementation, translate progress callbacks, and keep pool lifecycle concerns
            outside API routes.
        """

        payload = {
            "progress": float(progress),
            "message": str(message),
            "timestamp": time.time(),
        }

        if progress_queue is not None:
            try:
                progress_queue.put_nowait(payload)
            except Exception as exc:
                logger.debug(
                    "progress_queue.put_nowait failed, retrying with put: %s", exc
                )
                try:
                    progress_queue.put(payload)
                except Exception as put_exc:
                    logger.debug(
                        "progress_queue.put failed; dropping progress payload: %s",
                        put_exc,
                    )

        if progress_callback is not None:
            try:
                progress_callback(progress, message)
            except Exception as exc:
                logger.debug("progress_callback invocation failed: %s", exc)

    return _cb


def ldaca_import_task(
    user_id: str,
    workspace_id: str,
    url: str,
    filename: str | None = None,
    api_token: str | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    progress_queue: Any | None = None,
) -> dict[str, Any]:
    """Run the ldaca import task background job submitted by API routes.

    Used by:
    - core workspace and worker services because background jobs need one lifecycle owner
      for submission, progress, cancellation, and artifact cleanup.

    Flow: configure the child-process environment, delegate to the registered task
        implementation, translate progress callbacks, and keep pool lifecycle concerns
        outside API routes.
    """

    cb = _build_progress_callback(progress_queue, progress_callback)
    return run_ldaca_import_task(
        configure_worker_environment,
        user_id,
        workspace_id,
        url,
        filename,
        api_token,
        cb,
    )


def workspace_download_task(
    user_id: str,
    workspace_id: str,
    target_workspace_dir: str | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    progress_queue: Any | None = None,
) -> dict[str, Any]:
    """Run the workspace download task background job submitted by API routes.

    Used by:
    - core workspace and worker services because background jobs need one lifecycle owner
      for submission, progress, cancellation, and artifact cleanup.

    Flow: configure the child-process environment, delegate to the registered task
        implementation, translate progress callbacks, and keep pool lifecycle concerns
        outside API routes.
    """

    cb = _build_progress_callback(progress_queue, progress_callback)
    return run_workspace_download_task(
        configure_worker_environment,
        user_id,
        workspace_id,
        target_workspace_dir,
        cb,
    )


def concordance_detach_task(
    user_id: str,
    workspace_id: str,
    workspace_dir: str,
    node_corpus: list[str],
    parent_node_id: str,
    document_column: str,
    search_word: str,
    num_left_tokens: int,
    num_right_tokens: int,
    regex: bool,
    whole_word: bool,
    case_sensitive: bool,
    new_node_name: str,
    include_document_column: bool = False,
    include_extraction: bool = False,
    selected_generated_columns: list[str] | None = None,
    extra_columns_data: dict[str, list] | None = None,
    extra_columns_dtypes: dict[str, Any] | None = None,
    materialized_path: str | None = None,
    input_snapshot_dir: str | None = None,
    extra_column_names: list[str] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    progress_queue: Any | None = None,
) -> dict[str, Any]:
    """Run the concordance detach task background job submitted by API routes.

    Used by:
    - backend tests, core workspace and worker services because tests need the same
      observable contract that production routes and workers rely on.

    Flow: configure the child-process environment, delegate to the registered task
        implementation, translate progress callbacks, and keep pool lifecycle concerns
        outside API routes.
    """

    cb = _build_progress_callback(progress_queue, progress_callback)
    return run_concordance_detach_task(
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
        include_document_column=include_document_column,
        include_extraction=include_extraction,
        selected_generated_columns=selected_generated_columns,
        extra_columns_data=extra_columns_data,
        extra_columns_dtypes=extra_columns_dtypes,
        materialized_path=materialized_path,
        input_snapshot_dir=input_snapshot_dir,
        extra_column_names=extra_column_names,
        user_id=user_id,
        progress_callback=cb,
    )


def concordance_dispersion_detach_task(
    user_id: str,
    workspace_id: str,
    workspace_dir: str,
    node_corpus: list[str],
    parent_node_id: str,
    document_column: str,
    search_word: str,
    num_left_tokens: int,
    num_right_tokens: int,
    regex: bool,
    whole_word: bool,
    case_sensitive: bool,
    new_node_name: str,
    child_task_id: str | None = None,
    parent_task_id: str | None = None,
    include_document_column: bool = True,
    extra_columns_data: dict[str, list] | None = None,
    extra_columns_dtypes: dict[str, Any] | None = None,
    materialized_path: str | None = None,
    selected_bins: list[int] | None = None,
    total_bins: int | None = None,
    selected_matched_texts: list[str] | None = None,
    match_case_insensitive: bool = False,
    input_snapshot_dir: str | None = None,
    extra_column_names: list[str] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    progress_queue: Any | None = None,
) -> dict[str, Any]:
    """Run the concordance dispersion detach task background job submitted by API routes.

    Used by:
    - core workspace and worker services because background jobs need one lifecycle owner
      for submission, progress, cancellation, and artifact cleanup.

    Flow: configure the child-process environment, delegate to the registered task
        implementation, translate progress callbacks, and keep pool lifecycle concerns
        outside API routes.
    """

    cb = _build_progress_callback(progress_queue, progress_callback)
    return run_concordance_dispersion_detach_task(
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
        child_task_id=child_task_id,
        parent_task_id=parent_task_id,
        include_document_column=include_document_column,
        extra_columns_data=extra_columns_data,
        extra_columns_dtypes=extra_columns_dtypes,
        materialized_path=materialized_path,
        selected_bins=selected_bins,
        total_bins=total_bins,
        selected_matched_texts=selected_matched_texts,
        match_case_insensitive=match_case_insensitive,
        input_snapshot_dir=input_snapshot_dir,
        extra_column_names=extra_column_names,
        user_id=user_id,
        progress_callback=cb,
    )


def concordance_materialize_task(
    user_id: str,
    workspace_id: str,
    workspace_dir: str,
    node_corpus: list[str],
    child_task_id: str,
    parent_task_id: str,
    parent_node_id: str,
    document_column: str,
    search_word: str,
    num_left_tokens: int,
    num_right_tokens: int,
    regex: bool,
    whole_word: bool,
    case_sensitive: bool,
    extra_columns_data: dict[str, list] | None = None,
    extra_columns_dtypes: dict[str, Any] | None = None,
    search_mode: str = "regex",
    node_tokens: list[Any] | None = None,
    input_snapshot_dir: str | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    progress_queue: Any | None = None,
) -> dict[str, Any]:
    """Run the concordance materialize task background job submitted by API routes.

    Used by:
    - core workspace and worker services because background jobs need one lifecycle owner
      for submission, progress, cancellation, and artifact cleanup.

    Flow: configure the child-process environment, delegate to the registered task
        implementation, translate progress callbacks, and keep pool lifecycle concerns
        outside API routes.
    """

    cb = _build_progress_callback(progress_queue, progress_callback)
    return run_concordance_materialize_task(
        configure_worker_environment,
        workspace_dir,
        node_corpus,
        child_task_id,
        parent_task_id,
        parent_node_id,
        document_column,
        search_word,
        num_left_tokens,
        num_right_tokens,
        regex,
        whole_word,
        case_sensitive,
        extra_columns_data=extra_columns_data,
        extra_columns_dtypes=extra_columns_dtypes,
        search_mode=search_mode,
        node_tokens=node_tokens,
        input_snapshot_dir=input_snapshot_dir,
        user_id=user_id,
        progress_callback=cb,
    )


def quotation_detach_task(
    user_id: str,
    workspace_id: str,
    workspace_dir: str,
    node_corpus: list[str],
    parent_node_id: str,
    document_column: str,
    engine_config: dict[str, Any],
    new_node_name: str,
    include_document_column: bool = False,
    include_extraction: bool = False,
    selected_generated_columns: list[str] | None = None,
    extra_columns_data: dict[str, list] | None = None,
    extra_columns_dtypes: dict[str, Any] | None = None,
    materialized_path: str | None = None,
    input_snapshot_dir: str | None = None,
    extra_column_names: list[str] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    progress_queue: Any | None = None,
) -> dict[str, Any]:
    """Run the quotation detach task background job submitted by API routes.

    Used by:
    - core workspace and worker services because background jobs need one lifecycle owner
      for submission, progress, cancellation, and artifact cleanup.

    Flow: configure the child-process environment, delegate to the registered task
        implementation, translate progress callbacks, and keep pool lifecycle concerns
        outside API routes.
    """

    cb = _build_progress_callback(progress_queue, progress_callback)
    return run_quotation_detach_task(
        configure_worker_environment,
        workspace_dir,
        node_corpus,
        parent_node_id,
        document_column,
        engine_config,
        new_node_name,
        include_document_column=include_document_column,
        include_extraction=include_extraction,
        selected_generated_columns=selected_generated_columns,
        extra_columns_data=extra_columns_data,
        extra_columns_dtypes=extra_columns_dtypes,
        materialized_path=materialized_path,
        input_snapshot_dir=input_snapshot_dir,
        extra_column_names=extra_column_names,
        progress_callback=cb,
    )


def quotation_materialize_task(
    user_id: str,
    workspace_id: str,
    workspace_dir: str,
    node_corpus: list[str],
    child_task_id: str,
    parent_task_id: str,
    parent_node_id: str,
    document_column: str,
    engine_config: dict[str, Any],
    extra_columns_data: dict[str, list] | None = None,
    extra_columns_dtypes: dict[str, Any] | None = None,
    input_snapshot_dir: str | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    progress_queue: Any | None = None,
) -> dict[str, Any]:
    """Run the quotation materialize task background job submitted by API routes.

    Used by:
    - core workspace and worker services because background jobs need one lifecycle owner
      for submission, progress, cancellation, and artifact cleanup.

    Flow: configure the child-process environment, delegate to the registered task
        implementation, translate progress callbacks, and keep pool lifecycle concerns
        outside API routes.
    """

    cb = _build_progress_callback(progress_queue, progress_callback)
    return run_quotation_materialize_task(
        configure_worker_environment,
        workspace_dir,
        node_corpus,
        child_task_id,
        parent_task_id,
        parent_node_id,
        document_column,
        engine_config,
        extra_columns_data=extra_columns_data,
        extra_columns_dtypes=extra_columns_dtypes,
        input_snapshot_dir=input_snapshot_dir,
        progress_callback=cb,
    )


def topic_modeling_task(
    user_id: str,
    workspace_id: str,
    node_infos: list[dict[str, Any]],
    artifact_dir: str,
    artifact_prefix: str,
    min_topic_size: int,
    workspace_dir: str | None = None,
    input_snapshot_dir: str | None = None,
    corpora: list[list[str]] | None = None,
    random_seed: int = 42,
    representative_words_count: int = 5,
    progress_callback: Callable[[float, str], None] | None = None,
    progress_queue: Any | None = None,
    sample_fractions: list[float | None] | None = None,
) -> dict[str, Any]:
    """Run the topic modeling task background job submitted by API routes.

    Used by:
    - core workspace and worker services because background jobs need one lifecycle owner
      for submission, progress, cancellation, and artifact cleanup.

    Flow: configure the child-process environment, delegate to the registered task
        implementation, translate progress callbacks, and keep pool lifecycle concerns
        outside API routes.
    """

    cb = _build_progress_callback(progress_queue, progress_callback)
    return run_topic_modeling_task(
        configure_worker_environment=configure_worker_environment,
        user_id=user_id,
        workspace_id=workspace_id,
        workspace_dir=workspace_dir,
        input_snapshot_dir=input_snapshot_dir,
        corpora=corpora,
        node_infos=node_infos,
        artifact_dir=artifact_dir,
        artifact_prefix=artifact_prefix,
        min_topic_size=min_topic_size,
        random_seed=random_seed,
        representative_words_count=representative_words_count,
        progress_callback=cb,
        sample_fractions=sample_fractions,
    )


def concordance_task(
    user_id: str,
    workspace_id: str,
    input_snapshot_dir: str,
    request_payload: dict[str, Any],
    progress_callback: Callable[[float, str], None] | None = None,
    progress_queue: Any | None = None,
) -> dict[str, Any]:
    """Run the primary concordance analysis worker submitted by API routes.

    Used by:
    - concordance submit routes because their initial HTTP response must only
      register/submit work and return a task id.
    """

    cb = _build_progress_callback(progress_queue, progress_callback)
    return run_concordance_analysis_task(
        configure_worker_environment=configure_worker_environment,
        user_id=user_id,
        workspace_id=workspace_id,
        input_snapshot_dir=input_snapshot_dir,
        request_payload=request_payload,
        progress_callback=cb,
    )


def quotation_task(
    user_id: str,
    workspace_id: str,
    input_snapshot_dir: str,
    node_id: str,
    request_payload: dict[str, Any],
    progress_callback: Callable[[float, str], None] | None = None,
    progress_queue: Any | None = None,
) -> dict[str, Any]:
    """Run the primary quotation analysis worker submitted by API routes.

    Used by:
    - quotation submit routes because their initial HTTP response must only
      register/submit work and return a task id.
    """

    cb = _build_progress_callback(progress_queue, progress_callback)
    return run_quotation_analysis_task(
        configure_worker_environment=configure_worker_environment,
        user_id=user_id,
        workspace_id=workspace_id,
        input_snapshot_dir=input_snapshot_dir,
        node_id=node_id,
        request_payload=request_payload,
        progress_callback=cb,
    )


def sequential_analysis_task(
    user_id: str,
    workspace_id: str,
    input_snapshot_dir: str,
    node_id: str,
    request_payload: dict[str, Any],
    progress_callback: Callable[[float, str], None] | None = None,
    progress_queue: Any | None = None,
) -> dict[str, Any]:
    """Run the sequential analysis background job submitted by API routes.

    Used by:
    - sequential-analysis submit routes because their initial HTTP response
      must only register/submit work and return a task id.
    """

    cb = _build_progress_callback(progress_queue, progress_callback)
    return run_sequential_analysis_task(
        configure_worker_environment=configure_worker_environment,
        user_id=user_id,
        workspace_id=workspace_id,
        input_snapshot_dir=input_snapshot_dir,
        node_id=node_id,
        request_payload=request_payload,
        progress_callback=cb,
    )


def token_frequencies_task(
    user_id: str,
    workspace_id: str,
    artifact_dir: str,
    artifact_prefix: str,
    token_limit: int,
    node_corpora: dict[str, list[str]] | None = None,
    node_display_names: dict[str, str] | None = None,
    stop_words: list[str] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    progress_queue: Any | None = None,
    node_token_streams: dict[str, str] | None = None,
    tokenizer_model: str | None = None,
    node_tokenizer_models: dict[str, str] | None = None,
    input_snapshot_dir: str | None = None,
    node_ids: list[str] | None = None,
    node_columns: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run the token frequencies task background job submitted by API routes.

    Used by:
    - backend tests, core workspace and worker services because tests need the same
      observable contract that production routes and workers rely on.

    Flow: configure the child-process environment, delegate to the registered task
        implementation, translate progress callbacks, and keep pool lifecycle concerns
        outside API routes.
    """

    cb = _build_progress_callback(progress_queue, progress_callback)
    return run_token_frequencies_task(
        configure_worker_environment=configure_worker_environment,
        user_id=user_id,
        workspace_id=workspace_id,
        node_corpora=node_corpora or {},
        node_display_names=node_display_names or {},
        artifact_dir=artifact_dir,
        artifact_prefix=artifact_prefix,
        token_limit=token_limit,
        stop_words=stop_words,
        progress_callback=cb,
        node_token_streams=node_token_streams,
        tokenizer_model=tokenizer_model,
        node_tokenizer_models=node_tokenizer_models,
        input_snapshot_dir=input_snapshot_dir,
        node_ids=node_ids,
        node_columns=node_columns,
    )


def _pid_reporting_wrapper(task_func: Any, **kwargs: Any) -> Any:
    """Wrapper executed inside the worker process.

    Sends the worker's own PID as the first message on the progress queue so
    the main process can terminate it if the user requests cancellation.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need a
      backend boundary that validates inputs before delegating to workspace or worker state.

    Flow: configure the child-process environment, delegate to the registered task
        implementation, translate progress callbacks, and keep pool lifecycle concerns
        outside API routes.
    """
    pq = kwargs.get("progress_queue")
    if pq is not None:
        try:
            pq.put_nowait({"type": "pid", "pid": os.getpid()})
        except Exception:
            pass
    return task_func(**kwargs)


TASK_REGISTRY: dict[str, Any] = {
    "ldaca_import": ldaca_import_task,
    "workspace_download": workspace_download_task,
    "concordance": concordance_task,
    "concordance_detach": concordance_detach_task,
    "concordance_dispersion_detach": concordance_dispersion_detach_task,
    "concordance_materialize": concordance_materialize_task,
    "quotation_detach": quotation_detach_task,
    "quotation": quotation_task,
    "quotation_materialize": quotation_materialize_task,
    "topic_modeling": topic_modeling_task,
    "sequential_analysis": sequential_analysis_task,
    "token_frequencies": token_frequencies_task,
}

_worker_pool: WorkerPool | None = None


def get_worker_pool(max_workers: int = 2) -> "WorkerPool":
    """Return worker pool data used by background worker process orchestration.

    Used by:
    - FastAPI application startup, core workspace and worker services because they need a
      backend boundary that validates inputs before delegating to workspace or worker state.

    Flow: configure the child-process environment, delegate to the registered task
        implementation, translate progress callbacks, and keep pool lifecycle concerns
        outside API routes.
    """

    global _worker_pool
    if _worker_pool is None:
        _worker_pool = WorkerPool(max_workers=max_workers)
    return _worker_pool


class WorkerPool:
    """Simple process-pool task manager for CPU-heavy operations.

    Used by:
    - backend tests, core workspace and worker services because tests need the same
      observable contract that production routes and workers rely on.

    Flow: configure the child-process environment, delegate to the registered task
        implementation, translate progress callbacks, and keep pool lifecycle concerns
        outside API routes.
    """

    def __init__(self, max_workers: int = 2):
        """Initialize WorkerPool state used by background worker process orchestration.

        Called by:
        - `WorkerPool` construction in backend services and tests because tests need the
          same observable contract that production routes and workers rely on.

        Flow: configure the child-process environment, delegate to the registered task
            implementation, translate progress callbacks, and keep pool lifecycle concerns
            outside API routes.
        """

        self.max_workers = max_workers
        self.executor: ProcessPoolExecutor | None = None
        self.is_running = False

    def start(self) -> None:
        """Start WorkerPool resources used by background worker process orchestration.

        Called by:
        - `WorkerPool` instances owned by backend services, routes, and tests because
          they need a backend boundary that validates inputs before delegating to workspace or
          worker state.

        Flow: configure the child-process environment, delegate to the registered task
            implementation, translate progress callbacks, and keep pool lifecycle concerns
            outside API routes.
        """

        if self.executor is None:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self.is_running = True

    def submit_task(self, task_func: Any, **kwargs: Any) -> Future:
        """Submit background work to the pool used by task routes.

        Called by:
        - `WorkerPool` instances owned by backend services, routes, and tests because
          they need a backend boundary that validates inputs before delegating to workspace or
          worker state.

        Flow: configure the child-process environment, delegate to the registered task
            implementation, translate progress callbacks, and keep pool lifecycle concerns
            outside API routes.
        """

        if self.executor is None:
            self.start()

        assert self.executor is not None
        return self.executor.submit(_pid_reporting_wrapper, task_func, **kwargs)

    def shutdown(self, wait: bool = True, timeout: float | None = None) -> None:
        """Shutdown WorkerPool resources used by background worker process orchestration.

        Called by:
        - `WorkerPool` instances owned by backend services, routes, and tests because
          they need a backend boundary that validates inputs before delegating to workspace or
          worker state.

        Flow: configure the child-process environment, delegate to the registered task
            implementation, translate progress callbacks, and keep pool lifecycle concerns
            outside API routes.
        """

        if self.executor is None:
            self.is_running = False
            return
        self.executor.shutdown(wait=wait)
        self.executor = None
        self.is_running = False
