"""Workspace-owned Analysis commands and side-effect-free projections."""

from __future__ import annotations

import logging
import math
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import anyio
from pydantic import ValidationError

from ..domain.workspace import (
    Analysis,
    AnalysisArtifactRecord,
    AnalysisQuerySnapshotRecord,
    AnalysisRecord,
    AnalysisState,
    AnalysisSubmission,
    AnnotationAnalysisRequest,
    AnnotationAnalysisSubmission,
    ChildAnalysisRequest,
    ConcordanceAnalysisRequest,
    ConcordanceDetachmentAnalysisRequest,
    ConcordanceDispersionDetachmentAnalysisRequest,
    CorruptAnalysis,
    Failure,
    InvalidAnalysisIntegrity,
    Progress,
    QuotationAnalysisRequest,
    QuotationDetachmentAnalysisRequest,
    TopicModelingAnalysisRequest,
    TopicModelingDetachmentAnalysisRequest,
    ValidAnalysisIntegrity,
    Workspace,
    analysis_input_ids,
    persisted_submission,
    public_analysis,
)
from ..models.analyses import AnalysisPage
from ..shared.errors import (
    AppError,
    AnalysisCorruptError,
    AnalysisInputGoneError,
    AnalysisInputMissingError,
    AnalysisKindMismatchError,
    AnalysisNotCancellableError,
    AnalysisNotFoundError,
    AnalysisParentInvalidError,
    AnalysisNotSucceededError,
    BackendStoppingError,
    TabAnalysisExistsError,
    TabNotFoundError,
)
from ..shared.json_data import JsonData
from .analysis_execution_types import (
    AnalysisExecutionControl,
    AnalysisExecutionKey,
    AnalysisInvocation,
    AnalysisSchedulingStopped,
)
from .workspace import WorkspaceLease, WorkspaceService
from .provider_credentials import ProviderCredentialStore

logger = logging.getLogger(__name__)
ExecutionPreparer = Callable[
    [WorkspaceLease, AnalysisRecord, str | None], Awaitable[AnalysisInvocation]
]
LaunchControl = Callable[[AnalysisExecutionKey], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class PublishedAnalysisResult:
    """Validated result payload and resources published by one execution."""

    payload: dict[str, JsonData]
    artifacts: list[AnalysisArtifactRecord]
    output_node_ids: list[uuid.UUID]
    query_snapshot: AnalysisQuerySnapshotRecord | None = None


class AnalysisResultPublisher(Protocol):
    async def publish_result(
        self,
        lease: WorkspaceLease,
        record: AnalysisRecord,
        raw_result: object,
    ) -> PublishedAnalysisResult: ...


class AnalysisService:
    """Own Analysis lifecycle state while WorkspaceService owns persistence."""

    def __init__(
        self,
        workspaces: WorkspaceService,
        execution: AnalysisExecutionControl,
        artifacts: AnalysisResultPublisher,
        *,
        credentials: ProviderCredentialStore,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._workspaces = workspaces
        self._execution = execution
        self._artifacts = artifacts
        self._credentials = credentials
        self._clock = clock or (lambda: datetime.now(UTC))
        self._live_progress: dict[tuple[str, str], Progress] = {}
        self._accepting = True

    @staticmethod
    def _key(user_id: str, workspace_id: str, analysis_id: str) -> AnalysisExecutionKey:
        return AnalysisExecutionKey(user_id, workspace_id, analysis_id)

    @staticmethod
    def _missing_input_ids(
        lease: WorkspaceLease,
        record: AnalysisRecord,
    ) -> list[uuid.UUID]:
        return [
            node_id
            for node_id in analysis_input_ids(record.request)
            if str(node_id) not in lease.workspace.nodes
        ]

    def _project(self, lease: WorkspaceLease, record: AnalysisRecord) -> Analysis:
        missing = self._missing_input_ids(lease, record)
        integrity = (
            InvalidAnalysisIntegrity(missing_input_ids=missing)
            if missing
            else ValidAnalysisIntegrity()
        )
        progress = self._live_progress.get((lease.workspace.id, str(record.id)))
        return public_analysis(record, integrity=integrity, progress=progress)

    def _current_progress(
        self,
        workspace_id: str,
        record: AnalysisRecord,
    ) -> Progress:
        return self._live_progress.get((workspace_id, str(record.id)), record.progress)

    @staticmethod
    def _detached_root_id(lease: WorkspaceLease, record: AnalysisRecord) -> str | None:
        root_id = str(record.parent_analysis_id or record.id)
        return root_id if lease.workspace.analysis_tab_id(root_id) is None else None

    @staticmethod
    def _remove_terminal_detached_tree(
        lease: WorkspaceLease,
        root_id: str | None,
    ) -> None:
        if root_id is None:
            return
        root = lease.workspace.analyses.get(root_id)
        if root is None:
            return
        tree = [root, *lease.workspace.analysis_children(root_id)]
        if all(
            record.state
            in {
                AnalysisState.SUCCEEDED,
                AnalysisState.FAILED,
                AnalysisState.CANCELLED,
            }
            for record in tree
        ):
            lease.workspace.remove_analysis(root_id)

    @staticmethod
    def _require_live_record(
        lease: WorkspaceLease,
        analysis_id: str,
    ) -> AnalysisRecord:
        if analysis_id not in lease.workspace.live_analysis_ids():
            raise AnalysisNotFoundError("Analysis not found")
        if analysis_id in lease.workspace.corrupt_analysis_ids:
            raise AnalysisCorruptError("Analysis data is corrupt")
        record = lease.workspace.analyses.get(analysis_id)
        if record is None:
            raise AnalysisNotFoundError("Analysis not found")
        return record

    async def submit_root(
        self,
        user_id: str,
        workspace_id: str,
        tab_id: str,
        submission: AnalysisSubmission,
    ) -> Analysis:
        """Atomically create one queued root and assign it to an empty Tab."""

        if not self._accepting:
            raise BackendStoppingError()
        request = persisted_submission(submission)
        credential = (
            await self._credentials.annotation_credential(
                request.provider,
                supplied=(
                    submission.api_key
                    if isinstance(submission, AnnotationAnalysisSubmission)
                    else None
                ),
            )
            if isinstance(request, AnnotationAnalysisRequest)
            else None
        )
        timestamp = self._clock()
        async with self._workspaces.mutation_context(user_id, workspace_id) as lease:
            tab = lease.workspace.tabs.get(tab_id)
            if tab is None:
                raise TabNotFoundError("Tab not found")
            if request.kind != tab.kind.value:
                raise AnalysisKindMismatchError("Analysis kind does not match the Tab")
            if tab.analysis_id is not None:
                raise TabAnalysisExistsError("Tab already has an Analysis")
            missing = [
                node_id
                for node_id in analysis_input_ids(request)
                if str(node_id) not in lease.workspace.nodes
            ]
            if missing:
                raise AnalysisInputMissingError(
                    "Analysis input is missing",
                    details={
                        "missing_input_ids": [str(node_id) for node_id in missing]
                    },
                )
            record = AnalysisRecord.create(request, timestamp=timestamp)
            lease.workspace.add_analysis(record)
            tab.analysis_id = record.id
            tab.modified_at = timestamp
            tab.revision += 1
            resource = self._project(lease, record)

        key = self._key(user_id, workspace_id, str(record.id))
        await self._schedule_created(
            key,
            created_at=record.created_at,
            credential=credential,
        )
        return resource

    async def submit_child(
        self,
        user_id: str,
        workspace_id: str,
        parent_analysis_id: str,
        request: ChildAnalysisRequest,
    ) -> Analysis:
        """Create one independently observable child beneath a successful root."""

        if not self._accepting:
            raise BackendStoppingError()
        timestamp = self._clock()
        async with self._workspaces.mutation_context(user_id, workspace_id) as lease:
            parent = self._require_live_record(lease, parent_analysis_id)
            if parent.parent_analysis_id is not None:
                raise AnalysisParentInvalidError("Child Analyses cannot have children")
            if parent.state is not AnalysisState.SUCCEEDED:
                raise AnalysisNotSucceededError("Parent Analysis has not succeeded")
            missing = self._missing_input_ids(lease, parent)
            if missing:
                raise AnalysisInputGoneError(
                    "Analysis input is missing",
                    details={
                        "missing_input_ids": [str(node_id) for node_id in missing]
                    },
                )
            compatible = (
                isinstance(parent.request, ConcordanceAnalysisRequest)
                and isinstance(
                    request,
                    ConcordanceDetachmentAnalysisRequest
                    | ConcordanceDispersionDetachmentAnalysisRequest,
                )
            ) or (
                isinstance(parent.request, QuotationAnalysisRequest)
                and isinstance(request, QuotationDetachmentAnalysisRequest)
            ) or (
                isinstance(parent.request, TopicModelingAnalysisRequest)
                and isinstance(request, TopicModelingDetachmentAnalysisRequest)
            )
            requested_ids = set(analysis_input_ids(request))
            parent_ids = set(analysis_input_ids(parent.request))
            if not compatible or not requested_ids.issubset(parent_ids):
                raise AnalysisParentInvalidError(
                    "Child Analysis does not match its parent"
                )
            record = AnalysisRecord.create(
                request,
                timestamp=timestamp,
                parent_analysis_id=parent.id,
            )
            lease.workspace.add_analysis(record)
            resource = self._project(lease, record)

        key = self._key(user_id, workspace_id, str(record.id))
        await self._schedule_created(
            key,
            created_at=record.created_at,
            credential=None,
        )
        return resource

    async def _schedule_created(
        self,
        key: AnalysisExecutionKey,
        *,
        created_at: datetime,
        credential: str | None,
    ) -> None:
        """Schedule one durable creation or compensate the complete mutation."""

        try:
            await self._execution.enqueue(
                key,
                created_at=created_at,
                credential=credential,
            )
        except AnalysisSchedulingStopped:
            with anyio.CancelScope(shield=True):
                await self._remove_unscheduled_analysis(key)
            raise BackendStoppingError() from None
        except BaseException:
            with anyio.CancelScope(shield=True):
                await self._remove_unscheduled_analysis(key)
            raise

    async def _remove_unscheduled_analysis(self, key: AnalysisExecutionKey) -> None:
        """Compensate a durable creation rejected by the stopping scheduler."""

        async with self._workspaces.mutation_context(
            key.user_id,
            key.workspace_id,
            internal=True,
        ) as lease:
            record = lease.workspace.analyses.get(key.analysis_id)
            if record is None or record.state is not AnalysisState.QUEUED:
                lease.commit_requested = False
                return
            if record.parent_analysis_id is None:
                tab_id = lease.workspace.analysis_tab_id(key.analysis_id)
                if tab_id is not None:
                    tab = lease.workspace.tabs[tab_id]
                    tab.analysis_id = None
                    tab.modified_at = self._clock()
                    tab.revision += 1
            lease.workspace.remove_analysis(key.analysis_id)
            self._live_progress.pop((key.workspace_id, key.analysis_id), None)

    async def current_for_tab(
        self,
        user_id: str,
        workspace_id: str,
        tab_id: str,
    ) -> Analysis:
        """Return the current root Analysis for one Tab."""

        async with self._workspaces.read_context(user_id, workspace_id) as lease:
            tab = lease.workspace.tabs.get(tab_id)
            if tab is None:
                raise TabNotFoundError("Tab not found")
            if tab.analysis_id is None:
                raise AnalysisNotFoundError("Analysis not found")
            record = self._require_live_record(lease, str(tab.analysis_id))
            return self._project(lease, record)

    async def get(
        self,
        user_id: str,
        workspace_id: str,
        analysis_id: str,
    ) -> Analysis:
        """Return one live valid Analysis without mutating integrity state."""

        async with self._workspaces.read_context(user_id, workspace_id) as lease:
            record = self._require_live_record(lease, analysis_id)
            return self._project(lease, record)

    @asynccontextmanager
    async def successful_record_context(
        self,
        user_id: str,
        workspace_id: str,
        analysis_id: str,
        *,
        allow_closing: bool,
    ) -> AsyncIterator[tuple[WorkspaceLease, AnalysisRecord]]:
        """Yield one usable successful record for brief Result materialization."""

        context = (
            self._workspaces.read_context(user_id, workspace_id)
            if allow_closing
            else self._workspaces.submission_context(user_id, workspace_id)
        )
        async with context as lease:
            record = self._require_live_record(lease, analysis_id)
            if record.state is not AnalysisState.SUCCEEDED:
                raise AnalysisNotSucceededError("Analysis has not succeeded")
            missing = self._missing_input_ids(lease, record)
            if missing:
                raise AnalysisInputGoneError(
                    "Analysis input is missing",
                    details={
                        "missing_input_ids": [str(node_id) for node_id in missing]
                    },
                )
            yield lease, record

    async def list_analyses(
        self,
        user_id: str,
        workspace_id: str,
        *,
        page: int,
        page_size: int,
    ) -> AnalysisPage:
        """Return valid live Analyses first, followed by corrupt root items."""

        async with self._workspaces.read_context(user_id, workspace_id) as lease:
            live_ids = lease.workspace.live_analysis_ids()
            records = [
                record
                for analysis_id, record in lease.workspace.analyses.items()
                if analysis_id in live_ids
            ]
            records.sort(key=lambda record: str(record.id))
            records.sort(key=lambda record: record.created_at, reverse=True)
            items: list[Analysis | CorruptAnalysis] = [
                self._project(lease, record) for record in records
            ]
            for analysis_id in sorted(lease.workspace.corrupt_analysis_ids & live_ids):
                tab_id = lease.workspace.analysis_tab_id(analysis_id)
                if tab_id is not None:
                    items.append(
                        CorruptAnalysis(
                            id=uuid.UUID(analysis_id),
                            tab_id=uuid.UUID(tab_id),
                        )
                    )

            total_items = len(items)
            start = (page - 1) * page_size
            return AnalysisPage(
                items=items[start : start + page_size],
                page=page,
                page_size=page_size,
                total_items=total_items,
                total_pages=math.ceil(total_items / page_size) if total_items else 0,
            )

    async def cancel(
        self,
        user_id: str,
        workspace_id: str,
        analysis_id: str,
    ) -> tuple[Analysis, bool]:
        """Request cancellation and report whether termination remains pending."""

        key = self._key(user_id, workspace_id, analysis_id)
        should_signal = False
        pending = False
        async with self._workspaces.mutation_context(
            user_id,
            workspace_id,
            internal=True,
        ) as lease:
            record = self._require_live_record(lease, analysis_id)
            if record.state is AnalysisState.CANCELLED:
                lease.commit_requested = False
            elif record.state in {AnalysisState.SUCCEEDED, AnalysisState.FAILED}:
                raise AnalysisNotCancellableError("Analysis is not cancellable")
            elif record.state is AnalysisState.QUEUED:
                record = record.cancel_queued(self._clock())
                lease.workspace.replace_analysis(record)
                should_signal = True
            else:
                pending = True
                if record.cancellation_requested_at is None:
                    record = record.request_running_cancellation(self._clock())
                    lease.workspace.replace_analysis(record)
                    should_signal = True
                else:
                    lease.commit_requested = False
            resource = self._project(lease, record)

        if should_signal:
            await self._execution.cancel(key)
        return resource, pending

    async def clear_tab(
        self,
        user_id: str,
        workspace_id: str,
        tab_id: str,
    ) -> None:
        """Detach one root immediately and leave non-terminal cleanup private."""

        keys_to_cancel: list[AnalysisExecutionKey] = []
        async with self._workspaces.mutation_context(user_id, workspace_id) as lease:
            tab = lease.workspace.tabs.get(tab_id)
            if tab is None:
                raise TabNotFoundError("Tab not found")
            if tab.analysis_id is None:
                lease.commit_requested = False
                return
            root_id = str(tab.analysis_id)
            timestamp = self._clock()
            tab.analysis_id = None
            tab.modified_at = timestamp
            tab.revision += 1
            keys_to_cancel = self._detach_analysis_tree(
                lease,
                user_id,
                workspace_id,
                root_id,
                timestamp,
            )

        for key in keys_to_cancel:
            await self._execution.cancel(key)

    async def delete_tab(
        self,
        user_id: str,
        workspace_id: str,
        tab_id: str,
    ) -> None:
        """Delete one Tab and detach its Analysis tree through one mutation."""

        keys_to_cancel: list[AnalysisExecutionKey] = []
        async with self._workspaces.mutation_context(user_id, workspace_id) as lease:
            tab = lease.workspace.remove_tab(tab_id)
            if tab is None:
                raise TabNotFoundError("Tab not found")
            if tab.analysis_id is not None:
                keys_to_cancel = self._detach_analysis_tree(
                    lease,
                    user_id,
                    workspace_id,
                    str(tab.analysis_id),
                    self._clock(),
                )

        for key in keys_to_cancel:
            await self._execution.cancel(key)

    def _detach_analysis_tree(
        self,
        lease: WorkspaceLease,
        user_id: str,
        workspace_id: str,
        root_id: str,
        timestamp: datetime,
    ) -> list[AnalysisExecutionKey]:
        if root_id in lease.workspace.corrupt_analysis_ids:
            lease.workspace.remove_analysis(root_id)
            return []

        root = lease.workspace.analyses.get(root_id)
        if root is None:
            return []
        keys_to_cancel: list[AnalysisExecutionKey] = []
        tree = [root, *lease.workspace.analysis_children(root_id)]
        has_running = False
        for record in tree:
            if record.state is AnalysisState.QUEUED:
                updated = record.cancel_queued(timestamp)
                lease.workspace.replace_analysis(updated)
                keys_to_cancel.append(
                    self._key(user_id, workspace_id, str(record.id))
                )
            elif record.state is AnalysisState.RUNNING:
                has_running = True
                if record.cancellation_requested_at is None:
                    updated = record.request_running_cancellation(timestamp)
                    lease.workspace.replace_analysis(updated)
                keys_to_cancel.append(
                    self._key(user_id, workspace_id, str(record.id))
                )
        if not has_running:
            lease.workspace.remove_analysis(root_id)
        return keys_to_cancel

    async def admit_execution(
        self,
        key: AnalysisExecutionKey,
        *,
        credential: str | None,
        prepare: ExecutionPreparer,
        reserve_launch: LaunchControl,
        discard_launch: LaunchControl,
    ) -> AnalysisInvocation | None:
        """Snapshot and durably admit one queued Analysis under its Workspace gate."""

        reserved = False
        try:
            async with self._workspaces.mutation_context(
                key.user_id,
                key.workspace_id,
                internal=True,
            ) as lease:
                record = lease.workspace.analyses.get(key.analysis_id)
                if record is None or record.state is not AnalysisState.QUEUED:
                    lease.commit_requested = False
                    return None
                missing = self._missing_input_ids(lease, record)
                if missing:
                    failed = record.fail(
                        self._clock(),
                        failure=Failure(
                            code="analysis_input_missing",
                            message="Analysis input is missing",
                        ),
                        progress=record.progress,
                    )
                    lease.workspace.replace_analysis(failed)
                    self._remove_terminal_detached_tree(
                        lease,
                        self._detached_root_id(lease, failed),
                    )
                    return None
                try:
                    invocation = await prepare(lease, record, credential)
                    await reserve_launch(key)
                    reserved = True
                    lease.workspace.replace_analysis(record.start(self._clock()))
                except Exception as exc:
                    if isinstance(exc, AppError):
                        failure = Failure(
                            code=exc.code,
                            message=(
                                exc.message
                                if exc.status_code < 500 or exc.expose_message
                                else "Analysis failed"
                            ),
                        )
                    else:
                        logger.exception(
                            "Analysis dispatch admission failed analysis_id=%s user_id=%s",
                            key.analysis_id,
                            key.user_id,
                        )
                        failure = Failure(
                            code="analysis_start_failed",
                            message="Analysis could not start",
                        )
                    if reserved:
                        await discard_launch(key)
                        reserved = False
                    failed = record.fail(
                        self._clock(),
                        failure=failure,
                        progress=record.progress,
                    )
                    lease.workspace.replace_analysis(failed)
                    self._remove_terminal_detached_tree(
                        lease,
                        self._detached_root_id(lease, failed),
                    )
                    return None
            return invocation
        except BaseException:
            if reserved:
                with anyio.CancelScope(shield=True):
                    await discard_launch(key)
            raise

    async def report_progress(
        self,
        key: AnalysisExecutionKey,
        payload: object,
    ) -> None:
        """Validate one live report or fail only its owning Analysis."""

        should_terminate = False
        live_progress: Progress | None = None
        async with self._workspaces.mutation_context(
            key.user_id,
            key.workspace_id,
            internal=True,
        ) as lease:
            record = lease.workspace.analyses.get(key.analysis_id)
            if (
                record is None
                or record.state is not AnalysisState.RUNNING
                or key.analysis_id not in lease.workspace.live_analysis_ids()
            ):
                lease.commit_requested = False
                return
            progress_key = (key.workspace_id, key.analysis_id)
            previous_live = self._live_progress.get(progress_key)
            try:
                progress = Progress.model_validate(payload)
                if progress.fraction == 1.0:
                    raise ValueError("Workers cannot report completion")
                if progress.fraction is None:
                    if previous_live is not None and previous_live.fraction is not None:
                        raise ValueError("Progress cannot return to indeterminate")
                else:
                    previous_fraction = (
                        previous_live.fraction
                        if previous_live is not None
                        else record.progress.fraction
                    )
                    if (
                        previous_fraction is not None
                        and progress.fraction < previous_fraction
                    ):
                        raise ValueError("Progress cannot decrease")
            except ValidationError, ValueError:
                logger.warning(
                    "Invalid Analysis progress analysis_id=%s user_id=%s",
                    key.analysis_id,
                    key.user_id,
                    exc_info=True,
                )
                failed = record.fail(
                    self._clock(),
                    failure=Failure(
                        code="progress_invalid",
                        message="Analysis reported invalid progress",
                    ),
                    progress=self._current_progress(key.workspace_id, record),
                )
                lease.workspace.replace_analysis(failed)
                self._live_progress.pop(progress_key, None)
                should_terminate = True
            else:
                self._live_progress[progress_key] = progress
                live_progress = progress
                lease.commit_requested = False
        if live_progress is not None:
            await self._workspaces.publish_analysis_progress(
                key.user_id,
                key.workspace_id,
                key.analysis_id,
                live_progress,
            )
        if should_terminate:
            await self._execution.cancel(key)

    async def complete_execution(
        self,
        key: AnalysisExecutionKey,
        result_payload: object,
    ) -> AnalysisState | None:
        """Commit success only while inputs and the live ownership path remain valid."""

        async with self._workspaces.mutation_context(
            key.user_id,
            key.workspace_id,
            internal=True,
        ) as lease:
            record = lease.workspace.analyses.get(key.analysis_id)
            if record is None or record.state is not AnalysisState.RUNNING:
                lease.commit_requested = False
                return None
            progress = self._current_progress(key.workspace_id, record)
            detached_root = self._detached_root_id(lease, record)
            missing = self._missing_input_ids(lease, record)
            if (
                detached_root is not None
                and record.cancellation_requested_at is not None
            ):
                terminal = record.confirm_cancelled(self._clock(), progress=progress)
            elif missing:
                terminal = record.fail(
                    self._clock(),
                    failure=Failure(
                        code="analysis_input_missing",
                        message="Analysis input is missing",
                    ),
                    progress=progress,
                )
            else:
                try:
                    publication = await self._artifacts.publish_result(
                        lease,
                        record,
                        result_payload,
                    )
                    terminal = record.succeed(
                        self._clock(),
                        result_payload=publication.payload,
                        artifact_references=publication.artifacts,
                        output_node_ids=publication.output_node_ids,
                        query_snapshot=publication.query_snapshot,
                    )
                except OSError, TypeError, ValidationError, ValueError:
                    logger.exception(
                        "Analysis result validation failed analysis_id=%s user_id=%s",
                        key.analysis_id,
                        key.user_id,
                    )
                    terminal = record.fail(
                        self._clock(),
                        failure=Failure(
                            code="analysis_execution_failed",
                            message="Analysis failed",
                        ),
                        progress=progress,
                    )
            lease.workspace.replace_analysis(terminal)
            self._live_progress.pop((key.workspace_id, key.analysis_id), None)
            self._remove_terminal_detached_tree(lease, detached_root)
            return terminal.state

    async def fail_execution(
        self,
        key: AnalysisExecutionKey,
        *,
        code: str = "analysis_execution_failed",
        message: str = "Analysis failed",
    ) -> None:
        """Persist one safe isolated failure when execution cannot complete."""

        async with self._workspaces.mutation_context(
            key.user_id,
            key.workspace_id,
            internal=True,
        ) as lease:
            record = lease.workspace.analyses.get(key.analysis_id)
            if record is None or record.state not in {
                AnalysisState.QUEUED,
                AnalysisState.RUNNING,
            }:
                lease.commit_requested = False
                return
            progress = self._current_progress(key.workspace_id, record)
            failed = record.fail(
                self._clock(),
                failure=Failure(code=code, message=message),
                progress=progress,
            )
            lease.workspace.replace_analysis(failed)
            self._live_progress.pop((key.workspace_id, key.analysis_id), None)
            self._remove_terminal_detached_tree(
                lease,
                self._detached_root_id(lease, failed),
            )

    async def confirm_cancellation(self, key: AnalysisExecutionKey) -> None:
        """Commit cancelled only after the executor confirms process exit."""

        async with self._workspaces.mutation_context(
            key.user_id,
            key.workspace_id,
            internal=True,
        ) as lease:
            record = lease.workspace.analyses.get(key.analysis_id)
            if (
                record is None
                or record.state is not AnalysisState.RUNNING
                or record.cancellation_requested_at is None
            ):
                lease.commit_requested = False
                return
            cancelled = record.confirm_cancelled(
                self._clock(),
                progress=self._current_progress(key.workspace_id, record),
            )
            lease.workspace.replace_analysis(cancelled)
            self._live_progress.pop((key.workspace_id, key.analysis_id), None)
            self._remove_terminal_detached_tree(
                lease,
                self._detached_root_id(lease, cancelled),
            )

    async def interrupt_execution(self, key: AnalysisExecutionKey) -> None:
        """Commit a confirmed graceful-shutdown interruption."""

        async with self._workspaces.mutation_context(
            key.user_id,
            key.workspace_id,
            internal=True,
        ) as lease:
            record = lease.workspace.analyses.get(key.analysis_id)
            if record is None or record.state not in {
                AnalysisState.QUEUED,
                AnalysisState.RUNNING,
            }:
                lease.commit_requested = False
                return
            progress = self._current_progress(key.workspace_id, record)
            if (
                record.state is AnalysisState.RUNNING
                and record.cancellation_requested_at is not None
            ):
                terminal = record.confirm_cancelled(self._clock(), progress=progress)
            else:
                terminal = record.fail(
                    self._clock(),
                    failure=Failure(
                        code="analysis_interrupted",
                        message="Analysis was interrupted",
                    ),
                    progress=progress,
                )
            lease.workspace.replace_analysis(terminal)
            self._live_progress.pop((key.workspace_id, key.analysis_id), None)
            self._remove_terminal_detached_tree(
                lease,
                self._detached_root_id(lease, terminal),
            )

    async def interrupt_queued_execution(self, key: AnalysisExecutionKey) -> None:
        """Fail queued work after dispatch has stopped, without touching runners."""

        async with self._workspaces.mutation_context(
            key.user_id,
            key.workspace_id,
            internal=True,
        ) as lease:
            record = lease.workspace.analyses.get(key.analysis_id)
            if record is None or record.state is not AnalysisState.QUEUED:
                lease.commit_requested = False
                return
            failed = record.fail(
                self._clock(),
                failure=Failure(
                    code="analysis_interrupted",
                    message="Analysis was interrupted",
                ),
                progress=record.progress,
            )
            lease.workspace.replace_analysis(failed)
            self._remove_terminal_detached_tree(
                lease,
                self._detached_root_id(lease, failed),
            )

    async def reconcile_interrupted_analyses(self) -> None:
        """Fail durable work that lost its private executor on restart."""

        def reconcile(workspace: Workspace) -> bool:
            changed = False
            timestamp = self._clock()
            for record in list(workspace.analyses.values()):
                if record.state not in {
                    AnalysisState.QUEUED,
                    AnalysisState.RUNNING,
                }:
                    continue
                workspace.replace_analysis(
                    record.fail(
                        timestamp,
                        failure=Failure(
                            code="analysis_interrupted",
                            message="Analysis was interrupted",
                        ),
                        progress=record.progress,
                    )
                )
                changed = True

            detached_roots = [
                analysis_id
                for analysis_id, record in workspace.analyses.items()
                if record.parent_analysis_id is None
                and workspace.analysis_tab_id(analysis_id) is None
            ]
            for root_id in detached_roots:
                root = workspace.analyses.get(root_id)
                if root is None:
                    continue
                tree = [root, *workspace.analysis_children(root_id)]
                if all(
                    record.state
                    in {
                        AnalysisState.SUCCEEDED,
                        AnalysisState.FAILED,
                        AnalysisState.CANCELLED,
                    }
                    for record in tree
                ):
                    workspace.remove_analysis(root_id)
                    changed = True
            return changed

        await self._workspaces.reconcile_durable_workspaces(reconcile)

    def stop_accepting(self) -> None:
        """Reject new submissions before scheduler shutdown begins."""

        self._accepting = False


__all__ = [
    "AnalysisService",
    "PublishedAnalysisResult",
]
