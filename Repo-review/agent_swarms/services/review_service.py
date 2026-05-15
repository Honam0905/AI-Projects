"""Review orchestration service."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
import shutil
from threading import Thread

from agent_swarms.config import Settings
from agent_swarms.graphs.review_graph import build_review_graph
from agent_swarms.observability.logging import get_logger
from agent_swarms.sandbox.service import SandboxService
from agent_swarms.services.fix_mode import apply_fix_mode
from agent_swarms.services.repo_sources import RepoSourceResolver
from agent_swarms.services.review_run_tracking import (
    ReviewEventBroker,
    ReviewRunStore,
)
from agent_swarms.services.spawn_policy import build_review_plan
from agent_swarms.state.enums import (
    ReviewIntent,
    RunEventType,
    RunStatus,
    SandboxBackend,
    WorkerRole,
)
from agent_swarms.state.schemas import (
    FixResult,
    ReviewPlan,
    ReviewReport,
    ReviewRequest,
    ReviewResponse,
    ReviewArtifactContentResponse,
    ReviewRunDetail,
    ReviewRunEvent,
    ReviewRunListResponse,
    ReviewRunSubmittedResponse,
    SandboxArtifactEntry,
    SandboxRunCreateRequest,
)

logger = get_logger(__name__)


class ReviewService:
    """Run repository reviews through the review graph and sandbox layer."""

    def __init__(
        self,
        settings: Settings,
        sandbox_service: SandboxService,
        run_store: ReviewRunStore,
        event_broker: ReviewEventBroker,
    ) -> None:
        self._settings = settings
        self._sandbox_service = sandbox_service
        self._run_store = run_store
        self._event_broker = event_broker
        self._repo_source_resolver = RepoSourceResolver(settings)
        self._graph = build_review_graph(sandbox_service, event_broker, settings)
        self._workers: dict[str, Thread] = {}

    async def review(self, request: ReviewRequest) -> ReviewResponse:
        """Run a synchronous review request and return the final report."""

        run_id, plan = await self._prepare_run(request)
        return await self._execute_review_run(run_id=run_id, request=request, plan=plan)

    async def submit_review(self, request: ReviewRequest) -> ReviewRunSubmittedResponse:
        """Create a background review run and return immediately."""

        run_id, plan = await self._prepare_run(request)
        worker = Thread(
            target=self._run_background_review,
            args=(run_id, request.model_copy(deep=True), plan.model_copy(deep=True)),
            name=f"review-run-{run_id}",
            daemon=True,
        )
        self._workers[run_id] = worker
        worker.start()
        run = self._run_store.read_run(run_id)
        return ReviewRunSubmittedResponse(
            run_id=run_id,
            status=RunStatus.PENDING,
            intent=run.intent,
            repo_path=run.repo_path,
            repo_source_type=run.repo_source_type,
            repo_source=run.repo_source,
            sandbox_backend=run.sandbox_backend,
            plan=plan,
            stream_path=f"{self._settings.api_prefix}/review/runs/{run_id}/stream",
        )

    def list_runs(self) -> ReviewRunListResponse:
        """Return the persisted review run history."""

        return ReviewRunListResponse(runs=self._run_store.list_runs())

    def get_run(self, run_id: str) -> ReviewRunDetail:
        """Return the current persisted state for a review run."""

        return self._run_store.read_run(run_id, artifacts=self._safe_list_artifacts(run_id))

    def read_artifact_content(
        self,
        run_id: str,
        artifact_path: str,
    ) -> ReviewArtifactContentResponse:
        """Return one persisted review artifact as text."""

        return ReviewArtifactContentResponse(
            run_id=run_id,
            path=artifact_path,
            content=self._sandbox_service.read_artifact_text(run_id, artifact_path),
        )

    def list_events(self, run_id: str) -> list[ReviewRunEvent]:
        """Return replayable events for a review run."""

        return self._event_broker.list_events(run_id)

    async def subscribe(self, run_id: str) -> asyncio.Queue[ReviewRunEvent]:
        """Subscribe to live events for a review run."""

        return await self._event_broker.subscribe(run_id)

    async def unsubscribe(
        self,
        run_id: str,
        queue: asyncio.Queue[ReviewRunEvent],
    ) -> None:
        """Remove a websocket subscriber queue."""

        await self._event_broker.unsubscribe(run_id, queue)

    async def _prepare_run(self, request: ReviewRequest) -> tuple[str, ReviewPlan]:
        repo_source = await self._repo_source_resolver.resolve(request)
        requested_backend = request.sandbox_backend or self._settings.sandbox_backend
        if request.intent is ReviewIntent.FIX and requested_backend is not SandboxBackend.DOCKER:
            raise ValueError("Fix mode currently requires sandbox_backend=docker.")

        plan = build_review_plan(request.user_query, intent=request.intent)
        run = await self._sandbox_service.create_run(
            SandboxRunCreateRequest(
                repo_path=repo_source.repo_path,
                sandbox_backend=request.sandbox_backend,
            )
        )
        self._run_store.create_run(
            run_id=run.run_id,
            repo_path=repo_source.repo_path,
            repo_source_type=repo_source.repo_source_type,
            repo_source=repo_source.repo_source,
            user_query=request.user_query,
            intent=request.intent,
            sandbox_backend=run.sandbox_backend,
            plan=plan,
            status=RunStatus.PENDING,
        )
        await self._event_broker.publish(
            run_id=run.run_id,
            event_type=RunEventType.RUN_CREATED,
            message="Review run created.",
            status=RunStatus.PENDING,
            payload={
                "intent": request.intent.value,
                "repo_path": repo_source.repo_path,
                "repo_source_type": repo_source.repo_source_type.value,
                "repo_source": repo_source.repo_source,
            },
        )
        await self._event_broker.publish(
            run_id=run.run_id,
            event_type=RunEventType.SANDBOX_READY,
            message="Sandbox workspace is ready.",
            status=RunStatus.PENDING,
            payload={
                "sandbox_backend": run.sandbox_backend.value,
                "workspace_path": str(run.workspace_path),
                "repo_source_path": run.repo_source_path and str(run.repo_source_path),
            },
        )
        await self._event_broker.publish(
            run_id=run.run_id,
            event_type=RunEventType.PLAN_READY,
            message="Review plan prepared.",
            status=RunStatus.PENDING,
            payload=plan.model_dump(mode="json"),
        )
        return run.run_id, plan

    def _run_background_review(
        self,
        run_id: str,
        request: ReviewRequest,
        plan: ReviewPlan,
    ) -> None:
        try:
            asyncio.run(
                self._execute_review_task(
                    run_id=run_id,
                    request=request,
                    plan=plan,
                )
            )
        finally:
            self._workers.pop(run_id, None)

    async def _execute_review_task(
        self,
        *,
        run_id: str,
        request: ReviewRequest,
        plan: ReviewPlan,
    ) -> None:
        try:
            await self._execute_review_run(run_id=run_id, request=request, plan=plan)
        except Exception:
            logger.exception("Background review run %s failed", run_id)

    async def _execute_review_run(
        self,
        *,
        run_id: str,
        request: ReviewRequest,
        plan: ReviewPlan,
    ) -> ReviewResponse:
        await self._mark_run_started(run_id)

        try:
            run = self._run_store.read_run(run_id)
            result = await self._graph.ainvoke(
                {
                    "run_id": run_id,
                    "repo_path": run.repo_path,
                    "user_query": request.user_query,
                    "sandbox_backend": request.sandbox_backend,
                    "plan": plan.model_dump(),
                }
            )
            report = ReviewReport.model_validate(result["final_report"])
            fix_result: FixResult | None = None
            if request.intent is ReviewIntent.FIX:
                report, fix_result = await self._run_fix_mode(
                    run_id=run_id,
                    report=report,
                    target_repo_path=run.repo_path,
                    user_query=request.user_query,
                )
            response = await self._finalize_success(
                run_id=run_id,
                plan=plan,
                result=result,
                report=report,
                fix_result=fix_result,
            )
            return response
        except Exception as exc:
            await self._finalize_failure(run_id=run_id, error_message=str(exc))
            raise
        finally:
            await self._close_sandbox_run(run_id)

    async def _mark_run_started(self, run_id: str) -> None:
        self._run_store.update_run(
            run_id,
            status=RunStatus.IN_PROGRESS,
            started_at=datetime.now(UTC),
            error_message=None,
        )
        await self._event_broker.publish(
            run_id=run_id,
            event_type=RunEventType.RUN_STARTED,
            message="Review run started.",
            status=RunStatus.IN_PROGRESS,
        )

    async def _finalize_success(
        self,
        *,
        run_id: str,
        plan: ReviewPlan,
        result: dict,
        report: ReviewReport,
        fix_result: FixResult | None,
    ) -> ReviewResponse:
        self._sandbox_service.write_artifact(
            run_id,
            "review-report",
            {
                "plan": result["plan"],
                "report": report.model_dump(mode="json"),
                "fix_result": fix_result and fix_result.model_dump(mode="json"),
            },
        )
        if fix_result is not None:
            self._sandbox_service.write_artifact(
                run_id,
                "fix-summary",
                fix_result.model_dump(mode="json"),
            )
        artifacts = self._safe_list_artifacts(run_id)
        report = ReviewReport.model_validate(
            {
                **report.model_dump(mode="json"),
                "artifacts": [artifact.model_dump(mode="json") for artifact in artifacts],
            }
        )
        stored_run = self._run_store.read_run(run_id)
        self._run_store.update_run(
            run_id,
            status=RunStatus.COMPLETED,
            report=report,
            fix_result=fix_result,
            artifacts=artifacts,
            completed_at=datetime.now(UTC),
            error_message=None,
        )
        await self._event_broker.publish(
            run_id=run_id,
            event_type=RunEventType.RUN_COMPLETED,
            message="Review run completed successfully.",
            status=RunStatus.COMPLETED,
            payload={
                "overall_verdict": report.overall_verdict.value,
                "artifacts_count": len(artifacts),
            },
        )
        return ReviewResponse(
            run_id=run_id,
            status=RunStatus.COMPLETED,
            intent=stored_run.intent,
            repo_path=stored_run.repo_path,
            repo_source_type=stored_run.repo_source_type,
            repo_source=stored_run.repo_source,
            sandbox_backend=stored_run.sandbox_backend,
            plan=plan,
            report=report,
            fix_result=fix_result,
        )

    async def _finalize_failure(self, *, run_id: str, error_message: str) -> None:
        self._run_store.update_run(
            run_id,
            status=RunStatus.FAILED,
            completed_at=datetime.now(UTC),
            error_message=error_message,
            artifacts=self._safe_list_artifacts(run_id),
        )
        await self._event_broker.publish(
            run_id=run_id,
            event_type=RunEventType.RUN_FAILED,
            message="Review run failed.",
            status=RunStatus.FAILED,
            payload={"error_message": error_message},
        )

    async def _run_fix_mode(
        self,
        *,
        run_id: str,
        report: ReviewReport,
        target_repo_path: str,
        user_query: str,
    ) -> tuple[ReviewReport, FixResult]:
        await self._event_broker.publish(
            run_id=run_id,
            event_type=RunEventType.FIX_STARTED,
            message="Automatic fix mode started in Docker sandbox.",
            status=RunStatus.IN_PROGRESS,
            payload={"worker_role": WorkerRole.DOCS_DEVEX_REVIEWER.value},
        )

        execution = await apply_fix_mode(
            run_id=run_id,
            sandbox_service=self._sandbox_service,
            report=report,
            user_query=user_query,
            settings=self._settings,
        )
        apply_back_target = Path(target_repo_path).expanduser().resolve()
        applied_to_local_repo = False
        apply_back_error: str | None = None
        if execution.fix_result.applied and execution.fix_result.changed_files:
            try:
                self._apply_back_to_local_repo(
                    run_id=run_id,
                    target_root=apply_back_target,
                    changed_files=execution.fix_result.changed_files,
                )
                applied_to_local_repo = True
            except Exception as exc:
                apply_back_error = str(exc)

        fix_result = execution.fix_result.model_copy(
            update={
                "applied_to_local_repo": applied_to_local_repo,
                "apply_back_target": str(apply_back_target),
                "apply_back_error": apply_back_error,
            }
        )

        for changed_file in fix_result.changed_files:
            await self._event_broker.publish(
                run_id=run_id,
                event_type=RunEventType.FILE_UPDATED,
                message=f"Updated {changed_file}.",
                status=RunStatus.IN_PROGRESS,
                payload={
                    "path": changed_file,
                    "worker_role": WorkerRole.DOCS_DEVEX_REVIEWER.value,
                },
            )

        await self._event_broker.publish(
            run_id=run_id,
            event_type=RunEventType.FIX_COMPLETED,
            message=fix_result.summary,
            status=RunStatus.IN_PROGRESS,
            payload={
                "applied": fix_result.applied,
                "changed_files": fix_result.changed_files,
                "diff_artifact_path": fix_result.diff_artifact_path,
                "applied_to_local_repo": fix_result.applied_to_local_repo,
                "apply_back_target": fix_result.apply_back_target,
                "apply_back_error": fix_result.apply_back_error,
                "worker_role": WorkerRole.DOCS_DEVEX_REVIEWER.value,
            },
        )
        return execution.report, fix_result

    async def _close_sandbox_run(self, run_id: str) -> None:
        try:
            await self._sandbox_service.close_run(run_id)
        except FileNotFoundError:
            return

    def _safe_list_artifacts(self, run_id: str) -> list[SandboxArtifactEntry]:
        try:
            return self._sandbox_service.list_artifacts(run_id).artifacts
        except FileNotFoundError:
            return []

    def _apply_back_to_local_repo(
        self,
        *,
        run_id: str,
        target_root: Path,
        changed_files: list[str],
    ) -> None:
        workspace_root = self._sandbox_service.get_workspace_path(run_id).resolve()
        resolved_target_root = target_root.resolve()
        for relative_path in changed_files:
            source_path = (workspace_root / relative_path).resolve()
            destination_path = (resolved_target_root / relative_path).resolve()
            if not source_path.is_relative_to(workspace_root):
                raise ValueError(
                    "Refusing to read sandbox path outside the workspace: "
                    f"{relative_path}"
                )
            if not destination_path.is_relative_to(resolved_target_root):
                raise ValueError(
                    "Refusing to write outside the target repository: "
                    f"{relative_path}"
                )
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
