"""Persistent review run tracking and event streaming support."""

from __future__ import annotations

import asyncio
import json
import threading
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_swarms.state.enums import (
    RepoSourceType,
    ReviewIntent,
    RunEventType,
    RunStatus,
    SandboxBackend,
)
from agent_swarms.state.schemas import (
    FixResult,
    ReviewPlan,
    ReviewReport,
    ReviewRunDetail,
    ReviewRunEvent,
    ReviewRunSummary,
    SandboxArtifactEntry,
)

TERMINAL_RUN_STATUSES = frozenset({RunStatus.COMPLETED, RunStatus.FAILED})


class ReviewRunStore:
    """Persist review run state and replayable events to disk."""

    def __init__(self, data_dir: Path) -> None:
        self._runs_dir = data_dir / "runs"
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def create_run(
        self,
        *,
        run_id: str,
        repo_path: str,
        repo_source_type: RepoSourceType,
        repo_source: str,
        user_query: str,
        intent: ReviewIntent,
        sandbox_backend: SandboxBackend,
        plan: ReviewPlan,
        status: RunStatus,
    ) -> ReviewRunDetail:
        """Create and persist a review run record."""

        now = datetime.now(UTC)
        run = ReviewRunDetail(
            run_id=run_id,
            status=status,
            intent=intent,
            repo_path=repo_path,
            repo_source_type=repo_source_type,
            repo_source=repo_source,
            user_query=user_query,
            sandbox_backend=sandbox_backend,
            plan=plan,
            created_at=now,
            updated_at=now,
            started_at=now if status is RunStatus.IN_PROGRESS else None,
        )
        self._write_run(run)
        return run

    def update_run(self, run_id: str, **changes: Any) -> ReviewRunDetail:
        """Apply partial updates to a persisted review run."""

        with self._lock:
            run = self.read_run(run_id)
            payload = run.model_dump(mode="json")

            for key, value in changes.items():
                if isinstance(value, ReviewPlan | ReviewReport | FixResult):
                    payload[key] = value.model_dump(mode="json")
                elif (
                    isinstance(value, list)
                    and value
                    and isinstance(value[0], SandboxArtifactEntry)
                ):
                    payload[key] = [item.model_dump(mode="json") for item in value]
                else:
                    payload[key] = value

            payload["updated_at"] = datetime.now(UTC).isoformat()
            updated_run = ReviewRunDetail.model_validate(payload)
            self._write_run(updated_run)
            return updated_run

    def read_run(
        self,
        run_id: str,
        *,
        artifacts: list[SandboxArtifactEntry] | None = None,
    ) -> ReviewRunDetail:
        """Read one persisted review run."""

        path = self._run_file(run_id)
        if not path.exists():
            raise FileNotFoundError(f"Review run '{run_id}' was not found.")

        payload = self._normalize_payload(json.loads(path.read_text()))
        if artifacts is not None:
            payload["artifacts"] = [
                artifact.model_dump(mode="json") for artifact in artifacts
            ]
        return ReviewRunDetail.model_validate(payload)

    def list_runs(self) -> list[ReviewRunSummary]:
        """Return saved review runs ordered by most recent first."""

        runs: list[ReviewRunSummary] = []
        for path in sorted(self._runs_dir.glob("*/review-run.json"), reverse=True):
            payload = self._normalize_payload(json.loads(path.read_text()))
            runs.append(ReviewRunSummary.model_validate(payload))

        return sorted(runs, key=lambda item: item.created_at, reverse=True)

    def run_exists(self, run_id: str) -> bool:
        """Return whether a persisted review run exists."""

        return self._run_file(run_id).exists()

    def append_event(self, event: ReviewRunEvent) -> None:
        """Append one replayable lifecycle event."""

        with self._lock:
            path = self._events_file(event.run_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.model_dump(mode="json"), sort_keys=True))
                handle.write("\n")

    def list_events(self, run_id: str) -> list[ReviewRunEvent]:
        """Return replayable lifecycle events for a run."""

        path = self._events_file(run_id)
        if not path.exists():
            if not self.run_exists(run_id):
                raise FileNotFoundError(f"Review run '{run_id}' was not found.")
            return []

        return [
            ReviewRunEvent.model_validate(json.loads(line))
            for line in path.read_text().splitlines()
            if line.strip()
        ]

    def _run_file(self, run_id: str) -> Path:
        return self._runs_dir / run_id / "review-run.json"

    def _events_file(self, run_id: str) -> Path:
        return self._runs_dir / run_id / "review-events.jsonl"

    def _write_run(self, run: ReviewRunDetail) -> None:
        path = self._run_file(run.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run.model_dump(mode="json"), indent=2, sort_keys=True))

    @staticmethod
    def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
        if "intent" not in payload:
            payload["intent"] = "review"
        if "repo_source_type" not in payload:
            payload["repo_source_type"] = RepoSourceType.LOCAL.value
        if "repo_source" not in payload:
            payload["repo_source"] = payload.get("repo_path", "")
        return payload


class ReviewEventBroker:
    """Fan out persisted review events to websocket subscribers."""

    def __init__(self, store: ReviewRunStore) -> None:
        self._store = store
        self._subscribers: dict[
            str,
            set[tuple[asyncio.AbstractEventLoop, asyncio.Queue[ReviewRunEvent]]],
        ] = defaultdict(set)
        self._lock = threading.Lock()

    async def publish(
        self,
        *,
        run_id: str,
        event_type: RunEventType,
        message: str,
        status: RunStatus | None = None,
        payload: dict[str, Any] | None = None,
    ) -> ReviewRunEvent:
        """Persist an event and broadcast it to active subscribers."""

        event = ReviewRunEvent(
            event_id=uuid4().hex,
            run_id=run_id,
            event_type=event_type,
            message=message,
            timestamp=datetime.now(UTC),
            status=status,
            payload=payload or {},
        )
        self._store.append_event(event)

        with self._lock:
            subscribers = list(self._subscribers.get(run_id, set()))

        for loop, queue in subscribers:
            loop.call_soon_threadsafe(queue.put_nowait, event)
        return event

    async def subscribe(self, run_id: str) -> asyncio.Queue[ReviewRunEvent]:
        """Subscribe to future events for an existing run."""

        if not self._store.run_exists(run_id):
            raise FileNotFoundError(f"Review run '{run_id}' was not found.")

        queue: asyncio.Queue[ReviewRunEvent] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        with self._lock:
            self._subscribers[run_id].add((loop, queue))
        return queue

    async def unsubscribe(self, run_id: str, queue: asyncio.Queue[ReviewRunEvent]) -> None:
        """Detach an event subscriber queue."""

        with self._lock:
            subscribers = self._subscribers.get(run_id)
            if subscribers is None:
                return
            subscribers = {item for item in subscribers if item[1] is not queue}
            if subscribers:
                self._subscribers[run_id] = subscribers
                return
            if not subscribers:
                self._subscribers.pop(run_id, None)

    def list_events(self, run_id: str) -> list[ReviewRunEvent]:
        """Return persisted events for one run."""

        return self._store.list_events(run_id)
