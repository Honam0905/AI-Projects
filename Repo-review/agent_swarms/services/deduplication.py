"""Finding deduplication helpers."""

from agent_swarms.state.enums import FindingSeverity, WorkerRole
from agent_swarms.state.schemas import ReviewFinding

SEVERITY_RANK = {
    FindingSeverity.LOW: 1,
    FindingSeverity.MEDIUM: 2,
    FindingSeverity.HIGH: 3,
    FindingSeverity.CRITICAL: 4,
}


def deduplicate_findings(findings: list[ReviewFinding]) -> list[ReviewFinding]:
    """Merge duplicate findings while preserving the strongest signal."""

    merged: dict[tuple[str, tuple[str, ...]], ReviewFinding] = {}

    for finding in findings:
        key = (
            finding.title.strip().lower(),
            tuple(sorted(set(finding.file_paths))),
        )

        existing = merged.get(key)
        if existing is None:
            merged[key] = finding
            continue

        if SEVERITY_RANK[finding.severity] > SEVERITY_RANK[existing.severity]:
            severity = finding.severity
        else:
            severity = existing.severity

        merged[key] = ReviewFinding(
            title=existing.title,
            summary=(
                existing.summary
                if len(existing.summary) >= len(finding.summary)
                else finding.summary
            ),
            severity=severity,
            confidence=max(existing.confidence, finding.confidence),
            source_agents=_merge_roles(existing.source_agents, finding.source_agents),
            evidence=_merge_text(existing.evidence, finding.evidence),
            file_paths=sorted(set(existing.file_paths + finding.file_paths)),
            reproduction_steps=_merge_text(
                existing.reproduction_steps,
                finding.reproduction_steps,
            ),
            suggested_fix=existing.suggested_fix or finding.suggested_fix,
        )

    return list(merged.values())


def _merge_roles(left: list[WorkerRole], right: list[WorkerRole]) -> list[WorkerRole]:
    seen = set()
    merged: list[WorkerRole] = []
    for role in left + right:
        if role in seen:
            continue
        seen.add(role)
        merged.append(role)
    return merged


def _merge_text(left: list[str], right: list[str]) -> list[str]:
    seen = set()
    merged: list[str] = []
    for item in left + right:
        if item in seen:
            continue
        seen.add(item)
        merged.append(item)
    return merged
