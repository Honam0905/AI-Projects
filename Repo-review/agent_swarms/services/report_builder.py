"""Review report assembly."""

from agent_swarms.services.scoring import determine_verdict, sort_findings
from agent_swarms.state.enums import FindingSeverity
from agent_swarms.state.schemas import (
    RepoMapSummary,
    ReviewFinding,
    ReviewReport,
    ReviewWorkerSummary,
    SandboxArtifactEntry,
)


def build_review_report(
    *,
    repo_map: RepoMapSummary,
    findings: list[ReviewFinding],
    worker_summaries: list[ReviewWorkerSummary],
    commands_run: list[str],
    artifacts: list[SandboxArtifactEntry] | None = None,
) -> ReviewReport:
    """Assemble the final structured review report."""

    sorted_findings = sort_findings(findings)
    verdict = determine_verdict(sorted_findings)
    artifacts = artifacts or []

    return ReviewReport(
        overall_verdict=verdict,
        summary=_build_summary(repo_map, sorted_findings),
        repo_map=repo_map,
        top_findings=sorted_findings[:10],
        worker_summaries=worker_summaries,
        commands_run=_unique(commands_run),
        artifacts=artifacts,
    )


def _build_summary(repo_map: RepoMapSummary, findings: list[ReviewFinding]) -> str:
    if not findings:
        return (
            "The review completed without material findings. "
            f"Detected languages: {', '.join(repo_map.languages) or 'unknown'}."
        )

    high_count = sum(
        1
        for finding in findings
        if finding.severity in {FindingSeverity.HIGH, FindingSeverity.CRITICAL}
    )
    medium_count = sum(
        1
        for finding in findings
        if finding.severity is FindingSeverity.MEDIUM
    )
    low_count = sum(1 for finding in findings if finding.severity is FindingSeverity.LOW)

    return (
        f"Review completed with {len(findings)} findings "
        f"({high_count} high/critical, {medium_count} medium, {low_count} low). "
        f"Detected languages: {', '.join(repo_map.languages) or 'unknown'}."
    )


def _unique(items: list[str]) -> list[str]:
    seen = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered
