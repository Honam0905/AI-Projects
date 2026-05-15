"""Finding ranking and overall verdict helpers."""

from agent_swarms.state.enums import FindingSeverity, ReviewVerdict
from agent_swarms.state.schemas import ReviewFinding

SEVERITY_RANK = {
    FindingSeverity.CRITICAL: 4,
    FindingSeverity.HIGH: 3,
    FindingSeverity.MEDIUM: 2,
    FindingSeverity.LOW: 1,
}


def sort_findings(findings: list[ReviewFinding]) -> list[ReviewFinding]:
    """Sort findings by severity, confidence, and title."""

    return sorted(
        findings,
        key=lambda finding: (
            -SEVERITY_RANK[finding.severity],
            -finding.confidence,
            finding.title.lower(),
        ),
    )


def determine_verdict(findings: list[ReviewFinding]) -> ReviewVerdict:
    """Compute the overall verdict for a review run."""

    severities = {finding.severity for finding in findings}
    if FindingSeverity.CRITICAL in severities or FindingSeverity.HIGH in severities:
        return ReviewVerdict.RED
    if FindingSeverity.MEDIUM in severities:
        return ReviewVerdict.CAUTION
    return ReviewVerdict.GREEN
