import type {
  AgentSnapshot,
  AgentState,
  AgentStatus,
  RepoSourceType,
  ReviewIntent,
  ReviewPlanSummary,
  ReviewRunEvent,
  WorkerRole,
} from '../types'
import type { ReviewRunDetailResponse } from './api'

type ReviewFinding = NonNullable<ReviewRunDetailResponse['report']>['top_findings'][number]

export const TOTAL_PROGRESS_STEPS = 14

const WORKER_META: Record<
  WorkerRole,
  {
    name: string
    task: string
    idleLabel: string
    activeLabel: string
    finalLabel: string
  }
> = {
  repo_mapper: {
    name: 'Mapper',
    task: 'Map repository structure and surface key entry points.',
    idleLabel: 'Queued',
    activeLabel: 'Mapping',
    finalLabel: 'Mapped',
  },
  static_reviewer: {
    name: 'Static',
    task: 'Inspect code structure, TODOs, and maintainability risks.',
    idleLabel: 'Queued',
    activeLabel: 'Reviewing',
    finalLabel: 'Reviewed',
  },
  runtime_tester: {
    name: 'Runtime',
    task: 'Run build, compile, and test checks inside the sandbox.',
    idleLabel: 'Queued',
    activeLabel: 'Executing',
    finalLabel: 'Verified',
  },
  security_reviewer: {
    name: 'Security',
    task: 'Scan for credentials, unsafe execution, and risky patterns.',
    idleLabel: 'Queued',
    activeLabel: 'Scanning',
    finalLabel: 'Secured',
  },
  docs_devex_reviewer: {
    name: 'Docs',
    task: 'Check README quality, onboarding flow, and developer experience.',
    idleLabel: 'Queued',
    activeLabel: 'Checking',
    finalLabel: 'Documented',
  },
  external_validator: {
    name: 'Validator',
    task: 'Cross-check dependencies, validation signals, and external-facing readiness.',
    idleLabel: 'Queued',
    activeLabel: 'Validating',
    finalLabel: 'Validated',
  },
}

export function getWorkerMeta(role: string, fallbackIndex = 0) {
  const typedRole = role as WorkerRole
  const meta = WORKER_META[typedRole]
  if (meta) {
    return meta
  }

  return {
    name: `Agent ${String(fallbackIndex + 1).padStart(2, '0')}`,
    task: 'Handle a delegated task from the supervisor.',
    idleLabel: 'Queued',
    activeLabel: 'Working',
    finalLabel: 'Completed',
  }
}

export function createAgentsFromWorkers(selectedWorkers: string[]): AgentState[] {
  return selectedWorkers.map((role, index) => {
    const meta = getWorkerMeta(role, index)
    return {
      id: role,
      role,
      name: meta.name,
      task: meta.task,
      status: 'idle',
      progress: 0,
      totalSteps: TOTAL_PROGRESS_STEPS,
      logs: [],
    }
  })
}

function summarizeWorkerNames(selectedWorkers: string[]): string {
  return selectedWorkers
    .map((role, index) => getWorkerMeta(role, index).name)
    .join(', ')
}

export function buildChatThinkingSteps(message: string): string[] {
  return [
    `Analyzing user query: "${truncate(message)}"`,
    'Intent detected: direct supervisor conversation.',
    'Decision: answer inline without spawning sub-agents.',
    'Sandbox tools are not required for this request.',
  ]
}

export function buildReviewThinkingSteps(
  message: string,
  plan: ReviewPlanSummary,
  repoTarget?: string,
  intent: ReviewIntent = 'review',
): string[] {
  const workerNames = summarizeWorkerNames(plan.selectedWorkers)

  return [
    `Analyzing user query: "${truncate(message)}"`,
    intent === 'fix'
      ? 'Intent detected: repository fix request with safe sandbox edits.'
      : 'Intent detected: repository review or tool-backed request.',
    ...(repoTarget ? [`Repository target: ${truncate(repoTarget, 88)}`] : []),
    `Route reason: ${plan.routeReason}`,
    `Decision: spawn ${plan.spawnCount} specialized sub-agent${plan.spawnCount === 1 ? '' : 's'}.`,
    ...plan.selectedWorkers.map((role, index) => {
      const meta = getWorkerMeta(role, index)
      return `Agent ${String(index + 1).padStart(2, '0')} (${meta.name}): ${meta.task}`
    }),
    ...(intent === 'fix'
      ? ['After the review, prepare supported fixes and capture a patch diff.']
      : []),
    `Dispatching swarm: ${workerNames}.`,
  ]
}

export function buildPendingAssistantMessage(
  plan: ReviewPlanSummary,
  repoTarget?: string,
  intent: ReviewIntent = 'review',
): string {
  const targetDetail = repoTarget
    ? ` Target: ${truncate(repoTarget, 88)}.`
    : ''
  const actionLabel =
    intent === 'fix'
      ? 'review and safe automatic fixes'
      : 'review'
  return `I started ${plan.spawnCount} worker${plan.spawnCount === 1 ? '' : 's'} for ${actionLabel}.${targetDetail} Watch the live work in the Agent's Computer panel.`
}

export function getAgentStatusLabel(agent: {
  role: string
  status: AgentStatus
  progress: number
  totalSteps: number
}): string {
  const meta = getWorkerMeta(agent.role)
  if (agent.status === 'failed') {
    return 'Failed'
  }
  if (agent.status === 'completed') {
    return meta.finalLabel
  }
  if (agent.status === 'idle') {
    return meta.idleLabel
  }
  if (agent.progress >= agent.totalSteps - 2) {
    return 'Finalizing'
  }
  return meta.activeLabel
}

export function nextProgressFromLogs(
  agent: Pick<AgentState, 'logs' | 'totalSteps' | 'status'>,
): number {
  if (agent.status === 'completed') {
    return agent.totalSteps
  }
  if (agent.status === 'failed') {
    return Math.max(1, Math.min(agent.totalSteps, agent.logs.length + 1))
  }

  const tentative = 2 + agent.logs.length * 2
  return Math.min(agent.totalSteps - 1, tentative)
}

export function formatWorkerLog(event: ReviewRunEvent): string | null {
  const workerRole = typeof event.payload.worker_role === 'string'
    ? event.payload.worker_role
    : null
  const agentMessage = typeof event.payload.agent_message === 'string'
    ? event.payload.agent_message
    : null

  switch (event.event_type) {
    case 'worker_started':
      return `${timestampPrefix(event.timestamp)} ${agentMessage || `${workerRole || 'worker'} started`}`
    case 'tool_completed': {
      const kind = typeof event.payload.kind === 'string' ? event.payload.kind : 'tool'
      const toolName = typeof event.payload.tool_name === 'string'
        ? event.payload.tool_name
        : kind
      const command = typeof event.payload.command === 'string'
        ? event.payload.command
        : event.message
      const exitCode =
        typeof event.payload.exit_code === 'number' ? event.payload.exit_code : 0
      const duration =
        typeof event.payload.duration_ms === 'number'
          ? `${event.payload.duration_ms}ms`
          : 'unknown time'
      return `${timestampPrefix(event.timestamp)} ${toolName} > ${command} [exit ${exitCode}, ${duration}]`
    }
    case 'worker_completed':
      if (agentMessage) {
        return `${timestampPrefix(event.timestamp)} ${agentMessage}`
      }
      if (workerRole === 'repo_mapper') {
        const mappedFileCount =
          typeof event.payload.mapped_file_count === 'number'
            ? event.payload.mapped_file_count
            : 0
        const languages = Array.isArray(event.payload.languages)
          ? event.payload.languages.filter((value): value is string => typeof value === 'string')
          : []
        const scanStderr =
          typeof event.payload.scan_stderr === 'string' ? event.payload.scan_stderr : null
        const languageLabel = languages.length > 0 ? languages.join(', ') : 'none'
        const stderrLabel = scanStderr ? ` stderr: ${truncate(scanStderr, 90)}` : ''
        return `${timestampPrefix(event.timestamp)} mapped ${mappedFileCount} files, languages: ${languageLabel}.${stderrLabel}`
      }
      return `${timestampPrefix(event.timestamp)} task completed`
    case 'fix_started':
      return `${timestampPrefix(event.timestamp)} fix mode started`
    case 'file_updated': {
      const path = typeof event.payload.path === 'string' ? event.payload.path : 'workspace file'
      return `${timestampPrefix(event.timestamp)} wrote ${path}`
    }
    case 'fix_completed':
      if (typeof event.payload.apply_back_error === 'string' && event.payload.apply_back_error) {
        return `${timestampPrefix(event.timestamp)} fix mode completed, but apply-back failed: ${truncate(event.payload.apply_back_error, 72)}`
      }
      if (typeof event.payload.apply_back_target === 'string' && event.payload.apply_back_target) {
        return `${timestampPrefix(event.timestamp)} fix mode completed and copied changes to ${truncate(event.payload.apply_back_target, 56)}`
      }
      return `${timestampPrefix(event.timestamp)} fix mode completed`
    case 'run_failed':
      return `${timestampPrefix(event.timestamp)} run failed`
    default:
      return null
  }
}

export function buildAgentWorkResult(
  detail: ReviewRunDetailResponse,
  query: string,
  agents: AgentSnapshot[],
): {
  answer: string
  summary: string
  findingsCount: number
  overallVerdict?: string
  fixResult?: {
    applied: boolean
    summary: string
    changedFiles: string[]
    diffArtifactPath?: string | null
    appliedToLocalRepo?: boolean
    applyBackTarget?: string | null
    applyBackError?: string | null
  }
} {
  void query
  if (!detail.report) {
    return {
      answer: [
        detail.intent === 'fix'
          ? 'I started fix mode, but the run ended before I could finish.'
          : 'I started the review, but the run ended before I could finish.',
        detail.error_message || 'The final report was not produced.',
        '',
        'Open "View Agent Work" if you want the raw run details.',
      ]
        .filter(Boolean)
        .join('\n'),
      summary: detail.error_message || 'No report was generated.',
      findingsCount: 0,
    }
  }

  const findings = detail.report.top_findings
  const repoMap = detail.report.repo_map
  const blockingFindings = findings.filter((finding) => isBlockingSeverity(finding.severity))
  const importantFindings = findings.filter((finding) => finding.severity === 'medium')
  const minorFindings = findings.filter((finding) => finding.severity === 'low')
  const focusFindings =
    blockingFindings.length > 0
      ? blockingFindings
      : importantFindings.length > 0
        ? importantFindings
        : minorFindings
  const answerLines: string[] = [
    `Verdict: ${buildVerdictLine(detail)}`,
  ]

  if (detail.repo_source) {
    answerLines.push(`Target: ${detail.repo_source}`)
  }

  if (detail.intent === 'fix' && detail.fix_result?.applied) {
    answerLines.push('')
    answerLines.push('Fix result:')
    if (detail.fix_result.applied_to_local_repo) {
      answerLines.push(
        `- Updated local repo: ${detail.fix_result.apply_back_target || detail.repo_path}`,
      )
    } else if (detail.fix_result.apply_back_error) {
      answerLines.push(`- Local repo was not updated: ${detail.fix_result.apply_back_error}`)
    } else {
      answerLines.push('- Changes were prepared in the sandbox only.')
    }
    if (detail.fix_result.changed_files.length > 0) {
      answerLines.push(`- Changed files: ${formatCollection(detail.fix_result.changed_files, 3)}`)
    }
  }

  if (focusFindings.length > 0) {
    answerLines.push('')
    answerLines.push(blockingFindings.length > 0 ? 'Main things to fix:' : 'Things to improve:')
    focusFindings.slice(0, 3).forEach((finding) => {
      answerLines.push(`- ${formatFindingSummary(finding)}`)
    })
  } else {
    answerLines.push('')
    answerLines.push('Main things to fix:')
    answerLines.push('- None surfaced in this pass.')
  }

  const nextSteps = buildNextSteps(detail, findings)
  if (nextSteps.length > 0) {
    answerLines.push('')
    answerLines.push('Next steps:')
    nextSteps.forEach((step, index) => {
      answerLines.push(`${index + 1}. ${step}`)
    })
  }

  answerLines.push('')
  answerLines.push(
    `Scanned: ${repoMap.mapped_file_count} files across ${formatCollection(repoMap.languages, 4)}.`,
  )
  if (repoMap.scan_stderr) {
    answerLines.push(`Mapper warning: ${truncate(repoMap.scan_stderr, 120)}`)
  }
  answerLines.push('Open "View Agent Work" for evidence, commands, and patch diff.')

  return {
    answer: answerLines.join('\n'),
    summary:
      findings.length > 0
        ? buildVerdictLine(detail)
        : detail.intent === 'fix' && detail.fix_result?.applied
          ? 'Fixes applied'
          : 'No blocking issues found',
    findingsCount: findings.length,
    overallVerdict: detail.report.overall_verdict,
    fixResult: detail.fix_result
      ? {
          applied: detail.fix_result.applied,
          summary: detail.fix_result.summary,
          changedFiles: detail.fix_result.changed_files,
          diffArtifactPath: detail.fix_result.diff_artifact_path,
          appliedToLocalRepo: detail.fix_result.applied_to_local_repo,
          applyBackTarget: detail.fix_result.apply_back_target,
          applyBackError: detail.fix_result.apply_back_error,
        }
      : undefined,
  }
}

export function formatRepoTargetLabel(
  repoSourceType: RepoSourceType | undefined,
  repoSource: string | undefined,
  repoPath?: string,
): string | undefined {
  if (repoSourceType === 'remote' && repoSource) {
    if (repoSource.includes('/pull/')) {
      return `GitHub PR ${repoSource}`
    }
    return `GitHub repo ${repoSource}`
  }
  if (repoSource) {
    return `Local repo ${repoSource}`
  }
  if (repoPath) {
    return `Local repo ${repoPath}`
  }
  return undefined
}

function formatCollection(values: string[], limit = 3): string {
  if (values.length === 0) {
    return 'none'
  }
  const visible = values.slice(0, limit).join(', ')
  if (values.length <= limit) {
    return visible
  }
  return `${visible}, +${values.length - limit} more`
}

function formatFindingSummary(
  finding: ReviewFinding,
): string {
  const location = finding.file_paths[0]
  const cleanSummary = finding.summary.replace(/`/g, '').trim()
  const baseSummary = location && !cleanSummary.includes(location)
    ? `${cleanSummary} (${location})`
    : cleanSummary
  const trimmedSummary = baseSummary.length > 132
    ? `${baseSummary.slice(0, 129)}...`
    : baseSummary
  return `${capitalize(finding.severity)}: ${trimmedSummary}`
}

function truncate(value: string, limit = 72): string {
  if (value.length <= limit) {
    return value
  }
  return `${value.slice(0, limit - 3)}...`
}

function capitalize(value: string): string {
  if (!value) {
    return value
  }
  return `${value[0].toUpperCase()}${value.slice(1)}`
}

function buildVerdictLine(detail: ReviewRunDetailResponse): string {
  if (detail.intent === 'fix' && detail.fix_result?.applied) {
    if (detail.fix_result.applied_to_local_repo) {
      return detail.report?.top_findings.length
        ? 'I updated the local repo, but a few issues still need attention.'
        : 'I updated the local repo. Review the patch, then run your checks.'
    }
    if (detail.fix_result.apply_back_error) {
      return 'Fixes were prepared in the sandbox, but they were not copied back to the local repo.'
    }
  }

  if (detail.report?.overall_verdict === 'green') {
    return 'Looks ready to share based on this pass.'
  }
  if (detail.report?.overall_verdict === 'red') {
    return 'Not ready to publish yet.'
  }
  return 'Almost ready, but fix a few items first.'
}

function buildNextSteps(
  detail: ReviewRunDetailResponse,
  findings: ReviewFinding[],
): string[] {
  const steps: string[] = []
  const seen = new Set<string>()

  if (detail.intent === 'fix' && detail.fix_result?.applied) {
    pushUnique(
      steps,
      seen,
      detail.fix_result.applied_to_local_repo
        ? 'Review the changed files before committing.'
        : 'Review the generated patch before applying it to your real repo.',
    )
    pushUnique(steps, seen, 'Run the relevant tests and a quick manual check.')
  }

  findings.forEach((finding) => {
    const suggestion = normalizeSuggestedFix(finding)
    if (suggestion) {
      pushUnique(steps, seen, suggestion)
    }
  })

  if (steps.length === 0) {
    pushUnique(steps, seen, 'Do one final local verification pass before publishing.')
  }
  return steps.slice(0, 3)
}

function normalizeSuggestedFix(
  finding: ReviewFinding,
): string | null {
  const text = `${finding.title} ${finding.summary}`.toLowerCase()
  const paths = finding.file_paths.join(' ').toLowerCase()
  if (text.includes('secret')) {
    if (paths.includes('.env')) {
      return 'Keep real keys out of git, rotate anything that was shared, and use .env.example for placeholders.'
    }
    return 'Remove secret-like values from the repo and rotate anything that may have been exposed.'
  }
  if (text.includes('readme')) {
    return 'Make the README clear enough for a new developer to install, run, and test the project.'
  }
  if (text.includes('.env.example') || text.includes('environment example')) {
    return 'Add a safe .env.example with placeholder values only.'
  }
  if (text.includes('test')) {
    return 'Add or document the main test command before publishing.'
  }
  if (text.includes('unsafe') || text.includes('deserialization') || text.includes('eval')) {
    return 'Replace risky eval, exec, or unsafe loading code with a safer path.'
  }
  if (text.includes('large source file')) {
    return 'Split oversized files into smaller modules by responsibility.'
  }
  if (finding.suggested_fix) {
    return tidySentence(finding.suggested_fix)
  }
  return null
}

function pushUnique(items: string[], seen: Set<string>, value: string) {
  const normalized = tidySentence(value)
  if (!normalized || seen.has(normalized)) {
    return
  }
  seen.add(normalized)
  items.push(normalized)
}

function tidySentence(value: string): string {
  const cleaned = value.replace(/\s+/g, ' ').trim().replace(/[.]+$/, '')
  if (!cleaned) {
    return cleaned
  }
  return `${cleaned}.`
}

function isBlockingSeverity(value: string): boolean {
  return value === 'high' || value === 'critical'
}

function timestampPrefix(timestamp: string): string {
  try {
    const date = new Date(timestamp)
    return `[${date.toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    })}]`
  } catch {
    return '[--:--:--]'
  }
}
