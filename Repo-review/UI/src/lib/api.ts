import type {
  ReviewIntent,
  RepoSourceType,
  ReviewRunEvent,
  SandboxBackend,
  RunMode,
  RunStatus,
  WorkerRole,
} from '../types'

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '')
const DEFAULT_REPO_PATH = import.meta.env.VITE_DEFAULT_REPO_PATH?.trim() || undefined

export interface ChatApiResponse {
  request_id: string
  status: RunStatus
  provider: 'mock' | 'nvidia'
  plan: {
    mode: RunMode
    spawn_count: number
    selected_workers: string[]
    needs_sandbox: boolean
    route_reason: string
  }
  response: string
}

export interface ReviewRunSubmittedResponse {
  run_id: string
  status: RunStatus
  intent: ReviewIntent
  repo_path: string
  repo_source_type: RepoSourceType
  repo_source: string
  sandbox_backend: SandboxBackend
  plan: {
    intent: string
    needs_repo: boolean
    needs_sandbox: boolean
    spawn_count: number
    selected_workers: WorkerRole[]
    route_reason: string
    expected_outputs: string[]
  }
  stream_path: string
}

export interface ReviewRunDetailResponse {
  run_id: string
  mode: RunMode
  status: RunStatus
  intent: ReviewIntent
  repo_path: string
  repo_source_type: RepoSourceType
  repo_source: string
  user_query: string
  sandbox_backend: string
  plan: ReviewRunSubmittedResponse['plan'] | null
  created_at: string
  updated_at: string
  started_at: string | null
  completed_at: string | null
  error_message: string | null
  fix_result: {
    applied: boolean
    summary: string
    changed_files: string[]
    diff_artifact_path: string | null
    fixed_finding_titles: string[]
    unsupported_finding_titles: string[]
    applied_to_local_repo: boolean
    apply_back_target: string | null
    apply_back_error: string | null
  } | null
  report: {
    overall_verdict: string
    summary: string
    repo_map: {
      mapped_file_count: number
      languages: string[]
      test_locations: string[]
      docs_files: string[]
      entry_points: string[]
      top_level_files: string[]
      dependency_files: string[]
      scan_stderr: string | null
    }
    top_findings: Array<{
      title: string
      summary: string
      severity: string
      confidence: number
      source_agents: string[]
      file_paths: string[]
      evidence: string[]
      suggested_fix: string | null
    }>
    worker_summaries: Array<{
      agent_name: string
      summary: string
      commands_run: string[]
      artifacts: string[]
    }>
    commands_run: string[]
    artifacts: Array<{
      path: string
      size_bytes: number
      created_at: string
    }>
  } | null
  artifacts: Array<{
    path: string
    size_bytes: number
    created_at: string
  }>
}

export interface ReviewArtifactContentResponse {
  run_id: string
  path: string
  content: string
}

function buildPath(path: string): string {
  return `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(buildPath(path), {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers || {}),
    },
  })

  if (!response.ok) {
    let detail = response.statusText
    try {
      const payload = await response.json()
      detail = payload.detail || JSON.stringify(payload)
    } catch {
      detail = await response.text()
    }
    throw new Error(detail || 'Unexpected API error')
  }

  return response.json() as Promise<T>
}

export function getDefaultRepoPath(): string | undefined {
  return DEFAULT_REPO_PATH
}

export async function sendChatMessage(
  message: string,
  sessionId?: string | null,
  requestedMode?: 'review',
) {
  return apiFetch<ChatApiResponse>('/chat', {
    method: 'POST',
    body: JSON.stringify({
      message,
      session_id: sessionId || undefined,
      requested_mode: requestedMode,
    }),
  })
}

export async function submitReviewRun(input: {
  userQuery: string
  intent?: ReviewIntent
  repoPath?: string
  repoUrl?: string
  sandboxBackend?: SandboxBackend
  sessionId?: string | null
}) {
  return apiFetch<ReviewRunSubmittedResponse>('/review/runs', {
    method: 'POST',
    body: JSON.stringify({
      repo_path: input.repoPath,
      repo_url: input.repoUrl,
      intent: input.intent,
      sandbox_backend: input.sandboxBackend,
      user_query: input.userQuery,
      session_id: input.sessionId || undefined,
    }),
  })
}

export async function getReviewRun(runId: string) {
  return apiFetch<ReviewRunDetailResponse>(`/review/runs/${runId}`)
}

export async function getReviewArtifactContent(runId: string, artifactPath: string) {
  const query = new URLSearchParams({ path: artifactPath })
  return apiFetch<ReviewArtifactContentResponse>(`/review/runs/${runId}/artifacts/content?${query.toString()}`)
}

export function buildReviewStreamUrl(streamPath: string): string {
  const url = new URL(streamPath, window.location.origin)
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  return url.toString()
}

export function openReviewStream(
  streamPath: string,
  handlers: {
    onEvent: (event: ReviewRunEvent) => void
    onError?: () => void
    onClose?: () => void
  },
): WebSocket {
  const socket = new WebSocket(buildReviewStreamUrl(streamPath))
  socket.onmessage = (message) => {
    const event = JSON.parse(message.data) as ReviewRunEvent
    handlers.onEvent(event)
  }
  socket.onerror = () => {
    handlers.onError?.()
  }
  socket.onclose = () => {
    handlers.onClose?.()
  }
  return socket
}
