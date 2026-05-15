export type MessageRole = 'user' | 'assistant'
export type RunMode = 'chat' | 'review'
export type RunStatus = 'pending' | 'in_progress' | 'completed' | 'failed'
export type AgentStatus = 'idle' | 'working' | 'completed' | 'failed'
export type RepoSourceType = 'local' | 'remote'
export type ReviewIntent = 'review' | 'fix'
export type ExecutionIntent = 'auto' | ReviewIntent
export type ReviewTargetMode = 'local' | 'github'
export type SandboxBackend = 'local' | 'docker' | 'opensandbox'
export type WorkerRole =
  | 'repo_mapper'
  | 'static_reviewer'
  | 'runtime_tester'
  | 'security_reviewer'
  | 'docs_devex_reviewer'
  | 'external_validator'

export type RunEventType =
  | 'run_created'
  | 'run_started'
  | 'sandbox_ready'
  | 'plan_ready'
  | 'worker_started'
  | 'worker_completed'
  | 'tool_completed'
  | 'fix_started'
  | 'file_updated'
  | 'fix_completed'
  | 'report_ready'
  | 'run_completed'
  | 'run_failed'

export interface Message {
  id: string
  role: MessageRole
  content: string
  timestamp: number
  agentWork?: AgentWorkResult
  thinking?: ThinkingData
}

export interface ThinkingData {
  steps: string[]
  durationMs: number
  isActive: boolean
}

export interface ReviewPlanSummary {
  mode: RunMode
  spawnCount: number
  selectedWorkers: string[]
  needsSandbox: boolean
  routeReason: string
}

export interface AgentSnapshot {
  id: string
  role: string
  name: string
  task: string
  status: AgentStatus
  progress: number
  totalSteps: number
  logs: string[]
}

export type AgentState = AgentSnapshot

export interface AgentWorkResult {
  runId?: string
  mode: RunMode
  intent?: ReviewIntent
  query: string
  answer: string
  agents: AgentSnapshot[]
  completed: boolean
  status: RunStatus
  routeReason?: string
  repoPath?: string
  repoSourceType?: RepoSourceType
  repoSource?: string
  overallVerdict?: string
  findingsCount?: number
  summary?: string
  fixResult?: {
    applied: boolean
    summary: string
    changedFiles: string[]
    diffArtifactPath?: string | null
    diffText?: string
    appliedToLocalRepo?: boolean
    applyBackTarget?: string | null
    applyBackError?: string | null
  }
}

export interface ActiveSwarmRun {
  runId: string
  query: string
  intent?: ReviewIntent
  routeReason: string
  status: RunStatus
  streamPath: string
  repoPath?: string
  repoSourceType?: RepoSourceType
  repoSource?: string
  sandboxBackend?: SandboxBackend
  overallVerdict?: string
}

export interface ReviewTargetConfig {
  mode: ReviewTargetMode
  localPath: string
  repoUrl: string
}

export interface Project {
  id: string
  name: string
  sessionIds: string[]
  createdAt: number
}

export interface ChatSession {
  id: string
  title: string
  messages: Message[]
  createdAt: number
  projectId?: string
}

export interface ReviewRunEvent {
  event_id: string
  run_id: string
  event_type: RunEventType
  message: string
  timestamp: string
  status: RunStatus | null
  payload: Record<string, unknown>
}
