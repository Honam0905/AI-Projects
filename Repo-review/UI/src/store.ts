import { create } from 'zustand'
import { nextProgressFromLogs } from './lib/swarm'
import type {
  ActiveSwarmRun,
  AgentState,
  AgentStatus,
  AgentWorkResult,
  ChatSession,
  ExecutionIntent,
  Message,
  Project,
  ReviewTargetMode,
  SandboxBackend,
  RunStatus,
} from './types'

type ScreenMode = 'planning' | 'agent-work' | 'task-review'

interface AppStore {
  isDarkMode: boolean
  toggleDarkMode: () => void

  sessions: ChatSession[]
  activeSessionId: string | null
  createSession: (projectId?: string) => string
  setActiveSession: (id: string) => void
  deleteSession: (id: string) => void
  addMessage: (sessionId: string, msg: Omit<Message, 'id' | 'timestamp'>) => string
  updateMessage: (
    sessionId: string,
    messageId: string,
    updates: Partial<Message>,
  ) => void

  projects: Project[]
  createProject: (name: string) => string
  renameProject: (id: string, name: string) => void
  deleteProject: (id: string) => void
  moveSessionToProject: (sessionId: string, projectId: string | undefined) => void
  expandedProjectIds: string[]
  toggleProjectExpanded: (id: string) => void

  searchQuery: string
  setSearchQuery: (q: string) => void

  reviewTargetMode: ReviewTargetMode
  executionIntent: ExecutionIntent
  localRepoPath: string
  remoteRepoUrl: string
  reviewSandboxBackend: SandboxBackend
  setExecutionIntent: (value: ExecutionIntent) => void
  setReviewTargetMode: (mode: ReviewTargetMode) => void
  setLocalRepoPath: (value: string) => void
  setRemoteRepoUrl: (value: string) => void
  setReviewSandboxBackend: (value: SandboxBackend) => void

  isSwarmActive: boolean
  activeSwarmRun: ActiveSwarmRun | null
  swarmAgents: AgentState[]
  selectedAgentId: string | null
  computerScreen: ScreenMode
  completedTaskIds: string[]
  viewingWork: AgentWorkResult | null

  startSwarmRun: (run: ActiveSwarmRun, agents: AgentState[]) => void
  updateSwarmRunStatus: (status: RunStatus) => void
  finishSwarmRun: () => void
  setSelectedAgent: (id: string) => void
  setAgentStatus: (agentId: string, status: AgentStatus) => void
  setAgentProgress: (agentId: string, progress: number) => void
  addAgentLog: (agentId: string, log: string) => void
  setComputerScreen: (screen: ScreenMode) => void
  addCompletedTask: (agentId: string) => void
  resetSwarmState: () => void
  setViewingWork: (work: AgentWorkResult | null) => void
}

const uid = () => crypto.randomUUID()
const DEFAULT_LOCAL_REPO_PATH = import.meta.env.VITE_DEFAULT_REPO_PATH?.trim() || ''
const DEFAULT_REMOTE_REPO_URL = import.meta.env.VITE_DEFAULT_GITHUB_REPO_URL?.trim() || ''
const DEFAULT_SANDBOX_BACKEND =
  (import.meta.env.VITE_DEFAULT_SANDBOX_BACKEND?.trim() as SandboxBackend | undefined) ||
  'local'

function patchAgent(
  agents: AgentState[],
  agentId: string,
  updater: (agent: AgentState) => AgentState,
): AgentState[] {
  return agents.map((agent) => (agent.id === agentId ? updater(agent) : agent))
}

export const useAppStore = create<AppStore>((set) => ({
  isDarkMode: true,
  toggleDarkMode: () => set((state) => ({ isDarkMode: !state.isDarkMode })),

  sessions: [],
  activeSessionId: null,

  createSession: (projectId?: string) => {
    const id = uid()
    const session: ChatSession = {
      id,
      title: 'New Chat',
      messages: [],
      createdAt: Date.now(),
      projectId,
    }
    set((state) => ({
      sessions: [session, ...state.sessions],
      activeSessionId: id,
      isSwarmActive: false,
      activeSwarmRun: null,
      swarmAgents: [],
      selectedAgentId: null,
      viewingWork: null,
      projects: projectId
        ? state.projects.map((project) =>
            project.id === projectId
              ? { ...project, sessionIds: [id, ...project.sessionIds] }
              : project,
          )
        : state.projects,
    }))
    return id
  },

  setActiveSession: (id) =>
    set({
      activeSessionId: id,
      isSwarmActive: false,
      activeSwarmRun: null,
      swarmAgents: [],
      selectedAgentId: null,
      viewingWork: null,
      computerScreen: 'planning',
      completedTaskIds: [],
    }),

  deleteSession: (id) =>
    set((state) => {
      const session = state.sessions.find((candidate) => candidate.id === id)
      const sessions = state.sessions.filter((candidate) => candidate.id !== id)
      const activeSessionId =
        state.activeSessionId === id ? (sessions[0]?.id ?? null) : state.activeSessionId
      return {
        sessions,
        activeSessionId,
        projects: session?.projectId
          ? state.projects.map((project) =>
              project.id === session.projectId
                ? {
                    ...project,
                    sessionIds: project.sessionIds.filter((sessionId) => sessionId !== id),
                  }
                : project,
            )
          : state.projects,
      }
    }),

  addMessage: (sessionId, msg) => {
    const id = uid()
    const message: Message = { ...msg, id, timestamp: Date.now() }
    set((state) => ({
      sessions: state.sessions.map((session) => {
        if (session.id !== sessionId) {
          return session
        }
        const updatedSession = {
          ...session,
          messages: [...session.messages, message],
        }
        if (session.messages.length === 0 && msg.role === 'user') {
          updatedSession.title = msg.content.slice(0, 40) || 'New Chat'
        }
        return updatedSession
      }),
    }))
    return id
  },

  updateMessage: (sessionId, messageId, updates) =>
    set((state) => ({
      sessions: state.sessions.map((session) => {
        if (session.id !== sessionId) {
          return session
        }
        return {
          ...session,
          messages: session.messages.map((message) =>
            message.id === messageId ? { ...message, ...updates } : message,
          ),
        }
      }),
    })),

  projects: [],
  expandedProjectIds: [],

  createProject: (name: string) => {
    const id = uid()
    set((state) => ({
      projects: [
        ...state.projects,
        { id, name, sessionIds: [], createdAt: Date.now() },
      ],
      expandedProjectIds: [...state.expandedProjectIds, id],
    }))
    return id
  },

  renameProject: (id, name) =>
    set((state) => ({
      projects: state.projects.map((project) =>
        project.id === id ? { ...project, name } : project,
      ),
    })),

  deleteProject: (id) =>
    set((state) => ({
      projects: state.projects.filter((project) => project.id !== id),
      sessions: state.sessions.map((session) =>
        session.projectId === id ? { ...session, projectId: undefined } : session,
      ),
      expandedProjectIds: state.expandedProjectIds.filter((projectId) => projectId !== id),
    })),

  moveSessionToProject: (sessionId, projectId) =>
    set((state) => {
      const session = state.sessions.find((candidate) => candidate.id === sessionId)
      const oldProjectId = session?.projectId

      return {
        sessions: state.sessions.map((candidate) =>
          candidate.id === sessionId ? { ...candidate, projectId } : candidate,
        ),
        projects: state.projects.map((project) => {
          let sessionIds = project.sessionIds
          if (project.id === oldProjectId) {
            sessionIds = sessionIds.filter((id) => id !== sessionId)
          }
          if (project.id === projectId) {
            sessionIds = [sessionId, ...sessionIds.filter((id) => id !== sessionId)]
          }
          return { ...project, sessionIds }
        }),
      }
    }),

  toggleProjectExpanded: (id) =>
    set((state) => ({
      expandedProjectIds: state.expandedProjectIds.includes(id)
        ? state.expandedProjectIds.filter((projectId) => projectId !== id)
        : [...state.expandedProjectIds, id],
    })),

  searchQuery: '',
  setSearchQuery: (q) => set({ searchQuery: q }),

  reviewTargetMode: 'local',
  executionIntent: 'auto',
  localRepoPath: DEFAULT_LOCAL_REPO_PATH,
  remoteRepoUrl: DEFAULT_REMOTE_REPO_URL,
  reviewSandboxBackend: DEFAULT_SANDBOX_BACKEND,
  setExecutionIntent: (value) =>
    set((state) => ({
      executionIntent: value,
      reviewSandboxBackend:
        value === 'fix' ? 'docker' : state.reviewSandboxBackend,
    })),
  setReviewTargetMode: (mode) => set({ reviewTargetMode: mode }),
  setLocalRepoPath: (value) => set({ localRepoPath: value }),
  setRemoteRepoUrl: (value) => set({ remoteRepoUrl: value }),
  setReviewSandboxBackend: (value) =>
    set((state) => ({
      reviewSandboxBackend: state.executionIntent === 'fix' ? 'docker' : value,
    })),

  isSwarmActive: false,
  activeSwarmRun: null,
  swarmAgents: [],
  selectedAgentId: null,
  computerScreen: 'planning',
  completedTaskIds: [],
  viewingWork: null,

  startSwarmRun: (run, agents) =>
    set({
      isSwarmActive: true,
      activeSwarmRun: run,
      swarmAgents: agents,
      selectedAgentId: agents[0]?.id ?? null,
      computerScreen: 'planning',
      completedTaskIds: [],
      viewingWork: null,
    }),

  updateSwarmRunStatus: (status) =>
    set((state) => ({
      activeSwarmRun: state.activeSwarmRun
        ? { ...state.activeSwarmRun, status }
        : null,
    })),

  finishSwarmRun: () =>
    set({
      isSwarmActive: false,
      activeSwarmRun: null,
      swarmAgents: [],
      selectedAgentId: null,
      computerScreen: 'planning',
      completedTaskIds: [],
    }),

  setSelectedAgent: (id) => set({ selectedAgentId: id }),

  setAgentStatus: (agentId, status) =>
    set((state) => {
      const swarmAgents = patchAgent(state.swarmAgents, agentId, (agent) => {
        const nextStatus = status
        const nextProgress =
          nextStatus === 'completed'
            ? agent.totalSteps
            : nextStatus === 'working'
              ? Math.max(agent.progress, 1)
              : agent.progress
        return {
          ...agent,
          status: nextStatus,
          progress: nextProgress,
        }
      })

      const completedTaskIds =
        status === 'completed' && !state.completedTaskIds.includes(agentId)
          ? [...state.completedTaskIds, agentId]
          : state.completedTaskIds

      return {
        swarmAgents,
        completedTaskIds,
      }
    }),

  setAgentProgress: (agentId, progress) =>
    set((state) => ({
      swarmAgents: patchAgent(state.swarmAgents, agentId, (agent) => ({
        ...agent,
        progress: Math.max(0, Math.min(agent.totalSteps, progress)),
      })),
    })),

  addAgentLog: (agentId, log) =>
    set((state) => ({
      swarmAgents: patchAgent(state.swarmAgents, agentId, (agent) => {
        const nextAgent = {
          ...agent,
          logs: [...agent.logs, log],
        }
        return {
          ...nextAgent,
          progress: Math.max(nextAgent.progress, nextProgressFromLogs(nextAgent)),
        }
      }),
    })),

  setComputerScreen: (screen) => set({ computerScreen: screen }),

  addCompletedTask: (agentId) =>
    set((state) => ({
      completedTaskIds: state.completedTaskIds.includes(agentId)
        ? state.completedTaskIds
        : [...state.completedTaskIds, agentId],
    })),

  resetSwarmState: () =>
    set({
      isSwarmActive: false,
      activeSwarmRun: null,
      swarmAgents: [],
      selectedAgentId: null,
      computerScreen: 'planning',
      completedTaskIds: [],
    }),

  setViewingWork: (work) =>
    set({
      viewingWork: work,
      selectedAgentId: work?.agents[0]?.id ?? null,
      computerScreen: work ? 'agent-work' : 'planning',
    }),
}))
