import { useCallback, useEffect, useRef, useState } from 'react'
import { Bot } from 'lucide-react'
import {
  getDefaultRepoPath,
  getReviewArtifactContent,
  getReviewRun,
  openReviewStream,
  sendChatMessage,
  submitReviewRun,
} from '../lib/api'
import {
  buildAgentWorkResult,
  buildChatThinkingSteps,
  buildPendingAssistantMessage,
  buildReviewThinkingSteps,
  createAgentsFromWorkers,
  formatRepoTargetLabel,
  formatWorkerLog,
} from '../lib/swarm'
import { useAppStore } from '../store'
import ChatInput from './ChatInput'
import ChatMessage from './ChatMessage'
import StatusMatrix from './StatusMatrix'
import type {
  AgentWorkResult,
  AgentSnapshot,
  ReviewPlanSummary,
  ReviewRunEvent,
  SandboxBackend,
  ThinkingData,
} from '../types'

interface RunMessageContext {
  runId: string
  sessionId: string
  assistantMessageId: string
  query: string
  routeReason: string
}

function normalizePlan(plan: {
  mode?: 'chat' | 'review'
  spawn_count: number
  selected_workers: string[]
  needs_sandbox: boolean
  route_reason: string
}): ReviewPlanSummary {
  return {
    mode: plan.mode ?? 'review',
    spawnCount: plan.spawn_count,
    selectedWorkers: plan.selected_workers,
    needsSandbox: plan.needs_sandbox,
    routeReason: plan.route_reason,
  }
}

export default function ChatArea() {
  const {
    sessions,
    activeSessionId,
    createSession,
    addMessage,
    updateMessage,
    isSwarmActive,
    swarmAgents,
    startSwarmRun,
    updateSwarmRunStatus,
    finishSwarmRun,
    setAgentProgress,
    setAgentStatus,
    addAgentLog,
    setViewingWork,
    viewingWork,
    setComputerScreen,
    addCompletedTask,
    setSelectedAgent,
    resetSwarmState,
    executionIntent,
    reviewTargetMode,
    localRepoPath,
    remoteRepoUrl,
    reviewSandboxBackend,
    setExecutionIntent,
    setReviewTargetMode,
    setLocalRepoPath,
    setRemoteRepoUrl,
    setReviewSandboxBackend,
  } = useAppStore()

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const timersRef = useRef<ReturnType<typeof setTimeout>[]>([])
  const socketRef = useRef<WebSocket | null>(null)
  const activeRunRef = useRef<RunMessageContext | null>(null)
  const eventPlaybackAtRef = useRef(0)
  const [thinkingMsgId, setThinkingMsgId] = useState<string | null>(null)
  const [reviewTargetExpanded, setReviewTargetExpanded] = useState(true)

  const activeSession = sessions.find((session) => session.id === activeSessionId)

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  const clearTimers = useCallback(() => {
    timersRef.current.forEach(clearTimeout)
    timersRef.current = []
  }, [])

  const closeSocket = useCallback(() => {
    socketRef.current?.close()
    socketRef.current = null
    eventPlaybackAtRef.current = 0
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [activeSession?.messages.length, scrollToBottom])

  useEffect(() => {
    return () => {
      clearTimers()
      closeSocket()
    }
  }, [clearTimers, closeSocket])

  useEffect(() => {
    if (!activeSession || activeSession.messages.length === 0) {
      setReviewTargetExpanded(true)
    }
  }, [activeSession])

  const revealThinkingSteps = useCallback(
    (sessionId: string, messageId: string, steps: string[]) =>
      new Promise<ThinkingData>((resolve) => {
        clearTimers()
        setThinkingMsgId(messageId)

        const startTime = Date.now()
        let index = 0
        const visibleSteps: string[] = []

        const tick = () => {
          if (index >= steps.length) {
            const thinking: ThinkingData = {
              steps: [...visibleSteps],
              durationMs: Date.now() - startTime,
              isActive: false,
            }
            updateMessage(sessionId, messageId, { thinking })
            setThinkingMsgId((current) => (current === messageId ? null : current))
            resolve(thinking)
            return
          }

          visibleSteps.push(steps[index])
          updateMessage(sessionId, messageId, {
            thinking: {
              steps: [...visibleSteps],
              durationMs: 0,
              isActive: true,
            },
          })
          index += 1

          const timer = setTimeout(tick, 85 + Math.random() * 55)
          timersRef.current.push(timer)
        }

        const firstTimer = setTimeout(tick, 180)
        timersRef.current.push(firstTimer)
      }),
    [clearTimers, updateMessage],
  )

  const startThinkingPreview = useCallback(
    (sessionId: string, messageId: string, steps: string[]) => {
      clearTimers()
      setThinkingMsgId(messageId)

      let index = 0
      const visibleSteps: string[] = []

      const tick = () => {
        if (index < steps.length) {
          visibleSteps.push(steps[index])
          updateMessage(sessionId, messageId, {
            thinking: {
              steps: [...visibleSteps],
              durationMs: 0,
              isActive: true,
            },
          })
          index += 1
          const timer = setTimeout(tick, 150 + Math.random() * 90)
          timersRef.current.push(timer)
          return
        }

        const timer = setTimeout(tick, 520)
        timersRef.current.push(timer)
      }

      const firstTimer = setTimeout(tick, 80)
      timersRef.current.push(firstTimer)
    },
    [clearTimers, updateMessage],
  )

  const queueReturnToAgentWork = useCallback(() => {
    const timer = setTimeout(() => {
      const state = useAppStore.getState()
      const workingAgent = state.swarmAgents.find((agent) => agent.status === 'working')
      if (!workingAgent) {
        return
      }
      state.setSelectedAgent(workingAgent.id)
      state.setComputerScreen('agent-work')
    }, 1300)
    timersRef.current.push(timer)
  }, [])

  const finalizeReviewRun = useCallback(
    async (context: RunMessageContext, fallbackMessage?: string) => {
      const detail = await getReviewRun(context.runId)
      let diffText: string | undefined
      if (detail.fix_result?.diff_artifact_path) {
        try {
          const artifact = await getReviewArtifactContent(
            context.runId,
            detail.fix_result.diff_artifact_path,
          )
          diffText = artifact.content
        } catch {
          diffText = undefined
        }
      }
      const storeState = useAppStore.getState()
      const currentAgents = storeState.swarmAgents.map((agent) => ({
        ...agent,
        status:
          detail.status === 'failed' && agent.status !== 'completed'
            ? 'failed'
            : agent.status,
      }))

      const resultMeta = buildAgentWorkResult(detail, context.query, currentAgents)
      const answer = fallbackMessage || resultMeta.answer

      const work: AgentWorkResult = {
        runId: context.runId,
        mode: 'review',
        query: context.query,
        answer,
        completed: true,
        status: detail.status,
        intent: detail.intent,
        routeReason: context.routeReason,
        repoPath: detail.repo_path,
        repoSourceType: detail.repo_source_type,
        repoSource: detail.repo_source,
        overallVerdict: resultMeta.overallVerdict,
        findingsCount: resultMeta.findingsCount,
        summary: resultMeta.summary,
        fixResult: resultMeta.fixResult
          ? {
              ...resultMeta.fixResult,
              diffText,
            }
          : undefined,
        agents: currentAgents.map(
          (agent): AgentSnapshot => ({
            id: agent.id,
            role: agent.role,
            name: agent.name,
            task: agent.task,
            status: agent.status,
            progress: agent.progress,
            totalSteps: agent.totalSteps,
            logs: [...agent.logs],
          }),
        ),
      }

      updateMessage(context.sessionId, context.assistantMessageId, {
        content: answer,
        agentWork: work,
      })

      finishSwarmRun()
      activeRunRef.current = null
      closeSocket()
    },
    [closeSocket, finishSwarmRun, updateMessage],
  )

  const handleReviewEvent = useCallback(
    async (event: ReviewRunEvent) => {
      const activeRun = activeRunRef.current
      if (!activeRun || activeRun.runId !== event.run_id) {
        return
      }

      const workerRole =
        typeof event.payload.worker_role === 'string'
          ? event.payload.worker_role
          : null

      if (event.status) {
        updateSwarmRunStatus(event.status)
      }

      switch (event.event_type) {
        case 'run_started':
          setComputerScreen('planning')
          return
        case 'worker_started':
          if (!workerRole) {
            return
          }
          setAgentStatus(workerRole, 'working')
          setSelectedAgent(workerRole)
          setComputerScreen('agent-work')
          {
            const log = formatWorkerLog(event)
            if (log) {
              addAgentLog(workerRole, log)
            }
          }
          return
        case 'tool_completed':
          if (!workerRole) {
            return
          }
          setAgentStatus(workerRole, 'working')
          setSelectedAgent(workerRole)
          setComputerScreen('agent-work')
          {
            const log = formatWorkerLog(event)
            if (log) {
              addAgentLog(workerRole, log)
            }
          }
          return
        case 'fix_started':
        case 'file_updated':
          if (!workerRole) {
            return
          }
          setAgentStatus(workerRole, 'working')
          setSelectedAgent(workerRole)
          setComputerScreen('agent-work')
          {
            const log = formatWorkerLog(event)
            if (log) {
              addAgentLog(workerRole, log)
            }
          }
          return
        case 'fix_completed':
          if (!workerRole) {
            return
          }
          {
            const log = formatWorkerLog(event)
            if (log) {
              addAgentLog(workerRole, log)
            }
          }
          setAgentStatus(workerRole, 'completed')
          setComputerScreen('task-review')
          return
        case 'worker_completed':
          if (!workerRole) {
            return
          }
          {
            const log = formatWorkerLog(event)
            if (log) {
              addAgentLog(workerRole, log)
            }
          }
          setAgentStatus(workerRole, 'completed')
          setAgentProgress(workerRole, 14)
          addCompletedTask(workerRole)
          setComputerScreen('task-review')
          queueReturnToAgentWork()
          return
        case 'run_completed':
          await finalizeReviewRun(activeRun)
          return
        case 'run_failed':
          await finalizeReviewRun(
            activeRun,
            `The review run failed before the final report completed.\n\n${String(
              event.payload.error_message || event.message,
            )}`,
          )
          return
        default:
          return
      }
    },
    [
      addAgentLog,
      addCompletedTask,
      finalizeReviewRun,
      queueReturnToAgentWork,
      setAgentProgress,
      setAgentStatus,
      setComputerScreen,
      setSelectedAgent,
      updateSwarmRunStatus,
    ],
  )

  const scheduleReviewEventPlayback = useCallback((event: ReviewRunEvent) => {
    const now = Date.now()
    const playbackStart = Math.max(now, eventPlaybackAtRef.current)
    const delay = playbackStart - now
    eventPlaybackAtRef.current = playbackStart + getEventPlaybackDuration(event)
    return delay
  }, [])

  const beginReviewRun = useCallback(
    (
      sessionId: string,
      assistantMessageId: string,
      query: string,
      plan: ReviewPlanSummary,
        submitted: {
          run_id: string
          status: 'pending' | 'in_progress' | 'completed' | 'failed'
          intent: 'review' | 'fix'
          stream_path: string
          repo_path: string
          repo_source_type: 'local' | 'remote'
        repo_source: string
        sandbox_backend: 'local' | 'docker' | 'opensandbox'
      },
    ) => {
      const agents = createAgentsFromWorkers(plan.selectedWorkers)

      startSwarmRun(
        {
          runId: submitted.run_id,
          query,
          intent: submitted.intent,
          routeReason: plan.routeReason,
          status: submitted.status,
          streamPath: submitted.stream_path,
          repoPath: submitted.repo_path,
          repoSourceType: submitted.repo_source_type,
          repoSource: submitted.repo_source,
          sandboxBackend: submitted.sandbox_backend as SandboxBackend,
        },
        agents,
      )

      activeRunRef.current = {
        runId: submitted.run_id,
        sessionId,
        assistantMessageId,
        query,
        routeReason: plan.routeReason,
      }

      closeSocket()
      eventPlaybackAtRef.current = 0
      socketRef.current = openReviewStream(submitted.stream_path, {
        onEvent: (event) => {
          const delay = scheduleReviewEventPlayback(event)
          const timer = setTimeout(() => {
            void handleReviewEvent(event)
          }, delay)
          timersRef.current.push(timer)
        },
        onError: () => {
          updateMessage(sessionId, assistantMessageId, {
            content: 'The live event stream disconnected unexpectedly. The run may still be processing on the backend.',
          })
        },
      })
    },
    [closeSocket, handleReviewEvent, scheduleReviewEventPlayback, startSwarmRun, updateMessage],
  )

  const resolveReviewInput = useCallback(() => {
    if (reviewTargetMode === 'github') {
      const repoUrl = remoteRepoUrl.trim()
      if (!repoUrl) {
        throw new Error('Enter a GitHub repository URL before starting a review.')
      }
      return {
        repoPath: undefined,
        repoUrl,
        repoTargetLabel: `GitHub target ${repoUrl}`,
      }
    }

    const repoPath = localRepoPath.trim() || getDefaultRepoPath()
      return {
        repoPath,
        repoUrl: undefined,
        repoTargetLabel: repoPath ? `Local repo ${repoPath}` : 'Configured default local repo',
      }
  }, [localRepoPath, remoteRepoUrl, reviewTargetMode])

  const handleSend = useCallback(
    async (content: string) => {
      closeSocket()
      clearTimers()
      activeRunRef.current = null
      resetSwarmState()
      setViewingWork(null)
      setReviewTargetExpanded(false)

      let sessionId = activeSessionId
      if (!sessionId) {
        sessionId = createSession()
      }

      addMessage(sessionId, { role: 'user', content })
      const assistantMessageId = addMessage(sessionId, {
        role: 'assistant',
        content: '',
        thinking: {
          steps: [],
          durationMs: 0,
          isActive: true,
        },
      })

      const previewTarget = reviewTargetMode === 'github'
        ? remoteRepoUrl.trim() || undefined
        : localRepoPath.trim() || getDefaultRepoPath() || undefined
      startThinkingPreview(
        sessionId,
        assistantMessageId,
        buildSupervisorRoutingPreviewSteps(content, previewTarget),
      )

      try {
        const requestedMode = executionIntent === 'auto' ? undefined : 'review'
        const chatResponse = await sendChatMessage(content, sessionId, requestedMode)

        if (executionIntent === 'auto' && chatResponse.plan.mode === 'chat') {
          const thinking = await revealThinkingSteps(
            sessionId,
            assistantMessageId,
            buildChatThinkingSteps(content),
          )
          updateMessage(sessionId, assistantMessageId, {
            content: chatResponse.response,
            thinking,
          })
          return
        }

        const reviewInput = resolveReviewInput()
        const plan = normalizePlan({
          ...chatResponse.plan,
        })
        const thinking = await revealThinkingSteps(
          sessionId,
          assistantMessageId,
          buildReviewThinkingSteps(
            content,
            plan,
            reviewInput.repoTargetLabel,
            executionIntent === 'fix' ? 'fix' : 'review',
          ),
        )

        updateMessage(sessionId, assistantMessageId, {
          content: buildPendingAssistantMessage(
            plan,
            reviewInput.repoTargetLabel,
            executionIntent === 'fix' ? 'fix' : 'review',
          ),
          thinking,
        })

        const submittedRun = await submitReviewRun({
          userQuery: content,
          intent: executionIntent === 'fix' ? 'fix' : 'review',
          repoPath: reviewInput.repoPath,
          repoUrl: reviewInput.repoUrl,
          sandboxBackend: reviewSandboxBackend,
          sessionId,
        })
        updateMessage(sessionId, assistantMessageId, {
          content: buildPendingAssistantMessage(
            plan,
            formatRepoTargetLabel(
              submittedRun.repo_source_type,
              submittedRun.repo_source,
              submittedRun.repo_path,
            ),
            submittedRun.intent,
          ),
          thinking,
        })

        beginReviewRun(sessionId, assistantMessageId, content, plan, {
          run_id: submittedRun.run_id,
          status: submittedRun.status,
          intent: submittedRun.intent,
          stream_path: submittedRun.stream_path,
          repo_path: submittedRun.repo_path,
          repo_source_type: submittedRun.repo_source_type,
          repo_source: submittedRun.repo_source,
          sandbox_backend: submittedRun.sandbox_backend,
        })
      } catch (error) {
        closeSocket()
        activeRunRef.current = null
        resetSwarmState()
        setThinkingMsgId(null)
        updateMessage(sessionId, assistantMessageId, {
          content:
            error instanceof Error
              ? formatReviewError(error.message)
              : 'Something went wrong while contacting the backend.',
          thinking: {
            steps: [],
            durationMs: 0,
            isActive: false,
          },
        })
      }
    },
    [
      activeSessionId,
      addMessage,
      beginReviewRun,
      clearTimers,
      closeSocket,
      createSession,
      resetSwarmState,
      revealThinkingSteps,
      resolveReviewInput,
      reviewTargetMode,
      localRepoPath,
      remoteRepoUrl,
      reviewSandboxBackend,
      executionIntent,
      startThinkingPreview,
      setViewingWork,
      updateMessage,
    ],
  )

  const handleViewWork = useCallback(
    (work: AgentWorkResult) => {
      setViewingWork(work)
    },
    [setViewingWork],
  )

  const showSwarmStatus =
    isSwarmActive && swarmAgents.length > 0 && !viewingWork && thinkingMsgId === null
  const executionIntentLabel = executionIntent === 'auto'
    ? 'Auto mode'
    : executionIntent === 'fix'
      ? 'Fix mode'
      : 'Review mode'
  const reviewTargetSummary = reviewTargetMode === 'github'
    ? `${executionIntentLabel} • ${remoteRepoUrl.trim() || 'GitHub repo URL not set'} • ${formatSandboxBackendLabel(reviewSandboxBackend)}`
    : `${executionIntentLabel} • ${localRepoPath.trim() || getDefaultRepoPath() || 'Configured default local repo'} • ${formatSandboxBackendLabel(reviewSandboxBackend)}`

  return (
    <div className="flex flex-col h-full bg-white dark:bg-surface-950">
      <div className="flex-1 overflow-y-auto">
        {!activeSession || activeSession.messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center px-4">
            <div className="w-12 h-12 rounded-2xl bg-emerald-100 dark:bg-emerald-900/30 flex items-center justify-center mb-4">
              <Bot size={24} className="text-emerald-600 dark:text-emerald-400" />
            </div>
            <h2 className="text-lg font-semibold text-zinc-800 dark:text-zinc-200 mb-1">
              Agent Swarm Chat
            </h2>
            <p className="text-sm text-zinc-400 dark:text-zinc-500 text-center max-w-sm">
              Send a simple question for supervisor-only chat, or choose a local path or GitHub repo
              and ask for a review to launch the swarm.
            </p>
          </div>
        ) : (
          <div className="max-w-2xl mx-auto px-4 py-6 space-y-6">
            {activeSession.messages.map((message) => (
              <ChatMessage
                key={message.id}
                message={message}
                onViewWork={
                  message.agentWork?.completed
                    ? () => handleViewWork(message.agentWork!)
                    : undefined
                }
              />
            ))}

            {showSwarmStatus && (
              <div className="animate-fade-in-up">
                <div className="flex gap-3">
                  <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-emerald-600 dark:bg-emerald-500 flex items-center justify-center">
                    <Bot size={16} className="text-white" />
                  </div>
                  <div className="flex-1">
                    <div
                      className="rounded-xl border p-3 space-y-2
                        bg-zinc-50 dark:bg-zinc-800/50
                        border-zinc-200 dark:border-zinc-700/50"
                    >
                      <div className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400 flex items-center gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                        Agent Swarm Working
                      </div>
                      {swarmAgents.map((agent) => (
                        <div
                          key={agent.id}
                          className="flex items-center justify-between gap-3"
                        >
                          <span className="text-xs font-medium text-zinc-700 dark:text-zinc-300 min-w-[72px]">
                            {agent.name}
                          </span>
                          <StatusMatrix
                            progress={agent.progress}
                            cols={agent.totalSteps}
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      <div className="border-t border-zinc-100 dark:border-zinc-800/50">
        <div className="max-w-2xl mx-auto px-4 py-4">
          <ChatInput
            onSend={handleSend}
            disabled={isSwarmActive || thinkingMsgId !== null}
            executionIntent={executionIntent}
            reviewTargetMode={reviewTargetMode}
            localRepoPath={localRepoPath}
            remoteRepoUrl={remoteRepoUrl}
            reviewSandboxBackend={reviewSandboxBackend}
            reviewTargetExpanded={reviewTargetExpanded}
            reviewTargetSummary={reviewTargetSummary}
            onExecutionIntentChange={setExecutionIntent}
            onReviewTargetModeChange={setReviewTargetMode}
            onLocalRepoPathChange={setLocalRepoPath}
            onRemoteRepoUrlChange={setRemoteRepoUrl}
            onReviewSandboxBackendChange={setReviewSandboxBackend}
            onToggleReviewTargetExpanded={setReviewTargetExpanded}
          />
        </div>
      </div>
    </div>
  )
}

function getEventPlaybackDuration(event: ReviewRunEvent): number {
  switch (event.event_type) {
    case 'worker_started':
    case 'worker_completed':
      return 650
    case 'tool_completed':
      return 360
    case 'run_completed':
    case 'run_failed':
      return 900
    default:
      return 240
  }
}

function buildSupervisorRoutingPreviewSteps(
  message: string,
  reviewTarget?: string,
): string[] {
  return [
    `Received query: "${truncate(message, 84)}"`,
    'Parsing intent, expected output, and urgency.',
    'Checking whether the supervisor can answer directly or should route into the swarm.',
    ...(reviewTarget
      ? [`Inspecting configured review target: ${truncate(reviewTarget, 88)}`]
      : []),
    'Estimating whether sandbox execution and specialist workers are required.',
    'Preparing the routing decision and execution plan.',
  ]
}

function truncate(value: string, limit: number): string {
  if (value.length <= limit) {
    return value
  }
  return `${value.slice(0, limit - 3)}...`
}

function formatSandboxBackendLabel(value: SandboxBackend): string {
  if (value === 'docker') {
    return 'Docker sandbox'
  }
  if (value === 'opensandbox') {
    return 'OpenSandbox'
  }
  return 'Local sandbox'
}

function formatReviewError(message: string): string {
  const normalized = message.toLowerCase()
  if (normalized.includes('remote fix mode requires a matching local clone')) {
    return [
      'GitHub fix mode needs a local clone first.',
      '',
      'Clone the repository onto this machine, make sure its `origin` remote matches the GitHub URL, then run Fix mode again. Review mode can still use the GitHub URL directly.',
    ].join('\n')
  }
  if (normalized.includes('multiple repositories') || normalized.includes('parent folder')) {
    return [
      'This path looks like a folder that contains multiple projects.',
      '',
      'Choose the exact repo root you want to review, then run the request again.',
    ].join('\n')
  }
  if (normalized.includes('docker') && normalized.includes('not')) {
    return [
      'Docker is required for Fix mode.',
      '',
      'Start Docker Desktop, keep Docker selected as the sandbox, then run the request again.',
    ].join('\n')
  }
  if (normalized.includes('repository mapping returned zero files')) {
    return [
      'The repo was staged, but the mapper found no files.',
      '',
      'Check that the selected path is the repo root and that the backend can read it.',
    ].join('\n')
  }
  if (normalized.includes('failed to clone remote repository')) {
    return [
      'The backend could not clone the GitHub repository.',
      '',
      message,
    ].join('\n')
  }
  return message
}
