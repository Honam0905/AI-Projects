import { useEffect, useMemo, useRef, useState } from 'react'
import type { RepoSourceType, ReviewIntent } from '../types'
import AgentAvatar from './AgentAvatar'

interface TaskScreenProps {
  agents: { id: string; name: string; task: string }[]
  completedTaskIds: string[]
  isPlanning: boolean
  query: string
  intent?: ReviewIntent
  routeReason?: string
  repoPath?: string
  repoSourceType?: RepoSourceType
  repoSource?: string
}

interface PlanningLine {
  text: string
  delay: number
  taskAgentId?: string
}

function buildPlanningLines(
  query: string,
  intent: ReviewIntent | undefined,
  routeReason: string | undefined,
  repoPath: string | undefined,
  repoSourceType: RepoSourceType | undefined,
  repoSource: string | undefined,
  agents: TaskScreenProps['agents'],
): PlanningLine[] {
  const lines: PlanningLine[] = [
    { text: '> supervisor --init', delay: 0 },
    { text: `> Received query: "${truncate(query, 58)}"`, delay: 260 },
    { text: '> Analyzing requirements...', delay: 540 },
    {
      text: `> Operation: ${intent === 'fix' ? 'fix mode' : 'review mode'}`,
      delay: 700,
    },
    {
      text: `> Route: ${truncate(routeReason || 'dynamic swarm planning', 64)}`,
      delay: 860,
    },
  ]

  if (repoSourceType === 'remote' && repoSource) {
    lines.push({
      text: `> Source: remote GitHub clone`,
      delay: 1140,
    })
    lines.push({
      text: `> Remote target: ${truncate(repoSource, 64)}`,
      delay: 1320,
    })
    lines.push({
      text: `> Sandbox staging path: ${truncate(repoPath || 'managed cache clone', 64)}`,
      delay: 1500,
    })
  } else if (repoSource || repoPath) {
    lines.push({
      text: `> Source: local filesystem workspace`,
      delay: 1140,
    })
    lines.push({
      text: `> Repo target: ${truncate(repoSource || repoPath || '', 64)}`,
      delay: 1320,
    })
  }

  lines.push(
    { text: '', delay: 1680 },
    { text: '─── Task Plan ──────────────────', delay: 1820 },
  )

  agents.forEach((agent, index) => {
    lines.push({
      text: agent.task,
      delay: 2040 + index * 180,
      taskAgentId: agent.id,
    })
  })

  lines.push(
    {
      text: '────────────────────────────────',
      delay: 2220 + agents.length * 180,
    },
    { text: '', delay: 2340 + agents.length * 180 },
    { text: '> Dispatching agents...', delay: 2460 + agents.length * 180 },
  )

  return lines
}

export default function TaskScreen({
  agents,
  completedTaskIds,
  isPlanning,
  query,
  intent,
  routeReason,
  repoPath,
  repoSourceType,
  repoSource,
}: TaskScreenProps) {
  const planningLines = useMemo(
    () => buildPlanningLines(query, intent, routeReason, repoPath, repoSourceType, repoSource, agents),
    [agents, intent, query, repoPath, repoSource, repoSourceType, routeReason],
  )
  const [visibleCount, setVisibleCount] = useState(
    isPlanning ? 0 : planningLines.length,
  )
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!isPlanning) {
      setVisibleCount(planningLines.length)
      return
    }

    setVisibleCount(0)
    const timers: ReturnType<typeof setTimeout>[] = []
    planningLines.forEach((line, index) => {
      timers.push(setTimeout(() => setVisibleCount(index + 1), line.delay))
    })
    return () => timers.forEach(clearTimeout)
  }, [isPlanning, planningLines])

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [visibleCount, completedTaskIds])

  const completedCount = completedTaskIds.length

  return (
    <div className="flex flex-col h-full">
      <div
        className="flex-1 min-h-0 flex flex-col rounded-2xl overflow-hidden
          bg-zinc-800 dark:bg-zinc-900
          shadow-[0_8px_32px_rgba(0,0,0,0.3),inset_0_1px_0_rgba(255,255,255,0.05)]
          border border-zinc-700/50"
      >
        <div className="flex items-center gap-2 px-4 py-2.5 bg-zinc-750 dark:bg-[#1e1e1e] border-b border-zinc-700/40 flex-shrink-0">
          <div className="flex gap-[6px]">
            <div className="w-[11px] h-[11px] rounded-full bg-[#ff5f57] shadow-[inset_0_-1px_1px_rgba(0,0,0,0.2)]" />
            <div className="w-[11px] h-[11px] rounded-full bg-[#febc2e] shadow-[inset_0_-1px_1px_rgba(0,0,0,0.2)]" />
            <div className="w-[11px] h-[11px] rounded-full bg-[#28c840] shadow-[inset_0_-1px_1px_rgba(0,0,0,0.2)]" />
          </div>
          <div className="flex-1 text-center">
            <span className="text-[11px] font-medium text-zinc-400 font-mono">
              Supervisor — Task Planner
            </span>
          </div>
          <div className="w-[56px]" />
        </div>

        <div
          ref={scrollRef}
          className="retro-screen flex-1 min-h-0 overflow-y-auto p-4 bg-[#0c0c0c] dark:bg-[#080808]"
        >
          <div className="flex items-center gap-2 mb-3">
            <div
              className={`w-2 h-2 rounded-full ${
                isPlanning ? 'bg-amber-400 animate-pulse' : 'bg-emerald-400'
              }`}
            />
            <span
              className={`font-mono text-xs ${
                isPlanning ? 'text-amber-400/80' : 'text-emerald-400/80'
              }`}
            >
              {isPlanning
                ? 'Planning tasks...'
                : `Task review — ${completedCount}/${agents.length} completed`}
            </span>
          </div>

          <div className="space-y-[4px]">
            {planningLines.slice(0, visibleCount).map((line, index) => {
              const agent = agents.find(
                (candidate) => candidate.id === line.taskAgentId,
              )

              if (agent) {
                const isDone = completedTaskIds.includes(agent.id)
                return (
                  <div key={`${agent.id}-${index}`} className="flex items-center gap-3 py-1 pl-2">
                    <div
                      className={`w-[18px] h-[18px] rounded flex items-center justify-center border ${
                        isDone
                          ? 'border-emerald-400/60 bg-emerald-400/15'
                          : 'border-emerald-700/40'
                      }`}
                    >
                      {isDone && (
                        <span className="text-emerald-400 text-[11px] font-bold">✓</span>
                      )}
                    </div>
                    <span
                      className={`font-mono text-[12px] flex-1 ${
                        isDone
                          ? 'text-emerald-400/60 line-through'
                          : 'text-emerald-300/90'
                      }`}
                    >
                      {agent.task}
                    </span>
                    <div className="flex items-center gap-1.5">
                      <span className="text-emerald-500/50 font-mono text-[11px]">→</span>
                      <AgentAvatar agentId={agent.id} label={agent.name} size={18} />
                      <span className="text-emerald-500/60 font-mono text-[11px]">
                        {agent.name}
                      </span>
                    </div>
                    {isDone && (
                      <span className="text-emerald-400/50 font-mono text-[10px]">
                        [done]
                      </span>
                    )}
                  </div>
                )
              }

              return (
                <div
                  key={`${line.text}-${index}`}
                  className="text-emerald-300/90 font-mono text-[12px] leading-relaxed"
                >
                  {line.text || '\u00A0'}
                </div>
              )
            })}
          </div>

          {!isPlanning && completedCount > 0 && (
            <div className="mt-4 pt-3 border-t border-emerald-900/40">
              <div className="flex items-center gap-2">
                <div className="flex-1 h-1.5 rounded-full bg-emerald-900/30 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-emerald-400 transition-all duration-500"
                    style={{ width: `${(completedCount / Math.max(agents.length, 1)) * 100}%` }}
                  />
                </div>
                <span className="text-emerald-400/70 font-mono text-[11px]">
                  {completedCount}/{agents.length}
                </span>
              </div>
              {completedCount < agents.length && (
                <span className="text-emerald-500/50 font-mono text-[11px] mt-2 block">
                  Awaiting remaining agents...
                </span>
              )}
              {completedCount === agents.length && agents.length > 0 && (
                <span className="text-emerald-400 font-mono text-[11px] mt-2 block">
                  All tasks completed successfully.
                </span>
              )}
            </div>
          )}

          {isPlanning && visibleCount < planningLines.length && (
            <div className="mt-2">
              <span className="text-emerald-500/60 font-mono text-xs">{'> '}</span>
              <span className="typewriter-cursor" />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function truncate(value: string, limit: number): string {
  if (value.length <= limit) {
    return value
  }
  return `${value.slice(0, limit - 3)}...`
}
