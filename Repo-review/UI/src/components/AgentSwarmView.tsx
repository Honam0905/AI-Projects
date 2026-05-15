import { Cpu, X } from 'lucide-react'
import { formatRepoTargetLabel, getAgentStatusLabel } from '../lib/swarm'
import { useAppStore } from '../store'
import AgentAvatar from './AgentAvatar'
import RetroComputer from './RetroComputer'
import TaskScreen from './TaskScreen'

export default function AgentSwarmView() {
  const {
    activeSwarmRun,
    swarmAgents,
    selectedAgentId,
    setSelectedAgent,
    viewingWork,
    setViewingWork,
    computerScreen,
    completedTaskIds,
    setComputerScreen,
  } = useAppStore()

  const isViewingPast = viewingWork !== null
  const agents = isViewingPast ? viewingWork.agents : swarmAgents
  const activeId = selectedAgentId ?? agents[0]?.id
  const selectedAgent = agents.find((agent) => agent.id === activeId)
  const completedCount = isViewingPast ? agents.length : completedTaskIds.length
  const totalTasks = agents.length
  const query = isViewingPast ? viewingWork.query : activeSwarmRun?.query || ''
  const intent = isViewingPast ? viewingWork.intent : activeSwarmRun?.intent
  const routeReason = isViewingPast
    ? viewingWork.routeReason
    : activeSwarmRun?.routeReason
  const repoPath = isViewingPast ? viewingWork.repoPath : activeSwarmRun?.repoPath
  const repoSourceType = isViewingPast
    ? viewingWork.repoSourceType
    : activeSwarmRun?.repoSourceType
  const repoSource = isViewingPast ? viewingWork.repoSource : activeSwarmRun?.repoSource
  const repoTargetLabel = formatRepoTargetLabel(repoSourceType, repoSource, repoPath)

  const handleClose = () => {
    setViewingWork(null)
  }

  const showTaskScreen = computerScreen === 'planning' || computerScreen === 'task-review'

  return (
    <div className="flex flex-col h-full animate-slide-in-right bg-zinc-50 dark:bg-surface-950">
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-200 dark:border-zinc-800 flex-shrink-0">
        <div className="flex items-center gap-2">
          <Cpu size={16} className="text-emerald-500" />
          <span className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">
            Agent's Computer
          </span>
          {repoTargetLabel && (
            <span className="hidden xl:inline-flex items-center rounded-full bg-zinc-200 dark:bg-zinc-800 px-2 py-0.5 text-[10px] font-medium text-zinc-600 dark:text-zinc-300">
              {repoSourceType === 'remote' ? 'GitHub target' : 'Local target'}
            </span>
          )}
          {!isViewingPast && (
            <span className="flex items-center gap-1.5 ml-1 px-2 py-0.5 rounded-full bg-emerald-100 dark:bg-emerald-900/30">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-[10px] font-medium text-emerald-700 dark:text-emerald-400">
                Task Progress {completedCount}/{totalTasks}
              </span>
            </span>
          )}
          {isViewingPast && (
            <span className="ml-1 px-2 py-0.5 rounded-full bg-zinc-200 dark:bg-zinc-800">
              <span className="text-[10px] font-medium text-zinc-500 dark:text-zinc-400">
                {viewingWork.status === 'failed'
                  ? 'Completed with issues'
                  : `Completed ${totalTasks}/${totalTasks}`}
              </span>
            </span>
          )}
        </div>
        {isViewingPast && (
          <button
            onClick={handleClose}
            className="p-1.5 rounded-lg hover:bg-zinc-200 dark:hover:bg-zinc-800 transition-colors"
          >
            <X size={16} className="text-zinc-500" />
          </button>
        )}
      </div>

      <div className="flex-1 min-h-0 p-4 pb-2">
        {showTaskScreen && !selectedAgent ? null : showTaskScreen && !isViewingPast ? (
          <TaskScreen
            agents={agents}
            completedTaskIds={completedTaskIds}
            isPlanning={computerScreen === 'planning'}
            query={query}
            intent={intent}
            routeReason={routeReason}
            repoPath={repoPath}
            repoSourceType={repoSourceType}
            repoSource={repoSource}
          />
        ) : (
          selectedAgent && (
            <RetroComputer
              agentName={selectedAgent.name}
              logs={selectedAgent.logs}
              status={selectedAgent.status}
              patchPreview={isViewingPast ? viewingWork.fixResult?.diffText : undefined}
              patchTarget={isViewingPast ? viewingWork.fixResult?.applyBackTarget : undefined}
            />
          )
        )}
      </div>

      <div className="flex-shrink-0 border-t border-zinc-200 dark:border-zinc-800 px-3 py-3">
        <div className="flex items-stretch gap-1">
          {agents.map((agent, index) => {
            const isSelected = agent.id === activeId && computerScreen === 'agent-work'
            const isDone = completedTaskIds.includes(agent.id) || isViewingPast
            const statusLabel = isViewingPast
              ? agent.status === 'failed'
                ? 'Failed'
                : 'Completed'
              : getAgentStatusLabel(agent)

            return (
              <button
                key={agent.id}
                onClick={() => {
                  setSelectedAgent(agent.id)
                  if (computerScreen !== 'planning') {
                    setComputerScreen('agent-work')
                  }
                }}
                className={`flex-1 flex flex-col items-center gap-1.5 py-2.5 px-2 rounded-xl
                  transition-all duration-150 ${
                    isSelected
                      ? 'bg-zinc-200 dark:bg-zinc-800 ring-1 ring-zinc-300 dark:ring-zinc-700'
                      : 'hover:bg-zinc-100 dark:hover:bg-zinc-800/50'
                  }`}
              >
                <div className="flex items-center gap-1.5">
                  <AgentAvatar agentId={agent.id} label={agent.name} size={32} />
                  <span className="text-sm font-mono font-bold text-zinc-400 dark:text-zinc-500">
                    {String(index + 1).padStart(2, '0')}
                  </span>
                </div>
                <span
                  className={`text-[11px] font-medium ${
                    isDone
                      ? 'text-emerald-600 dark:text-emerald-400'
                      : 'text-zinc-500 dark:text-zinc-400'
                  }`}
                >
                  {statusLabel}
                </span>
              </button>
            )
          })}
        </div>
      </div>
    </div>
  )
}
