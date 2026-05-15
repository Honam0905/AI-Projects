import { Bot, User, Monitor } from 'lucide-react'
import type { Message } from '../types'
import StatusMatrix from './StatusMatrix'
import AgentAvatar from './AgentAvatar'
import ThinkingBlock from './ThinkingBlock'

interface ChatMessageProps {
  message: Message
  onViewWork?: () => void
}

export default function ChatMessage({ message, onViewWork }: ChatMessageProps) {
  const isUser = message.role === 'user'
  const hasThinking = message.thinking && message.thinking.steps.length > 0
  const showContent = !message.thinking?.isActive

  return (
    <div className="space-y-3 animate-fade-in-up">
      {/* Thinking block */}
      {hasThinking && <ThinkingBlock thinking={message.thinking!} />}

      {/* Main message */}
      {showContent && (
        <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
          <div
            className={`flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center ${
              isUser
                ? 'bg-zinc-900 dark:bg-zinc-100'
                : 'bg-emerald-600 dark:bg-emerald-500'
            }`}
          >
            {isUser ? (
              <User size={16} className="text-white dark:text-zinc-900" />
            ) : (
              <Bot size={16} className="text-white" />
            )}
          </div>

          <div className={`flex-1 max-w-[85%] ${isUser ? 'text-right' : ''}`}>
            <div
              className={`inline-block text-left rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
                isUser
                  ? 'bg-zinc-900 dark:bg-zinc-100 text-white dark:text-zinc-900'
                  : 'bg-zinc-100 dark:bg-zinc-800 text-zinc-800 dark:text-zinc-200'
              }`}
            >
              <div className="whitespace-pre-wrap">{message.content}</div>
            </div>

            {message.agentWork && message.agentWork.completed && (
              <div className="mt-3 space-y-3">
                <button
                  onClick={onViewWork}
                  className="flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-medium
                    bg-emerald-50 dark:bg-emerald-900/20
                    text-emerald-700 dark:text-emerald-400
                    border border-emerald-200 dark:border-emerald-800/50
                    hover:bg-emerald-100 dark:hover:bg-emerald-900/30
                    transition-colors"
                >
                  <Monitor size={14} />
                  View Agent Work
                </button>

                <div
                  className="rounded-xl border p-3 space-y-2
                    bg-zinc-50 dark:bg-zinc-800/50
                    border-zinc-200 dark:border-zinc-700/50"
                >
                  <div className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400 mb-2">
                    Agent Status
                  </div>
                  {message.agentWork.agents.map((agent) => (
                    <div
                      key={agent.id}
                      className="flex items-center justify-between gap-3"
                    >
                      <div className="flex items-center gap-2">
                        <AgentAvatar agentId={agent.id} label={agent.name} size={22} />
                        <span className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
                          {agent.name}
                        </span>
                      </div>
                      <StatusMatrix
                        progress={agent.progress}
                        cols={agent.totalSteps}
                        animated={false}
                      />
                    </div>
                  ))}
                </div>

                {message.agentWork.fixResult && (
                  <div
                    className="rounded-xl border p-3 space-y-2
                      bg-zinc-50 dark:bg-zinc-800/50
                      border-zinc-200 dark:border-zinc-700/50"
                  >
                    <div className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                      Patch
                    </div>
                    <div className="text-xs text-zinc-700 dark:text-zinc-300">
                      {message.agentWork.fixResult.appliedToLocalRepo
                        ? `Applied back to ${message.agentWork.fixResult.applyBackTarget || message.agentWork.repoPath || 'the local repo'}.`
                        : message.agentWork.fixResult.applyBackError
                          ? `Local repo update failed: ${message.agentWork.fixResult.applyBackError}`
                          : 'Patch is available in the sandbox output.'}
                    </div>
                    {message.agentWork.fixResult.diffText && (
                      <pre
                        className="rounded-lg bg-zinc-950 text-emerald-300 p-3 text-[11px] leading-relaxed overflow-x-auto"
                      >
                        {message.agentWork.fixResult.diffText.split('\n').slice(0, 18).join('\n')}
                      </pre>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
