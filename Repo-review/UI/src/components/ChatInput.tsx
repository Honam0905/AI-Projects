import { type ReactNode, useEffect, useRef, useState } from 'react'
import { ArrowUp, Box, FolderGit2, Github, Laptop, Sparkles, Wrench } from 'lucide-react'
import type { ExecutionIntent, ReviewTargetMode, SandboxBackend } from '../types'

interface ChatInputProps {
  onSend: (message: string) => void
  disabled?: boolean
  executionIntent: ExecutionIntent
  reviewTargetMode: ReviewTargetMode
  localRepoPath: string
  remoteRepoUrl: string
  reviewSandboxBackend: SandboxBackend
  reviewTargetExpanded: boolean
  reviewTargetSummary: string
  onExecutionIntentChange: (value: ExecutionIntent) => void
  onReviewTargetModeChange: (mode: ReviewTargetMode) => void
  onLocalRepoPathChange: (value: string) => void
  onRemoteRepoUrlChange: (value: string) => void
  onReviewSandboxBackendChange: (value: SandboxBackend) => void
  onToggleReviewTargetExpanded: (expanded: boolean) => void
}

export default function ChatInput({
  onSend,
  disabled,
  executionIntent,
  reviewTargetMode,
  localRepoPath,
  remoteRepoUrl,
  reviewSandboxBackend,
  reviewTargetExpanded,
  reviewTargetSummary,
  onExecutionIntentChange,
  onReviewTargetModeChange,
  onLocalRepoPathChange,
  onRemoteRepoUrlChange,
  onReviewSandboxBackendChange,
  onToggleReviewTargetExpanded,
}: ChatInputProps) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height =
        Math.min(textareaRef.current.scrollHeight, 160) + 'px'
    }
  }, [value])

  const handleSend = () => {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setValue('')
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="w-full space-y-3">
      {reviewTargetExpanded ? (
        <div
          className="rounded-2xl border px-3 py-3
            bg-zinc-50 dark:bg-zinc-900/70
            border-zinc-200 dark:border-zinc-800"
        >
          <div className="flex items-center justify-between gap-3 mb-3">
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.12em] text-zinc-500 dark:text-zinc-400">
                Review Target
              </div>
              <div className="text-xs text-zinc-500 dark:text-zinc-400 mt-1">
                Auto mode lets the supervisor decide. Review and Fix mode always use the selected repository source.
              </div>
            </div>
          </div>

          <div className="mb-4 space-y-2">
            <div className="text-[11px] font-semibold uppercase tracking-[0.12em] text-zinc-500 dark:text-zinc-400">
              Operation Mode
            </div>
            <div className="flex items-center gap-1 rounded-xl bg-white dark:bg-zinc-800 p-1 border border-zinc-200 dark:border-zinc-700">
              <TargetButton
                active={executionIntent === 'auto'}
                icon={<Sparkles size={14} />}
                label="Auto"
                onClick={() => onExecutionIntentChange('auto')}
              />
              <TargetButton
                active={executionIntent === 'review'}
                icon={<FolderGit2 size={14} />}
                label="Review"
                onClick={() => onExecutionIntentChange('review')}
              />
              <TargetButton
                active={executionIntent === 'fix'}
                icon={<Wrench size={14} />}
                label="Fix"
                onClick={() => onExecutionIntentChange('fix')}
              />
            </div>
            <p className="text-[11px] text-zinc-500 dark:text-zinc-400">
              Fix mode uses Docker, generates a patch, and applies supported changes back to the selected local repo.
            </p>
          </div>

          <div className="flex items-center justify-between gap-3 mb-3">
            <div className="text-[11px] font-semibold uppercase tracking-[0.12em] text-zinc-500 dark:text-zinc-400">
              Repository Source
            </div>
            <div className="flex items-center gap-1 rounded-xl bg-white dark:bg-zinc-800 p-1 border border-zinc-200 dark:border-zinc-700">
              <TargetButton
                active={reviewTargetMode === 'local'}
                icon={<FolderGit2 size={14} />}
                label="Local"
                onClick={() => onReviewTargetModeChange('local')}
              />
              <TargetButton
                active={reviewTargetMode === 'github'}
                icon={<Github size={14} />}
                label="GitHub"
                onClick={() => onReviewTargetModeChange('github')}
              />
            </div>
          </div>

          {reviewTargetMode === 'local' ? (
            <div className="space-y-2">
              <input
                value={localRepoPath}
                onChange={(e) => onLocalRepoPathChange(e.target.value)}
                disabled={disabled}
                placeholder="Local repository path. Leave blank to use the backend default."
                className="w-full rounded-xl border px-3 py-2.5 text-sm
                  bg-white dark:bg-zinc-800
                  border-zinc-200 dark:border-zinc-700
                  text-zinc-900 dark:text-zinc-100
                  placeholder:text-zinc-400 dark:placeholder:text-zinc-500
                  focus:outline-none focus:border-zinc-400 dark:focus:border-zinc-500"
              />
              <p className="text-[11px] text-zinc-500 dark:text-zinc-400">
                Use an absolute local path, or leave this empty to review the configured default repo.
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              <input
                value={remoteRepoUrl}
                onChange={(e) => onRemoteRepoUrlChange(e.target.value)}
                disabled={disabled}
                placeholder="https://github.com/owner/repo or /pull/123"
                className="w-full rounded-xl border px-3 py-2.5 text-sm
                  bg-white dark:bg-zinc-800
                  border-zinc-200 dark:border-zinc-700
                  text-zinc-900 dark:text-zinc-100
                  placeholder:text-zinc-400 dark:placeholder:text-zinc-500
                  focus:outline-none focus:border-zinc-400 dark:focus:border-zinc-500"
              />
              <p className="text-[11px] text-zinc-500 dark:text-zinc-400">
                Review mode can clone GitHub repos or PR heads directly. Fix mode requires the same repo to already exist as a local clone.
              </p>
            </div>
          )}

          <div className="mt-4 space-y-2">
            <div className="text-[11px] font-semibold uppercase tracking-[0.12em] text-zinc-500 dark:text-zinc-400">
              Execution Sandbox
            </div>
            <div className="flex items-center gap-1 rounded-xl bg-white dark:bg-zinc-800 p-1 border border-zinc-200 dark:border-zinc-700">
              <TargetButton
                active={reviewSandboxBackend === 'local'}
                icon={<Laptop size={14} />}
                label="Local"
                onClick={() => onReviewSandboxBackendChange('local')}
                disabled={executionIntent === 'fix'}
              />
              <TargetButton
                active={reviewSandboxBackend === 'docker'}
                icon={<Box size={14} />}
                label="Docker"
                onClick={() => onReviewSandboxBackendChange('docker')}
              />
            </div>
            <p className="text-[11px] text-zinc-500 dark:text-zinc-400">
              Local is a lightweight development backend. Docker is the isolated container backend and is required for Fix mode.
            </p>
          </div>
        </div>
      ) : (
        <div
          className="flex items-center justify-between gap-3 rounded-2xl border px-3 py-2.5
            bg-zinc-50 dark:bg-zinc-900/70
            border-zinc-200 dark:border-zinc-800"
        >
          <div className="min-w-0">
            <div className="text-[11px] font-semibold uppercase tracking-[0.12em] text-zinc-500 dark:text-zinc-400">
              Review Target
            </div>
            <div className="truncate text-sm text-zinc-700 dark:text-zinc-200">
              {reviewTargetSummary}
            </div>
          </div>
          <button
            type="button"
            onClick={() => onToggleReviewTargetExpanded(true)}
            disabled={disabled}
            className="rounded-lg border border-zinc-200 dark:border-zinc-700 px-2.5 py-1.5 text-xs font-medium
              text-zinc-600 dark:text-zinc-300
              hover:bg-white dark:hover:bg-zinc-800
              disabled:opacity-50"
          >
            Change
          </button>
        </div>
      )}

      <div className="relative">
        <div
          className="flex items-end rounded-2xl border
          bg-white dark:bg-zinc-800
          border-zinc-200 dark:border-zinc-700
          shadow-sm dark:shadow-none
          transition-colors duration-200
          focus-within:border-zinc-400 dark:focus-within:border-zinc-500"
      >
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          placeholder="Send a message..."
          rows={1}
          className="flex-1 resize-none bg-transparent px-4 py-3 pr-12
            text-sm text-zinc-900 dark:text-zinc-100
            placeholder:text-zinc-400 dark:placeholder:text-zinc-500
            focus:outline-none disabled:opacity-50
            max-h-40"
        />
        <button
          onClick={handleSend}
          disabled={!value.trim() || disabled}
          className="absolute right-2 bottom-2 p-1.5 rounded-lg
            bg-zinc-900 dark:bg-zinc-100
            text-white dark:text-zinc-900
            hover:bg-zinc-700 dark:hover:bg-zinc-300
            disabled:opacity-30 disabled:cursor-not-allowed
            transition-all duration-150"
        >
          <ArrowUp size={16} strokeWidth={2.5} />
        </button>
      </div>
      </div>
    </div>
  )
}

interface TargetButtonProps {
  active: boolean
  icon: ReactNode
  label: string
  onClick: () => void
  disabled?: boolean
}

function TargetButton({ active, icon, label, onClick, disabled }: TargetButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium transition-colors ${
        active
          ? 'bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900'
          : 'text-zinc-600 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-700'
      } ${disabled ? 'opacity-40 cursor-not-allowed hover:bg-transparent dark:hover:bg-transparent' : ''}`}
    >
      {icon}
      {label}
    </button>
  )
}
