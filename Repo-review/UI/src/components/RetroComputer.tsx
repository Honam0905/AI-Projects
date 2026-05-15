import { useEffect, useRef } from 'react'

interface RetroComputerProps {
  agentName: string
  logs: string[]
  status: string
  patchPreview?: string
  patchTarget?: string | null
}

export default function RetroComputer({
  agentName,
  logs,
  status,
  patchPreview,
  patchTarget,
}: RetroComputerProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [logs])

  return (
    <div className="flex flex-col h-full">
      <div
        className="flex-1 min-h-0 flex flex-col rounded-2xl overflow-hidden
          bg-zinc-800 dark:bg-zinc-900
          shadow-[0_8px_32px_rgba(0,0,0,0.3),inset_0_1px_0_rgba(255,255,255,0.05)]
          border border-zinc-700/50"
      >
        {/* Title bar */}
        <div className="flex items-center gap-2 px-4 py-2.5 bg-zinc-750 dark:bg-[#1e1e1e] border-b border-zinc-700/40 flex-shrink-0">
          <div className="flex gap-[6px]">
            <div className="w-[11px] h-[11px] rounded-full bg-[#ff5f57] shadow-[inset_0_-1px_1px_rgba(0,0,0,0.2)]" />
            <div className="w-[11px] h-[11px] rounded-full bg-[#febc2e] shadow-[inset_0_-1px_1px_rgba(0,0,0,0.2)]" />
            <div className="w-[11px] h-[11px] rounded-full bg-[#28c840] shadow-[inset_0_-1px_1px_rgba(0,0,0,0.2)]" />
          </div>
          <div className="flex-1 text-center">
            <span className="text-[11px] font-medium text-zinc-400 font-mono">
              {agentName} — Terminal
            </span>
          </div>
          <div className="w-[56px]" />
        </div>

        {/* Screen */}
        <div
          ref={scrollRef}
          className="retro-screen flex-1 min-h-0 overflow-y-auto p-4 bg-[#0c0c0c] dark:bg-[#080808]"
        >
          <div className="mb-3 pb-2 border-b border-emerald-900/40">
            <span className="text-emerald-500/70 font-mono text-[11px]">
              agent@swarm ~ % sandbox.exec --worker "{agentName}"
            </span>
          </div>

          <div className="flex items-center gap-2 mb-3">
            <div
              className={`w-2 h-2 rounded-full ${
                status === 'completed'
                  ? 'bg-emerald-400'
                  : status === 'failed'
                    ? 'bg-red-400'
                    : 'bg-emerald-400 animate-pulse'
              }`}
            />
            <span
              className={`font-mono text-xs ${
                status === 'failed'
                  ? 'text-red-400/80'
                  : 'text-emerald-400/80'
              }`}
            >
              {status === 'completed'
                ? 'Task completed'
                : status === 'failed'
                  ? 'Task failed'
                  : 'Executing task...'}
            </span>
          </div>

          <div className="space-y-[6px]">
            {logs.length === 0 && (
              <div className="text-emerald-300/50 font-mono text-[12px] leading-relaxed">
                Waiting for streamed tool events...
              </div>
            )}
            {logs.map((log, i) => (
              <div
                key={`${log}-${i}`}
                className="text-emerald-300/90 font-mono text-[12px] leading-relaxed"
              >
                {log}
              </div>
            ))}
          </div>

          {status !== 'completed' && logs.length > 0 && (
            <div className="mt-2">
              <span className="text-emerald-500/60 font-mono text-xs">{'> '}</span>
              <span className="typewriter-cursor" />
            </div>
          )}

          {status === 'completed' && (
            <div className="mt-3 pt-2 border-t border-emerald-900/40">
              <span className="text-emerald-400 font-mono text-xs">
                Process exited with code 0
              </span>
            </div>
          )}
          {status === 'failed' && (
            <div className="mt-3 pt-2 border-t border-red-900/40">
              <span className="text-red-400 font-mono text-xs">
                Process exited with errors
              </span>
            </div>
          )}

          {patchPreview && (
            <div className="mt-4 pt-3 border-t border-emerald-900/40 space-y-2">
              <div className="text-emerald-400 font-mono text-xs">
                Patch Preview{patchTarget ? ` -> ${patchTarget}` : ''}
              </div>
              <pre className="overflow-x-auto text-emerald-300/90 font-mono text-[11px] leading-relaxed whitespace-pre-wrap">
                {patchPreview}
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
