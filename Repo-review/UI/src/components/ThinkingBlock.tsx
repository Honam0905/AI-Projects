import { useEffect, useRef, useState } from 'react'
import { Brain, ChevronDown } from 'lucide-react'
import type { ThinkingData } from '../types'

interface ThinkingBlockProps {
  thinking: ThinkingData
}

export default function ThinkingBlock({ thinking }: ThinkingBlockProps) {
  const [isExpanded, setIsExpanded] = useState(thinking.isActive)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    setIsExpanded(thinking.isActive)
  }, [thinking.isActive])

  useEffect(() => {
    if (scrollRef.current && isExpanded) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [thinking.steps.length, isExpanded])

  if (thinking.isActive) {
    return (
      <div className="animate-fade-in-up">
        <div className="flex gap-3">
          <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-amber-500 dark:bg-amber-500 flex items-center justify-center">
            <Brain size={16} className="text-white" />
          </div>
          <div className="flex-1">
            <div
              className="rounded-xl border p-3
                bg-amber-50/50 dark:bg-amber-900/10
                border-amber-200/50 dark:border-amber-800/30"
            >
              <div className="flex items-center gap-2 mb-2">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
                <span className="text-[11px] font-semibold uppercase tracking-wider text-amber-600 dark:text-amber-400">
                  Supervisor Thinking
                </span>
              </div>
              <div ref={scrollRef} className="max-h-[200px] overflow-y-auto space-y-1">
                {thinking.steps.map((step, index) => (
                  <div
                    key={`${step}-${index}`}
                    className="text-xs text-amber-800/80 dark:text-amber-300/70 font-mono leading-relaxed"
                  >
                    {step}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="animate-fade-in-up">
      <button
        onClick={() => setIsExpanded((value) => !value)}
        className="flex items-center gap-2 px-1 py-1 group w-full text-left"
      >
        <div className="flex items-center gap-2 flex-1">
          <div className="h-px flex-1 bg-zinc-200 dark:bg-zinc-800" />
          <span className="text-[11px] font-medium text-zinc-400 dark:text-zinc-500 whitespace-nowrap">
            Thinking finished ({thinking.durationMs}ms)
          </span>
          <ChevronDown
            size={12}
            className={`text-zinc-400 dark:text-zinc-500 transition-transform duration-200 ${
              isExpanded ? 'rotate-180' : ''
            }`}
          />
          <div className="h-px flex-1 bg-zinc-200 dark:bg-zinc-800" />
        </div>
      </button>

      {isExpanded && (
        <div className="mt-2 ml-11">
          <div
            className="rounded-xl border p-3
              bg-zinc-50 dark:bg-zinc-800/30
              border-zinc-200/50 dark:border-zinc-700/30"
          >
            <div className="space-y-1 max-h-[180px] overflow-y-auto">
              {thinking.steps.map((step, index) => (
                <div
                  key={`${step}-${index}`}
                  className="text-xs text-zinc-500 dark:text-zinc-400 font-mono leading-relaxed"
                >
                  {step}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
