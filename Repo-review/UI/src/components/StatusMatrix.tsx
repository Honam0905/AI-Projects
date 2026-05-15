import { useEffect, useState } from 'react'

interface StatusMatrixProps {
  progress: number
  rows?: number
  cols?: number
  size?: 'sm' | 'md'
  animated?: boolean
}

export default function StatusMatrix({
  progress,
  rows = 3,
  cols = 14,
  size = 'sm',
  animated = true,
}: StatusMatrixProps) {
  const [displayProgress, setDisplayProgress] = useState(animated ? 0 : progress)

  useEffect(() => {
    if (!animated) {
      setDisplayProgress(progress)
      return
    }
    if (progress > displayProgress) {
      const timer = setTimeout(
        () => setDisplayProgress((p) => Math.min(p + 1, progress)),
        60,
      )
      return () => clearTimeout(timer)
    }
    setDisplayProgress(progress)
  }, [progress, displayProgress, animated])

  const cellPx = size === 'sm' ? 6 : 8
  const gap = size === 'sm' ? 2 : 3

  return (
    <div
      className="inline-grid"
      style={{
        gridTemplateRows: `repeat(${rows}, ${cellPx}px)`,
        gridAutoFlow: 'column',
        gridAutoColumns: `${cellPx}px`,
        gap: `${gap}px`,
      }}
    >
      {Array.from({ length: cols * rows }, (_, i) => {
        const col = Math.floor(i / rows)
        const isActive = col < displayProgress

        return (
          <div
            key={i}
            className={`rounded-[1px] transition-all ${
              isActive ? 'matrix-cell-active' : 'matrix-cell-inactive'
            }`}
            style={{
              width: cellPx,
              height: cellPx,
              transitionDuration: '250ms',
              transitionDelay: animated && isActive ? `${col * 30}ms` : '0ms',
            }}
          />
        )
      })}
    </div>
  )
}
