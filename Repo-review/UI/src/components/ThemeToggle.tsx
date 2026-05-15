import { Moon, Sun } from 'lucide-react'
import { useAppStore } from '../store'

export default function ThemeToggle() {
  const { isDarkMode, toggleDarkMode } = useAppStore()

  return (
    <button
      onClick={toggleDarkMode}
      className="relative flex items-center justify-center w-9 h-9 rounded-xl
        bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700
        transition-colors duration-200"
      aria-label="Toggle theme"
    >
      {isDarkMode ? (
        <Sun size={16} className="text-amber-400" />
      ) : (
        <Moon size={16} className="text-zinc-600" />
      )}
    </button>
  )
}
