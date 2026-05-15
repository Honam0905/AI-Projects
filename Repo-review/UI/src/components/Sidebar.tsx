import { useState, useMemo, useRef } from 'react'
import {
  Plus,
  MessageSquare,
  Trash2,
  Search,
  FolderClosed,
  FolderOpen,
  ChevronDown,
  X,
  FolderPlus,
} from 'lucide-react'
import { useAppStore } from '../store'
import ThemeToggle from './ThemeToggle'

export default function Sidebar() {
  const {
    sessions,
    activeSessionId,
    createSession,
    setActiveSession,
    deleteSession,
    projects,
    createProject,
    deleteProject,
    renameProject,
    expandedProjectIds,
    toggleProjectExpanded,
    moveSessionToProject,
    searchQuery,
    setSearchQuery,
  } = useAppStore()

  const [isSearchOpen, setIsSearchOpen] = useState(false)
  const [isCreatingProject, setIsCreatingProject] = useState(false)
  const [newProjectName, setNewProjectName] = useState('')
  const [editingProjectId, setEditingProjectId] = useState<string | null>(null)
  const [editingName, setEditingName] = useState('')
  const [dragOverTarget, setDragOverTarget] = useState<string | null>(null)
  const dragSessionId = useRef<string | null>(null)

  const filteredSessions = useMemo(() => {
    if (!searchQuery.trim()) return null
    const q = searchQuery.toLowerCase()
    return sessions.filter(
      (s) =>
        s.title.toLowerCase().includes(q) ||
        s.messages.some((m) => m.content.toLowerCase().includes(q)),
    )
  }, [searchQuery, sessions])

  const isSearchActive = isSearchOpen && searchQuery.trim().length > 0
  const looseSessions = sessions.filter((s) => !s.projectId)

  const handleCreateProject = () => {
    const name = newProjectName.trim()
    if (!name) return
    createProject(name)
    setNewProjectName('')
    setIsCreatingProject(false)
  }

  const handleRenameProject = (id: string) => {
    const name = editingName.trim()
    if (name) renameProject(id, name)
    setEditingProjectId(null)
    setEditingName('')
  }

  const handleDragStart = (sessionId: string) => {
    dragSessionId.current = sessionId
  }

  const handleDragEnd = () => {
    dragSessionId.current = null
    setDragOverTarget(null)
  }

  const handleDropOnProject = (projectId: string) => {
    if (dragSessionId.current) {
      moveSessionToProject(dragSessionId.current, projectId)
      const store = useAppStore.getState()
      if (!store.expandedProjectIds.includes(projectId)) {
        toggleProjectExpanded(projectId)
      }
    }
    dragSessionId.current = null
    setDragOverTarget(null)
  }

  const handleDropOnLoose = () => {
    if (dragSessionId.current) {
      moveSessionToProject(dragSessionId.current, undefined)
    }
    dragSessionId.current = null
    setDragOverTarget(null)
  }

  const dragOverHandler = (e: React.DragEvent, targetId: string) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
    setDragOverTarget(targetId)
  }

  return (
    <div
      className="w-[260px] flex flex-col h-full
        bg-zinc-50 dark:bg-surface-900
        border-r border-zinc-200 dark:border-zinc-800"
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 pt-3 pb-1">
        <span className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">
          Chats
        </span>
        <ThemeToggle />
      </div>

      {/* Action buttons */}
      <div className="px-3 py-2 space-y-1">
        <button
          onClick={() => createSession()}
          className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm
            text-zinc-700 dark:text-zinc-300
            hover:bg-zinc-200/60 dark:hover:bg-zinc-800
            transition-colors duration-100"
        >
          <Plus size={16} strokeWidth={2} className="text-zinc-500 dark:text-zinc-400" />
          New chat
        </button>

        <button
          onClick={() => {
            setIsSearchOpen(!isSearchOpen)
            if (isSearchOpen) setSearchQuery('')
          }}
          className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm
            transition-colors duration-100
            ${isSearchOpen
              ? 'bg-zinc-200/60 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100'
              : 'text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200/60 dark:hover:bg-zinc-800'
            }`}
        >
          <Search size={16} strokeWidth={2} className="text-zinc-500 dark:text-zinc-400" />
          Search
        </button>

        <button
          onClick={() => setIsCreatingProject(!isCreatingProject)}
          className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm
            text-zinc-700 dark:text-zinc-300
            hover:bg-zinc-200/60 dark:hover:bg-zinc-800
            transition-colors duration-100"
        >
          <FolderPlus size={16} strokeWidth={2} className="text-zinc-500 dark:text-zinc-400" />
          Projects
        </button>
      </div>

      {/* Search input */}
      {isSearchOpen && (
        <div className="px-3 pb-2">
          <div className="relative">
            <Search
              size={14}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-400"
            />
            <input
              autoFocus
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search chats..."
              className="w-full pl-8 pr-8 py-1.5 text-xs rounded-lg
                bg-white dark:bg-zinc-800
                border border-zinc-200 dark:border-zinc-700
                text-zinc-800 dark:text-zinc-200
                placeholder:text-zinc-400 dark:placeholder:text-zinc-500
                focus:outline-none focus:ring-1 focus:ring-zinc-300 dark:focus:ring-zinc-600"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="absolute right-2 top-1/2 -translate-y-1/2 p-0.5 rounded hover:bg-zinc-200 dark:hover:bg-zinc-700"
              >
                <X size={12} className="text-zinc-400" />
              </button>
            )}
          </div>
        </div>
      )}

      {/* New project input */}
      {isCreatingProject && (
        <div className="px-3 pb-2">
          <div className="flex gap-1">
            <input
              autoFocus
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleCreateProject()
                if (e.key === 'Escape') setIsCreatingProject(false)
              }}
              placeholder="Project name..."
              className="flex-1 px-2.5 py-1.5 text-xs rounded-lg
                bg-white dark:bg-zinc-800
                border border-zinc-200 dark:border-zinc-700
                text-zinc-800 dark:text-zinc-200
                placeholder:text-zinc-400
                focus:outline-none focus:ring-1 focus:ring-zinc-300 dark:focus:ring-zinc-600"
            />
            <button
              onClick={handleCreateProject}
              disabled={!newProjectName.trim()}
              className="px-2 py-1 text-[11px] font-medium rounded-lg
                bg-zinc-900 dark:bg-zinc-100
                text-white dark:text-zinc-900
                disabled:opacity-30
                hover:bg-zinc-700 dark:hover:bg-zinc-300
                transition-colors"
            >
              Create
            </button>
          </div>
        </div>
      )}

      <div className="mx-3 border-t border-zinc-200 dark:border-zinc-800" />

      {/* Session/Search Results */}
      <div className="flex-1 overflow-y-auto px-2 py-2 space-y-0.5">
        {isSearchActive && filteredSessions ? (
          <>
            <div className="px-2 py-1">
              <span className="text-[11px] font-medium text-zinc-400 dark:text-zinc-500">
                {filteredSessions.length} result{filteredSessions.length !== 1 ? 's' : ''} for "{searchQuery}"
              </span>
            </div>
            {filteredSessions.length === 0 && (
              <div className="px-3 py-6 text-center">
                <p className="text-xs text-zinc-400 dark:text-zinc-500">
                  No matches found
                </p>
              </div>
            )}
            {filteredSessions.map((session) => (
              <SessionItem
                key={session.id}
                session={session}
                isActive={session.id === activeSessionId}
                onSelect={() => setActiveSession(session.id)}
                onDelete={() => deleteSession(session.id)}
                highlight={searchQuery}
              />
            ))}
          </>
        ) : (
          <>
            {/* Projects as drop targets */}
            {projects.map((project) => {
              const isExpanded = expandedProjectIds.includes(project.id)
              const projectSessions = sessions.filter(
                (s) => s.projectId === project.id,
              )
              const isEditing = editingProjectId === project.id
              const isDragOver = dragOverTarget === project.id

              return (
                <div
                  key={project.id}
                  className="mb-1"
                  onDragOver={(e) => dragOverHandler(e, project.id)}
                  onDragLeave={() => setDragOverTarget(null)}
                  onDrop={() => handleDropOnProject(project.id)}
                >
                  <div
                    className={`group flex items-center rounded-lg transition-all duration-150
                      ${isDragOver
                        ? 'ring-2 ring-amber-400 dark:ring-amber-500 bg-amber-50/50 dark:bg-amber-900/15'
                        : ''
                      }`}
                  >
                    <button
                      onClick={() => toggleProjectExpanded(project.id)}
                      className="flex-1 flex items-center gap-2 px-2 py-1.5 rounded-lg text-sm
                        text-zinc-600 dark:text-zinc-400
                        hover:bg-zinc-100 dark:hover:bg-zinc-800/50
                        transition-colors"
                    >
                      {isExpanded || isDragOver ? (
                        <FolderOpen size={14} className="text-amber-500" />
                      ) : (
                        <FolderClosed size={14} className="text-zinc-400 dark:text-zinc-500" />
                      )}
                      {isEditing ? (
                        <input
                          autoFocus
                          value={editingName}
                          onChange={(e) => setEditingName(e.target.value)}
                          onKeyDown={(e) => {
                            e.stopPropagation()
                            if (e.key === 'Enter') handleRenameProject(project.id)
                            if (e.key === 'Escape') setEditingProjectId(null)
                          }}
                          onBlur={() => handleRenameProject(project.id)}
                          onClick={(e) => e.stopPropagation()}
                          className="flex-1 bg-transparent text-xs border-b border-zinc-400
                            focus:outline-none text-zinc-800 dark:text-zinc-200"
                        />
                      ) : (
                        <span className="flex-1 text-xs font-medium truncate text-left">
                          {project.name}
                        </span>
                      )}
                      <span className="text-[10px] text-zinc-400 dark:text-zinc-500">
                        {projectSessions.length}
                      </span>
                      <ChevronDown
                        size={12}
                        className={`text-zinc-400 transition-transform ${
                          isExpanded ? '' : '-rotate-90'
                        }`}
                      />
                    </button>
                    <div className="flex opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        onClick={() => createSession(project.id)}
                        className="p-1 rounded hover:bg-zinc-200 dark:hover:bg-zinc-700"
                        title="Add chat to project"
                      >
                        <Plus size={11} className="text-zinc-400" />
                      </button>
                      <button
                        onClick={() => {
                          setEditingProjectId(project.id)
                          setEditingName(project.name)
                        }}
                        className="p-1 rounded hover:bg-zinc-200 dark:hover:bg-zinc-700"
                        title="Rename"
                      >
                        <MessageSquare size={11} className="text-zinc-400" />
                      </button>
                      <button
                        onClick={() => deleteProject(project.id)}
                        className="p-1 rounded hover:bg-zinc-200 dark:hover:bg-zinc-700"
                        title="Delete project"
                      >
                        <Trash2 size={11} className="text-zinc-400" />
                      </button>
                    </div>
                  </div>

                  {(isExpanded || isDragOver) && (
                    <div className="ml-3 pl-2 border-l border-zinc-200 dark:border-zinc-800 space-y-0.5 mt-0.5">
                      {projectSessions.length === 0 && !isDragOver && (
                        <p className="text-[11px] text-zinc-400 dark:text-zinc-500 py-2 px-2">
                          No chats yet
                        </p>
                      )}
                      {isDragOver && projectSessions.length === 0 && (
                        <p className="text-[11px] text-amber-500 dark:text-amber-400 py-2 px-2 font-medium">
                          Drop here to move
                        </p>
                      )}
                      {projectSessions.map((session) => (
                        <SessionItem
                          key={session.id}
                          session={session}
                          isActive={session.id === activeSessionId}
                          onSelect={() => setActiveSession(session.id)}
                          onDelete={() => deleteSession(session.id)}
                          onDragStart={() => handleDragStart(session.id)}
                          onDragEnd={handleDragEnd}
                        />
                      ))}
                    </div>
                  )}
                </div>
              )
            })}

            {/* Loose sessions drop zone */}
            <div
              onDragOver={(e) => dragOverHandler(e, '__loose__')}
              onDragLeave={() => setDragOverTarget(null)}
              onDrop={handleDropOnLoose}
              className={`min-h-[20px] rounded-lg transition-all duration-150 ${
                dragOverTarget === '__loose__'
                  ? 'ring-2 ring-zinc-400 dark:ring-zinc-500 bg-zinc-100/50 dark:bg-zinc-800/30'
                  : ''
              }`}
            >
              {projects.length > 0 && looseSessions.length > 0 && (
                <div className="pt-1 mt-1">
                  <div className="px-2 py-1">
                    <span className="text-[10px] font-medium text-zinc-400 dark:text-zinc-500 uppercase tracking-wider">
                      Chats
                    </span>
                  </div>
                </div>
              )}
              {dragOverTarget === '__loose__' && looseSessions.length === 0 && projects.length > 0 && (
                <div className="px-3 py-2">
                  <p className="text-[11px] text-zinc-500 dark:text-zinc-400 font-medium">
                    Drop here to remove from project
                  </p>
                </div>
              )}
              {looseSessions.length === 0 && projects.length === 0 && (
                <div className="px-3 py-8 text-center">
                  <MessageSquare
                    size={28}
                    className="mx-auto mb-2 text-zinc-300 dark:text-zinc-600"
                  />
                  <p className="text-xs text-zinc-400 dark:text-zinc-500">
                    No conversations yet
                  </p>
                </div>
              )}
              {looseSessions.map((session) => (
                <SessionItem
                  key={session.id}
                  session={session}
                  isActive={session.id === activeSessionId}
                  onSelect={() => setActiveSession(session.id)}
                  onDelete={() => deleteSession(session.id)}
                  onDragStart={() => handleDragStart(session.id)}
                  onDragEnd={handleDragEnd}
                />
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  )
}

function SessionItem({
  session,
  isActive,
  onSelect,
  onDelete,
  highlight,
  onDragStart,
  onDragEnd,
}: {
  session: { id: string; title: string }
  isActive: boolean
  onSelect: () => void
  onDelete: () => void
  highlight?: string
  onDragStart?: () => void
  onDragEnd?: () => void
}) {
  const [isDragging, setIsDragging] = useState(false)
  const title = session.title

  const renderTitle = () => {
    if (!highlight) return title
    const idx = title.toLowerCase().indexOf(highlight.toLowerCase())
    if (idx === -1) return title
    return (
      <>
        {title.slice(0, idx)}
        <span className="bg-amber-200/60 dark:bg-amber-500/30 rounded px-0.5">
          {title.slice(idx, idx + highlight.length)}
        </span>
        {title.slice(idx + highlight.length)}
      </>
    )
  }

  return (
    <div
      draggable={!!onDragStart}
      onDragStart={(e) => {
        e.dataTransfer.effectAllowed = 'move'
        e.dataTransfer.setData('text/plain', session.id)
        setIsDragging(true)
        onDragStart?.()
      }}
      onDragEnd={() => {
        setIsDragging(false)
        onDragEnd?.()
      }}
      onClick={onSelect}
      className={`group flex items-center gap-2 px-2.5 py-1.5 rounded-lg
        transition-all duration-100
        ${onDragStart ? 'cursor-grab active:cursor-grabbing' : 'cursor-pointer'}
        ${isDragging ? 'opacity-40 scale-95' : ''}
        ${
          isActive
            ? 'bg-zinc-200 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100'
            : 'text-zinc-600 dark:text-zinc-400 hover:bg-zinc-100 dark:hover:bg-zinc-800/50'
        }`}
    >
      <MessageSquare size={13} className="flex-shrink-0 opacity-40" />
      <span className="flex-1 text-xs truncate">{renderTitle()}</span>
      <button
        onClick={(e) => {
          e.stopPropagation()
          onDelete()
        }}
        className="opacity-0 group-hover:opacity-60 hover:!opacity-100
          p-0.5 rounded transition-opacity"
      >
        <Trash2 size={11} />
      </button>
    </div>
  )
}
