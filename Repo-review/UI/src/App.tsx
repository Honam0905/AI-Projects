import { useAppStore } from './store'
import Sidebar from './components/Sidebar'
import ChatArea from './components/ChatArea'
import AgentSwarmView from './components/AgentSwarmView'

export default function App() {
  const { isDarkMode, isSwarmActive, viewingWork } = useAppStore()

  const showRightPanel = isSwarmActive || viewingWork !== null

  return (
    <div className={isDarkMode ? 'dark' : ''}>
      <div className="flex h-screen bg-white dark:bg-surface-950 transition-colors duration-300">
        {/* Sidebar - hidden during swarm */}
        {!showRightPanel && (
          <div className="flex-shrink-0">
            <Sidebar />
          </div>
        )}

        {/* Chat area */}
        <div
          className={`flex-1 min-w-0 transition-all duration-300 ${
            showRightPanel ? 'max-w-[50%]' : ''
          }`}
        >
          <ChatArea />
        </div>

        {/* Agent swarm panel */}
        {showRightPanel && (
          <div className="w-1/2 flex-shrink-0 border-l border-zinc-200 dark:border-zinc-800">
            <AgentSwarmView />
          </div>
        )}
      </div>
    </div>
  )
}
