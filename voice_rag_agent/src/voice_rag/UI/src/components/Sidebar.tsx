import { useState } from 'react';
import { 
  MessageSquare, 
  Folder, 
  Plus, 
  Search, 
  ChevronDown,
  Cpu,
  X
} from 'lucide-react';

interface Project {
  id: string;
  name: string;
  chats: Chat[];
}

interface Chat {
  id: string;
  title: string;
  projectId: string;
}

interface SidebarProps {
  projects: Project[];
  activeChatId: string | null;
  activeProjectId: string | null;
  onNewProject: (name: string) => void;
  onNewChat: (projectId: string) => void;
  onSelectChat: (chatId: string, projectId: string) => void;
  onDeleteChat: (chatId: string, projectId: string) => void;
  onDeleteProject: (projectId: string) => void;
}

const Sidebar = ({ 
  projects, 
  activeChatId, 
  activeProjectId,
  onNewProject, 
  onNewChat, 
  onSelectChat,
  onDeleteChat,
  onDeleteProject
}: SidebarProps) => {
  const [isCreatingProject, setIsCreatingProject] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [expandedProjects, setExpandedProjects] = useState<Set<string>>(new Set());

  const handleCreateProject = () => {
    if (newProjectName.trim()) {
      onNewProject(newProjectName.trim());
      setNewProjectName('');
      setIsCreatingProject(false);
    }
  };

  const toggleProject = (projectId: string) => {
    setExpandedProjects(prev => {
      const newSet = new Set(prev);
      if (newSet.has(projectId)) {
        newSet.delete(projectId);
      } else {
        newSet.add(projectId);
      }
      return newSet;
    });
  };

  const handleNewChat = () => {
    const defaultProject = projects[0];
    if (defaultProject) {
      onNewChat(defaultProject.id);
      setExpandedProjects(prev => new Set(prev).add(defaultProject.id));
    }
  };

  return (
    <aside className="w-[300px] h-screen bg-[#050508] border-r border-[#00f0ff]/30 flex flex-col scanlines">
      {/* Logo Section */}
      <div className="p-4 flex items-center gap-3 border-b border-[#00f0ff]/30">
        <div 
          className="flex items-center gap-2 px-3 py-2 cyber-btn"
        >
          <Cpu className="w-5 h-5 text-[#00f0ff]" />
          <span className="text-[#00f0ff] text-xs tracking-widest cyber-glow-cyan">NEURAL</span>
          <ChevronDown className="w-3 h-3 text-[#00f0ff]" />
        </div>
        <div className="flex-1" />
        <button className="w-8 h-8 cyber-btn flex items-center justify-center">
          <Search className="w-4 h-4 text-[#00f0ff]" />
        </button>
      </div>

      {/* New Chat Button */}
      <div className="px-3 py-3">
        <button 
          onClick={handleNewChat}
          className="w-full flex items-center gap-3 px-4 py-3 cyber-btn"
        >
          <MessageSquare className="w-4 h-4 text-[#00f0ff]" />
          <span className="text-[#00f0ff] text-xs tracking-widest">NEW CHAT</span>
        </button>
      </div>

      {/* Projects Section */}
      <div className="px-3 py-2 flex-1 overflow-y-auto">
        <div className="flex items-center justify-between mb-3">
          <span className="text-[10px] text-[#ff00ff] tracking-widest cyber-glow-pink">PROJECTS</span>
          <button 
            onClick={() => setIsCreatingProject(true)}
            className="w-6 h-6 cyber-btn flex items-center justify-center"
          >
            <Plus className="w-3 h-3 text-[#00f0ff]" />
          </button>
        </div>

        {/* Create Project Input */}
        {isCreatingProject && (
          <div className="mb-3 p-3 border border-[#00f0ff]/50 bg-[#00f0ff]/5">
            <input
              type="text"
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value)}
              placeholder="PROJECT NAME..."
              className="w-full bg-transparent text-[#00f0ff] text-xs outline-none placeholder-[#00f0ff]/30"
              style={{ fontFamily: "'Share Tech Mono', monospace" }}
              autoFocus
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleCreateProject();
                if (e.key === 'Escape') {
                  setIsCreatingProject(false);
                  setNewProjectName('');
                }
              }}
            />
            <div className="flex gap-2 mt-2">
              <button 
                onClick={handleCreateProject}
                className="flex-1 py-1 text-[10px] bg-[#00ff88]/20 text-[#00ff88] border border-[#00ff88]/50 hover:bg-[#00ff88]/30"
              >
                CREATE
              </button>
              <button 
                onClick={() => {
                  setIsCreatingProject(false);
                  setNewProjectName('');
                }}
                className="flex-1 py-1 text-[10px] bg-[#ff0040]/20 text-[#ff0040] border border-[#ff0040]/50 hover:bg-[#ff0040]/30"
              >
                CANCEL
              </button>
            </div>
          </div>
        )}

        {/* Projects List */}
        <div className="space-y-2">
          {projects.map((project) => (
            <div key={project.id} className="border border-[#1a1a2e]">
              {/* Project Header */}
              <div 
                className="flex items-center gap-2 px-3 py-2 cursor-pointer hover:bg-[#00f0ff]/5 transition-colors"
                onClick={() => toggleProject(project.id)}
              >
                <Folder className="w-4 h-4 text-[#b829dd]" />
                <span className="flex-1 text-[#e0e0e0] text-xs truncate">{project.name}</span>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onNewChat(project.id);
                    setExpandedProjects(prev => new Set(prev).add(project.id));
                  }}
                  className="w-5 h-5 flex items-center justify-center hover:bg-[#00f0ff]/20"
                >
                  <Plus className="w-3 h-3 text-[#00f0ff]" />
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteProject(project.id);
                  }}
                  className="w-5 h-5 flex items-center justify-center hover:bg-[#ff0040]/20"
                >
                  <X className="w-3 h-3 text-[#ff0040]" />
                </button>
              </div>

              {/* Project Chats */}
              {expandedProjects.has(project.id) && project.chats.length > 0 && (
                <div className="border-t border-[#1a1a2e]">
                  {project.chats.map((chat) => (
                    <div
                      key={chat.id}
                      className={`flex items-center ${
                        activeChatId === chat.id && activeProjectId === project.id
                          ? 'bg-gradient-to-r from-[#00f0ff]/20 to-transparent border-l-2 border-[#00f0ff]'
                          : 'hover:bg-[#00f0ff]/5'
                      }`}
                    >
                      <button
                        onClick={() => onSelectChat(chat.id, project.id)}
                        className="flex-1 flex items-center gap-2 px-6 py-2 text-left transition-all"
                      >
                        <MessageSquare className={`w-3 h-3 ${
                          activeChatId === chat.id && activeProjectId === project.id
                            ? 'text-[#00f0ff]'
                            : 'text-[#666]'
                        }`} />
                        <span className={`text-[10px] truncate ${
                          activeChatId === chat.id && activeProjectId === project.id
                            ? 'text-[#00f0ff] cyber-glow-cyan'
                            : 'text-[#888]'
                        }`}>
                          {chat.title}
                        </span>
                      </button>
                      <button
                        onClick={() => onDeleteChat(chat.id, project.id)}
                        className="w-6 h-6 mr-2 flex items-center justify-center hover:bg-[#ff0040]/20"
                      >
                        <X className="w-3 h-3 text-[#ff0040]" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* User Profile */}
      <div className="p-3 border-t border-[#00f0ff]/30">
        <div className="flex items-center gap-3 px-3 py-2 border border-[#1a1a2e]">
          <div 
            className="w-8 h-8 flex items-center justify-center"
            style={{ 
              background: 'linear-gradient(135deg, #00f0ff, #b829dd)',
              boxShadow: '0 0 10px rgba(0, 240, 255, 0.5)'
            }}
          >
            <span className="text-black text-[10px] font-bold">KV</span>
          </div>
          <div className="flex-1">
            <div className="text-xs text-[#00f0ff] cyber-glow-cyan">KiffatorZ9</div>
            <div className="text-[10px] text-[#b829dd]">ONLINE</div>
          </div>
          <div className="w-2 h-2 rounded-full bg-[#00ff88] animate-pulse" />
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
