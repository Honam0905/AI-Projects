import { useEffect, useMemo, useRef, useState } from 'react';

import { createKb, sendChat, uploadDocuments } from '@/lib/api';
import type { Message, Project } from '@/types/chat';

import ChatHeader from './ChatHeader';
import ChatInput from './ChatInput';
import MessageBubble from './MessageBubble';
import Sidebar from './Sidebar';

const DEFAULT_PROJECT_ID = 'default';
const DEFAULT_CHAT_ID = 'chat-1';

interface PendingUpload {
  id: string;
  file: File;
}

const ChatPage = () => {
  const [projects, setProjects] = useState<Project[]>([
    {
      id: DEFAULT_PROJECT_ID,
      name: 'GENERAL',
      kbId: null,
      chats: [
        {
          id: DEFAULT_CHAT_ID,
          title: 'Session 01',
          projectId: DEFAULT_PROJECT_ID,
          messages: [],
        },
      ],
    },
  ]);

  const [activeProjectId, setActiveProjectId] = useState<string>(DEFAULT_PROJECT_ID);
  const [activeChatId, setActiveChatId] = useState<string>(DEFAULT_CHAT_ID);
  const [isBusy, setIsBusy] = useState(false);
  const [pendingUploadsByChat, setPendingUploadsByChat] = useState<
    Record<string, PendingUpload[]>
  >({});

  const projectsRef = useRef<Project[]>(projects);
  const pendingUploadsRef = useRef<Record<string, PendingUpload[]>>(
    pendingUploadsByChat,
  );

  useEffect(() => {
    projectsRef.current = projects;
  }, [projects]);

  useEffect(() => {
    pendingUploadsRef.current = pendingUploadsByChat;
  }, [pendingUploadsByChat]);

  const currentProject = useMemo(() => {
    return projects.find((project) => project.id === activeProjectId) || projects[0] || null;
  }, [activeProjectId, projects]);

  const currentChat = useMemo(() => {
    if (!currentProject) {
      return null;
    }
    return (
      currentProject.chats.find((chat) => chat.id === activeChatId) ||
      currentProject.chats[0] ||
      null
    );
  }, [activeChatId, currentProject]);

  const pendingAttachments = useMemo(() => {
    if (!currentChat) {
      return [];
    }
    return (pendingUploadsByChat[currentChat.id] || []).map((item) => ({
      id: item.id,
      name: item.file.name,
    }));
  }, [currentChat, pendingUploadsByChat]);

  useEffect(() => {
    if (currentProject && !currentProject.chats.some((chat) => chat.id === activeChatId)) {
      const fallbackChat = currentProject.chats[0];
      if (fallbackChat) {
        setActiveChatId(fallbackChat.id);
      }
    }
  }, [activeChatId, currentProject]);

  useEffect(() => {
    const setupDefaultKb = async (): Promise<void> => {
      const defaultProject = projectsRef.current.find(
        (project) => project.id === DEFAULT_PROJECT_ID,
      );
      if (!defaultProject || defaultProject.kbId) {
        return;
      }

      try {
        const response = await createKb();
        setProjects((previousProjects) =>
          previousProjects.map((project) =>
            project.id === DEFAULT_PROJECT_ID
              ? { ...project, kbId: response.kb_id }
              : project,
          ),
        );
      } catch (error) {
        const message =
          error instanceof Error
            ? error.message
            : 'Could not initialize knowledge base.';
        setProjects((previousProjects) =>
          previousProjects.map((project) => {
            if (project.id !== DEFAULT_PROJECT_ID) {
              return project;
            }
            const firstChat = project.chats[0];
            if (!firstChat) {
              return project;
            }
            const errorMessage: Message = {
              id: generateId(),
              content: message,
              sender: 'ai',
              timestamp: getTimestamp(),
              citations: [],
              answerAudioBase64: null,
              isError: true,
            };
            return {
              ...project,
              chats: project.chats.map((chat) =>
                chat.id === firstChat.id
                  ? { ...chat, messages: [...chat.messages, errorMessage] }
                  : chat,
              ),
            };
          }),
        );
      }
    };

    void setupDefaultKb();
  }, []);

  const initializeProjectKb = async (projectId: string): Promise<void> => {
    try {
      await ensureKbId(projectId);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Could not initialize knowledge base.';
      const project = projectsRef.current.find((item) => item.id === projectId);
      const fallbackChatId = project?.chats[0]?.id;
      if (fallbackChatId) {
        appendAiMessage({
          projectId,
          chatId: fallbackChatId,
          content: message,
          isError: true,
        });
      }
    }
  };

  const ensureKbId = async (projectId: string): Promise<string> => {
    const existingProject = projectsRef.current.find((project) => project.id === projectId);
    if (!existingProject) {
      throw new Error('Project does not exist.');
    }

    if (existingProject.kbId) {
      return existingProject.kbId;
    }

    const response = await createKb();
    setProjects((previousProjects) =>
      previousProjects.map((project) =>
        project.id === projectId ? { ...project, kbId: response.kb_id } : project,
      ),
    );
    return response.kb_id;
  };

  const appendMessage = (projectId: string, chatId: string, message: Message): void => {
    setProjects((previousProjects) =>
      previousProjects.map((project) => {
        if (project.id !== projectId) {
          return project;
        }

        return {
          ...project,
          chats: project.chats.map((chat) => {
            if (chat.id !== chatId) {
              return chat;
            }
            return {
              ...chat,
              messages: [...chat.messages, message],
            };
          }),
        };
      }),
    );
  };

  const appendUserMessage = (projectId: string, chatId: string, content: string): void => {
    appendMessage(projectId, chatId, {
      id: generateId(),
      content,
      sender: 'user',
      timestamp: getTimestamp(),
      citations: [],
      answerAudioBase64: null,
      isError: false,
    });
  };

  const appendAiMessage = ({
    projectId,
    chatId,
    content,
    citations = [],
    answerAudioBase64 = null,
    isError = false,
  }: {
    projectId: string;
    chatId: string;
    content: string;
    citations?: Message['citations'];
    answerAudioBase64?: string | null;
    isError?: boolean;
  }): void => {
    appendMessage(projectId, chatId, {
      id: generateId(),
      content,
      sender: 'ai',
      timestamp: getTimestamp(),
      citations,
      answerAudioBase64,
      isError,
    });
  };

  const handleNewProject = (name: string): void => {
    const projectId = generateId();
    const chatId = generateId();

    const newProject: Project = {
      id: projectId,
      name: name.toUpperCase(),
      kbId: null,
      chats: [
        {
          id: chatId,
          title: 'Session 01',
          projectId,
          messages: [],
        },
      ],
    };

    setProjects((previousProjects) => [...previousProjects, newProject]);
    setActiveProjectId(projectId);
    setActiveChatId(chatId);

    void initializeProjectKb(projectId);
  };

  const handleNewChat = (projectId: string): void => {
    const chatId = generateId();

    setProjects((previousProjects) =>
      previousProjects.map((item) => {
        if (item.id !== projectId) {
          return item;
        }

        // Find the highest existing session number so we always increment
        let maxSessionNum = 0;
        for (const chat of item.chats) {
          const match = chat.title.match(/Session\s+(\d+)/i);
          if (match) {
            maxSessionNum = Math.max(maxSessionNum, parseInt(match[1], 10));
          }
        }
        const nextNum = maxSessionNum + 1;

        return {
          ...item,
          chats: [
            ...item.chats,
            {
              id: chatId,
              title: `Session ${String(nextNum).padStart(2, '0')}`,
              projectId,
              messages: [],
            },
          ],
        };
      }),
    );

    setActiveProjectId(projectId);
    setActiveChatId(chatId);
  };

  const handleSelectChat = (chatId: string, projectId: string): void => {
    setActiveChatId(chatId);
    setActiveProjectId(projectId);
  };

  const handleDeleteProject = (projectId: string): void => {
    if (projectId === DEFAULT_PROJECT_ID) {
      return;
    }

    const projectToDelete = projectsRef.current.find(
      (project) => project.id === projectId,
    );
    const removedChatIds = (projectToDelete?.chats || []).map((chat) => chat.id);

    const fallbackProject =
      projectsRef.current.find((project) => project.id === DEFAULT_PROJECT_ID) ||
      projectsRef.current.find((project) => project.id !== projectId) ||
      null;

    setProjects((previousProjects) => previousProjects.filter((project) => project.id !== projectId));
    setPendingUploadsByChat((previousUploads) => {
      const nextUploads = { ...previousUploads };
      for (const chatId of removedChatIds) {
        delete nextUploads[chatId];
      }
      return nextUploads;
    });

    if (activeProjectId === projectId && fallbackProject) {
      setActiveProjectId(fallbackProject.id);
      if (fallbackProject.chats.length > 0) {
        setActiveChatId(fallbackProject.chats[0].id);
      }
    }
  };

  const handleDeleteChat = (chatId: string, projectId: string): void => {
    const project = projectsRef.current.find((item) => item.id === projectId);
    if (!project) {
      return;
    }

    const remainingChats = project.chats.filter((chat) => chat.id !== chatId);
    const chatsAfterDelete =
      remainingChats.length > 0
        ? remainingChats
        : [
            {
              id: generateId(),
              title: 'Session 01',
              projectId,
              messages: [],
            },
          ];

    setProjects((previousProjects) =>
      previousProjects.map((item) => {
        if (item.id !== projectId) {
          return item;
        }
        return {
          ...item,
          chats: chatsAfterDelete,
        };
      }),
    );
    setPendingUploadsByChat((previousUploads) => {
      const nextUploads = { ...previousUploads };
      delete nextUploads[chatId];
      return nextUploads;
    });

    if (activeProjectId === projectId && activeChatId === chatId) {
      setActiveProjectId(projectId);
      setActiveChatId(chatsAfterDelete[0].id);
    }
  };

  const handleInputError = (message: string): void => {
    if (!currentProject || !currentChat) {
      return;
    }
    appendAiMessage({
      projectId: currentProject.id,
      chatId: currentChat.id,
      content: message,
      isError: true,
    });
  };

  const handleAddPendingFile = (file: File): void => {
    if (!currentChat) {
      return;
    }

    const chatId = currentChat.id;
    const pendingFile: PendingUpload = {
      id: generateId(),
      file,
    };
    setPendingUploadsByChat((previousUploads) => ({
      ...previousUploads,
      [chatId]: [...(previousUploads[chatId] || []), pendingFile],
    }));
  };

  const handleRemovePendingFile = (attachmentId: string): void => {
    if (!currentChat) {
      return;
    }

    const chatId = currentChat.id;
    setPendingUploadsByChat((previousUploads) => {
      const currentUploads = previousUploads[chatId] || [];
      const remainingUploads = currentUploads.filter(
        (item) => item.id !== attachmentId,
      );
      const nextUploads = { ...previousUploads };
      if (remainingUploads.length > 0) {
        nextUploads[chatId] = remainingUploads;
      } else {
        delete nextUploads[chatId];
      }
      return nextUploads;
    });
  };

  const prepareKbForChatRequest = async (
    projectId: string,
    chatId: string,
  ): Promise<string> => {
    const kbId = await ensureKbId(projectId);
    const pendingUploads = pendingUploadsRef.current[chatId] || [];
    if (pendingUploads.length === 0) {
      return kbId;
    }

    await uploadDocuments(
      kbId,
      pendingUploads.map((item) => item.file),
    );
    setPendingUploadsByChat((previousUploads) => {
      const nextUploads = { ...previousUploads };
      delete nextUploads[chatId];
      return nextUploads;
    });
    return kbId;
  };

  const handleSendTextMessage = async (content: string): Promise<void> => {
    if (!currentProject || !currentChat) {
      return;
    }

    const projectId = currentProject.id;
    const chatId = currentChat.id;

    appendUserMessage(projectId, chatId, content);
    setIsBusy(true);

    try {
      const kbId = await prepareKbForChatRequest(projectId, chatId);
      const response = await sendChat(kbId, {
        mode: 'text',
        question_text: content,
        read_aloud: true,
      });

      appendAiMessage({
        projectId,
        chatId,
        content: response.answer_text,
        citations: response.citations,
        answerAudioBase64: response.answer_audio_base64,
      });
    } catch (error) {
      appendAiMessage({
        projectId,
        chatId,
        content: error instanceof Error ? error.message : 'Failed to get response.',
        isError: true,
      });
    } finally {
      setIsBusy(false);
    }
  };

  const handleSendVoiceMessage = async (audioBase64: string): Promise<void> => {
    if (!currentProject || !currentChat) {
      return;
    }

    const projectId = currentProject.id;
    const chatId = currentChat.id;

    appendUserMessage(projectId, chatId, '[VOICE MESSAGE SENT]');
    setIsBusy(true);

    try {
      const kbId = await prepareKbForChatRequest(projectId, chatId);
      const response = await sendChat(kbId, {
        mode: 'voice',
        audio_base64: audioBase64,
      });

      appendAiMessage({
        projectId,
        chatId,
        content: response.answer_text,
        citations: response.citations,
        answerAudioBase64: response.answer_audio_base64,
      });
    } catch (error) {
      appendAiMessage({
        projectId,
        chatId,
        content: error instanceof Error ? error.message : 'Voice request failed.',
        isError: true,
      });
    } finally {
      setIsBusy(false);
    }
  };

  return (
    <div
      className="flex h-screen scanlines"
      style={{
        background: '#050508',
        backgroundImage: `
          radial-gradient(ellipse at 20% 20%, rgba(0, 240, 255, 0.08) 0%, transparent 50%),
          radial-gradient(ellipse at 80% 80%, rgba(184, 41, 221, 0.08) 0%, transparent 50%),
          linear-gradient(180deg, rgba(0, 240, 255, 0.02) 1px, transparent 1px)
        `,
        backgroundSize: '100% 100%, 100% 100%, 100% 4px',
      }}
    >
      <Sidebar
        projects={projects}
        activeChatId={activeChatId}
        activeProjectId={activeProjectId}
        onNewProject={handleNewProject}
        onNewChat={handleNewChat}
        onSelectChat={handleSelectChat}
        onDeleteChat={handleDeleteChat}
        onDeleteProject={handleDeleteProject}
      />

      <main className="flex-1 flex flex-col min-w-0">
        <ChatHeader />

        <div className="flex-1 overflow-y-auto px-4 py-6">
          <div className="max-w-3xl mx-auto">
            {currentChat?.messages.length ? (
              currentChat.messages.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-center py-20">
                <div
                  className="w-16 h-16 flex items-center justify-center mb-4"
                  style={{
                    background: 'linear-gradient(135deg, #00f0ff, #b829dd)',
                    boxShadow: '0 0 30px rgba(0, 240, 255, 0.5)',
                  }}
                >
                  <span className="text-black text-2xl font-bold">N</span>
                </div>
                <h2 className="text-[#00f0ff] text-lg cyber-glow-cyan mb-2">NEURAL CHAT INITIALIZED</h2>
                <p className="text-[#666] text-xs">Upload PDFs and ask grounded questions.</p>
              </div>
            )}
          </div>
        </div>

        <div className="max-w-3xl mx-auto w-full">
          <ChatInput
            disabled={isBusy}
            pendingAttachments={pendingAttachments}
            onSendTextMessage={handleSendTextMessage}
            onSendVoiceMessage={handleSendVoiceMessage}
            onAddPendingFile={handleAddPendingFile}
            onRemovePendingFile={handleRemovePendingFile}
            onInputError={handleInputError}
          />
        </div>
      </main>
    </div>
  );
};

function generateId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
}

function getTimestamp(): string {
  return new Date().toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });
}

export default ChatPage;
