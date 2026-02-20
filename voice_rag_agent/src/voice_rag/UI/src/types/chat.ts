import type { Citation } from '@/lib/api';

export interface Message {
  id: string;
  content: string;
  sender: 'user' | 'ai';
  timestamp: string;
  citations: Citation[];
  answerAudioBase64: string | null;
  isError: boolean;
}

export interface Chat {
  id: string;
  title: string;
  projectId: string;
  messages: Message[];
}

export interface Project {
  id: string;
  name: string;
  kbId: string | null;
  chats: Chat[];
}
