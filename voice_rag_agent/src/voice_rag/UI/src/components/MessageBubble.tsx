import { useMemo, useState } from 'react';
import {
  ChevronDown,
  ChevronUp,
  Copy,
  ThumbsDown,
  ThumbsUp,
  Zap,
} from 'lucide-react';

import type { Message } from '@/types/chat';

interface MessageBubbleProps {
  message: Message;
}

const MessageBubble = ({ message }: MessageBubbleProps) => {
  const [showCitations, setShowCitations] = useState(false);
  const isUser = message.sender === 'user';

  const audioSource = useMemo(() => {
    if (!message.answerAudioBase64) {
      return '';
    }
    const b64 = message.answerAudioBase64;
    // Detect audio format from base64 header bytes:
    // WAV starts with "RIFF" = "UklG" in base64
    // MP3 with ID3 tag starts with "SUQz", or frame sync FF FB = "//s", FF F3 = "//M"
    let mime = 'audio/wav';
    if (b64.startsWith('SUQz') || b64.startsWith('//s') || b64.startsWith('//M') || b64.startsWith('//u')) {
      mime = 'audio/mpeg';
    }
    return `data:${mime};base64,${b64}`;
  }, [message.answerAudioBase64]);

  if (isUser) {
    return (
      <div className="flex justify-end mb-6">
        <div className="flex items-start gap-3 max-w-[80%]">
          <div
            className="px-4 py-3"
            style={{
              background:
                'linear-gradient(135deg, rgba(0, 240, 255, 0.15), rgba(184, 41, 221, 0.15))',
              border: '1px solid #00f0ff',
              boxShadow:
                '0 0 15px rgba(0, 240, 255, 0.3), inset 0 0 20px rgba(0, 240, 255, 0.05)',
            }}
          >
            <p className="text-[#00f0ff] text-xs leading-relaxed cyber-glow-cyan">
              {message.content}
            </p>
          </div>
          <div
            className="w-8 h-8 flex items-center justify-center flex-shrink-0"
            style={{
              background: 'linear-gradient(135deg, #00f0ff, #b829dd)',
              boxShadow: '0 0 10px rgba(0, 240, 255, 0.5)',
            }}
          >
            <span className="text-black text-[10px] font-bold">U</span>
          </div>
        </div>
      </div>
    );
  }

  const hasCitations = message.citations.length > 0;

  return (
    <div className="flex gap-3 mb-6">
      <div
        className="w-10 h-10 flex items-center justify-center flex-shrink-0 mt-1"
        style={{
          background: 'linear-gradient(135deg, #ff00ff, #b829dd)',
          boxShadow: '0 0 15px rgba(255, 0, 255, 0.5)',
          border: '1px solid #ff00ff',
        }}
      >
        <Zap className="w-5 h-5 text-white" />
      </div>

      <div className="flex-1 max-w-[85%]">
        <div
          className="px-4 py-3 relative"
          style={{
            background: message.isError ? 'rgba(34, 8, 12, 0.8)' : 'rgba(15, 15, 26, 0.8)',
            border: message.isError ? '1px solid #ff0040' : '1px solid #b829dd',
            boxShadow: message.isError
              ? '0 0 15px rgba(255, 0, 64, 0.2), inset 0 0 20px rgba(255, 0, 64, 0.06)'
              : '0 0 15px rgba(184, 41, 221, 0.2), inset 0 0 20px rgba(184, 41, 221, 0.05)',
          }}
        >
          <div className="absolute top-0 left-0 w-2 h-2 border-t border-l border-[#ff00ff]" />
          <div className="absolute top-0 right-0 w-2 h-2 border-t border-r border-[#ff00ff]" />
          <div className="absolute bottom-0 left-0 w-2 h-2 border-b border-l border-[#ff00ff]" />
          <div className="absolute bottom-0 right-0 w-2 h-2 border-b border-r border-[#ff00ff]" />

          <p className="text-[#e0e0e0] text-xs leading-relaxed whitespace-pre-wrap break-words">
            {message.content}
          </p>
        </div>

        <div
          className="mt-2 px-4 py-3"
          style={{
            background: 'rgba(8, 8, 14, 0.9)',
            border: '1px solid rgba(0, 240, 255, 0.35)',
            boxShadow: 'inset 0 0 16px rgba(0, 240, 255, 0.06)',
          }}
        >
          {audioSource ? (
            <audio className="w-full" controls preload="metadata" src={audioSource} />
          ) : (
            <p className="text-[10px] text-[#777]">No audio response available.</p>
          )}
        </div>

        <button
          className="mt-2 w-full flex items-center justify-between px-3 py-2 border border-[#00f0ff]/35 hover:bg-[#00f0ff]/10 transition-all text-[10px] text-[#00f0ff] tracking-wider"
          onClick={() => setShowCitations((value) => !value)}
          type="button"
        >
          <span>{showCitations ? 'HIDE CITATIONS' : 'SHOW CITATIONS'}</span>
          {showCitations ? (
            <ChevronUp className="w-3 h-3" />
          ) : (
            <ChevronDown className="w-3 h-3" />
          )}
        </button>

        {showCitations && (
          <div
            className="mt-2 px-3 py-3"
            style={{
              background: 'rgba(10, 12, 18, 0.92)',
              border: '1px solid rgba(0, 240, 255, 0.25)',
            }}
          >
            {hasCitations ? (
              <div className="space-y-2">
                {message.citations.map((citation) => (
                  <div
                    key={`${citation.doc_id}-${citation.chunk_id}`}
                    className="p-2 border border-[#1a1a2e] bg-[#0f0f1a]"
                  >
                    <div className="text-[10px] text-[#00f0ff]">
                      {citation.source_name} · page {citation.page} · {citation.doc_id}
                    </div>
                    <p className="mt-1 text-[10px] text-[#c9c9c9] leading-relaxed whitespace-pre-wrap break-words">
                      {citation.snippet}
                    </p>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-[10px] text-[#777]">No citations available.</p>
            )}
          </div>
        )}

        <div className="flex items-center gap-3 mt-2">
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 bg-[#ff00ff] animate-pulse" />
            <span className="text-[10px] text-[#b829dd]">{message.timestamp}</span>
          </div>

          <div className="flex items-center gap-1">
            <button
              className="w-7 h-7 flex items-center justify-center border border-[#00ff88]/30 hover:bg-[#00ff88]/20 hover:border-[#00ff88] transition-all"
              type="button"
            >
              <ThumbsUp className="w-3 h-3 text-[#00ff88]" />
            </button>
            <button
              className="w-7 h-7 flex items-center justify-center border border-[#ff0040]/30 hover:bg-[#ff0040]/20 hover:border-[#ff0040] transition-all"
              type="button"
            >
              <ThumbsDown className="w-3 h-3 text-[#ff0040]" />
            </button>
            <button
              className="w-7 h-7 flex items-center justify-center border border-[#ffee00]/30 hover:bg-[#ffee00]/20 hover:border-[#ffee00] transition-all"
              type="button"
              onClick={() => {
                if (navigator.clipboard) {
                  void navigator.clipboard.writeText(message.content);
                }
              }}
            >
              <Copy className="w-3 h-3 text-[#ffee00]" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MessageBubble;
