import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type KeyboardEvent,
} from 'react';
import { Mic, Send, Upload, X } from 'lucide-react';

interface PendingAttachment {
  id: string;
  name: string;
}

interface ChatInputProps {
  disabled: boolean;
  pendingAttachments: PendingAttachment[];
  onSendTextMessage: (message: string) => Promise<void> | void;
  onSendVoiceMessage: (audioBase64: string) => Promise<void> | void;
  onAddPendingFile: (file: File) => void;
  onRemovePendingFile: (attachmentId: string) => void;
  onInputError: (message: string) => void;
}

const ChatInput = ({
  disabled,
  pendingAttachments,
  onSendTextMessage,
  onSendVoiceMessage,
  onAddPendingFile,
  onRemovePendingFile,
  onInputError,
}: ChatInputProps) => {
  const [inputValue, setInputValue] = useState('');
  const [isRecording, setIsRecording] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const handleSend = async () => {
    const text = inputValue.trim();
    if (!text || disabled) {
      return;
    }

    await onSendTextMessage(text);
    setInputValue('');
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void handleSend();
    }
  };

  const handleFileClick = () => {
    if (disabled) {
      return;
    }
    fileInputRef.current?.click();
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = event.target.files;
    if (!selectedFiles || selectedFiles.length === 0) {
      return;
    }

    let validFileCount = 0;
    for (const file of Array.from(selectedFiles)) {
      if (!file.name.toLowerCase().endsWith('.pdf')) {
        onInputError(`"${file.name}" is not a PDF file.`);
        continue;
      }
      onAddPendingFile(file);
      validFileCount += 1;
    }

    event.target.value = '';
    if (validFileCount === 0) {
      onInputError('Only PDF files are supported.');
    }
  };

  const startRecording = async () => {
    if (disabled) {
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      onInputError('Voice recording is not supported in this browser.');
      return;
    }

    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(mediaStream);

      mediaStreamRef.current = mediaStream;
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        void handleRecordedAudio();
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch {
      onInputError('Microphone permission is required for voice input.');
    }
  };

  const stopRecording = () => {
    if (!mediaRecorderRef.current) {
      return;
    }

    if (mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
  };

  const handleRecordedAudio = async () => {
    try {
      const rawBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });

      // Convert WebM → WAV (16kHz mono PCM) so backend ASR can process it
      const wavBlob = await convertBlobToWav(rawBlob);
      const dataUrl = await blobToDataUrl(wavBlob);
      const encodedAudio = dataUrl.split(',')[1] || '';

      if (!encodedAudio) {
        onInputError('Could not encode recorded audio. Please try again.');
      } else {
        await onSendVoiceMessage(encodedAudio);
      }
    } finally {
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      }
      mediaRecorderRef.current = null;
      mediaStreamRef.current = null;
      audioChunksRef.current = [];
    }
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
      return;
    }
    void startRecording();
  };

  return (
    <div className="px-4 pb-4">
      <div
        className="p-4 relative"
        style={{
          background: 'rgba(15, 15, 26, 0.9)',
          border: '1px solid #00f0ff',
          boxShadow:
            '0 0 20px rgba(0, 240, 255, 0.2), inset 0 0 30px rgba(0, 240, 255, 0.05)',
        }}
      >
        <div className="absolute top-0 left-0 w-3 h-3 border-t-2 border-l-2 border-[#00f0ff]" />
        <div className="absolute top-0 right-0 w-3 h-3 border-t-2 border-r-2 border-[#00f0ff]" />
        <div className="absolute bottom-0 left-0 w-3 h-3 border-b-2 border-l-2 border-[#00f0ff]" />
        <div className="absolute bottom-0 right-0 w-3 h-3 border-b-2 border-r-2 border-[#00f0ff]" />

        <textarea
          value={inputValue}
          onChange={(event) => setInputValue(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="ENTER COMMAND..."
          className="w-full bg-transparent text-[#00f0ff] placeholder-[#00f0ff]/30 text-xs resize-none outline-none min-h-[50px] max-h-[150px]"
          rows={2}
          style={{ fontFamily: "'Share Tech Mono', monospace" }}
          disabled={disabled}
        />

        {pendingAttachments.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-2">
            {pendingAttachments.map((attachment) => (
              <div
                key={attachment.id}
                className="flex items-center gap-2 px-2 py-1 border border-[#b829dd]/60 bg-[#b829dd]/10"
              >
                <span
                  className="text-[10px] text-[#b829dd] max-w-[220px] truncate"
                  title={attachment.name}
                >
                  {attachment.name}
                </span>
                <button
                  type="button"
                  onClick={() => onRemovePendingFile(attachment.id)}
                  className="w-4 h-4 flex items-center justify-center hover:bg-[#ff0040]/20"
                  disabled={disabled}
                >
                  <X className="w-3 h-3 text-[#ff0040]" />
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="flex items-center justify-between mt-4">
          <div className="flex items-center gap-2">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              className="hidden"
              accept="application/pdf,.pdf"
              multiple
              disabled={disabled}
            />
            <button
              onClick={handleFileClick}
              className="flex items-center gap-2 px-3 py-2 border border-[#b829dd]/50 hover:bg-[#b829dd]/20 hover:border-[#b829dd] transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              type="button"
              disabled={disabled}
            >
              <Upload className="w-4 h-4 text-[#b829dd]" />
              <span className="text-[10px] text-[#b829dd] tracking-wider">UPLOAD</span>
            </button>

            <button
              onClick={toggleRecording}
              className={`flex items-center gap-2 px-3 py-2 border transition-all ${
                isRecording
                  ? 'border-[#ff0040] bg-[#ff0040]/20 animate-pulse'
                  : 'border-[#00ff88]/50 hover:bg-[#00ff88]/20 hover:border-[#00ff88]'
              } disabled:opacity-50 disabled:cursor-not-allowed`}
              type="button"
              disabled={disabled && !isRecording}
            >
              <Mic className={`w-4 h-4 ${isRecording ? 'text-[#ff0040]' : 'text-[#00ff88]'}`} />
              <span
                className={`text-[10px] tracking-wider ${isRecording ? 'text-[#ff0040]' : 'text-[#00ff88]'}`}
              >
                {isRecording ? 'STOP & SEND' : 'VOICE'}
              </span>
            </button>
          </div>

          <button
            onClick={() => {
              void handleSend();
            }}
            className="w-10 h-10 flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed"
            style={{
              background: 'linear-gradient(135deg, #00f0ff, #b829dd)',
              boxShadow: '0 0 15px rgba(0, 240, 255, 0.5)',
            }}
            type="button"
            disabled={disabled || !inputValue.trim()}
          >
            <Send className="w-4 h-4 text-black" />
          </button>
        </div>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Audio helpers — convert browser WebM recording to 16 kHz mono PCM WAV
// ---------------------------------------------------------------------------

async function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const value = reader.result;
      if (typeof value === 'string') {
        resolve(value);
      } else {
        reject(new Error('Could not read audio blob.'));
      }
    };
    reader.onerror = () => {
      reject(new Error('Could not read audio blob.'));
    };
    reader.readAsDataURL(blob);
  });
}

async function convertBlobToWav(blob: Blob): Promise<Blob> {
  const arrayBuffer = await blob.arrayBuffer();
  const audioCtx = new AudioContext({ sampleRate: 16000 });
  try {
    const decoded = await audioCtx.decodeAudioData(arrayBuffer);
    return audioBufferToWavBlob(decoded);
  } finally {
    void audioCtx.close();
  }
}

function audioBufferToWavBlob(buffer: AudioBuffer): Blob {
  const sampleRate = buffer.sampleRate;
  const samples = buffer.getChannelData(0);
  const numSamples = samples.length;
  const bps = 2;
  const dataBytes = numSamples * bps;
  const buf = new ArrayBuffer(44 + dataBytes);
  const v = new DataView(buf);

  writeStr(v, 0, 'RIFF');
  v.setUint32(4, 36 + dataBytes, true);
  writeStr(v, 8, 'WAVE');
  writeStr(v, 12, 'fmt ');
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * bps, true);
  v.setUint16(32, bps, true);
  v.setUint16(34, 16, true);
  writeStr(v, 36, 'data');
  v.setUint32(40, dataBytes, true);

  let off = 44;
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    off += 2;
  }
  return new Blob([buf], { type: 'audio/wav' });
}

function writeStr(view: DataView, offset: number, text: string) {
  for (let i = 0; i < text.length; i++) {
    view.setUint8(offset + i, text.charCodeAt(i));
  }
}

export default ChatInput;
