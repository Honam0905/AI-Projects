export interface ApiErrorEnvelope {
  error: {
    code: string;
    message: string;
    details: Record<string, unknown>;
  };
  request_id: string;
}

export interface CreateKbResponse {
  kb_id: string;
}

export interface UploadedDocument {
  doc_id: string;
  source_name: string;
  pages: number;
}

export interface UploadDocumentsResponse {
  kb_id: string;
  documents: UploadedDocument[];
}

export interface Citation {
  source_name: string;
  doc_id: string;
  page: number;
  chunk_id: string;
  snippet: string;
  score: number;
  bbox: number[] | null;
}

export type ChatMode = 'text' | 'voice';

export interface ChatRequest {
  mode: ChatMode;
  question_text?: string;
  audio_base64?: string;
  read_aloud?: boolean;
}

export interface ChatResponse {
  answer_text: string;
  citations: Citation[];
  answer_audio_base64: string | null;
}

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').trim();

function buildUrl(path: string): string {
  if (!API_BASE_URL) {
    return path;
  }
  const normalizedBaseUrl = API_BASE_URL.endsWith('/')
    ? API_BASE_URL.slice(0, -1)
    : API_BASE_URL;
  return `${normalizedBaseUrl}${path}`;
}

async function parseJson<T>(response: Response): Promise<T> {
  const contentType = response.headers.get('content-type') || '';
  if (!contentType.includes('application/json')) {
    throw new Error('Unexpected server response type.');
  }
  return (await response.json()) as T;
}

async function requestJson<T>(path: string, init: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(buildUrl(path), init);
  } catch {
    throw new Error(
      'Cannot connect to backend API. Start FastAPI server and verify UI proxy target.',
    );
  }

  if (!response.ok) {
    const fallbackMessage = `Request failed with status ${response.status}.`;
    let errorMessage = fallbackMessage;
    const responseForJson = response.clone();

    try {
      const errorEnvelope = await parseJson<ApiErrorEnvelope>(responseForJson);
      errorMessage = errorEnvelope.error.message || fallbackMessage;
    } catch {
      try {
        const rawBody = await response.text();
        if (rawBody.includes('ECONNREFUSED')) {
          errorMessage =
            'Cannot connect to backend API. Start FastAPI server and verify UI proxy target.';
        } else {
          errorMessage = fallbackMessage;
        }
      } catch {
        errorMessage = fallbackMessage;
      }
    }

    throw new Error(errorMessage);
  }

  return parseJson<T>(response);
}

export function createKb(): Promise<CreateKbResponse> {
  return requestJson<CreateKbResponse>('/v1/kb', {
    method: 'POST',
    headers: {
      Accept: 'application/json',
    },
  });
}

export function uploadDocuments(
  kbId: string,
  files: File[],
): Promise<UploadDocumentsResponse> {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append('files', file, file.name);
  });

  return requestJson<UploadDocumentsResponse>(`/v1/kb/${kbId}/documents`, {
    method: 'POST',
    body: formData,
  });
}

export function sendChat(
  kbId: string,
  payload: ChatRequest,
): Promise<ChatResponse> {
  return requestJson<ChatResponse>(`/v1/kb/${kbId}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify(payload),
  });
}
