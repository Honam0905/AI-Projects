# Voice RAG UI (React + Tailwind)

This frontend integrates with the backend API in `/v1`:

- `POST /v1/kb`
- `POST /v1/kb/{kb_id}/documents`
- `POST /v1/kb/{kb_id}/chat`

## Run

```bash
cd /Users/honam/Cursor/voice_rag_agent/src/voice_rag/UI
npm install
npm run dev
```

Run backend in a second terminal:

```bash
cd /Users/honam/Cursor/voice_rag_agent
source .venv310/bin/activate
uvicorn voice_rag.api.main:create_app --factory --host 127.0.0.1 --port 8000 --app-dir src
```

## Backend URL

By default, Vite proxies `/v1` to `http://127.0.0.1:8000`.

Optional environment variables:

- `VITE_BACKEND_TARGET=http://127.0.0.1:8000` (controls Vite dev proxy target)
- `VITE_API_BASE_URL=http://127.0.0.1:8000` (forces direct absolute API calls)

If `VITE_API_BASE_URL` is empty, the app uses relative `/v1/...` URLs.

## Troubleshooting

If UI shows `ECONNREFUSED` or `Request failed with status 500` from Vite proxy:

1. Ensure FastAPI is running on `127.0.0.1:8000`.
2. Check backend health:
   `curl http://127.0.0.1:8000/healthz`
3. If backend uses another port, set:
   `VITE_BACKEND_TARGET=http://127.0.0.1:<port> npm run dev`

## UI behavior

- Upload PDF(s) from the input toolbar (`UPLOAD`)
- Uploaded files stay pending in the input box (not sent yet)
- Press `SEND` with your query to ingest pending PDFs first, then run RAG
- Voice query via browser microphone (`VOICE` then `STOP & SEND`)
- AI response includes:
  - text panel
  - audio panel
  - citations panel toggled by `SHOW CITATIONS`
