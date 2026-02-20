# Voice RAG Agent

## 1. Project Introduction
Voice RAG Agent is a production-oriented retrieval-augmented generation system that answers questions from uploaded PDF documents in both text and voice modes. The project combines document ingestion, hybrid retrieval, grounded answer generation, and optional speech interfaces (ASR and TTS) into one API + UI workflow.

Core goals:
- Ground answers in user-provided documents with citations.
- Support both text and voice interaction.
- Keep the codebase simple, testable, and reproducible.

## 2. System Pipelines
The project has three main pipelines.

### 2.1 Data Processing Pipeline
- User uploads PDF files to a knowledge base (`kb_id`).
- PDF text is extracted page by page.
- Text is chunked with overlap.
- Chunks are embedded and stored in zvec.
- Chunk metadata is saved for sparse retrieval and citation mapping.

### 2.2 RAG Pipeline
- Input question is normalized.
- Hybrid retrieval runs dense vector search + BM25 sparse retrieval.
- Optional NIM reranker reorders retrieved chunks.
- LangGraph agent generates grounded answer (LLM path if NIM is available, fallback heuristic if unavailable).
- Citations are returned with source metadata and snippets.

### 2.3 Audio Pipeline (ASR/TTS)
- Voice mode:
  - ASR transcribes input audio to text.
  - Text query goes through the same RAG pipeline.
  - TTS synthesizes final answer audio.
- Text mode can optionally return read-aloud audio.

## 3. Requirements
- Python 3.10
- Node.js 18+ and npm (for React UI)
- macOS/Linux/WSL environment recommended
- NVIDIA NIM endpoint and API key (optional but recommended for best quality)

## 4. Environment Setup
```bash
cd /Users/honam/Cursor/voice_rag_agent
python3.10 -m venv .venv310
source .venv310/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Set up `.env` values:
```env
VOICE_RAG_APP_NAME=voice-rag-agent
VOICE_RAG_ENV=development
VOICE_RAG_LOG_LEVEL=INFO
VOICE_RAG_DATA_DIR=data

# ── NVIDIA NIM API (REQUIRED for LLM answer generation) ──
VOICE_RAG_NIM_BASE_URL=https://integrate.api.nvidia.com
VOICE_RAG_NIM_API_KEY=<your_nvidia_nim_api_key>
VOICE_RAG_NIM_TIMEOUT_SECONDS=30
VOICE_RAG_NIM_RETRY_COUNT=2

# ── LLM Settings ──
VOICE_RAG_RAG_USE_LLM=true
VOICE_RAG_NIM_LLM_MODEL=meta/llama-3.1-8b-instruct
VOICE_RAG_NIM_LLM_TEMPERATURE=0.2
VOICE_RAG_NIM_LLM_MAX_TOKENS=256

# ── Embedding (NVIDIA NIM nv-embedqa-e5-v5 = 1024-dim) ──
VOICE_RAG_EMBEDDING_DIMENSION=1024
VOICE_RAG_NIM_EMBEDDING_PATH=/v1/embeddings
VOICE_RAG_NIM_EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5

# ── RAG Retrieval ──
VOICE_RAG_RETRIEVAL_TOP_K=5
VOICE_RAG_CITATION_TOP_K=2
VOICE_RAG_MAX_ANSWER_CHARS=500
VOICE_RAG_CHUNK_SIZE_CHARS=1200
VOICE_RAG_CHUNK_OVERLAP_CHARS=150

# ── Voice ──
VOICE_RAG_VOICE_BACKEND=local
VOICE_RAG_MAX_AUDIO_SIZE_BYTES=10485760

```

If NIM is unavailable, the project still runs with built-in local fallbacks for embedding and local voice tools.

## 5. How To Run The Project

### 5.1 Run Backend API
```bash
cd /Users/honam/Cursor/voice_rag_agent
source .venv310/bin/activate
uvicorn voice_rag.api.main:create_app --factory --host 127.0.0.1 --port 8000 --app-dir src
```

Health check:
```bash
curl http://127.0.0.1:8000/healthz
```

### 5.2 Run Frontend UI (React + Vite)
```bash
cd /Users/honam/Cursor/voice_rag_agent/src/voice_rag/UI
npm install
npm run dev
```

Open the local Vite URL shown in terminal (typically `http://127.0.0.1:5173`).

### 5.3 Run API Directly (Without UI)
Create KB:
```bash
curl -X POST http://127.0.0.1:8000/v1/kb
```

Upload PDF:
```bash
curl -X POST \
  -F "files=@/absolute/path/to/file.pdf" \
  http://127.0.0.1:8000/v1/kb/<kb_id>/documents
```

Text chat:
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"mode":"text","question_text":"What does this document say?"}' \
  http://127.0.0.1:8000/v1/kb/<kb_id>/chat
```

Voice chat:
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"mode":"voice","audio_base64":"<base64-audio>"}' \
  http://127.0.0.1:8000/v1/kb/<kb_id>/chat
```

## 6. UI Usage Instructions (for people who clone this repo)
1. Start backend API (`:8000`) and UI app (`:5173`).
2. In UI, create/select a chat/project.
3. Upload one or more PDF files first.
4. Choose query mode:
   - `text` mode: type a question.
   - `voice` mode: record/send voice query.
5. Press send.
6. Review three answer parts:
   - answer text
   - answer audio playback
   - citations panel (expand/collapse)

## 7. Evaluation Metrics Summary (from `evaluation/metrics.csv`)
Final metric values below are the mean scores from the latest run.

### RAG Metrics
- `answer_relevancy`: `0.3645`
- `context_precision`: `0.5000`
- `context_recall`: `0.6724`
- `context_entity_recall`: `0.7093`
- `faithfulness`: `0.6710`

### ASR Metrics
- `WER`: `0.3793`
- `CER`: `0.1509`

### TTS Metrics
- `duration_abs_error_sec`: `1.4824`
- `duration_rel_error`: `0.1083`
- `mcd_db`: `89.2799`
- `spectral_convergence`: `2.0566`

Run evaluation:
```bash
cd /Users/honam/Cursor/voice_rag_agent
source .venv310/bin/activate
PYTHONPATH=src python evaluation/run_milestone7_eval.py
```

Artifacts:
- `/Users/honam/Cursor/voice_rag_agent/evaluation/results.jsonl`
- `/Users/honam/Cursor/voice_rag_agent/evaluation/metrics.csv`

## 8. Conclusion and Future Improvements
This project delivers a full voice-enabled RAG workflow with clean backend APIs, modern UI integration, retrieval + reranking support, and an evaluation pipeline.

Practical next improvements:
- Improve answer relevancy via stronger query rewriting and retrieval tuning.
- Add production observability (structured traces/metrics per pipeline node).
- Add CI workflows for lint/test/eval checks.
- Expand domain-specific evaluation datasets and regression tracking.
