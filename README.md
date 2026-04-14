# Document QA (Corrective RAG)

Full-stack PDF question-answering app with:

- `backend/`: FastAPI API for upload, indexing, and QA
- `frontend/`: Streamlit UI
- local mode (filesystem) and AWS mode (S3 + Lambda/ECS)

The `/ask` endpoint is currently wired to `backend/corrective_rag.py::answer_question`, a LangGraph-based corrective RAG flow.

## What this app does

1. Upload a PDF via `POST /upload_pdf`
2. Index it via `POST /index?path=<uploaded-path>`
3. Ask questions via `POST /ask`

Answers include source chunks (`content`, `page`, `source`) used by retrieval.

## Architecture

### Backend (`backend/`)

- `main.py`: FastAPI app, API routes, rate limiting, metrics
- `ingest.py`: PDF ingestion + chunking + embeddings + vectorstore persistence
- `rag.py`: hybrid retrieval primitives (FAISS + BM25)
- `corrective_rag.py`: graph-based answer generation with relevance grading and rewrite loop
- `schemas.py`: request/response models
- `logger.py`: structured logging setup

### Frontend (`frontend/`)

- `app.py`: Streamlit app that calls backend endpoints

### Storage modes

- **Local mode** (default): uses `data/uploads/` and `vectorstore/`
- **AWS mode** (`S3_BUCKET_NAME` set): uses S3 for uploads and vectorstore

## Corrective RAG flow (`backend/corrective_rag.py`)

The graph runs these steps:

1. **Agent**: decides whether to call `search_documents`
2. **Tools**: executes retrieval via `hybrid_search`
3. **Grader**: checks relevance per chunk with a grader LLM
4. **Router**:
   - enough relevant evidence -> back to agent to answer
   - weak evidence + retries left -> rewrite query and retry
   - retries exhausted -> fallback response

Key environment knobs:

- `RELEVANCE_THRESHOLD` (default `0.65`)
- `GRADER_MODEL` (default `gpt-4.1-mini`)
- `GRADER_TEMPERATURE` (default `0.0`)
- `MAX_CORRECTIVE_RETRIES` (default `2`)

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)
- OpenAI API key
- (optional) Docker / Docker Compose
- (optional) AWS CLI + SAM CLI

## Local setup (uv)

From repo root:

```bash
uv venv
source .venv/bin/activate
uv pip install -r backend/requirements.txt
```

Create `.env`:

```bash
cat > .env << 'EOF'
OPENAI_API_KEY=your-key-here
RELEVANCE_THRESHOLD=0.65
GRADER_MODEL=gpt-4.1-mini
GRADER_TEMPERATURE=0.0
MAX_CORRECTIVE_RETRIES=2
EOF
```

## Run locally

### Backend

```bash
source .venv/bin/activate
cd backend && uvicorn main:app --reload
```

### Frontend (separate terminal)

```bash
cd frontend
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
BACKEND_URL=http://localhost:8000 streamlit run app.py
```

## Run with Docker Compose

```bash
docker compose up -d --build
docker compose logs -f
docker compose down
```

## API quickstart

### 1) Health

```bash
curl http://localhost:8000/health
```

### 2) Upload

```bash
curl -X POST http://localhost:8000/upload_pdf -F "file=@document.pdf"
```

### 3) Index

```bash
curl -X POST "http://localhost:8000/index?path=<path-from-upload>"
```

### 4) Ask

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the key conclusions?"}'
```

## Testing

Run from project root:

```bash
source .venv/bin/activate
pytest backend/test/
```

## AWS deployment (SAM)

The project supports serverless deployment:

- Backend: FastAPI on Lambda via Mangum
- Frontend: ECS Fargate
- Storage: S3

Deploy:

```bash
sam build
sam deploy --guided
```

## Notes

- `/ask` is rate-limited to `10/minute` per IP.
- Prometheus metrics are available at `/metrics`.
- JSON metrics debug endpoint: `/metrics-json`.
- Avoid committing `.env` or secrets.
