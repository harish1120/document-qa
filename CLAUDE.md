# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Python environment (uv)

Use a virtual environment managed by [uv](https://github.com/astral-sh/uv). Create it once, then **activate** it in each shell (uv does not auto-activate).

```bash
# Backend + pytest (one venv at repo root)
uv venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -r backend/requirements.txt

# Frontend (separate terminal — own venv under frontend/)
cd frontend
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

After activation, `python`, `pip`, `uvicorn`, `pytest`, etc. use the venv.

### Run locally (without Docker)

With the **root** `.venv` activated and backend dependencies installed:

```bash
cd backend && uvicorn main:app --reload
```

Frontend (with **frontend** `.venv` activated):

```bash
BACKEND_URL=http://localhost:8000 streamlit run app.py
```

### Run with Docker Compose

```bash
docker compose up -d --build
docker compose logs -f        # tail logs
docker compose down
```

### Run tests

Tests must be run from the **project root** (not from `backend/`), because `conftest.py` imports via `backend.*` package paths. Use the **root** `.venv` (backend requirements installed):

```bash
source .venv/bin/activate   # if not already active

# All tests
pytest backend/test/

# Single test file
pytest backend/test/test_api.py

# Single test
pytest backend/test/test_api.py::test_health
```

Note: `test_rag.py` requires a populated vectorstore and a valid `OPENAI_API_KEY`.

### Environment setup

```bash
echo "OPENAI_API_KEY=your-key-here" > .env
```

### AWS / SAM deployment

The backend is also deployable as an AWS Lambda (via [Mangum](https://mangum.io/)) with the vectorstore and uploads stored in S3. The frontend runs on ECS Fargate.

```bash
sam build
sam deploy --guided   # prompts for OPENAI_API_KEY, BackendImageUri, FrontendImageUri
```

The SAM template (`template.yaml`) provisions: Lambda + HTTP API Gateway (backend), S3 bucket (uploads + vectorstore), ECS Fargate service (frontend), and the required IAM roles.

## Architecture

This is a two-service Dockerized RAG application:

- **Backend** (`backend/`, FastAPI on `:8000`) — handles PDF ingestion, vector indexing, hybrid search, and LLM answer generation.
- **Frontend** (`frontend/`, Streamlit on `:8501`) — thin UI that calls the backend REST API. The frontend `BACKEND_URL` is hardcoded to `http://backend:8000` (Docker network name).

### RAG pipeline flow

1. **Upload**: `POST /upload_pdf` — validates file (size, extension, magic bytes), saves with a UUID-prefixed filename to `data/uploads/`.
2. **Index**: `POST /index?path=<path>` — calls `ingest.py::ingest_pdf`, which loads the PDF with `PyPDFLoader`, splits into 500-char chunks (100 overlap), embeds with OpenAI, and writes a FAISS index + `docs.pkl` to `vectorstore/`.
3. **Ask**: `POST /ask` — calls `rag.py::answer_question`, which runs `hybrid_search` (FAISS dense + BM25 sparse, α=0.5), builds a context string, and calls `gpt-5-nano` via `ChatOpenAI`. Rate-limited to 10 req/min per IP. Retries up to 3× with exponential backoff (tenacity).

### Two RAG implementations

- **`rag.py`** — production implementation, wired to `POST /ask`. Direct chain: hybrid search → LLM call.
- **`rag_graph.py`** — experimental LangGraph alternative (NOT connected to the API). Defines a `StateGraph` with `retrieve` → `generate` nodes. Also contains a stub `grade_documents` node (LLM-graded relevance filtering, not yet wired in) and an incomplete `rewrite_query` function. The API request/response shapes are defined in `backend/schemas.py` (`AskRequest`, `AskResponse`).

### Deployment modes

The app has two modes, controlled by the `S3_BUCKET_NAME` environment variable:

| Mode | Storage | Vectorstore path |
|------|---------|-----------------|
| Local (unset) | Local disk | `vectorstore/` directory |
| AWS (set) | S3 bucket | `vectorstore/` prefix in S3, downloaded to `/tmp/vectorstore/` |

In AWS mode, `ingest.py` maintains a `vectorstore/manifest.json` in S3 that tracks ingested documents by SHA256 hash to prevent duplicate indexing. `rag.py` uses S3 ETag-based in-process caching (`_cache` dict) to avoid re-downloading the vectorstore on every request.

### Key data paths

- `vectorstore/index.faiss` + `vectorstore/index.pkl` — FAISS index (persisted via Docker volume mount in local mode)
- `vectorstore/docs.pkl` — all chunked `Document` objects (needed for BM25 at query time)
- `vectorstore/manifest.json` — SHA256 deduplication manifest (S3 mode only)
- `data/uploads/` — uploaded PDFs (local mode)
- `logs/app.log` — DEBUG-level file log; console is INFO-level

### Hybrid search scoring

```python
final_score = alpha * (1 - dense_score) + (1 - alpha) * sparse_score  # alpha=0.5
```

BM25 scores are min-max normalized before combining. The FAISS scores are L2 distances (lower = better), so they are inverted with `(1 - dense_score)`.

### Monitoring

Prometheus metrics are exposed at `/metrics` (via `prometheus-fastapi-instrumentator`). A JSON debug view is at `/metrics-json`.

### Logging

All modules call `logger.setup_logger(__name__)` from `backend/logger.py`. Logs go to both stdout (INFO) and `logs/app.log` (DEBUG).
