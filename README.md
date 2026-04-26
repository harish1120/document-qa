<h1 align="center">Document QA — Corrective RAG</h1>

<p align="center">
  A production-ready, full-stack PDF question-answering system powered by a <strong>LangGraph Corrective RAG</strong> pipeline.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-0.x-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/AWS-SAM%20%7C%20Lambda%20%7C%20ECS-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white" />
</p>

---

## Overview

Document QA lets you upload PDFs and ask questions grounded in their content. Unlike a naive RAG pipeline, this app uses **Corrective RAG (CRAG)**, a LangGraph state machine that retrieves, grades, and self-corrects before generating an answer.

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Mangum (Lambda-compatible) |
| Frontend UI | Streamlit |
| RAG Orchestration | LangGraph (`StateGraph`) |
| Retrieval | Hybrid — FAISS (dense) + BM25 (sparse) |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o-mini` |
| Storage (local) | Filesystem (`data/uploads/`, `vectorstore/`) |
| Storage (AWS) | S3 + Lambda + ECS Fargate |
| Observability | Prometheus `/metrics`, structured file + console logging |

---

## Corrective RAG Pipeline

```
┌─────────┐     tool call?     ┌───────┐    grade docs    ┌────────┐
│  Agent  │ ────────────────► │ Tools │ ───────────────► │ Grader │
└─────────┘                    └───────┘                   └────────┘
     ▲                                                          │
     │                                          ┌──────────────┼──────────────┐
     │                                     pass │         fail │         exhausted │
     │                                          ▼              ▼                ▼
     │                                    ┌──────────┐  ┌─────────┐  ┌──────────────┐
     └────────────────────────────────────│ Generate │  │ Rewrite │  │   Fallback   │
                                          └──────────┘  └─────────┘  └──────────────┘
                                               │              │
                                              END           Agent (retry)
```

**Step-by-step:**

1. **Agent** — the LLM decides whether to call `search_documents` or answer directly.
2. **Tools** — executes hybrid retrieval (`FAISS` dense + `BM25` sparse, α=0.5).
3. **Grader** — an LLM grader scores each retrieved chunk for relevance. Chunks below `RELEVANCE_THRESHOLD` are discarded.
4. **Router** decides the next step:
   - **Relevant docs found** → `Generate` — produces a grounded answer from only the passing chunks.
   - **Weak evidence, retries left** → `Rewrite` — rewrites the query and retries retrieval.
   - **Retries exhausted** → `Fallback` — returns a transparent "insufficient evidence" response.

> Grading runs **before** generation, so the LLM never sees irrelevant context.

### Tunable environment variables

| Variable | Default | Description |
|---|---|---|
| `RELEVANCE_THRESHOLD` | `0.65` | Minimum relevance score to keep a chunk |
| `GRADER_MODEL` | `gpt-4.1-mini` | Model used for relevance grading |
| `GRADER_TEMPERATURE` | `0.0` | Grader LLM temperature |
| `MAX_CORRECTIVE_RETRIES` | `2` | Max rewrite-and-retry attempts |

---

## Project Structure

```
document_qa/
├── backend/
│   ├── main.py            # FastAPI app — routes, rate limiting, metrics
│   ├── ingest.py          # PDF ingestion, chunking, embedding, vectorstore persistence
│   ├── rag.py             # Hybrid retrieval primitives (FAISS + BM25)
│   ├── corrective_rag.py  # LangGraph CRAG pipeline (primary)
│   ├── rag_graph.py       # Experimental ReAct agent (not wired to API)
│   ├── schemas.py         # Pydantic request / response models
│   ├── logger.py          # Structured logging (console INFO + file DEBUG)
│   └── test/              # pytest test suite
├── frontend/
│   └── app.py             # Streamlit UI
├── template.yaml          # AWS SAM deployment template
├── docker-compose.yml
└── .env                   # Local secrets (never commit)
```

---

## Prerequisites

- **Python 3.10+**
- **[uv](https://github.com/astral-sh/uv)** — fast Python package manager
- **OpenAI API key**
- *(optional)* Docker & Docker Compose
- *(optional)* AWS CLI + SAM CLI

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone <repo-url> && cd document_qa

# Backend virtualenv (repo root)
uv venv && source .venv/bin/activate
uv pip install -r backend/requirements.txt
```

### 2. Configure secrets

```bash
cat > .env << 'EOF'
OPENAI_API_KEY=your-key-here
RELEVANCE_THRESHOLD=0.65
GRADER_MODEL=gpt-4.1-mini
GRADER_TEMPERATURE=0.0
MAX_CORRECTIVE_RETRIES=2
EOF
```

### 3. Run the backend

```bash
source .venv/bin/activate
cd backend && uvicorn main:app --reload
# API available at http://localhost:8000
```

### 4. Run the frontend *(separate terminal)*

```bash
cd frontend
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
BACKEND_URL=http://localhost:8000 streamlit run app.py
# UI available at http://localhost:8501
```

---

## Docker Compose

```bash
# Start all services
docker compose up -d --build

# Tail logs
docker compose logs -f

# Stop
docker compose down
```

---

## API Reference

### Health check

```bash
curl http://localhost:8000/health
```

### Upload a PDF

```bash
curl -X POST http://localhost:8000/upload_pdf \
  -F "file=@document.pdf"
```

### Index the uploaded PDF

```bash
curl -X POST "http://localhost:8000/index?path=<path-from-upload-response>"
```

### Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key conclusions?"}'
```

**Response shape:**

```json
{
  "answer": "...",
  "sources": [
    { "content": "...", "page": 4, "source": "file.pdf", "relevance_score": 0.82 }
  ]
}
```

> `/ask` is rate-limited to **10 requests / minute per IP**.

---

## Testing

Tests must be run from the **project root** (not `backend/`) so package imports resolve correctly.

```bash
source .venv/bin/activate
pytest backend/test/
```

> `test_rag.py` requires a populated vectorstore and a valid `OPENAI_API_KEY`.

---

## AWS Deployment (SAM)

The backend deploys as an **AWS Lambda** function (via [Mangum](https://mangum.io/)) and the frontend runs on **ECS Fargate**. The vectorstore and uploads are stored in **S3**.

```bash
sam build
sam deploy --guided   # prompts for OPENAI_API_KEY, image URIs, etc.
```

The SAM template (`template.yaml`) provisions:

| Resource | Purpose |
|---|---|
| Lambda + HTTP API Gateway | Backend |
| S3 Bucket | Uploads + vectorstore |
| ECS Fargate Service | Frontend |
| IAM Roles | Least-privilege access |

In AWS mode (`S3_BUCKET_NAME` set), the vectorstore is cached in-process using S3 ETags to avoid re-downloading on every request.

---

## Observability

| Endpoint | Description |
|---|---|
| `/metrics` | Prometheus metrics (request counts, latencies) |
| `/metrics-json` | Human-readable JSON metrics snapshot |
| `logs/app.log` | DEBUG-level file log |
| stdout | INFO-level console log |

---

## Security Notes

- Never commit `.env` or any file containing your `OPENAI_API_KEY`.
- The `.gitignore` already excludes `.env`, `vectorstore/`, and `data/uploads/`.
