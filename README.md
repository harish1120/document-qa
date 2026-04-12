# PDF Question Answering — RAG on AWS

A full-stack application that lets you upload PDF documents and ask questions about them in natural language. The system finds the most relevant passages from your documents and uses an LLM to write a grounded answer.

This README explains **what the app does**, **how each piece works**, and **why the infrastructure is designed the way it is** — written to be useful if you're learning.

---

## Table of Contents

1. [What is RAG?](#1-what-is-rag)
2. [Project Structure](#2-project-structure)
3. [How the App Works End-to-End](#3-how-the-app-works-end-to-end)
4. [Backend Deep Dive](#4-backend-deep-dive)
5. [Frontend](#5-frontend)
6. [Infrastructure: Why Serverless?](#6-infrastructure-why-serverless)
7. [AWS Architecture](#7-aws-architecture)
8. [Running Locally](#8-running-locally)
9. [Deploying to AWS](#9-deploying-to-aws)
10. [API Reference](#10-api-reference)

---

## 1. What is RAG?

**RAG (Retrieval-Augmented Generation)** is a pattern for answering questions using your own documents rather than relying on what an LLM already knows from training.

The problem with asking an LLM a question directly:
- It can hallucinate (make up facts confidently)
- It doesn't know about your private documents
- Its knowledge has a cutoff date

RAG solves this in two steps:

```
Step 1 — Retrieval: Find the most relevant chunks of text from your documents
Step 2 — Generation: Give those chunks to the LLM as context, then ask your question
```

The LLM is instructed to answer **only** using the provided context, so it can't make things up. If the answer isn't there, it says "I don't know".

---

## 2. Project Structure

```
document_qa/
├── backend/
│   ├── main.py              # FastAPI app — HTTP endpoints
│   ├── ingest.py            # PDF processing and vector indexing
│   ├── rag.py               # Hybrid search + LLM answer generation
│   ├── schemas.py           # Pydantic request/response models
│   ├── logger.py            # Logging setup
│   ├── Dockerfile           # Standard image (Docker Compose / EC2)
│   ├── Dockerfile.lambda    # Lambda-compatible image
│   └── requirements.txt
├── frontend/
│   ├── app.py               # Streamlit UI
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml       # Local development setup
├── template.yaml            # AWS SAM — defines all cloud infrastructure
└── .env                     # Your secrets (never commit this)
```

---

## 3. How the App Works End-to-End

There are three user actions, each hitting a different endpoint:

### Action 1: Upload a PDF (`POST /upload_pdf`)

```
User picks a file
      │
      ▼
Frontend sends multipart/form-data to backend
      │
      ▼
Backend validates:
  - File size ≤ 10MB
  - Filename ends in .pdf
  - Magic bytes confirm it's actually a PDF (not a renamed .exe)
      │
      ▼
File is saved:
  - Locally:  data/uploads/<uuid>_filename.pdf
  - On AWS:   S3 bucket at uploads/<uuid>_filename.pdf
      │
      ▼
Returns the file path (used in the next step)
```

The UUID prefix prevents filename collisions if two people upload files with the same name.

### Action 2: Index the document (`POST /index?path=<path>`)

```
Frontend sends the path returned from upload
      │
      ▼
ingest_pdf() is called:
  1. Download PDF from S3 (or read from local disk)
  2. Compute SHA256 hash of the file bytes
  3. Check manifest.json in S3 — if this hash exists, skip (already indexed)
  4. Load PDF pages with PyPDFLoader
  5. Split pages into 500-character chunks (100-char overlap)
  6. Embed each chunk with OpenAI's embedding model → float vectors
  7. Store vectors in a FAISS index
  8. If a vectorstore already exists (prior uploads), merge the new index into it
  9. Upload index.faiss, index.pkl, docs.pkl back to S3
  10. Update manifest.json with the SHA256 so this file won't be re-indexed
```

**Why chunking?** Embedding models have token limits, and shorter focused chunks retrieve better than entire pages. Overlap ensures sentences at chunk boundaries aren't cut off.

**Why SHA256 deduplication?** If you upload the same PDF twice, the hash matches and indexing is skipped — saving time and OpenAI API cost.

**Why merge instead of overwrite?** The app supports multiple documents. Each new upload adds to the shared index, so you can ask questions that span multiple PDFs.

### Action 3: Ask a question (`POST /ask`)

```
User types a question
      │
      ▼
hybrid_search() runs:
  ┌─────────────────────────────────────────────────────┐
  │  Dense search (FAISS)                               │
  │  - Embed the question with OpenAI                   │
  │  - Find 10 nearest vectors by L2 distance           │
  │  - Closer = more semantically similar               │
  └─────────────────────────────────────────────────────┘
          +
  ┌─────────────────────────────────────────────────────┐
  │  Sparse search (BM25)                               │
  │  - Classic keyword scoring (like a search engine)   │
  │  - Scores every document chunk for word overlap     │
  └─────────────────────────────────────────────────────┘
          │
          ▼
  Combine scores:
  final = 0.5 × (1 - dense_score) + 0.5 × sparse_score
          │
          ▼
  Top 5 chunks become the "context"
      │
      ▼
LLM (gpt-5-nano) receives:
  "Answer using ONLY this context: <chunks>\n\nQuestion: <question>"
      │
      ▼
Returns answer + source citations
```

**Why hybrid search?** Dense search understands meaning ("car" ≈ "vehicle") but misses exact keywords. Sparse search finds exact terms but doesn't understand synonyms. Combining them covers both cases.

**Why α = 0.5?** This gives equal weight to both methods. You could tune this — a higher α favors semantic search, lower favors keyword search.

---

## 4. Backend Deep Dive

### `main.py` — The API layer

FastAPI handles HTTP. Key things to know:

- **`UploadFile`** — FastAPI's type for file uploads; `await file.read()` reads bytes
- **`filetype.guess()`** — checks the file's magic bytes (first few bytes that identify file format), not just the extension. A renamed `.exe` would fail this check.
- **`@limiter.limit("10/minute")`** — rate limiting on the `/ask` endpoint to prevent abuse; uses slowapi + Redis-compatible storage
- **`Mangum(app)`** — a thin wrapper that translates AWS Lambda's event format into ASGI (what FastAPI speaks). Without this, Lambda wouldn't know how to call FastAPI.

### `ingest.py` — PDF processing

- **`PyPDFLoader`** — LangChain wrapper around pypdf; extracts text page by page
- **`RecursiveCharacterTextSplitter`** — splits text preferring natural boundaries (paragraphs → sentences → words) before hard-cutting at 500 chars
- **`OpenAIEmbeddings`** — calls OpenAI's embedding API; converts text to a list of ~1536 floats that encode semantic meaning
- **`FAISS`** — Facebook AI Similarity Search; an in-memory index optimized for finding nearest vectors fast. `save_local()` writes two files: `index.faiss` (the vectors) and `index.pkl` (metadata)
- **`merge_from()`** — FAISS method to combine two indices without re-embedding anything

### `rag.py` — Retrieval and generation

- **`BM25Okapi`** — BM25 is a ranking algorithm (used by Elasticsearch, Lucene). "Okapi" is a specific variant. It scores documents based on term frequency and document length normalization.
- **ETag caching** — S3 returns an ETag (a hash of the file contents) with every `HEAD` request. If the ETag hasn't changed since last time, we reuse the in-memory vectorstore and skip the S3 download. This matters in Lambda where the container stays warm between invocations.
- **`@retry(stop_after_attempt(3), wait_exponential(...))`** — tenacity decorator; if OpenAI returns a rate-limit error, it waits 4s, then 8s, then 16s before giving up

### `schemas.py` — Data contracts

Pydantic models validate request/response shapes:

```python
class AskRequest(BaseModel):
    question: str          # input must have a "question" string field

class AskResponse(BaseModel):
    answer: str
    sources: list[dict]    # list of {content, page, source}
```

If the client sends malformed JSON, FastAPI automatically returns a 422 error.

### `logger.py` — Logging

Previously wrote to both stdout and a `logs/app.log` file. The file handler was removed for Lambda because:
- Lambda's filesystem is ephemeral (gone after the container is recycled)
- Lambda automatically captures stdout and ships it to **CloudWatch Logs**
- Writing to a file you can't read is pointless overhead

---

## 5. Frontend

`frontend/app.py` is a Streamlit app — a Python library that turns scripts into interactive web UIs without writing any HTML/JS.

Key things:
- `st.file_uploader()` — renders a file picker; returns file bytes when something is selected
- The upload → index flow is sequential: upload first, get the path back, then index using that path
- `BACKEND_URL` is read from an environment variable so the same Docker image works locally (`http://backend:8000`) and on AWS (the API Gateway URL)

---

## 6. Infrastructure: Why Serverless?

The original setup ran everything on a single EC2 instance with Docker Compose. That works, but has problems:

| Problem | Explanation |
|---|---|
| Always-on cost | EC2 charges by the hour even when nobody is using the app |
| Manual scaling | If traffic spikes, you have to manually resize the instance |
| OS management | You're responsible for patching, security updates, disk space |
| Single point of failure | If the instance goes down, everything goes down |

The serverless migration fixes these:

| Service | Replaces | Why |
|---|---|---|
| **AWS Lambda** | EC2 backend process | Runs only when a request arrives; billed per 100ms; auto-scales |
| **API Gateway** | nginx / port exposure | Managed HTTPS endpoint; routes requests to Lambda |
| **ECS Fargate** | EC2 frontend process | Containers without managing servers; Streamlit needs persistent connections so Lambda isn't suitable |
| **S3** | Local filesystem | Durable, cheap, accessible from both Lambda and Fargate; survives container restarts |

**Why keep Streamlit on Fargate instead of Lambda?**
Lambda is stateless and has a 15-minute timeout. Streamlit keeps a WebSocket connection open while you're on the page. Lambda would kill that connection after a short idle period. Fargate runs containers that stay alive.

**Why keep local mode working?**
Both `ingest.py` and `rag.py` check `if not S3_BUCKET:` and fall back to the local filesystem. This means `docker compose up` still works without any AWS credentials — useful for development and testing.

---

## 7. AWS Architecture

```
                        ┌─────────────────────────────┐
                        │         User Browser         │
                        └──────────────┬──────────────┘
                                       │ HTTP :8501
                                       ▼
                        ┌─────────────────────────────┐
                        │     ECS Fargate (Frontend)   │
                        │       Streamlit :8501        │
                        └──────────────┬──────────────┘
                                       │ HTTPS
                                       ▼
                        ┌─────────────────────────────┐
                        │   API Gateway (HTTP API)     │
                        │   CORS enabled               │
                        └──────────────┬──────────────┘
                                       │ invoke
                                       ▼
                        ┌─────────────────────────────┐
                        │    AWS Lambda (Backend)      │
                        │    FastAPI + Mangum          │
                        │    1024MB, 300s timeout      │
                        └──────────────┬──────────────┘
                                       │ GetObject / PutObject
                                       ▼
                        ┌─────────────────────────────┐
                        │         S3 Bucket            │
                        │  uploads/       (PDFs)       │
                        │  vectorstore/   (FAISS idx)  │
                        │  vectorstore/manifest.json   │
                        └─────────────────────────────┘
```

All infrastructure is defined in `template.yaml` using **AWS SAM** (Serverless Application Model) — a superset of CloudFormation that adds shortcuts for Lambda, API Gateway, and more. `sam deploy` reads this file and creates/updates everything in one command.

### What `template.yaml` defines

| Resource | Type | Purpose |
|---|---|---|
| `DocsBucket` | S3 Bucket | Stores PDFs and vectorstore; versioning on for safety |
| `BackendFunction` | Lambda Function | Runs the FastAPI app via Mangum |
| `BackendApi` | HTTP API Gateway | Public HTTPS endpoint for the Lambda |
| `BackendExecutionRole` | IAM Role | Grants Lambda permission to read/write the S3 bucket |
| `FrontendCluster` | ECS Cluster | Logical grouping for Fargate tasks |
| `FrontendTaskDef` | ECS Task Definition | Container spec: image, CPU/memory, env vars |
| `FrontendService` | ECS Service | Keeps 1 Fargate task running; restarts if it crashes |
| `FrontendSecurityGroup` | Security Group | Firewall rule allowing inbound TCP on port 8501 |

---

## 8. Running Locally

### Prerequisites

- Docker Desktop
- OpenAI API key

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/harish1120/document_qa.git
cd document_qa

# 2. Create .env (never commit this file)
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Start both services
docker compose up -d --build

# 4. Open the UI
open http://localhost:8501
```

When `S3_BUCKET_NAME` is not set (the default), the app stores everything locally:
- PDFs go to `data/uploads/`
- Vectorstore goes to `vectorstore/`

### Run tests

```bash
# From the project root (not from backend/)
pytest backend/test/
```

Tests use `backend.*` import paths defined in `conftest.py`, so the working directory matters.

---

## 9. Deploying to AWS

### Prerequisites

```bash
brew install awscli aws-sam-cli
aws configure        # enter your Access Key, Secret, region
```

### Step 1 — Create ECR repositories

ECR (Elastic Container Registry) is AWS's Docker registry — where your images live.

```bash
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1

aws ecr create-repository --repository-name rag-backend-lambda --region $REGION
aws ecr create-repository --repository-name rag-frontend        --region $REGION
```

### Step 2 — Authenticate Docker to ECR

```bash
aws ecr get-login-password --region $REGION \
  | docker login --username AWS --password-stdin \
    $ACCOUNT.dkr.ecr.$REGION.amazonaws.com
```

### Step 3 — Build and push images

```bash
# Backend Lambda image (uses Dockerfile.lambda — AWS Lambda base image)
docker build -f backend/Dockerfile.lambda -t rag-backend-lambda ./backend
docker tag  rag-backend-lambda:latest \
  $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/rag-backend-lambda:latest
docker push $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/rag-backend-lambda:latest

# Frontend Fargate image (uses standard Dockerfile)
docker build -f frontend/Dockerfile -t rag-frontend ./frontend
docker tag  rag-frontend:latest \
  $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/rag-frontend:latest
docker push $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/rag-frontend:latest
```

**Why two Dockerfiles for the backend?**
`Dockerfile` uses a standard Python base image and starts uvicorn — for Docker Compose. `Dockerfile.lambda` uses AWS's Lambda base image, which includes the Lambda runtime that knows how to invoke your `handler` function. The app code is the same; only the entrypoint differs.

### Step 4 — Deploy with SAM

```bash
sam build
sam deploy --guided
```

SAM will ask:

| Prompt | What to enter |
|---|---|
| Stack name | `rag-app` |
| Region | `us-east-1` |
| `OpenAIApiKey` | your key (hidden input) |
| `BackendImageUri` | `<account>.dkr.ecr.us-east-1.amazonaws.com/rag-backend-lambda:latest` |
| `FrontendImageUri` | `<account>.dkr.ecr.us-east-1.amazonaws.com/rag-frontend:latest` |

Your answers are saved to `samconfig.toml`. Future deploys just need `sam deploy`.

### Step 5 — Verify

```bash
# SAM prints the API Gateway URL when deploy finishes
curl https://<api-gw-id>.execute-api.us-east-1.amazonaws.com/health
# → {"status":"ok","service":"rag-backend"}
```

### Updating after code changes

```bash
# Rebuild and push the changed image, then:
sam deploy    # picks up the new image and updates Lambda/Fargate
```

### Teardown

```bash
sam delete --stack-name rag-app

# S3 bucket is retained by default — delete manually if needed:
aws s3 rb s3://<bucket-name> --force
```

---

## 10. API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Returns `{"status":"ok"}` |
| `POST` | `/upload_pdf` | Upload a PDF; returns `{"path": "<s3-key or local-path>"}` |
| `POST` | `/index?path=<path>` | Index an uploaded PDF; returns `{"skipped": bool, "chunks": int}` |
| `POST` | `/ask` | Ask a question; body: `{"question": "..."}` |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/metrics-json` | Same metrics as JSON (useful for debugging) |

### Upload PDF

```bash
curl -X POST http://localhost:8000/upload_pdf \
  -F "file=@document.pdf"
# → {"message":"PDF Uploaded","path":"data/uploads/abc123_document.pdf"}
```

### Index Document

```bash
curl -X POST "http://localhost:8000/index?path=data/uploads/abc123_document.pdf"
# → {"message":"Document indexed","skipped":false,"chunks":142}
# Second upload of same file:
# → {"message":"Document already indexed","skipped":true,"sha256":"a3f..."}
```

### Ask a Question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main conclusions?"}'
# → {"answer":"...","sources":[{"content":"...","page":3,"source":"doc.pdf"}]}
```

---

## Key Technologies

| Library | What it does |
|---|---|
| **FastAPI** | Python web framework; async, auto-generates OpenAPI docs |
| **Streamlit** | Turns Python scripts into interactive web apps |
| **LangChain** | Framework for building LLM pipelines; handles loaders, splitters, vectorstores |
| **FAISS** | Facebook's fast approximate nearest-neighbor search library |
| **BM25** | Classic probabilistic keyword ranking (used in Elasticsearch) |
| **OpenAI** | Embedding model + GPT for generation |
| **Mangum** | ASGI adapter for AWS Lambda |
| **boto3** | AWS SDK for Python |
| **AWS SAM** | Infrastructure-as-code tool for serverless AWS apps |
| **tenacity** | Retry logic with exponential backoff |
| **slowapi** | Rate limiting middleware for FastAPI |
| **Pydantic** | Data validation using Python type hints |
