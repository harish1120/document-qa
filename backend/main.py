from mangum import Mangum
import os
import re
from uuid import uuid4

import boto3
import filetype
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from pathlib import Path
from prometheus_client import REGISTRY
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ingest import ingest_pdf
from logger import setup_logger
from rag_graph import answer_question
from schemas import AskRequest, AskResponse

load_dotenv()

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".pdf"}

S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
UPLOAD_DIR = Path("data/uploads")
if not S3_BUCKET:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

logger = setup_logger(__name__)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "rag-backend"
    }


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    logger.info(f"Upload request received for file: {file.filename}")

    # 1. Check file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        logger.warning(f"File too large: {len(content)} bytes")
        raise HTTPException(
            status_code=413, detail="File too large (max 10MB)")

    # 2. Sanitize filename (prevent path traversal)
    safe_filename = Path(file.filename).name
    safe_filename = re.sub(r'[^\w\s.-]', '', safe_filename)

    if not safe_filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    # 3. Validate actual file type (magic bytes)
    kind = filetype.guess(content)
    if kind is None or kind.mime != 'application/pdf':
        logger.warning(f"Invalid file type for {safe_filename}")
        raise HTTPException(status_code=400, detail="File is not a valid PDF")

    # 4. Generate unique filename
    unique_filename = f"{uuid4()}_{safe_filename}"

    # 5. Store file
    try:
        if S3_BUCKET:
            s3_key = f"uploads/{unique_filename}"
            boto3.client("s3").put_object(
                Bucket=S3_BUCKET, Key=s3_key, Body=content)
            logger.info(f"PDF uploaded to S3: {s3_key}")
            return {"message": "PDF Uploaded", "path": s3_key}
        else:
            pdf_path = UPLOAD_DIR / unique_filename
            with open(pdf_path, "wb") as f:
                f.write(content)
            logger.info(f"PDF uploaded locally: {pdf_path}")
            return {"message": "PDF Uploaded", "path": str(pdf_path)}
    except Exception as e:
        logger.error(f"Failed to save PDF: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to save file")


@app.post("/index")
async def index_pdf(path: str):
    logger.info(f"Indexing request for: {path}")
    try:
        result = ingest_pdf(path)
        if result.get("skipped"):
            logger.info(f"Skipped (already indexed): {path}")
            return {"message": "Document already indexed", "skipped": True, "sha256": result.get("sha256")}
        logger.info(
            f"Successfully indexed: {path} ({result.get('chunks')} chunks)")
        return {"message": "Document indexed", "skipped": False, "chunks": result.get("chunks")}
    except Exception as e:
        logger.error(f"Failed to index {path}: {str(e)}", exc_info=True)
        raise


@app.post("/ask", response_model=AskResponse)
@limiter.limit("10/minute")
def ask_question(request: Request, req: AskRequest):
    logger.info(f"Question received: {req.question[:100]}...")
    try:
        answer, sources = answer_question(req.question)
        logger.info(f"Answer generated with {len(sources)} sources")
        logger.debug(f"Sources: {sources}")
        return AskResponse(answer=answer, sources=sources)
    except RuntimeError as e:
        logger.error(f"Timeout answering question: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT, detail=str(e))
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}", exc_info=True)
        raise


@app.get("/metrics-json")
async def metrics_json():
    """Simple JSON view of metrics for debugging"""
    logger.debug("Metrics endpoint called")
    metrics = {}
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            metrics[sample.name] = {
                "value": sample.value,
                "labels": sample.labels
            }
    return metrics


handler = Mangum(app, lifespan="off")
