import hashlib
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
VECTOR_PREFIX = "vectorstore"
TMP_VS = "/tmp/vectorstore"
LOCAL_VECTOR_DIR = "vectorstore"


def _s3():
    import boto3
    return boto3.client("s3")


def _download_vectorstore():
    """Download index.faiss, index.pkl, docs.pkl from S3 to /tmp/vectorstore/."""
    Path(TMP_VS).mkdir(exist_ok=True)
    s3 = _s3()
    exists = False
    for fname in ("index.faiss", "index.pkl", "docs.pkl"):
        try:
            s3.download_file(S3_BUCKET, f"{VECTOR_PREFIX}/{fname}", f"{TMP_VS}/{fname}")
            exists = True
        except s3.exceptions.ClientError:
            pass
    return exists


def _upload_vectorstore(docs):
    s3 = _s3()
    for fname in ("index.faiss", "index.pkl"):
        s3.upload_file(f"{TMP_VS}/{fname}", S3_BUCKET, f"{VECTOR_PREFIX}/{fname}")
    buf = pickle.dumps(docs)
    s3.put_object(Bucket=S3_BUCKET, Key=f"{VECTOR_PREFIX}/docs.pkl", Body=buf)


def _get_manifest():
    try:
        obj = _s3().get_object(Bucket=S3_BUCKET, Key=f"{VECTOR_PREFIX}/manifest.json")
        return json.loads(obj["Body"].read())
    except Exception:
        return {}


def _put_manifest(manifest):
    _s3().put_object(
        Bucket=S3_BUCKET, Key=f"{VECTOR_PREFIX}/manifest.json",
        Body=json.dumps(manifest).encode(), ContentType="application/json"
    )


def ingest_pdf(s3_key: str) -> dict:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS

    if not S3_BUCKET:
        # Local fallback for dev without S3
        loader = PyPDFLoader(s3_key)
        raw_docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(raw_docs)
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(chunks, embeddings)
        Path(LOCAL_VECTOR_DIR).mkdir(exist_ok=True)
        db.save_local(LOCAL_VECTOR_DIR)
        with open(f"{LOCAL_VECTOR_DIR}/docs.pkl", "wb") as f:
            pickle.dump(chunks, f)
        return {"skipped": False, "chunks": len(chunks)}

    # Download PDF from S3
    tmp_pdf = f"/tmp/{Path(s3_key).name}"
    _s3().download_file(S3_BUCKET, s3_key, tmp_pdf)
    pdf_bytes = Path(tmp_pdf).read_bytes()
    sha256 = hashlib.sha256(pdf_bytes).hexdigest()

    # Deduplication check
    manifest = _get_manifest()
    if sha256 in manifest:
        return {"skipped": True, "reason": "already indexed", "sha256": sha256}

    # Load + split
    loader = PyPDFLoader(tmp_pdf)
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    new_chunks = splitter.split_documents(raw_docs)
    embeddings = OpenAIEmbeddings()

    vs_exists = _download_vectorstore()

    if vs_exists:
        existing_db = FAISS.load_local(TMP_VS, embeddings, allow_dangerous_deserialization=True)
        with open(f"{TMP_VS}/docs.pkl", "rb") as f:
            existing_docs = pickle.load(f)
        new_db = FAISS.from_documents(new_chunks, embeddings)
        existing_db.merge_from(new_db)
        existing_db.save_local(TMP_VS)
        all_docs = existing_docs + new_chunks
    else:
        db = FAISS.from_documents(new_chunks, embeddings)
        db.save_local(TMP_VS)
        all_docs = new_chunks

    _upload_vectorstore(all_docs)

    manifest[sha256] = {
        "filename": Path(s3_key).name,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    _put_manifest(manifest)

    return {"skipped": False, "chunks": len(new_chunks), "sha256": sha256}


if __name__ == "__main__":
    ingest_pdf("/Users/harishsundaralingam/myworkspace/document_qa/data/uploads/AI Engineering.pdf")
