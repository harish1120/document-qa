import os
import pickle
from pathlib import Path

import boto3
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from aider_sdk import AiderChat, AiderEmbeddings
from openai import OpenAIError
from tenacity import retry, stop_after_attempt, wait_exponential

from logger import setup_logger

S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
LOCAL_VECTOR_DIR = "vectorstore"
TMP_VS = "/tmp/vectorstore"

_cache = {"etag": None, "db": None, "bm25": None}

PROMPT = """
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

logger = setup_logger(__name__)


def _load_vectorstore():
    """Load vectorstore and BM25 from S3 (with ETag cache) or local disk."""
    embeddings = AiderEmbeddings()

    if not S3_BUCKET:
        db = FAISS.load_local(LOCAL_VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
        with open(f"{LOCAL_VECTOR_DIR}/bm25.pkl", "rb") as f:
            bm25 = pickle.load(f)
        return db, bm25

    s3 = boto3.client("s3")
    head = s3.head_object(Bucket=S3_BUCKET, Key="vectorstore/index.faiss")
    etag = head["ETag"]
    if _cache["etag"] == etag and _cache["db"] is not None:
        return _cache["db"], _cache["bm25"]

    Path(TMP_VS).mkdir(exist_ok=True)
    for fname in ("index.faiss", "index.pkl", "bm25.pkl"):
        s3.download_file(S3_BUCKET, f"vectorstore/{fname}", f"{TMP_VS}/{fname}")

    db = FAISS.load_local(TMP_VS, embeddings, allow_dangerous_deserialization=True)
    with open(f"{TMP_VS}/bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    _cache.update({"etag": etag, "db": db, "bm25": bm25})
    return db, bm25


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def answer_question(question: str):
    try:
        docs = hybrid_search(question, k=10)
        context = "\n\n".join([d.page_content for d in docs])

        llm = AiderChat(model="aider-model", temperature=0.5)

        prompt = PromptTemplate(
            template=PROMPT,
            input_variables=["context", "question"]
        )

        response = llm.invoke(prompt.format(
            context=context,
            question=question
        ))

        sources = []
        for d in docs:
            sources.append({
                "content": d.page_content,
                "page": d.metadata.get("page"),
                "source": d.metadata.get("source"),
            })

        return response.content, sources

    except OpenAIError as e:
        logger.error(f"%OpenAI API error: {e}")
        raise


def hybrid_search(query: str, k: int = 5, alpha: float = 0.5):
    db, bm25 = _load_vectorstore()

    # dense
    dense_docs = db.similarity_search_with_score(query, k=10)

    # sparse
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (
        np.max(bm25_scores) - np.min(bm25_scores) + 1e-9
    )

    scored = []
    for idx, (doc, dense_score) in enumerate(dense_docs):
        sparse_score = bm25_scores[idx]
        final_score = alpha * (1 - dense_score) + (1 - alpha) * sparse_score
        scored.append((doc, final_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:k]]


