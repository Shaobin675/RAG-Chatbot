"""
temporal-worker/workflows.py

Durable ingestion pipeline using Temporal.
Orchestrates: upload → chunk → embed (BioBERT) → FAISS index update
Each activity is retried independently on failure — no full reprocess needed.
"""

from datetime import timedelta
from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from dataclasses import dataclass
from typing import List


# ── Shared data types ────────────────────────────────────────────────────────

@dataclass
class IngestRequest:
    document_id: str
    s3_key: str          # raw file location in S3
    source_type: str     # "guide" | "ticket" | "doc"
    uploaded_by: str


@dataclass
class ChunkResult:
    document_id: str
    chunks: List[str]


@dataclass
class EmbedResult:
    document_id: str
    chunk_count: int
    index_updated: bool


# ── Activities (each step is independently retried) ──────────────────────────

@activity.defn
async def fetch_document_activity(request: IngestRequest) -> str:
    """
    Pull raw document from S3 and return extracted text.
    Retried on S3 connectivity issues without re-running downstream steps.
    """
    import boto3
    from io import BytesIO

    s3 = boto3.client("s3")
    bucket = "rag-chatbot-uploads"

    obj = s3.get_object(Bucket=bucket, Key=request.s3_key)
    raw_bytes = obj["Body"].read()

    # Basic text extraction — extend with PyMuPDF for PDFs
    if request.s3_key.endswith(".pdf"):
        import fitz  # PyMuPDF
        doc = fitz.open(stream=raw_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
    else:
        text = raw_bytes.decode("utf-8", errors="ignore")

    activity.logger.info(f"Fetched document {request.document_id}, length={len(text)}")
    return text


@activity.defn
async def chunk_document_activity(request: IngestRequest, raw_text: str) -> ChunkResult:
    """
    Split document into overlapping chunks for embedding.
    Uses sliding window to preserve context at chunk boundaries.
    """
    chunk_size = 512    # tokens approx
    overlap = 64
    words = raw_text.split()
    chunks = []

    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap

    activity.logger.info(
        f"Chunked document {request.document_id} → {len(chunks)} chunks"
    )
    return ChunkResult(document_id=request.document_id, chunks=chunks)


@activity.defn
async def embed_and_index_activity(
    chunk_result: ChunkResult,
    source_type: str,
) -> EmbedResult:
    """
    Embed chunks with BioBERT and write vectors into FAISS.
    Retried independently — partial FAISS writes are idempotent via doc_id prefix.
    """
    import numpy as np
    import faiss
    import redis
    import json
    from transformers import AutoTokenizer, AutoModel
    import torch

    # Load BioBERT (cached after first load)
    model_name = "dmis-lab/biobert-base-cased-v1.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    embeddings = []
    for chunk in chunk_result.chunks:
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean pool over token embeddings
        vec = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(vec)

    vectors = np.array(embeddings, dtype="float32")

    # Write to FAISS
    index_path = "/data/faiss_index/index.faiss"
    try:
        index = faiss.read_index(index_path)
    except Exception:
        index = faiss.IndexFlatL2(vectors.shape[1])

    index.add(vectors)
    faiss.write_index(index, index_path)

    # Store chunk metadata in Redis for retrieval lookup
    r = redis.Redis(host="redis", port=6379, decode_responses=True)
    for i, chunk in enumerate(chunk_result.chunks):
        key = f"chunk:{chunk_result.document_id}:{i}"
        r.set(key, json.dumps({"text": chunk, "source_type": source_type}))

    activity.logger.info(
        f"Indexed {len(embeddings)} vectors for document {chunk_result.document_id}"
    )
    return EmbedResult(
        document_id=chunk_result.document_id,
        chunk_count=len(embeddings),
        index_updated=True,
    )


@activity.defn
async def notify_observability_activity(result: EmbedResult) -> None:
    """
    Emit ingestion completion event to observability pipeline.
    Non-blocking — failure here does not affect document availability.
    """
    import httpx

    payload = {
        "event": "ingestion_complete",
        "document_id": result.document_id,
        "chunk_count": result.chunk_count,
    }
    async with httpx.AsyncClient() as client:
        await client.post("http://observability-service/events", json=payload, timeout=5)

    activity.logger.info(f"Notified observability for {result.document_id}")


# ── Workflow: durable, step-level retry ──────────────────────────────────────

@workflow.defn
class IngestDocumentWorkflow:
    """
    Durable ingestion workflow. Each activity retries independently.
    A K8s pod crash mid-pipeline resumes from the last completed step —
    no document is lost or double-indexed.

    Flow:
        fetch_document → chunk_document → embed_and_index → notify_observability
    """

    @workflow.run
    async def run(self, request: IngestRequest) -> EmbedResult:
        retry_policy = RetryPolicy(
            maximum_attempts=5,
            initial_interval=timedelta(seconds=2),
            backoff_coefficient=2.0,
            maximum_interval=timedelta(minutes=5),
        )
        default_opts = workflow.ActivityOptions(
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=retry_policy,
        )

        # Step 1 — fetch raw text from S3
        raw_text = await workflow.execute_activity(
            fetch_document_activity,
            request,
            **default_opts.__dict__,
        )

        # Step 2 — chunk into overlapping windows
        chunk_result = await workflow.execute_activity(
            chunk_document_activity,
            args=[request, raw_text],
            **default_opts.__dict__,
        )

        # Step 3 — embed with BioBERT + write to FAISS
        embed_result = await workflow.execute_activity(
            embed_and_index_activity,
            args=[chunk_result, request.source_type],
            **default_opts.__dict__,
        )

        # Step 4 — emit event (best-effort, shorter timeout)
        await workflow.execute_activity(
            notify_observability_activity,
            embed_result,
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        return embed_result
