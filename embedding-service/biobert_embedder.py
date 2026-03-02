"""
embedding-service/biobert_embedder.py

BioBERT-based embedding and FAISS retrieval.
Used by:
  - chat-orchestrator: embed query → retrieve top-k chunks at inference time
  - temporal-worker:   embed document chunks → write to FAISS at ingestion time

BioBERT is chosen over generic embeddings because the knowledge corpus
(system guides, operational docs, support tickets) contains domain-specific
terminology that benefits from biomedical/technical pretraining.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

import faiss
import numpy as np
import redis
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
FAISS_INDEX_PATH = "/data/faiss_index/index.faiss"
REDIS_HOST = "redis"
REDIS_PORT = 6379
EMBEDDING_DIM = 768  # BioBERT hidden size


# ── Model singleton (loaded once per process) ────────────────────────────────

@lru_cache(maxsize=1)
def load_model():
    logger.info(f"Loading BioBERT model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("BioBERT running on GPU")
    return tokenizer, model


# ── Core embedding function ──────────────────────────────────────────────────

def embed_text(text: str) -> np.ndarray:
    """
    Embed a single text string using BioBERT mean pooling.
    Returns a float32 numpy vector of shape (768,).
    """
    tokenizer, model = load_model()
    device = next(model.parameters()).device

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pool over token dimension, move to CPU
    vec = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return vec.astype("float32")


def embed_batch(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Embed a list of texts in batches.
    Returns a float32 numpy array of shape (N, 768).
    """
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs = np.array([embed_text(t) for t in batch])
        all_vecs.append(vecs)
        logger.debug(f"Embedded batch {i // batch_size + 1} ({len(batch)} texts)")
    return np.vstack(all_vecs)


# ── FAISS retrieval ──────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    text: str
    source_type: str      # "guide" | "ticket" | "doc"
    document_id: str
    chunk_index: int
    score: float          # L2 distance (lower = more similar)


def load_faiss_index() -> faiss.Index:
    """Load FAISS index from disk. Called fresh per request to pick up new writes."""
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        logger.debug(f"Loaded FAISS index with {index.ntotal} vectors")
        return index
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")
        raise


def retrieve_top_k(
    query: str,
    k: int = 5,
    source_filter: Optional[str] = None,
) -> List[RetrievedChunk]:
    """
    Embed query with BioBERT and retrieve top-k most similar chunks from FAISS.

    Args:
        query:         Natural language query from the user.
        k:             Number of chunks to retrieve.
        source_filter: Optionally restrict to "guide", "ticket", or "doc".

    Returns:
        List of RetrievedChunk sorted by ascending L2 distance (most relevant first).
    """
    query_vec = embed_text(query).reshape(1, -1)
    index = load_faiss_index()

    # Retrieve more candidates if filtering, then trim to k after filter
    fetch_k = k * 3 if source_filter else k
    distances, indices = index.search(query_vec, fetch_k)

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    results: List[RetrievedChunk] = []

    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue

        # Resolve chunk metadata from Redis
        # Key format: chunk:{document_id}:{chunk_index}
        pattern = f"chunk:*:{idx}"
        keys = r.keys(pattern)
        if not keys:
            continue

        key = keys[0]
        raw = r.get(key)
        if not raw:
            continue

        meta = json.loads(raw)

        # Apply source filter if requested
        if source_filter and meta.get("source_type") != source_filter:
            continue

        parts = key.split(":")
        doc_id = parts[1] if len(parts) >= 3 else "unknown"
        chunk_idx = int(parts[2]) if len(parts) >= 3 else idx

        results.append(
            RetrievedChunk(
                text=meta["text"],
                source_type=meta.get("source_type", "unknown"),
                document_id=doc_id,
                chunk_index=chunk_idx,
                score=float(dist),
            )
        )

        if len(results) >= k:
            break

    logger.info(f"Retrieved {len(results)} chunks for query (top-{k})")
    return results


def format_context(chunks: List[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a structured context string
    for injection into the LLM prompt.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Source {i} | type={chunk.source_type} | doc={chunk.document_id}]\n"
            f"{chunk.text.strip()}"
        )
    return "\n\n---\n\n".join(parts)
