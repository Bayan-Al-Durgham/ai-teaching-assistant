"""A minimal, file-backed vector store for course materials.

Each course gets one JSON file of chunks + embeddings under
``TEACHING_ASSISTANT_DATA_DIR/rag_index/<course_id>.json``. Search does a
brute-force cosine-similarity scan with numpy, which is fine for the small
per-course corpora this demo is built for.

Swap this module out for a real vector database (pgvector, Chroma, Pinecone,
...) in production — `ingest_text` and `search` are the only two functions
the rest of the app calls, so that's the whole surface to replace.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic_ai import Embedder

from teaching_assistant.config import MOCK_MODE
from teaching_assistant.rag.chunking import chunk_text


def _data_dir() -> Path:
    return Path(os.environ.get("TEACHING_ASSISTANT_DATA_DIR", "./data"))


def _index_path(course_id: str) -> Path:
    return _data_dir() / "rag_index" / f"{course_id}.json"


def _embedding_model() -> str:
    return os.environ.get("EMBEDDING_MODEL", "openai:text-embedding-3-small")


def _get_embedder():
    """Real `Embedder` normally; a deterministic offline mock when MOCK_MODE is on
    (see `teaching_assistant.mock.MockEmbedder` for why a real embeddings API key isn't
    needed either, in that mode)."""
    if MOCK_MODE:
        from teaching_assistant.mock import MockEmbedder

        return MockEmbedder()
    return Embedder(_embedding_model())


def _load_index(course_id: str) -> list[dict]:
    path = _index_path(course_id)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _save_index(course_id: str, chunks: list[dict]) -> None:
    path = _index_path(course_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chunks), encoding="utf-8")


def ingest_text(text: str, *, course_id: str, source: str) -> int:
    """Chunk, embed, and add `text` to `course_id`'s index. Returns chunk count."""
    pieces = chunk_text(text)
    if not pieces:
        return 0

    embedder = _get_embedder()
    result = embedder.embed_documents_sync(pieces)

    existing = _load_index(course_id)
    next_id = len(existing)
    for offset, (piece, vector) in enumerate(zip(pieces, result.embeddings)):
        existing.append(
            {
                "id": next_id + offset,
                "source": source,
                "text": piece,
                "embedding": list(vector),
            }
        )
    _save_index(course_id, existing)
    return len(pieces)


@dataclass
class RetrievedChunk:
    source: str
    text: str
    score: float


def search(query: str, *, course_id: str, k: int = 4) -> list[RetrievedChunk]:
    """Return the `k` most relevant chunks for `query` within `course_id`."""
    index = _load_index(course_id)
    if not index:
        return []

    embedder = _get_embedder()
    query_vector = np.array(embedder.embed_query_sync(query).embeddings[0], dtype=np.float32)

    doc_vectors = np.array([chunk["embedding"] for chunk in index], dtype=np.float32)
    query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
    doc_norms = doc_vectors / (np.linalg.norm(doc_vectors, axis=1, keepdims=True) + 1e-10)
    scores = doc_norms @ query_norm

    top_k = min(k, len(index))
    top_indices = np.argsort(-scores)[:top_k]

    return [
        RetrievedChunk(source=index[i]["source"], text=index[i]["text"], score=float(scores[i]))
        for i in top_indices
    ]


def has_materials(course_id: str) -> bool:
    return bool(_load_index(course_id))
