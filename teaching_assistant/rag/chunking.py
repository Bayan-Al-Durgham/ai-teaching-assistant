"""Plain word-based chunking with overlap.

This is intentionally simple: no sentence-boundary detection, no PDF/HTML
structure awareness. It's enough to make retrieval useful for the demo and
is easy to swap for a smarter chunker (e.g. one that respects headings or
sentence boundaries) without touching the rest of the RAG pipeline.
"""

from __future__ import annotations


def chunk_text(text: str, *, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split `text` into ~chunk_size-character chunks, each overlapping the
    previous one by ~overlap characters, breaking only on word boundaries.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    n = len(words)

    while start < n:
        chunk_words: list[str] = []
        length = 0
        i = start
        while i < n and (length < chunk_size or not chunk_words):
            chunk_words.append(words[i])
            length += len(words[i]) + 1
            i += 1
        chunks.append(" ".join(chunk_words))

        if i >= n:
            break

        # Back up from the end of this chunk by ~overlap characters so the
        # next chunk shares context with this one.
        overlap_words = 0
        overlap_len = 0
        j = i - 1
        while j > start and overlap_len < overlap:
            overlap_len += len(words[j]) + 1
            overlap_words += 1
            j -= 1
        start = max(i - overlap_words, start + 1)

    return chunks
