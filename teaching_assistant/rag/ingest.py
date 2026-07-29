"""File-level ingestion helpers on top of `teaching_assistant.rag.store`."""

from __future__ import annotations

from pathlib import Path

from teaching_assistant.rag.store import RetrievedChunk, ingest_text

SUPPORTED_SUFFIXES = {".txt", ".md"}


def ingest_file(path: str | Path, *, course_id: str) -> int:
    """Read a plain-text course material file and add it to `course_id`'s index.

    Only .txt/.md are read directly. For PDFs or slides, extract the text
    first (e.g. with the `pdf` skill) and pass the resulting .txt file here.
    """
    file_path = Path(path)
    if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type '{file_path.suffix}'. Convert to .txt or .md first "
            f"(supported: {sorted(SUPPORTED_SUFFIXES)})."
        )
    text = file_path.read_text(encoding="utf-8")
    return ingest_text(text, course_id=course_id, source=file_path.name)


def format_chunks_for_prompt(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as labeled excerpts for inclusion in a prompt."""
    if not chunks:
        return "No matching course materials were found."
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[{i}] Source: {chunk.source} (relevance {chunk.score:.2f})\n{chunk.text}")
    return "\n\n".join(parts)
