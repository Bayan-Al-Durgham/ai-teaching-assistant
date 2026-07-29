import numpy as np

from teaching_assistant.rag import store as rag_store
from teaching_assistant.rag.chunking import chunk_text


def test_chunk_text_covers_all_words_with_overlap():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert all(chunks)
    assert "word0" in chunks[0]
    assert "word299" in chunks[-1]


def test_chunk_text_handles_empty_or_blank_input():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_never_infinite_loops_on_a_single_giant_word():
    # A single "word" longer than chunk_size must still terminate.
    chunks = chunk_text("x" * 5000, chunk_size=100, overlap=20)
    assert chunks == ["x" * 5000]


class _FakeEmbeddingResult:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class _FakeEmbedder:
    """Deterministic stand-in for pydantic_ai.Embedder: same text -> same vector,
    so ingest/search can be tested without a real embeddings API call."""

    def __init__(self, *_args, **_kwargs):
        pass

    @staticmethod
    def _vector(text: str) -> list[float]:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.normal(size=8).tolist()

    def embed_documents_sync(self, documents, **_kwargs):
        return _FakeEmbeddingResult([self._vector(d) for d in documents])

    def embed_query_sync(self, query, **_kwargs):
        return _FakeEmbeddingResult([self._vector(query)])


def test_ingest_and_search_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("TEACHING_ASSISTANT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(rag_store, "Embedder", _FakeEmbedder)

    assert not rag_store.has_materials("test-course")

    count = rag_store.ingest_text(
        "Photosynthesis converts light energy into chemical energy stored in glucose.",
        course_id="test-course",
        source="notes.txt",
    )
    assert count >= 1
    assert rag_store.has_materials("test-course")

    results = rag_store.search("What does photosynthesis produce?", course_id="test-course", k=2)
    assert len(results) >= 1
    assert results[0].source == "notes.txt"
    assert "Photosynthesis" in results[0].text


def test_search_on_a_course_with_nothing_ingested_returns_no_results(monkeypatch, tmp_path):
    monkeypatch.setenv("TEACHING_ASSISTANT_DATA_DIR", str(tmp_path))
    assert rag_store.search("anything", course_id="never-ingested", k=3) == []


def test_courses_are_isolated_from_each_other(monkeypatch, tmp_path):
    monkeypatch.setenv("TEACHING_ASSISTANT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(rag_store, "Embedder", _FakeEmbedder)

    rag_store.ingest_text("Course A material.", course_id="course-a", source="a.txt")

    assert rag_store.has_materials("course-a")
    assert not rag_store.has_materials("course-b")
    assert rag_store.search("material", course_id="course-b", k=3) == []
