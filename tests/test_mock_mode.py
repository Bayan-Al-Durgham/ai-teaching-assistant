"""Tests for mock mode itself: the whole point is "runs with zero API keys," so these
tests explicitly unset every provider key rather than relying on conftest's placeholders.
"""

import importlib

import pytest

from teaching_assistant import mock as mock_module
from teaching_assistant.deps import StudentContext
from teaching_assistant.models import LessonPlan, Quiz, ResourceList, Rubric


@pytest.fixture()
def mock_agent(monkeypatch, tmp_path):
    """Reimport the whole package with MOCK_MODE on and zero API keys, isolated to a temp
    data dir. Reimporting (not monkeypatching config.MOCK_MODE post-hoc) matters because
    every module in this project reads MOCK_MODE once, at import time, to decide which
    model to build each Agent with."""
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TEACHING_ASSISTANT_MOCK", "1")
    monkeypatch.setenv("TEACHING_ASSISTANT_DATA_DIR", str(tmp_path))

    import teaching_assistant.agent as agent_module
    import teaching_assistant.capabilities.lesson_planning as lesson_planning_module
    import teaching_assistant.capabilities.quiz_generation as quiz_generation_module
    import teaching_assistant.capabilities.resource_recommendation as resource_recommendation_module
    import teaching_assistant.capabilities.rubric_generation as rubric_generation_module
    import teaching_assistant.config as config_module
    import teaching_assistant.rag.store as store_module

    for module in (
        config_module,
        mock_module,
        lesson_planning_module,
        quiz_generation_module,
        rubric_generation_module,
        resource_recommendation_module,
        agent_module,
        store_module,
    ):
        importlib.reload(module)

    assert config_module.MOCK_MODE is True
    yield agent_module.teaching_assistant_agent

    # Reload back to non-mock so later tests in the same process see normal config again.
    # Restore the placeholder key conftest.py set (this fixture deleted it above) - the
    # non-mock branch constructs Agent('anthropic:...') at reload time, which needs *a*
    # key present even though nothing here ever calls it for real.
    monkeypatch.delenv("TEACHING_ASSISTANT_MOCK", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-placeholder")
    for module in (
        config_module,
        mock_module,
        lesson_planning_module,
        quiz_generation_module,
        rubric_generation_module,
        resource_recommendation_module,
        agent_module,
        store_module,
    ):
        importlib.reload(module)


def _deps(**overrides) -> StudentContext:
    defaults = dict(student_name="Alex", course_id="mock-course", learner_level="beginner")
    defaults.update(overrides)
    return StudentContext(**defaults)


def test_agent_builds_and_runs_with_zero_api_keys(mock_agent):
    result = mock_agent.run_sync("What is entropy?", deps=_deps())
    assert "MOCK MODE" in result.output
    assert "search_course_materials" in result.output


def test_router_dispatches_lesson_plan_requests(mock_agent):
    result = mock_agent.run_sync("Please build me a lesson plan on this", deps=_deps())
    assert "create_lesson_plan" in result.output
    assert "How Plants Make Their Own Food: Photosynthesis" in result.output


def test_router_dispatches_quiz_requests(mock_agent):
    result = mock_agent.run_sync("Give me a quiz on this", deps=_deps())
    assert "generate_quiz" in result.output
    assert "Photosynthesis Check" in result.output


def test_router_dispatches_rubric_requests(mock_agent):
    result = mock_agent.run_sync("I need a grading rubric for this assignment", deps=_deps())
    assert "create_rubric" in result.output
    assert "Lab Report Rubric" in result.output


def test_router_dispatches_resource_requests(mock_agent):
    result = mock_agent.run_sync("Can you recommend more resources?", deps=_deps())
    assert "recommend_resources" in result.output
    assert "Khan Academy" in result.output


def test_specialist_mock_returns_the_exact_sample_model():
    from teaching_assistant.mock import SAMPLE_LESSON_PLAN, make_specialist_mock
    from pydantic_ai import Agent

    probe_agent = Agent(make_specialist_mock(SAMPLE_LESSON_PLAN), name="probe", output_type=LessonPlan)
    result = probe_agent.run_sync("anything")
    assert result.output == SAMPLE_LESSON_PLAN


@pytest.mark.parametrize(
    "sample,output_type",
    [
        (mock_module.SAMPLE_LESSON_PLAN, LessonPlan),
        (mock_module.SAMPLE_QUIZ, Quiz),
        (mock_module.SAMPLE_RUBRIC, Rubric),
        (mock_module.SAMPLE_RESOURCE_LIST, ResourceList),
    ],
)
def test_every_sample_validates_as_its_model(sample, output_type):
    assert isinstance(sample, output_type)


def test_mock_embedder_gives_higher_similarity_to_related_text():
    from teaching_assistant.mock import MockEmbedder

    embedder = MockEmbedder()
    photosynthesis_a = embedder.embed_query_sync("What does photosynthesis produce?").embeddings[0]
    photosynthesis_b = embedder.embed_query_sync("How does photosynthesis work in plants?").embeddings[0]
    unrelated = embedder.embed_query_sync("How do I fix a flat bicycle tire?").embeddings[0]

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    related_score = cosine(photosynthesis_a, photosynthesis_b)
    unrelated_score = cosine(photosynthesis_a, unrelated)
    assert related_score > unrelated_score
