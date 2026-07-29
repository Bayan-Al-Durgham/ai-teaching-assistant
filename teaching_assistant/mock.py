"""A zero-API-key mock layer, for studying the architecture without any credentials.

What's mocked and what isn't matters here:

- The 5 `Agent` objects in this project (the main agent + 4 specialist sub-agents) each
  normally send a real request to a real LLM provider. In mock mode, each is constructed
  with a `pydantic_ai.models.function.FunctionModel` instead of a provider model string, so
  no network call and no API key is ever needed to construct or run them.
- Every *tool* those agents call is still the real project code: `search_course_materials`
  really searches the real (mock-embedded) index, `create_lesson_plan` really calls
  `lesson_plan_agent.run(...)` and really renders the result with `format_lesson_plan`. The
  only thing swapped out is the "which tool should I call, and what should a specialist
  agent's structured answer be" decision that a real LLM would normally make.
- `Embedder` (used to turn course-material text into vectors for RAG) also needs a real API
  key normally (OpenAI by default - see README). `MockEmbedder` replaces it with a small
  deterministic hashing-trick bag-of-words vector, entirely offline. It's not a real
  embedding model, but text that shares words *does* score as more similar, so RAG retrieval
  in mock mode still behaves sensibly instead of returning arbitrary chunks.

None of this is meant to look like a real LLM. Every mock response is prefixed so it's
obvious in the transcript which parts are canned.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from teaching_assistant.models import (
    LessonActivity,
    LessonPlan,
    Quiz,
    QuizQuestion,
    ResourceList,
    ResourceRecommendation,
    Rubric,
    RubricCriterion,
    RubricLevel,
)

MOCK_BANNER = "[MOCK MODE — no LLM was called]"

# ---------------------------------------------------------------------------
# Canned sample outputs for the 4 specialist sub-agents.
#
# Themed around the bundled sample_course_materials/intro_to_photosynthesis.txt so a
# mock-mode walkthrough (ingest that file, then ask for a lesson plan / quiz / rubric /
# resources) reads as one coherent example instead of unrelated canned text.
# ---------------------------------------------------------------------------

SAMPLE_LESSON_PLAN = LessonPlan(
    title="How Plants Make Their Own Food: Photosynthesis",
    subject="Science",
    grade_level="7-8",
    duration_minutes=45,
    objectives=[
        "State the word equation for photosynthesis.",
        "Explain the role of chlorophyll, sunlight, water, and carbon dioxide.",
        "Describe why photosynthesis matters beyond the individual plant.",
    ],
    materials=["Whiteboard", "Diagram of a leaf cross-section", "Exit ticket slips"],
    warm_up="Ask: 'Plants don't eat food like we do — so where does their energy come from?' Collect a few guesses.",
    activities=[
        LessonActivity(
            name="Diagram walkthrough",
            duration_minutes=15,
            description="Label a chloroplast diagram together, tracing carbon dioxide and water in, sugar and oxygen out.",
        ),
        LessonActivity(
            name="Equation build",
            duration_minutes=15,
            description="Students build the word equation for photosynthesis from labeled cards, then explain it in their own words to a partner.",
        ),
        LessonActivity(
            name="Think-pair-share",
            duration_minutes=10,
            description="Pairs discuss: 'What would happen to the oxygen in the air if plants disappeared?'",
        ),
    ],
    assessment="Circulate during the equation-build activity; listen for whether students can name all 4 inputs/outputs.",
    closure="Cold-call 2-3 students to state the word equation in one sentence, in their own words.",
    differentiation_notes="[MOCK] A real run would tailor this to the student's learner_level dependency.",
)

SAMPLE_QUIZ = Quiz(
    title="Photosynthesis Check",
    topic="Photosynthesis",
    difficulty="medium",
    questions=[
        QuizQuestion(
            question="What gas do plants take in during photosynthesis?",
            question_type="multiple_choice",
            options=["Oxygen", "Carbon dioxide", "Nitrogen"],
            correct_answer="Carbon dioxide",
            explanation="Plants absorb carbon dioxide through stomata and combine it with water to build glucose.",
        ),
        QuizQuestion(
            question="True or false: photosynthesis happens only in the roots.",
            question_type="true_false",
            options=["True", "False"],
            correct_answer="False",
            explanation="Photosynthesis happens mainly in the leaves, inside chloroplasts.",
        ),
        QuizQuestion(
            question="Name the green pigment that absorbs light energy for photosynthesis.",
            question_type="short_answer",
            options=None,
            correct_answer="Chlorophyll",
            explanation="Chlorophyll absorbs red and blue light and reflects green light, which is why leaves look green.",
        ),
        QuizQuestion(
            question="Which gas is released as a waste product of photosynthesis?",
            question_type="multiple_choice",
            options=["Oxygen", "Carbon dioxide", "Hydrogen"],
            correct_answer="Oxygen",
            explanation="Splitting water in the light-dependent reactions releases oxygen through the stomata.",
        ),
        QuizQuestion(
            question="What sugar molecule does the Calvin cycle ultimately build?",
            question_type="short_answer",
            options=None,
            correct_answer="Glucose",
            explanation="ATP and NADPH from stage one power the Calvin cycle, which builds glucose from carbon dioxide.",
        ),
    ],
)

SAMPLE_RUBRIC = Rubric(
    title="Photosynthesis Rate Lab Report Rubric",
    assignment_description="A short lab report investigating how light intensity affects the rate of photosynthesis.",
    total_points=20,
    criteria=[
        RubricCriterion(
            name="Data analysis",
            weight_percent=50,
            levels=[
                RubricLevel(label="Proficient", score_range="9-10", description="Correctly identifies the trend and connects it to light intensity as a limiting factor."),
                RubricLevel(label="Developing", score_range="5-8", description="Describes the data but doesn't connect it to a limiting factor."),
                RubricLevel(label="Beginning", score_range="0-4", description="Restates data points without identifying any trend."),
            ],
        ),
        RubricCriterion(
            name="Use of vocabulary",
            weight_percent=25,
            levels=[
                RubricLevel(label="Proficient", score_range="5", description="Uses chlorophyll, chloroplast, and limiting factor correctly."),
                RubricLevel(label="Developing", score_range="2-4", description="Uses at least one term correctly."),
                RubricLevel(label="Beginning", score_range="0-1", description="Does not use key vocabulary."),
            ],
        ),
        RubricCriterion(
            name="Conclusion",
            weight_percent=25,
            levels=[
                RubricLevel(label="Proficient", score_range="5", description="States a claim supported by specific data from the experiment."),
                RubricLevel(label="Developing", score_range="2-4", description="States a claim with weak or no supporting data."),
                RubricLevel(label="Beginning", score_range="0-1", description="No clear claim."),
            ],
        ),
    ],
)

SAMPLE_RESOURCE_LIST = ResourceList(
    topic="Photosynthesis",
    resources=[
        ResourceRecommendation(
            title="Introduction to Photosynthesis (this course's own material)",
            resource_type="course_material",
            reference="sample_course_materials/intro_to_photosynthesis.txt",
            level="beginner",
            why_recommended="Already matches the vocabulary and depth used in class.",
        ),
        ResourceRecommendation(
            title="Khan Academy — Photosynthesis topic",
            resource_type="interactive",
            reference="Khan Academy, Biology > Photosynthesis (search their site for the current link)",
            level="beginner",
            why_recommended="Short videos and practice questions matched to a beginner level.",
        ),
        ResourceRecommendation(
            title="A general biology textbook chapter on the light-dependent and Calvin cycle reactions",
            resource_type="book",
            reference="Any current middle/high school biology textbook, chapter on photosynthesis",
            level="intermediate",
            why_recommended="Goes one level deeper into the two-stage mechanism than the course notes do.",
        ),
    ],
)


def make_specialist_mock(sample: BaseModel) -> FunctionModel:
    """Build a FunctionModel for a specialist sub-agent that always 'answers' with `sample`.

    Structured output in Pydantic AI is implemented as a tool call to a tool named
    `final_result` by default (confirmed by inspecting `AgentInfo.output_tools` against a
    real specialist agent) - so satisfying `output_type=SomeModel` from a FunctionModel just
    means returning one `ToolCallPart(tool_name='final_result', args=<matching dict>)`.
    """

    def respond(_messages: list[ModelRequest], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(tool_name="final_result", args=sample.model_dump(mode="json"))])

    return FunctionModel(respond)


# ---------------------------------------------------------------------------
# Main agent mock: a tiny keyword-based router standing in for the LLM's judgment
# about which tool to call.
# ---------------------------------------------------------------------------

_LESSON_PLAN_KEYWORDS = ("lesson plan", "lesson-plan", "plan a lesson", "teach a lesson")
_QUIZ_KEYWORDS = ("quiz", "test question", "practice question")
_RUBRIC_KEYWORDS = ("rubric", "grading criteria", "grading rubric")
_RESOURCE_KEYWORDS = ("resource", "recommend", "further reading", "more to read", "practice material")


def _latest_user_text(messages: list[ModelRequest]) -> str:
    for message in reversed(messages):
        if not isinstance(message, ModelRequest):
            continue
        for part in reversed(message.parts):
            if isinstance(part, UserPromptPart):
                content = part.content
                return content if isinstance(content, str) else " ".join(str(item) for item in content)
    return ""


def _pending_tool_return(messages: list[ModelRequest]) -> ToolReturnPart | None:
    """If the most recent message is a tool's result, return it; else None (fresh turn)."""
    if not messages:
        return None
    last = messages[-1]
    if not isinstance(last, ModelRequest):
        return None
    for part in last.parts:
        if isinstance(part, ToolReturnPart):
            return part
    return None


def _classify_intent(user_text: str) -> tuple[str, dict]:
    """Decide which tool a real model would plausibly call for `user_text`."""
    lowered = user_text.lower()
    if any(kw in lowered for kw in _LESSON_PLAN_KEYWORDS):
        return "create_lesson_plan", {"topic": user_text, "grade_level": "8", "duration_minutes": 45}
    if any(kw in lowered for kw in _QUIZ_KEYWORDS):
        return "generate_quiz", {"topic": user_text, "num_questions": 5, "difficulty": "medium"}
    if any(kw in lowered for kw in _RUBRIC_KEYWORDS):
        return "create_rubric", {"assignment_description": user_text, "total_points": 100}
    if any(kw in lowered for kw in _RESOURCE_KEYWORDS):
        return "recommend_resources", {"topic": user_text}
    # Default: this is what the real base instructions ask for on any content question.
    return "search_course_materials", {"query": user_text}


def _main_agent_router(messages: list[ModelRequest], _info: AgentInfo) -> ModelResponse:
    pending = _pending_tool_return(messages)
    if pending is not None:
        content = pending.content
        text = content if isinstance(content, str) else str(content)
        return ModelResponse(
            parts=[
                TextPart(
                    content=(
                        f"{MOCK_BANNER} The router below picked `{pending.tool_name}` for your message. "
                        f"Everything after this line is that real tool's actual return value.\n\n{text}"
                    )
                )
            ]
        )

    tool_name, args = _classify_intent(_latest_user_text(messages))
    return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args)])


def make_main_agent_mock() -> FunctionModel:
    """Build the FunctionModel used for the main teaching_assistant_agent in mock mode.

    Real behavior this stands in for: the model reading BASE_INSTRUCTIONS, deciding whether
    a request needs course-material grounding or a specialist capability, and (for a real
    model) loading that capability first. This router skips simulating `load_capability`
    entirely - probing confirmed a FunctionModel-backed agent can call a deferred
    capability's tool directly, so the mock stays focused on demonstrating tool dispatch
    rather than replicating that handshake move-for-move.
    """
    return FunctionModel(_main_agent_router)


# ---------------------------------------------------------------------------
# Offline stand-in for pydantic_ai.Embedder, used by rag/store.py in mock mode.
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9']+")
_HASH_DIM = 128


def _hash_embed(text: str) -> list[float]:
    """A hashing-trick bag-of-words vector: crude, but texts sharing words score as more
    similar than unrelated texts, so RAG retrieval in mock mode behaves sensibly rather than
    returning arbitrary chunks."""
    vector = [0.0] * _HASH_DIM
    for word in _WORD_RE.findall(text.lower()):
        index = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % _HASH_DIM
        vector[index] += 1.0
    norm = sum(component * component for component in vector) ** 0.5
    if norm > 0:
        vector = [component / norm for component in vector]
    return vector


@dataclass
class _MockEmbeddingResult:
    embeddings: list[list[float]]


class MockEmbedder:
    """Drop-in replacement for `pydantic_ai.Embedder` covering the two methods
    `teaching_assistant.rag.store` actually calls. See module docstring for why."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def embed_documents_sync(self, documents: str | list[str], **_kwargs) -> _MockEmbeddingResult:
        texts = [documents] if isinstance(documents, str) else list(documents)
        return _MockEmbeddingResult([_hash_embed(text) for text in texts])

    def embed_query_sync(self, query: str | list[str], **_kwargs) -> _MockEmbeddingResult:
        texts = [query] if isinstance(query, str) else list(query)
        return _MockEmbeddingResult([_hash_embed(text) for text in texts])
