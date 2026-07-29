"""Quiz generation with an answer key, as a deferred capability."""

from __future__ import annotations

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Capability

from teaching_assistant.config import DEFAULT_MODEL, MOCK_MODE
from teaching_assistant.deps import StudentContext
from teaching_assistant.formatting import format_quiz
from teaching_assistant.models import Quiz
from teaching_assistant.rag import format_chunks_for_prompt, search

if MOCK_MODE:
    from teaching_assistant.mock import SAMPLE_QUIZ, make_specialist_mock

    _MODEL = make_specialist_mock(SAMPLE_QUIZ)
else:
    _MODEL = DEFAULT_MODEL

quiz_agent = Agent(
    _MODEL,
    name="quiz_agent",
    output_type=Quiz,
    instructions=(
        "You write quizzes for students. Every question needs a correct_answer and an "
        "explanation suitable for an answer key. For multiple_choice, correct_answer must "
        "exactly match one of the options. Vary question types unless the request asks for a "
        "single type. Ground questions in the provided course-material excerpts when relevant; "
        "otherwise write standard questions for the stated topic and difficulty."
    ),
)

quiz_generation_capability = Capability(
    id="quiz_generation",
    description=(
        "Generate a quiz with an answer key for a topic, question count, and difficulty. Load "
        "when asked to create a quiz, test, or practice questions."
    ),
    instructions=(
        "Call generate_quiz with the topic, question count, and difficulty the student or "
        "teacher specified (default to 5 medium-difficulty questions if unstated), then present "
        "the quiz followed by the answer key. Offer to change the count, difficulty, or format."
    ),
    defer_loading=True,
)


@quiz_generation_capability.tool
async def generate_quiz(
    ctx: RunContext[StudentContext],
    topic: str,
    num_questions: int = 5,
    difficulty: str = "medium",
) -> str:
    """Generate a `num_questions`-question quiz on `topic` at the given `difficulty`, with an answer key."""
    grounding = format_chunks_for_prompt(search(topic, course_id=ctx.deps.course_id, k=4))
    prompt = (
        f"Topic: {topic}\n"
        f"Number of questions: {num_questions}\n"
        f"Difficulty: {difficulty}\n\n"
        f"Relevant course material excerpts:\n{grounding}"
    )
    result = await quiz_agent.run(prompt, usage=ctx.usage)
    return format_quiz(result.output)
