"""The main teaching assistant agent.

Design:
- RAG search over course materials is eager (a plain tool) because answering
  from course content is the hot-path behavior, used on most turns.
- Lesson plans, quizzes, rubrics, and resource recommendations are each
  behind a deferred (`defer_loading=True`) capability, because any one
  student session probably only needs zero or one of them. The base prompt
  stays small; the model loads a capability's instructions and tools only
  when a request actually calls for it.
- Step-by-step explanation and learner-level adaptation are core behavior,
  not tools, so they live in instructions (static + dynamic) that apply on
  every turn.
"""

from __future__ import annotations

import os

from pydantic_ai import Agent, ModelMessage, RunContext
from pydantic_ai.capabilities import ProcessHistory

from teaching_assistant.capabilities import (
    lesson_planning_capability,
    quiz_generation_capability,
    resource_recommendation_capability,
    rubric_generation_capability,
)
from teaching_assistant.config import DEFAULT_MODEL, MOCK_MODE
from teaching_assistant.deps import LearnerLevel, StudentContext
from teaching_assistant.rag import format_chunks_for_prompt, search

MAX_HISTORY_MESSAGES = int(os.environ.get("TEACHING_ASSISTANT_MAX_HISTORY_MESSAGES", "40"))

if MOCK_MODE:
    from teaching_assistant.mock import make_main_agent_mock

    _MODEL = make_main_agent_mock()
else:
    _MODEL = DEFAULT_MODEL

BASE_INSTRUCTIONS = """
You are an AI Teaching Assistant. You help one specific student, in one specific course,
learn effectively, honestly, and independently.

Core behavior, on every turn:
- When explaining a concept, break it into clearly numbered steps that build from
  foundational ideas up to the full idea, unless the student explicitly asks for a short
  answer instead.
- For questions about course content, call search_course_materials before answering from
  general knowledge. If the materials don't cover it, say so and be clear you're answering
  from general knowledge instead.
- Do not simply do a student's graded work for them. Teach the reasoning and let them apply
  it, unless they are asking you to check or grade work they already completed themselves.
- Be encouraging but honest: if an answer is wrong, say so plainly and explain why.

You have four specialist capabilities. Load only the one a request actually calls for:
- lesson_planning: the student or teacher asks you to build a lesson plan.
- quiz_generation: asks for a quiz, test, or practice questions with an answer key.
- rubric_generation: asks for a grading rubric or assessment criteria.
- resource_recommendation: asks for further reading, videos, or practice resources on a topic.
Do not load a capability for a question you can already answer directly.
""".strip()


async def _trim_history(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Keep context bounded in long-running conversations."""
    if len(messages) > MAX_HISTORY_MESSAGES:
        return messages[-MAX_HISTORY_MESSAGES:]
    return messages


teaching_assistant_agent = Agent(
    _MODEL,
    name="teaching_assistant_agent",
    deps_type=StudentContext,
    instructions=BASE_INSTRUCTIONS,
    capabilities=[
        lesson_planning_capability,
        quiz_generation_capability,
        rubric_generation_capability,
        resource_recommendation_capability,
        ProcessHistory(_trim_history),
    ],
)


@teaching_assistant_agent.instructions
def adapt_to_learner_level(ctx: RunContext[StudentContext]) -> str:
    """Dynamic instruction: shapes every response to the student's level."""
    guidance: dict[LearnerLevel, str] = {
        "beginner": (
            "Use simple, everyday language and concrete analogies. Avoid jargon; when a "
            "technical term is unavoidable, define it in plain words the first time you use it."
        ),
        "intermediate": (
            "Use standard terminology for the subject. You can assume familiarity with "
            "foundational concepts, but still define less common terms on first use."
        ),
        "advanced": (
            "Use precise technical language. You can reference edge cases, nuances, and "
            "connections to related advanced topics without re-explaining the basics."
        ),
    }
    level = ctx.deps.learner_level
    return f"You are talking with {ctx.deps.student_name}, a {level}-level learner. {guidance[level]}"


@teaching_assistant_agent.tool
def search_course_materials(ctx: RunContext[StudentContext], query: str) -> str:
    """Search this student's uploaded course materials for content relevant to `query`.

    Always try this before answering a course-content question from general knowledge.
    """
    chunks = search(query, course_id=ctx.deps.course_id, k=4)
    return format_chunks_for_prompt(chunks)
