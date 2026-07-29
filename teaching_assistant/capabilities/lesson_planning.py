"""Lesson-plan generation, as a deferred capability.

Loaded only when a request actually asks for a lesson plan. Delegates to a
specialist sub-agent with `output_type=LessonPlan` so the result is
validated structure, not free-form text — see the "Coordinate Multiple
Agents" pattern (agent-as-tool) in Pydantic AI's orchestration docs.
"""

from __future__ import annotations

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Capability

from teaching_assistant.config import DEFAULT_MODEL, MOCK_MODE
from teaching_assistant.deps import StudentContext
from teaching_assistant.formatting import format_lesson_plan
from teaching_assistant.models import LessonPlan
from teaching_assistant.rag import format_chunks_for_prompt, search

if MOCK_MODE:
    from teaching_assistant.mock import SAMPLE_LESSON_PLAN, make_specialist_mock

    _MODEL = make_specialist_mock(SAMPLE_LESSON_PLAN)
else:
    _MODEL = DEFAULT_MODEL

lesson_plan_agent = Agent(
    _MODEL,
    name="lesson_plan_agent",
    output_type=LessonPlan,
    instructions=(
        "You are an instructional designer. Write a complete, classroom-ready lesson plan "
        "from the request and any course-material excerpts given to you. Timings across all "
        "activities (plus the warm-up) must sum to the requested duration. Objectives must be "
        "specific and measurable. Ground content in the provided excerpts when they're relevant; "
        "otherwise rely on sound general pedagogy for the subject and grade level."
    ),
)

lesson_planning_capability = Capability(
    id="lesson_planning",
    description=(
        "Build a complete lesson plan (objectives, materials, activities, assessment) for a "
        "topic, grade level, and duration. Load when asked to create or draft a lesson plan."
    ),
    instructions=(
        "The student or teacher wants a lesson plan. Call create_lesson_plan with the topic, "
        "grade level, and duration they specified (ask only if truly ambiguous), then present "
        "the result. Offer to adjust pacing, add differentiation, or regenerate a section."
    ),
    defer_loading=True,
)


@lesson_planning_capability.tool
async def create_lesson_plan(
    ctx: RunContext[StudentContext],
    topic: str,
    grade_level: str,
    duration_minutes: int = 45,
) -> str:
    """Create a full lesson plan for `topic`, tailored to `grade_level` and `duration_minutes`."""
    grounding = format_chunks_for_prompt(search(topic, course_id=ctx.deps.course_id, k=4))
    prompt = (
        f"Topic: {topic}\n"
        f"Grade level: {grade_level}\n"
        f"Duration: {duration_minutes} minutes\n\n"
        f"Relevant course material excerpts:\n{grounding}"
    )
    result = await lesson_plan_agent.run(prompt, usage=ctx.usage)
    return format_lesson_plan(result.output)
