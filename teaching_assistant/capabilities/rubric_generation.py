"""Assessment rubric generation, as a deferred capability."""

from __future__ import annotations

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Capability

from teaching_assistant.config import DEFAULT_MODEL, MOCK_MODE
from teaching_assistant.deps import StudentContext
from teaching_assistant.formatting import format_rubric
from teaching_assistant.models import Rubric
from teaching_assistant.rag import format_chunks_for_prompt, search

if MOCK_MODE:
    from teaching_assistant.mock import SAMPLE_RUBRIC, make_specialist_mock

    _MODEL = make_specialist_mock(SAMPLE_RUBRIC)
else:
    _MODEL = DEFAULT_MODEL

rubric_agent = Agent(
    _MODEL,
    name="rubric_agent",
    output_type=Rubric,
    instructions=(
        "You write assessment rubrics. Criteria weights must sum to 100. Each criterion needs "
        "at least three performance levels, ordered from highest to lowest, with score ranges "
        "that partition total_points without gaps or overlaps. Descriptions must be specific "
        "enough that two different graders would reach the same score."
    ),
)

rubric_generation_capability = Capability(
    id="rubric_generation",
    description=(
        "Generate a grading rubric with weighted criteria and performance-level descriptions "
        "for an assignment. Load when asked to create a rubric or grading criteria."
    ),
    instructions=(
        "Call create_rubric with the assignment description and total points the student or "
        "teacher specified (default to 100 points if unstated), then present the rubric. Offer "
        "to adjust criteria, weights, or the number of performance levels."
    ),
    defer_loading=True,
)


@rubric_generation_capability.tool
async def create_rubric(
    ctx: RunContext[StudentContext],
    assignment_description: str,
    total_points: int = 100,
) -> str:
    """Create a weighted grading rubric for the given assignment, worth `total_points` points."""
    grounding = format_chunks_for_prompt(search(assignment_description, course_id=ctx.deps.course_id, k=4))
    prompt = (
        f"Assignment: {assignment_description}\n"
        f"Total points: {total_points}\n\n"
        f"Relevant course material excerpts:\n{grounding}"
    )
    result = await rubric_agent.run(prompt, usage=ctx.usage)
    return format_rubric(result.output)
