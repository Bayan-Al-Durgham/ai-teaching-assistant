"""Further-learning resource recommendations, as a deferred capability.

No live search is wired in by default, so recommendations come from the
model's own knowledge plus the student's course materials. That makes
fabricated URLs a real risk, so the sub-agent is explicitly told to prefer
citing by title/publisher over guessing a link. For production use, give
`resource_agent` a `WebSearch()` capability (see Native Tools reference) so
recommendations are grounded in a live search instead.
"""

from __future__ import annotations

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Capability

from teaching_assistant.config import DEFAULT_MODEL, MOCK_MODE
from teaching_assistant.deps import StudentContext
from teaching_assistant.formatting import format_resource_list
from teaching_assistant.models import ResourceList
from teaching_assistant.rag import format_chunks_for_prompt, search

if MOCK_MODE:
    from teaching_assistant.mock import SAMPLE_RESOURCE_LIST, make_specialist_mock

    _MODEL = make_specialist_mock(SAMPLE_RESOURCE_LIST)
else:
    _MODEL = DEFAULT_MODEL

resource_agent = Agent(
    _MODEL,
    name="resource_agent",
    output_type=ResourceList,
    instructions=(
        "You recommend further-learning resources on a topic, matched to the student's level. "
        "Do not invent a URL you are not highly confident is correct and stable. Prefer citing "
        "well-known, stable sources (major textbooks, Khan Academy, MDN, official documentation, "
        "Wikipedia) by title/publisher; only include a URL for a well-known top-level domain you "
        "are confident about, and otherwise leave `reference` as a precise citation and say the "
        "student should search for it by that title. Never present a resource as more specific "
        "or authoritative than you're actually sure of."
    ),
)

resource_recommendation_capability = Capability(
    id="resource_recommendation",
    description=(
        "Recommend further reading, videos, or practice resources on a topic, matched to the "
        "student's level. Load when asked for more resources, further reading, or practice material."
    ),
    instructions=(
        "Call recommend_resources with the topic and the student's current learner level, then "
        "present the list. Make clear which recommendations come from the course materials "
        "versus general knowledge, and don't present any link you're not confident is real."
    ),
    defer_loading=True,
)


@resource_recommendation_capability.tool
async def recommend_resources(ctx: RunContext[StudentContext], topic: str) -> str:
    """Recommend further-learning resources on `topic` for this student's level."""
    grounding = format_chunks_for_prompt(search(topic, course_id=ctx.deps.course_id, k=4))
    prompt = (
        f"Topic: {topic}\n"
        f"Student level: {ctx.deps.learner_level}\n\n"
        f"Relevant course material excerpts (cite these as course_material resources when used):\n{grounding}"
    )
    result = await resource_agent.run(prompt, usage=ctx.usage)
    return format_resource_list(result.output)
