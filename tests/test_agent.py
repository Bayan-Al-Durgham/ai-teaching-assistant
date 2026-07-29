from pydantic_ai.models.test import TestModel

from teaching_assistant.agent import teaching_assistant_agent
from teaching_assistant.capabilities import (
    lesson_planning_capability,
    quiz_generation_capability,
    resource_recommendation_capability,
    rubric_generation_capability,
)
from teaching_assistant.deps import StudentContext

ALL_CAPABILITIES = [
    lesson_planning_capability,
    quiz_generation_capability,
    rubric_generation_capability,
    resource_recommendation_capability,
]


def _deps(**overrides) -> StudentContext:
    defaults = dict(student_name="Alex", course_id="bio101", learner_level="intermediate")
    defaults.update(overrides)
    return StudentContext(**defaults)


def test_agent_has_a_stable_name():
    # Matters for telling agents apart in Logfire traces.
    assert teaching_assistant_agent.name == "teaching_assistant_agent"


def test_capability_ids_are_stable_and_deferred():
    ids = {c.id for c in ALL_CAPABILITIES}
    assert ids == {
        "lesson_planning",
        "quiz_generation",
        "rubric_generation",
        "resource_recommendation",
    }
    assert all(c.defer_loading for c in ALL_CAPABILITIES), (
        "Specialist capabilities should stay out of the base prompt until requested."
    )


def test_rag_tool_is_eager_and_specialist_tools_are_not():
    model = TestModel(call_tools=["search_course_materials"])
    with teaching_assistant_agent.override(model=model):
        teaching_assistant_agent.run_sync("What is photosynthesis?", deps=_deps())

    tool_names = {t.name for t in model.last_model_request_parameters.function_tools}

    # Hot-path tool: visible from turn one.
    assert "search_course_materials" in tool_names
    assert "load_capability" in tool_names

    # Specialist tools stay hidden until their capability is loaded.
    for hidden in ("create_lesson_plan", "generate_quiz", "create_rubric", "recommend_resources"):
        assert hidden not in tool_names


def test_instructions_adapt_to_learner_level_and_name():
    model = TestModel(call_tools=["search_course_materials"])

    with teaching_assistant_agent.override(model=model):
        beginner_result = teaching_assistant_agent.run_sync(
            "hi", deps=_deps(student_name="Priya", learner_level="beginner")
        )
    beginner_instructions = beginner_result.all_messages()[0].instructions
    assert "Priya" in beginner_instructions
    assert "everyday language" in beginner_instructions
    assert "precise technical language" not in beginner_instructions

    with teaching_assistant_agent.override(model=model):
        advanced_result = teaching_assistant_agent.run_sync(
            "hi", deps=_deps(student_name="Priya", learner_level="advanced")
        )
    advanced_instructions = advanced_result.all_messages()[0].instructions
    assert "precise technical language" in advanced_instructions
    assert "everyday language" not in advanced_instructions


def test_rag_tool_reports_no_materials_for_a_course_with_none_ingested():
    model = TestModel(call_tools=["search_course_materials"])
    with teaching_assistant_agent.override(model=model):
        result = teaching_assistant_agent.run_sync(
            "What is entropy?", deps=_deps(course_id="never-ingested-course-xyz")
        )
    assert "No matching course materials were found." in result.output
