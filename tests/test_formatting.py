from teaching_assistant.formatting import (
    format_lesson_plan,
    format_quiz,
    format_resource_list,
    format_rubric,
)
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


def test_format_lesson_plan_includes_all_sections():
    plan = LessonPlan(
        title="Intro to Photosynthesis",
        subject="Biology",
        grade_level="7",
        duration_minutes=45,
        objectives=["Explain inputs and outputs of photosynthesis."],
        materials=["Whiteboard"],
        warm_up="Ask what plants need to grow.",
        activities=[LessonActivity(name="Diagram walkthrough", duration_minutes=15, description="Label the diagram.")],
        assessment="Exit ticket.",
        closure="Recap.",
        differentiation_notes="Provide sentence starters for beginners.",
    )
    rendered = format_lesson_plan(plan)
    for expected in [
        "Intro to Photosynthesis",
        "Explain inputs and outputs",
        "Whiteboard",
        "Diagram walkthrough",
        "Exit ticket.",
        "Recap.",
        "sentence starters",
    ]:
        assert expected in rendered


def test_format_quiz_includes_answer_key():
    quiz = Quiz(
        title="Photosynthesis Quiz",
        topic="Photosynthesis",
        difficulty="easy",
        questions=[
            QuizQuestion(
                question="Where does photosynthesis mainly occur?",
                question_type="multiple_choice",
                options=["Roots", "Leaves", "Stem"],
                correct_answer="Leaves",
                explanation="Leaves contain the most chloroplasts.",
            )
        ],
    )
    rendered = format_quiz(quiz)
    assert "Where does photosynthesis mainly occur?" in rendered
    assert "Leaves" in rendered
    assert "Answer key" in rendered
    assert "chloroplasts" in rendered


def test_format_rubric_includes_all_criteria_and_levels():
    rubric = Rubric(
        title="Lab Report Rubric",
        assignment_description="Photosynthesis rate lab report.",
        total_points=20,
        criteria=[
            RubricCriterion(
                name="Data analysis",
                weight_percent=50,
                levels=[RubricLevel(label="Proficient", score_range="8-10", description="Interprets trends correctly.")],
            )
        ],
    )
    rendered = format_rubric(rubric)
    assert "Data analysis" in rendered
    assert "Proficient" in rendered
    assert "Interprets trends correctly." in rendered


def test_format_resource_list_includes_why_recommended():
    resources = ResourceList(
        topic="Photosynthesis",
        resources=[
            ResourceRecommendation(
                title="Photosynthesis overview",
                resource_type="article",
                reference="Course materials: intro_to_photosynthesis.txt",
                level="beginner",
                why_recommended="Matches the vocabulary already used in class.",
            )
        ],
    )
    rendered = format_resource_list(resources)
    assert "Photosynthesis overview" in rendered
    assert "Matches the vocabulary already used in class." in rendered
