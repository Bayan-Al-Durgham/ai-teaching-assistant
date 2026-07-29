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


def test_lesson_plan_round_trips_through_dump_and_validate():
    plan = LessonPlan(
        title="Intro to Photosynthesis",
        subject="Biology",
        grade_level="7",
        duration_minutes=45,
        objectives=["Explain the inputs and outputs of photosynthesis."],
        materials=["Whiteboard", "Diagram handout"],
        warm_up="Ask what plants need to grow.",
        activities=[
            LessonActivity(
                name="Diagram walkthrough",
                duration_minutes=15,
                description="Label the chloroplast diagram together.",
            )
        ],
        assessment="Exit ticket: write the word equation for photosynthesis.",
        closure="Recap the two stages.",
    )
    assert LessonPlan.model_validate(plan.model_dump()) == plan


def test_quiz_question_options_are_optional_for_short_answer():
    question = QuizQuestion(
        question="What pigment absorbs light in photosynthesis?",
        question_type="short_answer",
        options=None,
        correct_answer="Chlorophyll",
        explanation="Chlorophyll is the main light-absorbing pigment in chloroplasts.",
    )
    quiz = Quiz(title="Photosynthesis Quiz", topic="Photosynthesis", difficulty="easy", questions=[question])
    assert quiz.questions[0].options is None


def test_rubric_criteria_and_levels_construct_cleanly():
    rubric = Rubric(
        title="Lab Report Rubric",
        assignment_description="A short lab report on the photosynthesis rate experiment.",
        total_points=20,
        criteria=[
            RubricCriterion(
                name="Data analysis",
                weight_percent=50,
                levels=[
                    RubricLevel(label="Proficient", score_range="8-10", description="Correctly interprets trends."),
                    RubricLevel(label="Developing", score_range="0-7", description="Misses key trends."),
                ],
            )
        ],
    )
    assert rubric.criteria[0].weight_percent == 50


def test_resource_list_holds_multiple_recommendations():
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
    assert len(resources.resources) == 1
