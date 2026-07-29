"""Structured output models for the teaching assistant's specialist capabilities."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Lesson plans
# ---------------------------------------------------------------------------


class LessonActivity(BaseModel):
    name: str = Field(description="Short name of the activity, e.g. 'Think-pair-share'.")
    duration_minutes: int = Field(description="How many minutes this activity takes.")
    description: str = Field(description="What the teacher and students do during this activity.")


class LessonPlan(BaseModel):
    title: str
    subject: str
    grade_level: str
    duration_minutes: int = Field(description="Total lesson length in minutes.")
    objectives: list[str] = Field(description="Specific, measurable learning objectives.")
    materials: list[str] = Field(description="Everything needed to teach the lesson.")
    warm_up: str = Field(description="How the lesson opens, to activate prior knowledge.")
    activities: list[LessonActivity] = Field(description="The main sequence of the lesson.")
    assessment: str = Field(description="How the teacher will check understanding during or after the lesson.")
    closure: str = Field(description="How the lesson wraps up and connects back to the objectives.")
    differentiation_notes: str = Field(
        default="", description="Optional notes on adapting the lesson for different learner levels."
    )


# ---------------------------------------------------------------------------
# Quizzes
# ---------------------------------------------------------------------------

QuestionType = Literal["multiple_choice", "true_false", "short_answer"]


class QuizQuestion(BaseModel):
    question: str
    question_type: QuestionType
    options: list[str] | None = Field(
        default=None, description="Answer choices for multiple_choice or true_false questions; null for short_answer."
    )
    correct_answer: str = Field(description="The correct answer, exactly matching one of the options when present.")
    explanation: str = Field(description="Why this is the correct answer, for the answer key.")


class Quiz(BaseModel):
    title: str
    topic: str
    difficulty: Literal["easy", "medium", "hard"]
    questions: list[QuizQuestion]


# ---------------------------------------------------------------------------
# Rubrics
# ---------------------------------------------------------------------------


class RubricLevel(BaseModel):
    label: str = Field(description="e.g. 'Exemplary', 'Proficient', 'Developing', 'Beginning'.")
    score_range: str = Field(description="e.g. '9-10 points' or '90-100%'.")
    description: str = Field(description="What performance at this level looks like for this criterion.")


class RubricCriterion(BaseModel):
    name: str = Field(description="e.g. 'Thesis clarity', 'Use of evidence'.")
    weight_percent: float = Field(description="This criterion's share of the total score, 0-100.")
    levels: list[RubricLevel]


class Rubric(BaseModel):
    title: str
    assignment_description: str
    criteria: list[RubricCriterion]
    total_points: int = Field(description="Total possible points across all criteria.")


# ---------------------------------------------------------------------------
# Resource recommendations
# ---------------------------------------------------------------------------

ResourceType = Literal["article", "video", "interactive", "book", "practice_problems", "course_material"]


class ResourceRecommendation(BaseModel):
    title: str
    resource_type: ResourceType
    reference: str = Field(description="URL if known, otherwise a precise citation (author/title/publisher).")
    level: Literal["beginner", "intermediate", "advanced"]
    why_recommended: str = Field(description="One or two sentences on why this fits the student's need.")


class ResourceList(BaseModel):
    topic: str
    resources: list[ResourceRecommendation]
