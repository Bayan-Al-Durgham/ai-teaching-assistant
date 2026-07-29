"""Runtime dependencies injected into the teaching assistant agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

LearnerLevel = Literal["beginner", "intermediate", "advanced"]


@dataclass
class StudentContext:
    """Per-run context: who's asking, at what level, and in which course.

    Passed as ``deps=`` on every ``agent.run(...)`` call. Tools and dynamic
    instructions read this via ``RunContext.deps`` to scope RAG search to the
    right course and to adapt explanations to the student's level.
    """

    student_name: str
    course_id: str
    learner_level: LearnerLevel = "intermediate"
    conversation_id: str = field(default="default")
