"""Render structured capability outputs as markdown for the chat conversation.

Keeping this separate from the capability modules means the validated
Pydantic objects (available via each sub-agent's `AgentRunResult.output`)
stay the source of truth for anything that needs the structured data — the
markdown here is just one presentation of it.
"""

from __future__ import annotations

from teaching_assistant.models import LessonPlan, Quiz, ResourceList, Rubric


def format_lesson_plan(plan: LessonPlan) -> str:
    lines = [
        f"# {plan.title}",
        f"**Subject:** {plan.subject} | **Grade:** {plan.grade_level} | **Duration:** {plan.duration_minutes} min",
        "",
        "## Objectives",
        *[f"- {o}" for o in plan.objectives],
        "",
        "## Materials",
        *[f"- {m}" for m in plan.materials],
        "",
        "## Warm-up",
        plan.warm_up,
        "",
        "## Activities",
        *[f"- **{a.name}** ({a.duration_minutes} min): {a.description}" for a in plan.activities],
        "",
        "## Assessment",
        plan.assessment,
        "",
        "## Closure",
        plan.closure,
    ]
    if plan.differentiation_notes:
        lines += ["", "## Differentiation", plan.differentiation_notes]
    return "\n".join(lines)


def format_quiz(quiz: Quiz) -> str:
    lines = [f"# {quiz.title}", f"**Topic:** {quiz.topic} | **Difficulty:** {quiz.difficulty}", ""]
    for i, q in enumerate(quiz.questions, start=1):
        lines.append(f"**{i}. {q.question}**")
        if q.options:
            lines.extend(f"   - {option}" for option in q.options)
        lines.append("")
    lines.append("## Answer key")
    for i, q in enumerate(quiz.questions, start=1):
        lines.append(f"{i}. **{q.correct_answer}** — {q.explanation}")
    return "\n".join(lines)


def format_rubric(rubric: Rubric) -> str:
    lines = [
        f"# {rubric.title}",
        rubric.assignment_description,
        f"**Total points:** {rubric.total_points}",
        "",
    ]
    for criterion in rubric.criteria:
        lines.append(f"## {criterion.name} ({criterion.weight_percent:.0f}%)")
        lines.extend(f"- **{lvl.label}** ({lvl.score_range}): {lvl.description}" for lvl in criterion.levels)
        lines.append("")
    return "\n".join(lines)


def format_resource_list(resources: ResourceList) -> str:
    lines = [f"# Recommended resources: {resources.topic}", ""]
    for r in resources.resources:
        lines.append(f"- **{r.title}** ({r.resource_type}, {r.level}) — {r.reference}")
        lines.append(f"  _{r.why_recommended}_")
    return "\n".join(lines)
