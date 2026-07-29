"""AI Teaching Assistant built with Pydantic AI.

`teaching_assistant_agent` and `StudentContext` are exposed lazily (PEP 562) rather than
imported eagerly here. Eagerly importing `teaching_assistant.agent` at package-import time
would construct every Agent (including the real-provider ones) as a side effect of importing
*any* submodule - e.g. `from teaching_assistant.web import app` would import this package
first, which would construct the real-model agents before `web.py`'s own module body (which
sets mock mode) ever ran. Deferring the import until the attribute is actually accessed avoids
that ordering trap entirely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from teaching_assistant.agent import teaching_assistant_agent as teaching_assistant_agent
    from teaching_assistant.deps import StudentContext as StudentContext

__all__ = ["teaching_assistant_agent", "StudentContext"]


def __getattr__(name: str):
    if name == "teaching_assistant_agent":
        from teaching_assistant.agent import teaching_assistant_agent

        return teaching_assistant_agent
    if name == "StudentContext":
        from teaching_assistant.deps import StudentContext

        return StudentContext
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
