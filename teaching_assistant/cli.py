"""A minimal interactive CLI for the teaching assistant.

Usage:
    teaching-assistant ingest --course bio101 sample_course_materials/intro_to_photosynthesis.txt
    teaching-assistant chat --student Alex --course bio101 --level beginner --session alex-bio101

In a chat session:
    /level beginner|intermediate|advanced   change how explanations are pitched
    /ingest <path>                          add a course material file to the current course
    /reset                                  clear this session's remembered history
    /exit                                   leave
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

# Must run before any `teaching_assistant.*` import: agent.py reads provider API keys (real
# mode) or TEACHING_ASSISTANT_MOCK (mock mode) as soon as it's imported, which happens at
# module-import time below - well before a `load_dotenv()` call inside main() would ever run.
load_dotenv()

from teaching_assistant import memory  # noqa: E402
from teaching_assistant.agent import teaching_assistant_agent  # noqa: E402
from teaching_assistant.deps import StudentContext  # noqa: E402
from teaching_assistant.rag import ingest_file  # noqa: E402


def _cmd_ingest(args: argparse.Namespace) -> None:
    count = ingest_file(args.path, course_id=args.course)
    print(f"Ingested {count} chunk(s) from {args.path} into course '{args.course}'.")


def _cmd_chat(args: argparse.Namespace) -> None:
    deps = StudentContext(
        student_name=args.student,
        course_id=args.course,
        learner_level=args.level,
        conversation_id=args.session,
    )
    history = memory.load_history(args.session)
    print(
        f"Chatting as {deps.student_name} ({deps.learner_level}) in course '{deps.course_id}'. "
        f"Type /exit to quit.\n"
    )

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input in ("/exit", "/quit"):
            break

        if user_input == "/reset":
            memory.clear_history(args.session)
            history = []
            print("(session history cleared)")
            continue

        if user_input.startswith("/level "):
            new_level = user_input.removeprefix("/level ").strip()
            if new_level in ("beginner", "intermediate", "advanced"):
                deps.learner_level = new_level  # type: ignore[assignment]
                print(f"(learner level set to {new_level})")
            else:
                print("(usage: /level beginner|intermediate|advanced)")
            continue

        if user_input.startswith("/ingest "):
            path = user_input.removeprefix("/ingest ").strip()
            try:
                count = ingest_file(path, course_id=deps.course_id)
                print(f"(ingested {count} chunk(s) from {path})")
            except (OSError, ValueError) as exc:
                print(f"(couldn't ingest {path}: {exc})")
            continue

        result = teaching_assistant_agent.run_sync(user_input, deps=deps, message_history=history)
        print(f"\n{result.output}\n")

        history = result.all_messages()
        memory.save_history(args.session, history)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="teaching-assistant", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Add a course material file to the RAG index.")
    ingest_parser.add_argument("path", help="Path to a .txt or .md course material file.")
    ingest_parser.add_argument("--course", required=True, help="Course id to ingest into.")
    ingest_parser.set_defaults(func=_cmd_ingest)

    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session.")
    chat_parser.add_argument("--student", required=True, help="Student's display name.")
    chat_parser.add_argument("--course", required=True, help="Course id to scope RAG search to.")
    chat_parser.add_argument(
        "--level",
        choices=["beginner", "intermediate", "advanced"],
        default="intermediate",
        help="Starting learner level.",
    )
    chat_parser.add_argument(
        "--session",
        default=None,
        help="Conversation id to persist/resume history under (default: <student>-<course>).",
    )
    chat_parser.set_defaults(func=_cmd_chat)

    args = parser.parse_args(argv)
    if args.command == "chat" and args.session is None:
        args.session = f"{args.student}-{args.course}"

    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
