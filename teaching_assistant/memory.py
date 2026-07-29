"""Persist conversation history to disk so a session survives across runs.

One JSON file per conversation, under
``TEACHING_ASSISTANT_DATA_DIR/conversations/<conversation_id>.json``. Each
file holds the full `ModelMessagesTypeAdapter`-serialized message list, which
round-trips through `agent.run(..., message_history=...)` unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_ai import ModelMessage
from pydantic_ai.messages import ModelMessagesTypeAdapter


def _data_dir() -> Path:
    return Path(os.environ.get("TEACHING_ASSISTANT_DATA_DIR", "./data"))


def _conversation_path(conversation_id: str) -> Path:
    return _data_dir() / "conversations" / f"{conversation_id}.json"


def load_history(conversation_id: str) -> list[ModelMessage]:
    """Load prior turns for `conversation_id`, or an empty list for a new conversation."""
    path = _conversation_path(conversation_id)
    if not path.exists():
        return []
    return ModelMessagesTypeAdapter.validate_json(path.read_bytes())


def save_history(conversation_id: str, messages: list[ModelMessage]) -> None:
    """Persist the full message list for `conversation_id`, overwriting any prior save."""
    path = _conversation_path(conversation_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ModelMessagesTypeAdapter.dump_json(messages))


def clear_history(conversation_id: str) -> None:
    """Delete a conversation's saved history, e.g. for a "start over" command."""
    path = _conversation_path(conversation_id)
    path.unlink(missing_ok=True)
