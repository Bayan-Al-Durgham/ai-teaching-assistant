"""Shared configuration. Kept separate from agent.py so specialist capability
modules can read the default model without importing agent.py and creating a
circular import (agent.py imports the capabilities package)."""

from __future__ import annotations

import os

DEFAULT_MODEL = os.environ.get("TEACHING_ASSISTANT_MODEL", "anthropic:claude-sonnet-4-6")

# When set, every Agent in this project is built with a FunctionModel-based mock instead of
# a real provider (see mock.py), and RAG uses a mock embedder instead of a real embeddings
# API. No API key of any kind is needed in this mode. Construction, not just running, must
# branch on this flag: Agent('provider:model', ...) validates that a provider API key is
# *present* as soon as it's constructed, even if you never end up calling it - passing a
# FunctionModel instance instead skips that check entirely, but only if it's passed to
# Agent(...) directly rather than swapped in afterward with `.override()`.
MOCK_MODE = os.environ.get("TEACHING_ASSISTANT_MOCK", "").strip().lower() in ("1", "true", "yes", "on")
