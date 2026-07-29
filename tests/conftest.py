"""Test environment setup.

Runs at collection time (before any test module imports `teaching_assistant.agent`),
because Pydantic AI's Anthropic/OpenAI providers validate that an API key is *present*
as soon as an `Agent(...)` is constructed - even when the model is swapped out for
`TestModel` immediately afterward. These are obviously-fake placeholders; no test in
this suite makes a real network call.
"""

import os

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-placeholder")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")
os.environ.setdefault("TEACHING_ASSISTANT_DATA_DIR", ".pytest_data")
