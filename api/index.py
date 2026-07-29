import os

# Vercel's serverless filesystem is read-only except /tmp. Must be set before
# teaching_assistant.web is imported, since memory.py/rag/store.py read this env
# var at call time but default to a relative "./data" path under the read-only
# deployment bundle otherwise.
os.environ.setdefault("TEACHING_ASSISTANT_DATA_DIR", "/tmp/data")

from teaching_assistant.web import app

__all__ = ["app"]
