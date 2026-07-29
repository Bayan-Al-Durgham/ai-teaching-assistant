"""A minimal local web UI for the teaching assistant.

Run with `teaching-assistant-web` (or `python -m teaching_assistant.web`) and open
http://127.0.0.1:8000 in a browser. This entrypoint forces mock mode - no external AI API is
ever called, and no API key of any kind is required - see mock.py / ARCHITECTURE.md for what
that does and doesn't simulate. The page is a single self-contained HTML string with inline
CSS/JS (no CDN, no build step, no other files to serve), so nothing here needs network access
apart from the local browser <-> local server connection.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# Both must run before any `teaching_assistant.*` module is imported: config.py reads
# TEACHING_ASSISTANT_MOCK (and provider API keys, in real mode) exactly once, at import time.
# load_dotenv() first so an explicit choice in .env (e.g. TEACHING_ASSISTANT_MOCK=0 to use a
# real model here too) is honored; setdefault() after so this entrypoint still defaults to
# mock mode - no external AI API, no API key required - when nothing else has set it.
load_dotenv()
os.environ.setdefault("TEACHING_ASSISTANT_MOCK", "1")

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from teaching_assistant import memory
from teaching_assistant.agent import teaching_assistant_agent
from teaching_assistant.deps import LearnerLevel, StudentContext
from teaching_assistant.rag import has_materials, ingest_file

DEFAULT_COURSE = "bio101"
DEFAULT_SESSION = "web-session"
HOST = os.environ.get("TEACHING_ASSISTANT_WEB_HOST", "127.0.0.1")
PORT = int(os.environ.get("TEACHING_ASSISTANT_WEB_PORT", "8000"))

_SAMPLE_MATERIAL = (
    Path(__file__).resolve().parent.parent / "sample_course_materials" / "intro_to_photosynthesis.txt"
)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # First-run convenience: so the RAG demo works the moment the page loads, with nothing
    # else to run first.
    if not has_materials(DEFAULT_COURSE) and _SAMPLE_MATERIAL.exists():
        ingest_file(_SAMPLE_MATERIAL, course_id=DEFAULT_COURSE)
    yield


app = FastAPI(title="AI Teaching Assistant", lifespan=_lifespan)


class ChatRequest(BaseModel):
    message: str
    student_name: str = "Student"
    course_id: str = DEFAULT_COURSE
    learner_level: LearnerLevel = "beginner"
    session_id: str = DEFAULT_SESSION


class ChatResponse(BaseModel):
    reply: str


class ResetRequest(BaseModel):
    session_id: str = DEFAULT_SESSION


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    deps = StudentContext(
        student_name=req.student_name,
        course_id=req.course_id,
        learner_level=req.learner_level,
        conversation_id=req.session_id,
    )
    history = memory.load_history(req.session_id)
    result = teaching_assistant_agent.run_sync(req.message, deps=deps, message_history=history)
    memory.save_history(req.session_id, result.all_messages())
    return ChatResponse(reply=result.output)


@app.post("/api/reset")
def reset(req: ResetRequest) -> dict[str, bool]:
    memory.clear_history(req.session_id)
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE


def main() -> None:
    print(f"AI Teaching Assistant (mock mode) starting at http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Teaching Assistant (mock mode)</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: system-ui, -apple-system, Segoe UI, sans-serif;
    background: #14161c; color: #e7e9ee; height: 100vh; display: flex; flex-direction: column;
  }
  header {
    padding: 0.9rem 1.2rem; border-bottom: 1px solid #262a35; display: flex;
    align-items: center; gap: 0.75rem; flex-wrap: wrap;
  }
  header h1 { font-size: 1.05rem; margin: 0; font-weight: 600; }
  .badge {
    background: #3a2a12; color: #ffb74d; border: 1px solid #5a3f18; border-radius: 999px;
    padding: 0.15rem 0.65rem; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.02em;
  }
  .controls { margin-left: auto; display: flex; gap: 0.5rem; align-items: center; }
  .controls input, .controls select {
    background: #1c1f27; border: 1px solid #2c303c; color: #e7e9ee; border-radius: 6px;
    padding: 0.35rem 0.5rem; font-size: 0.85rem;
  }
  .controls input { width: 8rem; }
  main { flex: 1; overflow-y: auto; padding: 1rem 1.2rem; display: flex; flex-direction: column; gap: 0.75rem; }
  .msg { max-width: 70ch; padding: 0.6rem 0.85rem; border-radius: 10px; white-space: pre-wrap; line-height: 1.4; }
  .msg.user { align-self: flex-end; background: #2447c9; color: white; }
  .msg.assistant { align-self: flex-start; background: #1c1f27; border: 1px solid #2c303c; }
  .msg.assistant .mock-banner { color: #ffb74d; font-weight: 600; display: block; margin-bottom: 0.35rem; }
  .msg.system { align-self: center; color: #8a8f9c; font-size: 0.8rem; }
  form {
    display: flex; gap: 0.6rem; padding: 0.9rem 1.2rem; border-top: 1px solid #262a35;
  }
  form textarea {
    flex: 1; resize: none; background: #1c1f27; border: 1px solid #2c303c; color: #e7e9ee;
    border-radius: 8px; padding: 0.6rem 0.75rem; font: inherit; height: 3rem;
  }
  form button {
    background: #2447c9; color: white; border: none; border-radius: 8px; padding: 0 1.2rem;
    font-weight: 600; cursor: pointer;
  }
  form button:disabled { opacity: 0.5; cursor: default; }
  .hint { color: #8a8f9c; font-size: 0.78rem; padding: 0 1.2rem 0.6rem; }
</style>
</head>
<body>
<header>
  <h1>AI Teaching Assistant</h1>
  <span class="badge">MOCK MODE — no external AI API is called</span>
  <div class="controls">
    <input id="studentName" value="Alex" title="Student name">
    <select id="level" title="Learner level">
      <option value="beginner" selected>beginner</option>
      <option value="intermediate">intermediate</option>
      <option value="advanced">advanced</option>
    </select>
    <button id="resetBtn" type="button" title="Clear conversation history">Reset</button>
  </div>
</header>
<main id="log"></main>
<div class="hint">
  Try: "What does photosynthesis produce?" · "Make me a 5-question quiz on this" ·
  "Build a 45-minute lesson plan on this" · "Create a grading rubric for a lab report" ·
  "Recommend more resources on this topic"
</div>
<form id="chatForm">
  <textarea id="input" placeholder="Ask a question, or ask for a quiz / lesson plan / rubric / resources..."></textarea>
  <button id="sendBtn" type="submit">Send</button>
</form>
<script>
  const log = document.getElementById('log');
  const form = document.getElementById('chatForm');
  const input = document.getElementById('input');
  const sendBtn = document.getElementById('sendBtn');
  const resetBtn = document.getElementById('resetBtn');
  const studentName = document.getElementById('studentName');
  const level = document.getElementById('level');
  const sessionId = 'web-session-' + Math.random().toString(36).slice(2);

  function addMessage(role, text) {
    const div = document.createElement('div');
    div.className = 'msg ' + role;
    if (role === 'assistant' && text.startsWith('[MOCK MODE')) {
      const closeIdx = text.indexOf(']');
      const banner = document.createElement('span');
      banner.className = 'mock-banner';
      banner.textContent = text.slice(0, closeIdx + 1);
      div.appendChild(banner);
      div.appendChild(document.createTextNode(text.slice(closeIdx + 1).trim()));
    } else {
      div.textContent = text;
    }
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
    return div;
  }

  addMessage('system', 'Course materials on photosynthesis are pre-loaded for this demo. Ask away.');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = input.value.trim();
    if (!message) return;
    input.value = '';
    sendBtn.disabled = true;
    addMessage('user', message);
    const thinking = addMessage('assistant', 'Thinking...');
    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          student_name: studentName.value || 'Student',
          learner_level: level.value,
          session_id: sessionId,
        }),
      });
      if (!resp.ok) throw new Error('Request failed: ' + resp.status);
      const data = await resp.json();
      thinking.remove();
      addMessage('assistant', data.reply);
    } catch (err) {
      thinking.remove();
      addMessage('system', 'Error: ' + err.message);
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  });

  resetBtn.addEventListener('click', async () => {
    await fetch('/api/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId }),
    });
    log.innerHTML = '';
    addMessage('system', 'Conversation history cleared.');
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
