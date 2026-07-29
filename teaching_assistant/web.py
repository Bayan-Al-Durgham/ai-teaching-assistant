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
<script>
  (function () {
    var saved = null;
    try { saved = localStorage.getItem('ta-theme'); } catch (e) {}
    var theme = saved || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    document.documentElement.setAttribute('data-theme', theme);
  })();
</script>
<style>
  :root {
    --primary: #4f5fe8;
    --primary-hover: #4351d4;
    --primary-soft: rgba(79,95,232,0.12);
    --accent: #17b3a3;
    --accent-soft: rgba(23,179,163,0.14);
    --warn: #c9820f;
    --warn-bg: #fbf1de;
    --warn-border: #edd8a8;
    --danger: #e0475a;
    --danger-bg: #fbe9eb;
    --success: #1ea672;
    --success-bg: #e6f6ef;
    --sp-1: 4px; --sp-2: 8px; --sp-3: 12px; --sp-4: 16px; --sp-5: 24px; --sp-6: 32px; --sp-7: 48px;
    --radius-sm: 8px; --radius-md: 12px; --radius-lg: 18px; --radius-xl: 24px; --radius-full: 999px;
    --ease: cubic-bezier(.4,0,.2,1);
    --sidebar-w: 264px;
  }
  :root[data-theme="light"] {
    color-scheme: light;
    --bg: #f5f6fb;
    --surface: #ffffff;
    --surface-2: #f4f5fa;
    --surface-3: #eaecf5;
    --border: #e3e5f0;
    --text: #1a1c2b;
    --text-dim: #5c6079;
    --text-faint: #8d91a8;
    --shadow-sm: 0 1px 2px rgba(23,26,53,0.06);
    --shadow-md: 0 12px 28px rgba(23,26,53,0.09);
    --primary-contrast: #ffffff;
    --warn-bg: #fdf3e0; --warn-border: #f0dcae;
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --bg: #0e1016;
    --surface: #161922;
    --surface-2: #1b1f2a;
    --surface-3: #222634;
    --border: #2a2f3f;
    --text: #edeef5;
    --text-dim: #a3a8bd;
    --text-faint: #6d7288;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
    --shadow-md: 0 12px 28px rgba(0,0,0,0.4);
    --primary-contrast: #ffffff;
    --primary: #7c86f2;
    --primary-hover: #6870e8;
    --warn-bg: #352a13; --warn-border: #55431c; --warn: #f0b429;
    --danger-bg: #3a1b20;
    --success-bg: #103322;
  }
  * { box-sizing: border-box; }
  ::selection { background: var(--primary-soft); }
  html, body { height: 100%; }
  body {
    margin: 0; font-family: -apple-system, "Segoe UI", system-ui, Roboto, sans-serif;
    background: var(--bg); color: var(--text); overflow: hidden;
    -webkit-font-smoothing: antialiased; transition: background 200ms var(--ease), color 200ms var(--ease);
  }
  button, input, select, textarea { font-family: inherit; }
  a { color: inherit; }
  h1, h2, h3 { letter-spacing: -0.02em; }
  .icon { width: 18px; height: 18px; flex-shrink: 0; stroke: currentColor; fill: none; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }

  .app { display: flex; height: 100vh; }

  /* Sidebar */
  .sidebar {
    width: var(--sidebar-w); flex-shrink: 0; background: var(--surface);
    border-right: 1px solid var(--border); display: flex; flex-direction: column;
    padding: var(--sp-5) var(--sp-4); gap: var(--sp-5);
    transition: transform 240ms var(--ease), background 200ms var(--ease), border-color 200ms var(--ease);
  }
  .brand { display: flex; align-items: center; gap: var(--sp-3); padding: 0 var(--sp-2); }
  .brand .logo {
    width: 38px; height: 38px; border-radius: var(--radius-md); flex-shrink: 0;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    display: flex; align-items: center; justify-content: center; color: white;
    box-shadow: var(--shadow-sm);
  }
  .brand .name { font-weight: 700; font-size: 0.98rem; letter-spacing: -0.01em; line-height: 1.2; }
  .brand .sub { color: var(--text-faint); font-size: 0.74rem; margin-top: 1px; }

  .nav { display: flex; flex-direction: column; gap: var(--sp-1); }
  .nav-label { color: var(--text-faint); font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; padding: 0 var(--sp-3); margin: var(--sp-2) 0; }
  .nav-item {
    display: flex; align-items: center; gap: var(--sp-3); padding: var(--sp-3);
    border-radius: var(--radius-sm); color: var(--text-dim); cursor: pointer; border: none;
    background: transparent; font-size: 0.87rem; font-weight: 500; text-align: left; width: 100%;
    transition: background 150ms var(--ease), color 150ms var(--ease);
  }
  .nav-item:hover { background: var(--surface-2); color: var(--text); }
  .nav-item.active { background: var(--primary-soft); color: var(--primary); font-weight: 600; }
  .nav-item.active .icon { color: var(--primary); }

  .sidebar-footer { margin-top: auto; display: flex; flex-direction: column; gap: var(--sp-3); }
  .mode-pill {
    display: flex; align-items: center; gap: var(--sp-2); background: var(--warn-bg);
    border: 1px solid var(--warn-border); color: var(--warn); border-radius: var(--radius-md);
    padding: var(--sp-3); font-size: 0.72rem; font-weight: 600; line-height: 1.35;
  }
  .mode-pill .icon { width: 15px; height: 15px; flex-shrink: 0; }

  .theme-toggle {
    display: flex; align-items: center; justify-content: space-between; gap: var(--sp-2);
    background: var(--surface-2); border: 1px solid var(--border); border-radius: var(--radius-md);
    padding: var(--sp-2) var(--sp-3); cursor: pointer; color: var(--text-dim); font-size: 0.8rem; font-weight: 600;
    transition: background 150ms var(--ease), color 150ms var(--ease);
  }
  .theme-toggle:hover { background: var(--surface-3); color: var(--text); }
  .theme-toggle .icon-sun { display: none; }
  :root[data-theme="dark"] .theme-toggle .icon-moon { display: none; }
  :root[data-theme="dark"] .theme-toggle .icon-sun { display: block; }

  .sidebar-overlay {
    display: none; position: fixed; inset: 0; background: rgba(10,11,20,0.55); z-index: 30;
    opacity: 0; pointer-events: none; transition: opacity 200ms var(--ease);
  }

  /* Main column */
  .main { flex: 1; display: flex; flex-direction: column; min-width: 0; }

  .topbar {
    display: flex; align-items: center; gap: var(--sp-4); padding: var(--sp-4) var(--sp-6);
    border-bottom: 1px solid var(--border); flex-wrap: wrap; flex-shrink: 0;
    transition: border-color 200ms var(--ease);
  }
  .menu-btn {
    display: none; background: var(--surface-2); border: 1px solid var(--border); color: var(--text);
    border-radius: var(--radius-sm); width: 36px; height: 36px; align-items: center; justify-content: center;
    cursor: pointer; flex-shrink: 0;
  }
  .topbar h1 { font-size: 1rem; margin: 0; font-weight: 700; }
  .topbar .view-desc { color: var(--text-faint); font-size: 0.78rem; margin-top: 1px; }
  .title-group { display: flex; flex-direction: column; }
  .controls { margin-left: auto; display: flex; gap: var(--sp-2); align-items: center; flex-wrap: wrap; }
  .field {
    background: var(--surface-2); border: 1px solid var(--border); color: var(--text);
    border-radius: var(--radius-sm); padding: 0.45rem 0.65rem; font-size: 0.82rem;
    transition: border-color 150ms var(--ease), box-shadow 150ms var(--ease);
  }
  .field:focus-visible, button:focus-visible, .nav-item:focus-visible, .card:focus-visible {
    outline: none; box-shadow: 0 0 0 2px var(--bg), 0 0 0 4px var(--primary);
  }
  .field:focus { border-color: var(--primary); }
  .controls input.field { width: 8.5rem; }

  .btn {
    display: inline-flex; align-items: center; gap: var(--sp-2); border: none; border-radius: var(--radius-sm);
    padding: 0.5rem 0.95rem; font-size: 0.84rem; font-weight: 600; cursor: pointer;
    transition: transform 120ms var(--ease), background 150ms var(--ease), box-shadow 150ms var(--ease), opacity 150ms var(--ease);
  }
  .btn:active { transform: translateY(1px); }
  .btn-ghost { background: var(--surface-2); color: var(--text-dim); border: 1px solid var(--border); }
  .btn-ghost:hover { background: var(--surface-3); color: var(--text); }
  .btn-primary { background: var(--primary); color: var(--primary-contrast); box-shadow: var(--shadow-sm); }
  .btn-primary:hover { background: var(--primary-hover); }
  .btn-primary:disabled { opacity: 0.5; cursor: default; transform: none; }

  .content { flex: 1; overflow-y: auto; padding: var(--sp-6); }
  .view { display: none; max-width: 960px; margin: 0 auto; animation: fadeInUp 280ms var(--ease); }
  .view.active { display: block; }

  @keyframes fadeInUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
  @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
  @keyframes popIn { from { opacity: 0; transform: scale(0.94) translateY(4px); } to { opacity: 1; transform: scale(1) translateY(0); } }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Landing / home view */
  .landing-hero {
    text-align: center; padding: var(--sp-7) var(--sp-5) var(--sp-6);
  }
  .landing-hero .eyebrow {
    display: inline-flex; align-items: center; gap: var(--sp-2); background: var(--primary-soft); color: var(--primary);
    border-radius: var(--radius-full); padding: 0.3rem 0.85rem; font-size: 0.76rem; font-weight: 700;
    letter-spacing: 0.02em; margin-bottom: var(--sp-5);
  }
  .landing-hero h1 {
    font-size: 2.3rem; margin: 0 0 var(--sp-4); font-weight: 800; letter-spacing: -0.03em; line-height: 1.15;
  }
  .landing-hero h1 .accent-text {
    background: linear-gradient(135deg, var(--primary), var(--accent)); -webkit-background-clip: text;
    background-clip: text; color: transparent;
  }
  .landing-hero p {
    color: var(--text-dim); font-size: 1.02rem; line-height: 1.6; max-width: 56ch; margin: 0 auto var(--sp-6);
  }
  .landing-cta { display: flex; gap: var(--sp-3); justify-content: center; flex-wrap: wrap; }
  .btn-lg { padding: 0.75rem 1.4rem; font-size: 0.92rem; border-radius: var(--radius-md); }

  .feature-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: var(--sp-4); margin-top: var(--sp-7); }
  .feature-item {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg);
    padding: var(--sp-5); text-align: left; transition: transform 180ms var(--ease), box-shadow 180ms var(--ease);
  }
  .feature-item:hover { transform: translateY(-3px); box-shadow: var(--shadow-md); }
  .feature-item .card-icon {
    width: 40px; height: 40px; border-radius: var(--radius-md); background: var(--accent-soft); color: var(--accent);
    display: flex; align-items: center; justify-content: center; margin-bottom: var(--sp-3);
  }
  .feature-item .card-title { font-weight: 700; font-size: 0.92rem; margin-bottom: var(--sp-1); }
  .feature-item .card-sub { color: var(--text-faint); font-size: 0.8rem; line-height: 1.5; }

  .stat-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--sp-4); margin-top: var(--sp-6); }
  .stat-tile {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg);
    padding: var(--sp-4) var(--sp-5); display: flex; align-items: center; gap: var(--sp-3);
  }
  .stat-tile .stat-icon {
    width: 36px; height: 36px; border-radius: var(--radius-md); background: var(--primary-soft); color: var(--primary);
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
  }
  .stat-tile .stat-label { color: var(--text-faint); font-size: 0.74rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; }
  .stat-tile .stat-value { font-size: 0.92rem; font-weight: 700; }

  /* Hero / landing (chat welcome) */
  .hero { margin-bottom: var(--sp-6); overflow: hidden; transition: max-height 320ms var(--ease), opacity 220ms var(--ease), margin 320ms var(--ease); max-height: 600px; opacity: 1; }
  .hero.collapsed { max-height: 0; opacity: 0; margin-bottom: 0; }
  .hero h2 { font-size: 1.5rem; margin: 0 0 var(--sp-2); }
  .hero p { color: var(--text-dim); margin: 0 0 var(--sp-5); font-size: 0.92rem; line-height: 1.55; max-width: 62ch; }
  .card-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: var(--sp-3); }
  .card {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-md);
    padding: var(--sp-4); cursor: pointer; text-align: left; color: var(--text); font: inherit;
    transition: transform 160ms var(--ease), border-color 160ms var(--ease), background 160ms var(--ease), box-shadow 160ms var(--ease);
    display: flex; flex-direction: column; gap: var(--sp-2);
  }
  .card:hover { transform: translateY(-2px); border-color: var(--primary); box-shadow: var(--shadow-sm); }
  .card .card-icon {
    width: 32px; height: 32px; border-radius: var(--radius-sm); background: var(--primary-soft);
    display: flex; align-items: center; justify-content: center; color: var(--primary);
  }
  .card .card-title { font-weight: 600; font-size: 0.88rem; }
  .card .card-sub { color: var(--text-faint); font-size: 0.78rem; line-height: 1.4; }

  /* Chat - ChatGPT-style full width rows */
  .chat-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg);
    display: flex; flex-direction: column; box-shadow: var(--shadow-md); overflow: hidden;
    height: calc(100vh - 220px); min-height: 420px; transition: border-color 200ms var(--ease);
  }
  .hero.collapsed ~ .chat-card { height: calc(100vh - 160px); }
  main.log { flex: 1; overflow-y: auto; display: flex; flex-direction: column; padding: var(--sp-2) 0; }
  .msg-row { width: 100%; padding: var(--sp-4) var(--sp-5); animation: fadeInUp 220ms var(--ease); }
  .msg-row.assistant { background: var(--surface-2); }
  .msg-row.user { background: transparent; }
  .msg-row.system { padding: var(--sp-3) var(--sp-5); }
  .msg-inner { max-width: 720px; margin: 0 auto; display: flex; gap: var(--sp-3); align-items: flex-start; }
  .avatar {
    width: 30px; height: 30px; border-radius: var(--radius-md); flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 0.72rem; font-weight: 700;
    color: white; margin-top: 2px;
  }
  .msg-row.user .avatar { background: linear-gradient(135deg, var(--primary), #8f6bf0); }
  .msg-row.assistant .avatar { background: linear-gradient(135deg, var(--accent), #1791c9); }
  .msg-content { flex: 1; min-width: 0; white-space: pre-wrap; line-height: 1.65; font-size: 0.91rem; padding-top: 3px; }
  .empty-state {
    max-width: 460px; margin: var(--sp-7) auto; text-align: center; color: var(--text-faint);
    display: flex; flex-direction: column; align-items: center; gap: var(--sp-3); animation: fadeInUp 260ms var(--ease);
  }
  .empty-state .empty-icon {
    width: 52px; height: 52px; border-radius: var(--radius-full); background: var(--primary-soft); color: var(--primary);
    display: flex; align-items: center; justify-content: center;
  }
  .empty-state .empty-title { color: var(--text); font-weight: 700; font-size: 0.95rem; }
  .empty-state .empty-sub { font-size: 0.82rem; line-height: 1.5; }

  .error-note {
    max-width: 720px; margin: 0 auto; display: flex; gap: var(--sp-3); align-items: flex-start;
    background: var(--danger-bg); border: 1px solid var(--danger); color: var(--danger); border-radius: var(--radius-md);
    padding: var(--sp-3) var(--sp-4); font-size: 0.85rem; line-height: 1.5;
  }
  .error-note .icon { flex-shrink: 0; margin-top: 2px; }
  .error-note .error-detail { color: var(--text-faint); font-size: 0.76rem; margin-top: var(--sp-1); }

  .mock-banner { color: var(--warn); font-weight: 700; display: block; margin-bottom: 0.35rem; font-size: 0.78rem; }

  .typing-dots { display: inline-flex; gap: 4px; padding: 0.3rem 0; }
  .typing-dots span {
    width: 6px; height: 6px; border-radius: 50%; background: var(--text-faint);
    animation: typing 1.2s infinite ease-in-out;
  }
  .typing-dots span:nth-child(2) { animation-delay: 0.15s; }
  .typing-dots span:nth-child(3) { animation-delay: 0.3s; }
  @keyframes typing { 0%, 60%, 100% { opacity: 0.3; transform: translateY(0); } 30% { opacity: 1; transform: translateY(-3px); } }

  .composer { display: flex; gap: var(--sp-3); padding: var(--sp-4); border-top: 1px solid var(--border); background: var(--surface); flex-shrink: 0; }
  .composer textarea {
    flex: 1; resize: none; background: var(--surface-2); border: 1px solid var(--border); color: var(--text);
    border-radius: var(--radius-md); padding: 0.65rem 0.85rem; font: inherit; font-size: 0.89rem; height: 2.9rem;
    transition: border-color 150ms var(--ease);
  }
  .composer textarea:focus { outline: none; border-color: var(--primary); }
  .send-btn {
    background: var(--primary); color: white; border: none; border-radius: var(--radius-md); padding: 0 1.1rem;
    font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: var(--sp-2);
    transition: background 150ms var(--ease), transform 120ms var(--ease);
  }
  .send-btn:hover { background: var(--primary-hover); }
  .send-btn:disabled { opacity: 0.6; cursor: default; }
  .send-btn:active { transform: translateY(1px); }
  .spinner {
    width: 15px; height: 15px; border-radius: 50%; border: 2px solid rgba(255,255,255,0.4);
    border-top-color: white; animation: spin 0.7s linear infinite;
  }

  /* Static info views */
  .info-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg);
    padding: var(--sp-6); margin-bottom: var(--sp-4);
  }
  .info-card h3 { margin: 0 0 var(--sp-2); font-size: 1.05rem; }
  .info-card p { color: var(--text-dim); font-size: 0.88rem; line-height: 1.6; margin: 0 0 var(--sp-2); }
  .info-card .tag {
    display: inline-flex; align-items: center; gap: var(--sp-1); background: var(--surface-3);
    color: var(--text-dim); border-radius: var(--radius-full); padding: 0.2rem 0.7rem; font-size: 0.74rem;
    font-weight: 600; margin-right: var(--sp-2);
  }
  .cap-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: var(--sp-3); margin-top: var(--sp-4); }
  .cap-item { background: var(--surface-2); border: 1px solid var(--border); border-radius: var(--radius-md); padding: var(--sp-4); }
  .cap-item .card-title { font-weight: 600; font-size: 0.86rem; margin-bottom: var(--sp-1); }
  .cap-item .card-sub { color: var(--text-faint); font-size: 0.78rem; line-height: 1.45; }

  /* Toasts */
  .toast-stack {
    position: fixed; top: var(--sp-5); right: var(--sp-5); z-index: 60;
    display: flex; flex-direction: column; gap: var(--sp-2); max-width: 340px;
  }
  .toast {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-md);
    box-shadow: var(--shadow-md); padding: var(--sp-3) var(--sp-4); display: flex; gap: var(--sp-3);
    align-items: flex-start; font-size: 0.83rem; animation: popIn 200ms var(--ease);
  }
  .toast .icon { margin-top: 1px; }
  .toast.success { border-color: var(--success); color: var(--success); }
  .toast.error { border-color: var(--danger); color: var(--danger); }
  .toast.info { border-color: var(--primary); color: var(--primary); }
  .toast .toast-msg { color: var(--text); flex: 1; line-height: 1.4; }
  .toast.leaving { animation: fadeIn 180ms var(--ease) reverse forwards; }

  /* Scrollbars */
  .content::-webkit-scrollbar, main.log::-webkit-scrollbar { width: 8px; }
  .content::-webkit-scrollbar-thumb, main.log::-webkit-scrollbar-thumb { background: var(--surface-3); border-radius: var(--radius-full); }

  @media (max-width: 860px) {
    .sidebar {
      position: fixed; top: 0; left: 0; bottom: 0; z-index: 40; transform: translateX(-100%);
      box-shadow: var(--shadow-md);
    }
    .sidebar.open { transform: translateX(0); }
    .sidebar-overlay.open { display: block; opacity: 1; pointer-events: auto; }
    .menu-btn { display: flex; }
    .topbar { padding: var(--sp-3) var(--sp-4); }
    .content { padding: var(--sp-4); }
    .card-grid { grid-template-columns: 1fr; }
    .cap-grid { grid-template-columns: 1fr; }
    .feature-grid { grid-template-columns: repeat(2, 1fr); }
    .stat-row { grid-template-columns: 1fr; }
    .controls input.field { width: 6.5rem; }
    .chat-card { height: calc(100vh - 260px); }
    .hero.collapsed ~ .chat-card { height: calc(100vh - 190px); }
    .landing-hero h1 { font-size: 1.7rem; }
    .toast-stack { left: var(--sp-3); right: var(--sp-3); max-width: none; }
  }
</style>
</head>
<body>
<div class="toast-stack" id="toastStack"></div>
<div class="app">
  <div class="sidebar-overlay" id="sidebarOverlay"></div>
  <aside class="sidebar" id="sidebar">
    <div class="brand">
      <div class="logo">
        <svg viewBox="0 0 24 24" style="width:20px;height:20px;stroke:white;fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round">
          <path d="M4 6.5c2.5-1.3 5.5-1.3 8 0 2.5-1.3 5.5-1.3 8 0v11c-2.5-1.3-5.5-1.3-8 0-2.5-1.3-5.5-1.3-8 0v-11Z"/>
          <path d="M12 6.5v11"/>
        </svg>
      </div>
      <div>
        <div class="name">AI Teaching Assistant</div>
        <div class="sub">bio101 · demo workspace</div>
      </div>
    </div>

    <nav class="nav">
      <div class="nav-label">Overview</div>
      <button class="nav-item active" data-view="home" type="button">
        <svg class="icon" viewBox="0 0 24 24"><path d="m3 11 9-8 9 8"/><path d="M5 10v10h14V10"/></svg>
        Home
      </button>
      <div class="nav-label">Workspace</div>
      <button class="nav-item" data-view="chat" type="button">
        <svg class="icon" viewBox="0 0 24 24"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5Z"/></svg>
        Chat
      </button>
      <button class="nav-item" data-view="materials" type="button">
        <svg class="icon" viewBox="0 0 24 24"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2Z"/></svg>
        Course Materials
      </button>
      <button class="nav-item" data-view="about" type="button">
        <svg class="icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 16v-5"/><path d="M12 8h.01"/></svg>
        About This Demo
      </button>
    </nav>

    <div class="sidebar-footer">
      <button class="theme-toggle" id="themeToggle" type="button">
        <span>Appearance</span>
        <svg class="icon icon-moon" viewBox="0 0 24 24"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z"/></svg>
        <svg class="icon icon-sun" viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.9 4.9 1.4 1.4"/><path d="m17.7 17.7 1.4 1.4"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m4.9 19.1 1.4-1.4"/><path d="m17.7 6.3 1.4-1.4"/></svg>
      </button>
      <div class="mode-pill">
        <svg class="icon" viewBox="0 0 24 24"><path d="M12 9v4"/><path d="M12 17h.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/></svg>
        <span>MOCK MODE — no external AI API is called</span>
      </div>
    </div>
  </aside>

  <div class="main">
    <header class="topbar">
      <button class="menu-btn" id="menuBtn" type="button" aria-label="Toggle sidebar">
        <svg class="icon" viewBox="0 0 24 24"><path d="M4 6h16"/><path d="M4 12h16"/><path d="M4 18h16"/></svg>
      </button>
      <div class="title-group">
        <h1 id="viewTitle">Home</h1>
        <div class="view-desc" id="viewDesc">An overview of this workspace.</div>
      </div>
      <div class="controls">
        <input id="studentName" class="field" value="Alex" title="Student name">
        <select id="level" class="field" title="Learner level">
          <option value="beginner" selected>beginner</option>
          <option value="intermediate">intermediate</option>
          <option value="advanced">advanced</option>
        </select>
        <button id="resetBtn" class="btn btn-ghost" type="button" title="Clear conversation history">
          <svg class="icon" viewBox="0 0 24 24" style="width:15px;height:15px"><path d="M3 12a9 9 0 1 0 2.6-6.4"/><path d="M3 4v5h5"/></svg>
          Reset
        </button>
      </div>
    </header>

    <div class="content">
      <section class="view active" id="view-home">
        <div class="landing-hero">
          <span class="eyebrow">
            <svg class="icon" viewBox="0 0 24 24" style="width:14px;height:14px"><path d="M12 2 2 7l10 5 10-5-10-5Z"/><path d="M6 10.5v5c0 1.5 3 3 6 3s6-1.5 6-3v-5"/></svg>
            AI-powered teaching companion
          </span>
          <h1>Help every student learn,<br><span class="accent-text">one conversation at a time</span></h1>
          <p>
            This assistant answers course questions grounded in your materials, and builds lesson
            plans, quizzes, rubrics, and resource lists on request &mdash; adapting explanations to
            each learner's level. This demo runs fully offline in mock mode, so you can try the
            whole experience with no API key.
          </p>
          <div class="landing-cta">
            <button class="btn btn-primary btn-lg" type="button" data-goto="chat">
              <svg class="icon" viewBox="0 0 24 24" style="width:16px;height:16px"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5Z"/></svg>
              Start chatting
            </button>
            <button class="btn btn-ghost btn-lg" type="button" data-goto="about">Learn how it works</button>
          </div>

          <div class="stat-row">
            <div class="stat-tile">
              <div class="stat-icon"><svg class="icon" viewBox="0 0 24 24" style="width:17px;height:17px"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2Z"/></svg></div>
              <div>
                <div class="stat-label">Course</div>
                <div class="stat-value">bio101 &middot; Photosynthesis</div>
              </div>
            </div>
            <div class="stat-tile">
              <div class="stat-icon"><svg class="icon" viewBox="0 0 24 24" style="width:17px;height:17px"><path d="M12 9v4"/><path d="M12 17h.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/></svg></div>
              <div>
                <div class="stat-label">Mode</div>
                <div class="stat-value">Mock &middot; no API key needed</div>
              </div>
            </div>
            <div class="stat-tile">
              <div class="stat-icon"><svg class="icon" viewBox="0 0 24 24" style="width:17px;height:17px"><path d="m9 11 3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg></div>
              <div>
                <div class="stat-label">Capabilities</div>
                <div class="stat-value">4 specialist tools on demand</div>
              </div>
            </div>
          </div>

          <div class="feature-grid">
            <div class="feature-item">
              <div class="card-icon"><svg class="icon" viewBox="0 0 24 24" style="width:18px;height:18px"><rect x="3" y="4" width="18" height="17" rx="2"/><path d="M3 9h18"/><path d="M8 2v4"/><path d="M16 2v4"/></svg></div>
              <div class="card-title">Lesson planning</div>
              <div class="card-sub">Builds a structured lesson plan for any topic and duration.</div>
            </div>
            <div class="feature-item">
              <div class="card-icon"><svg class="icon" viewBox="0 0 24 24" style="width:18px;height:18px"><path d="M9 11.3 12 14l4.5-6"/><circle cx="12" cy="12" r="9"/></svg></div>
              <div class="card-title">Quiz generation</div>
              <div class="card-sub">Creates practice questions with a full answer key.</div>
            </div>
            <div class="feature-item">
              <div class="card-icon"><svg class="icon" viewBox="0 0 24 24" style="width:18px;height:18px"><path d="M9 12h6"/><path d="M9 16h6"/><rect x="4" y="3" width="16" height="18" rx="2"/></svg></div>
              <div class="card-title">Rubric generation</div>
              <div class="card-sub">Drafts clear grading criteria for any assignment.</div>
            </div>
            <div class="feature-item">
              <div class="card-icon"><svg class="icon" viewBox="0 0 24 24" style="width:18px;height:18px"><circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/></svg></div>
              <div class="card-title">Resource recommendations</div>
              <div class="card-sub">Suggests further reading and practice on a topic.</div>
            </div>
          </div>
        </div>
      </section>

      <section class="view" id="view-chat">
        <div class="hero" id="hero">
          <h2>Welcome back, <span id="heroName">Alex</span> 👋</h2>
          <p>Course materials on photosynthesis are pre-loaded for this demo. Ask a question, or try one of these:</p>
          <div class="card-grid">
            <button class="card suggestion" type="button" data-prompt="What does photosynthesis produce?">
              <div class="card-icon"><svg class="icon" viewBox="0 0 24 24" style="width:16px;height:16px"><circle cx="12" cy="12" r="9"/><path d="M12 8v4l3 3"/></svg></div>
              <div class="card-title">Ask a concept question</div>
              <div class="card-sub">"What does photosynthesis produce?"</div>
            </button>
            <button class="card suggestion" type="button" data-prompt="Make me a 5-question quiz on this">
              <div class="card-icon"><svg class="icon" viewBox="0 0 24 24" style="width:16px;height:16px"><path d="M9 11.3 12 14l4.5-6"/><circle cx="12" cy="12" r="9"/></svg></div>
              <div class="card-title">Generate a quiz</div>
              <div class="card-sub">"Make me a 5-question quiz on this"</div>
            </button>
            <button class="card suggestion" type="button" data-prompt="Build a 45-minute lesson plan on this">
              <div class="card-icon"><svg class="icon" viewBox="0 0 24 24" style="width:16px;height:16px"><rect x="3" y="4" width="18" height="17" rx="2"/><path d="M3 9h18"/><path d="M8 2v4"/><path d="M16 2v4"/></svg></div>
              <div class="card-title">Build a lesson plan</div>
              <div class="card-sub">"Build a 45-minute lesson plan on this"</div>
            </button>
            <button class="card suggestion" type="button" data-prompt="Create a grading rubric for a lab report">
              <div class="card-icon"><svg class="icon" viewBox="0 0 24 24" style="width:16px;height:16px"><path d="M9 12h6"/><path d="M9 16h6"/><rect x="4" y="3" width="16" height="18" rx="2"/></svg></div>
              <div class="card-title">Create a rubric</div>
              <div class="card-sub">"Create a grading rubric for a lab report"</div>
            </button>
          </div>
        </div>

        <div class="chat-card">
          <main class="log" id="log"></main>
          <form id="chatForm" class="composer">
            <textarea id="input" placeholder="Ask a question, or ask for a quiz / lesson plan / rubric / resources..."></textarea>
            <button id="sendBtn" class="send-btn" type="submit">
              <svg class="icon send-icon" viewBox="0 0 24 24" style="width:16px;height:16px;stroke:currentColor"><path d="m22 2-7 20-4-9-9-4 20-7Z"/><path d="M22 2 11 13"/></svg>
              <span class="send-label">Send</span>
            </button>
          </form>
        </div>
      </section>

      <section class="view" id="view-materials">
        <div class="info-card">
          <span class="tag">Course: bio101</span>
          <span class="tag">Auto-loaded on startup</span>
          <h3>Introduction to Photosynthesis</h3>
          <p>This demo pre-loads a short course document on photosynthesis so the retrieval-augmented
          question answering works immediately, with nothing to upload first. When you ask a
          course-content question in Chat, the assistant searches this material and grounds its
          answer in the retrieved passages before falling back to general knowledge.</p>
        </div>
      </section>

      <section class="view" id="view-about">
        <div class="info-card">
          <span class="tag">Mock Mode</span>
          <h3>No external AI API is called</h3>
          <p>Every response in this demo comes from a deterministic offline mock instead of a real
          model provider, so you can try the full assistant without needing an API key. Responses
          are prefixed with a "[MOCK MODE]" banner so it's always clear what you're looking at.</p>
        </div>
        <div class="info-card">
          <h3>What this assistant can do</h3>
          <p>Beyond answering course questions, it has four specialist capabilities it loads on demand:</p>
          <div class="cap-grid">
            <div class="cap-item">
              <div class="card-title">Lesson planning</div>
              <div class="card-sub">Builds a structured lesson plan for a given topic and duration.</div>
            </div>
            <div class="cap-item">
              <div class="card-title">Quiz generation</div>
              <div class="card-sub">Creates practice questions with an answer key.</div>
            </div>
            <div class="cap-item">
              <div class="card-title">Rubric generation</div>
              <div class="card-sub">Drafts grading criteria for an assignment.</div>
            </div>
            <div class="cap-item">
              <div class="card-title">Resource recommendation</div>
              <div class="card-sub">Suggests further reading or practice material on a topic.</div>
            </div>
          </div>
        </div>
      </section>
    </div>
  </div>
</div>

<script>
  const log = document.getElementById('log');
  const form = document.getElementById('chatForm');
  const input = document.getElementById('input');
  const sendBtn = document.getElementById('sendBtn');
  const sendIcon = sendBtn.querySelector('.send-icon');
  const sendLabel = sendBtn.querySelector('.send-label');
  const resetBtn = document.getElementById('resetBtn');
  const studentName = document.getElementById('studentName');
  const level = document.getElementById('level');
  const hero = document.getElementById('hero');
  const heroName = document.getElementById('heroName');
  const sessionId = 'web-session-' + Math.random().toString(36).slice(2);

  // ---- Toast notifications (additive UI polish only) ----
  const toastStack = document.getElementById('toastStack');
  const TOAST_ICONS = {
    success: '<svg class="icon" viewBox="0 0 24 24"><path d="M20 6 9 17l-5-5"/></svg>',
    error: '<svg class="icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 8v4"/><path d="M12 16h.01"/></svg>',
    info: '<svg class="icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 16v-5"/><path d="M12 8h.01"/></svg>',
  };
  function showToast(message, type) {
    type = type || 'info';
    const el = document.createElement('div');
    el.className = 'toast ' + type;
    el.innerHTML = (TOAST_ICONS[type] || TOAST_ICONS.info) + '<span class="toast-msg"></span>';
    el.querySelector('.toast-msg').textContent = message;
    toastStack.appendChild(el);
    setTimeout(() => {
      el.classList.add('leaving');
      setTimeout(() => el.remove(), 200);
    }, 3800);
  }

  function initials(name) {
    const trimmed = (name || '').trim();
    if (!trimmed) return '?';
    const parts = trimmed.split(/\\s+/);
    return (parts[0][0] + (parts[1] ? parts[1][0] : '')).toUpperCase();
  }

  function emptyStateNode() {
    const wrap = document.createElement('div');
    wrap.className = 'empty-state';
    wrap.id = 'emptyState';
    wrap.innerHTML =
      '<div class="empty-icon"><svg class="icon" viewBox="0 0 24 24" style="width:22px;height:22px">' +
      '<path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5Z"/></svg></div>' +
      '<div class="empty-title">Course materials on photosynthesis are pre-loaded for this demo</div>' +
      '<div class="empty-sub">Ask a question, or try one of the suggestions above to see what this assistant can do.</div>';
    return wrap;
  }

  function addMessage(role, text) {
    const emptyState = document.getElementById('emptyState');
    if (emptyState) emptyState.remove();

    if (role === 'system') {
      const row = document.createElement('div');
      row.className = 'msg-row system';
      const inner = document.createElement('div');
      inner.className = 'msg-inner';
      inner.style.justifyContent = 'center';
      const pill = document.createElement('div');
      pill.style.cssText = 'border:1px dashed var(--border); color: var(--text-faint); font-size: 0.78rem; text-align: center; padding: 0.4rem 0.9rem; border-radius: var(--radius-full);';
      pill.textContent = text;
      inner.appendChild(pill);
      row.appendChild(inner);
      log.appendChild(row);
      log.scrollTop = log.scrollHeight;
      return row;
    }

    const row = document.createElement('div');
    row.className = 'msg-row ' + role;
    const inner = document.createElement('div');
    inner.className = 'msg-inner';

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = role === 'user' ? initials(studentName.value) : 'AI';
    inner.appendChild(avatar);

    const content = document.createElement('div');
    content.className = 'msg-content';

    if (text === 'Thinking...') {
      content.innerHTML = '<span class="typing-dots"><span></span><span></span><span></span></span>';
    } else if (role === 'assistant' && text.startsWith('[MOCK MODE')) {
      const closeIdx = text.indexOf(']');
      const banner = document.createElement('span');
      banner.className = 'mock-banner';
      banner.textContent = text.slice(0, closeIdx + 1);
      content.appendChild(banner);
      content.appendChild(document.createTextNode(text.slice(closeIdx + 1).trim()));
    } else {
      content.textContent = text;
    }

    inner.appendChild(content);
    row.appendChild(inner);
    log.appendChild(row);
    log.scrollTop = log.scrollHeight;
    return row;
  }

  function addErrorNote(friendlyMessage, detail) {
    const row = document.createElement('div');
    row.className = 'msg-row system';
    const inner = document.createElement('div');
    inner.className = 'msg-inner';
    const note = document.createElement('div');
    note.className = 'error-note';
    note.innerHTML = '<svg class="icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 8v4"/><path d="M12 16h.01"/></svg><div></div>';
    const textWrap = note.querySelector('div');
    const title = document.createElement('div');
    title.textContent = friendlyMessage;
    textWrap.appendChild(title);
    if (detail) {
      const det = document.createElement('div');
      det.className = 'error-detail';
      det.textContent = detail;
      textWrap.appendChild(det);
    }
    inner.appendChild(note);
    row.appendChild(inner);
    log.appendChild(row);
    log.scrollTop = log.scrollHeight;
  }

  log.appendChild(emptyStateNode());

  function collapseHero() {
    hero.classList.add('collapsed');
  }

  studentName.addEventListener('input', () => {
    heroName.textContent = studentName.value || 'Student';
  });

  document.querySelectorAll('.suggestion').forEach((el) => {
    el.addEventListener('click', () => {
      input.value = el.dataset.prompt;
      input.focus();
    });
  });

  function setSending(sending) {
    sendBtn.disabled = sending;
    sendIcon.style.display = sending ? 'none' : '';
    sendLabel.textContent = sending ? '' : 'Send';
    if (sending) {
      const spin = document.createElement('span');
      spin.className = 'spinner';
      spin.id = 'sendSpinner';
      sendBtn.insertBefore(spin, sendLabel);
    } else {
      const spin = document.getElementById('sendSpinner');
      if (spin) spin.remove();
      sendLabel.textContent = 'Send';
    }
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = input.value.trim();
    if (!message) return;
    input.value = '';
    setSending(true);
    collapseHero();
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
      if (!resp.ok) throw new Error('Request failed with status ' + resp.status);
      const data = await resp.json();
      thinking.remove();
      addMessage('assistant', data.reply);
    } catch (err) {
      thinking.remove();
      addErrorNote('Something went wrong reaching the assistant. Please try again.', err.message);
      showToast('Message failed to send', 'error');
    } finally {
      setSending(false);
      input.focus();
    }
  });

  resetBtn.addEventListener('click', async () => {
    try {
      await fetch('/api/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      });
      log.innerHTML = '';
      log.appendChild(emptyStateNode());
      hero.classList.remove('collapsed');
      showToast('Conversation history cleared', 'success');
    } catch (err) {
      showToast('Could not clear conversation history', 'error');
    }
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  // ---- Theme toggle (light/dark, persisted; purely presentational) ----
  const themeToggle = document.getElementById('themeToggle');
  themeToggle.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('ta-theme', next); } catch (e) {}
  });

  // ---- Sidebar navigation (view switching only - no new backend calls) ----
  const sidebar = document.getElementById('sidebar');
  const sidebarOverlay = document.getElementById('sidebarOverlay');
  const menuBtn = document.getElementById('menuBtn');
  const viewTitle = document.getElementById('viewTitle');
  const viewDesc = document.getElementById('viewDesc');
  const viewMeta = {
    home: { title: 'Home', desc: 'An overview of this workspace.' },
    chat: { title: 'Chat', desc: 'Ask about the course material, or request a quiz, lesson plan, rubric, or resources.' },
    materials: { title: 'Course Materials', desc: 'What this session has pre-loaded for retrieval-augmented answers.' },
    about: { title: 'About This Demo', desc: 'How mock mode works and what the assistant can do.' },
  };

  function setSidebarOpen(open) {
    sidebar.classList.toggle('open', open);
    sidebarOverlay.classList.toggle('open', open);
  }

  function goToView(target) {
    document.querySelectorAll('.nav-item[data-view]').forEach((b) => b.classList.toggle('active', b.dataset.view === target));
    document.querySelectorAll('.view').forEach((v) => v.classList.toggle('active', v.id === 'view-' + target));
    viewTitle.textContent = viewMeta[target].title;
    viewDesc.textContent = viewMeta[target].desc;
    setSidebarOpen(false);
  }

  document.querySelectorAll('.nav-item[data-view]').forEach((btn) => {
    btn.addEventListener('click', () => goToView(btn.dataset.view));
  });
  document.querySelectorAll('[data-goto]').forEach((btn) => {
    btn.addEventListener('click', () => goToView(btn.dataset.goto));
  });

  menuBtn.addEventListener('click', () => setSidebarOpen(!sidebar.classList.contains('open')));
  sidebarOverlay.addEventListener('click', () => setSidebarOpen(false));
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
