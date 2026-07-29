# AI Teaching Assistant

A teaching assistant agent built with [Pydantic AI](https://ai.pydantic.dev). One base agent
handles conversation, adapts to the student's level, and answers from uploaded course
materials; four specialist capabilities (lesson plans, quizzes, rubrics, resource
recommendations) load on demand only when a request actually needs them.

**Studying the code rather than deploying it?** Set one environment variable and the whole
project runs with **zero API keys** — see [Run without any API keys](#run-without-any-api-keys-mock-mode)
below. For a full component-by-component explanation of how everything fits together, read
[ARCHITECTURE.md](ARCHITECTURE.md).

## Try it in your browser on Windows — no API keys, nothing external

This is the fastest path: a local web page, talking only to a local mock (see
[Run without any API keys](#run-without-any-api-keys-mock-mode) for exactly what that means) —
no external AI API of any kind is called.

Open **PowerShell** in the `ai-teaching-assistant` folder and run:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[web]"
teaching-assistant-web
```

You'll see:

```
AI Teaching Assistant (mock mode) starting at http://127.0.0.1:8000
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Open **http://127.0.0.1:8000** in your browser. Course material on photosynthesis is pre-loaded
automatically, so you can immediately try things like "What does photosynthesis produce?", "Make
me a quiz on this", "Build a lesson plan on this", "Create a grading rubric", or "Recommend more
resources" — each response is clearly labeled `MOCK MODE`, and the page never makes a network
call to any AI provider. Press `Ctrl+C` in the PowerShell window to stop the server.

If `python` isn't recognized, install Python 3.10+ from [python.org](https://www.python.org/downloads/)
first (check "Add python.exe to PATH" during install) and reopen PowerShell. `pip install -e ".[web]"`
installs every dependency this needs (Pydantic AI, FastAPI, Uvicorn, etc.) automatically — nothing
else to install by hand.

Prefer a terminal chat instead of a browser? See
[Run without any API keys](#run-without-any-api-keys-mock-mode) for the CLI equivalent.

## What it does

| Requirement | How it's implemented |
|---|---|
| Answer questions using uploaded course materials (RAG) | `search_course_materials` tool, eager — file-backed embedding index, cosine similarity search, scoped per course |
| Create lesson plans | `lesson_planning` capability → `LessonPlan` structured output |
| Generate quizzes with answer keys | `quiz_generation` capability → `Quiz` structured output |
| Produce rubrics | `rubric_generation` capability → `Rubric` structured output |
| Explain concepts step by step | Base instructions, on every turn |
| Adapt to learner level | Dynamic instruction reading `StudentContext.learner_level` (beginner / intermediate / advanced) |
| Remember previous conversations | Message history persisted to disk per conversation, reloaded and passed as `message_history=` on each run |
| Recommend additional resources | `resource_recommendation` capability → `ResourceList` structured output |

## Architecture

```
teaching_assistant/
├── agent.py                    # base agent: identity, step-by-step + level-adaptive
│                                # instructions, eager RAG tool, capability wiring
├── config.py                   # DEFAULT_MODEL (shared, avoids a circular import)
├── deps.py                     # StudentContext: student_name, course_id, learner_level
├── models.py                   # LessonPlan, Quiz, Rubric, ResourceList (+ nested models)
├── formatting.py                # render each structured model as markdown for chat
├── memory.py                   # save/load conversation history as JSON
├── cli.py                      # interactive demo: `ingest` and `chat` subcommands
├── web.py                      # local browser demo: FastAPI + one self-contained HTML page
├── mock.py                     # zero-API-key FunctionModel mocks, see below
├── rag/
│   ├── chunking.py             # word-based chunking with overlap
│   └── store.py                # embed + file-backed cosine-similarity index, per course_id
└── capabilities/
    ├── lesson_planning.py      # Capability(defer_loading=True) + specialist sub-agent
    ├── quiz_generation.py      # same pattern
    ├── rubric_generation.py    # same pattern
    └── resource_recommendation.py
```

For what each file does and why, and a full step-by-step trace of a request through the system,
see [ARCHITECTURE.md](ARCHITECTURE.md) — the rest of this section is just the two design
decisions worth knowing before you read the code.

**Why capabilities, not just tools.** Only `search_course_materials` is a plain eager tool,
because answering from course content is what happens on nearly every turn. Lesson plans,
quizzes, rubrics, and resource recommendations are each a
[`Capability(defer_loading=True)`](https://ai.pydantic.dev/capabilities/) — the model sees just
an `id` and one-line `description` for each until it decides one is actually needed, then calls
`load_capability` to pull in that bundle's instructions and tool. This keeps the base prompt
small instead of loading four specialist tool schemas and instruction sets on every single
message.

**Why capabilities call sub-agents ("agent-as-tool").** Each specialist tool
(`create_lesson_plan`, `generate_quiz`, `create_rubric`, `recommend_resources`) delegates to its
own `Agent(..., output_type=<Model>)` rather than trying to make the main conversational agent
return structured output directly. That keeps the main agent free to have an ordinary text
conversation while still getting schema-validated lesson plans/quizzes/rubrics/resources
whenever one is generated — and each specialist sub-agent gets instructions tuned to its own
task instead of sharing one generic prompt.

**RAG is intentionally simple.** `rag/store.py` embeds chunks with
[`Embedder`](https://ai.pydantic.dev/embeddings/) and stores them as JSON, searched with a
brute-force numpy cosine-similarity scan — fine for a course's worth of material, not meant to
scale past that. `ingest_text` and `search` are the only two functions the rest of the app
calls, so swapping in a real vector database (pgvector, Chroma, Pinecone, ...) means rewriting
this one file.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
cp .env.example .env             # then fill in API keys
```

By default the chat model is `anthropic:claude-sonnet-4-6` and the embedding model is
`openai:text-embedding-3-small` (Anthropic has no public embeddings endpoint, so RAG needs an
OpenAI key even if chat doesn't). Both are overridable via `.env` — any
[Pydantic AI model string](https://ai.pydantic.dev/models/) works for `TEACHING_ASSISTANT_MODEL`.

## Run without any API keys (mock mode)

```bash
export TEACHING_ASSISTANT_MOCK=1      # $env:TEACHING_ASSISTANT_MOCK=1 in PowerShell
teaching-assistant ingest --course bio101 sample_course_materials/intro_to_photosynthesis.txt
teaching-assistant chat --student Alex --course bio101 --level beginner
```

With `TEACHING_ASSISTANT_MOCK=1`, no `.env` file and no API keys are needed at all — every agent
is built with a [`FunctionModel`](https://ai.pydantic.dev/api/models/function/) instead of a real
provider, and RAG uses a deterministic offline embedder instead of a real embeddings API. Every
reply is prefixed `[MOCK MODE — no LLM was called]` so it's always obvious you're not talking to
a real model.

**This is not a stub that skips the architecture — it exercises all of it.** The router that picks
`search_course_materials` vs. `create_lesson_plan` vs. `generate_quiz` vs. `create_rubric` vs.
`recommend_resources` is simulated, and each specialist sub-agent always returns one canned (but
fully valid, schema-checked) example object — but every tool function, the real RAG index, the
`load_capability`/capability-loading mechanism, `formatting.py`, and conversation memory are the
exact same code that runs in real mode. It's the fastest way to see the whole system move without
spending on API calls. See [ARCHITECTURE.md § Run without any API keys](ARCHITECTURE.md#run-without-any-api-keys-mock-mode)
for exactly what's real vs. simulated at each step, and `mock.py` for the implementation.

## Usage

Ingest a course material file (`.txt` or `.md` — extract text from PDFs/slides first):

```bash
teaching-assistant ingest --course bio101 sample_course_materials/intro_to_photosynthesis.txt
```

Start a chat session (history persists to `data/conversations/<session>.json` and reloads
automatically next time you use the same `--session`):

```bash
teaching-assistant chat --student Alex --course bio101 --level beginner
```

In-session commands: `/level beginner|intermediate|advanced`, `/ingest <path>`, `/reset`, `/exit`.

Example:

```
> What does photosynthesis produce?
[searches course materials, answers step by step at the beginner level]

> Can you make me a 5-question quiz on this?
[loads the quiz_generation capability, returns a quiz + answer key grounded in the ingested text]

> Build me a 30-minute lesson plan on this for 7th graders
[loads lesson_planning, returns objectives/materials/activities/assessment summing to 30 min]
```

## Testing

```bash
pytest
```

All 32 tests run offline against `TestModel` / `FunctionModel` / a fake embedder — no API keys or
network calls required, ever, for any test. They check: capability catalog and `defer_loading`
flags, that the RAG tool is eager while specialist tools stay hidden until loaded, that
instructions actually change with learner level, chunking correctness, RAG
ingest/search/isolation-by-course, structured-model validation, markdown rendering,
conversation-memory round-tripping, and (`test_mock_mode.py`) that mock mode itself builds and
runs correctly with every provider API key unset.

## Known limitations / next steps

- **`web.py` is a local demo, not a deployable server.** It binds to `127.0.0.1` only, has no
  auth, keeps conversation state in a plain JSON file on disk, and defaults to mock mode (see
  `web.py`'s top-of-file comment for how `.env` can override that to use a real model instead).
  Fine for "run it on my own machine and look at it in a browser"; not fine to expose on a
  network as-is.
- **Mock mode's router is a keyword match, not a model.** `mock.py`'s `_classify_intent()` looks
  for words like "quiz" or "lesson plan" in the latest message — it won't understand an indirect
  or oddly-phrased request the way a real model would, and every specialist sub-agent returns the
  same canned example regardless of what you actually asked for. It's built for tracing
  architecture, not for evaluating output quality — use a real model for that.
- **No PDF/slide parsing.** Course materials must be `.txt`/`.md`; extract text first for other
  formats.
- **Resource recommendations aren't live-searched by default** — they come from the model's own
  knowledge plus your course materials, which risks a confidently wrong URL. The sub-agent is
  instructed to prefer citing by title/publisher over guessing a link, but for production use,
  add a [`WebSearch()`](https://ai.pydantic.dev/capabilities/web-search/) capability to
  `resource_agent` in `capabilities/resource_recommendation.py` so recommendations are grounded
  in a live search instead.
- **The vector store is single-machine, file-backed**, sized for "one course's worth of
  material," not a multi-tenant production corpus — see the RAG note above.
- **No authentication/authorization** — `course_id` scoping keeps RAG search from crossing
  courses, but nothing stops a caller from passing any `course_id`; add real access control
  before exposing this beyond a trusted single-user CLI.
- **Observability**: wire up [Logfire](https://ai.pydantic.dev/logfire/) with
  `logfire.instrument_pydantic_ai()` to trace runs, tool calls, and capability loads in
  production.
