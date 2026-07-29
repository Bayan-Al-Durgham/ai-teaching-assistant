# Architecture: a component-by-component walkthrough

This document is for reading the code, not just running it. It explains what every file does,
why it exists, and how a request actually flows through the system — including exactly what's
real and what's simulated when you run with `TEACHING_ASSISTANT_MOCK=1` (see
[Run without any API keys](#run-without-any-api-keys-mock-mode) at the end).

## Mental model

One conversational agent (`teaching_assistant_agent`) talks to the student. It can do two kinds
of things:

1. **Answer directly**, using an eager tool to search the student's course materials (RAG).
2. **Delegate to a specialist**, one of four capabilities that each wrap their own sub-agent and
   return validated, structured data (a lesson plan, a quiz, a rubric, or a resource list).

```
                         ┌─────────────────────────────┐
   student message  ───► │   teaching_assistant_agent   │ ───► reply
                         │  (agent.py)                  │
                         └──────────────┬────────────────┘
                                         │
              always available ─────────┤────────── loaded on demand
                                         │
                    ┌────────────────────┴──────────────────────┐
                    │                                            │
        search_course_materials                    lesson_planning / quiz_generation /
           (agent.py, eager tool)                   rubric_generation / resource_recommendation
                    │                                (capabilities/*.py, defer_loading=True)
                    ▼                                            │
            rag/store.py: search()                               ▼
         (embed query, cosine similarity                 each calls its own specialist
          over the course's chunk index)                 Agent(output_type=<Model>)
                                                           and renders it with formatting.py
```

Every turn also passes through `deps.py` (who's asking, at what level, in which course) and
`memory.py` (so the conversation survives across separate runs).

## File-by-file

### `deps.py` — what the agent knows about who it's talking to

```python
@dataclass
class StudentContext:
    student_name: str
    course_id: str
    learner_level: Literal["beginner", "intermediate", "advanced"]
    conversation_id: str
```

This is Pydantic AI's dependency-injection mechanism: pass `deps=StudentContext(...)` to
`agent.run(...)`, and any tool or instruction function that takes a `RunContext[StudentContext]`
can read `ctx.deps`. Two things read it directly: `search_course_materials` (to scope RAG to the
right `course_id`) and `adapt_to_learner_level` (to read `learner_level`).

### `config.py` — the one place that decides "real or mock"

Two module-level values, read from environment variables at import time:

- `DEFAULT_MODEL` — the real model string (`anthropic:claude-sonnet-4-6` by default).
- `MOCK_MODE` — `True` if `TEACHING_ASSISTANT_MOCK` is set to a truthy value.

It's a separate file from `agent.py` for a mechanical reason: every specialist capability module
needs `DEFAULT_MODEL`/`MOCK_MODE`, and `agent.py` imports the capabilities package — if the
capabilities imported from `agent.py` instead, that would be a circular import.

### `agent.py` — the main agent

Three things happen here, all layered onto one `Agent`:

1. **Static instructions** (`BASE_INSTRUCTIONS`) — the persona, the "explain step by step" rule,
   the "don't just do the student's graded work" rule, and a routing hint listing the four
   capabilities so the model knows when to reach for one.
2. **A dynamic instruction** (`adapt_to_learner_level`) — a function decorated with
   `@teaching_assistant_agent.instructions` that runs on every turn and returns different
   guidance text depending on `ctx.deps.learner_level`. This is how "adapt explanations to
   different learner levels" is implemented: not a tool, just instructions that change based on
   who's asking.
3. **One eager tool** (`search_course_materials`) — decorated with `@teaching_assistant_agent.tool`
   (needs `RunContext` for `ctx.deps.course_id`). It's a plain tool, not a capability, because
   answering from course content is the hot path — used on most turns — so its schema and
   instructions should always be in the prompt, not loaded on demand.

The agent is also given `capabilities=[...]` — the four specialist bundles (see below) plus
`ProcessHistory(_trim_history)`, which trims the message list to the last `N` messages before
each model request so a long-running conversation doesn't grow its context unboundedly.

### `capabilities/*.py` — specialist work, loaded on demand

Each of the four files follows the same shape. Take `lesson_planning.py` apart:

```python
lesson_plan_agent = Agent(DEFAULT_MODEL, name="lesson_plan_agent", output_type=LessonPlan, ...)

lesson_planning_capability = Capability(
    id="lesson_planning",
    description="Build a complete lesson plan ...",   # what the model sees before loading
    instructions="...",                                 # what the model sees after loading
    defer_loading=True,
)

@lesson_planning_capability.tool
async def create_lesson_plan(ctx, topic, grade_level, duration_minutes=45) -> str:
    grounding = format_chunks_for_prompt(search(topic, course_id=ctx.deps.course_id, k=4))
    prompt = f"Topic: {topic}\n...\n{grounding}"
    result = await lesson_plan_agent.run(prompt, usage=ctx.usage)
    return format_lesson_plan(result.output)
```

Two design decisions worth calling out:

**Why `defer_loading=True`.** Without it, all four capabilities' instructions and tool schemas
would be in the prompt on every single turn, even for a student just asking a question. With it,
the main agent initially sees only each capability's `id` and one-line `description`, plus a
framework-provided `load_capability` tool. Only when the model decides a request actually needs
one does it call `load_capability("lesson_planning")`, at which point that capability's
instructions and `create_lesson_plan` tool become visible for the rest of the run. This is
Pydantic AI's "capabilities on demand" progressive-disclosure mechanism — see
`.claude/skills/building-pydantic-ai-agents/references/ON-DEMAND-CAPABILITIES.md` for the
framework docs this project follows.

**Why a whole separate `Agent`, not just a tool that returns text.** `create_lesson_plan` doesn't
generate the lesson plan itself — it builds a grounding prompt (pulling relevant course-material
chunks via the same `rag/store.py:search()` the main agent's own tool uses) and hands it to
`lesson_plan_agent`, a *second* agent whose `output_type=LessonPlan` forces a schema-validated
response (see `models.py`). This is Pydantic AI's "agent delegation" / agent-as-tool pattern:
`await lesson_plan_agent.run(prompt, usage=ctx.usage)` — passing `usage=ctx.usage` so token usage
from the delegated call rolls up into the parent run's totals. The result is guaranteed to be a
real `LessonPlan` object (right field types, required fields present) before
`format_lesson_plan()` ever touches it — not "hopefully valid-looking text."

`quiz_generation.py`, `rubric_generation.py`, and `resource_recommendation.py` are the same
pattern with a different `output_type` (`Quiz`, `Rubric`, `ResourceList`) and a prompt tailored to
that task. `resource_recommendation.py` has one extra wrinkle: its sub-agent's instructions
explicitly forbid inventing URLs it isn't confident about, because — unlike the other three —
recommending resources risks citing a source that doesn't exist; see the comment at the top of
that file for the production fix (add a `WebSearch()` capability to ground it in a live search).

### `models.py` — the structured-output contracts

Plain Pydantic `BaseModel` classes: `LessonPlan` (+ `LessonActivity`), `Quiz` (+ `QuizQuestion`),
`Rubric` (+ `RubricCriterion`, `RubricLevel`), `ResourceList` (+ `ResourceRecommendation`). Each
field has a `Field(description=...)` where the meaning isn't obvious from the name alone — that
description becomes part of the JSON schema the model sees when generating that field, so it
doubles as documentation for both you and the LLM.

### `formatting.py` — turning validated data into a chat-readable reply

Four `format_*` functions, one per model, each walking the Pydantic object's fields into a
markdown string (headers, bullet lists, an answer key section for quizzes, etc.). This is
deliberately a separate module from the capability files: the *validated data* is the source of
truth (available as `result.output` inside each capability's tool function), and markdown is just
one way to present it — a future JSON API endpoint could reuse the same models and skip this file
entirely.

### `rag/` — retrieval over uploaded course materials

Three small files with a narrow, swappable interface:

- **`chunking.py`** — `chunk_text(text, chunk_size=800, overlap=100)`. Plain word-based splitting
  with overlap between consecutive chunks, so a fact sitting right at a chunk boundary is still
  likely to appear whole in at least one chunk. No sentence- or heading-awareness — intentionally
  simple.
- **`store.py`** — the actual index. `ingest_text(text, course_id, source)` chunks the text,
  embeds every chunk with `pydantic_ai.Embedder(...)`, and appends `{id, source, text, embedding}`
  records to a JSON file at `data/rag_index/<course_id>.json`. `search(query, course_id, k)` embeds
  the query, loads that course's JSON file, and does a brute-force numpy cosine-similarity scan to
  return the top `k` chunks. `has_materials(course_id)` is a cheap existence check the CLI/agent
  can use. **`ingest_text` and `search` are the entire surface the rest of the app depends on** —
  swapping this file for a real vector database means changing nothing else.
- **`ingest.py`** — `ingest_file(path, course_id)` reads a `.txt`/`.md` file and calls
  `ingest_text`; `format_chunks_for_prompt(chunks)` renders retrieved chunks as a numbered,
  source-labeled block of text for inclusion in a prompt (used by the main agent's tool *and*
  every capability's grounding step).

### `memory.py` — remembering previous conversations

Three functions, one JSON file per conversation, under
`data/conversations/<conversation_id>.json`:

- `save_history(conversation_id, messages)` — serializes the full message list with
  `pydantic_ai.messages.ModelMessagesTypeAdapter.dump_json(...)`.
- `load_history(conversation_id)` — the inverse, `.validate_json(...)`, or `[]` for a new
  conversation.
- `clear_history(conversation_id)` — deletes the file (used by the CLI's `/reset`).

The CLI is what actually wires this into a conversation: load history before the first turn, pass
it as `message_history=` on every `agent.run_sync(...)` call, and save `result.all_messages()`
back to disk after every turn. `ModelMessagesTypeAdapter` round-trips message objects (including
tool calls and their results) exactly, so resuming a saved conversation looks identical to
Pydantic AI as one that never stopped.

### `cli.py` — the demo that ties it together

Two subcommands: `ingest` (calls `rag.ingest_file`) and `chat` (the REPL). The chat loop is short
enough to read as a script: build a `StudentContext` from the flags, load history, then for each
line of input either handle a `/command` locally (`/level`, `/ingest`, `/reset`, `/exit`) or call
`teaching_assistant_agent.run_sync(text, deps=deps, message_history=history)`, print the output,
and persist the new history. This file has no logic of its own beyond argument parsing and I/O —
everything it calls is a public function from another module.

### `mock.py` — the zero-API-key study mode

See the dedicated section below.

### `tests/` — what's actually verified

- `test_agent.py` — the capability catalog exists with the right `id`s and `defer_loading=True`;
  the RAG tool is visible from turn one while specialist tools aren't (checked via
  `TestModel(call_tools=[...])` and inspecting `model.last_model_request_parameters.function_tools`);
  the dynamic learner-level instruction actually changes the text sent to the model (checked via
  `result.all_messages()[0].instructions`).
- `test_rag.py` — chunking covers all input text and terminates even on pathological input;
  ingest→search round-trips and returns the right source; courses are isolated from each other.
  Uses a small fake embedder (`monkeypatch.setattr(rag_store, "Embedder", ...)`) so it runs
  offline.
- `test_models.py` / `test_formatting.py` — the Pydantic models validate and round-trip; every
  `format_*` function includes the content it's supposed to.
- `test_memory.py` — save/load/clear round-trip correctly and conversations don't leak into each
  other.
- `test_mock_mode.py` — mock mode specifically: the agent builds and runs with every provider API
  key env var unset; the router dispatches each of the 4 request types to the right tool; every
  sample object actually validates as its model; the mock embedder scores related text as more
  similar than unrelated text.

None of the tests call a real model or a real embeddings API — the whole suite runs offline.

## Tracing one request end to end

Take "Can you make me a 5-question quiz on this?" as a concrete example, assuming a real model
(not mock mode) and a course that already has material ingested:

1. `cli.py`'s chat loop calls `teaching_assistant_agent.run_sync(text, deps=deps, message_history=history)`.
2. The model sees `BASE_INSTRUCTIONS` + the dynamic learner-level instruction + the compact
   capability catalog (`lesson_planning`, `quiz_generation`, `rubric_generation`,
   `resource_recommendation` — descriptions only) + `search_course_materials`'s full tool schema.
3. It recognizes this as a quiz request and calls `load_capability("quiz_generation")`.
4. Pydantic AI resolves that call, and `quiz_generation_capability`'s instructions plus its
   `generate_quiz` tool schema become visible for the rest of this run.
5. The model calls `generate_quiz(topic="...", num_questions=5, difficulty="medium")`.
6. Inside `capabilities/quiz_generation.py`, that function calls `search(...)` for grounding
   (real course-material chunks), builds a prompt, and calls
   `await quiz_agent.run(prompt, usage=ctx.usage)` — a second, independent agent run with
   `output_type=Quiz`.
7. `quiz_agent` returns a validated `Quiz` object. `format_quiz(result.output)` renders it to
   markdown (questions, then an answer key section).
8. That markdown string is the tool's return value, fed back to the *main* agent as a
   `ToolReturnPart`.
9. The main model reads it and produces its final reply (typically presenting or lightly
   introducing the quiz).
10. Back in `cli.py`, `result.all_messages()` — which now includes the `load_capability` call, the
    `generate_quiz` call and its result, and the final text — is saved via
    `memory.save_history(session, ...)`, so the next turn (and the next process run) picks up
    right where this one left off.

## Run without any API keys (mock mode)

Set `TEACHING_ASSISTANT_MOCK=1` (see `.env.example`) and every step above still happens — except
steps 2-3 and 6-7 don't call a real model or a real embeddings API. Concretely:

| Real mode | Mock mode |
|---|---|
| `teaching_assistant_agent`'s model is the string `"anthropic:claude-sonnet-4-6"` | Its model is a `FunctionModel` built by `mock.make_main_agent_mock()` |
| The LLM reads instructions and decides which tool to call | `mock._classify_intent()` keyword-matches the latest user message (`"quiz"` → `generate_quiz`, `"lesson plan"` → `create_lesson_plan`, `"rubric"` → `create_rubric`, `"resource"`/`"recommend"` → `recommend_resources`, anything else → `search_course_materials`) |
| Each specialist sub-agent (`quiz_agent`, etc.) calls a real model with `output_type=Quiz` | Its model is `mock.make_specialist_mock(SAMPLE_QUIZ)` — a `FunctionModel` that always returns one canned `Quiz` object via a `final_result` tool call, regardless of the prompt |
| `Embedder("openai:text-embedding-3-small")` calls a real embeddings API | `rag/store.py`'s `_get_embedder()` returns `mock.MockEmbedder()` — a deterministic hashing-trick bag-of-words vector, so retrieval still behaves sensibly (related text scores higher) without any network call |

Everything else — `search_course_materials` actually searching the real (mock-embedded) index,
`create_lesson_plan` actually calling `format_lesson_plan()` on a real (canned) `LessonPlan`
object, `load_capability` and the deferred-capability mechanism, memory persistence — is the exact
same code path as real mode. Mock mode only replaces "what would a real LLM decide/generate,"
never the surrounding architecture.

Two implementation details worth knowing if you extend this:

- **The branch happens at `Agent(...)` construction, not via `.override()`.** Pydantic AI
  validates that a provider API key is *present* as soon as `Agent("anthropic:...", ...)` is
  constructed — before you'd ever get a chance to swap the model out afterward. So every agent
  module (`agent.py` and each `capabilities/*.py`) checks `config.MOCK_MODE` *before* calling
  `Agent(...)` and passes either the real model string or an already-built `FunctionModel`
  instance. Passing a model *instance* (rather than a string Pydantic AI has to resolve to a
  provider) skips that validation entirely — which is exactly what makes zero-API-key
  construction possible.
- **The mock main-agent router never simulates the `load_capability` handshake.** It calls
  specialist tools like `create_lesson_plan` directly. Probing confirmed this executes correctly
  even though a real model would normally call `load_capability` first — `FunctionModel` exposes
  the full tool catalog to the routing function regardless of `defer_loading`. The mock is
  therefore honest about tool *dispatch and execution* but doesn't reproduce that one piece of
  request-shaping wire protocol; `.claude/skills/building-pydantic-ai-agents/references/ON-DEMAND-CAPABILITIES.md`
  is the place to read about how that handshake works with a real model.
