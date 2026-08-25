# CLAUDE.md

Guidance for Claude Code working in this repository. Keep it concise and current; delete anything that goes stale.

## Project

`edge-kb-ai` is a FastAPI retrieval-augmented-generation (RAG) service. Users upload documents (Excel/PDF/PowerPoint) which are parsed, chunked, embedded, and indexed into a vector store; the `/ai/chat` endpoints answer questions about them.

Chat is **agentic**: a LangGraph tool-calling agent decides whether to search the knowledge base or query Jira (read-only, via an MCP server), and falls back to a classic retrieve-then-answer chain when no tool-capable LLM/agent is available.

Python 3.13, Windows-first (venv at `./venv`). LLM, embeddings, vector store, and file storage are all swappable via env vars.

## Commands

Run from the repo root with the venv Python (`venv\Scripts\python.exe`), or activate first with `venv\Scripts\Activate.ps1`.

```powershell
# Install dependencies
venv\Scripts\python.exe -m pip install -r requirements.txt

# Run the API — http://127.0.0.1:8000 (interactive docs at /docs)
venv\Scripts\python.exe -m uvicorn main:app --reload

# Interactive CLI chat (uses the legacy sync path)
venv\Scripts\python.exe main.py

# Run all tests — pytest runs the service unittests AND the endpoint suite (offline)
venv\Scripts\python.exe -m pytest -q

# Just the HTTP endpoint suite (pytest + TestClient) / just the service tests
venv\Scripts\python.exe -m pytest tests/api -v
venv\Scripts\python.exe -m pytest tests/unit -v

# Backward-compatible: legacy unittest discovery still finds the moved tests
venv\Scripts\python.exe -m unittest discover -s tests
venv\Scripts\python.exe -m unittest tests.unit.test_agent_service

# Lint (pylint, config in .pylintrc) — expected to stay at 10.00/10
venv\Scripts\python.exe -m pylint api core models services tests main.py conftest.py
```

Config files: `pytest.ini` (`pythonpath=.`, `testpaths=tests`) and `.pylintrc`; no formatter, `pyproject.toml`, or `Makefile`. Service tests are `unittest.TestCase` (run natively by both `pytest` and `unittest`); the endpoint tests are pytest-style and run under `pytest` only. `.pylintrc` fixes genuine defects in code and disables only intentional-pattern / false-positive / design-metric checks (each documented inline) — keep it at 10.00/10.

## Architecture

Thin routers over service modules. All routes are prefixed `/ai`.

- `main.py` — FastAPI app + lifespan (starts the job queue, opens the agent/MCP session). Also a `__main__` CLI loop that calls the legacy sync path.
- `api/routes/` — `chat.py` (`/ai/chat`, `/ai/chat/stream` SSE, `/ai/agent-status`), `upload.py`, `files.py`, `delete.py`, `jobs.py`.
- `core/config.py` — every setting as a frozen `Settings` dataclass read from env/`.env`. Add new config here via the `_env_*` helpers.
- `models/schemas.py` — Pydantic request/response models.
- `services/` — key modules:
  - `agent_service.py` — the agentic core: LangGraph `create_react_agent`, KB tools, Jira MCP loading, SSE streaming (`astream_answer`), and `AgentStatus`.
  - `rag_service.py` — the classic LCEL chain, `create_llm()`, in-memory `chat_sessions`, and the exhaustive-dump fast path. This is the agent's fallback.
  - `embed_service.py` — vector store + `retrieve_from_all_sources`. **Importing it runs `initialize_on_startup()`** (vector store + S3 sync) — heavy; keep it out of unit tests.
  - `vectorstore_service.py` (Chroma / SAP HANA), `storage_service.py` (local / S3), `job_queue.py` (async upload/reindex).
  - Parsing & chunking: `excel_service.py`, `pdf_service.py`, `ppt_service.py`, `chunking_service.py`, `classifier_service.py`.
  - Support: `file_service.py`, `metadata_service.py`, `hash_embedding_service.py`.

## Conventions

- **Every code change goes through Ponytail — no exceptions.** Before writing anything, apply the **Ponytail — lazy senior dev mode** rules (full text at the bottom of this file): climb the reuse ladder first (does it need building? does it already exist here / in the stdlib / in an installed dep?), prefer the shortest working diff, delete over add, fix bugs at the root not the symptom, and add no abstraction/dependency/boilerplate nobody asked for.
- **Always verify before finishing — no exceptions.** After *every* change, run both the tests and the linter and confirm they pass before calling the work done:
  ```powershell
  venv\Scripts\python.exe -m pytest -q                                       # all tests must pass
  venv\Scripts\python.exe -m pylint api core models services tests main.py conftest.py   # must stay 10.00/10
  ```
  If either fails, fix it (or, for a justified lint case, disable the specific check in `.pylintrc` with an inline rationale) — never leave the tree red or claim done on unrun checks.
- **Config:** don't read `os.getenv` outside `core/config.py`; add a field to `Settings` and reference `settings.<field>`.
- **Lazy imports:** `agent_service` imports `rag_service`/`embed_service` *inside functions* so the module stays cheap to import (tests depend on this). Follow that pattern for heavy deps.
- **Tests:** two layers, both offline. `tests/unit/` — service `unittest` tests; fakes injected via `sys.modules` (`inject_services` / `inject_modules` in `tests/unit/test_agent_service.py`), the agent driven by fake tool-calling models or a scripted `astream`. `tests/api/` — pytest + `TestClient` endpoint tests; the root `conftest.py` neutralizes `.env` and pins offline settings, and `tests/api/conftest.py` injects a fake `embed_service` before importing `main` (so the import stays cheap) and drives scenarios with `monkeypatch`. Never import real services (especially `embed_service`) into a test.
- **Style:** 4-space indent, type hints and short docstrings on new functions; match the surrounding module.
- **Docs:** user-facing config lives in `CONFIGURATION.md`, features/endpoints in `README.md` — update them when behavior changes.
- **Secrets:** never commit credentials; real values live in `.env` (gitignored). `CLAUDE.md` becomes part of the prompt — keep it secret-free.

## Ponytail — lazy senior dev mode

Vendored from the [ponytail](https://github.com/DietrichGebert/ponytail) skill (`AGENTS.md`) and inlined here because installing Claude Code skills is blocked in this environment. Apply it to all code work in this repo.

You are a lazy senior developer. Lazy means efficient, not careless. The best code is the code never written.

Before writing any code, stop at the first rung that holds:

1. Does this need to be built at all? (YAGNI)
2. Does it already exist in this codebase? Reuse the helper, util, or pattern that's already here — don't rewrite it.
3. Does the standard library already do this? Use it.
4. Does a native platform feature cover it? Use it.
5. Does an already-installed dependency solve it? Use it.
6. Can this be one line? Make it one line.
7. Only then: write the minimum code that works.

The ladder runs *after* you understand the problem, not instead of it: read the task and the code it touches, trace the real flow end to end, then climb.

**Bug fix = root cause, not symptom.** Grep every caller of the function you touch and fix the shared function once — one guard there is a smaller diff than one per caller, and patching only the path the ticket names leaves a sibling caller broken.

Rules:

- No abstractions that weren't explicitly requested.
- No new dependency if it can be avoided.
- No boilerplate nobody asked for.
- Deletion over addition. Boring over clever. Fewest files possible.
- Shortest working diff wins — but only once you understand the problem. The smallest change in the wrong place isn't lazy, it's a second bug.
- Question complex requests: "Do you actually need X, or does Y cover it?"
- When two stdlib approaches are the same size, pick the edge-case-correct one — lazy means less code, not the flimsier algorithm.
- Mark deliberate simplifications that cut a real corner with a `ponytail:` comment naming the ceiling and upgrade path (e.g. global lock, O(n²) scan, naive heuristic).

Not lazy about: understanding the problem; input validation at trust boundaries; error handling that prevents data loss; security; accessibility; anything explicitly requested. Non-trivial logic leaves ONE runnable check behind — the smallest thing that fails if the logic breaks (an assert-based self-check or one small test file; no frameworks, no fixtures). Trivial one-liners need no test.

## Gotchas (project-specific)

- **The agent needs a tool-capable LLM.** The local default `OLLAMA_LLM_MODEL=gemma4` can't tool-call, so agentic features silently fall back to the classic RAG chain. Use hosted OpenAI/Groq or a tool-capable Ollama model (`llama3.1`, `qwen2.5`).
- **`GET /ai/agent-status` is the source of truth** for whether the agent and Jira tools loaded (`jira_loaded`, `tools`, `jira_error`). Failures degrade silently to a retrieval-only or legacy answer.
- **Jira MCP over stdio spawns `JIRA_MCP_COMMAND` via PATH.** On Windows set it to the absolute `...\venv\Scripts\mcp-atlassian.exe`, or launch uvicorn from the activated venv.
- **Too many tool schemas can exceed the LLM's token/rate limit.** Binding all ~38 Jira tools adds thousands of tokens per request; on low tiers (e.g. Groq's 8000 TPM) this returns `413`, the agent throws, and it falls back to the legacy chain ("I don't have that information"). Restrict `JIRA_ENABLED_TOOLS` to a few tools.
- **`.env` changes require a full server restart** — `--reload` only watches `.py` files.
- `langgraph.prebuilt.create_react_agent` is deprecated (moved to `langchain.agents`) but intentionally kept under the pinned `langgraph<2`.

## Reference

- `README.md` — features, endpoints, usage examples, project structure.
- `CONFIGURATION.md` — every env var with defaults (LLM, embeddings, chunking, vector store, storage, Jira/MCP).
- `.env.example` — template `.env` to copy.
