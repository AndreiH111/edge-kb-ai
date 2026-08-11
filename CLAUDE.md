# Plan: Make the chat agentic + add Jira via MCP

## Context

Today `edge-kb-ai` is a **classic (non-agentic) RAG pipeline**. `services/rag_service.py`
builds one LCEL chain `prompt | model | StrOutputParser` (`rag_service.py:73`) and
`answer_question()` (`rag_service.py:172-194`) *always* runs retrieval, stuffs the result
into a `{context}` variable, and returns a string. The LLM never decides anything — the only
"routing" is a hand-rolled keyword heuristic `is_exhaustive_data_request` (`rag_service.py:112-152`).

Goal:
1. **Make the chat agentic** — let the LLM decide which tool(s) to call (search the knowledge
   base, and now query Jira) instead of a fixed retrieve-then-answer flow.
2. **Add Jira via MCP** — connect a Model Context Protocol server that exposes Jira tools.

**Decisions locked with the user:** community **`mcp-atlassian`** server · **Jira Cloud** ·
**read-only** access.

### Why this is straightforward here
- **LangGraph 1.2.10 is already installed** (transitively via LangChain 1.x), including
  `langgraph.prebuilt.create_react_agent` and `ToolNode`. The agent runtime exists today.
- `ChatOpenAI` / `ChatGroq` (already used) support `.bind_tools()`. Only the **local Ollama
  path is a problem**: it uses `OllamaLLM` (a text-completion class that can't tool-call) and
  defaults to `gemma4` (no native tool support).
- Retrieval is a single function `embed_service.retrieve_from_all_sources(query) -> List[Document]`
  (`embed_service.py:297-340`) — trivial to wrap as an agent tool.
- `main.py` already has an `@asynccontextmanager` **lifespan** (`main.py:14-20`) — the correct
  place to open/close the async MCP session.

## What's missing / must be added
- `langchain-mcp-adapters` + the `mcp-atlassian` server package (neither installed).
- An **agent** that binds tools to the chat model.
- An **async path**: MCP tools are async, so the chat route/service must go `async`.
- Config + docs for the new `JIRA_*` / `AGENT_*` settings.

---

## Implementation

### 1. `requirements.txt` — add deps (unpinned, matching repo convention)
Append an optional block:
```
# Optional: agentic chat + Jira via MCP (requires JIRA_* env vars)
langgraph
langchain-mcp-adapters
mcp-atlassian
```
(`langgraph` is currently only transitive; pin it explicitly since we now depend on it directly.
`mcp` SDK arrives transitively. Installing `mcp-atlassian` gives a `mcp-atlassian` console script
we can spawn over stdio — **no `uv`/Docker required**.)

### 2. `core/config.py` — add settings (frozen-dataclass + `os.getenv` style, per existing convention)
Add fields to `Settings` (`config.py:137+`):
- `agent_enabled: bool = _env_bool("AGENT_ENABLED", True)`
- `jira_enabled: bool = _env_bool("JIRA_ENABLED", False)`
- `jira_url: str = os.getenv("JIRA_URL", "")`
- `jira_username: str = os.getenv("JIRA_USERNAME", "")`
- `jira_api_token: str = _first_env_value("JIRA_API_TOKEN", default="")`
- `jira_read_only: bool = _env_bool("JIRA_READ_ONLY", True)`
- `jira_enabled_tools: str = os.getenv("JIRA_ENABLED_TOOLS", "")`  # optional comma list
- `jira_mcp_transport: str = os.getenv("JIRA_MCP_TRANSPORT", "stdio")`  # stdio | streamable_http
- `jira_mcp_command: str = os.getenv("JIRA_MCP_COMMAND", "mcp-atlassian")`
- `jira_mcp_url: str = os.getenv("JIRA_MCP_URL", "")`  # only for streamable_http

### 3. NEW `services/agent_service.py` — the agentic core
Holds tool definitions, the MCP client lifecycle, agent construction, and the async answer fn.

- **Retrieval tools** (reuse existing code):
  - `search_knowledge_base(query: str) -> str` — wraps `embed_service.retrieve_from_all_sources`
    + `rag_service.format_context`. Docstring tells the LLM to use it for the uploaded docs.
  - `list_knowledge_base_sources() -> str` — wraps `embed_service.get_indexed_sources()`.
- **Jira MCP config builder** `_jira_mcp_config()` — returns the `MultiServerMCPClient` dict:
  - stdio: `{"jira": {"command": settings.jira_mcp_command, "args": [...], "transport": "stdio",
    "env": {"JIRA_URL":..., "JIRA_USERNAME":..., "JIRA_API_TOKEN":...,
    "READ_ONLY_MODE": "true", ["ENABLED_TOOLS": settings.jira_enabled_tools]}}}`
  - streamable_http: `{"jira": {"transport": "streamable_http", "url": settings.jira_mcp_url}}`
- **`lifespan_agent()`** — `@asynccontextmanager`. Uses `contextlib.AsyncExitStack`:
  builds `create_llm()` + retrieval tools; if `settings.jira_enabled`, opens
  `MultiServerMCPClient(...).session("jira")` via the stack and `load_mcp_tools(session)`
  (keeps ONE MCP subprocess/connection alive for the whole app lifetime). Builds
  `create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)` and stores it in a module global.
  **Resilient**: if Jira is disabled or the connection fails, log a warning and build a
  retrieval-only agent (still agentic). Clears the agent on shutdown.
- **`async def aanswer_question(question, session_id) -> str`**:
  - Keep the existing fast-path `is_exhaustive_data_request` → `format_all_records` shortcut.
  - Reuse the existing `chat_sessions` history dict: convert stored `{role,content}` →
    `HumanMessage`/`AIMessage`, append the new question, `await agent.ainvoke({"messages": ...})`,
    take the final message content, persist via `add_chat_history`.
  - **Graceful fallback**: if no agent is built (`agent_enabled` false) OR the invoke raises a
    tool-calling error (e.g. gemma4), fall back to the legacy sync `rag_service.answer_question`.

### 4. `services/rag_service.py` — one swap + keep legacy intact
- `create_llm()` local branch: `OllamaLLM` → **`ChatOllama`** (`rag_service.py:32-34`). Safe for
  the legacy LCEL chain too (chat model works with `ChatPromptTemplate | model | StrOutputParser`)
  and is the tool-calling-capable class. Everything else in `rag_service.py` stays as the fallback.

### 5. `main.py` — wire the agent into lifespan
Wrap the existing body: after `job_queue.start()`, `async with agent_service.lifespan_agent(): yield`
(`main.py:14-17`). The MCP session opens on startup and closes on shutdown.

### 6. `api/routes/chat.py` — go async
Make the route `async def` and `answer = await agent_service.aanswer_question(user_question, user_session_id)`
(`chat.py:35`). Session/cookie handling unchanged. (CLI in `main.py` keeps calling the sync
legacy `answer_question` to avoid async/MCP lifecycle off-server.)

### 7. `models/schemas.py` — (optional, additive) surface tool usage
Add `tools_used: Optional[List[str]] = None` to `ChatResponse` (`schemas.py:11-14`); populate by
scanning the agent result messages for tool calls. Non-breaking.

### 8. Docs
- `.env.example`: new `# ---- Jira / MCP (agentic chat) ----` block with the `JIRA_*` / `AGENT_ENABLED` keys.
- `CONFIGURATION.md`: new `## Jira (via MCP)` section — `Variable | Default | Description` table +
  a `>` note that it needs `pip install langchain-mcp-adapters mcp-atlassian` and `JIRA_*` env vars.
- `README.md`: note that chat is agentic and can query Jira read-only.

### 9. Tests
`tests/test_agent_service.py`: a fake tool-calling chat model + a fake tool; assert the agent
invokes the tool and that `aanswer_question` falls back to the legacy chain when the model raises.

---

## Important caveat — the agent needs a tool-capable LLM
The local default `OLLAMA_LLM_MODEL=gemma4` **cannot** tool-call, so agentic features won't
activate on it (the code gracefully falls back to today's RAG chain). To actually get agentic
behavior, use one of: hosted **OpenAI/Groq** (support tools natively), or a tool-capable **Ollama**
model (e.g. `llama3.1`, `qwen2.5`).

## Verification
1. `pip install -r requirements.txt`.
2. `.env`: a tool-capable LLM (e.g. `LLM_MODE=hosted`, `HOSTED_LLM_PROVIDER=openai`, key set) +
   `AGENT_ENABLED=true`, `JIRA_ENABLED=true`, `JIRA_URL=https://<site>.atlassian.net`,
   `JIRA_USERNAME=<email>`, `JIRA_API_TOKEN=<token>`.
3. `uvicorn main:app --reload` → logs should show Jira MCP tools loaded.
4. `POST /ai/chat` "what internships are in the docs?" → agent calls `search_knowledge_base`.
5. `POST /ai/chat` "what are my open Jira issues in project ABC?" → agent calls `jira_search`.
6. Regression: set local `gemma4` → request still returns an answer (legacy fallback path).
7. `python -m unittest discover -s tests` (existing suite + new `test_agent_service`).

## Files touched
`requirements.txt`, `core/config.py`, `services/agent_service.py` (new), `services/rag_service.py`,
`main.py`, `api/routes/chat.py`, `models/schemas.py`, `.env.example`, `CONFIGURATION.md`,
`README.md`, `tests/test_agent_service.py` (new).

---

## Reference: how the MCP wiring looks in code

```python
# services/agent_service.py (sketch)
import contextlib, logging
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from core.config import settings
from services import embed_service, rag_service

logger = logging.getLogger(__name__)
_agent = None

@tool
def search_knowledge_base(query: str) -> str:
    """Search the uploaded documents (Excel/PDF/PowerPoint) for info relevant to the query."""
    docs = embed_service.retrieve_from_all_sources(query)
    return rag_service.format_context(docs) if docs else "No relevant context found."

def _jira_mcp_config() -> dict:
    if settings.jira_mcp_transport == "streamable_http":
        return {"jira": {"transport": "streamable_http", "url": settings.jira_mcp_url}}
    return {"jira": {
        "command": settings.jira_mcp_command, "args": [], "transport": "stdio",
        "env": {"JIRA_URL": settings.jira_url, "JIRA_USERNAME": settings.jira_username,
                "JIRA_API_TOKEN": settings.jira_api_token, "READ_ONLY_MODE": "true"},
    }}

@contextlib.asynccontextmanager
async def lifespan_agent():
    global _agent
    if not settings.agent_enabled:
        yield; return
    tools = [search_knowledge_base]
    async with contextlib.AsyncExitStack() as stack:
        if settings.jira_enabled:
            try:
                client = MultiServerMCPClient(_jira_mcp_config())
                session = await stack.enter_async_context(client.session("jira"))
                tools += await load_mcp_tools(session)
            except Exception as e:
                logger.warning("Jira MCP unavailable, retrieval-only agent: %s", e)
        _agent = create_react_agent(rag_service.create_llm(), tools, prompt=SYSTEM_PROMPT)
        yield
    _agent = None
```
