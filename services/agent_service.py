"""Agentic chat core.

Turns the classic retrieve-then-answer RAG flow into a tool-calling agent. The
LLM decides which tool(s) to invoke: search the uploaded knowledge base and,
when configured, query Jira through a Model Context Protocol (MCP) server.

The heavy service modules (``embed_service`` / ``rag_service``) and the optional
MCP adapter packages are imported lazily inside functions so that importing this
module stays cheap and side-effect free — useful for the running app (``main.py``
already imports ``rag_service``) and essential for unit tests.
"""

import asyncio
import contextlib
import json
import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import date
from typing import AsyncIterator, List, Literal, Optional, Tuple, Union

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from core.config import settings

logger = logging.getLogger(__name__)

# The single agent instance, built during the app lifespan. ``None`` means the
# agent is unavailable and callers fall back to the legacy RAG chain.
_agent = None


# Indexed files named in the system prompt; the rest are summarised as "+N more".
MAX_PROMPT_SOURCES = 30

PROMPT_RULES = """Rules:
- You know nothing about the user's data except what your tools return. Call a tool before answering any question \
about it, and never say information is missing before you have searched.
- Search with short standalone keyword queries; resolve names and references from the conversation. If results are \
thin, search again with different wording. To compare items, search once per item.
- Never estimate counts, totals or averages from a few passages: use a tool that computes over all rows if you have \
one, otherwise say the passages may not cover every row.
- If the results do not answer the question, or it needs a system none of your tools reach, say plainly what is \
missing instead of guessing.
- Reply in the user's language. Use markdown tables for multi-row data.
- Cite the passages you used as [n] with their file name; do not cite passages you did not use."""


@dynamic_prompt
def system_prompt(request: ModelRequest) -> str:
    """Build the system prompt for one model call: date, files, bound tools, rules."""
    from services import embed_service

    tool_names = [t.get("name", "") if isinstance(t, dict) else t.name for t in request.tools]
    # File names come from uploaders and the client: JSON-quote them so they stay data, not instructions.
    pinned = embed_service.pinned_source.get()
    if pinned:
        files = (
            f"The user pinned the file {json.dumps(pinned, ensure_ascii=False)}: "
            "answer knowledge-base questions only from it."
        )
    else:
        # shortcut: rescans every chunk per model call (~7 ms at 200k), cache it if the corpus grows far past that.
        sources = embed_service.get_indexed_sources()
        files = f"Indexed files: {json.dumps(sources[:MAX_PROMPT_SOURCES], ensure_ascii=False)}"
        if len(sources) > MAX_PROMPT_SOURCES:
            files += f" (+{len(sources) - MAX_PROMPT_SOURCES} more)"
        files += "."

    lines = [
        "You are an expert assistant for the edge-kb knowledge base.",
        # shortcut: server-local date, pass the client's time zone if users span zones.
        f"Today is {date.today():%A, %d %B %Y}.",
        files,
        f"Your tools: {', '.join(tool_names)}.",
    ]
    if any(name.startswith("jira_") for name in tool_names):
        lines.append(
            "Jira tools are read-only: use them for Jira projects, issues, sprints and boards, "
            "and decline any request to create, change or delete anything in Jira. "
            "Cite Jira data by its issue or project key."
        )
    return "\n".join(lines) + "\n\n" + PROMPT_RULES


@dataclass
class AgentResult:
    """The outcome of an agent turn: the answer, which tools it called and the passages it cited."""

    answer: str
    tools_used: List[str] = field(default_factory=list)
    sources: List[dict] = field(default_factory=list)


@dataclass
class AgentStatus:
    """A snapshot of how the agent was built during the app lifespan.

    Exposed via ``GET /ai/agent-status`` so operators can tell whether the agent
    and its Jira MCP tools actually loaded, instead of the failure being swallowed
    into a silent degradation to a retrieval-only (or legacy) chain.
    """

    agent_enabled: bool = False
    agent_ready: bool = False
    jira_enabled: bool = False
    jira_loaded: bool = False
    tool_count: int = 0
    tools: List[str] = field(default_factory=list)
    jira_error: Optional[str] = None


# Populated by ``lifespan_agent`` on startup; reset on shutdown.
_status = AgentStatus()


def get_agent_status() -> AgentStatus:
    """Return the current agent build status (see ``AgentStatus``)."""
    return _status


# The user's own words for this chat turn, set by the public wrappers below, so the search
# tool's retrieval graph judges passages against the question and not only its query.
user_question: ContextVar[Optional[str]] = ContextVar("user_question", default=None)


@tool
def search_knowledge_base(query: str) -> str:
    """Search the uploaded documents (Excel/PDF/PowerPoint) for information
    relevant to the query. Use this for any question about the knowledge base's
    contents. Returns the most relevant passages, or a message if none match."""
    from services import rag_graph, rag_service

    documents = rag_graph.retrieve(query, question=user_question.get())
    if not documents:
        return "No relevant context found in the knowledge base."

    context = rag_service.format_context(documents)
    if any(doc.metadata.get("low_relevance") for doc in documents):
        return (
            "Note: no passage below was judged relevant to the question. Unless they clearly answer "
            "it, say the knowledge base doesn't seem to contain the answer instead of guessing.\n\n"
            + context
        )
    return context


@tool
def list_knowledge_base_sources() -> str:
    """List the source files currently indexed in the knowledge base. Use this
    when the user asks what documents or files are available."""
    from services import embed_service

    sources = embed_service.get_indexed_sources()
    pinned = embed_service.pinned_source.get()
    if pinned:
        sources = [source for source in sources if source == pinned]
    if not sources:
        return "No documents are currently indexed."

    return "Indexed source files:\n" + "\n".join(f"- {source}" for source in sources)


def _knowledge_base_tools() -> list:
    return [search_knowledge_base, list_knowledge_base_sources, query_spreadsheet]


class SpreadsheetFilter(BaseModel):
    """Keep only the rows where ``column <op> value``."""

    column: str
    op: Literal["==", "!=", ">", ">=", "<", "<=", "contains"]
    value: Union[str, int, float]


@tool
def query_spreadsheet(
    file: Optional[str] = None,
    sheet: Optional[str] = None,
    filters: Optional[List[SpreadsheetFilter]] = None,
    group_by: Optional[str] = None,
    aggregate: Optional[Literal["count", "sum", "mean", "min", "max"]] = None,
    column: Optional[str] = None,
    sort_by: Optional[str] = None,
    descending: bool = False,
    limit: int = 20,
) -> str:
    """Compute exact answers over EVERY row of an indexed Excel workbook. Use this,
    not `search_knowledge_base` (which only sees a few rows), for counts, totals,
    averages, min/max, group-bys, top-N and complete row lists.

    `file` is the workbook's file name (it defaults to the file the chat is pinned
    to); `sheet` defaults to the first sheet. Rows must match every filter: numbers
    compare numerically, text ignoring case, dates as YYYY-MM-DD text. `group_by`
    alone counts rows per group; aggregate sum/mean/min/max needs `column`.
    `sort_by` (with `descending` for top-N) orders the rows or groups and `limit`
    caps them (default 20, max 200). Each result starts with the sheet, its row
    count and its column names."""
    try:
        from services import embed_service, excel_service

        workbooks = {path.name: path for path in excel_service.get_excel_files()}
        pinned = embed_service.pinned_source.get()
        if pinned:  # a chat pinned to one file must not read any other
            workbooks = {name: path for name, path in workbooks.items() if name == pinned}
        path = workbooks.get(file or pinned)
        if path is None:
            return f"Unknown or missing `file`. Indexed workbooks: {', '.join(workbooks) or 'none'}."
        rules = [(rule.column, rule.op, rule.value) for rule in filters or []]
        return excel_service.query_sheet(path, sheet, rules, group_by, aggregate, column, sort_by, descending, limit)
    except Exception as exc:  # noqa: BLE001 - a raised tool error would abort the agent into the legacy chain
        return f"query_spreadsheet failed: {exc}"


def _jira_mcp_config() -> dict:
    """Build the ``MultiServerMCPClient`` configuration for the Jira server."""
    if settings.jira_mcp_transport == "streamable_http":
        return {"jira": {"transport": "streamable_http", "url": settings.jira_mcp_url}}

    env = {
        "JIRA_URL": settings.jira_url,
        "JIRA_USERNAME": settings.jira_username,
        "JIRA_API_TOKEN": settings.jira_api_token,
        "READ_ONLY_MODE": "true" if settings.jira_read_only else "false",
    }
    if settings.jira_enabled_tools:
        env["ENABLED_TOOLS"] = settings.jira_enabled_tools

    return {
        "jira": {
            "command": settings.jira_mcp_command,
            "args": [],
            "transport": "stdio",
            "env": env,
        }
    }


async def _load_jira_tools(stack: contextlib.AsyncExitStack) -> list:
    """Open a single long-lived MCP session (kept alive by ``stack``) and return
    the Jira tools it exposes. Imports the MCP adapters lazily so the package is
    only required when Jira is actually enabled."""
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools

    client = MultiServerMCPClient(_jira_mcp_config())
    session = await stack.enter_async_context(client.session("jira"))
    return await load_mcp_tools(session)


@contextlib.asynccontextmanager
async def lifespan_agent():
    """Build the agent on startup and tear down the MCP connection on shutdown.

    Resilient by design: if the agent is disabled, or Jira is unavailable, or the
    LLM cannot be built into an agent, this logs a warning and leaves ``_agent``
    as ``None`` (or retrieval-only) so callers fall back to the legacy RAG chain.
    The outcome is recorded in the module ``_status`` (see ``get_agent_status``)
    so a degradation is inspectable rather than silent.
    """
    global _agent, _status

    _status = AgentStatus(
        agent_enabled=settings.agent_enabled,
        jira_enabled=settings.jira_enabled,
    )

    if not settings.agent_enabled:
        logger.info("Agentic chat disabled (AGENT_ENABLED=false); using legacy RAG chain.")
        _agent = None
        yield
        return

    from services import rag_service

    tools = _knowledge_base_tools()

    async with contextlib.AsyncExitStack() as stack:
        if settings.jira_enabled:
            try:
                jira_tools = await _load_jira_tools(stack)
                tools.extend(jira_tools)
                _status.jira_loaded = True
                logger.info("Loaded %d Jira MCP tool(s).", len(jira_tools))
            except ImportError as exc:
                # A missing package is actionable — say exactly how to fix it,
                # rather than lumping it in with connection failures.
                _status.jira_error = (
                    f"MCP packages not installed ({exc}). "
                    "Run: pip install langchain-mcp-adapters mcp-atlassian"
                )
                logger.warning(
                    "Jira enabled but MCP packages missing; building a "
                    "retrieval-only agent. %s",
                    _status.jira_error,
                )
            except Exception as exc:  # noqa: BLE001 - stay up on any MCP failure
                _status.jira_error = str(exc)
                logger.warning(
                    "Jira MCP unavailable; building a retrieval-only agent: %s", exc
                )

        try:
            _agent = create_agent(
                rag_service.create_llm(), tools, middleware=[system_prompt]
            )
            logger.info("Agentic chat ready with %d tool(s).", len(tools))
        except Exception as exc:  # noqa: BLE001 - fall back to the legacy chain
            logger.warning(
                "Could not build the agent; falling back to legacy RAG chain: %s", exc
            )
            _agent = None

        _status.agent_ready = _agent is not None
        _status.tools = [tool.name for tool in tools]
        _status.tool_count = len(tools)

        try:
            yield
        finally:
            _agent = None
            _status = AgentStatus()


def _chunk_text(content) -> str:
    """Flatten a message's ``content`` into plain text.

    Chat models return either a plain ``str`` or a list of content parts (each a
    ``str`` or a ``{"type": ..., "text": ...}`` dict). This collapses both shapes
    to a string and is shared by ``_extract_answer`` and the streaming loop.
    """
    if isinstance(content, list):
        parts = []
        for chunk in content:
            if isinstance(chunk, str):
                parts.append(chunk)
            elif isinstance(chunk, dict):
                parts.append(chunk.get("text", ""))
        return "".join(parts)

    return str(content)


def _extract_answer(result) -> str:
    """Pull the final assistant text out of an agent result."""
    messages = result.get("messages") if isinstance(result, dict) else None
    if not messages:
        return ""

    return _chunk_text(messages[-1].content).strip()


def _extract_tools_used(messages) -> List[str]:
    """Collect, in order and de-duplicated, the names of tools the agent called."""
    used: List[str] = []
    for message in messages or []:
        for call in getattr(message, "tool_calls", None) or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name and name not in used:
                used.append(name)
    return used


async def aanswer_question(
    question: str, session_id: str, source: Optional[str] = None
) -> AgentResult:
    """Answer a question with the agent, reusing the legacy chat-history store.

    Falls back to the synchronous legacy RAG chain when the agent is unavailable
    or an invocation fails (e.g. a local model that cannot tool-call).
    ``source`` limits retrieval to that one indexed file.
    """
    from services import embed_service

    # asyncio.to_thread and LangChain's executor copy context vars, so the pin
    # reaches retrieval running in worker threads too.
    user_question.set(question)  # no reset: each request runs in its own asyncio task context
    with embed_service.use_source(source):
        return await _aanswer_question(question, session_id)


async def _aanswer_question(question: str, session_id: str) -> AgentResult:
    from services import rag_service

    history = rag_service.get_history(session_id)

    # Fast path: exhaustive "list everything" requests bypass the LLM entirely,
    # matching the legacy behaviour.
    if rag_service.is_exhaustive_data_request(question):
        answer = rag_service.format_all_records()
        rag_service.add_chat_history(history, question, answer)
        return AgentResult(answer=answer, tools_used=[])

    # One [n] numbering per turn: tool threads copy this context, so they share the list.
    # No reset: each request runs in its own task, and every turn sets a fresh list.
    passages: list = []
    rag_service.turn_passages.set(passages)

    if _agent is None:
        answer = await asyncio.to_thread(rag_service.answer_question, question, session_id)
        return AgentResult(answer=answer, tools_used=[], sources=rag_service.cited_sources(answer))

    messages = [*rag_service.recent_messages(history), HumanMessage(content=question)]

    try:
        result = await _agent.ainvoke({"messages": messages})
    except Exception:
        logger.exception("Agent invocation failed; falling back to legacy RAG chain.")
        passages.clear()
        answer = await asyncio.to_thread(rag_service.answer_question, question, session_id)
        return AgentResult(answer=answer, tools_used=[], sources=rag_service.cited_sources(answer))

    answer = _extract_answer(result)
    tools_used = _extract_tools_used(result.get("messages", []))
    rag_service.add_chat_history(history, question, answer)
    return AgentResult(answer=answer, tools_used=tools_used, sources=rag_service.cited_sources(answer))


async def astream_answer(
    question: str, session_id: str, source: Optional[str] = None
) -> AsyncIterator[Tuple[str, dict]]:
    """Stream an answer as ``(event_name, payload)`` pairs.

    Event names:
    - ``token`` — an incremental piece of the answer text (``{"text": ...}``).
    - ``tool``  — a tool the agent invoked, emitted once per unique tool
      (``{"name": ...}``).
    - ``done``  — the final payload (``{"session_id", "answer", "tools_used", "sources"}``);
      ``sources`` is ``[{"n", "locator"}]`` for each passage the answer cites as ``[n]``.
    - ``error`` — a mid-stream failure *after* tokens already started.

    Mirrors ``aanswer_question``: same exhaustive fast-path, and the same legacy
    fallback when the agent is unavailable or fails before emitting any token.
    History persists only on the normal ``done`` path (a client that disconnects
    mid-stream will not save the turn). ``source`` limits retrieval to one file.
    """
    from services import embed_service

    user_question.set(question)  # no reset: each request runs in its own asyncio task context
    with embed_service.use_source(source):
        async for event in _astream_answer(question, session_id):
            yield event


async def _astream_answer(question: str, session_id: str) -> AsyncIterator[Tuple[str, dict]]:
    from services import rag_service

    history = rag_service.get_history(session_id)

    # Fast path: exhaustive "list everything" requests bypass the LLM entirely,
    # matching the legacy behaviour.
    if rag_service.is_exhaustive_data_request(question):
        answer = rag_service.format_all_records()
        rag_service.add_chat_history(history, question, answer)
        yield ("token", {"text": answer})
        yield ("done", {"session_id": session_id, "answer": answer, "tools_used": [], "sources": []})
        return

    # Same per-turn numbering as _aanswer_question.
    passages: list = []
    rag_service.turn_passages.set(passages)

    if _agent is None:
        async for event in _astream_legacy(question, session_id):
            yield event
        return

    messages = [*rag_service.recent_messages(history), HumanMessage(content=question)]

    parts: List[str] = []
    tools_used: List[str] = []
    started = False

    try:
        async for mode, data in _agent.astream(
            {"messages": messages}, stream_mode=["updates", "messages"]
        ):
            if mode == "messages":
                chunk, meta = data
                # Only the agent's answer node: LLM calls inside tools also emit chunks.
                if isinstance(chunk, AIMessageChunk) and meta.get("langgraph_node") == "model":
                    text = _chunk_text(chunk.content)
                    if text:
                        started = True
                        parts.append(text)
                        yield ("token", {"text": text})
            elif mode == "updates":
                for _node, update in (data or {}).items():
                    for message in (update or {}).get("messages", []):
                        for call in getattr(message, "tool_calls", None) or []:
                            name = (
                                call.get("name")
                                if isinstance(call, dict)
                                else getattr(call, "name", None)
                            )
                            if name and name not in tools_used:
                                tools_used.append(name)
                                yield ("tool", {"name": name})
    except Exception:
        logger.exception("Agent streaming failed.")
        if not started:
            # Nothing emitted yet — fall back to the legacy chain, consistent
            # with aanswer_question's behaviour on invocation failure.
            passages.clear()
            async for event in _astream_legacy(question, session_id):
                yield event
            return
        # Tokens already streamed to the client; we cannot cleanly restart, so
        # surface the failure and stop.
        yield ("error", {"message": "The agent failed while generating the answer."})
        return

    answer = "".join(parts).strip()
    rag_service.add_chat_history(history, question, answer)
    sources = rag_service.cited_sources(answer)
    yield ("done", {"session_id": session_id, "answer": answer, "tools_used": tools_used, "sources": sources})


async def _astream_legacy(
    question: str, session_id: str
) -> AsyncIterator[Tuple[str, dict]]:
    """Stream the classic RAG chain token-by-token.

    Retrieves through ``query_service`` exactly as ``rag_service.answer_question``
    does (in a thread), then streams the LCEL chain. If the chain stream itself
    fails, falls back to the fully synchronous
    ``answer_question`` (which persists its own turn) emitted as a single token.
    """
    from services import query_service, rag_service

    history = rag_service.get_history(session_id)
    chat_history = rag_service.format_chat_history(history)

    try:
        context = await asyncio.to_thread(
            query_service.build_retrieval_documents,
            question,
            chat_history,
            rag_service.model,
        )
        parts: List[str] = []
        async for chunk in rag_service.chain.astream(
            {
                "chat_history": chat_history,
                "context": rag_service.format_context(context),
                "question": question,
            }
        ):
            parts.append(chunk)
            yield ("token", {"text": chunk})
    except Exception:
        logger.exception("Legacy streaming failed; using the synchronous fallback.")
        rag_service.turn_passages.get().clear()
        answer = await asyncio.to_thread(
            rag_service.answer_question, question, session_id
        )
        sources = rag_service.cited_sources(answer)
        yield ("token", {"text": answer})
        yield ("done", {"session_id": session_id, "answer": answer, "tools_used": [], "sources": sources})
        return

    answer = "".join(parts).strip()
    rag_service.add_chat_history(history, question, answer)
    sources = rag_service.cited_sources(answer)
    yield ("done", {"session_id": session_id, "answer": answer, "tools_used": [], "sources": sources})
