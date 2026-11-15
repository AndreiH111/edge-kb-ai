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
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional, Tuple

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
)
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from core.config import settings

logger = logging.getLogger(__name__)

# The single agent instance, built during the app lifespan. ``None`` means the
# agent is unavailable and callers fall back to the legacy RAG chain.
_agent = None


SYSTEM_PROMPT = """You are an expert assistant for the edge-kb knowledge base. You can also query Jira (read-only).

IMPORTANT: You have NO prior knowledge of this user's Jira instance or their uploaded documents. You MUST call the appropriate tool to answer any question about them. Never answer such a question from memory, and never claim you lack the information before you have actually called a tool.

Tools:
- `search_knowledge_base`: search the uploaded documents (Excel, PDF, PowerPoint) for anything about their contents.
- `list_knowledge_base_sources`: list which files are currently indexed.
- Jira tools (e.g. `jira_get_all_projects`, `jira_search`, `jira_get_issue`, and the board/sprint tools, when available): answer questions about Jira projects, issues, sprints, or boards. Jira access is read-only.

How to decide:
- Any question about Jira (projects, issues, sprints, boards, or "my" items in Jira) -> call a Jira tool first.
- Any question about the uploaded documents or knowledge base -> call `search_knowledge_base` first.
- Only answer once a tool has returned. If a tool returns nothing useful, say so plainly instead of inventing an answer.
- Use the conversation history to resolve follow-up questions and references to earlier messages.
- Answer clearly and concisely, and mention the source file or the Jira tool the information came from.
"""


@dataclass
class AgentResult:
    """The outcome of an agent turn: the answer plus which tools it called."""

    answer: str
    tools_used: List[str] = field(default_factory=list)


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


@tool
def search_knowledge_base(query: str) -> str:
    """Search the uploaded documents (Excel/PDF/PowerPoint) for information
    relevant to the query. Use this for any question about the knowledge base's
    contents. Returns the most relevant passages, or a message if none match."""
    from services import embed_service, rag_service

    documents = embed_service.retrieve_from_all_sources(query)
    if not documents:
        return "No relevant context found in the knowledge base."

    return rag_service.format_context(documents)


@tool
def list_knowledge_base_sources() -> str:
    """List the source files currently indexed in the knowledge base. Use this
    when the user asks what documents or files are available."""
    from services import embed_service

    sources = embed_service.get_indexed_sources()
    if not sources:
        return "No documents are currently indexed."

    return "Indexed source files:\n" + "\n".join(f"- {source}" for source in sources)


def _knowledge_base_tools() -> list:
    return [search_knowledge_base, list_knowledge_base_sources]


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
            _agent = create_react_agent(
                rag_service.create_llm(), tools, prompt=SYSTEM_PROMPT
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


def _history_to_messages(history: List[dict]) -> List[BaseMessage]:
    """Convert the stored ``{role, content}`` history into chat messages."""
    messages: List[BaseMessage] = []
    for entry in history[-settings.max_chat_history_messages :]:
        content = entry.get("content", "")
        if entry.get("role") == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    return messages


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


async def aanswer_question(question: str, session_id: str) -> AgentResult:
    """Answer a question with the agent, reusing the legacy chat-history store.

    Falls back to the synchronous legacy RAG chain when the agent is unavailable
    or an invocation fails (e.g. a local model that cannot tool-call).
    """
    from services import rag_service

    history = rag_service.chat_sessions.setdefault(session_id, [])

    # Fast path: exhaustive "list everything" requests bypass the LLM entirely,
    # matching the legacy behaviour.
    if rag_service.is_exhaustive_data_request(question):
        answer = rag_service.format_all_records()
        rag_service.add_chat_history(history, question, answer)
        return AgentResult(answer=answer, tools_used=[])

    if _agent is None:
        answer = await asyncio.to_thread(rag_service.answer_question, question, session_id)
        return AgentResult(answer=answer, tools_used=[])

    messages = _history_to_messages(history)
    messages.append(HumanMessage(content=question))

    try:
        result = await _agent.ainvoke({"messages": messages})
    except Exception:
        logger.exception("Agent invocation failed; falling back to legacy RAG chain.")
        answer = await asyncio.to_thread(rag_service.answer_question, question, session_id)
        return AgentResult(answer=answer, tools_used=[])

    answer = _extract_answer(result)
    tools_used = _extract_tools_used(result.get("messages", []))
    rag_service.add_chat_history(history, question, answer)
    return AgentResult(answer=answer, tools_used=tools_used)


async def astream_answer(
    question: str, session_id: str
) -> AsyncIterator[Tuple[str, dict]]:
    """Stream an answer as ``(event_name, payload)`` pairs.

    Event names:
    - ``token`` — an incremental piece of the answer text (``{"text": ...}``).
    - ``tool``  — a tool the agent invoked, emitted once per unique tool
      (``{"name": ...}``).
    - ``done``  — the final payload (``{"session_id", "answer", "tools_used"}``).
    - ``error`` — a mid-stream failure *after* tokens already started.

    Mirrors ``aanswer_question``: same exhaustive fast-path, and the same legacy
    fallback when the agent is unavailable or fails before emitting any token.
    History persists only on the normal ``done`` path (a client that disconnects
    mid-stream will not save the turn).
    """
    from services import rag_service

    history = rag_service.chat_sessions.setdefault(session_id, [])

    # Fast path: exhaustive "list everything" requests bypass the LLM entirely,
    # matching the legacy behaviour.
    if rag_service.is_exhaustive_data_request(question):
        answer = rag_service.format_all_records()
        rag_service.add_chat_history(history, question, answer)
        yield ("token", {"text": answer})
        yield ("done", {"session_id": session_id, "answer": answer, "tools_used": []})
        return

    if _agent is None:
        async for event in _astream_legacy(question, session_id):
            yield event
        return

    messages = _history_to_messages(history)
    messages.append(HumanMessage(content=question))

    parts: List[str] = []
    tools_used: List[str] = []
    started = False

    try:
        async for mode, data in _agent.astream(
            {"messages": messages}, stream_mode=["updates", "messages"]
        ):
            if mode == "messages":
                chunk, _meta = data
                if isinstance(chunk, AIMessageChunk):
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
            async for event in _astream_legacy(question, session_id):
                yield event
            return
        # Tokens already streamed to the client; we cannot cleanly restart, so
        # surface the failure and stop.
        yield ("error", {"message": "The agent failed while generating the answer."})
        return

    answer = "".join(parts).strip()
    rag_service.add_chat_history(history, question, answer)
    yield ("done", {"session_id": session_id, "answer": answer, "tools_used": tools_used})


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

    history = rag_service.chat_sessions.setdefault(session_id, [])
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
        answer = await asyncio.to_thread(
            rag_service.answer_question, question, session_id
        )
        yield ("token", {"text": answer})
        yield ("done", {"session_id": session_id, "answer": answer, "tools_used": []})
        return

    answer = "".join(parts).strip()
    rag_service.add_chat_history(history, question, answer)
    yield ("done", {"session_id": session_id, "answer": answer, "tools_used": []})
