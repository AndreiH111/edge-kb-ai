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
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from core.config import settings

logger = logging.getLogger(__name__)

# The single agent instance, built during the app lifespan. ``None`` means the
# agent is unavailable and callers fall back to the legacy RAG chain.
_agent = None


SYSTEM_PROMPT = """You are an expert assistant for the edge-kb knowledge base.

You have tools available. Decide which to use based on the question:
- Use `search_knowledge_base` to find information inside the uploaded documents
  (Excel, PDF, PowerPoint). Use it whenever the question is about their contents.
- Use `list_knowledge_base_sources` to see which files are currently indexed.
- Use the Jira tools (when available) to answer questions about Jira issues,
  projects, sprints, or boards. Jira access is read-only.

Guidelines:
- Prefer calling a tool over guessing. If a tool returns nothing useful, say so
  plainly instead of inventing an answer.
- Use the conversation history to resolve follow-up questions and references to
  earlier messages.
- Answer clearly and concisely. When you used the knowledge base, mention the
  source file the information came from.
"""


@dataclass
class AgentResult:
    """The outcome of an agent turn: the answer plus which tools it called."""

    answer: str
    tools_used: List[str] = field(default_factory=list)


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
    """
    global _agent

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
                logger.info("Loaded %d Jira MCP tool(s).", len(jira_tools))
            except Exception as exc:  # noqa: BLE001 - stay up on any MCP failure
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

        try:
            yield
        finally:
            _agent = None


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


def _extract_answer(result) -> str:
    """Pull the final assistant text out of an agent result."""
    messages = result.get("messages") if isinstance(result, dict) else None
    if not messages:
        return ""

    content = messages[-1].content
    if isinstance(content, list):
        parts = []
        for chunk in content:
            if isinstance(chunk, str):
                parts.append(chunk)
            elif isinstance(chunk, dict):
                parts.append(chunk.get("text", ""))
        return "".join(parts).strip()

    return str(content).strip()


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
