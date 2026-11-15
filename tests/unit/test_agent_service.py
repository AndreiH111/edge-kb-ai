"""Tests for the agentic chat core.

These tests never import the heavy ``embed_service`` / ``rag_service`` modules.
``agent_service`` imports them lazily, so we inject lightweight fakes via
``inject_services`` and drive the agent with a fake tool-calling chat model.
"""

import asyncio
import contextlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from services import agent_service


_MISSING = object()


def _collect(async_gen):
    """Drain an async generator into a list, running a fresh event loop.

    Mirrors how the sync tests wrap ``aanswer_question`` in ``asyncio.run`` — the
    caller stays inside the relevant ``inject_services`` context while draining.
    """

    async def _run():
        return [item async for item in async_gen]

    return asyncio.run(_run())


@contextlib.contextmanager
def inject_services(**fakes):
    """Temporarily register fake ``services.<name>`` modules.

    Sets both the ``sys.modules`` entry and the attribute on the ``services``
    package so a lazy ``from services import <name>`` resolves to the fake
    deterministically, then restores the original state on exit.
    """
    import services as services_pkg

    saved_module = {}
    saved_attr = {}
    for name, fake in fakes.items():
        full_name = f"services.{name}"
        saved_module[name] = sys.modules.get(full_name, _MISSING)
        saved_attr[name] = getattr(services_pkg, name, _MISSING)
        sys.modules[full_name] = fake
        setattr(services_pkg, name, fake)

    try:
        yield
    finally:
        for name in fakes:
            full_name = f"services.{name}"
            if saved_module[name] is _MISSING:
                sys.modules.pop(full_name, None)
            else:
                sys.modules[full_name] = saved_module[name]

            if saved_attr[name] is _MISSING:
                if hasattr(services_pkg, name):
                    delattr(services_pkg, name)
            else:
                setattr(services_pkg, name, saved_attr[name])


@contextlib.contextmanager
def inject_modules(**modules):
    """Temporarily register fake modules by full dotted name in ``sys.modules``.

    Unlike ``inject_services`` (which targets ``services.<name>``), this injects
    arbitrary top-level/dotted modules — used to stand in for the uninstalled
    ``langchain_mcp_adapters`` packages that ``_load_jira_tools`` imports lazily.
    Dotted names reach ``**modules`` via ``inject_modules(**{"pkg.sub": fake})``.
    """
    saved = {}
    for full_name, fake in modules.items():
        saved[full_name] = sys.modules.get(full_name, _MISSING)
        sys.modules[full_name] = fake

    try:
        yield
    finally:
        for full_name in modules:
            if saved[full_name] is _MISSING:
                sys.modules.pop(full_name, None)
            else:
                sys.modules[full_name] = saved[full_name]


def make_fake_rag(
    *,
    answer="LEGACY ANSWER",
    exhaustive=False,
    all_records="ALL RECORDS",
    stream_tokens=None,
    llm=None,
):
    """Build a stand-in for ``services.rag_service`` with a fresh history store.

    ``stream_tokens`` (a list of strings) scripts the fake LCEL ``chain.astream``
    used by the legacy streaming path; ``llm`` overrides what ``create_llm``
    returns for the lifespan tests.
    """
    module = types.ModuleType("services.rag_service")
    module.chat_sessions = {}
    module.is_exhaustive_data_request = lambda question: exhaustive
    module.format_all_records = lambda: all_records
    module.format_context = lambda documents: "\n\n".join(
        getattr(doc, "page_content", str(doc)) for doc in documents
    )
    module.format_chat_history = lambda history: (
        "No previous messages in this session." if not history else "HISTORY"
    )
    module.answer_question = lambda question, session_id: answer
    module.create_llm = lambda: (
        llm
        if llm is not None
        else FakeToolCallingModel(messages=iter([AIMessage(content="ok")]))
    )

    def add_chat_history(history, question, answer_text):
        history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer_text},
            ]
        )

    module.add_chat_history = add_chat_history

    class _FakeChain:
        async def astream(self, payload):
            for token in stream_tokens or []:
                yield token

    module.chain = _FakeChain()
    return module


def make_fake_embed(*, docs=None, sources=None):
    """Build a stand-in for ``services.embed_service``."""
    module = types.ModuleType("services.embed_service")
    module.retrieve_from_all_sources = lambda query: list(docs or [])
    module.get_indexed_sources = lambda: list(sources or [])
    return module


def make_fake_rag_graph(*, docs=None) -> types.ModuleType:
    """Build a stand-in for ``services.rag_graph``."""
    module = types.ModuleType("services.rag_graph")
    module.retrieve = lambda question: list(docs or [])
    return module


def make_fake_mcp_adapters(*, tools=None, load_error=None):
    """Build fake ``langchain_mcp_adapters`` ``.client`` / ``.tools`` modules.

    ``MultiServerMCPClient(config).session("jira")`` returns an async context
    manager yielding a sentinel session; ``load_mcp_tools(session)`` returns
    ``tools`` (or raises ``load_error``). Returns ``(modules_by_fullname,
    FakeMultiServerMCPClient)`` — the dict is ready to splat into
    ``inject_modules(**modules)``.
    """
    sentinel_session = object()

    class FakeMultiServerMCPClient:
        instances = []

        def __init__(self, config):
            self.config = config
            FakeMultiServerMCPClient.instances.append(self)

        def session(self, name):
            @contextlib.asynccontextmanager
            async def _session_cm():
                yield sentinel_session

            return _session_cm()

    async def load_mcp_tools(session):
        assert session is sentinel_session
        if load_error is not None:
            raise load_error
        return list(tools or [])

    parent = types.ModuleType("langchain_mcp_adapters")
    client_mod = types.ModuleType("langchain_mcp_adapters.client")
    client_mod.MultiServerMCPClient = FakeMultiServerMCPClient
    tools_mod = types.ModuleType("langchain_mcp_adapters.tools")
    tools_mod.load_mcp_tools = load_mcp_tools

    modules = {
        "langchain_mcp_adapters": parent,
        "langchain_mcp_adapters.client": client_mod,
        "langchain_mcp_adapters.tools": tools_mod,
    }
    return modules, FakeMultiServerMCPClient


class FakeToolCallingModel(GenericFakeChatModel):
    """A GenericFakeChatModel that accepts ``bind_tools`` (returns itself)."""

    def bind_tools(self, tools, **kwargs):
        return self


class FakeAgent:
    """Stands in for a compiled agent; records calls and can raise on demand."""

    def __init__(self, *, result=None, exc=None):
        self.result = result
        self.exc = exc
        self.called = False

    async def ainvoke(self, payload):
        self.called = True
        if self.exc is not None:
            raise self.exc
        return self.result


class FakeStreamingAgent:
    """Stands in for a compiled agent; yields scripted ``(mode, data)`` tuples.

    A real ``create_react_agent`` cannot be driven by ``GenericFakeChatModel``
    under ``stream_mode="messages"`` (it raises "No generations found in stream"),
    so streaming is exercised by scripting the agent's ``astream`` directly — the
    same approach ``FakeAgent`` takes for ``ainvoke``.
    """

    def __init__(self, events):
        self.events = events
        self.astream_called = False

    async def astream(self, payload, stream_mode=None):
        self.astream_called = True
        for event in self.events:
            yield event


class AgentInvocationTests(unittest.TestCase):
    def _use_agent(self, agent):
        self.addCleanup(setattr, agent_service, "_agent", agent_service._agent)
        agent_service._agent = agent

    def test_agent_invokes_tool_and_records_history(self):
        """The agent calls the tool, and aanswer_question surfaces the answer,
        the tool name, and appends the turn to the shared history store."""
        captured = []

        @tool
        def echo_tool(query: str) -> str:
            """Echo the query back (test tool)."""
            captured.append(query)
            return f"ECHO:{query}"

        model = FakeToolCallingModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "echo_tool",
                                "args": {"query": "hello"},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Final answer using ECHO:hello"),
                ]
            )
        )
        agent = create_react_agent(model, [echo_tool], prompt=agent_service.SYSTEM_PROMPT)
        self._use_agent(agent)

        fake_rag = make_fake_rag()
        with inject_services(rag_service=fake_rag):
            result = asyncio.run(agent_service.aanswer_question("please echo", "s1"))

        self.assertEqual(captured, ["hello"])  # the tool actually ran
        self.assertEqual(result.answer, "Final answer using ECHO:hello")
        self.assertEqual(result.tools_used, ["echo_tool"])
        self.assertEqual(
            fake_rag.chat_sessions["s1"],
            [
                {"role": "user", "content": "please echo"},
                {"role": "assistant", "content": "Final answer using ECHO:hello"},
            ],
        )

    def test_fallback_to_legacy_when_agent_raises(self):
        """A tool-calling failure (e.g. gemma4) falls back to the legacy chain."""
        self._use_agent(FakeAgent(exc=RuntimeError("model cannot tool-call")))

        fake_rag = make_fake_rag(answer="LEGACY ANSWER")
        # assertLogs both verifies the fallback is logged and keeps the expected
        # traceback out of the test console output.
        with inject_services(rag_service=fake_rag), self.assertLogs(
            agent_service.logger, level="ERROR"
        ):
            result = asyncio.run(agent_service.aanswer_question("hello", "s2"))

        self.assertEqual(result.answer, "LEGACY ANSWER")
        self.assertEqual(result.tools_used, [])

    def test_fallback_to_legacy_when_no_agent(self):
        """With no agent built, aanswer_question uses the legacy chain."""
        self._use_agent(None)

        fake_rag = make_fake_rag(answer="LEGACY ANSWER")
        with inject_services(rag_service=fake_rag):
            result = asyncio.run(agent_service.aanswer_question("hello", "s3"))

        self.assertEqual(result.answer, "LEGACY ANSWER")
        self.assertEqual(result.tools_used, [])

    def test_exhaustive_request_bypasses_agent(self):
        """Exhaustive 'list everything' requests skip the LLM entirely."""
        never_call = FakeAgent(exc=AssertionError("agent must not be called"))
        self._use_agent(never_call)

        fake_rag = make_fake_rag(exhaustive=True, all_records="ALL RECORDS")
        with inject_services(rag_service=fake_rag):
            result = asyncio.run(agent_service.aanswer_question("list all records", "s4"))

        self.assertEqual(result.answer, "ALL RECORDS")
        self.assertEqual(result.tools_used, [])
        self.assertFalse(never_call.called)


class KnowledgeBaseToolTests(unittest.TestCase):
    def test_search_returns_formatted_context(self):
        docs = [SimpleNamespace(page_content="internship at ACME")]
        with inject_services(
            rag_graph=make_fake_rag_graph(docs=docs),
            rag_service=make_fake_rag(),
        ):
            result = agent_service.search_knowledge_base.invoke({"query": "internships"})

        self.assertEqual(result, "internship at ACME")

    def test_search_returns_message_when_no_docs(self):
        with inject_services(
            rag_graph=make_fake_rag_graph(docs=[]),
            rag_service=make_fake_rag(),
        ):
            result = agent_service.search_knowledge_base.invoke({"query": "nothing"})

        self.assertEqual(result, "No relevant context found in the knowledge base.")

    def test_list_sources_lists_files(self):
        with inject_services(embed_service=make_fake_embed(sources=["a.pdf", "b.xlsx"])):
            result = agent_service.list_knowledge_base_sources.invoke({})

        self.assertIn("a.pdf", result)
        self.assertIn("b.xlsx", result)

    def test_list_sources_when_empty(self):
        with inject_services(embed_service=make_fake_embed(sources=[])):
            result = agent_service.list_knowledge_base_sources.invoke({})

        self.assertEqual(result, "No documents are currently indexed.")


class JiraMcpConfigTests(unittest.TestCase):
    def _settings(self, **overrides):
        base = dict(
            jira_mcp_transport="stdio",
            jira_url="https://site.atlassian.net",
            jira_username="me@example.com",
            jira_api_token="secret-token",
            jira_read_only=True,
            jira_enabled_tools="",
            jira_mcp_command="mcp-atlassian",
            jira_mcp_url="",
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_stdio_config_shape(self):
        with patch.object(agent_service, "settings", self._settings()):
            config = agent_service._jira_mcp_config()

        jira = config["jira"]
        self.assertEqual(jira["command"], "mcp-atlassian")
        self.assertEqual(jira["transport"], "stdio")
        self.assertEqual(jira["args"], [])
        self.assertEqual(jira["env"]["JIRA_URL"], "https://site.atlassian.net")
        self.assertEqual(jira["env"]["JIRA_USERNAME"], "me@example.com")
        self.assertEqual(jira["env"]["JIRA_API_TOKEN"], "secret-token")
        self.assertEqual(jira["env"]["READ_ONLY_MODE"], "true")
        self.assertNotIn("ENABLED_TOOLS", jira["env"])

    def test_stdio_config_with_enabled_tools(self):
        settings = self._settings(jira_enabled_tools="jira_search,jira_get_issue")
        with patch.object(agent_service, "settings", settings):
            config = agent_service._jira_mcp_config()

        self.assertEqual(
            config["jira"]["env"]["ENABLED_TOOLS"], "jira_search,jira_get_issue"
        )

    def test_read_only_false(self):
        with patch.object(agent_service, "settings", self._settings(jira_read_only=False)):
            config = agent_service._jira_mcp_config()

        self.assertEqual(config["jira"]["env"]["READ_ONLY_MODE"], "false")

    def test_streamable_http_config(self):
        settings = self._settings(
            jira_mcp_transport="streamable_http",
            jira_mcp_url="http://localhost:9000/mcp",
        )
        with patch.object(agent_service, "settings", settings):
            config = agent_service._jira_mcp_config()

        self.assertEqual(
            config,
            {"jira": {"transport": "streamable_http", "url": "http://localhost:9000/mcp"}},
        )


class HelperTests(unittest.TestCase):
    def test_extract_answer_string_content(self):
        result = {"messages": [AIMessage(content="the answer")]}
        self.assertEqual(agent_service._extract_answer(result), "the answer")

    def test_extract_answer_list_content(self):
        result = {
            "messages": [
                AIMessage(content=[{"type": "text", "text": "part one "}, "part two"])
            ]
        }
        self.assertEqual(agent_service._extract_answer(result), "part one part two")

    def test_extract_answer_no_messages(self):
        self.assertEqual(agent_service._extract_answer({}), "")
        self.assertEqual(agent_service._extract_answer({"messages": []}), "")

    def test_extract_tools_used_dedup_and_order(self):
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search_knowledge_base", "args": {}, "id": "1", "type": "tool_call"},
                    {"name": "jira_search", "args": {}, "id": "2", "type": "tool_call"},
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search_knowledge_base", "args": {}, "id": "3", "type": "tool_call"}
                ],
            ),
            AIMessage(content="done"),
        ]
        self.assertEqual(
            agent_service._extract_tools_used(messages),
            ["search_knowledge_base", "jira_search"],
        )

    def test_extract_tools_used_empty(self):
        self.assertEqual(agent_service._extract_tools_used([]), [])
        self.assertEqual(agent_service._extract_tools_used(None), [])


class JiraMcpLoadTests(unittest.TestCase):
    """Exercise the MCP-loading path with fake adapter modules (no MCP install,
    no live Jira). Fakes are injected via ``inject_modules``."""

    def _settings(self, **overrides):
        base = dict(
            agent_enabled=True,
            jira_enabled=True,
            jira_mcp_transport="stdio",
            jira_url="https://site.atlassian.net",
            jira_username="me@example.com",
            jira_api_token="secret-token",
            jira_read_only=True,
            jira_enabled_tools="",
            jira_mcp_command="mcp-atlassian",
            jira_mcp_url="",
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def _preserve_agent_state(self):
        """Restore module-level ``_agent`` / ``_status`` after the test."""
        self.addCleanup(setattr, agent_service, "_agent", agent_service._agent)
        self.addCleanup(setattr, agent_service, "_status", agent_service._status)

    def test_load_jira_tools_returns_mcp_tools(self):
        @tool
        def jira_search(query: str) -> str:
            """Fake Jira search tool."""
            return "issues"

        modules, fake_client_cls = make_fake_mcp_adapters(tools=[jira_search])

        async def run():
            async with contextlib.AsyncExitStack() as stack:
                return await agent_service._load_jira_tools(stack)

        with inject_modules(**modules), patch.object(
            agent_service, "settings", self._settings()
        ):
            tools = asyncio.run(run())
            expected_config = agent_service._jira_mcp_config()

        self.assertEqual([t.name for t in tools], ["jira_search"])
        self.assertEqual(len(fake_client_cls.instances), 1)
        self.assertEqual(fake_client_cls.instances[0].config, expected_config)

    def test_lifespan_builds_agent_with_jira_tools(self):
        self._preserve_agent_state()

        @tool
        def jira_search(query: str) -> str:
            """Fake Jira search tool."""
            return "issues"

        modules, _ = make_fake_mcp_adapters(tools=[jira_search])
        fake_rag = make_fake_rag()

        async def run():
            async with agent_service.lifespan_agent():
                return agent_service._agent, agent_service.get_agent_status()

        with inject_services(rag_service=fake_rag), inject_modules(
            **modules
        ), patch.object(agent_service, "settings", self._settings()):
            agent, agent_status = asyncio.run(run())

        self.assertIsNotNone(agent)
        self.assertTrue(agent_status.agent_ready)
        self.assertTrue(agent_status.jira_loaded)
        self.assertIsNone(agent_status.jira_error)
        self.assertIn("jira_search", agent_status.tools)
        self.assertIn("search_knowledge_base", agent_status.tools)

    def test_lifespan_jira_failure_degrades_to_retrieval_only(self):
        self._preserve_agent_state()

        modules, _ = make_fake_mcp_adapters(
            load_error=RuntimeError("connection refused")
        )
        fake_rag = make_fake_rag()

        async def run():
            async with agent_service.lifespan_agent():
                return agent_service._agent, agent_service.get_agent_status()

        with inject_services(rag_service=fake_rag), inject_modules(
            **modules
        ), patch.object(agent_service, "settings", self._settings()), self.assertLogs(
            agent_service.logger, level="WARNING"
        ):
            agent, agent_status = asyncio.run(run())

        self.assertIsNotNone(agent)  # KB-only agent still built
        self.assertFalse(agent_status.jira_loaded)
        self.assertIsNotNone(agent_status.jira_error)
        self.assertIn("connection refused", agent_status.jira_error)
        self.assertNotIn("jira_search", agent_status.tools)

    def test_lifespan_missing_mcp_packages(self):
        self._preserve_agent_state()

        fake_rag = make_fake_rag()

        async def run():
            async with agent_service.lifespan_agent():
                return agent_service._agent, agent_service.get_agent_status()

        # Force ImportError deterministically, whether or not the adapter
        # packages happen to be installed in this environment.
        blocked = {
            "langchain_mcp_adapters": None,
            "langchain_mcp_adapters.client": None,
            "langchain_mcp_adapters.tools": None,
        }
        with inject_services(rag_service=fake_rag), patch.dict(
            sys.modules, blocked
        ), patch.object(
            agent_service, "settings", self._settings()
        ), self.assertLogs(agent_service.logger, level="WARNING"):
            agent, agent_status = asyncio.run(run())

        self.assertIsNotNone(agent)  # retrieval-only agent still built
        self.assertFalse(agent_status.jira_loaded)
        self.assertIsNotNone(agent_status.jira_error)
        self.assertIn("pip install", agent_status.jira_error)

    def test_lifespan_agent_disabled(self):
        self._preserve_agent_state()

        fake_rag = make_fake_rag()

        async def run():
            async with agent_service.lifespan_agent():
                return agent_service._agent, agent_service.get_agent_status()

        with inject_services(rag_service=fake_rag), patch.object(
            agent_service, "settings", self._settings(agent_enabled=False)
        ):
            agent, agent_status = asyncio.run(run())

        self.assertIsNone(agent)
        self.assertFalse(agent_status.agent_ready)
        self.assertFalse(agent_status.agent_enabled)


class AgentStreamingTests(unittest.TestCase):
    def _use_agent(self, agent):
        self.addCleanup(setattr, agent_service, "_agent", agent_service._agent)
        agent_service._agent = agent

    def test_astream_answer_streams_tokens_tools_and_persists(self):
        """A scripted updates(tool) + two messages(chunks) stream yields
        ``tool -> token -> token -> done``, joins the answer, reports the tool,
        and persists the turn."""
        events = [
            (
                "updates",
                {
                    "agent": {
                        "messages": [
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": "search_knowledge_base",
                                        "args": {},
                                        "id": "1",
                                        "type": "tool_call",
                                    }
                                ],
                            )
                        ]
                    }
                },
            ),
            ("messages", (AIMessageChunk(content="Hello "), {})),
            ("messages", (AIMessageChunk(content="world"), {})),
        ]
        agent = FakeStreamingAgent(events)
        self._use_agent(agent)

        fake_rag = make_fake_rag()
        with inject_services(rag_service=fake_rag):
            result = _collect(agent_service.astream_answer("q", "s1"))

        names = [name for name, _ in result]
        self.assertEqual(names, ["tool", "token", "token", "done"])
        self.assertEqual(result[0][1], {"name": "search_knowledge_base"})
        self.assertEqual(result[1][1]["text"], "Hello ")
        self.assertEqual(result[2][1]["text"], "world")

        done = result[-1][1]
        self.assertEqual(done["session_id"], "s1")
        self.assertEqual(done["answer"], "Hello world")
        self.assertEqual(done["tools_used"], ["search_knowledge_base"])
        self.assertEqual(
            fake_rag.chat_sessions["s1"],
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "Hello world"},
            ],
        )

    def test_astream_answer_fallback_when_no_agent(self):
        """With no agent built, streaming uses the legacy LCEL chain and persists."""
        self._use_agent(None)

        fake_rag = make_fake_rag(stream_tokens=["LEG", "ACY"])
        with inject_services(
            rag_service=fake_rag, embed_service=make_fake_embed(docs=[])
        ):
            result = _collect(agent_service.astream_answer("q", "s2"))

        names = [name for name, _ in result]
        self.assertEqual(names, ["token", "token", "done"])
        self.assertEqual(result[0][1]["text"], "LEG")
        self.assertEqual(result[1][1]["text"], "ACY")

        done = result[-1][1]
        self.assertEqual(done["answer"], "LEGACY")
        self.assertEqual(done["tools_used"], [])
        self.assertEqual(
            fake_rag.chat_sessions["s2"],
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "LEGACY"},
            ],
        )

    def test_astream_answer_exhaustive_bypass(self):
        """Exhaustive requests emit a single all-records token and never touch
        the agent's ``astream``."""
        agent = FakeStreamingAgent([("messages", (AIMessageChunk(content="X"), {}))])
        self._use_agent(agent)

        fake_rag = make_fake_rag(exhaustive=True, all_records="ALL RECORDS")
        with inject_services(rag_service=fake_rag):
            result = _collect(agent_service.astream_answer("list all records", "s3"))

        names = [name for name, _ in result]
        self.assertEqual(names, ["token", "done"])
        self.assertEqual(result[0][1]["text"], "ALL RECORDS")
        self.assertEqual(result[-1][1]["answer"], "ALL RECORDS")
        self.assertFalse(agent.astream_called)


if __name__ == "__main__":
    unittest.main()
