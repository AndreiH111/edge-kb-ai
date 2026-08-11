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
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from services import agent_service


_MISSING = object()


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


def make_fake_rag(*, answer="LEGACY ANSWER", exhaustive=False, all_records="ALL RECORDS"):
    """Build a stand-in for ``services.rag_service`` with a fresh history store."""
    module = types.ModuleType("services.rag_service")
    module.chat_sessions = {}
    module.is_exhaustive_data_request = lambda question: exhaustive
    module.format_all_records = lambda: all_records
    module.format_context = lambda documents: "\n\n".join(
        getattr(doc, "page_content", str(doc)) for doc in documents
    )
    module.answer_question = lambda question, session_id: answer

    def add_chat_history(history, question, answer_text):
        history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer_text},
            ]
        )

    module.add_chat_history = add_chat_history
    return module


def make_fake_embed(*, docs=None, sources=None):
    """Build a stand-in for ``services.embed_service``."""
    module = types.ModuleType("services.embed_service")
    module.retrieve_from_all_sources = lambda query: list(docs or [])
    module.get_indexed_sources = lambda: list(sources or [])
    return module


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
            embed_service=make_fake_embed(docs=docs),
            rag_service=make_fake_rag(),
        ):
            result = agent_service.search_knowledge_base.invoke({"query": "internships"})

        self.assertEqual(result, "internship at ACME")

    def test_search_returns_message_when_no_docs(self):
        with inject_services(
            embed_service=make_fake_embed(docs=[]),
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


if __name__ == "__main__":
    unittest.main()
