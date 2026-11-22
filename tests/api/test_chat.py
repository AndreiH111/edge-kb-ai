"""Endpoint tests for /ai/chat, /ai/chat/stream, and /ai/agent-status.

The agent core is driven per-test by monkeypatching ``agent_service`` functions
(``aanswer_question``, ``astream_answer``, ``get_agent_status``); the routers call
them via attribute access, so patching the module attribute takes effect.
"""

import json
from collections import defaultdict
from types import SimpleNamespace

import pytest
from langchain_core.chat_history import InMemoryChatMessageHistory

from services import agent_service


def _parse_sse(text: str):
    """Parse an SSE body into a list of ``(event, data)`` tuples."""
    frames = []
    for block in text.strip().split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event = data = None
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data = line[len("data:") :].strip()
        frames.append((event, json.loads(data) if data is not None else None))
    return frames


def _fake_answer(answer="Hello from the agent.", tools_used=None):
    async def _aanswer(question, session_id, source=None):
        return agent_service.AgentResult(
            answer=answer, tools_used=list(tools_used or [])
        )

    return _aanswer


class TestChat:
    def test_answer_via_body(self, client, monkeypatch):
        monkeypatch.setattr(
            agent_service,
            "aanswer_question",
            _fake_answer(tools_used=["search_knowledge_base"]),
        )
        resp = client.post(
            "/ai/chat", json={"question": "hi", "session_id": "sess-body"}
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "sess-body"
        assert body["question"] == "hi"
        assert body["answer"] == "Hello from the agent."
        assert body["tools_used"] == ["search_knowledge_base"]
        assert resp.cookies.get("ai_session_id") == "sess-body"
        assert "httponly" in resp.headers["set-cookie"].lower()

    def test_answer_via_query_param(self, client, monkeypatch):
        monkeypatch.setattr(agent_service, "aanswer_question", _fake_answer())
        resp = client.post("/ai/chat?question=hello&session_id=sess-q")

        assert resp.status_code == 200
        body = resp.json()
        assert body["question"] == "hello"
        assert body["session_id"] == "sess-q"

    def test_query_session_beats_body(self, client, monkeypatch):
        """Query param session id wins over the body session id."""
        monkeypatch.setattr(agent_service, "aanswer_question", _fake_answer())
        resp = client.post(
            "/ai/chat?session_id=from-query",
            json={"question": "hi", "session_id": "from-body"},
        )

        assert resp.status_code == 200
        assert resp.json()["session_id"] == "from-query"

    def test_cookie_session_roundtrip(self, client, monkeypatch):
        """With no query/body session id, the ai_session_id cookie is honored."""
        monkeypatch.setattr(agent_service, "aanswer_question", _fake_answer())
        client.cookies.set("ai_session_id", "sess-cookie")
        resp = client.post("/ai/chat", json={"question": "hi"})

        assert resp.status_code == 200
        assert resp.json()["session_id"] == "sess-cookie"

    def test_generates_session_when_absent(self, client, monkeypatch):
        """A fresh uuid session is minted and set as an httponly cookie."""
        monkeypatch.setattr(agent_service, "aanswer_question", _fake_answer())
        resp = client.post("/ai/chat", json={"question": "hi"})

        assert resp.status_code == 200
        session_id = resp.json()["session_id"]
        assert isinstance(session_id, str) and len(session_id) >= 32
        assert resp.cookies.get("ai_session_id") == session_id

    def test_empty_tools_used_collapses_to_null(self, client, monkeypatch):
        """``tools_used=[]`` renders as JSON null (chat.py: ``or None``)."""
        monkeypatch.setattr(
            agent_service, "aanswer_question", _fake_answer(tools_used=[])
        )
        resp = client.post("/ai/chat", json={"question": "hi"})

        assert resp.status_code == 200
        assert resp.json()["tools_used"] is None

    def test_missing_question_returns_400(self, client, monkeypatch):
        monkeypatch.setattr(agent_service, "aanswer_question", _fake_answer())
        resp = client.post("/ai/chat")

        assert resp.status_code == 400

    def test_empty_query_question_returns_422(self, client):
        """``question`` query param has ``min_length=1``."""
        resp = client.post("/ai/chat?question=")

        assert resp.status_code == 422

    def test_empty_body_question_returns_422(self, client):
        """``ChatRequest.question`` has ``min_length=1``."""
        resp = client.post("/ai/chat", json={"question": ""})

        assert resp.status_code == 422

    def test_filename_is_forwarded_as_source(self, client, monkeypatch):
        seen = {}

        async def _aanswer(question, session_id, source=None):
            seen["source"] = source
            return agent_service.AgentResult(answer="ok", tools_used=[])

        monkeypatch.setattr(agent_service, "aanswer_question", _aanswer)
        resp = client.post(
            "/ai/chat", json={"question": "hi", "filename": "Resort1.xlsx"}
        )

        assert resp.status_code == 200
        assert seen["source"] == "Resort1.xlsx"


class TestListEverythingShortcut:
    """Drives the real ``aanswer_question`` (no agent) to check which questions
    skip the LLM and dump every record (``format_all_records``: "Found N ...")."""

    @pytest.mark.parametrize(
        "question, dumps",
        [
            ("List all records", True),
            ("Show me every row", True),
            ("What are all the internships?", True),
            ("What data is available?", True),
            ("list all intenships", True),
            ("What is the overall total in the sales sheet?", False),
            ("Which department grows fastest? Really curious", False),
            ("Summarize the full PDF please", False),
            ("List all documents", False),
            ("Show me all records where status is Open", False),
            ("List all Jira issues about data migration", False),
            ("Print the full row for ID 42", False),
        ],
    )
    def test_only_pure_list_requests_dump(self, client, monkeypatch, question, dumps):
        # Imported here, after ``client`` injected the fake embed_service: a
        # top-level import would load the real one.
        from services import rag_service

        monkeypatch.setattr(agent_service, "_agent", None)
        monkeypatch.setattr(rag_service, "answer_question", lambda q, s: "NORMAL ANSWER")
        resp = client.post("/ai/chat", json={"question": question})

        assert resp.status_code == 200
        assert resp.json()["answer"].startswith("Found ") is dumps


class TestChatHistory:
    def test_history_trims_to_limit_starting_on_user(self, client, monkeypatch):
        from services import rag_service  # after ``client`` injected the fake embed_service

        monkeypatch.setattr(
            rag_service,
            "settings",
            SimpleNamespace(max_chat_history_messages=3, max_chat_history_content_chars=100),
        )
        history = InMemoryChatMessageHistory()
        assert rag_service.format_chat_history(history) == "No previous messages in this session."

        rag_service.add_chat_history(history, "q1", "a1")
        rag_service.add_chat_history(history, "q2", "a2")

        # Last 3 would be [a1, q2, a2]; start_on="human" drops the leading AI turn.
        assert rag_service.format_chat_history(history) == "User: q2\nAssistant: a2"

    def test_client_history_replaces_session_memory(self, client, monkeypatch):
        """A reopened chat sends its stored turns; they replace whatever the server held."""
        from services import rag_service  # after ``client`` injected the fake embed_service

        seen = {}

        def fake_answer(question, session_id):
            seen["history"] = rag_service.format_chat_history(rag_service.get_history(session_id))
            return "ok"

        monkeypatch.setattr(rag_service, "chat_sessions", defaultdict(InMemoryChatMessageHistory))
        monkeypatch.setattr(agent_service, "_agent", None)
        monkeypatch.setattr(rag_service, "answer_question", fake_answer)
        rag_service.add_chat_history(rag_service.get_history("chat-1"), "stale q", "stale a")

        resp = client.post("/ai/chat", json={
            "question": "can you paraphrase your answer",
            "session_id": "chat-1",
            "history": [{"question": "How do I request access?", "answer": "Use User Management."}],
        })

        assert resp.status_code == 200
        assert seen["history"] == "User: How do I request access?\nAssistant: Use User Management."


class TestChatStream:
    def test_stream_frames_order_and_done(self, client, monkeypatch):
        async def fake_astream(question, session_id, source=None):
            yield ("tool", {"name": "search_knowledge_base"})
            yield ("token", {"text": "Hel"})
            yield ("token", {"text": "lo"})
            yield (
                "done",
                {
                    "session_id": session_id,
                    "answer": "Hello",
                    "tools_used": ["search_knowledge_base"],
                },
            )

        monkeypatch.setattr(agent_service, "astream_answer", fake_astream)
        resp = client.post(
            "/ai/chat/stream", json={"question": "hi", "session_id": "s-stream"}
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers["cache-control"] == "no-cache"
        assert resp.headers["x-accel-buffering"] == "no"
        assert resp.cookies.get("ai_session_id") == "s-stream"

        frames = _parse_sse(resp.text)
        assert [name for name, _ in frames] == ["tool", "token", "token", "done"]
        assert frames[0][1] == {"name": "search_knowledge_base"}
        assert frames[1][1]["text"] == "Hel"
        assert frames[2][1]["text"] == "lo"

        done = frames[-1][1]
        assert done["session_id"] == "s-stream"
        assert done["answer"] == "Hello"
        assert done["tools_used"] == ["search_knowledge_base"]

    def test_stream_surfaces_error_frame(self, client, monkeypatch):
        async def fake_astream(question, session_id, source=None):
            yield ("token", {"text": "partial"})
            yield ("error", {"message": "The agent failed while generating the answer."})

        monkeypatch.setattr(agent_service, "astream_answer", fake_astream)
        resp = client.post("/ai/chat/stream", json={"question": "hi"})

        assert resp.status_code == 200
        frames = _parse_sse(resp.text)
        names = [name for name, _ in frames]
        assert "error" in names
        assert frames[-1][0] == "error"

    def test_stream_missing_question_returns_400(self, client, monkeypatch):
        """The no-question guard runs before streaming starts."""
        called = False

        async def fake_astream(question, session_id, source=None):
            nonlocal called
            called = True
            yield ("done", {})

        monkeypatch.setattr(agent_service, "astream_answer", fake_astream)
        resp = client.post("/ai/chat/stream")

        assert resp.status_code == 400
        assert called is False


class TestAgentStatus:
    def test_default_shape(self, client):
        resp = client.get("/ai/agent-status")

        assert resp.status_code == 200
        body = resp.json()
        assert set(body) == {
            "agent_enabled",
            "agent_ready",
            "jira_enabled",
            "jira_loaded",
            "tool_count",
            "tools",
            "jira_error",
        }
        assert body["tools"] == []
        assert body["tool_count"] == 0
        assert body["jira_error"] is None

    def test_full_field_mapping(self, client, monkeypatch):
        status = agent_service.AgentStatus(
            agent_enabled=True,
            agent_ready=True,
            jira_enabled=True,
            jira_loaded=True,
            tool_count=3,
            tools=[
                "search_knowledge_base",
                "list_knowledge_base_sources",
                "jira_search",
            ],
            jira_error="connection refused",
        )
        monkeypatch.setattr(agent_service, "get_agent_status", lambda: status)
        resp = client.get("/ai/agent-status")

        assert resp.status_code == 200
        body = resp.json()
        assert body["agent_ready"] is True
        assert body["jira_loaded"] is True
        assert body["tool_count"] == 3
        assert body["tools"] == [
            "search_knowledge_base",
            "list_knowledge_base_sources",
            "jira_search",
        ]
        assert body["jira_error"] == "connection refused"
