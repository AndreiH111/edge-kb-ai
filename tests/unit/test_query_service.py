"""Tests for the conversational query rewriting + multi-query retrieval helpers.

``query_service`` imports ``rag_service`` / ``embed_service`` lazily, so these
tests pass scripted ``GenericFakeChatModel`` instances directly and inject a fake
``embed_service`` via ``inject_services`` — the real modules are never imported.
"""

import dataclasses
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from services import query_service
from tests.unit.test_agent_service import inject_services, make_fake_embed


def fake_llm(*replies):
    return GenericFakeChatModel(messages=iter(AIMessage(content=reply) for reply in replies))


class RaisingLLM:
    def invoke(self, *args, **kwargs):
        raise RuntimeError("llm down")


def with_settings(**overrides):
    return patch.object(
        query_service, "settings", dataclasses.replace(query_service.settings, **overrides)
    )


class CondenseQuestionTests(unittest.TestCase):
    def test_skipped_without_history(self):
        llm = RaisingLLM()
        self.assertEqual(query_service.condense_question("q", "", llm), "q")
        self.assertEqual(query_service.condense_question("q", query_service.NO_HISTORY, llm), "q")

    def test_skipped_when_disabled(self):
        with with_settings(query_rewrite_enabled=False):
            self.assertEqual(
                query_service.condense_question("q", "User: hi", RaisingLLM()), "User: hi\nCurrent question: q"
            )

    def test_uses_llm_output(self):
        llm = fake_llm("  What was ACME revenue in 2024?  ")
        with with_settings(query_rewrite_enabled=True):
            result = query_service.condense_question("and in 2024?", "User: ACME revenue 2023?", llm)
        self.assertEqual(result, "What was ACME revenue in 2024?")


class ExpandQueriesTests(unittest.TestCase):
    def test_count_one_returns_question_only(self):
        self.assertEqual(query_service.expand_queries("q", RaisingLLM(), 1), ["q"])

    def test_dedupes_and_strips_list_markers(self):
        llm = fake_llm("1. ACME 2024 revenue\n- acme 2024 REVENUE\n\n2) Q\n* 2024 sales at ACME\nextra")
        self.assertEqual(
            query_service.expand_queries("Q", llm, 3), ["Q", "ACME 2024 revenue", "2024 sales at ACME"]
        )


class RetrieveMultiTests(unittest.TestCase):
    def test_dedupes_by_id_preserving_order(self):
        a = Document(id="f:s:1", page_content="A")
        b = Document(id="f:s:2", page_content="B")
        c = Document(page_content="C")
        calls = []

        def retrieve(query):
            calls.append(query)
            return {"q1": [a, b], "q2": [Document(id="f:s:2", page_content="B'"), c, c]}[query]

        fake_embed = make_fake_embed()
        fake_embed.retrieve_from_all_sources = retrieve
        with inject_services(embed_service=fake_embed):
            result = query_service.retrieve_multi(["q1", "q2"])

        self.assertEqual(calls, ["q1", "q2"])
        self.assertEqual([doc.page_content for doc in result], ["A", "B", "C"])


class BuildRetrievalDocumentsTests(unittest.TestCase):
    def test_llm_failure_falls_back_to_history_query(self):
        queries = []
        fake_embed = make_fake_embed()
        fake_embed.retrieve_from_all_sources = lambda query: queries.append(query) or []
        with (
            with_settings(query_rewrite_enabled=True, multi_query_count=1),
            inject_services(embed_service=fake_embed),
        ):
            self.assertEqual(
                query_service.build_retrieval_documents("and in 2024?", "User: hi", RaisingLLM()), []
            )

        self.assertEqual(queries, ["User: hi\nCurrent question: and in 2024?"])

    def test_expansion_failure_keeps_condensed_question(self):
        queries = []
        llm = fake_llm("STANDALONE")  # second (expansion) call finds the script exhausted and raises
        fake_embed = make_fake_embed()
        fake_embed.retrieve_from_all_sources = lambda query: queries.append(query) or []
        with (
            with_settings(query_rewrite_enabled=True, multi_query_count=3),
            inject_services(embed_service=fake_embed),
        ):
            query_service.build_retrieval_documents("and in 2024?", "User: hi", llm)

        self.assertEqual(queries, ["STANDALONE"])

    def test_condensed_question_drives_retrieval(self):
        queries = []
        llm = fake_llm("STANDALONE", "PARAPHRASE")
        fake_embed = make_fake_embed()
        fake_embed.retrieve_from_all_sources = lambda query: queries.append(query) or []
        with (
            with_settings(query_rewrite_enabled=True, multi_query_count=2),
            inject_services(embed_service=fake_embed),
        ):
            query_service.build_retrieval_documents("and in 2024?", "User: hi", llm)

        self.assertEqual(queries, ["STANDALONE", "PARAPHRASE"])


if __name__ == "__main__":
    unittest.main()
