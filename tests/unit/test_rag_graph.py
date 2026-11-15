"""Offline tests for the self-corrective retrieval graph (``services/rag_graph.py``)."""

import unittest
from types import SimpleNamespace
from unittest import mock

from langchain_core.documents import Document
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from services import rag_graph
from tests.unit.test_agent_service import inject_services, make_fake_embed


def _doc(text, source="a.pdf"):
    return Document(page_content=text, metadata={"source": source})


def _settings(**overrides):
    values = {
        "rag_grading_enabled": True,
        "rag_max_retries": 1,
        "rag_rerank_enabled": True,
        "rag_rerank_top_n": 5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class RagGraphTests(unittest.TestCase):
    def _run(self, docs, replies, *, corpus=(), matched=(), **overrides):
        """Run ``rag_graph.retrieve`` with scripted LLM replies; return (result, queries, unused replies)."""
        queries = []
        embed = make_fake_embed()
        embed.documents = list(corpus)
        embed.get_sources_matching_query = lambda query: list(matched)
        embed.retrieve_from_all_sources = lambda query: queries.append(query) or list(docs)
        script = iter(AIMessage(content=reply) for reply in replies)
        llm = GenericFakeChatModel(messages=script)
        with inject_services(embed_service=embed), mock.patch.object(
            rag_graph, "_get_llm", return_value=llm
        ), mock.patch.object(rag_graph, "settings", _settings(**overrides)):
            result = rag_graph.retrieve("remote internships?")
        return result, queries, list(script)

    def test_grade_filters_then_rerank_reorders_and_cuts(self):
        a, b, c = _doc("A"), _doc("B"), _doc("C")
        result, queries, unused = self._run(
            [a, b, c], ["yes\nno\nyes", "2, 1"], rag_rerank_top_n=1
        )
        self.assertEqual(result, [c])
        self.assertEqual(queries, ["remote internships?"])
        self.assertEqual(unused, [])

    def test_retry_loop_stops_at_max_retries(self):
        docs = [_doc("A"), _doc("B")]
        replies = ["no\nno", "rewritten one", "no\nno", "rewritten two", "no\nno", "1, 2"]
        result, queries, unused = self._run(docs, replies, rag_max_retries=2)
        self.assertEqual(queries, ["remote internships?", "rewritten one", "rewritten two"])
        # Retries spent: graded-out passages are returned rather than nothing.
        self.assertEqual(result, docs)
        self.assertEqual(unused, [])

    def test_unparseable_replies_keep_original_order(self):
        docs = [_doc("A"), _doc("B")]
        result, _, _ = self._run(docs, ["maybe", "no idea"])
        self.assertEqual(result, docs)

    def test_whole_named_files_skip_grade_and_rerank(self):
        docs = [_doc(f"row {i}", "jobs.xlsx") for i in range(8)]
        result, _, unused = self._run(
            docs, ["no\n" * 8, "1"], corpus=docs + [_doc("other")], matched=["jobs.xlsx"]
        )
        self.assertEqual(result, docs)
        self.assertEqual(len(unused), 2)

    def test_disabled_grading_and_rerank_make_no_llm_calls(self):
        docs = [_doc(str(i)) for i in range(7)]
        result, _, unused = self._run(
            docs, ["no"], rag_grading_enabled=False, rag_rerank_enabled=False
        )
        self.assertEqual(result, docs)
        self.assertEqual(len(unused), 1)

    def test_empty_retrieval_does_not_rewrite(self):
        result, queries, unused = self._run([], ["rewritten"])
        self.assertEqual(result, [])
        self.assertEqual(queries, ["remote internships?"])
        self.assertEqual(len(unused), 1)

    def test_rewrite_keeps_last_line_of_reply(self):
        _, queries, _ = self._run(
            [_doc("A")], ["no", "Here is a better query:\n\"remote jobs\""], rag_rerank_enabled=False
        )
        self.assertEqual(queries, ["remote internships?", "remote jobs"])


if __name__ == "__main__":
    unittest.main()
