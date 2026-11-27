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
        "retriever_k": 5,
        "retriever_max_file_chars": 12000,
        "rag_grading_enabled": True,
        "rag_max_retries": 1,
        "rag_rerank_enabled": True,
        "rag_rerank_top_n": 5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class RagGraphTests(unittest.TestCase):
    def setUp(self):
        self.prompts, self.ks = [], []  # filled by _run

    def _run(self, docs, replies, *, corpus=(), matched=(), question=None, **overrides):
        """Run ``rag_graph.retrieve`` with scripted LLM replies; return (result, queries, unused replies).

        Also leaves the LLM prompts in ``self.prompts`` and each search's ``k`` in ``self.ks``.
        """
        searches = []
        embed = make_fake_embed()
        embed.documents = list(corpus)
        embed.get_sources_matching_query = lambda query: list(matched)
        embed.retrieve_from_all_sources = lambda query, k=None: searches.append((query, k)) or list(docs)
        script = iter(AIMessage(content=reply) for reply in replies)
        llm = GenericFakeChatModel(messages=script)
        with inject_services(embed_service=embed), mock.patch.object(
            rag_graph, "_get_llm", return_value=llm
        ), mock.patch.object(rag_graph, "settings", _settings(**overrides)), mock.patch.object(
            rag_graph, "_ask", wraps=rag_graph._ask
        ) as ask:
            result = rag_graph.retrieve("remote internships?", question=question)
        self.prompts = [call.args[0] for call in ask.call_args_list]
        self.ks = [k for _, k in searches]
        return result, [query for query, _ in searches], list(script)

    def test_grade_filters_then_rerank_reorders_and_cuts(self):
        a, b, c = _doc("A"), _doc("B"), _doc("C")
        result, queries, unused = self._run(
            [a, b, c], ["yes\nno\nyes", "2, 1"], rag_rerank_top_n=1
        )
        self.assertEqual(result, [c])
        self.assertEqual(queries, ["remote internships?"])
        self.assertEqual(unused, [])

    def test_grade_and_rerank_judge_against_the_users_question(self):
        self._run([_doc("A"), _doc("B")], ["yes\nyes", "2, 1"], question="And in Cebu?")
        self.assertEqual(len(self.prompts), 2)  # grade, then rerank
        for prompt in self.prompts:
            self.assertIn("User question: And in Cebu?\nSearch query: remote internships?", prompt)

    def test_grade_takes_the_first_verdict_on_each_line(self):
        a, b, c = _doc("A"), _doc("B"), _doc("C")
        reply = "Grades:\n\n1. Yes - remote roles, no doubt\n2. no, unrelated\n3. yes"
        result, _, _ = self._run([a, b, c], [reply], rag_rerank_enabled=False)
        self.assertEqual(result, [a, c])

    def test_rerank_widens_the_search_to_three_times_top_n(self):
        cases = {(True, 5): 15, (True, 1): 5, (True, 0): None, (False, 5): None}
        for (enabled, top_n), k in cases.items():
            with self.subTest(rerank=enabled, top_n=top_n):
                self._run([], [], rag_rerank_enabled=enabled, rag_rerank_top_n=top_n)
                self.assertEqual(self.ks, [k])

    def test_retry_loop_stops_at_max_retries_and_flags_copies(self):
        docs = [_doc("A"), _doc("B")]
        replies = ["no\nno", "rewritten one", "no\nno", "rewritten two", "no\nno", "2, 1"]
        result, queries, unused = self._run(docs, replies, rag_max_retries=2)
        self.assertEqual(queries, ["remote internships?", "rewritten one", "rewritten two"])
        # Retries spent: the graded-out passages come back as reranked, flagged copies,
        # and the shared corpus documents stay unmarked.
        self.assertEqual([doc.page_content for doc in result], ["B", "A"])
        self.assertTrue(all(doc.metadata["low_relevance"] for doc in result))
        self.assertEqual([doc.metadata for doc in docs], [{"source": "a.pdf"}] * 2)
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

    def test_searched_named_file_is_graded_even_if_every_chunk_came_back(self):
        """Over the size budget the file was searched, not sent whole, however few chunks it has."""
        docs = [_doc(f"row {i}", "jobs.xlsx") for i in range(8)]
        result, _, unused = self._run(
            docs, ["yes\n" * 8, "1"], corpus=docs, matched=["jobs.xlsx"], retriever_max_file_chars=10
        )
        self.assertEqual(result, docs[:5])
        self.assertEqual(unused, [])

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

    def test_failed_rewrite_retries_the_agents_query(self):
        """A follow-up alone ("And in Cebu?") searches badly, so an empty rewrite keeps the query."""
        _, queries, _ = self._run(
            [_doc("A")], ["no", "", "yes"], question="And in Cebu?", rag_rerank_enabled=False
        )
        self.assertEqual(queries, ["remote internships?", "remote internships?"])

    def test_rewrite_keeps_last_line_of_reply(self):
        _, queries, _ = self._run(
            [_doc("A")], ["no", "Here is a better query:\n\"remote jobs\""], rag_rerank_enabled=False
        )
        self.assertEqual(queries, ["remote internships?", "remote jobs"])


if __name__ == "__main__":
    unittest.main()
