"""Tests for ``rag_service.format_context`` / ``cited_sources``: numbered passages with citation locators.

``rag_service`` does ``from services import embed_service`` at module top, so the
module is executed under a private name with a fake ``embed_service`` injected
(the same trick ``test_embed_service.py`` uses), keeping the load offline and
``sys.modules`` untouched for the endpoint suite. Its import also builds the LLM
client, so the class is skipped unless conftest's offline settings are active.
"""

import importlib.util
import types
import unittest
from pathlib import Path

from langchain_core.documents import Document

from tests.unit.test_agent_service import inject_services
from tests.unit.test_embed_service import offline_only

RAG_SERVICE_PATH = Path(__file__).resolve().parents[2] / "services" / "rag_service.py"


@offline_only
class FormatContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("rag_service_under_test", RAG_SERVICE_PATH)
        cls.module = importlib.util.module_from_spec(spec)
        with inject_services(embed_service=types.ModuleType("services.embed_service")):
            spec.loader.exec_module(cls.module)

    def fmt(self, *documents) -> str:
        return self.module.format_context(list(documents))

    def test_empty_keeps_sentinel(self) -> None:
        self.assertEqual(self.fmt(), "No relevant context found.")

    def test_pdf_page_locator(self) -> None:
        doc = Document(page_content="body", metadata={"source": "file.pdf", "page": 3})
        self.assertEqual(self.fmt(doc), "[1] file.pdf (page 3)\nbody")

    def test_excel_row_locator(self) -> None:
        doc = Document(page_content="r", metadata={"source": "sheet.xlsx", "sheet": "Data", "row": 12})
        self.assertEqual(self.fmt(doc), "[1] sheet.xlsx (sheet Data, row 14)\nr")  # 0-based index 12 = sheet row 14

    def test_excel_row_range_wins_over_row(self) -> None:
        doc = Document(
            page_content="r",
            metadata={"source": "sheet.xlsx", "sheet": "Data", "row": 12, "row_range": "12-20"},
        )
        self.assertEqual(self.fmt(doc), "[1] sheet.xlsx (sheet Data, rows 14-22)\nr")

    def test_powerpoint_slide_locator(self) -> None:
        doc = Document(page_content="s", metadata={"source": "deck.pptx", "slide": 4})
        self.assertEqual(self.fmt(doc), "[1] deck.pptx (slide 4)\ns")

    def test_missing_metadata_falls_back(self) -> None:
        only_source = Document(page_content="a", metadata={"source": "notes.txt"})
        nothing = Document(page_content="b", metadata={})
        self.assertEqual(self.fmt(only_source, nothing), "[1] notes.txt\na\n\n[2] unknown source\nb")

    def test_dedupes_by_id_first_wins_and_renumbers(self) -> None:
        first = Document(page_content="v1", metadata={"source": "a.pdf", "page": 1}, id="a:1")
        dup = Document(page_content="v2", metadata={"source": "a.pdf", "page": 1}, id="a:1")
        other = Document(page_content="o", metadata={"source": "b.pdf", "page": 2}, id="b:2")
        self.assertEqual(self.fmt(first, dup, other), "[1] a.pdf (page 1)\nv1\n\n[2] b.pdf (page 2)\no")

    def test_dedupes_by_content_when_id_missing(self) -> None:
        doc = Document(page_content="same", metadata={"source": "a.pdf", "page": 1})
        self.assertEqual(self.fmt(doc, doc), "[1] a.pdf (page 1)\nsame")

    def test_numbering_continues_across_a_turn_and_reuses_numbers(self) -> None:
        a, b, c = (Document(page_content=x, metadata={"source": f"{x}.pdf"}, id=x) for x in "abc")
        passages = []
        self.addCleanup(self.module.turn_passages.reset, self.module.turn_passages.set(passages))

        self.assertEqual(self.fmt(a, b), "[1] a.pdf\na\n\n[2] b.pdf\nb")
        self.assertEqual(self.fmt(c, b), "[3] c.pdf\nc\n\n[2] b.pdf\nb")  # second search: b keeps [2]
        self.assertEqual(passages, [("a", "a.pdf"), ("b", "b.pdf"), ("c", "c.pdf")])

    def test_cited_sources_keeps_only_cited_numbers_of_the_turn(self) -> None:
        self.assertEqual(self.module.cited_sources("[1]"), [])  # no turn: nothing numbered
        turn = [("a", "a.pdf"), ("b", "b.xlsx (sheet S, row 5)"), ("c", "c.pdf (page 2)")]
        self.addCleanup(self.module.turn_passages.reset, self.module.turn_passages.set(turn))

        self.assertEqual(
            self.module.cited_sources("Per [2] and [3, 9], not 1 or items[x]."),
            [{"n": 2, "locator": "b.xlsx (sheet S, row 5)"}, {"n": 3, "locator": "c.pdf (page 2)"}],
        )


if __name__ == "__main__":
    unittest.main()
