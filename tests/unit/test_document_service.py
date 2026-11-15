"""Docling loader tests, offline: a fake ``docling`` package stands in for the real one."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from services import document_service


def _fake_docling(pages):
    """Fake ``docling`` modules whose converter returns a doc with the scripted ``{page_no: markdown}``."""
    doc = SimpleNamespace(pages={number: object() for number in pages})
    doc.export_to_markdown = lambda page_no, image_placeholder: pages[page_no]
    converter_cls = MagicMock()
    converter_cls.return_value.convert.return_value = SimpleNamespace(document=doc, status="success", errors=[])
    input_format = SimpleNamespace(PDF="pdf", PPTX="pptx")
    return converter_cls, {
        "docling": MagicMock(),
        "docling.datamodel": MagicMock(),
        "docling.datamodel.base_models": SimpleNamespace(InputFormat=input_format),
        "docling.document_converter": SimpleNamespace(DocumentConverter=converter_cls),
    }


class LoadFileTests(unittest.TestCase):
    def setUp(self) -> None:
        document_service._CONVERTER = None
        self.addCleanup(setattr, document_service, "_CONVERTER", None)

    def _load(self, name, pages):
        converter_cls, modules = _fake_docling(pages)
        with patch.dict(sys.modules, modules):
            return converter_cls, document_service.load_file(Path(name))

    def test_pdf_pages_become_documents_and_empty_pages_are_skipped(self) -> None:
        converter_cls, documents = self._load("report.pdf", {2: "| a | b |", 1: " Intro \n", 3: "  \n"})

        converter_cls.assert_called_once_with(allowed_formats=["pdf", "pptx"])
        self.assertEqual([d.id for d in documents], ["report.pdf:page:1", "report.pdf:page:2"])
        self.assertEqual(documents[0].metadata, {"source": "report.pdf", "file_type": "pdf", "page": 1})
        self.assertEqual(documents[0].page_content, "Source file: report.pdf\nFile type: PDF\nPage: 1\nIntro")

    def test_pptx_slides_use_slide_ids_and_metadata(self) -> None:
        _, documents = self._load("deck.pptx", {1: "# Title", 2: "Body"})

        self.assertEqual([d.id for d in documents], ["deck.pptx:slide:1", "deck.pptx:slide:2"])
        self.assertEqual(documents[1].metadata, {"source": "deck.pptx", "file_type": "powerpoint", "slide": 2})
        self.assertEqual(documents[1].page_content, "Source file: deck.pptx\nFile type: PowerPoint\nSlide: 2\nBody")

    def test_load_documents_skips_a_file_that_fails_to_convert(self) -> None:
        paths = [Path("bad.pdf"), Path("good.pdf")]
        loaded = [Document(page_content="ok", id="good.pdf:page:1")]
        with (
            patch.object(document_service, "get_supported_files", return_value=paths),
            patch.object(Path, "is_file", return_value=True),
            patch.object(document_service, "load_file", side_effect=[RuntimeError("encrypted"), loaded]),
        ):
            self.assertEqual(document_service.load_documents(), loaded)

    def test_is_document_file(self) -> None:
        self.assertTrue(document_service.is_document_file(Path("A.PDF")))
        self.assertTrue(document_service.is_document_file(Path("deck.pptx")))
        self.assertFalse(document_service.is_document_file(Path("sheet.xlsx")))


if __name__ == "__main__":
    unittest.main()
