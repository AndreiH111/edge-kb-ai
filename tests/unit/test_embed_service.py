"""Retrieval routing tests for ``embed_service.retrieve_from_all_sources``.

``embed_service`` does heavy work at import time (it builds the vector store and
runs ``initialize_on_startup``), so these tests execute its code under a private
module name with ``build_vector_store`` patched to a mock. That keeps the load
offline, never touches the real ``chrome_langchain_db`` folder, and leaves the
endpoint suite's fake ``services.embed_service`` in ``sys.modules`` untouched.
The class is skipped unless the offline settings pinned by the root
``conftest.py`` are active (the legacy ``unittest discover`` runner reads the
real ``.env``), and loads the module in ``setUpClass`` so a skip runs nothing.
"""

import dataclasses
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from core.config import settings

EMBED_SERVICE_PATH = Path(__file__).resolve().parents[2] / "services" / "embed_service.py"


def _doc(source: str, text: str = "content") -> Document:
    return Document(page_content=text, metadata={"source": source})


@unittest.skipUnless(
    settings.storage_backend == "local"
    and settings.use_custom_hash_embeddings
    and settings.use_local_chunking,
    "needs the offline settings pinned by conftest.py (run with pytest)",
)
class RetrieveFromAllSourcesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("embed_service_under_test", EMBED_SERVICE_PATH)
        cls.module = importlib.util.module_from_spec(spec)
        with patch("services.vectorstore_service.build_vector_store", return_value=MagicMock()):
            spec.loader.exec_module(cls.module)

    def setUp(self) -> None:
        # Fresh store per test: the load-time mock already recorded the startup search.
        self.store = MagicMock()
        self.store.similarity_search.return_value = []
        self.module.vector_store = self.store

    def test_no_file_named_runs_one_global_search(self) -> None:
        """Short words and chat-history labels no longer pin unrelated files."""
        self.module.documents = [_doc("layout.pptx"), _doc("information.pdf"), _doc("user_guide.pdf")]
        query = "User: hi\nAssistant: hello\nCurrent question: Can you share the policy for remote work?"

        self.module.retrieve_from_all_sources(query)

        self.store.similarity_search.assert_called_once_with(query, k=self.module.settings.hybrid_candidates)

    def test_small_named_file_is_returned_whole(self) -> None:
        internships = [_doc("Internships_2024.xlsx", f"row {index}") for index in range(3)]
        self.module.documents = internships + [_doc("handbook.pdf")]

        result = self.module.retrieve_from_all_sources("Which internships are remote?")

        self.assertEqual(result, internships)
        self.store.similarity_search.assert_not_called()

    def test_large_named_file_is_searched_not_dumped(self) -> None:
        self.module.documents = [_doc("Internships_2024.xlsx", "x" * 50), _doc("handbook.pdf")]
        tight = dataclasses.replace(self.module.settings, retriever_max_file_chars=10)

        with patch.object(self.module, "settings", tight):
            self.module.retrieve_from_all_sources("Which internships are remote?")

        self.store.similarity_search.assert_called_once_with(
            "Which internships are remote?",
            k=tight.hybrid_candidates,
            filter={"source": "Internships_2024.xlsx"},
        )

    def test_global_search_fuses_bm25_and_vector_results(self) -> None:
        doc_a = _doc("a.pdf", "unrelated vector hit")
        doc_b = _doc("b.pdf", "habitat notes")
        doc_c = _doc("c.pdf", "quokka habitat on the island")
        self.module.documents = [doc_b, doc_c]
        self.store.similarity_search.return_value = [doc_a, doc_b]

        result = self.module.retrieve_from_all_sources("quokka habitat")

        # B is in both rankings so it fuses highest; C (BM25 only) and A (vector only) follow.
        self.assertEqual(result[0], doc_b)
        self.assertCountEqual(result[1:], [doc_a, doc_c])

    def test_hybrid_disabled_returns_vector_results_untouched(self) -> None:
        doc_a, doc_b = _doc("a.pdf", "first"), _doc("b.pdf", "second")
        self.module.documents = [doc_b, _doc("c.pdf", "quokka")]
        self.store.similarity_search.return_value = [doc_a, doc_b]
        classic = dataclasses.replace(self.module.settings, hybrid_search_enabled=False)

        with patch.object(self.module, "settings", classic):
            result = self.module.retrieve_from_all_sources("quokka")

        self.assertEqual(result, [doc_a, doc_b])
        self.store.similarity_search.assert_called_once_with("quokka", k=classic.retriever_k)


if __name__ == "__main__":
    unittest.main()
