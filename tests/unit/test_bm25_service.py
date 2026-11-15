"""Tests for the stdlib BM25 index and reciprocal rank fusion."""

import unittest

from langchain_core.documents import Document

from services.bm25_service import BM25Index, fuse_rrf, get_index


def _doc(doc_id: str, text: str, source: str = "a.xlsx") -> Document:
    return Document(id=doc_id, page_content=text, metadata={"source": source})


class BM25IndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.docs = [
            _doc("1", "Zebra stripes and savanna grass"),
            _doc("2", "The quokka lives on Rottnest Island; quokka habitat is scrub", source="b.pdf"),
            _doc("3", "Lemur troops of Madagascar"),
            _doc("4", "Island ferry timetable"),
        ]
        self.index = BM25Index(self.docs)

    def test_exact_term_ranks_matching_doc_first(self) -> None:
        result = self.index.search("quokka habitat", k=3)

        self.assertEqual(result[0].id, "2")
        # "island" alone also matches doc 4; docs without any query term are dropped.
        self.assertNotIn("1", [document.id for document in result])

    def test_source_filter_excludes_other_files(self) -> None:
        self.assertEqual([d.id for d in self.index.search("island", k=5, source="a.xlsx")], ["4"])
        self.assertEqual(self.index.search("island", k=5, source="missing.pdf"), [])

    def test_empty_corpus_and_empty_query_return_nothing(self) -> None:
        self.assertEqual(BM25Index([]).search("anything", k=5), [])
        self.assertEqual(self.index.search("!!!", k=5), [])

    def test_get_index_caches_until_corpus_changes(self) -> None:
        cached = get_index(self.docs)
        self.assertIs(cached, get_index(self.docs))
        self.docs.append(_doc("5", "new quokka row"))
        # A search on the stale index (corpus grew under it) must not crash or see the new row.
        self.assertNotIn("5", [d.id for d in cached.search("quokka", k=5)])
        self.assertEqual(len(get_index(self.docs).documents), 5)

    def test_tokenizer_keeps_non_ascii_words(self) -> None:
        self.assertEqual(BM25Index([_doc("z", "Büro in Zürich")]).search("zürich", k=1)[0].id, "z")


class FuseRrfTests(unittest.TestCase):
    def test_doc_in_both_rankings_wins_and_is_deduped(self) -> None:
        a, b, c = _doc("a", "a"), _doc("b", "b"), _doc("c", "c")

        fused = fuse_rrf([[c, b], [a, b]])

        self.assertEqual([d.id for d in fused], ["b", "c", "a"])

    def test_merges_same_chunk_when_one_ranking_lacks_ids(self) -> None:
        first = _doc("chunk-1", "same text")
        second = Document(page_content="same text", metadata={"source": "a.xlsx"})

        self.assertEqual(len(fuse_rrf([[first], [second]])), 1)

    def test_empty_rankings(self) -> None:
        self.assertEqual(fuse_rrf([[], []]), [])


if __name__ == "__main__":
    unittest.main()
