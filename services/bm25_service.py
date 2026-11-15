"""Stdlib BM25 (Okapi) over the in-memory corpus plus reciprocal rank fusion.

Lexical search complements the vector store: exact identifiers, codes and rare
words that embeddings blur still rank first. ``rank_bm25`` is not a dependency,
and the corpus is already in memory, so a few dozen lines of ``math`` and
``Counter`` cover it.
"""

import math
import re
from collections import Counter
from typing import Dict, List, Optional

from langchain_core.documents import Document

_TOKEN = re.compile(r"\w+")
K1 = 1.5
B = 0.75


def tokenize(text: str) -> List[str]:
    """Casefolded word tokens (Unicode-aware)."""
    return _TOKEN.findall(text.casefold())


def _doc_key(document: Document) -> tuple:
    """Fusion key: source + text, so a hit merges even when one retriever drops ``Document.id``."""
    return document.metadata.get("source"), document.page_content


class BM25Index:
    """Okapi BM25 index over a fixed list of documents."""

    def __init__(self, documents: List[Document]) -> None:
        self.documents = documents
        self.term_freqs = [Counter(tokenize(document.page_content)) for document in documents]
        self.doc_lens = [sum(counter.values()) for counter in self.term_freqs]
        self.avg_len = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 0.0
        doc_freqs: Counter = Counter()
        for counter in self.term_freqs:
            doc_freqs.update(counter.keys())
        total = len(documents)
        self.idf: Dict[str, float] = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5)) for term, freq in doc_freqs.items()
        }

    def _score(self, index: int, query_terms: List[str]) -> float:
        counter = self.term_freqs[index]
        norm = K1 * (1 - B + B * self.doc_lens[index] / self.avg_len) if self.avg_len else K1
        return sum(
            self.idf[term] * counter[term] * (K1 + 1) / (counter[term] + norm)
            for term in query_terms
            if term in counter
        )

    def search(self, query: str, k: int, source: Optional[str] = None) -> List[Document]:
        """Top-``k`` documents by BM25 score, optionally only those from ``source``."""
        query_terms = tokenize(query)
        if not query_terms or not self.documents:
            return []
        # term_freqs, not documents: the live corpus list may grow under a running search.
        candidates = range(len(self.term_freqs))
        if source is not None:
            candidates = [i for i in candidates if self.documents[i].metadata.get("source") == source]
        scored = [(self._score(i, query_terms), i) for i in candidates]
        # Stable sort on score only, so equal scores keep corpus order.
        scored.sort(key=lambda pair: -pair[0])
        return [self.documents[i] for score, i in scored[:k] if score > 0]


# shortcut: cache keyed on id(documents)+len(documents), rebuilt in full on the first
# query after any upload/delete (holding the live list keeps its id from being reused);
# an in-place edit that keeps the length would be missed. Upgrade to incremental
# updates from index_file/remove_file if large corpora make that rebuild slow.
_cache: Dict[tuple, BM25Index] = {}


def get_index(documents: List[Document]) -> BM25Index:
    """Return the cached BM25 index for ``documents``, rebuilding when the corpus changed."""
    key = (id(documents), len(documents))
    index = _cache.get(key)
    if index is None:  # concurrent misses just build twice; never read back after clear()
        index = BM25Index(documents)
        _cache.clear()
        _cache[key] = index
    return index


def fuse_rrf(rankings: List[List[Document]], k: int = 60) -> List[Document]:
    """Reciprocal rank fusion: sum 1/(k+rank) per document, ties broken by first appearance."""
    scores: Dict[tuple, float] = {}
    first_seen: Dict[tuple, Document] = {}
    for ranking in rankings:
        for rank, document in enumerate(ranking, start=1):
            key = _doc_key(document)
            scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
            first_seen.setdefault(key, document)
    # dict preserves insertion (first appearance) order; sort is stable.
    return [first_seen[key] for key in sorted(first_seen, key=lambda key: -scores[key])]
