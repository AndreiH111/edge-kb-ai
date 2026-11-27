"""Self-corrective retrieval graph used by the ``search_knowledge_base`` tool.

    retrieve -> (named small files sent whole) END
             \\-> grade -> (nothing relevant & retries left) rewrite -> retrieve
                        \\-> rerank -> END

``grade``, ``rewrite`` and ``rerank`` each make one non-streamed LLM call and
degrade to "keep what we have" on any LLM or parse failure, so the tool never
breaks. ``grade`` and ``rerank`` judge passages against the user's question and
the search query. If grading rejects every passage and the retries are spent,
the last search comes back as copies flagged ``metadata["low_relevance"]``.
``embed_service`` / ``rag_service`` are imported lazily (see CLAUDE.md).
"""

import functools
import logging
import re
from typing import List, Optional, TypedDict

from langchain_core.documents import Document
from langgraph.constants import TAG_NOSTREAM
from langgraph.graph import END, START, StateGraph

from core.config import settings

logger = logging.getLogger(__name__)


class RetrievalState(TypedDict, total=False):
    """Graph state. ``question`` is the user's wording, ``query`` the current search text."""

    question: str
    query: str
    documents: List[Document]
    relevant: List[Document]
    retries: int


@functools.cache
def _get_llm():
    """Build the internal LLM once (a failed build is not cached, so it is retried)."""
    from services import rag_service

    return rag_service.create_llm()


def _ask(prompt: str) -> str:
    """One internal LLM call, never streamed to clients; ``""`` on failure."""
    try:
        return _get_llm().invoke(prompt, config={"tags": [TAG_NOSTREAM]}).text
    except Exception:
        logger.warning("Retrieval LLM call failed; continuing without it", exc_info=True)
        return ""


def _numbered(documents: List[Document]) -> str:
    """Passages as ``[1] text`` blocks for the grade/rerank prompts."""
    return "\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(documents, 1))


def _retrieve(state: RetrievalState) -> RetrievalState:
    """Search the knowledge base with the current query."""
    from services import embed_service

    # Rerank keeps only its top N, so give grade and rerank a bigger pool to choose from
    # (like any failed step, a failed rerank call passes the whole pool on).
    top_n = settings.rag_rerank_top_n
    k = max(settings.retriever_k, 3 * top_n) if settings.rag_rerank_enabled and top_n > 0 else None
    return {"documents": embed_service.retrieve_from_all_sources(state["query"], k=k)}


def _after_retrieve(state: RetrievalState) -> str:
    """Named small files arrive whole so filtered questions see every row: don't grade or cut them."""
    from services import embed_service

    # shortcut: re-derives retrieve_from_all_sources' whole-file branch (same source match, same
    # character budget); have embed_service report the branch if the two ever drift apart.
    documents = state["documents"]
    pinned = embed_service.pinned_source.get()
    if pinned:
        matched = {pinned} if documents else set()
    else:
        matched = set(embed_service.get_sources_matching_query(state["query"])) if documents else set()
    size = sum(len(doc.page_content) for doc in embed_service.documents if doc.metadata.get("source") in matched)
    return END if matched and size <= settings.retriever_max_file_chars else "grade"


def _grade(state: RetrievalState) -> RetrievalState:
    """Keep only passages the LLM marks relevant (one batched call); keep all on doubt."""
    documents = state["documents"]
    if not settings.rag_grading_enabled or not documents:
        return {"relevant": documents}

    reply = _ask(
        "Grade each passage: does it help answer the user's question? The search query is what "
        "was searched for; it may spell out what a short follow-up question refers to.\n"
        f"User question: {state['question']}\nSearch query: {state['query']}\n\n"
        f"Passages:\n{_numbered(documents)}\n\n"
        f"Reply with exactly {len(documents)} lines, one per passage in order, "
        'each line a single word: "yes" or "no". No other text.'
    )
    # The first yes/no on each line, so "2. no - wrong city, yes it is a job" counts once.
    lines = reply.lower().splitlines()
    verdicts = [match[1] for line in lines if (match := re.search(r"\b(yes|no)\b", line))]
    if len(verdicts) != len(documents):
        return {"relevant": documents}
    return {"relevant": [doc for doc, verdict in zip(documents, verdicts) if verdict == "yes"]}


def _route(state: RetrievalState) -> str:
    """Rewrite and retry only when grading rejected every passage and retries remain."""
    if state["documents"] and not state["relevant"] and state["retries"] < settings.rag_max_retries:
        return "rewrite"
    return "rerank"


def _rewrite(state: RetrievalState) -> RetrievalState:
    """Ask the LLM for a better search query, then retry retrieval."""
    query = _ask(
        "Rewrite the search as a better query for a document knowledge base (key terms and "
        "synonyms, no filler). The previous query may spell out what a short follow-up "
        "question refers to. Reply with only the new query.\n"
        f"User question: {state['question']}\n"
        f"Previous query (found nothing relevant): {state['query']}"
    ).strip()
    # Drop any preamble ("Here is a better query:") by keeping the last line.
    query = query.splitlines()[-1].strip().strip('"') if query else ""
    logger.info("No relevant passages for %r; retrying with %r", state["query"], query)
    # Fall back to the agent's query: a follow-up question alone ("And in Cebu?") searches badly.
    return {"query": query or state["query"], "retries": state["retries"] + 1}


def _rerank(state: RetrievalState) -> RetrievalState:
    """Listwise LLM rerank by index, then keep the top N; untouched on LLM or parse failure."""
    documents = state["relevant"]
    if not documents:
        # Retries spent and every passage graded out: flag copies, never the shared corpus
        # objects, so the tool can warn the model instead of it answering from them.
        documents = [
            doc.model_copy(update={"metadata": {**doc.metadata, "low_relevance": True}})
            for doc in state["documents"]
        ]
    if settings.rag_rerank_enabled and len(documents) > 1:
        order = _ask(
            "Rank the passages by how well they answer the user's question, most relevant first. "
            "The search query is what was searched for.\n"
            f"User question: {state['question']}\nSearch query: {state['query']}\n\n"
            f"Passages:\n{_numbered(documents)}\n\n"
            "Reply with only the passage numbers as a comma-separated list, e.g. 2, 1, 3."
        )
        ranked = [int(i) - 1 for i in re.findall(r"\d+", order)]
        picked = list(dict.fromkeys(i for i in ranked if 0 <= i < len(documents)))
        if picked:
            rest = [i for i in range(len(documents)) if i not in picked]
            documents = [documents[i] for i in picked + rest]
            if settings.rag_rerank_top_n > 0:
                documents = documents[: settings.rag_rerank_top_n]
    return {"documents": documents}


def _build() -> StateGraph:
    """Wire the nodes as drawn in the module docstring."""
    graph = StateGraph(RetrievalState)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("grade", _grade)
    graph.add_node("rewrite", _rewrite)
    graph.add_node("rerank", _rerank)
    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges("retrieve", _after_retrieve, ["grade", END])
    graph.add_conditional_edges("grade", _route, ["rewrite", "rerank"])
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("rerank", END)
    return graph


# checkpointer=False: never inherit the agent's checkpointer when run inside a tool.
_graph = _build().compile(checkpointer=False)


def retrieve(query: str, question: Optional[str] = None) -> List[Document]:
    """Run the retrieval graph for ``query`` synchronously and return the final passages.

    ``question`` is the user's own wording (default ``query``); grade and rerank see both.
    """
    return _graph.invoke({"question": question or query, "query": query, "retries": 0})["documents"]
