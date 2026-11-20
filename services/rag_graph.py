"""Self-corrective retrieval graph used by the ``search_knowledge_base`` tool.

    retrieve -> (named small files sent whole) END
             \\-> grade -> (nothing relevant & retries left) rewrite -> retrieve
                        \\-> rerank -> END

``grade``, ``rewrite`` and ``rerank`` each make one non-streamed LLM call and
degrade to "keep what we have" on any LLM or parse failure, so the tool never
breaks. ``embed_service`` / ``rag_service`` are imported lazily (see CLAUDE.md).
"""

import functools
import logging
import re
from typing import List, TypedDict

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

    return {"documents": embed_service.retrieve_from_all_sources(state["query"])}


def _after_retrieve(state: RetrievalState) -> str:
    """Named small files arrive whole so filtered questions see every row: don't grade or cut them."""
    from services import embed_service

    # shortcut: infers retrieve_from_all_sources' whole-file branch by re-matching sources and
    # counting their chunks; have embed_service report the branch if this ever misfires.
    documents = state["documents"]
    pinned = embed_service.pinned_source.get()
    if pinned:
        matched = {pinned} if documents else set()
    else:
        matched = set(embed_service.get_sources_matching_query(state["query"])) if documents else set()
    named = sum(doc.metadata.get("source") in matched for doc in embed_service.documents)
    return END if matched and len(documents) == named else "grade"


def _grade(state: RetrievalState) -> RetrievalState:
    """Keep only passages the LLM marks relevant (one batched call); keep all on doubt."""
    documents = state["documents"]
    if not settings.rag_grading_enabled or not documents:
        return {"relevant": documents}

    verdicts = re.findall(
        r"\b(yes|no)\b",
        _ask(
            "Grade each passage for relevance to the question.\n"
            f"Question: {state['question']}\n\nPassages:\n{_numbered(documents)}\n\n"
            f"Reply with exactly {len(documents)} lines, one per passage in order, "
            'each line a single word: "yes" or "no". No other text.'
        ).lower(),
    )
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
        "Rewrite the question as a better search query for a document knowledge base "
        "(key terms and synonyms, no filler). Reply with only the new query.\n"
        f"Question: {state['question']}\n"
        f"Previous query (found nothing relevant): {state['query']}"
    ).strip()
    # Drop any preamble ("Here is a better query:") by keeping the last line.
    query = query.splitlines()[-1].strip().strip('"') if query else ""
    logger.info("No relevant passages for %r; retrying with %r", state["query"], query)
    return {"query": query or state["question"], "retries": state["retries"] + 1}


def _rerank(state: RetrievalState) -> RetrievalState:
    """Listwise LLM rerank by index, then keep the top N; untouched on LLM or parse failure."""
    # Once retries are spent, the last search's graded-out passages beat returning nothing.
    documents = state["relevant"] or state["documents"]
    if settings.rag_rerank_enabled and len(documents) > 1:
        order = _ask(
            "Rank the passages by relevance to the question, most relevant first.\n"
            f"Question: {state['question']}\n\nPassages:\n{_numbered(documents)}\n\n"
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


def retrieve(question: str) -> List[Document]:
    """Run the retrieval graph synchronously and return the final passages."""
    return _graph.invoke({"question": question, "query": question, "retries": 0})["documents"]
