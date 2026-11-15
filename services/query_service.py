"""Conversational query rewriting and multi-query retrieval for the legacy RAG path.

``build_retrieval_documents`` condenses a follow-up question plus chat history
into a standalone question, optionally expands it into paraphrases, retrieves
for each query and returns the de-duplicated union. Any LLM failure falls back
to the pre-rewrite query so retrieval never breaks.
"""

import logging
import re
from typing import List, Optional

from langchain_core.documents import Document

from core.config import settings

logger = logging.getLogger(__name__)

NO_HISTORY = "No previous messages in this session."  # rag_service.format_chat_history's empty sentinel
# Keeps internal rewrite calls off the SSE token stream (langgraph TAG_NOSTREAM).
_NOSTREAM = {"tags": ["nostream"]}
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")

CONDENSE_PROMPT = """Rewrite the follow-up question as a standalone search question that includes \
every name, entity, time period and detail it refers to from the chat history. \
Return only the rewritten question.

Chat history:
{history}

Follow-up question: {question}"""

EXPAND_PROMPT = """Write {count} alternative phrasings of the question below for searching a \
document knowledge base. Keep every name, entity and time period. \
Return one per line with no numbering and nothing else.

Question: {question}"""


def _complete(llm, prompt: str) -> str:
    return llm.invoke(prompt, config=_NOSTREAM).text.strip()


def condense_question(question: str, history_text: str, llm) -> str:
    """Turn a follow-up plus history into a standalone question.

    Without history the question is returned unchanged. When rewriting is
    disabled or the LLM fails, the pre-rewrite query (history + question) is
    returned so follow-ups still match files named in earlier turns.
    """
    if not history_text or history_text == NO_HISTORY:
        return question

    history_query = f"{history_text}\nCurrent question: {question}"
    if not settings.query_rewrite_enabled:
        return history_query

    try:
        return _complete(llm, CONDENSE_PROMPT.format(history=history_text, question=question)) or history_query
    except Exception:
        logger.exception("Question rewriting failed; retrieving with history + question.")
        return history_query


def expand_queries(question: str, llm, n: Optional[int] = None) -> List[str]:
    """Return ``question`` plus up to ``n - 1`` de-duplicated LLM paraphrases (``n <= 1`` = off)."""
    count = settings.multi_query_count if n is None else n
    queries = [question]
    if count <= 1:
        return queries

    seen = {question.casefold()}
    for line in _complete(llm, EXPAND_PROMPT.format(count=count - 1, question=question)).splitlines():
        candidate = _LIST_MARKER.sub("", line).strip()
        if candidate and candidate.casefold() not in seen:
            seen.add(candidate.casefold())
            queries.append(candidate)
        if len(queries) >= count:
            break

    return queries


def retrieve_multi(queries: List[str]) -> List[Document]:
    """Union of ``retrieve_from_all_sources`` over ``queries``, de-duplicated by id in first-seen order."""
    from services import embed_service

    seen = set()
    documents: List[Document] = []
    for query in queries:
        for document in embed_service.retrieve_from_all_sources(query):
            key = document.id or document.page_content
            if key not in seen:
                seen.add(key)
                documents.append(document)

    return documents


def build_retrieval_documents(question: str, history_text: str, llm) -> List[Document]:
    """condense -> expand -> retrieve_multi; a failed expansion retrieves with the condensed query alone."""
    query = condense_question(question, history_text, llm)
    try:
        queries = expand_queries(query, llm)
    except Exception:
        logger.exception("Multi-query expansion failed; retrieving with the single query.")
        queries = [query]

    return retrieve_multi(queries)
