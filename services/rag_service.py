import re
import threading
from collections import defaultdict
from contextvars import ContextVar
from typing import DefaultDict, List, Optional

from langchain.chat_models import init_chat_model
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    get_buffer_string,
    trim_messages,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from core.config import settings
from services import embed_service, query_service


chat_sessions: DefaultDict[str, InMemoryChatMessageHistory] = defaultdict(
    InMemoryChatMessageHistory
)

template = """
You are an expert assistant. Answer the user's question clearly and concisely.
Use the chat history to understand follow-up questions and references to earlier messages.
Answer only from the retrieved context and the chat history.
If they do not contain the answer, say so plainly and name what is missing; never guess or use outside knowledge.
If the user names a specific file, prioritize records from that source file.
Reply in the user's language. Use markdown tables for multi-row data.
Cite the context passages you used as [n] and name their file; do not cite passages you did not use.

Chat history:
{chat_history}

Retrieved context:
{context}

Current question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


def create_llm():
    if settings.use_local_llm:
        return init_chat_model(settings.ollama_llm_model, model_provider="ollama")

    optional = {
        key: value
        for key, value in (
            ("api_key", settings.hosted_llm_api_key),
            ("base_url", settings.hosted_llm_base_url),
        )
        if value
    }
    if settings.hosted_llm_provider == "groq":
        optional["reasoning_effort"] = settings.groq_reasoning_effort

    return init_chat_model(
        settings.hosted_llm_model,
        model_provider=settings.hosted_llm_provider,
        temperature=settings.hosted_llm_temperature,
        max_tokens=settings.hosted_llm_max_tokens,
        model_kwargs={"top_p": settings.hosted_llm_top_p},
        **optional,
    )


model = create_llm()
chain = prompt | model | StrOutputParser()


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return the session's chat history, creating it on first use."""
    return chat_sessions[session_id]


def recent_messages(history: InMemoryChatMessageHistory) -> List[BaseMessage]:
    """The last ``MAX_CHAT_HISTORY_MESSAGES`` messages, starting on a user turn."""
    return trim_messages(
        history.messages,
        max_tokens=settings.max_chat_history_messages,
        token_counter=len,
        strategy="last",
        start_on="human",
    )


def format_chat_history(history: InMemoryChatMessageHistory) -> str:
    recent = recent_messages(history)
    if not recent:
        return "No previous messages in this session."

    return get_buffer_string(recent, human_prefix="User", ai_prefix="Assistant")


def _locator(metadata: dict) -> str:
    """One-line citation locator from loader metadata, e.g. ``a.xlsx (sheet Data, row 5)``."""
    parts = [f"{key} {metadata[key]}" for key in ("sheet", "page", "slide") if key in metadata]
    # shortcut: index + 2 assumes the header is on row 1, store real sheet rows at load if titles sit above it.
    if "row_range" in metadata:
        start, end = metadata["row_range"].split("-")
        parts.append(f"rows {int(start) + 2}-{int(end) + 2}")
    elif "row" in metadata:
        parts.append(f"row {int(metadata['row']) + 2}")
    source = metadata.get("source") or "unknown source"
    return f"{source} ({', '.join(parts)})" if parts else source


def format_context(documents) -> str:
    """Number the passages as ``[n] <locator>`` blocks, deduped by ``id`` (or content).

    Inside a chat turn (``turn_passages`` set) numbering continues across calls and a
    passage already numbered this turn keeps its number.
    """
    if not documents:
        return "No relevant context found."

    numbered = turn_passages.get()
    if numbered is None:
        numbered = []
    passages = {}
    # Parallel tool calls share the turn's list; the lock keeps their numbers apart.
    with _numbering_lock:
        numbers = {key: n for n, (key, _) in enumerate(numbered, 1)}
        for document in documents:
            key = document.id if document.id is not None else document.page_content
            if key in passages:
                continue
            locator = _locator(document.metadata)
            if key not in numbers:
                numbered.append((key, locator))
                numbers[key] = len(numbered)
            passages[key] = f"[{numbers[key]}] {locator}\n{document.page_content}"
    return "\n\n".join(passages.values())


# The chat turn's numbered passages as ``(key, locator)`` pairs, ``[n]`` = index + 1 (set by agent_service).
turn_passages: ContextVar[Optional[list]] = ContextVar("turn_passages", default=None)
_numbering_lock = threading.Lock()  # shortcut: one lock for all turns, give each turn its own if it ever contends


def cited_sources(answer: str) -> List[dict]:
    """``{"n", "locator"}`` of each passage numbered this turn that ``answer`` cites as ``[n]`` or ``[n, m]``."""
    cited = {int(n) for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer) for n in group.split(",")}
    passages = turn_passages.get() or []
    return [{"n": n, "locator": locator} for n, (_, locator) in enumerate(passages, 1) if n in cited]


def truncate_history_content(content: str) -> str:
    if len(content) <= settings.max_chat_history_content_chars:
        return content

    return f"{content[:settings.max_chat_history_content_chars]}... [truncated]"


def add_chat_history(
    history: InMemoryChatMessageHistory, question: str, answer: str
) -> None:
    # One add_messages call keeps the question/answer pair adjacent.
    history.add_messages(
        [
            HumanMessage(truncate_history_content(question)),
            AIMessage(truncate_history_content(answer)),
        ]
    )
    history.messages = recent_messages(history)


def restore_history(session_id: str, turns) -> None:
    """Replace a session's memory with the ``question``/``answer`` turns a client stored,
    so a reopened chat keeps its context across API restarts and instances."""
    history = get_history(session_id)
    history.clear()
    for turn in turns:
        add_chat_history(history, turn.question, turn.answer)


def is_exhaustive_data_request(question: str) -> bool:
    words = set(re.findall(r"[a-z0-9]+", question.lower()))
    data_terms = {
        "record",
        "records",
        "row",
        "rows",
        "entry",
        "entries",
        "data",
        "internship",
        "internships",
        "intenship",
        "intenships",
    }
    exhaustive_terms = {"all", "every", "complete", "full", "entire", "available"}
    filler_terms = {
        "list", "show", "give", "display", "print", "me", "us", "the", "a", "of", "please",
        "can", "could", "you", "what", "is", "are", "there", "my", "our", "current",
        "uploaded", "indexed", "single", "do", "have",
    }

    # Whole words only, and every word must come from these sets: substring
    # matching let "row" hit "grows" and "all" hit "overall", and any extra word
    # ("in Manila", "where status is Open", "Jira issues") means the user wants
    # a filtered answer, not a dump of every KB record.
    # ponytail: allowlist, so "show me all records in the file" also goes to the
    # LLM; widen filler_terms if a real "list everything" phrasing gets missed.
    return (
        bool(words & data_terms)
        and bool(words & exhaustive_terms)
        and words <= data_terms | exhaustive_terms | filler_terms
    )


def format_all_records() -> str:
    source = embed_service.pinned_source.get()
    records = [
        document
        for document in embed_service.documents
        if not source or document.metadata.get("source") == source
    ]
    lines = [f"Found {len(records)} records in the current data files:"]

    for index, document in enumerate(records, start=1):
        fields = [
            line.strip()
            for line in document.page_content.splitlines()
            if line.strip()
        ]

        lines.append(f"- Record {index}")
        for field in fields:
            lines.append(f"  - {field}")

    return "\n".join(lines)


def answer_question(question: str, session_id: str) -> str:
    history = get_history(session_id)

    if is_exhaustive_data_request(question):
        answer = format_all_records()
        add_chat_history(history, question, answer)
        return answer

    chat_history = format_chat_history(history)
    context = query_service.build_retrieval_documents(question, chat_history, model)
    answer = chain.invoke(
        {
            "chat_history": chat_history,
            "context": format_context(context),
            "question": question,
        }
    )

    add_chat_history(history, question, answer)
    return answer
