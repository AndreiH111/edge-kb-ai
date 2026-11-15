import re
from typing import Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from core.config import settings
from services import embed_service


chat_sessions: Dict[str, List[Dict[str, str]]] = {}

template = """
You are an expert assistant. Answer the user's question clearly and concisely.
Use the chat history to understand follow-up questions and references to earlier messages.
Use the retrieved context when it is relevant to the user's question.
If the user names a specific file, prioritize records from that source file.
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
        from langchain_ollama import ChatOllama

        return ChatOllama(model=settings.ollama_llm_model)

    if settings.hosted_llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": settings.hosted_llm_model,
            "temperature": settings.hosted_llm_temperature,
            "max_tokens": settings.hosted_llm_max_tokens,
            "model_kwargs": {"top_p": settings.hosted_llm_top_p},
        }
        if settings.hosted_llm_api_key:
            kwargs["api_key"] = settings.hosted_llm_api_key
        if settings.hosted_llm_base_url:
            kwargs["base_url"] = settings.hosted_llm_base_url

        return ChatOpenAI(**kwargs)

    if settings.hosted_llm_provider == "groq":
        from langchain_groq import ChatGroq

        kwargs = {
            "model": settings.hosted_llm_model,
            "temperature": settings.hosted_llm_temperature,
            "max_tokens": settings.hosted_llm_max_tokens,
            "model_kwargs": {"top_p": settings.hosted_llm_top_p},
            "reasoning_effort": settings.groq_reasoning_effort,
        }
        if settings.hosted_llm_api_key:
            kwargs["api_key"] = settings.hosted_llm_api_key
        if settings.hosted_llm_base_url:
            kwargs["groq_api_base"] = settings.hosted_llm_base_url

        return ChatGroq(**kwargs)

    raise ValueError(f"Unsupported hosted LLM provider: {settings.hosted_llm_provider}")


model = create_llm()
chain = prompt | model | StrOutputParser()


def format_chat_history(history: List[Dict[str, str]]) -> str:
    if not history:
        return "No previous messages in this session."

    recent_history = history[-settings.max_chat_history_messages :]
    return "\n".join(
        f"{message['role'].title()}: {message['content']}" for message in recent_history
    )


def _locator(metadata: dict) -> str:
    """One-line citation locator from loader metadata, e.g. ``a.xlsx (sheet Data, row 3)``."""
    parts = [f"{key} {metadata[key]}" for key in ("sheet", "page", "slide") if key in metadata]
    if "row_range" in metadata:
        parts.append(f"rows {metadata['row_range']}")
    elif "row" in metadata:
        parts.append(f"row {metadata['row']}")
    source = metadata.get("source") or "unknown source"
    return f"{source} ({', '.join(parts)})" if parts else source


def format_context(documents) -> str:
    """Number the passages as ``[n] <locator>`` blocks, deduped by ``id`` (or content)."""
    if not documents:
        return "No relevant context found."

    seen = set()
    passages = []
    for document in documents:
        key = document.id if document.id is not None else document.page_content
        if key in seen:
            continue
        seen.add(key)
        passages.append(
            f"[{len(passages) + 1}] {_locator(document.metadata)}\n{document.page_content}"
        )
    return "\n\n".join(passages)


def truncate_history_content(content: str) -> str:
    if len(content) <= settings.max_chat_history_content_chars:
        return content

    return f"{content[:settings.max_chat_history_content_chars]}... [truncated]"


def add_chat_history(history: List[Dict[str, str]], question: str, answer: str) -> None:
    history.extend(
        [
            {"role": "user", "content": truncate_history_content(question)},
            {"role": "assistant", "content": truncate_history_content(answer)},
        ]
    )

    if len(history) > settings.max_chat_history_messages:
        del history[: -settings.max_chat_history_messages]


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
    lines = [f"Found {len(embed_service.documents)} records in the current data files:"]

    for index, document in enumerate(embed_service.documents, start=1):
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
    history = chat_sessions.setdefault(session_id, [])

    if is_exhaustive_data_request(question):
        answer = format_all_records()
        add_chat_history(history, question, answer)
        return answer

    chat_history = format_chat_history(history)
    retrieval_query = (
        question if not history else f"{chat_history}\nCurrent question: {question}"
    )
    context = embed_service.retrieve_from_all_sources(retrieval_query)
    answer = chain.invoke(
        {
            "chat_history": chat_history,
            "context": format_context(context),
            "question": question,
        }
    )

    add_chat_history(history, question, answer)
    return answer
