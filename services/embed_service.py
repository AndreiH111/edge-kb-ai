import re
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document

from core.config import settings
from services.chunking_service import chunk_documents
from services.classifier_service import classify_excel_file, classify_excel_files
from services.excel_service import (
    is_supported_excel_file,
    load_excel_documents,
    row_to_document,
)
from services.file_service import get_supported_files, sync_storage_to_local
from services.pdf_service import load_pdf_documents, page_to_document
from services.ppt_service import load_powerpoint_documents, slide_to_document
from services.vectorstore_service import build_vector_store


def create_embeddings():
    if settings.use_local_embeddings:
        from langchain_ollama import OllamaEmbeddings

        print("Using Ollama embeddings")

        return OllamaEmbeddings(model=settings.ollama_embedding_model)

    if settings.use_custom_hash_embeddings:
        print("Using Custom Hash embeddings")
        from services.hash_embedding_service import HashEmbeddings

        return HashEmbeddings(dimensions=settings.local_embedding_dimensions)

    if settings.hosted_embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        kwargs = {"model": settings.hosted_embedding_model}
        if settings.hosted_embedding_api_key:
            kwargs["api_key"] = settings.hosted_embedding_api_key
        if settings.hosted_embedding_base_url:
            kwargs["base_url"] = settings.hosted_embedding_base_url

        return OpenAIEmbeddings(**kwargs)

    if settings.hosted_embedding_provider == "nvidia":
        from  langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
        print("Using NVIDIA embeddings")
        kwargs = {"model": settings.hosted_embedding_model}
        if settings.hosted_embedding_api_key:
            kwargs["api_key"] = settings.hosted_embedding_api_key
        if settings.hosted_embedding_base_url:
            kwargs["base_url"] = settings.hosted_embedding_base_url

        return NVIDIAEmbeddings(**kwargs)


    raise ValueError(
        "Unsupported hosted embedding provider: "
        f"{settings.hosted_embedding_provider}. "
        "Set HOSTED_EMBEDDING_PROVIDER=openai."
    )


embeddings = create_embeddings()
vector_store = build_vector_store(embeddings)

documents: List[Document] = []
ids: List[str] = []
retriever = vector_store.as_retriever(search_kwargs={"k": settings.retriever_k})


def refresh_vector_store() -> int:
    global documents, ids, retriever

    sync_storage_to_local()
    classify_excel_files()

    excel_documents, excel_ids = load_excel_documents()
    print(f"Loaded {len(excel_documents)} Excel documents")
    pdf_documents, pdf_ids = load_pdf_documents()
    powerpoint_documents, powerpoint_ids = load_powerpoint_documents()

    documents = excel_documents + pdf_documents + powerpoint_documents
    ids = excel_ids + pdf_ids + powerpoint_ids

    documents, ids = chunk_documents(documents, ids)

    vector_store.reset_collection()

    if documents:
        batch_size = 5000
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            vector_store.add_documents(batch_docs, ids=batch_ids)
            print(f"Indexed batch {i // batch_size + 1}: {len(batch_docs)} documents")

    retriever = vector_store.as_retriever(search_kwargs={"k": settings.retriever_k})
    return len(documents)


def parse_file(file_path: Path) -> Tuple[List[Document], List[str]]:
    suffix = file_path.suffix.lower()

    if is_supported_excel_file(file_path.name):
        import pandas as pd

        classify_excel_file(file_path)
        sheets = pd.read_excel(file_path, sheet_name=None)
        file_documents = []
        file_ids = []
        for sheet_name, df in sheets.items():
            for row_index, row in df.dropna(how="all").iterrows():
                document = row_to_document(file_path, sheet_name, row_index, row)
                if document.page_content.strip():
                    file_documents.append(document)
                    file_ids.append(document.id)
        return file_documents, file_ids

    if suffix in settings.supported_pdf_extensions:
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        file_documents = []
        file_ids = []
        for page_index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            document = page_to_document(file_path, page_index, text)
            file_documents.append(document)
            file_ids.append(document.id)
        return file_documents, file_ids

    if suffix in settings.supported_powerpoint_extensions:
        from pptx import Presentation

        presentation = Presentation(str(file_path))
        file_documents = []
        file_ids = []
        for slide_index, slide in enumerate(presentation.slides, start=1):
            document = slide_to_document(file_path, slide_index, slide)
            if document.page_content.strip():
                file_documents.append(document)
                file_ids.append(document.id)
        return file_documents, file_ids

    return [], []


def index_file(filename: str) -> int:
    global documents, ids, retriever

    sync_storage_to_local()

    file_path = None
    for path in get_supported_files():
        if path.name == filename:
            file_path = path
            break

    if file_path is None:
        return 0

    file_documents, file_ids = parse_file(file_path)
    if not file_documents:
        return 0

    file_documents, file_ids = chunk_documents(file_documents, file_ids)

    documents.extend(file_documents)
    ids.extend(file_ids)
    vector_store.add_documents(file_documents, ids=file_ids)
    retriever = vector_store.as_retriever(search_kwargs={"k": settings.retriever_k})
    return len(file_documents)


def remove_file(filename: str) -> int:
    global documents, ids, retriever

    prefix = f"{filename}:"
    remove_ids = [doc_id for doc_id in ids if doc_id.startswith(prefix)]

    if not remove_ids:
        return 0

    vector_store.delete_by_ids(remove_ids)

    remove_set = set(remove_ids)
    new_documents = []
    new_ids = []
    for doc, doc_id in zip(documents, ids):
        if doc_id not in remove_set:
            new_documents.append(doc)
            new_ids.append(doc_id)

    documents = new_documents
    ids = new_ids

    retriever = vector_store.as_retriever(search_kwargs={"k": settings.retriever_k})
    return len(remove_ids)


def initialize_on_startup() -> int:
    global documents, ids, retriever

    sync_storage_to_local()
    classify_excel_files()

    try:
        # Plain retriever, not similarity_search: RETRIEVER_SCORE_THRESHOLD must not hide existing data.
        results = vector_store.as_retriever(search_kwargs={"k": 1}).invoke("test")
        has_data = len(results) > 0
    except Exception:
        has_data = False

    if has_data:
        excel_documents, excel_ids = load_excel_documents()
        pdf_documents, pdf_ids = load_pdf_documents()
        powerpoint_documents, powerpoint_ids = load_powerpoint_documents()

        documents = excel_documents + pdf_documents + powerpoint_documents
        ids = excel_ids + pdf_ids + powerpoint_ids

        documents, ids = chunk_documents(documents, ids)

        retriever = vector_store.as_retriever(search_kwargs={"k": settings.retriever_k})
        print(f"Vector store has data. Rebuilt in-memory lists: {len(documents)} documents")
        return len(documents)

    return refresh_vector_store()


def get_indexed_sources() -> List[str]:
    return sorted(
        {
            document.metadata["source"]
            for document in documents
            if document.metadata.get("source")
        }
    )


def normalize_search_tokens(text: str) -> List[str]:
    # The chat-history labels are ignored so follow-up queries (which embed
    # "User: ... Assistant: ... Current question: ...") don't pin user_guide.pdf.
    ignored_terms = {
        "file",
        "look",
        "into",
        "what",
        "where",
        "when",
        "user",
        "assistant",
        "current",
        "question",
    }
    # Drop one trailing "s" before the length check, so "internships" matches
    # "Internship.xlsx" but "this" doesn't become "thi"; 3-letter words like
    # "you" or "for" matched filenames by accident.
    tokens = [
        token[:-1] if token.endswith("s") else token
        for token in re.findall(r"[a-z0-9]+", text.lower())
    ]
    return [token for token in tokens if len(token) >= 4 and token not in ignored_terms]


def get_sources_matching_query(query: str) -> List[str]:
    query_tokens = normalize_search_tokens(query)
    if not query_tokens:
        return []

    # Match the start of whole filename words (extension excluded), not any
    # substring of the name: "you" used to match "layout.pptx", "pdf" every PDF.
    matched_sources = []
    for source in get_indexed_sources():
        source_words = re.findall(r"[a-z0-9]+", Path(source).stem.lower())
        if any(word.startswith(token) for word in source_words for token in query_tokens):
            matched_sources.append(source)

    return matched_sources


def retrieve_from_all_sources(query: str) -> List[Document]:
    if not documents:
        return []

    matched_sources = get_sources_matching_query(query)
    if not matched_sources:
        return vector_store.similarity_search(query, k=settings.retriever_k)

    # Small named files go in whole so filtered questions ("which internships
    # are remote?") see every row; the character budget keeps the prompt bounded.
    named_documents = [
        document for document in documents if document.metadata.get("source") in matched_sources
    ]
    if sum(len(document.page_content) for document in named_documents) <= settings.retriever_max_file_chars:
        return named_documents

    # ponytail: one search per named file keeps comparisons between files fair;
    # many named files means many searches.
    return [
        document
        for source in matched_sources
        for document in vector_store.similarity_search(
            query, k=settings.retriever_k_per_source, filter={"source": source}
        )
    ]


initialize_on_startup()
