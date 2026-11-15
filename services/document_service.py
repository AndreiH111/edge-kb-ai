"""PDF and PowerPoint loading via Docling: one Markdown Document per page/slide."""

from pathlib import Path
from typing import Any, List

from langchain_core.documents import Document

from core.config import settings
from services.file_service import get_supported_files

_CONVERTER = None


def _converter() -> Any:
    """Docling converter built on first use: the import pulls torch and the
    first conversion downloads the layout models, so it never runs at import time."""
    global _CONVERTER
    if _CONVERTER is None:
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter

        _CONVERTER = DocumentConverter(allowed_formats=[InputFormat.PDF, InputFormat.PPTX])
    return _CONVERTER


def is_document_file(path: Path) -> bool:
    """True for the PDF/PowerPoint extensions Docling handles here."""
    return path.suffix.lower() in settings.supported_pdf_extensions | settings.supported_powerpoint_extensions


def load_file(file_path: Path) -> List[Document]:
    """Convert one PDF/PowerPoint file; empty pages/slides are skipped."""
    if file_path.suffix.lower() in settings.supported_pdf_extensions:
        label, file_type, unit = "PDF", "pdf", "page"
    else:
        label, file_type, unit = "PowerPoint", "powerpoint", "slide"

    result = _converter().convert(str(file_path))
    if result.status != "success":  # PARTIAL_SUCCESS: some pages failed and are missing below
        print(f"Docling converted {file_path.name} with status {result.status}: {result.errors}")
    doc = result.document
    documents = []
    for number in sorted(doc.pages):
        text = doc.export_to_markdown(page_no=number, image_placeholder="").strip()
        if not text:
            continue
        documents.append(
            Document(
                page_content=(
                    f"Source file: {file_path.name}\nFile type: {label}\n{unit.capitalize()}: {number}\n{text}"
                ),
                metadata={"source": file_path.name, "file_type": file_type, unit: number},
                id=f"{file_path.name}:{unit}:{number}",
            )
        )
    return documents


def load_documents() -> List[Document]:
    """Load every stored PDF/PowerPoint file, in filename order; a file that fails to convert is skipped."""
    documents = []
    for path in sorted(get_supported_files()):
        if not (path.is_file() and is_document_file(path)):
            continue
        try:
            documents.extend(load_file(path))
        except Exception as error:  # one corrupt/encrypted file must not abort startup or a full reindex
            print(f"Skipping {path.name}: Docling conversion failed: {error}")
    return documents
