from pathlib import Path
from typing import List

import pandas as pd
from langchain_core.documents import Document

from core.config import settings
from services.file_service import get_supported_files


def get_excel_files() -> List[Path]:
    return sorted(
        path
        for path in get_supported_files()
        if is_supported_excel_file(path.name) and path.is_file()
    )


def is_supported_excel_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in settings.supported_excel_extensions


def row_to_document(file_path: Path, sheet_name: str, row_index: int, row) -> Document:
    fields = []
    metadata = {
        "source": file_path.name,
        "file_type": "excel",
        "sheet": sheet_name,
        "row": int(row_index),
    }

    for column, value in row.items():
        if pd.isna(value):
            continue

        text = str(value).strip()
        if not text:
            continue

        fields.append(f"{column}: {text}")

    page_content = "\n".join(
        [
            f"Source file: {file_path.name}",
            f"Sheet: {sheet_name}",
            f"Row: {row_index}",
            *fields,
        ]
    )

    return Document(
        page_content=page_content,
        metadata=metadata,
        id=f"{file_path.name}:{sheet_name}:{row_index}",
    )


def load_excel_file(file_path: Path) -> List[Document]:
    """One Document per row (blank rows dropped) across every sheet of one workbook."""
    return [
        row_to_document(file_path, sheet_name, row_index, row)
        for sheet_name, df in pd.read_excel(file_path, sheet_name=None).items()
        for row_index, row in df.dropna(how="all").iterrows()
    ]


def load_excel_documents() -> List[Document]:
    """Rows of every stored workbook."""
    return [document for excel_file in get_excel_files() for document in load_excel_file(excel_file)]
