import operator
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


def read_sheets(file_path: Path) -> Dict[str, pd.DataFrame]:
    """Every sheet of one workbook by name, blank rows dropped."""
    return {name: df.dropna(how="all") for name, df in pd.read_excel(file_path, sheet_name=None).items()}


def load_excel_file(file_path: Path) -> List[Document]:
    """One Document per row (blank rows dropped) across every sheet of one workbook."""
    return [
        row_to_document(file_path, sheet_name, row_index, row)
        for sheet_name, df in read_sheets(file_path).items()
        for row_index, row in df.iterrows()
    ]


def load_excel_documents() -> List[Document]:
    """Rows of every stored workbook."""
    return [document for excel_file in get_excel_files() for document in load_excel_file(excel_file)]


_COMPARE = {
    "==": operator.eq, "!=": operator.ne, ">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le
}


def _pick(name: Any, choices: Iterable, kind: str) -> Any:
    """The choice equal to ``name`` ignoring case and outer spaces; else a ValueError listing them."""
    for choice in choices:
        if str(choice).strip().casefold() == str(name).strip().casefold():
            return choice
    raise ValueError(f"Unknown {kind} {name!r}. Available {kind}s: {', '.join(map(str, choices))}")


def _matches(values: pd.Series, op: str, value: Any) -> pd.Series:
    """Row mask for ``values <op> value``: numeric when both sides are numbers, else text ignoring case."""
    if pd.api.types.is_datetime64_any_dtype(values):
        values = values.dt.strftime("%Y-%m-%d")  # dates compare by day, as YYYY-MM-DD text
    # where(): pandas 2 turns blank cells into the text "nan"; keep them blank.
    text = values.astype(str).str.strip().str.casefold().where(values.notna())
    target = str(value).strip().casefold()
    if op == "contains":
        return text.str.contains(target, regex=False, na=False)
    numbers, number = pd.to_numeric(values, errors="coerce"), pd.to_numeric(value, errors="coerce")
    if pd.notna(number) and numbers.notna().any():
        return _COMPARE[op](numbers, number)
    return _COMPARE[op](text, target)


def query_sheet(
    file_path: Path,
    sheet: Optional[str] = None,
    filters: Iterable[Tuple[str, str, Any]] = (),
    group_by: Optional[str] = None,
    aggregate: Optional[str] = None,
    column: Optional[str] = None,
    sort_by: Optional[str] = None,
    descending: bool = False,
    limit: int = 20,
) -> str:
    """Filter, group, aggregate and sort every row of one sheet (the first by default).

    Returns a header line (sheet, row count, columns), the matching row count, then
    the scalar or the first ``limit`` rows/groups as CSV. ``group_by`` alone counts
    rows per group. Raises ValueError listing the valid names for an unknown sheet or column.
    """
    # shortcut: re-reads the workbook on every call; cache by (path, mtime) if big workbooks get slow.
    sheets = read_sheets(file_path)
    name = _pick(sheet, sheets, "sheet") if sheet else next(iter(sheets))
    df = sheets[name]
    lines = [
        f"{file_path.name}, sheet {name!r} (sheets: {', '.join(sheets)}): {len(df)} rows; "
        f"columns: {', '.join(map(str, df.columns))}"
    ]
    for filter_column, op, value in filters:
        df = df[_matches(df[_pick(filter_column, df.columns, "column")], op, value)]
    lines.append(f"Matching rows: {len(df)}")
    if aggregate in ("sum", "mean", "min", "max"):
        target = _pick(column or "", df.columns, "column")
        # min/max keep dates and text comparable; sum/mean need numbers.
        values = df[target] if aggregate in ("min", "max") else pd.to_numeric(df[target], errors="coerce")
        if not group_by:
            return "\n".join([*lines, f"{aggregate} of {target}: {values.agg(aggregate)}"])
        df = values.groupby(df[_pick(group_by, df.columns, "column")], dropna=False).agg(aggregate).reset_index()
    elif group_by:
        df = df.groupby(_pick(group_by, df.columns, "column"), dropna=False).size().reset_index(name="count")
    elif aggregate == "count":
        return "\n".join(lines)
    if sort_by:
        df = df.sort_values(_pick(sort_by, df.columns, "column"), ascending=not descending)
    limit = min(max(limit, 1), 200)
    lines.append(f"Showing {min(limit, len(df))} of {len(df)} {'groups' if group_by else 'rows'}:")
    return "\n".join([*lines, df.head(limit).to_csv(index=False)])
