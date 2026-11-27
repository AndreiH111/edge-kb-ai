import tempfile
import unittest
from pathlib import Path

import pandas as pd

from services.excel_service import load_excel_file


class LoadExcelFileTests(unittest.TestCase):
    def test_rows_carry_excel_file_type_and_blank_rows_are_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "people.xlsx"
            pd.DataFrame({"name": ["Alice", None, "Ben"], "city": ["Manila", None, "Cebu"]}).to_excel(
                path, sheet_name="Data", index=False
            )

            documents = load_excel_file(path)

        self.assertEqual([d.id for d in documents], ["people.xlsx:Data:0", "people.xlsx:Data:2"])
        self.assertEqual(
            documents[0].metadata, {"source": "people.xlsx", "file_type": "excel", "sheet": "Data", "row": 0}
        )
        self.assertIn("name: Alice", documents[0].page_content)
        # Row text shows the sheet row (header on row 1, blank row 3 dropped); ids above stay 0-based.
        self.assertEqual([d.page_content.splitlines()[2] for d in documents], ["Row: 2", "Row: 4"])


if __name__ == "__main__":
    unittest.main()
