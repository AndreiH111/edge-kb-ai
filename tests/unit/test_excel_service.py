import tempfile
import unittest
from pathlib import Path

import pandas as pd

from services.excel_service import _matches, load_excel_file, query_sheet


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


class QuerySheetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "people.xlsx"
        pd.DataFrame(
            {
                "Name": ["Alice", None, "Ben", "Cara", "Dan"],
                "City": ["Manila", None, "Manila", "Cebu", "Cebu"],
                "Stipend": [100, None, 1000, 300, 50],
            }
        ).to_excel(cls.path, sheet_name="Data", index=False)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_filters_ignore_case_and_compare_numbers_numerically(self) -> None:
        # As text, "1000" > "150" is false; numerically only Ben matches.
        count = query_sheet(self.path, filters=[("city", "==", "MANILA"), ("stipend", ">", "150")], aggregate="count")
        contains = query_sheet(self.path, filters=[("Name", "contains", "a")])

        self.assertEqual(
            count.splitlines(),
            ["people.xlsx, sheet 'Data' (sheets: Data): 4 rows; columns: Name, City, Stipend", "Matching rows: 1"],
        )
        self.assertTrue(contains.endswith("Name,City,Stipend\nAlice,Manila,100.0\nCara,Cebu,300.0\nDan,Cebu,50.0\n"))

    def test_group_by_aggregate_sort_and_top_n(self) -> None:
        grouped = query_sheet(
            self.path, group_by="CITY", aggregate="mean", column="stipend", sort_by="Stipend", descending=True
        )
        top = query_sheet(self.path, sort_by="Stipend", descending=True, limit=2)

        self.assertTrue(grouped.endswith("Showing 2 of 2 groups:\nCity,Stipend\nManila,550.0\nCebu,175.0\n"))
        self.assertTrue(top.endswith("Showing 2 of 4 rows:\nName,City,Stipend\nBen,Manila,1000.0\nCara,Cebu,300.0\n"))
        self.assertTrue(query_sheet(self.path, aggregate="sum", column="Stipend").endswith("sum of Stipend: 1450.0"))

    def test_dates_compare_by_day_and_blank_cells_never_contain_text(self) -> None:
        start = pd.Series(pd.to_datetime(["2024-03-01 09:30", None, "2024-02-29 00:00"]))
        names = pd.Series(["Ana", float("nan"), "Ben"])  # pandas 2 renders the blank as "nan"

        self.assertEqual(_matches(start, "==", "2024-03-01").tolist(), [True, False, False])
        self.assertEqual(_matches(names, "contains", "na").tolist(), [True, False, False])

    def test_unknown_column_lists_the_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown column 'Cty'. Available columns: Name, City, Stipend"):
            query_sheet(self.path, group_by="Cty")


if __name__ == "__main__":
    unittest.main()
