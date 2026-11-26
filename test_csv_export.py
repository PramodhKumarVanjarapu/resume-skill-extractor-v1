"""Compatibility checks for spreadsheet CSV imports."""

import csv
import io
import unittest

import pandas as pd

import app


class CsvExportTests(unittest.TestCase):
    def test_rfc4180_round_trip_preserves_special_cell_content(self):
        frame = pd.DataFrame([{
            "phone": "08810931580",
            "project_names": 'API, "Search"\nService',
            "tech_skills": "Python; Django; café",
        }])

        payload = app.to_csv_bytes(frame)
        self.assertTrue(payload.startswith(b"\xef\xbb\xbf"))
        text = payload.decode("utf-8-sig")
        self.assertIn("\r\n", text)
        self.assertTrue(text.startswith('"phone","project_names","tech_skills"\r\n'))
        rows = list(csv.reader(io.StringIO(text, newline="")))
        self.assertEqual(rows, [
            ["phone", "project_names", "tech_skills"],
            ["08810931580", 'API, "Search"\nService', "Python; Django; café"],
        ])

    def test_standard_export_does_not_turn_phone_numbers_into_formulas(self):
        frame = pd.DataFrame([{"phone": "08810931580"}])
        rows = list(csv.reader(io.StringIO(app.table_csv_bytes(frame, ["phone"]).decode("utf-8-sig"))))
        self.assertEqual(rows[1][0], "08810931580")


if __name__ == "__main__":
    unittest.main()
