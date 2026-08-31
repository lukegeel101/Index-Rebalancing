import csv
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_dataset import DEFAULT_DATASET, DEFAULT_EXPECTED, evaluate


class DatasetEvaluationTests(unittest.TestCase):
    def test_committed_dataset_matches_expected_results(self) -> None:
        import json

        expected = json.loads(DEFAULT_EXPECTED.read_text(encoding="utf-8"))
        self.assertEqual(evaluate(DEFAULT_DATASET), expected)

    def test_missing_required_column_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "incomplete.csv"
            with dataset.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=["Index Name", "Ticker", "GICS Sector", "Release date"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "Index Name": "Example Index",
                        "Ticker": "TEST",
                        "GICS Sector": "Example",
                        "Release date": "2026-01-01",
                    }
                )

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                evaluate(dataset)

    def test_empty_dataset_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "empty.csv"
            dataset.write_text(
                "Index Name,Ticker,GICS Sector,Release date,"
                "twentyfour_hr_post_release%change\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "no records"):
                evaluate(dataset)


if __name__ == "__main__":
    unittest.main()
