#!/usr/bin/env python3
"""Summarize the committed research dataset with deterministic checks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "added NYSE stocks dataset.csv"
DEFAULT_EXPECTED = ROOT / "data" / "evaluation" / "expected-results.json"
TARGET_COLUMN = "twentyfour_hr_post_release%change"
REQUIRED_COLUMNS = {
    "Index Name",
    "Ticker",
    "GICS Sector",
    "Release date",
    TARGET_COLUMN,
}


def evaluate(dataset_path: Path) -> dict[str, Any]:
    dataset_bytes = dataset_path.read_bytes()
    with dataset_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        columns = set(reader.fieldnames or [])
        missing_columns = REQUIRED_COLUMNS - columns
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Dataset is missing required columns: {missing}")
        rows = list(reader)

    if not rows:
        raise ValueError("Dataset contains no records")

    target_values = [
        float(row[TARGET_COLUMN])
        for row in rows
        if row[TARGET_COLUMN].strip().lower() not in {"", "na"}
    ]
    release_dates = [date.fromisoformat(row["Release date"]) for row in rows]
    ticker_counts = Counter(row["Ticker"] for row in rows)
    positive_count = sum(value > 0 for value in target_values)

    return {
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "duplicate_ticker_rows": sum(count - 1 for count in ticker_counts.values()),
        "first_release_date": min(release_dates).isoformat(),
        "index_count": len({row["Index Name"] for row in rows}),
        "last_release_date": max(release_dates).isoformat(),
        "mean_24h_post_release_change": round(statistics.fmean(target_values), 4),
        "median_24h_post_release_change": round(statistics.median(target_values), 4),
        "positive_24h_post_release_observations": positive_count,
        "positive_24h_post_release_percent": round(
            100 * positive_count / len(target_values), 2
        ),
        "records": len(rows),
        "sector_count": len({row["GICS Sector"] for row in rows}),
        "target_values_present": len(target_values),
        "unique_tickers": len(ticker_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    result = evaluate(args.dataset)
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.check:
        expected = json.loads(args.expected.read_text(encoding="utf-8"))
        if result != expected:
            print("Committed dataset metrics do not match the expected result.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
