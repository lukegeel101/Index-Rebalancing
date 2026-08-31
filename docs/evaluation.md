# Reproducible dataset evaluation

The repository includes a deterministic evaluator for the committed research CSV.
It validates the dataset fingerprint and reports row coverage, date range, index and sector counts, ticker uniqueness, and descriptive statistics for the 24-hour post-release target.
It does not retrain either model and does not recreate the notebook-era cross-validation split.

Run the evaluation with:

```bash
python3 scripts/evaluate_dataset.py --check
python3 -m unittest discover -s tests -v
```

## Committed result

| Metric | Result |
| --- | ---: |
| Dataset records | 92 |
| Target values present | 92 |
| Unique tickers | 91 |
| S&P indices | 3 |
| GICS sectors | 11 |
| Release-date coverage | 2021-05-03 to 2022-12-28 |
| Mean 24-hour post-release change | 5.4585% |
| Median 24-hour post-release change | 6.6303% |
| Positive 24-hour observations | 74 of 92, or 80.43% |

The evaluator uses the Python standard library and runs in CI on Python 3.11, 3.12, and 3.13.

## Notebook environment

The two notebooks preserve the original exploratory workflow.
Create their legacy environment with `conda env create -f environment.yml`.
The PDF-processing notebook also expects the original source PDFs and Java for `tabula-py`.
The LSTM notebook contains a Google Colab upload step and notebook-era Keras interfaces, so it should be treated as a historical artifact rather than the CI entry point.

The committed evaluator is the supported reproducibility path for this release.
