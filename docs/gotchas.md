# Gotchas

## A complete paper can still make a weak repository landing page

Issue: the README preserved the full academic write-up but made visitors read several long sections before seeing the research question, model comparison, visual results, or data limitation.

Verified fix: keep the complete paper intact while adding a concise discovery layer at the top with searchable terms, a one-row study summary, both result figures, and an explicit research limitation.

## Presentation improvements must not upgrade exploratory claims

Issue: making a quantitative-finance project more visually compelling can accidentally make limited-sample results sound production-ready.

Verified fix: describe the work as exploratory, surface the 92-record committed-dataset limitation above the fold, and include a visible non-advice note before the full paper.
## Avoid unmatched configuration globs in zsh

Issue: a repository inventory command used `requirements*.txt`, and zsh aborted because no requirements file existed yet.

Verified fix: discover configuration files with `find` or test exact paths individually so an absent optional file does not stop the rest of the inspection.

## Count CSV records rather than physical lines

Issue: the original report described the committed dataset as 93 observations, but the CSV contains 93 physical lines including its header and therefore 92 data records.

Verified fix: parse the CSV with `csv.DictReader`, report 92 committed observations, preserve the original research context, and enforce the count in the deterministic evaluator.

## Keep ambiguous historical error labels conservative

Issue: the original results narrative labels the same `0.132` value as both mean absolute error and mean squared error.

Verified fix: preserve the value in the historical results document, call it a reported error, and avoid claiming a metric label that cannot be disambiguated from committed notebook output.

## Quote dates in citation metadata for older YAML loaders

Issue: the system Ruby YAML loader converted an unquoted `date-released` value into a `Date` object and rejected it during safe loading.

Verified fix: quote the CFF release date as an ISO-formatted string and rerun safe parsing for the citation and workflow metadata.
