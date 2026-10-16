# Point-in-time text feature engineering

`src/text_features.py` joins timestamped public reports to the existing hourly
supervised frame and compares the same GBDT approach with and without text
signals on one chronological out-of-time holdout.

## Required report schema

The default text CSV has these columns:

- `text_timestamp`: the report's **first available/first observed time**, UTC.
  Do not substitute the time of the underlying event or a backdated time parsed
  from article text. If a post is ingested after publication, use the time it
  became available to this forecasting pipeline for a strict no-leakage audit.
- `text`: report body/title or another reproducible public-text field.
- `region` (optional): the matching region label; blank/`global`/`ukraine`/`*`
  denotes a nationwide report. Region-scoped labels must match the hourly data.
- `embedding_0`, `embedding_1`, ... (optional): precomputed, finite numeric
  embedding dimensions. They are averaged only over reports in past windows.
  If the embedding encoder is trained or fine-tuned on project data, fit it on
  training-fold texts only; a frozen external encoder may be used consistently.

## Run

First create the complete supervised time-series frame (before splitting):

```bash
python -m src.features data/processed/hourly.csv \
    --output-dir data/processed/model_data --test-fraction 0.2
```

Then join text features and compare models on the same chronological holdout:

```bash
python -m src.text_features \
    data/processed/model_data/supervised.csv \
    data/raw/public_text.csv \
    --output-dir artifacts/text_comparison \
    --text-timestamp-col text_timestamp \
    --text-col text \
    --test-fraction 0.2 \
    --n-splits 5 \
    --source-coverage-start "2024-01-01T00:00:00Z"
```

Replace the coverage bound with the actual date at which the feed is known to
be complete. If that date is unknown, omit the flag; the code warns that zero
counts cannot distinguish a quiet period from incomplete archive coverage.

The output directory contains the enriched supervised/train/test CSVs, paired
test predictions, the metrics JSON, and both sets of fitted classifier and
regressor pipelines.

## Signals and cutoff contract

Text features include rolling report counts, counts of reports matching
missile/drone/air-activity/attack/official-alert keyword groups, a heuristic
intensity sum/flag, time since the latest prior report, optional coverage
fractions, and optional rolling mean embeddings. These keyword rules are
transparent heuristics and should be validated against the source text/language.

For a forecast reference time `t`, every count window represents
`[t - window, t)`. The join uses `pandas.merge_asof(direction="backward",
allow_exact_matches=False)`, so a report timestamped exactly at `t` is excluded.
The module asserts the latest joined report is strictly earlier than `t` and
updates `feature_cutoff` to the latest prior input timestamp. Global reports are
replicated across regions; region-scoped reports cannot flow into another
region's features.

The baseline and text-enhanced models receive identical train/test rows and
expanding-window CV folds. Feature selection and tuning happen inside the
training split; the out-of-time test set is evaluated once after refitting.
The `text_minus_baseline_delta` section is enhanced minus baseline: positive
ROC-AUC/precision/recall changes and negative MAE/RMSE changes are favorable.
No real-data benefit is assumed—the delta must be measured on a sufficiently
large, representative holdout.

## Tests

Run the point-in-time join regression tests from the project root:

```bash
python -m unittest tests.test_text_features
```
