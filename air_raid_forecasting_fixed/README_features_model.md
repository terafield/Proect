# Air-Raid Alert Forecasting: Features and Baseline Models

## Pipeline

```bash
# 1) Convert source event intervals into a complete hourly panel.
python -m src.data_loader data/raw/alerts.csv \
  --coverage-start "2024-01-01T00:00:00Z" \
  --coverage-end "2026-10-15T23:00:00Z" \
  --output-dir data/processed

# 2) Create causal historical features and a chronological holdout.
python -m src.features data/processed/hourly.csv \
  --output-dir data/processed/model_data \
  --test-fraction 0.2 \
  --neighbors-json config/region_neighbors.json

# 3) Tune and evaluate on the out-of-time test set.
python -m src.model data/processed/model_data/train.csv \
  data/processed/model_data/test.csv \
  --output-dir artifacts/baseline \
  --n-splits 5
```

The coverage dates above are examples; replace them with the source's actual observation window. Each boundary represents an inclusive UTC hour-bucket start. Silent hours cannot be inferred reliably from event-only data, so provide coverage bounds when known.

## Neighbor graph format

`config/region_neighbors.json` must be a JSON object mapping each source region label to the labels of its adjacent regions, for example:

```json
{
  "Region A": ["Region B"],
  "Region B": ["Region A", "Region C"],
  "Region C": ["Region B"]
}
```

The feature module includes a starting adjacency map for common Ukrainian oblast labels, but region names and adjacency should be checked against the data source. Unknown labels raise an error instead of silently creating all-zero neighbor features.

## Leakage controls and evaluation semantics

- A row at bucket `t` predicts the outcome for bucket `t` using only fully preceding buckets.
- Lags and rolling windows are shifted before calculation; neighbor activity is also lagged.
- The split and cross-validation folds operate on unique timestamps so regions from one hour cannot appear in both sides of a fold.
- One-hot encoding is fitted inside each scikit-learn pipeline / CV fold. No scaler or imputer is fitted globally; tree models do not require scaling.
- The out-of-time test is evaluated after tuning on training-only expanding-window folds.
- The test metrics are for *rolling one-step-ahead* forecasts: as time moves forward, an earlier observed test-hour can become historical input for a later test-hour. For a single fixed-origin multi-hour forecast, implement recursive prediction instead of using actual intermediate test outcomes.

Dependencies: `pandas`, `numpy`, `scikit-learn`, and `joblib`.
