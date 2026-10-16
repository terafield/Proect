"""Leakage-aware GBDT baselines for hourly air-raid alert forecasting.

The classifier estimates whether any alert is active during the target hour.
The regressor estimates total alert-minutes in the target hour. Hyperparameters
are selected using expanding-window splits over unique timestamps. The final
metrics are computed once on a chronologically later out-of-time test period.

Typical use::

    python -m src.model data/processed/model_data/train.csv \
        data/processed/model_data/test.csv --output-dir artifacts/baseline

Dependencies: pandas, numpy, scikit-learn, joblib.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from .data_loader import FEATURE_CUTOFF_COLUMN, TARGET_COLUMNS, TIMESTAMP_COLUMN
    from .features import get_model_feature_columns
except ImportError:  # pragma: no cover - direct-script compatibility.
    from data_loader import (  # type: ignore[no-redef]
        FEATURE_CUTOFF_COLUMN, TARGET_COLUMNS, TIMESTAMP_COLUMN
    )
    from features import get_model_feature_columns  # type: ignore[no-redef]

LOGGER = logging.getLogger(__name__)
OCCURRENCE_TARGET = "alert_occurrence"
DURATION_TARGET = "alert_minutes"


def _canonicalize_model_frame(
    frame: pd.DataFrame, *, label: str = "input"
) -> pd.DataFrame:
    """Return a copy using canonical model column names and validate its schema.

    For convenience when calling the public Python API directly, common aliases
    are accepted: ``timestamp`` -> ``bucket_start``, ``occurrence`` ->
    ``alert_occurrence``, and ``duration`` -> ``alert_minutes``. The command-line
    pipeline always writes canonical names. Conflicting duplicate aliases are
    rejected instead of silently selecting one source.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame.")
    result = frame.copy()
    aliases = {
        TIMESTAMP_COLUMN: ("timestamp", "forecast_timestamp", "datetime"),
        OCCURRENCE_TARGET: ("occurrence",),
        DURATION_TARGET: ("duration",),
    }
    for canonical, candidates in aliases.items():
        present = [name for name in candidates if name in result.columns]
        if canonical not in result.columns and present:
            result = result.rename(columns={present[0]: canonical})
            present = present[1:]
        for alias in present:
            if canonical in result.columns:
                left = result[canonical]
                right = result[alias]
                equal = left.eq(right) | (left.isna() & right.isna())
                if not bool(equal.all()):
                    raise ValueError(
                        f"{label} has conflicting values in {canonical!r} and {alias!r}."
                    )
                result = result.drop(columns=[alias])

    required = {
        TIMESTAMP_COLUMN, FEATURE_CUTOFF_COLUMN, "region",
        OCCURRENCE_TARGET, DURATION_TARGET,
    }
    missing = sorted(required - set(result.columns))
    if missing:
        accepted_timestamp_aliases = "bucket_start (or timestamp/forecast_timestamp/datetime)"
        raise ValueError(
            f"{label} dataframe is missing required columns: {missing}. "
            f"Timestamp must be named {accepted_timestamp_aliases}; found columns: "
            f"{list(result.columns)!r}"
        )

    result[TIMESTAMP_COLUMN] = pd.to_datetime(
        result[TIMESTAMP_COLUMN], utc=True, errors="coerce"
    )
    result[FEATURE_CUTOFF_COLUMN] = pd.to_datetime(
        result[FEATURE_CUTOFF_COLUMN], utc=True, errors="coerce"
    )
    if result[[TIMESTAMP_COLUMN, FEATURE_CUTOFF_COLUMN]].isna().any().any():
        raise ValueError(f"{label} contains invalid timestamp metadata.")
    if result["region"].isna().any() or result["region"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{label} contains missing or blank region labels.")
    result["region"] = result["region"].astype(str).str.strip()
    return result.sort_values(
        [TIMESTAMP_COLUMN, "region"], kind="stable"
    ).reset_index(drop=True)


def read_model_frame(path: str | Path) -> pd.DataFrame:
    """Read a processed feature CSV and normalize its schema/timestamps."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {file_path}")
    frame = _canonicalize_model_frame(pd.read_csv(file_path), label=str(file_path))
    # The raw alert-start count is useful for lag generation but not required to
    # fit the two final estimators once causal features and target columns exist.
    if "alert_start_count" not in frame.columns:
        LOGGER.warning(
            "%s does not contain alert_start_count; ensure this is an already-"
            "engineered model frame, not raw event data.",
            file_path,
        )
    return frame


def assert_temporal_integrity(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Fail loudly if the temporal split or metadata reveals future information."""
    train = _canonicalize_model_frame(train, label="train")
    test = _canonicalize_model_frame(test, label="test")
    if train.empty or test.empty:
        raise ValueError("Training and test frames must both be non-empty.")
    for split_name, frame in (("train", train), ("test", test)):
        if not (frame[FEATURE_CUTOFF_COLUMN] < frame[TIMESTAMP_COLUMN]).all():
            raise AssertionError(
                f"Leakage detected: {split_name} features include a "
                "current/future target bucket."
            )
        if not frame[TIMESTAMP_COLUMN].eq(frame[TIMESTAMP_COLUMN].dt.floor("h")).all():
            raise ValueError(f"{split_name} timestamps must align to whole UTC hours.")
        unique_times = pd.Index(frame[TIMESTAMP_COLUMN].drop_duplicates()).sort_values()
        if len(unique_times) > 1 and not pd.Series(unique_times).diff().dropna().eq(
            pd.Timedelta(hours=1)
        ).all():
            raise AssertionError(
                f"{split_name} timestamps are not continuous hourly buckets; lag labels "
                "would not match their declared time offsets."
            )
        if frame[[TIMESTAMP_COLUMN, "region"]].duplicated().any():
            raise AssertionError(f"{split_name} has duplicate region/timestamp rows.")
        if frame[[OCCURRENCE_TARGET, DURATION_TARGET]].isna().any().any():
            raise ValueError(f"{split_name} target contains missing values.")
        if not frame[OCCURRENCE_TARGET].isin([0, 1]).all():
            raise ValueError(f"{split_name} occurrence target must contain only 0/1 values.")
        derived_occurrence = frame[DURATION_TARGET].astype(float).gt(0.0).astype(int)
        if not frame[OCCURRENCE_TARGET].astype(int).eq(derived_occurrence).all():
            raise AssertionError(
                f"{split_name} target inconsistency: alert_occurrence must equal "
                "(alert_minutes > 0).astype(int)."
            )
        expected_regions = set(frame["region"].astype(str))
        for timestamp, group in frame.groupby(TIMESTAMP_COLUMN, sort=False):
            if set(group["region"].astype(str)) != expected_regions:
                raise AssertionError(
                    f"{split_name} panel is incomplete at {timestamp}: every target "
                    "timestamp must contain the same region roster."
                )

    if set(train["region"].astype(str)) != set(test["region"].astype(str)):
        raise AssertionError("Train and test use different region rosters.")

    train_end = train[TIMESTAMP_COLUMN].max()
    test_start = test[TIMESTAMP_COLUMN].min()
    if train_end >= test_start:
        raise AssertionError(
            f"Leakage detected: train ends at {train_end}, test starts at {test_start}."
        )
    if test_start - train_end != pd.Timedelta(hours=1):
        raise AssertionError(
            "Train/test boundary skips one or more hourly buckets. Rebuild a continuous "
            "hourly panel before model training and evaluation."
        )
    if train[FEATURE_CUTOFF_COLUMN].max() >= test_start:
        raise AssertionError(
            "Leakage detected: a training feature cutoff reaches the test period."
        )

    allowed = set(get_model_feature_columns(train))
    prohibited = set(TARGET_COLUMNS) | {
        OCCURRENCE_TARGET, FEATURE_CUTOFF_COLUMN, TIMESTAMP_COLUMN
    }
    if allowed & prohibited:
        raise AssertionError(
            "Target/metadata leakage in feature whitelist: "
            f"{sorted(allowed & prohibited)}"
        )
    if not allowed:
        raise ValueError("No model feature columns were found in the processed data.")
    LOGGER.info(
        "Temporal checks passed: train %s..%s; test %s..%s UTC.",
        train[TIMESTAMP_COLUMN].min(), train_end,
        test_start, test[TIMESTAMP_COLUMN].max(),
    )


def build_expanding_window_splits(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
    test_size_hours: int | None = None,
    gap_hours: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build TimeSeriesSplit folds over timestamps rather than panel rows.

    Splitting unique timestamps first prevents regions from the same hour being
    distributed across train and validation. Returned indexes are row positions
    suitable for scikit-learn's GridSearchCV.

    The folds represent rolling one-step-ahead evaluation: outcomes observed in
    an earlier validation hour may be used as historical inputs for a later
    validation hour. For a *single fixed-origin multi-hour forecast*, use a
    recursive forecasting loop instead of feeding actual interim test outcomes.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2 for expanding-window CV.")
    if gap_hours < 0:
        raise ValueError("gap_hours cannot be negative.")
    if frame.empty:
        raise ValueError("Cannot build cross-validation folds from an empty frame.")
    timestamps = pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Cross-validation input has invalid timestamps.")
    unique_times = pd.Index(timestamps.drop_duplicates()).sort_values()
    if len(unique_times) > 1 and not pd.Series(unique_times).diff().dropna().eq(
        pd.Timedelta(hours=1)
    ).all():
        raise ValueError(
            "Expanding-window CV requires a continuous hourly time grid; missing "
            "timestamps make the requested gap/test size ambiguous."
        )
    if len(unique_times) < n_splits + 1:
        raise ValueError(
            f"Need at least {n_splits + 1} unique timestamps for {n_splits} folds; "
            f"received {len(unique_times)}. Reduce n_splits or provide more history."
        )
    splitter = TimeSeriesSplit(
        n_splits=n_splits, test_size=test_size_hours, gap=gap_hours
    )
    timestamp_values = timestamps.to_numpy()
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    time_indexes = np.arange(len(unique_times))
    for train_time_indexes, valid_time_indexes in splitter.split(time_indexes):
        train_times = unique_times.take(train_time_indexes)
        valid_times = unique_times.take(valid_time_indexes)
        train_mask = pd.Index(timestamp_values).isin(train_times)
        valid_mask = pd.Index(timestamp_values).isin(valid_times)
        train_indexes = np.flatnonzero(np.asarray(train_mask))
        valid_indexes = np.flatnonzero(np.asarray(valid_mask))
        if len(train_indexes) == 0 or len(valid_indexes) == 0:
            raise AssertionError(
                "A TimeSeriesSplit fold unexpectedly contains an empty partition."
            )
        if timestamps.iloc[train_indexes].max() >= timestamps.iloc[valid_indexes].min():
            raise AssertionError(
                "Cross-validation leakage: train timestamps overlap validation."
            )
        if set(timestamps.iloc[train_indexes]) & set(timestamps.iloc[valid_indexes]):
            raise AssertionError(
                "A timestamp was split across train and validation rows."
            )
        splits.append((train_indexes, valid_indexes))
    LOGGER.info(
        "Created %d expanding-window folds across unique timestamps.", len(splits)
    )
    return splits


def _make_pipeline(
    *,
    task: str,
    feature_columns: list[str],
    random_state: int = 42,
) -> Pipeline:
    """Create a model pipeline whose preprocessing is fitted inside each fold."""
    categorical = [column for column in feature_columns if column == "region"]
    numeric = [column for column in feature_columns if column != "region"]
    transformers: list[tuple[str, object, list[str]]] = []
    if categorical:
        transformers.append((
            "region_encoder",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical,
        ))
    if numeric:
        # No scaler is needed for tree-based models; numeric features are passed
        # through unchanged. If scaling is introduced later, put it in this
        # ColumnTransformer/Pipeline, never fit it on the full dataset.
        transformers.append(("numeric", "passthrough", numeric))
    preprocessing = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )
    if task == "classification":
        estimator = HistGradientBoostingClassifier(
            random_state=random_state,
            early_stopping=False,
            max_iter=100,
            max_leaf_nodes=15,
        )
    elif task == "regression":
        estimator = HistGradientBoostingRegressor(
            random_state=random_state,
            early_stopping=False,
            max_iter=100,
            max_leaf_nodes=15,
        )
    else:
        raise ValueError(f"Unknown task: {task}")
    return Pipeline([("preprocess", preprocessing), ("model", estimator)])


def _search_pipeline(
    pipeline: Pipeline,
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    scoring: str,
    task: str,
) -> Pipeline:
    """Tune small GBDT grids using chronological folds from training data only."""
    param_grid = {
        "model__max_iter": [100, 200],
        "model__max_leaf_nodes": [15, 31],
        "model__learning_rate": [0.05, 0.1],
    }
    if task == "classification":
        # ROC-AUC is undefined if any validation fold has only one class.
        usable_folds = [
            (train_idx, valid_idx)
            for train_idx, valid_idx in cv_splits
            if (
                y_train.iloc[train_idx].nunique() == 2
                and y_train.iloc[valid_idx].nunique() == 2
            )
        ]
        if len(usable_folds) < 2:
            LOGGER.warning(
                "Too few CV folds contain both classes in train and validation. "
                "Skipping classifier tuning and fitting default GBDT parameters."
            )
            pipeline.fit(X_train, y_train)
            return pipeline
        cv_splits = usable_folds
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv_splits,
        refit=True,
        n_jobs=-1,
        error_score="raise",
        return_train_score=False,
        verbose=0,
    )
    search.fit(X_train, y_train)
    LOGGER.info(
        "Best %s CV score: %.5f; parameters: %s",
        task, float(search.best_score_), search.best_params_,
    )
    return search.best_estimator_


def train_and_evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    n_splits: int = 5,
    test_size_hours: int | None = None,
    gap_hours: int = 0,
    random_state: int = 42,
) -> tuple[dict[str, object], pd.DataFrame, Pipeline, Pipeline]:
    """Tune two GBDT baselines and compute metrics on the out-of-time test set.

    Classification target: ``alert_occurrence`` (any alert active during the
    target hour). Regression target: ``alert_minutes`` (summed minutes active in
    the region during the target hour, which can exceed 60 for overlap).

    Returns metrics, test predictions, fitted classifier, and fitted regressor.
    Test labels are never used for fitting or model selection.
    """
    train_frame = _canonicalize_model_frame(train, label="train").reset_index(drop=True)
    test_frame = _canonicalize_model_frame(test, label="test").reset_index(drop=True)
    assert_temporal_integrity(train_frame, test_frame)
    feature_columns = get_model_feature_columns(train_frame)
    test_features = get_model_feature_columns(test_frame)
    if feature_columns != test_features:
        raise ValueError("Train and test feature schemas do not match exactly.")
    for split_name, frame in (("train", train_frame), ("test", test_frame)):
        if frame[feature_columns].isna().any().any():
            raise ValueError(
                f"{split_name} features contain missing values; "
                "clean them without future imputation."
            )
        numeric_features = [column for column in feature_columns if column != "region"]
        if not np.isfinite(frame[numeric_features].to_numpy(dtype=float)).all():
            raise ValueError(f"{split_name} contains infinite numeric features.")

    X_train = train_frame[feature_columns]
    X_test = test_frame[feature_columns]
    y_class_train = train_frame[OCCURRENCE_TARGET].astype(int)
    y_class_test = test_frame[OCCURRENCE_TARGET].astype(int)
    y_duration_train = train_frame[DURATION_TARGET].astype(float)
    y_duration_test = test_frame[DURATION_TARGET].astype(float)

    if y_class_train.nunique() < 2:
        raise ValueError(
            "The training period contains only one alert_occurrence class. "
            "Use a longer/more representative training period before "
            "fitting a classifier."
        )
    cv_splits = build_expanding_window_splits(
        train_frame,
        n_splits=n_splits,
        test_size_hours=test_size_hours,
        gap_hours=gap_hours,
    )
    classifier = _search_pipeline(
        _make_pipeline(
            task="classification",
            feature_columns=feature_columns,
            random_state=random_state,
        ),
        X_train=X_train,
        y_train=y_class_train,
        cv_splits=cv_splits,
        scoring="roc_auc",
        task="classification",
    )
    regressor = _search_pipeline(
        _make_pipeline(
            task="regression",
            feature_columns=feature_columns,
            random_state=random_state,
        ),
        X_train=X_train,
        y_train=y_duration_train,
        cv_splits=cv_splits,
        scoring="neg_mean_absolute_error",
        task="regression",
    )

    # The out-of-time test is touched only after all tuning/refitting is complete.
    probability = classifier.predict_proba(X_test)[:, 1]
    class_prediction = (probability >= 0.5).astype("int8")
    duration_prediction = np.maximum(regressor.predict(X_test), 0.0)

    if y_class_test.nunique() == 2:
        roc_auc: float | None = float(roc_auc_score(y_class_test, probability))
    else:
        roc_auc = None
        LOGGER.warning(
            "ROC-AUC is undefined because the test period contains only one class."
        )
    metrics: dict[str, object] = {
        "classification": {
            "target": OCCURRENCE_TARGET,
            "roc_auc": roc_auc,
            "precision_at_0_5": float(
                precision_score(y_class_test, class_prediction, zero_division=0)
            ),
            "recall_at_0_5": float(
                recall_score(y_class_test, class_prediction, zero_division=0)
            ),
            "threshold": 0.5,
            "test_positive_rate": float(y_class_test.mean()),
        },
        "duration_regression": {
            "target": DURATION_TARGET,
            "mae_minutes": float(
                mean_absolute_error(y_duration_test, duration_prediction)
            ),
            "rmse_minutes": float(
                np.sqrt(mean_squared_error(y_duration_test, duration_prediction))
            ),
            "test_mean_alert_minutes": float(y_duration_test.mean()),
        },
        "data": {
            "train_rows": int(len(train_frame)),
            "test_rows": int(len(test_frame)),
            "train_start": train_frame[TIMESTAMP_COLUMN].min().isoformat(),
            "train_end": train_frame[TIMESTAMP_COLUMN].max().isoformat(),
            "test_start": test_frame[TIMESTAMP_COLUMN].min().isoformat(),
            "test_end": test_frame[TIMESTAMP_COLUMN].max().isoformat(),
            "feature_count": int(len(feature_columns)),
            "evaluation_mode": (
                "rolling one-step-ahead; each row uses only earlier buckets"
            ),
        },
    }
    predictions = test_frame[
        [TIMESTAMP_COLUMN, "region", OCCURRENCE_TARGET, DURATION_TARGET]
    ].copy()
    predictions["alert_probability"] = probability
    predictions["alert_prediction_at_0_5"] = class_prediction
    predictions["alert_minutes_prediction"] = duration_prediction
    LOGGER.info("Out-of-time test metrics: %s", json.dumps(metrics, indent=2))
    return metrics, predictions, classifier, regressor


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create command-line parser for model training and evaluation."""
    parser = argparse.ArgumentParser(
        description="Train leakage-aware GBDT alert forecasting baselines."
    )
    parser.add_argument(
        "train_csv", type=Path,
        help="Chronological training dataset from features.py.",
    )
    parser.add_argument(
        "test_csv", type=Path, help="Out-of-time test dataset from features.py."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--cv-test-size-hours", type=int, default=None)
    parser.add_argument("--cv-gap-hours", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Load train/test files, run tuning, save models, predictions, and metrics."""
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    train = read_model_frame(args.train_csv)
    test = read_model_frame(args.test_csv)
    metrics, predictions, classifier, regressor = train_and_evaluate(
        train,
        test,
        n_splits=args.n_splits,
        test_size_hours=args.cv_test_size_hours,
        gap_hours=args.cv_gap_hours,
        random_state=args.random_state,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump(classifier, args.output_dir / "alert_occurrence_gbdt.joblib")
    dump(regressor, args.output_dir / "alert_minutes_gbdt.joblib")
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    with (args.output_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    LOGGER.info("Saved model artifacts to %s", args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
