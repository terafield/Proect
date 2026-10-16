"""Out-of-time model evaluation, drift diagnostics, plots, and feature importance.

This module deliberately does not fit or tune models. It loads models trained by
``src.model`` and evaluates them on a frozen chronological test frame. The test
set is used only for reporting, plotting, and permutation-importance diagnostics;
feature importance must not be fed back into tuning without a new untouched test
period.

Example::

    python -m src.evaluate \
        data/processed/model_data/train.csv \
        data/processed/model_data/test.csv \
        artifacts/baseline/alert_occurrence_gbdt.joblib \
        artifacts/baseline/alert_minutes_gbdt.joblib \
        --output-dir artifacts/evaluation

Dependencies: pandas, numpy, matplotlib, scikit-learn, joblib.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from .data_loader import FEATURE_CUTOFF_COLUMN, TIMESTAMP_COLUMN
    from .features import get_model_feature_columns
    from .model import (
        DURATION_TARGET,
        OCCURRENCE_TARGET,
        assert_temporal_integrity,
    )
except ImportError:  # pragma: no cover - direct-script compatibility.
    from data_loader import FEATURE_CUTOFF_COLUMN, TIMESTAMP_COLUMN  # type: ignore[no-redef]
    from features import get_model_feature_columns  # type: ignore[no-redef]
    from model import (  # type: ignore[no-redef]
        DURATION_TARGET,
        OCCURRENCE_TARGET,
        assert_temporal_integrity,
    )

LOGGER = logging.getLogger(__name__)
KEY_COLUMNS = (TIMESTAMP_COLUMN, "region")


def _read_frame(path: str | Path, *, label: str) -> pd.DataFrame:
    """Load a processed hourly feature CSV and normalize time columns to UTC."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {file_path}")
    frame = pd.read_csv(file_path)
    required = {
        TIMESTAMP_COLUMN,
        FEATURE_CUTOFF_COLUMN,
        "region",
        DURATION_TARGET,
        OCCURRENCE_TARGET,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} frame is missing required columns: {missing}")
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(
        frame[TIMESTAMP_COLUMN], utc=True, errors="coerce"
    )
    frame[FEATURE_CUTOFF_COLUMN] = pd.to_datetime(
        frame[FEATURE_CUTOFF_COLUMN], utc=True, errors="coerce"
    )
    if frame[[TIMESTAMP_COLUMN, FEATURE_CUTOFF_COLUMN]].isna().any().any():
        raise ValueError(f"{label} frame contains invalid timestamp metadata.")
    if frame["region"].isna().any() or frame["region"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{label} frame contains blank region labels.")
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"{label} frame contains duplicate region/timestamp rows.")
    if frame[[DURATION_TARGET, OCCURRENCE_TARGET]].isna().any().any():
        raise ValueError(f"{label} frame contains missing targets.")
    if (frame[DURATION_TARGET] < 0).any():
        raise ValueError(f"{label} frame contains negative alert-minute targets.")
    if not frame[OCCURRENCE_TARGET].isin([0, 1]).all():
        raise ValueError(f"{label} occurrence target must contain only 0/1 values.")
    frame = frame.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)
    return frame


def _assert_inputs(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    """Validate chronology, feature schemas, and strictly finite model inputs."""
    assert_temporal_integrity(train, test)
    train_features = get_model_feature_columns(train)
    test_features = get_model_feature_columns(test)
    if not train_features:
        raise ValueError("No whitelisted model features were found.")
    if train_features != test_features:
        raise ValueError("Train/test feature schemas differ; retrain with the same pipeline.")
    for label, frame in (("train", train), ("test", test)):
        if frame[train_features].isna().any().any():
            raise ValueError(
                f"{label} model features contain missing values. Drop incomplete warm-up "
                "rows or add a fold-fitted imputer inside the model Pipeline."
            )
        numeric = [column for column in train_features if column != "region"]
        numeric_values = frame[numeric].to_numpy(dtype=np.float64)
        if not np.isfinite(numeric_values).all():
            raise ValueError(f"{label} model features contain non-finite numeric values.")
    return train_features


def _target_profile(frame: pd.DataFrame) -> dict[str, Any]:
    """Describe zero inflation and the heavy tail of alert-minute targets."""
    minutes = frame[DURATION_TARGET].astype(float)
    positives = minutes[minutes > 0]
    return {
        "rows": int(len(frame)),
        "alert_occurrence_rate": float(frame[OCCURRENCE_TARGET].mean()),
        "zero_alert_minutes_fraction": float(minutes.eq(0).mean()),
        "mean_alert_minutes": float(minutes.mean()),
        "median_alert_minutes": float(minutes.median()),
        "positive_target_rows": int((minutes > 0).sum()),
        "positive_minutes_p95": (
            float(positives.quantile(0.95)) if not positives.empty else None
        ),
        "positive_minutes_p99": (
            float(positives.quantile(0.99)) if not positives.empty else None
        ),
        "max_alert_minutes": float(minutes.max()),
    }


def _distribution_drift_report(
    train: pd.DataFrame, test: pd.DataFrame
) -> dict[str, Any]:
    """Compare train/test target profiles without modifying either dataset."""
    train_profile = _target_profile(train)
    test_profile = _target_profile(test)
    train_p95 = train_profile["positive_minutes_p95"]
    test_p95 = test_profile["positive_minutes_p95"]
    train_rate = train_profile["alert_occurrence_rate"]
    return {
        "training": train_profile,
        "out_of_time_test": test_profile,
        "changes": {
            "alert_occurrence_rate_absolute": (
                test_profile["alert_occurrence_rate"] - train_rate
            ),
            "zero_alert_fraction_absolute": (
                test_profile["zero_alert_minutes_fraction"]
                - train_profile["zero_alert_minutes_fraction"]
            ),
            "positive_minutes_p95_ratio": (
                float(test_p95 / train_p95)
                if train_p95 not in (None, 0.0) and test_p95 is not None
                else None
            ),
            "max_alert_minutes_ratio": (
                float(test_profile["max_alert_minutes"] / train_profile["max_alert_minutes"])
                if train_profile["max_alert_minutes"] > 0
                else None
            ),
        },
        "interpretation_note": (
            "These are post-hoc monitoring diagnostics, not reasons to retune on the "
            "current holdout. Large changes indicate target drift or a different mix "
            "of quiet and high-intensity periods and warrant a new future validation window."
        ),
    }


def _weekly_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return weekly target and error diagnostics to expose time-varying drift."""
    frame = predictions.copy()
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=True)
    frame["absolute_duration_error"] = (
        frame[DURATION_TARGET] - frame["alert_minutes_prediction"]
    ).abs()
    frame["squared_duration_error"] = (
        frame[DURATION_TARGET] - frame["alert_minutes_prediction"]
    ) ** 2
    frame["classification_correct_at_0_5"] = (
        frame[OCCURRENCE_TARGET].astype(int)
        == frame["alert_prediction_at_threshold"].astype(int)
    ).astype(float)
    grouped = frame.groupby(pd.Grouper(key=TIMESTAMP_COLUMN, freq="W-MON"), dropna=True)
    weekly = grouped.agg(
        rows=(DURATION_TARGET, "size"),
        alert_occurrence_rate=(OCCURRENCE_TARGET, "mean"),
        zero_alert_minutes_fraction=(DURATION_TARGET, lambda values: float(values.eq(0).mean())),
        mean_alert_minutes=(DURATION_TARGET, "mean"),
        p95_alert_minutes=(DURATION_TARGET, lambda values: float(values.quantile(0.95))),
        max_alert_minutes=(DURATION_TARGET, "max"),
        duration_mae=("absolute_duration_error", "mean"),
        duration_rmse=("squared_duration_error", lambda values: float(np.sqrt(values.mean()))),
        occurrence_accuracy=("classification_correct_at_0_5", "mean"),
    ).reset_index()
    return weekly


def _compute_predictions(
    test: pd.DataFrame,
    feature_columns: list[str],
    classifier: Any,
    regressor: Any,
    *,
    threshold: float,
) -> pd.DataFrame:
    """Generate paired classifier and regressor predictions on the test frame."""
    X_test = test[feature_columns]
    if not hasattr(classifier, "predict_proba"):
        raise TypeError("Classifier artifact must expose predict_proba().")
    probability_matrix = classifier.predict_proba(X_test)
    classes = list(getattr(classifier, "classes_", []))
    if len(classes) != probability_matrix.shape[1] or 1 not in classes:
        raise ValueError("Classifier artifact does not expose a positive class labelled 1.")
    probability = probability_matrix[:, classes.index(1)]
    if not np.isfinite(probability).all():
        raise ValueError("Classifier produced non-finite alert probabilities.")
    raw_duration = np.asarray(regressor.predict(X_test), dtype=np.float64)
    if not np.isfinite(raw_duration).all():
        raise ValueError("Regressor produced non-finite duration predictions.")
    duration = np.maximum(raw_duration, 0.0)
    predictions = test[
        [TIMESTAMP_COLUMN, "region", OCCURRENCE_TARGET, DURATION_TARGET]
    ].copy()
    predictions["alert_probability"] = probability
    predictions["alert_prediction_at_threshold"] = (probability >= threshold).astype("int8")
    predictions["alert_minutes_prediction"] = duration
    return predictions


def _metric_report(
    predictions: pd.DataFrame, train: pd.DataFrame, threshold: float
) -> dict[str, Any]:
    """Compute overall and high-severity out-of-time metrics."""
    actual_occurrence = predictions[OCCURRENCE_TARGET].astype(int)
    probability = predictions["alert_probability"].to_numpy(dtype=float)
    predicted_occurrence = predictions["alert_prediction_at_threshold"].astype(int)
    actual_minutes = predictions[DURATION_TARGET].astype(float)
    predicted_minutes = predictions["alert_minutes_prediction"].astype(float)
    both_classes = actual_occurrence.nunique() == 2
    if both_classes:
        roc_auc: float | None = float(roc_auc_score(actual_occurrence, probability))
        average_precision: float | None = float(
            average_precision_score(actual_occurrence, probability)
        )
        brier: float | None = float(brier_score_loss(actual_occurrence, probability))
    else:
        roc_auc = None
        average_precision = None
        brier = None

    positive_train = train.loc[train[DURATION_TARGET] > 0, DURATION_TARGET].astype(float)
    threshold_minutes: float | None = (
        float(max(positive_train.quantile(0.95), 1.0)) if not positive_train.empty else None
    )
    if threshold_minutes is not None:
        high_mask = actual_minutes >= threshold_minutes
    else:
        high_mask = pd.Series(False, index=predictions.index)
    high_count = int(high_mask.sum())
    if high_count:
        high_actual = actual_minutes.loc[high_mask]
        high_predicted = predicted_minutes.loc[high_mask]
        high_mae: float | None = float(mean_absolute_error(high_actual, high_predicted))
        high_rmse: float | None = float(np.sqrt(mean_squared_error(high_actual, high_predicted)))
        underprediction_fraction: float | None = float((high_predicted < high_actual).mean())
        prediction_to_actual_median: float | None = float(
            np.median(high_predicted.to_numpy() / np.maximum(high_actual.to_numpy(), 1e-9))
        )
    else:
        high_mae = high_rmse = underprediction_fraction = prediction_to_actual_median = None

    zero_mask = actual_minutes.eq(0)
    return {
        "classification": {
            "roc_auc": roc_auc,
            "average_precision": average_precision,
            "brier_score": brier,
            "precision": float(
                precision_score(actual_occurrence, predicted_occurrence, zero_division=0)
            ),
            "recall": float(
                recall_score(actual_occurrence, predicted_occurrence, zero_division=0)
            ),
            "threshold": float(threshold),
            "actual_alert_rate": float(actual_occurrence.mean()),
            "false_positive_rate_on_zero_duration_rows": (
                float(predicted_occurrence.loc[zero_mask].mean()) if zero_mask.any() else None
            ),
        },
        "duration": {
            "mae_minutes": float(mean_absolute_error(actual_minutes, predicted_minutes)),
            "rmse_minutes": float(np.sqrt(mean_squared_error(actual_minutes, predicted_minutes))),
            "actual_zero_fraction": float(actual_minutes.eq(0).mean()),
            "mae_on_positive_rows": (
                float(
                    mean_absolute_error(
                        actual_minutes.loc[actual_minutes > 0],
                        predicted_minutes.loc[actual_minutes > 0],
                    )
                )
                if (actual_minutes > 0).any() else None
            ),
            "high_severity_definition_train_positive_p95_minutes": threshold_minutes,
            "high_severity_rows": high_count,
            "high_severity_mae_minutes": high_mae,
            "high_severity_rmse_minutes": high_rmse,
            "high_severity_underprediction_fraction": underprediction_fraction,
            "high_severity_median_prediction_to_actual_ratio": prediction_to_actual_median,
        },
        "note": (
            "High-severity threshold is computed from positive training targets only. "
            "These diagnostics quantify performance on spikes; they do not guarantee "
            "the tree model can extrapolate to events larger than its training experience."
        ),
    }


def _plot_aggregate_timeseries(predictions: pd.DataFrame, output_dir: Path) -> None:
    """Plot aggregated alert-minute targets and occurrence across regions."""
    grouped = predictions.groupby(TIMESTAMP_COLUMN, sort=True).agg(
        actual_alert_minutes=(DURATION_TARGET, "sum"),
        predicted_alert_minutes=("alert_minutes_prediction", "sum"),
        actual_active_regions=(OCCURRENCE_TARGET, "sum"),
        expected_active_regions=("alert_probability", "sum"),
    )
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(grouped.index, grouped["actual_alert_minutes"], label="Actual", linewidth=1.3)
    ax.plot(grouped.index, grouped["predicted_alert_minutes"], label="Predicted", linewidth=1.2)
    ax.set_title("Out-of-time test: aggregate alert-minutes across regions")
    ax.set_xlabel("Forecast bucket (UTC)")
    ax.set_ylabel("Sum of regional alert-minutes")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "actual_vs_predicted_alert_minutes.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        grouped.index, grouped["actual_active_regions"],
        label="Actual active regions", linewidth=1.3,
    )
    ax.plot(
        grouped.index, grouped["expected_active_regions"],
        label="Sum of predicted probabilities", linewidth=1.2,
    )
    ax.set_title("Out-of-time test: alert occurrence across regions")
    ax.set_xlabel("Forecast bucket (UTC)")
    ax.set_ylabel("Active-region count / expected count")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "actual_vs_predicted_occurrence.png", dpi=160)
    plt.close(fig)


def _plot_regional_panels(
    predictions: pd.DataFrame, output_dir: Path, *, max_regions: int
) -> None:
    """Plot actual and predicted alert-minutes for the highest-volume regions."""
    totals = predictions.groupby("region")[DURATION_TARGET].sum().sort_values(ascending=False)
    regions = totals.head(max_regions).index.tolist()
    if not regions:
        return
    ncols = 2
    nrows = int(np.ceil(len(regions) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(15, max(4, nrows * 3.6)), squeeze=False
    )
    for index, region in enumerate(regions):
        ax = axes[index // ncols][index % ncols]
        group = predictions.loc[predictions["region"].eq(region)].sort_values(TIMESTAMP_COLUMN)
        ax.plot(group[TIMESTAMP_COLUMN], group[DURATION_TARGET], label="Actual", linewidth=1.0)
        ax.plot(
            group[TIMESTAMP_COLUMN], group["alert_minutes_prediction"],
            label="Predicted", linewidth=1.0,
        )
        ax.set_title(str(region))
        ax.set_ylabel("Alert-minutes")
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis="x", labelrotation=25)
        if index == 0:
            ax.legend()
    for index in range(len(regions), nrows * ncols):
        axes[index // ncols][index % ncols].set_visible(False)
    fig.suptitle(
        "Regional actual vs predicted alert-minutes (highest test volume)", y=1.01
    )
    fig.tight_layout()
    fig.savefig(output_dir / "regional_actual_vs_predicted.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _feature_importance(
    estimator: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    task: str,
    output_dir: Path,
    top_n: int,
    n_repeats: int,
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    """Compute permutation importance on a bounded deterministic test sample."""
    if len(X_test) > max_rows:
        chosen = X_test.sample(n=max_rows, random_state=random_state).index
        X_eval = X_test.loc[chosen]
        y_eval = y_test.loc[chosen]
    else:
        X_eval, y_eval = X_test, y_test

    if task == "classification":
        scoring = "roc_auc" if y_eval.nunique() == 2 else "accuracy"
    elif task == "regression":
        scoring = "neg_mean_absolute_error"
    else:
        raise ValueError(f"Unknown importance task: {task}")

    result = permutation_importance(
        estimator,
        X_eval,
        y_eval,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    importance = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False, kind="stable")
    file_stem = "occurrence" if task == "classification" else "duration"
    importance.to_csv(output_dir / f"permutation_importance_{file_stem}.csv", index=False)
    top = importance.head(top_n).sort_values("importance_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.34)))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    ax.set_title(
        f"Permutation feature importance: {task}\n"
        f"Scoring={scoring}; calculated on out-of-time test (interpretation only)"
    )
    ax.set_xlabel("Mean score decrease after permutation")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"permutation_importance_{file_stem}.png", dpi=160)
    plt.close(fig)
    return importance


def evaluate_artifacts(
    train: pd.DataFrame,
    test: pd.DataFrame,
    classifier: Any,
    regressor: Any,
    *,
    output_dir: str | Path,
    threshold: float = 0.5,
    top_n_features: int = 20,
    n_repeats: int = 5,
    max_importance_rows: int = 5000,
    max_regions: int = 8,
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluate saved estimators, save charts, metrics, and drift reports."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1.")
    if top_n_features < 1 or n_repeats < 1 or max_importance_rows < 1 or max_regions < 1:
        raise ValueError("Plot/importance limits and repeat counts must be positive.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    feature_columns = _assert_inputs(train, test)
    predictions = _compute_predictions(
        test, feature_columns, classifier, regressor, threshold=threshold
    )
    predictions.to_csv(output / "evaluation_predictions.csv", index=False)
    metrics = _metric_report(predictions, train, threshold)
    metrics["data"] = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_start": train[TIMESTAMP_COLUMN].min().isoformat(),
        "train_end": train[TIMESTAMP_COLUMN].max().isoformat(),
        "test_start": test[TIMESTAMP_COLUMN].min().isoformat(),
        "test_end": test[TIMESTAMP_COLUMN].max().isoformat(),
        "n_model_features": int(len(feature_columns)),
        "evaluation_is_out_of_time": True,
        "feature_importance_source": (
            "permutation importance on a deterministic test sample; diagnostic only"
        ),
    }
    drift = _distribution_drift_report(train, test)
    with (output / "evaluation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    with (output / "target_drift_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(drift, handle, indent=2, ensure_ascii=False)

    _weekly_diagnostics(predictions).to_csv(output / "weekly_diagnostics.csv", index=False)
    _plot_aggregate_timeseries(predictions, output)
    _plot_regional_panels(predictions, output, max_regions=max_regions)

    X_test = test[feature_columns]
    classification_importance = _feature_importance(
        classifier,
        X_test,
        test[OCCURRENCE_TARGET].astype(int),
        task="classification",
        output_dir=output,
        top_n=top_n_features,
        n_repeats=n_repeats,
        max_rows=max_importance_rows,
        random_state=random_state,
    )
    regression_importance = _feature_importance(
        regressor,
        X_test,
        test[DURATION_TARGET].astype(float),
        task="regression",
        output_dir=output,
        top_n=top_n_features,
        n_repeats=n_repeats,
        max_rows=max_importance_rows,
        random_state=random_state,
    )
    metrics["top_occurrence_features"] = classification_importance.head(top_n_features)[
        ["feature", "importance_mean", "importance_std"]
    ].to_dict(orient="records")
    metrics["top_duration_features"] = regression_importance.head(top_n_features)[
        ["feature", "importance_mean", "importance_std"]
    ].to_dict(orient="records")
    with (output / "evaluation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    LOGGER.info("Evaluation complete. Outputs saved in %s", output.resolve())
    return metrics


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for saved-model evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate out-of-time forecast plots, feature importance, and drift diagnostics."
        )
    )
    parser.add_argument(
        "train_csv", type=Path, help="Processed chronological training feature frame."
    )
    parser.add_argument(
        "test_csv", type=Path, help="Processed out-of-time test feature frame."
    )
    parser.add_argument(
        "classifier_model", type=Path,
        help="Saved occurrence classifier joblib artifact.",
    )
    parser.add_argument(
        "regressor_model", type=Path,
        help="Saved alert-minutes regressor joblib artifact.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-n-features", type=int, default=20)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--max-importance-rows", type=int, default=5000)
    parser.add_argument("--max-regions", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Load data/artifacts and write evaluation deliverables."""
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    for path in (args.classifier_model, args.regressor_model):
        if not path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {path}")
    train = _read_frame(args.train_csv, label="Training")
    test = _read_frame(args.test_csv, label="Test")
    classifier = load(args.classifier_model)
    regressor = load(args.regressor_model)
    metrics = evaluate_artifacts(
        train,
        test,
        classifier,
        regressor,
        output_dir=args.output_dir,
        threshold=args.threshold,
        top_n_features=args.top_n_features,
        n_repeats=args.n_repeats,
        max_importance_rows=args.max_importance_rows,
        max_regions=args.max_regions,
        random_state=args.random_state,
    )
    LOGGER.info("Out-of-time metrics: %s", json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
