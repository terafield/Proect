"""End-to-end CLI for the Ukraine Air Raid Alerts forecasting project.

Example (time-series baseline):
    python -m src.pipeline data/raw/alerts.csv \
        --coverage-start 2024-01-01T00:00:00Z \
        --coverage-end 2026-10-15T23:00:00Z \
        --regions "Kyiv Oblast" "Lviv Oblast" \
        --output-dir artifacts/run

Example (with timestamped text events):
    python -m src.pipeline data/raw/alerts.csv \
        --coverage-start 2024-01-01T00:00:00Z \
        --coverage-end 2026-10-15T23:00:00Z \
        --regions "Kyiv Oblast" "Lviv Oblast" \
        --text-events data/raw/public_text.csv \
        --source-coverage-start 2024-01-01T00:00:00Z \
        --output-dir artifacts/run

A canonical neighbor map may be supplied using --neighbors-json. If the graph
is unavailable, --no-neighbor-features disables regional adjacency signals
explicitly instead of silently treating unknown regions as non-neighbors.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd
from joblib import dump

from .data_loader import aggregate_hourly, load_alerts, time_based_split
from .evaluate import evaluate_artifacts
from .features import build_features
from .model import train_and_evaluate
from .text_features import build_point_in_time_text_features, compare_time_series_vs_text

LOGGER = logging.getLogger(__name__)


def _read_neighbor_map(path: Path | None) -> Mapping[str, Sequence[str]] | None:
    """Load a user-supplied region adjacency JSON mapping."""
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Neighbor map not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in neighbor map {path}: {exc}") from exc
    if not isinstance(value, dict) or any(
        not isinstance(key, str)
        or not isinstance(regions, list)
        or any(not isinstance(region, str) for region in regions)
        for key, regions in value.items()
    ):
        raise ValueError(
            "Neighbor JSON must map each region name to a list of region names."
        )
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run hourly aggregation, causal feature engineering, chronological "
            "holdout evaluation, and optional point-in-time text comparison."
        )
    )
    parser.add_argument("alerts", type=Path, help="Raw alert event file (CSV/Parquet/JSON/JSONL).")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/run"))
    parser.add_argument("--coverage-start", default=None, help="Inclusive first observed UTC hour.")
    parser.add_argument("--coverage-end", default=None, help="Inclusive last observed UTC hour.")
    parser.add_argument(
        "--regions", nargs="+", default=None,
        help="Canonical region roster. Quote names containing spaces.",
    )
    parser.add_argument("--naive-timezone", default="UTC")
    parser.add_argument("--duration-unit", default="seconds")
    parser.add_argument("--calendar-timezone", default="Europe/Kyiv")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-time", default=None, help="Explicit start of the test period (UTC).")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--cv-test-size-hours", type=int, default=None)
    parser.add_argument("--cv-gap-hours", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--neighbors-json", type=Path, default=None)
    parser.add_argument(
        "--no-neighbor-features", action="store_true",
        help="Explicitly disable adjacency signals for non-standard region names.",
    )
    parser.add_argument("--text-events", type=Path, default=None, help="Optional timestamped reports CSV.")
    parser.add_argument("--text-timestamp-col", default="text_timestamp")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--text-region-col", default="region")
    parser.add_argument("--source-coverage-start", default=None)
    parser.add_argument("--importance-repeats", type=int, default=3)
    parser.add_argument("--max-importance-rows", type=int, default=2000)
    parser.add_argument("--skip-plots", action="store_true", help="Train/evaluate without chart and importance generation.")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    """Execute the full workflow and return the primary metrics dictionary."""
    if not args.alerts.is_file():
        raise FileNotFoundError(f"Raw alert input not found: {args.alerts}")
    if args.text_events is not None and not args.text_events.is_file():
        raise FileNotFoundError(f"Text event input not found: {args.text_events}")
    if args.neighbors_json is not None and args.no_neighbor_features:
        raise ValueError("Choose either --neighbors-json or --no-neighbor-features, not both.")
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2.")
    if args.importance_repeats < 1 or args.max_importance_rows < 1:
        raise ValueError("Importance repeats and max importance rows must be positive.")

    output = args.output_dir.resolve()
    data_dir = output / "data"
    model_dir = output / "models"
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    alerts = load_alerts(
        args.alerts,
        naive_timezone=args.naive_timezone,
        duration_unit=args.duration_unit,
    )
    hourly = aggregate_hourly(
        alerts,
        coverage_start=args.coverage_start,
        coverage_end=args.coverage_end,
        regions=args.regions,
        naive_timezone=args.naive_timezone,
    )
    hourly.to_csv(data_dir / "hourly.csv", index=False)

    neighbor_map = _read_neighbor_map(args.neighbors_json)
    if args.no_neighbor_features:
        neighbor_map = {str(region): [] for region in hourly["region"].drop_duplicates()}
        LOGGER.warning("Regional adjacency features explicitly disabled by --no-neighbor-features.")
    supervised = build_features(
        hourly,
        neighbors=neighbor_map,
        calendar_timezone=args.calendar_timezone,
        drop_incomplete=True,
    )

    if args.text_events is not None:
        text_events = pd.read_csv(args.text_events)
        supervised = build_point_in_time_text_features(
            supervised,
            text_events,
            text_timestamp_col=args.text_timestamp_col,
            text_col=args.text_col,
            text_region_col=args.text_region_col,
            source_coverage_start=args.source_coverage_start,
        )
        supervised.to_csv(data_dir / "supervised_with_text.csv", index=False)

    train, test = time_based_split(
        supervised,
        test_fraction=args.test_fraction,
        split_time=args.split_time,
    )
    supervised.to_csv(data_dir / "supervised.csv", index=False)
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)

    config = {
        "input_alerts": str(args.alerts.resolve()),
        "input_text_events": str(args.text_events.resolve()) if args.text_events else None,
        "coverage_start": args.coverage_start,
        "coverage_end": args.coverage_end,
        "regions": args.regions,
        "test_fraction": args.test_fraction,
        "split_time": args.split_time,
        "n_splits": args.n_splits,
        "random_state": args.random_state,
        "neighbor_features_disabled": bool(args.no_neighbor_features),
        "neighbors_json": str(args.neighbors_json.resolve()) if args.neighbors_json else None,
        "text_source_coverage_start": args.source_coverage_start,
    }
    (output / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.text_events is None:
        metrics, predictions, classifier, regressor = train_and_evaluate(
            train,
            test,
            n_splits=args.n_splits,
            test_size_hours=args.cv_test_size_hours,
            gap_hours=args.cv_gap_hours,
            random_state=args.random_state,
        )
        dump(classifier, model_dir / "alert_occurrence_gbdt.joblib")
        dump(regressor, model_dir / "alert_minutes_gbdt.joblib")
        predictions.to_csv(output / "test_predictions.csv", index=False)
        (output / "test_metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if not args.skip_plots:
            evaluate_artifacts(
                train,
                test,
                classifier,
                regressor,
                output_dir=output / "evaluation",
                n_repeats=args.importance_repeats,
                max_importance_rows=args.max_importance_rows,
                random_state=args.random_state,
            )
    else:
        metrics, predictions, fitted = compare_time_series_vs_text(
            train,
            test,
            n_splits=args.n_splits,
            cv_test_size_hours=args.cv_test_size_hours,
            cv_gap_hours=args.cv_gap_hours,
            random_state=args.random_state,
        )
        predictions.to_csv(output / "test_comparison_predictions.csv", index=False)
        (output / "comparison_metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        for model_name in ("time_series_only", "time_series_plus_text"):
            parts = fitted[model_name]
            variant_dir = model_dir / model_name
            variant_dir.mkdir(parents=True, exist_ok=True)
            classifier = parts["classifier"]
            regressor = parts["regressor"]
            dump(classifier, variant_dir / "alert_occurrence_gbdt.joblib")
            dump(regressor, variant_dir / "alert_minutes_gbdt.joblib")
            if not args.skip_plots:
                if model_name == "time_series_only":
                    text_columns = [column for column in train.columns if column.startswith("text_")]
                    eval_train = train.drop(columns=text_columns, errors="ignore")
                    eval_test = test.drop(columns=text_columns, errors="ignore")
                else:
                    eval_train, eval_test = train, test
                evaluate_artifacts(
                    eval_train,
                    eval_test,
                    classifier,
                    regressor,
                    output_dir=output / "evaluation" / model_name,
                    n_repeats=args.importance_repeats,
                    max_importance_rows=args.max_importance_rows,
                    random_state=args.random_state,
                )

    LOGGER.info("Pipeline finished successfully. Outputs: %s", output)
    LOGGER.info("Primary metrics: %s", json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
