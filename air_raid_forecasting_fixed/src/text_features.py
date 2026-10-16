"""Point-in-time text features for leakage-safe hourly alert forecasting.

The input text timestamp MUST be the first time the report was publicly
available to the forecasting system (for example, ``first_seen_at``), not a
retrospective event time extracted from the article body. Reports are joined
strictly before each forecast reference timestamp using ``merge_asof`` with
``direction='backward'`` and ``allow_exact_matches=False``.

Supported text signals include rolling report/keyword counts, a transparent
rule-based intensity score, time since the latest prior report, and optional
precomputed embeddings in columns named ``embedding_0``, ``embedding_1``, etc.
This module does not download a language model or fit an embedding model.
Any learned text encoder must be trained using training-fold text only.

Typical use::

    python -m src.text_features data/processed/model_data/supervised.csv \
        data/raw/public_text.csv --output-dir artifacts/text_comparison

Dependencies: pandas, numpy, scikit-learn, joblib (the last two are used for
model comparison and are already required by ``src.model``).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from joblib import dump

try:
    from .data_loader import (
        FEATURE_CUTOFF_COLUMN,
        TIMESTAMP_COLUMN,
        time_based_split,
    )
    from .features import get_model_feature_columns, normalize_region_name
except ImportError:  # pragma: no cover - direct-script compatibility.
    from data_loader import (  # type: ignore[no-redef]
        FEATURE_CUTOFF_COLUMN,
        TIMESTAMP_COLUMN,
        time_based_split,
    )
    from features import get_model_feature_columns, normalize_region_name  # type: ignore[no-redef]

LOGGER = logging.getLogger(__name__)
TEXT_TIMESTAMP_COLUMN = "text_timestamp"
TEXT_BODY_COLUMN = "text"
FORECAST_REFERENCE_COLUMN = "forecast_reference_timestamp"
DEFAULT_WINDOWS_HOURS: tuple[int, ...] = (1, 6, 24, 168)
GLOBAL_REGION_LABELS = {
    "", "*", "all", "global", "nationwide", "ukraine", "україна",
    "загальнонаціональний",
}

# Transparent, editable keyword dictionaries. Match on case-folded substrings;
# these are heuristic signals, not a classifier or a claim about actual threat.
KEYWORD_GROUPS: dict[str, tuple[str, ...]] = {
    "missile": (
        "missile", "ballistic", "cruise missile", "ракета", "ракетн",
        "балістич", "крилат",
    ),
    "drone": (
        "drone", "uav", "shahed", "шахед", "дрон", "безпілот", "бпла",
    ),
    "air_activity": (
        "aircraft", "aviation", "airfield", "takeoff", "take-off",
        "flight activity", "launch", "launched", "launching", "літак",
        "авіац", "аеродром", "зліт", "злет", "пуск", "запуск",
        "бомбардув",
    ),
    "attack": (
        "attack", "strike", "explosion", "blast", "shelling", "обстріл",
        "вибух", "удар", "атака", "атакує",
    ),
    "official_alert": (
        "air raid", "air alert", "air-raid", "повітрян", "тривог", "укриття",
    ),
}

# Used only to create a bounded feature from the keyword groups above.
INTENSITY_WEIGHTS: dict[str, float] = {
    "missile": 3.0,
    "drone": 3.0,
    "air_activity": 2.0,
    "attack": 2.0,
    "official_alert": 1.0,
}


def _parse_utc(series: pd.Series, *, field_name: str) -> pd.Series:
    """Parse a timestamp column to UTC and reject invalid/missing values."""
    result = pd.to_datetime(series, utc=True, errors="coerce")
    if result.isna().any():
        examples = series.loc[result.isna()].head(5).tolist()
        raise ValueError(f"{field_name} contains missing/invalid timestamps: {examples}")
    return result


def _region_key(value: object) -> str:
    """Normalize region labels consistently with the time-series feature graph."""
    return normalize_region_name(value)


def _is_global_region(value: object) -> bool:
    """Return whether a report is nationwide/unscoped."""
    if value is None or pd.isna(value):
        return True
    return _region_key(value) in GLOBAL_REGION_LABELS


def _detect_embedding_columns(text_events: pd.DataFrame) -> list[str]:
    """Return optional precomputed embedding columns in numeric-index order."""
    pattern = re.compile(r"^embedding_(\d+)$")
    indexed: list[tuple[int, str]] = []
    for column in text_events.columns:
        match = pattern.match(str(column))
        if match:
            indexed.append((int(match.group(1)), str(column)))
    indexed.sort()
    if indexed and [index for index, _ in indexed] != list(range(len(indexed))):
        raise ValueError(
            "Embedding columns must be consecutively named embedding_0..embedding_N."
        )
    return [column for _, column in indexed]


def extract_event_signals(
    text_events: pd.DataFrame,
    *,
    text_timestamp_col: str = TEXT_TIMESTAMP_COLUMN,
    text_col: str = TEXT_BODY_COLUMN,
) -> tuple[pd.DataFrame, list[str]]:
    """Parse timestamped text and extract report-level keyword/intensity signals.

    Args:
        text_events: One row per report, optionally containing ``region`` and
            precomputed numeric embeddings named ``embedding_0`` etc.
        text_timestamp_col: Availability/first-observed timestamp field.
        text_col: Main report text field.

    Returns:
        A copy of the event dataframe with normalized timestamp, keyword flags,
        ``_text_intensity`` and optional embedding fields validated as numeric,
        plus a list of discovered embedding columns.
    """
    missing = {text_timestamp_col, text_col} - set(text_events.columns)
    if missing:
        raise ValueError(f"Text events are missing required columns: {sorted(missing)}")
    if text_events.empty:
        LOGGER.warning(
            "The text-event table is empty; text features will be zero until "
            "reports become available. Verify source coverage before modeling."
        )

    frame = text_events.copy().reset_index(drop=True)
    frame["_text_timestamp_utc"] = _parse_utc(
        frame[text_timestamp_col], field_name=text_timestamp_col
    )
    frame["_text_body_normalized"] = frame[text_col].fillna("").astype(str).str.casefold()

    for group_name, terms in KEYWORD_GROUPS.items():
        # Escape keyword strings to avoid treating punctuation as regex syntax.
        pattern = "|".join(re.escape(term.casefold()) for term in terms)
        frame[f"_signal_{group_name}"] = (
            frame["_text_body_normalized"].str.contains(
            pattern, regex=True, na=False
        )
        ).astype("int8")

    score = np.zeros(len(frame), dtype=np.float64)
    for group_name, weight in INTENSITY_WEIGHTS.items():
        score += frame[f"_signal_{group_name}"].to_numpy(dtype=np.float64) * weight
    frame["_text_intensity"] = np.clip(score, 0.0, 5.0)
    frame["_high_intensity"] = frame["_text_intensity"].ge(2.0).astype("int8")
    frame["_report_count"] = 1.0

    embedding_columns = _detect_embedding_columns(frame)
    if embedding_columns:
        for column in embedding_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        embedding_values = frame[embedding_columns].to_numpy(dtype=np.float64)
        if not np.isfinite(embedding_values).all():
            raise ValueError("Precomputed embedding columns contain missing/non-finite values.")
    return frame, embedding_columns


def _lookup_cumulative(
    query_times: pd.Series,
    timeline: pd.DataFrame,
    *,
    cumulative_columns: Sequence[str],
) -> pd.DataFrame:
    """Look up cumulative features strictly before each query timestamp."""
    left = pd.DataFrame({
        "_row_id": np.arange(len(query_times), dtype=np.int64),
        "_query_timestamp": pd.to_datetime(query_times, utc=True).reset_index(drop=True),
    }).sort_values("_query_timestamp", kind="stable")
    right = timeline[["_event_timestamp", *cumulative_columns]].sort_values(
        "_event_timestamp", kind="stable"
    )

    if right.empty:
        result = pd.DataFrame(index=np.arange(len(query_times)), columns=cumulative_columns)
        result.loc[:, :] = 0.0
        result["_matched_event_timestamp"] = pd.Series(
            pd.NaT, index=result.index, dtype="datetime64[ns, UTC]"
        )
        return result

    joined = pd.merge_asof(
        left,
        right,
        left_on="_query_timestamp",
        right_on="_event_timestamp",
        direction="backward",
        allow_exact_matches=False,
    )
    joined = joined.sort_values("_row_id", kind="stable").reset_index(drop=True)
    joined[list(cumulative_columns)] = joined[list(cumulative_columns)].fillna(0.0)
    joined["_matched_event_timestamp"] = joined["_event_timestamp"]
    return joined[[*cumulative_columns, "_matched_event_timestamp"]]


def _build_region_timeline(
    event_frame: pd.DataFrame,
    *,
    embedding_columns: Sequence[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate same-time reports and build cumulative event counters."""
    signal_columns = [f"_signal_{name}" for name in KEYWORD_GROUPS]
    additive_columns = [
        "_report_count", *signal_columns, "_text_intensity", "_high_intensity",
    ]
    event_frame = event_frame.copy()
    for column in additive_columns:
        event_frame[column] = pd.to_numeric(event_frame[column], errors="raise")

    if embedding_columns:
        event_frame["_embedding_count"] = 1.0
        for column in embedding_columns:
            event_frame[f"_embedding_sum_{column}"] = event_frame[column].astype(float)
            additive_columns.append(f"_embedding_sum_{column}")
        additive_columns.append("_embedding_count")

    event_frame["_event_timestamp"] = event_frame["_text_timestamp_utc"]
    aggregated = (
        event_frame.groupby("_event_timestamp", sort=True)[additive_columns]
        .sum()
        .sort_index()
        .reset_index()
    )

    cumulative_columns: list[str] = []
    for column in additive_columns:
        cumulative_column = f"_cum{column}"
        aggregated[cumulative_column] = aggregated[column].cumsum()
        cumulative_columns.append(cumulative_column)
    return aggregated, cumulative_columns


def build_point_in_time_text_features(
    time_series: pd.DataFrame,
    text_events: pd.DataFrame,
    *,
    text_timestamp_col: str = TEXT_TIMESTAMP_COLUMN,
    text_col: str = TEXT_BODY_COLUMN,
    text_region_col: str = "region",
    reference_timestamp_col: str | None = None,
    windows_hours: Sequence[int] = DEFAULT_WINDOWS_HOURS,
    source_coverage_start: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Join causal text event features onto each region/forecast hour.

    Every rolling text window contains reports in ``[forecast_time - window,
    forecast_time)``. In particular, a report whose timestamp exactly equals
    the forecast reference is excluded. Global reports (missing region or a
    common global label) are available to every region; scoped reports are
    available only to their matching region.

    ``source_coverage_start`` is optional metadata documenting when the text
    feed became reliably complete. When supplied, window coverage fractions
    help distinguish genuinely quiet periods from a not-yet-observed archive.

    Raises:
        AssertionError: If a merged feature uses a report at/after its forecast.
        ValueError: For malformed timestamps, unmatched region scopes, invalid
            windows, duplicate panel keys, or invalid embeddings.
    """
    if time_series.empty:
        raise ValueError("The time-series feature frame is empty.")
    windows = sorted(set(windows_hours))
    if not windows or any(not isinstance(w, int) or w <= 0 for w in windows):
        raise ValueError("windows_hours must contain positive integer hour values.")

    required = {TIMESTAMP_COLUMN, "region", FEATURE_CUTOFF_COLUMN}
    missing = sorted(required - set(time_series.columns))
    if missing:
        raise ValueError(
            f"Time-series frame is missing {missing}; pass the supervised feature frame."
        )
    if reference_timestamp_col is None:
        reference_timestamp_col = (
            FORECAST_REFERENCE_COLUMN
            if FORECAST_REFERENCE_COLUMN in time_series.columns
            else TIMESTAMP_COLUMN
        )
    if reference_timestamp_col not in time_series.columns:
        raise ValueError(f"Missing forecast reference column: {reference_timestamp_col!r}")

    base = time_series.copy().reset_index(drop=True)
    existing_text_features = [
        column for column in base.columns if column.startswith("text_")
    ]
    if existing_text_features:
        raise ValueError(
            "Time-series frame already contains text features. Join text exactly "
            "once to avoid duplicate or stale feature columns."
        )
    base["_text_join_row_id"] = np.arange(len(base), dtype=np.int64)
    base[TIMESTAMP_COLUMN] = _parse_utc(
        base[TIMESTAMP_COLUMN], field_name=TIMESTAMP_COLUMN
    )
    base[FEATURE_CUTOFF_COLUMN] = _parse_utc(
        base[FEATURE_CUTOFF_COLUMN], field_name=FEATURE_CUTOFF_COLUMN
    )
    base[FORECAST_REFERENCE_COLUMN] = _parse_utc(
        base[reference_timestamp_col], field_name=reference_timestamp_col
    )
    if base[[TIMESTAMP_COLUMN, "region"]].duplicated().any():
        raise ValueError("Time-series frame has duplicate region/timestamp keys.")
    if not (base[FEATURE_CUTOFF_COLUMN] < base[FORECAST_REFERENCE_COLUMN]).all():
        raise AssertionError(
            "Time-series features must be available strictly before each forecast reference."
        )
    if not base[FORECAST_REFERENCE_COLUMN].eq(base[TIMESTAMP_COLUMN]).all():
        raise ValueError(
            "The current time-series lag builder forecasts the target bucket from its "
            "start boundary, so forecast_reference_timestamp must equal bucket_start. "
            "Earlier forecast references require rebuilding historical features from "
            "only buckets completed before that earlier reference; later references "
            "would not be forecasts for the declared target bucket."
        )

    event_frame, embedding_columns = extract_event_signals(
        text_events,
        text_timestamp_col=text_timestamp_col,
        text_col=text_col,
    )
    if text_region_col in event_frame.columns:
        event_frame["_scope_is_global"] = event_frame[text_region_col].map(_is_global_region)
        event_frame["_scope_region_key"] = event_frame[text_region_col].map(
            lambda value: "" if _is_global_region(value) else _region_key(value)
        )
    else:
        event_frame["_scope_is_global"] = True
        event_frame["_scope_region_key"] = ""

    actual_region_values = base["region"].astype(str).drop_duplicates().tolist()
    normalized_region_to_actual: dict[str, str] = {}
    for actual in actual_region_values:
        key = _region_key(actual)
        if key in normalized_region_to_actual and normalized_region_to_actual[key] != actual:
            raise ValueError(f"Ambiguous region labels after normalization: {actual!r}.")
        normalized_region_to_actual[key] = actual
    unknown_scopes = sorted(
        set(event_frame.loc[~event_frame["_scope_is_global"], "_scope_region_key"])
        - set(normalized_region_to_actual)
    )
    if unknown_scopes:
        raise ValueError(
            "Text event region labels do not match the time-series region labels: "
            f"{unknown_scopes}. Normalize/map the labels before joining."
        )

    if source_coverage_start is not None:
        coverage_start = pd.to_datetime(source_coverage_start, utc=True, errors="coerce")
        if pd.isna(coverage_start):
            raise ValueError("source_coverage_start is not a valid timestamp.")
    else:
        coverage_start = None
        LOGGER.warning(
            "No source_coverage_start supplied. Zero text counts cannot distinguish "
            "a quiet period from incomplete archive coverage."
        )

    # All counter signals are additive. For window [t-w, t), subtract the
    # cumulative total strictly before t-w from the cumulative total strictly
    # before t. allow_exact_matches=False includes a report at t-w but excludes
    # a report at t, satisfying the strict point-in-time contract.
    output_parts: list[pd.DataFrame] = []
    for region_key, region_rows in base.groupby(
        base["region"].map(_region_key), sort=False
    ):
        actual_region = normalized_region_to_actual[region_key]
        scoped_events = event_frame.loc[
            event_frame["_scope_is_global"]
            | event_frame["_scope_region_key"].eq(region_key)
        ].copy()
        timeline, cumulative_columns = _build_region_timeline(
            scoped_events, embedding_columns=embedding_columns
        )
        region_rows = region_rows.sort_values(
            FORECAST_REFERENCE_COLUMN, kind="stable"
        ).reset_index(drop=True)
        reference_times = region_rows[FORECAST_REFERENCE_COLUMN]
        current = _lookup_cumulative(
            reference_times, timeline, cumulative_columns=cumulative_columns
        )
        last_event_ts = current["_matched_event_timestamp"]

        if last_event_ts.notna().any():
            matched = last_event_ts.notna()
            if not (
                last_event_ts.loc[matched].reset_index(drop=True)
                < reference_times.loc[matched].reset_index(drop=True)
            ).all():
                raise AssertionError("Point-in-time join included text at/after forecast time.")

        region_features = pd.DataFrame(
            {"_text_join_row_id": region_rows["_text_join_row_id"]}
        )
        # Retain the exact UTC timestamp internally for audit metadata; do not
        # reconstruct it from the numeric age feature (which is minute-based).
        region_features["_latest_text_timestamp_before_ref"] = (
            last_event_ts.reset_index(drop=True)
        )
        region_features["text_has_prior_report"] = last_event_ts.notna().astype("int8")
        age_minutes = (
            reference_times.reset_index(drop=True) - last_event_ts.reset_index(drop=True)
        ).dt.total_seconds() / 60.0
        region_features["text_minutes_since_last_report"] = (
            age_minutes.fillna(10080.0).clip(lower=0.0)
        )

        current_by_column = {
            column: current[column].to_numpy(dtype=np.float64)
            for column in cumulative_columns
        }
        for window in windows:
            start_times = reference_times - pd.to_timedelta(window, unit="h")
            prior_window = _lookup_cumulative(
                start_times, timeline, cumulative_columns=cumulative_columns
            )
            for source_column, cumulative_column in zip(
                self_additive_columns(embedding_columns), cumulative_columns
            ):
                delta = (
                    current_by_column[cumulative_column]
                    - prior_window[cumulative_column].to_numpy(dtype=np.float64)
                )
                # Floating-point cancellation can produce tiny negatives only.
                if (delta < -1e-8).any():
                    raise AssertionError("Cumulative text counters are not monotone.")
                delta = np.maximum(delta, 0.0)
                feature_name = _window_feature_name(source_column, window)
                region_features[feature_name] = delta

            report_count = region_features[f"text_report_count_{window}h"]
            intensity_sum = region_features[f"text_intensity_sum_{window}h"]
            high_count = region_features[f"text_high_intensity_count_{window}h"]
            region_features[f"text_mean_intensity_{window}h"] = np.divide(
                intensity_sum.to_numpy(dtype=float),
                report_count.to_numpy(dtype=float),
                out=np.zeros(len(region_features), dtype=float),
                where=report_count.to_numpy(dtype=float) > 0,
            )
            region_features[f"text_has_high_intensity_{window}h"] = (
                high_count.gt(0).astype("int8")
            )
            if coverage_start is not None:
                elapsed_hours = (
                    reference_times - coverage_start
                ).dt.total_seconds() / 3600.0
                region_features[f"text_coverage_fraction_{window}h"] = (
                    elapsed_hours.clip(lower=0.0, upper=float(window)) / float(window)
                ).to_numpy(dtype=float)

            if embedding_columns:
                # Use average embeddings in each trailing window. Zero vector
                # means no embedding-bearing reports in that window; count
                # features let the model distinguish this from a real zero mean.
                embed_count_name = f"text_embedding_report_count_{window}h"
                region_features[embed_count_name] = region_features[
                    f"text_embedding_count_{window}h"
                ]
                count_values = region_features[embed_count_name].to_numpy(dtype=float)
                for column in embedding_columns:
                    embed_sum_name = f"text_embedding_sum_{column}_{window}h"
                    # Keep sums out of the model inputs; use only means.
                    embedding_index = column.removeprefix("embedding_")
                    mean_name = f"text_embedding_mean_{window}h_{embedding_index}"
                    mean_values = np.divide(
                        region_features[embed_sum_name].to_numpy(dtype=float),
                        count_values,
                        out=np.zeros(len(region_features), dtype=float),
                        where=count_values > 0,
                    )
                    region_features[mean_name] = mean_values
                drop_sums = [
                    f"text_embedding_sum_{column}_{window}h"
                    for column in embedding_columns
                ]
                region_features.drop(columns=drop_sums, inplace=True)
                region_features.drop(columns=[f"text_embedding_count_{window}h"], inplace=True)

        output_parts.append(region_features)

    feature_frame = pd.concat(output_parts, ignore_index=True)
    result = base.merge(
        feature_frame,
        on="_text_join_row_id",
        how="left",
        validate="one_to_one",
        sort=False,
    ).drop(columns="_text_join_row_id")
    text_feature_columns = [
        column for column in result.columns if column.startswith("text_")
    ]
    numeric_values = result[text_feature_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_values).all():
        raise AssertionError("Generated text features contain NaN or non-finite values.")
    if result[text_feature_columns].isna().any().any():
        raise AssertionError("Generated text features contain missing values.")
    # Update audit metadata to reflect the latest input used by ANY feature,
    # including a report published after the previous completed hourly bucket.
    latest_text_timestamp = result["_latest_text_timestamp_before_ref"]
    has_prior = latest_text_timestamp.notna()
    if has_prior.any():
        if not (
            latest_text_timestamp.loc[has_prior]
            < result.loc[has_prior, FORECAST_REFERENCE_COLUMN]
        ).all():
            raise AssertionError(
                "A joined text report is timestamped at/after its forecast reference."
            )
        prior_cutoffs = result.loc[has_prior, FEATURE_CUTOFF_COLUMN]
        result.loc[has_prior, FEATURE_CUTOFF_COLUMN] = [
            max(old_cutoff, latest_report)
            for old_cutoff, latest_report in zip(
                prior_cutoffs, latest_text_timestamp.loc[has_prior]
            )
        ]

    if not (result[FEATURE_CUTOFF_COLUMN] < result[FORECAST_REFERENCE_COLUMN]).all():
        raise AssertionError("Text features violate the forecast reference cutoff.")
    if not (result[FEATURE_CUTOFF_COLUMN] < result[TIMESTAMP_COLUMN]).all():
        raise AssertionError("Feature cutoff metadata reaches the target bucket or later.")
    result.drop(columns=["_latest_text_timestamp_before_ref"], inplace=True)

    LOGGER.info(
        "Built %d point-in-time text features across %d rows and %d region(s).",
        len(text_feature_columns),
        len(result),
        result["region"].nunique(),
    )
    return result.sort_values(
        [TIMESTAMP_COLUMN, "region"], kind="stable"
    ).reset_index(drop=True)


def self_additive_columns(embedding_columns: Sequence[str]) -> list[str]:
    """List report-level additive measures in cumulative column construction order."""
    base_columns = [
        "_report_count",
        *[f"_signal_{name}" for name in KEYWORD_GROUPS],
        "_text_intensity",
        "_high_intensity",
    ]
    if embedding_columns:
        base_columns.extend(
            f"_embedding_sum_{column}" for column in embedding_columns
        )
        base_columns.append("_embedding_count")
    return base_columns


def _window_feature_name(source_column: str, window: int) -> str:
    """Map internal cumulative measures to stable human-readable feature names."""
    mapping = {
        "_report_count": "text_report_count",
        "_signal_missile": "text_missile_report_count",
        "_signal_drone": "text_drone_report_count",
        "_signal_air_activity": "text_air_activity_report_count",
        "_signal_attack": "text_attack_report_count",
        "_signal_official_alert": "text_official_alert_report_count",
        "_text_intensity": "text_intensity_sum",
        "_high_intensity": "text_high_intensity_count",
        "_embedding_count": "text_embedding_count",
    }
    if source_column.startswith("_embedding_sum_embedding_"):
        suffix = source_column.removeprefix("_embedding_sum_embedding_")
        return f"text_embedding_sum_embedding_{suffix}_{window}h"
    if source_column not in mapping:
        raise ValueError(f"Unknown additive text signal: {source_column}")
    return f"{mapping[source_column]}_{window}h"


def _metric_delta(
    y_true: pd.Series,
    probability: np.ndarray,
    duration_true: pd.Series,
    duration_prediction: np.ndarray,
) -> dict[str, object]:
    """Calculate out-of-time classification and regression metrics."""
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    classes = (probability >= 0.5).astype("int8")
    roc_auc = (
        float(roc_auc_score(y_true, probability))
        if y_true.nunique() == 2
        else None
    )
    return {
        "alert_occurrence": {
            "roc_auc": roc_auc,
            "precision_at_0_5": float(precision_score(y_true, classes, zero_division=0)),
            "recall_at_0_5": float(recall_score(y_true, classes, zero_division=0)),
            "threshold": 0.5,
        },
        "alert_minutes": {
            "mae_minutes": float(mean_absolute_error(duration_true, duration_prediction)),
            "rmse_minutes": float(np.sqrt(mean_squared_error(duration_true, duration_prediction))),
        },
    }


def compare_time_series_vs_text(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    n_splits: int = 5,
    cv_test_size_hours: int | None = None,
    cv_gap_hours: int = 0,
    random_state: int = 42,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    """Compare time-series-only and time-series-plus-text models on one holdout.

    Both feature sets use identical chronological train/test rows and identical
    expanding-window CV folds. Hyperparameters are tuned on training data only;
    test rows are used once, after refitting. Returns metrics, paired predictions,
    and fitted pipelines. Comparison deltas are enhanced minus baseline, so lower
    MAE/RMSE and higher ROC-AUC/precision/recall indicate an improvement.
    """
    try:
        from .model import (
            DURATION_TARGET,
            OCCURRENCE_TARGET,
            _make_pipeline,
            _search_pipeline,
            assert_temporal_integrity,
            build_expanding_window_splits,
        )
    except ImportError:  # pragma: no cover
        from model import (  # type: ignore[no-redef]
            DURATION_TARGET,
            OCCURRENCE_TARGET,
            _make_pipeline,
            _search_pipeline,
            assert_temporal_integrity,
            build_expanding_window_splits,
        )

    train_frame = train.copy().reset_index(drop=True)
    test_frame = test.copy().reset_index(drop=True)
    assert_temporal_integrity(train_frame, test_frame)
    all_features = get_model_feature_columns(train_frame)
    test_features = get_model_feature_columns(test_frame)
    if all_features != test_features:
        raise ValueError("Train/test feature schemas differ; split after text feature generation.")
    baseline_features = [name for name in all_features if not name.startswith("text_")]
    enhanced_features = list(all_features)
    if not baseline_features:
        raise ValueError("No time-series baseline features were found.")
    text_only = [name for name in enhanced_features if name.startswith("text_")]
    if not text_only:
        raise ValueError("No text features were found; build text features before comparison.")

    for split_name, frame in (("train", train_frame), ("test", test_frame)):
        if frame[enhanced_features].isna().any().any():
            raise ValueError(f"{split_name} feature matrix contains missing values.")
        numeric_columns = [col for col in enhanced_features if col != "region"]
        if not np.isfinite(frame[numeric_columns].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{split_name} features contain non-finite values.")

    cv_splits = build_expanding_window_splits(
        train_frame,
        n_splits=n_splits,
        test_size_hours=cv_test_size_hours,
        gap_hours=cv_gap_hours,
    )
    fitted: dict[str, object] = {}
    predictions = test_frame[
        [TIMESTAMP_COLUMN, "region", OCCURRENCE_TARGET, DURATION_TARGET]
    ].copy()
    metrics: dict[str, object] = {
        "evaluation": {
            "test_start": test_frame[TIMESTAMP_COLUMN].min().isoformat(),
            "test_end": test_frame[TIMESTAMP_COLUMN].max().isoformat(),
            "train_end": train_frame[TIMESTAMP_COLUMN].max().isoformat(),
            "n_train_rows": int(len(train_frame)),
            "n_test_rows": int(len(test_frame)),
            "n_text_features": int(len(text_only)),
            "cv_strategy": "expanding window over unique timestamps",
            "test_used_for_tuning": False,
        },
    }

    for model_name, feature_columns in (
        ("time_series_only", baseline_features),
        ("time_series_plus_text", enhanced_features),
    ):
        fitted[model_name] = {}
        X_train = train_frame[feature_columns]
        X_test = test_frame[feature_columns]
        y_class_train = train_frame[OCCURRENCE_TARGET].astype(int)
        y_class_test = test_frame[OCCURRENCE_TARGET].astype(int)
        y_duration_train = train_frame[DURATION_TARGET].astype(float)

        if y_class_train.nunique() < 2:
            raise ValueError("Training data contains only one occurrence class.")

        classifier = _search_pipeline(
            _make_pipeline(
                task="classification", feature_columns=feature_columns,
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
                task="regression", feature_columns=feature_columns,
                random_state=random_state,
            ),
            X_train=X_train,
            y_train=y_duration_train,
            cv_splits=cv_splits,
            scoring="neg_mean_absolute_error",
            task="regression",
        )
        probability = classifier.predict_proba(X_test)[:, 1]
        duration_prediction = np.maximum(regressor.predict(X_test), 0.0)
        fitted[model_name] = {"classifier": classifier, "regressor": regressor}

        metrics[model_name] = _metric_delta(
            y_class_test,
            probability,
            test_frame[DURATION_TARGET].astype(float),
            duration_prediction,
        )
        predictions[f"{model_name}_alert_probability"] = probability
        predictions[f"{model_name}_alert_minutes_prediction"] = duration_prediction

    def metric_value(model_name: str, task: str, key: str) -> float | None:
        value = metrics[model_name][task][key]  # type: ignore[index]
        return None if value is None else float(value)

    deltas: dict[str, object] = {}
    for task, keys in (
        ("alert_occurrence", ("roc_auc", "precision_at_0_5", "recall_at_0_5")),
        ("alert_minutes", ("mae_minutes", "rmse_minutes")),
    ):
        deltas[task] = {}
        for key in keys:
            base_value = metric_value("time_series_only", task, key)
            text_value = metric_value("time_series_plus_text", task, key)
            deltas[task][key] = (
                None if base_value is None or text_value is None
                else text_value - base_value
            )
    metrics["text_minus_baseline_delta"] = deltas
    LOGGER.info("Out-of-time baseline vs text metrics: %s", json.dumps(metrics, indent=2))
    return metrics, predictions, fitted


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Join point-in-time text signals and compare forecasting models."
    )
    parser.add_argument(
        "features_csv", type=Path,
        help="Full supervised.csv from src.features.",
    )
    parser.add_argument(
        "text_events_csv", type=Path, help="Timestamped public-text CSV."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/text_comparison")
    )
    parser.add_argument("--text-timestamp-col", default=TEXT_TIMESTAMP_COLUMN)
    parser.add_argument("--text-col", default=TEXT_BODY_COLUMN)
    parser.add_argument("--text-region-col", default="region")
    parser.add_argument("--reference-timestamp-col", default=None)
    parser.add_argument("--source-coverage-start", default=None)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-time", default=None)
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
    """Run feature joining, holdout comparison, and persist output artifacts."""
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    for path in (args.features_csv, args.text_events_csv):
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")

    time_series = pd.read_csv(args.features_csv)
    text_events = pd.read_csv(args.text_events_csv)
    enriched = build_point_in_time_text_features(
        time_series,
        text_events,
        text_timestamp_col=args.text_timestamp_col,
        text_col=args.text_col,
        text_region_col=args.text_region_col,
        reference_timestamp_col=args.reference_timestamp_col,
        source_coverage_start=args.source_coverage_start,
    )
    train, test = time_based_split(
        enriched, test_fraction=args.test_fraction, split_time=args.split_time
    )
    metrics, predictions, fitted = compare_time_series_vs_text(
        train,
        test,
        n_splits=args.n_splits,
        cv_test_size_hours=args.cv_test_size_hours,
        cv_gap_hours=args.cv_gap_hours,
        random_state=args.random_state,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output_dir / "supervised_with_text.csv", index=False)
    train.to_csv(args.output_dir / "train_with_text.csv", index=False)
    test.to_csv(args.output_dir / "test_with_text.csv", index=False)
    predictions.to_csv(args.output_dir / "test_comparison_predictions.csv", index=False)
    with (args.output_dir / "comparison_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    for model_name, model_parts in fitted.items():
        dump(model_parts["classifier"], args.output_dir / f"{model_name}_classifier.joblib")
        dump(model_parts["regressor"], args.output_dir / f"{model_name}_regressor.joblib")
    LOGGER.info("Saved point-in-time text comparison artifacts to %s", args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
