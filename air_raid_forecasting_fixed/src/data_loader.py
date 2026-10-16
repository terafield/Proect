"""Leakage-aware loading and hourly aggregation for air-raid alert time series.

The hourly columns ``alert_start_count`` and ``alert_minutes`` are outcome
labels for their respective buckets. They must not be used as same-bucket
model features. :func:`make_supervised_frame` creates only causal features:
lagged and rolling statistics use completed prior buckets, and calendar
features are known at the forecast timestamp.

Input timestamps are converted to UTC. Naive timestamps are interpreted in
``naive_timezone`` (UTC by default). Alert records must describe closed events
with valid start and end timestamps. If the source includes an ongoing alert,
exclude it from this historical-event file or handle it separately as a
right-censored observation.

Dependencies: pandas and numpy.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

HOUR = pd.Timedelta(hours=1)
NANOSECOND = pd.Timedelta(nanoseconds=1)
TIMESTAMP_COLUMN = "bucket_start"
FEATURE_CUTOFF_COLUMN = "feature_cutoff"
TARGET_COLUMNS = ("alert_start_count", "alert_minutes")
CALENDAR_FEATURES = (
    "local_hour",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
)

_DURATION_FACTORS = {
    "second": 1.0,
    "seconds": 1.0,
    "s": 1.0,
    "minute": 60.0,
    "minutes": 60.0,
    "min": 60.0,
    "m": 60.0,
    "hour": 3600.0,
    "hours": 3600.0,
    "h": 3600.0,
    "millisecond": 0.001,
    "milliseconds": 0.001,
    "ms": 0.001,
}


def _read_table(path: Path) -> pd.DataFrame:
    """Read a supported tabular file based on its extension."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    raise ValueError(
        f"Unsupported file extension {suffix!r}. Use CSV, Parquet, JSON, or JSONL."
    )


def _normalise_timestamp(value: object, naive_timezone: str) -> pd.Timestamp | pd.NaT:
    """Parse one timestamp and return it in UTC, or ``NaT`` if invalid."""
    if value is None or pd.isna(value):
        return pd.NaT
    try:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            return pd.NaT
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(
                naive_timezone,
                ambiguous="NaT",
                nonexistent="NaT",
            )
        if pd.isna(timestamp):
            return pd.NaT
        return timestamp.tz_convert("UTC")
    except (TypeError, ValueError, OverflowError):
        return pd.NaT


def _parse_timestamp_series(
    values: pd.Series,
    *,
    naive_timezone: str,
    column_name: str,
) -> pd.Series:
    """Parse timestamps consistently and raise if any are invalid."""
    parsed = pd.Series(
        [_normalise_timestamp(value, naive_timezone) for value in values],
        index=values.index,
        dtype="datetime64[ns, UTC]",
        name=column_name,
    )
    bad_mask = parsed.isna()
    if bad_mask.any():
        examples = values.loc[bad_mask].head(5).tolist()
        raise ValueError(
            f"Column {column_name!r} has {int(bad_mask.sum())} missing or invalid "
            f"timestamps. Example values: {examples!r}"
        )
    return parsed


def _parse_duration_value(value: object, duration_unit: str) -> float:
    """Parse a numeric or timedelta-like duration into seconds."""
    if value is None or pd.isna(value):
        return float("nan")

    unit_key = duration_unit.strip().lower()
    if unit_key not in _DURATION_FACTORS:
        choices = ", ".join(sorted(_DURATION_FACTORS))
        raise ValueError(f"Unsupported duration_unit {duration_unit!r}. Use one of: {choices}.")

    if isinstance(value, (pd.Timedelta, np.timedelta64)):
        return float(pd.to_timedelta(value).total_seconds())

    # Numeric values (including numeric strings) use the configured source unit.
    try:
        return float(value) * _DURATION_FACTORS[unit_key]
    except (TypeError, ValueError):
        pass

    try:
        return float(pd.to_timedelta(value).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return float("nan")


def load_alerts(
    path: str | Path,
    *,
    start_col: str = "start_time",
    end_col: str = "end_time",
    duration_col: str = "duration",
    region_col: str = "region",
    naive_timezone: str = "UTC",
    duration_unit: str = "seconds",
) -> pd.DataFrame:
    """Load and validate historical alert events.

    Args:
        path: CSV, Parquet, JSON, or JSONL source file.
        start_col: Source column containing alert start timestamps.
        end_col: Source column containing alert end timestamps.
        duration_col: Optional source duration column used for quality checks.
        region_col: Source column containing the Ukrainian region name.
        naive_timezone: Time zone assigned to timestamps lacking a UTC offset.
        duration_unit: Unit for numeric source durations; textual timedeltas such
            as ``"00:15:00"`` are parsed as timedeltas.

    Returns:
        A sorted dataframe with canonical ``start_time``, ``end_time``,
        ``region``, and ``duration_seconds`` columns. If available, source
        durations are preserved as ``source_duration_seconds`` for QA only.
        ``duration_seconds`` is always derived from ``end_time - start_time``.

    Raises:
        FileNotFoundError: If the source does not exist.
        ValueError: If required columns, timestamps, regions, or intervals are
            invalid. Invalid rows are not silently discarded.
    """
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Alert data file not found: {source_path}")

    frame = _read_table(source_path)
    required = {start_col, end_col, region_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")
    # An empty but correctly headed event table can represent a genuinely quiet
    # observation interval, provided the caller later supplies explicit regions
    # and coverage bounds to ``aggregate_hourly``.
    result = pd.DataFrame(index=frame.index)
    result["start_time"] = _parse_timestamp_series(
        frame[start_col], naive_timezone=naive_timezone, column_name=start_col
    )
    result["end_time"] = _parse_timestamp_series(
        frame[end_col], naive_timezone=naive_timezone, column_name=end_col
    )
    result["region"] = frame[region_col].astype("string").str.strip()

    invalid_region = result["region"].isna() | result["region"].eq("")
    if invalid_region.any():
        raise ValueError(f"Found {int(invalid_region.sum())} rows with missing/blank region names.")

    negative_duration = result["end_time"] < result["start_time"]
    if negative_duration.any():
        bad_rows = result.index[negative_duration].tolist()[:5]
        raise ValueError(
            f"Found {int(negative_duration.sum())} events ending before they start. "
            f"Example row indices: {bad_rows}"
        )

    result["duration_seconds"] = (
        result["end_time"] - result["start_time"]
    ).dt.total_seconds()

    if duration_col in frame.columns:
        source_duration = pd.Series(
            [_parse_duration_value(value, duration_unit) for value in frame[duration_col]],
            index=frame.index,
            dtype="float64",
            name="source_duration_seconds",
        )
        result["source_duration_seconds"] = source_duration
        comparable = source_duration.notna()
        mismatched = comparable & ~np.isclose(
            source_duration.fillna(0.0),
            result["duration_seconds"],
            rtol=0.0,
            atol=1.0,
        )
        if mismatched.any():
            LOGGER.warning(
                "%d source duration value(s) differ from end_time - start_time by "
                "more than 1 second; duration_seconds is derived from timestamps.",
                int(mismatched.sum()),
            )
        if (~comparable).any():
            LOGGER.warning(
                "%d source duration value(s) could not be parsed; using timestamp-derived durations.",
                int((~comparable).sum()),
            )
    else:
        LOGGER.info("No source duration column %r; deriving durations from timestamps.", duration_col)

    result = result.reset_index(drop=True).sort_values(
        ["start_time", "region", "end_time"], kind="stable"
    )
    LOGGER.info(
        "Loaded %d valid alert events across %d regions (%s to %s UTC).",
        len(result),
        result["region"].nunique(),
        result["start_time"].min(),
        result["end_time"].max(),
    )
    return result.reset_index(drop=True)


def _hour_boundary(
    value: str | pd.Timestamp,
    *,
    name: str,
    naive_timezone: str,
) -> pd.Timestamp:
    """Parse an hourly bucket boundary and require hour alignment."""
    timestamp = _normalise_timestamp(value, naive_timezone)
    if pd.isna(timestamp):
        raise ValueError(f"{name} is not a valid timestamp: {value!r}")
    if timestamp != timestamp.floor("h"):
        raise ValueError(f"{name} must be aligned to a whole UTC hour; got {timestamp}.")
    return timestamp


def aggregate_hourly(
    alerts: pd.DataFrame,
    *,
    coverage_start: str | pd.Timestamp | None = None,
    coverage_end: str | pd.Timestamp | None = None,
    regions: Sequence[str] | None = None,
    naive_timezone: str = "UTC",
) -> pd.DataFrame:
    """Aggregate events into a complete hourly panel for each region.

    ``coverage_start`` and ``coverage_end`` are inclusive UTC bucket-start
    timestamps. Pass them when the source's observation window is known, so
    trailing hours with no alerts are represented as zeros. If omitted, bounds
    are inferred from the event records, which cannot recover silent periods
    outside the event-derived bounds. ``regions`` supplies a canonical roster,
    including regions with no alert events during the window. For an empty event
    table, coverage bounds and ``regions`` are all mandatory.

    The output labels are:
      * ``alert_start_count``: number of alerts starting in the bucket.
      * ``alert_minutes``: sum of each alert's overlap with the bucket, in
        minutes. It can exceed 60 if multiple alert intervals overlap.

    Each event is treated as a half-open interval ``[start_time, end_time)``.
    These same-bucket values are labels, not permissible same-bucket features.
    """
    required = {"start_time", "end_time", "region"}
    missing = sorted(required - set(alerts.columns))
    if missing:
        raise ValueError(f"alerts dataframe is missing canonical columns: {missing}")
    work = alerts.loc[:, ["start_time", "end_time", "region"]].copy()
    if work.empty and (regions is None or coverage_start is None or coverage_end is None):
        raise ValueError(
            "An empty alert table requires explicit coverage_start, coverage_end, "
            "and a canonical regions list; otherwise a zero-alert panel cannot "
            "be distinguished from missing data."
        )
    work["start_time"] = pd.to_datetime(work["start_time"], utc=True, errors="coerce")
    work["end_time"] = pd.to_datetime(work["end_time"], utc=True, errors="coerce")
    if work[["start_time", "end_time", "region"]].isna().any().any():
        raise ValueError("alerts contains missing/invalid start_time, end_time, or region values.")
    if (work["end_time"] < work["start_time"]).any():
        raise ValueError("alerts contains an event whose end_time precedes start_time.")

    if work.empty:
        inferred_start = pd.NaT
        inferred_end = pd.NaT
    else:
        inferred_start = work["start_time"].min().floor("h")
        last_start_bucket = work["start_time"].max().floor("h")
        positive_length = work["end_time"] > work["start_time"]
        if positive_length.any():
            last_overlap_bucket = (
                work.loc[positive_length, "end_time"].max() - NANOSECOND
            ).floor("h")
            inferred_end = max(last_start_bucket, last_overlap_bucket)
        else:
            inferred_end = last_start_bucket

    first_bucket = (
        _hour_boundary(coverage_start, name="coverage_start", naive_timezone=naive_timezone)
        if coverage_start is not None
        else inferred_start
    )
    last_bucket = (
        _hour_boundary(coverage_end, name="coverage_end", naive_timezone=naive_timezone)
        if coverage_end is not None
        else inferred_end
    )
    if first_bucket > last_bucket:
        raise ValueError("coverage_start must be less than or equal to coverage_end.")

    buckets = pd.date_range(first_bucket, last_bucket, freq="h", tz="UTC")
    observed_regions = set(work["region"].astype(str).unique().tolist())
    if regions is None:
        region_names = sorted(observed_regions)
    else:
        region_names = [str(value).strip() for value in regions]
        if any(not value for value in region_names):
            raise ValueError("The regions roster contains a blank label.")
        if len(set(region_names)) != len(region_names):
            raise ValueError("The regions roster contains duplicate labels.")
        unknown_events = sorted(observed_regions - set(region_names))
        if unknown_events:
            raise ValueError(
                "The regions roster omits region labels present in the alert events: "
                f"{unknown_events[:10]}"
            )
        region_names = sorted(region_names)
    if not region_names:
        raise ValueError("No regions are available to construct the hourly panel.")
    panel_index = pd.MultiIndex.from_product(
        [region_names, buckets], names=["region", TIMESTAMP_COLUMN]
    )
    panel = panel_index.to_frame(index=False)

    start_events = work.assign(bucket_start=work["start_time"].dt.floor("h"))
    start_counts = start_events.groupby(["region", "bucket_start"], sort=False).size()
    panel["alert_start_count"] = (
        start_counts.reindex(panel_index, fill_value=0).to_numpy(dtype=np.int64)
    )

    # Compute interval overlap against each fixed bucket and clip to requested
    # coverage. This calculates labels only; they become features only after a
    # one-or-more-hour shift in make_supervised_frame().
    overlap_seconds: dict[tuple[str, pd.Timestamp], float] = {}
    for event in work.itertuples(index=False):
        start = event.start_time
        end = event.end_time
        if end <= start:
            continue

        bucket = max(start.floor("h"), first_bucket)
        final_overlap_bucket = min((end - NANOSECOND).floor("h"), last_bucket)
        while bucket <= final_overlap_bucket:
            overlap_start = max(start, bucket)
            overlap_end = min(end, bucket + HOUR)
            if overlap_start < overlap_end:
                key = (str(event.region), bucket)
                overlap_seconds[key] = overlap_seconds.get(key, 0.0) + (
                    overlap_end - overlap_start
                ).total_seconds()
            bucket += HOUR

    overlap_series = pd.Series(overlap_seconds, dtype="float64")
    panel["alert_minutes"] = (
        overlap_series.reindex(panel_index, fill_value=0.0).to_numpy(dtype=np.float64) / 60.0
    )

    panel[TIMESTAMP_COLUMN] = pd.to_datetime(panel[TIMESTAMP_COLUMN], utc=True)
    panel = panel.sort_values(["region", TIMESTAMP_COLUMN], kind="stable").reset_index(drop=True)

    if (panel[list(TARGET_COLUMNS)] < 0).any().any():
        raise AssertionError("Hourly alert labels must be non-negative.")
    if panel.duplicated(["region", TIMESTAMP_COLUMN]).any():
        raise AssertionError("Hourly panel contains duplicate region/bucket rows.")

    LOGGER.info(
        "Created %d hourly region-bucket rows for %d regions (%s to %s UTC).",
        len(panel),
        len(region_names),
        first_bucket,
        last_bucket,
    )
    return panel


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return permissible model inputs, excluding targets and timestamp metadata."""
    candidates = set(CALENDAR_FEATURES) | {"region"}
    candidates.update(
        column
        for column in frame.columns
        if column.startswith("alert_start_count_lag_")
        or column.startswith("alert_minutes_lag_")
        or column.startswith("alert_start_count_rollmean_")
        or column.startswith("alert_minutes_rollmean_")
    )
    return [column for column in frame.columns if column in candidates]


def make_supervised_frame(
    hourly: pd.DataFrame,
    *,
    lags: Sequence[int] = (1, 2, 3, 6, 12, 24, 168),
    rolling_windows: Sequence[int] = (3, 6, 24, 168),
    calendar_timezone: str = "Europe/Kyiv",
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """Create a supervised frame using only information available at forecast time.

    For the row with ``bucket_start == t``, every lag and rolling statistic is
    computed from bucket labels strictly earlier than ``t``. Rolling windows
    shift the series by one bucket before calculating the statistic, so the
    current bucket's outcomes cannot enter its own features. No interpolation,
    forward/backfill, or global imputation is performed.

    The ``feature_cutoff`` metadata records the latest completed bucket that
    could be used as input (``bucket_start - 1 hour``). Use
    :func:`get_feature_columns` to select inputs and avoid passing target or
    timestamp metadata columns to a model.
    """
    required = {TIMESTAMP_COLUMN, "region", *TARGET_COLUMNS}
    missing = sorted(required - set(hourly.columns))
    if missing:
        raise ValueError(f"hourly dataframe is missing required columns: {missing}")
    if hourly.empty:
        raise ValueError("Cannot build features from an empty hourly dataframe.")
    if not lags or not rolling_windows:
        raise ValueError("lags and rolling_windows must each contain at least one positive integer.")
    if any(not isinstance(lag, int) or lag <= 0 for lag in lags):
        raise ValueError("All lags must be positive integers.")
    if any(not isinstance(window, int) or window <= 0 for window in rolling_windows):
        raise ValueError("All rolling_windows must be positive integers.")

    frame = hourly.copy()
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=True, errors="coerce")
    if frame[TIMESTAMP_COLUMN].isna().any() or frame["region"].isna().any():
        raise ValueError("Hourly data contains invalid timestamps or missing region names.")
    if frame.duplicated(["region", TIMESTAMP_COLUMN]).any():
        raise ValueError("Hourly data has duplicate region/bucket rows.")
    if (frame[list(TARGET_COLUMNS)] < 0).any().any():
        raise ValueError("Hourly target columns must be non-negative.")

    frame = frame.sort_values(["region", TIMESTAMP_COLUMN], kind="stable").reset_index(drop=True)
    for region, group in frame.groupby("region", sort=False):
        deltas = group[TIMESTAMP_COLUMN].diff().dropna()
        if not deltas.eq(HOUR).all():
            raise ValueError(
                f"Region {region!r} does not have a complete hourly sequence. "
                "Aggregate/reindex missing hours before creating lag features."
            )

    local_time = frame[TIMESTAMP_COLUMN].dt.tz_convert(calendar_timezone)
    frame["local_hour"] = local_time.dt.hour.astype("int8")
    frame["day_of_week"] = local_time.dt.dayofweek.astype("int8")
    frame["is_weekend"] = local_time.dt.dayofweek.isin([5, 6]).astype("int8")
    frame["hour_sin"] = np.sin(2.0 * np.pi * frame["local_hour"] / 24.0)
    frame["hour_cos"] = np.cos(2.0 * np.pi * frame["local_hour"] / 24.0)
    frame["dow_sin"] = np.sin(2.0 * np.pi * frame["day_of_week"] / 7.0)
    frame["dow_cos"] = np.cos(2.0 * np.pi * frame["day_of_week"] / 7.0)

    for target in TARGET_COLUMNS:
        grouped = frame.groupby("region", sort=False)[target]
        for lag in sorted(set(lags)):
            frame[f"{target}_lag_{lag}"] = grouped.shift(lag)
        for window in sorted(set(rolling_windows)):
            frame[f"{target}_rollmean_{window}"] = grouped.transform(
                lambda values, window=window: values.shift(1).rolling(
                    window=window, min_periods=window
                ).mean()
            )

    frame[FEATURE_CUTOFF_COLUMN] = frame[TIMESTAMP_COLUMN] - HOUR
    if not (frame[FEATURE_CUTOFF_COLUMN] < frame[TIMESTAMP_COLUMN]).all():
        raise AssertionError("Leakage check failed: a feature cutoff is not before its target bucket.")

    feature_columns = get_feature_columns(frame)
    if drop_incomplete:
        before = len(frame)
        frame = frame.dropna(subset=feature_columns).copy()
        LOGGER.info(
            "Dropped %d warm-up row(s) without sufficient causal lag/rolling history.",
            before - len(frame),
        )
    if frame.empty:
        raise ValueError(
            "No rows remain after feature construction. Provide more history or use shorter lags/windows."
        )

    # Check the actual generated inputs are numeric where expected and never
    # contain infinities. Region is categorical and excluded from this check.
    numeric_features = [column for column in feature_columns if column != "region"]
    if frame[numeric_features].select_dtypes(include=[np.number]).shape[1] != len(numeric_features):
        raise AssertionError("A generated numeric feature has an unexpected non-numeric dtype.")
    numeric_values = frame[numeric_features].to_numpy(dtype=np.float64)
    if np.isinf(numeric_values).any():
        raise AssertionError("Generated features contain infinite values.")
    if drop_incomplete and np.isnan(numeric_values).any():
        raise AssertionError(
            "Model-ready features contain missing values after warm-up rows were dropped."
        )
    if not drop_incomplete and np.isnan(numeric_values).any():
        LOGGER.warning(
            "Warm-up rows are retained with missing lag values; this frame is not "
            "accepted by the current model trainer without additional preprocessing."
        )

    LOGGER.info("Created supervised frame with %d rows and %d model feature columns.", len(frame), len(feature_columns))
    return frame.reset_index(drop=True)


def time_based_split(
    supervised: pd.DataFrame,
    *,
    test_fraction: float = 0.2,
    split_time: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split panel data chronologically, holding out the most recent period.

    The default test size is a fraction of unique hourly timestamps, not rows,
    so every region shares the same split boundary. Alternatively, ``split_time``
    can be supplied as an explicit UTC timestamp; training rows are strictly
    earlier than it and test rows are at or after it.

    Returns:
        ``(train, test)`` dataframes with original columns and index reset.

    Raises:
        ValueError: If the split is invalid or leaves an empty partition.
        AssertionError: If chronological or feature-cutoff invariants fail.
    """
    required = {TIMESTAMP_COLUMN, FEATURE_CUTOFF_COLUMN}
    missing = sorted(required - set(supervised.columns))
    if missing:
        raise ValueError(
            f"supervised dataframe is missing {missing}; call make_supervised_frame first."
        )
    if supervised.empty:
        raise ValueError("Cannot split an empty supervised dataframe.")

    frame = supervised.copy()
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=True, errors="coerce")
    frame[FEATURE_CUTOFF_COLUMN] = pd.to_datetime(
        frame[FEATURE_CUTOFF_COLUMN], utc=True, errors="coerce"
    )
    if frame[[TIMESTAMP_COLUMN, FEATURE_CUTOFF_COLUMN]].isna().any().any():
        raise ValueError("Invalid timestamp or feature_cutoff in supervised dataframe.")
    if not frame[TIMESTAMP_COLUMN].eq(frame[TIMESTAMP_COLUMN].dt.floor("h")).all():
        raise ValueError("Forecast bucket timestamps must align to whole UTC hours.")

    bad_cutoff = frame[FEATURE_CUTOFF_COLUMN] >= frame[TIMESTAMP_COLUMN]
    if bad_cutoff.any():
        raise AssertionError(
            f"Leakage check failed: {int(bad_cutoff.sum())} row(s) have features dated "
            "at or after their forecast bucket."
        )

    unique_times = pd.Index(frame[TIMESTAMP_COLUMN].drop_duplicates()).sort_values()
    if len(unique_times) < 2:
        raise ValueError("At least two unique bucket timestamps are required for a time split.")
    if not pd.Series(unique_times).diff().dropna().eq(HOUR).all():
        raise ValueError(
            "Supervised rows must cover a continuous hourly timestamp grid; missing "
            "whole hours make lag labels misaligned. Rebuild a complete hourly panel."
        )

    if split_time is None:
        if not 0.0 < test_fraction < 1.0:
            raise ValueError("test_fraction must be strictly between 0 and 1.")
        test_period_count = max(1, math.ceil(len(unique_times) * test_fraction))
        if test_period_count >= len(unique_times):
            raise ValueError("test_fraction leaves no training timestamps; reduce it.")
        boundary = unique_times[-test_period_count]
    else:
        boundary = _normalise_timestamp(split_time, "UTC")
        if pd.isna(boundary):
            raise ValueError(f"Invalid split_time: {split_time!r}")

    train = frame.loc[frame[TIMESTAMP_COLUMN] < boundary].copy()
    test = frame.loc[frame[TIMESTAMP_COLUMN] >= boundary].copy()
    if train.empty or test.empty:
        raise ValueError(
            f"Split at {boundary} created an empty partition (train={len(train)}, test={len(test)})."
        )

    # Runtime invariants (not Python's optimisable ``assert`` statement) ensure
    # leakage checks remain active even when Python is run with ``-O``.
    if not (train[TIMESTAMP_COLUMN] < boundary).all():
        raise AssertionError("Leakage check failed: training targets reach the test period.")
    if not (test[TIMESTAMP_COLUMN] >= boundary).all():
        raise AssertionError("Split check failed: test set contains a pre-boundary timestamp.")
    if train[TIMESTAMP_COLUMN].max() >= test[TIMESTAMP_COLUMN].min():
        raise AssertionError("Leakage check failed: training timestamps overlap or exceed test timestamps.")
    if not (train[FEATURE_CUTOFF_COLUMN] < boundary).all():
        raise AssertionError("Leakage check failed: a training feature cutoff reaches the test period.")
    if not (test[FEATURE_CUTOFF_COLUMN] < test[TIMESTAMP_COLUMN]).all():
        raise AssertionError("Leakage check failed: a test row contains future/current-bucket features.")
    if len(train) + len(test) != len(frame):
        raise AssertionError("Split check failed: rows were lost or duplicated.")

    train = train.sort_values([TIMESTAMP_COLUMN, "region"], kind="stable").reset_index(drop=True)
    test = test.sort_values([TIMESTAMP_COLUMN, "region"], kind="stable").reset_index(drop=True)
    LOGGER.info(
        "Time split at %s UTC: train=%d rows through %s; test=%d rows from %s.",
        boundary,
        len(train),
        train[TIMESTAMP_COLUMN].max(),
        len(test),
        test[TIMESTAMP_COLUMN].min(),
    )
    return train, test


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Create leakage-aware hourly air-raid alert datasets."
    )
    parser.add_argument("input", type=Path, help="Input CSV, Parquet, JSON, or JSONL file.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-time", default=None, help="Optional explicit test start timestamp.")
    parser.add_argument("--coverage-start", default=None, help="Inclusive first hourly bucket (UTC hour).")
    parser.add_argument("--coverage-end", default=None, help="Inclusive last hourly bucket (UTC hour).")
    parser.add_argument(
        "--regions", nargs="+", default=None,
        help=(
            "Canonical region roster; include regions that had zero alerts throughout "
            "the observation window. Quote labels containing spaces. Required with "
            "coverage bounds when the event table is empty."
        ),
    )
    parser.add_argument("--naive-timezone", default="UTC", help="Timezone for source timestamps without offsets.")
    parser.add_argument("--duration-unit", default="seconds", help="Unit of numeric source duration values.")
    parser.add_argument("--calendar-timezone", default="Europe/Kyiv", help="Timezone for calendar features.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Run the end-to-end preprocessing pipeline and save CSV outputs."""
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    alerts = load_alerts(
        args.input,
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
    supervised = make_supervised_frame(
        hourly,
        calendar_timezone=args.calendar_timezone,
    )
    train, test = time_based_split(
        supervised,
        test_fraction=args.test_fraction,
        split_time=args.split_time,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(args.output_dir / "hourly.csv", index=False)
    supervised.to_csv(args.output_dir / "supervised.csv", index=False)
    train.to_csv(args.output_dir / "train.csv", index=False)
    test.to_csv(args.output_dir / "test.csv", index=False)
    LOGGER.info("Saved processed datasets to %s", args.output_dir.resolve())
    LOGGER.info("Permissible model inputs: %s", get_feature_columns(supervised))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
