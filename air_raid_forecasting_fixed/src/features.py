"""Causal feature engineering for hourly air-raid alert forecasting.

The target row is indexed by the start of the hour being predicted. Every
historical alert feature is shifted by at least one complete hourly bucket.
Calendar features are deterministic and known at forecast time. Neighbor
features use only other regions' *preceding* hourly buckets.

Typical use::

    python -m src.features data/processed/hourly.csv \
        --output-dir data/processed/model_data --test-fraction 0.2

The default oblast adjacency graph is a starting point, not a substitute for
checking the labels/topology against the source data. Pass ``--neighbors-json``
to use a project-validated graph. JSON format is
``{"Region A": ["Region B", ...]}``.

Dependencies: pandas and numpy.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # Support both ``python -m src.features`` and direct script execution.
    from .data_loader import (
        FEATURE_CUTOFF_COLUMN,
        TARGET_COLUMNS,
        TIMESTAMP_COLUMN,
        time_based_split,
    )
except ImportError:  # pragma: no cover - exercised only as a direct script.
    from data_loader import (  # type: ignore[no-redef]
        FEATURE_CUTOFF_COLUMN,
        TARGET_COLUMNS,
        TIMESTAMP_COLUMN,
        time_based_split,
    )

LOGGER = logging.getLogger(__name__)
HOUR = pd.Timedelta(hours=1)

DEFAULT_LAGS_HOURS: tuple[int, ...] = (1, 3, 6, 24, 168)
DEFAULT_ROLLING_WINDOWS_HOURS: tuple[int, ...] = (3, 6, 24, 168)
DEFAULT_NEIGHBOR_LAGS_HOURS: tuple[int, ...] = (1, 3, 6, 24)
DEFAULT_NEIGHBOR_ROLLING_HOURS: tuple[int, ...] = (3, 6, 24)

# Keys use canonical, normalized region identifiers. The mapping is
# configurable because source datasets commonly use different regional labels.
_BASE_NEIGHBORS: dict[str, tuple[str, ...]] = {
    "vinnytsia": (
        "khmelnytskyi",
        "zhytomyr",
        "kyiv_oblast",
        "cherkasy",
        "kirovohrad",
        "odesa",
        "chernivtsi",
    ),
    "volyn": ("rivne", "lviv"),
    "dnipropetrovsk": (
        "poltava",
        "kharkiv",
        "donetsk",
        "zaporizhzhia",
        "kherson",
        "mykolaiv",
        "kirovohrad",
    ),
    "donetsk": (
        "kharkiv",
        "dnipropetrovsk",
        "zaporizhzhia",
        "luhansk",
    ),
    "zhytomyr": ("rivne", "khmelnytskyi", "vinnytsia", "kyiv_oblast"),
    "zakarpattia": ("lviv", "ivano_frankivsk"),
    "zaporizhzhia": ("dnipropetrovsk", "donetsk", "kherson"),
    "ivano_frankivsk": ("lviv", "zakarpattia", "ternopil", "chernivtsi"),
    "kyiv_oblast": (
        "zhytomyr",
        "vinnytsia",
        "cherkasy",
        "poltava",
        "chernihiv",
        "kyiv_city",
    ),
    "kyiv_city": ("kyiv_oblast",),
    "kirovohrad": (
        "vinnytsia",
        "cherkasy",
        "poltava",
        "dnipropetrovsk",
        "mykolaiv",
        "odesa",
    ),
    "luhansk": ("kharkiv", "donetsk"),
    "lviv": (
        "volyn",
        "rivne",
        "ternopil",
        "ivano_frankivsk",
        "zakarpattia",
    ),
    "mykolaiv": ("odesa", "kirovohrad", "dnipropetrovsk", "kherson"),
    "odesa": ("vinnytsia", "kirovohrad", "mykolaiv"),
    "poltava": (
        "kyiv_oblast",
        "chernihiv",
        "sumy",
        "kharkiv",
        "dnipropetrovsk",
        "kirovohrad",
        "cherkasy",
    ),
    "rivne": (
        "volyn",
        "lviv",
        "ternopil",
        "khmelnytskyi",
        "zhytomyr",
    ),
    "sumy": ("chernihiv", "poltava", "kharkiv"),
    "ternopil": (
        "lviv",
        "rivne",
        "khmelnytskyi",
        "chernivtsi",
        "ivano_frankivsk",
    ),
    "kharkiv": (
        "sumy",
        "poltava",
        "dnipropetrovsk",
        "donetsk",
        "luhansk",
    ),
    "kherson": (
        "dnipropetrovsk",
        "zaporizhzhia",
        "mykolaiv",
        "crimea",
    ),
    "khmelnytskyi": (
        "rivne",
        "ternopil",
        "chernivtsi",
        "vinnytsia",
        "zhytomyr",
    ),
    "cherkasy": ("kyiv_oblast", "poltava", "kirovohrad", "vinnytsia"),
    "chernivtsi": (
        "ivano_frankivsk",
        "ternopil",
        "khmelnytskyi",
        "vinnytsia",
    ),
    "chernihiv": ("kyiv_oblast", "sumy", "poltava"),
    "crimea": ("kherson",),
}

# Frequently seen English and Ukrainian labels. Unknown labels are normalized
# conservatively and can still be used when a custom adjacency map is provided.
_REGION_ALIASES: dict[str, str] = {
    "vinnytsia oblast": "vinnytsia",
    "вінницька область": "vinnytsia",
    "вінницька": "vinnytsia",
    "volyn oblast": "volyn",
    "volyn oblast (lutsk)": "volyn",
    "волинська область": "volyn",
    "волинська": "volyn",
    "dnipropetrovsk oblast": "dnipropetrovsk",
    "дніпропетровська область": "dnipropetrovsk",
    "дніпропетровська": "dnipropetrovsk",
    "donetsk oblast": "donetsk",
    "донецька область": "donetsk",
    "донецька": "donetsk",
    "zhytomyr oblast": "zhytomyr",
    "житомирська область": "zhytomyr",
    "житомирська": "zhytomyr",
    "zakarpattia oblast": "zakarpattia",
    "transcarpathia": "zakarpattia",
    "закарпатська область": "zakarpattia",
    "закарпатська": "zakarpattia",
    "zaporizhzhia oblast": "zaporizhzhia",
    "zaporizhia oblast": "zaporizhzhia",
    "запорізька область": "zaporizhzhia",
    "запорізька": "zaporizhzhia",
    "ivano-frankivsk oblast": "ivano_frankivsk",
    "ivano frankivsk oblast": "ivano_frankivsk",
    "івано-франківська область": "ivano_frankivsk",
    "івано-франківська": "ivano_frankivsk",
    "kyiv oblast": "kyiv_oblast",
    "київська область": "kyiv_oblast",
    "київська обл.": "kyiv_oblast",
    "kyiv": "kyiv_city",
    "kyiv city": "kyiv_city",
    "city of kyiv": "kyiv_city",
    "м. київ": "kyiv_city",
    "місто київ": "kyiv_city",
    "київ (місто)": "kyiv_city",
    "kirovohrad oblast": "kirovohrad",
    "kropyvnytskyi oblast": "kirovohrad",
    "кіровоградська область": "kirovohrad",
    "кіровоградська": "kirovohrad",
    "кропивницький": "kirovohrad",
    "luhansk oblast": "luhansk",
    "луганська область": "luhansk",
    "луганська": "luhansk",
    "lviv oblast": "lviv",
    "львівська область": "lviv",
    "львівська": "lviv",
    "mykolaiv oblast": "mykolaiv",
    "миколаївська область": "mykolaiv",
    "миколаївська": "mykolaiv",
    "odesa oblast": "odesa",
    "odessa oblast": "odesa",
    "одеська область": "odesa",
    "одеська": "odesa",
    "poltava oblast": "poltava",
    "полтавська область": "poltava",
    "полтавська": "poltava",
    "rivne oblast": "rivne",
    "рівненська область": "rivne",
    "рівненська": "rivne",
    "sumy oblast": "sumy",
    "сумська область": "sumy",
    "сумська": "sumy",
    "ternopil oblast": "ternopil",
    "тернопільська область": "ternopil",
    "тернопільська": "ternopil",
    "kharkiv oblast": "kharkiv",
    "харківська область": "kharkiv",
    "харківська": "kharkiv",
    "kherson oblast": "kherson",
    "херсонська область": "kherson",
    "херсонська": "kherson",
    "khmelnytskyi oblast": "khmelnytskyi",
    "khmelnitsky oblast": "khmelnytskyi",
    "хмельницька область": "khmelnytskyi",
    "хмельницька": "khmelnytskyi",
    "cherkasy oblast": "cherkasy",
    "черкаська область": "cherkasy",
    "черкаська": "cherkasy",
    "chernivtsi oblast": "chernivtsi",
    "чернівецька область": "chernivtsi",
    "чернівецька": "chernivtsi",
    "chernihiv oblast": "chernihiv",
    "чернігівська область": "chernihiv",
    "чернігівська": "chernihiv",
    "crimea": "crimea",
    "autonomous republic of crimea": "crimea",
    "ар крим": "crimea",
    "автономна республіка крим": "crimea",
    "крим": "crimea",
}


def normalize_region_name(value: object) -> str:
    """Normalize a region label to a stable key for adjacency lookup."""
    label = re.sub(r"\s+", " ", str(value).strip().casefold())
    if label in _REGION_ALIASES:
        return _REGION_ALIASES[label]
    label = label.replace("oblast", "").replace("область", "")
    label = re.sub(r"[^\w]+", "_", label, flags=re.UNICODE).strip("_")
    return label


def get_default_neighbors() -> dict[str, tuple[str, ...]]:
    """Return a normalized, symmetric copy of the default oblast graph."""
    graph: dict[str, set[str]] = {
        region: set(values) for region, values in _BASE_NEIGHBORS.items()
    }
    for region, adjacent_regions in list(graph.items()):
        for neighbor in adjacent_regions:
            graph.setdefault(neighbor, set()).add(region)
    return {region: tuple(sorted(values)) for region, values in graph.items()}


def _normalize_neighbor_mapping(
    neighbors: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    """Normalize keys and values in a caller-supplied adjacency mapping."""
    normalized: dict[str, set[str]] = {}
    for region, adjacent_regions in neighbors.items():
        region_key = normalize_region_name(region)
        normalized.setdefault(region_key, set())
        for adjacent in adjacent_regions:
            adjacent_key = normalize_region_name(adjacent)
            if adjacent_key == region_key:
                raise ValueError(f"Region {region!r} cannot be its own neighbor.")
            normalized[region_key].add(adjacent_key)
            normalized.setdefault(adjacent_key, set()).add(region_key)
    return {region: tuple(sorted(values)) for region, values in normalized.items()}


def _validate_hourly_panel(hourly: pd.DataFrame) -> pd.DataFrame:
    """Validate and return an hourly region panel in deterministic order."""
    required = {TIMESTAMP_COLUMN, "region", *TARGET_COLUMNS}
    missing = sorted(required - set(hourly.columns))
    if missing:
        raise ValueError(f"Hourly dataframe is missing required columns: {missing}")
    if hourly.empty:
        raise ValueError("Hourly dataframe is empty.")

    frame = hourly.copy()
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(
        frame[TIMESTAMP_COLUMN], utc=True, errors="coerce"
    )
    if frame[TIMESTAMP_COLUMN].isna().any() or frame["region"].isna().any():
        raise ValueError("Hourly data contains missing/invalid timestamps or regions.")
    if not frame[TIMESTAMP_COLUMN].eq(frame[TIMESTAMP_COLUMN].dt.floor("h")).all():
        raise ValueError("bucket_start timestamps must align to whole UTC hours.")
    frame["region"] = frame["region"].astype(str).str.strip()
    if frame["region"].eq("").any():
        raise ValueError("Hourly data contains blank region labels.")
    if frame.duplicated(["region", TIMESTAMP_COLUMN]).any():
        raise ValueError("Hourly data has duplicate region/bucket rows.")
    if frame[list(TARGET_COLUMNS)].isna().any().any():
        raise ValueError(
            "Hourly target columns contain missing values; "
            "do not impute labels silently."
        )
    if (frame[list(TARGET_COLUMNS)] < 0).any().any():
        raise ValueError("Hourly target columns must be non-negative.")

    frame = frame.sort_values(
        ["region", TIMESTAMP_COLUMN], kind="stable"
    ).reset_index(drop=True)
    # Regional co-occurrence uses a timestamp-by-region matrix. Every region
    # must therefore cover the same hourly grid; otherwise a missing region-hour
    # could be mistaken for a genuine zero-alert observation.
    common_times = pd.Index(frame[TIMESTAMP_COLUMN].drop_duplicates()).sort_values()
    for region, group in frame.groupby("region", sort=False):
        region_times = group[TIMESTAMP_COLUMN].reset_index(drop=True)
        differences = region_times.diff().dropna()
        if not differences.eq(HOUR).all():
            raise ValueError(
                f"Region {region!r} does not have a complete hourly sequence. "
                "Reindex/aggregate missing hours before feature generation."
            )
        if len(region_times) != len(common_times) or not np.array_equal(
            region_times.to_numpy(), common_times.to_numpy()
        ):
            raise ValueError(
                f"Region {region!r} does not cover the same hourly range as every "
                "other region. Build a complete region-hour panel first; missing "
                "rows must not be interpreted as zero alerts."
            )
    return frame


def get_model_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return a whitelist of causal model input columns, excluding labels/metadata."""
    allowed_exact = {
        "region", "local_hour", "day_of_week", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    }
    prefixes = (
        "alert_start_count_lag_", "alert_minutes_lag_",
        "alert_start_count_rollsum_", "alert_start_count_rollmean_",
        "alert_minutes_rollsum_", "alert_minutes_rollmean_",
        "neighbor_active_count_lag_", "neighbor_active_fraction_lag_",
        "neighbor_active_count_rollsum_", "neighbor_active_fraction_rollmean_",
        # Only known derived text signals are allowed. This avoids accidentally
        # admitting raw report bodies, timestamps, or arbitrary text metadata.
        "text_report_count_", "text_missile_report_count_",
        "text_drone_report_count_", "text_air_activity_report_count_",
        "text_attack_report_count_", "text_official_alert_report_count_",
        "text_intensity_sum_", "text_high_intensity_count_",
        "text_mean_intensity_", "text_has_high_intensity_",
        "text_embedding_report_count_", "text_embedding_mean_",
        "text_coverage_fraction_",
    )
    allowed_text_exact = {"text_minutes_since_last_report", "text_has_prior_report"}
    return [
        column for column in frame.columns
        if column in allowed_exact or column in allowed_text_exact
        or column.startswith(prefixes)
    ]


def build_features(
    hourly: pd.DataFrame,
    *,
    lags_hours: Sequence[int] = DEFAULT_LAGS_HOURS,
    rolling_windows_hours: Sequence[int] = DEFAULT_ROLLING_WINDOWS_HOURS,
    neighbor_lags_hours: Sequence[int] = DEFAULT_NEIGHBOR_LAGS_HOURS,
    neighbor_rolling_hours: Sequence[int] = DEFAULT_NEIGHBOR_ROLLING_HOURS,
    neighbors: Mapping[str, Sequence[str]] | None = None,
    calendar_timezone: str = "Europe/Kyiv",
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """Create a causal supervised dataset from hourly alert outcome labels.

    Rows target the bucket identified by ``bucket_start``. All lag and rolling
    values come from buckets strictly before that target bucket. For a standard
    one-hour-ahead rolling forecast, the model predicts bucket ``t`` at the
    forecast boundary ``t`` using completed data from before ``t``.

    Args:
        hourly: Panel from ``data_loader.aggregate_hourly`` or equivalent,
            with one row per region and UTC hourly bucket.
        lags_hours: Past-hour offsets for alert count and alert-minutes features.
        rolling_windows_hours: Windows for trailing sum and mean features; each
            window excludes the target bucket.
        neighbor_lags_hours: Past-hour offsets for neighbor alert activity.
        neighbor_rolling_hours: Trailing windows for neighbor co-occurrence.
        neighbors: Region adjacency mapping. The default map uses canonical
            Ukrainian regional names and common aliases; override it if source
            labels/topology differ. Every observed region must match the graph,
            preventing silently meaningless all-zero neighbor features.
        calendar_timezone: Time zone used for hour/day calendar features.
        drop_incomplete: Drop warm-up rows without enough lag history.

    Returns:
        A dataframe containing timestamp metadata, causal predictors, original
        hourly targets, and binary ``alert_occurrence`` (any alert active during
        the target hour).

    Raises:
        ValueError: For invalid panel data or invalid feature-window parameters.
        AssertionError: If a generated predictor violates the temporal cutoff.
    """
    for name, values in (
        ("lags_hours", lags_hours),
        ("rolling_windows_hours", rolling_windows_hours),
        ("neighbor_lags_hours", neighbor_lags_hours),
        ("neighbor_rolling_hours", neighbor_rolling_hours),
    ):
        if not values or any(
            not isinstance(value, int) or value <= 0 for value in values
        ):
            raise ValueError(f"{name} must contain positive integer hour values.")

    frame = _validate_hourly_panel(hourly)
    graph = (
        _normalize_neighbor_mapping(neighbors)
        if neighbors is not None
        else get_default_neighbors()
    )
    observed_regions = set(frame["region"].map(normalize_region_name))
    graph_regions = set(graph)
    unmatched = sorted(observed_regions - graph_regions)
    if unmatched:
        raise ValueError(
            "No neighbor adjacency configured for normalized region key(s): "
            f"{unmatched}. Validate the default graph or pass a complete custom "
            "neighbors mapping."
        )

    local_time = frame[TIMESTAMP_COLUMN].dt.tz_convert(calendar_timezone)
    frame["local_hour"] = local_time.dt.hour.astype("int8")
    frame["day_of_week"] = local_time.dt.dayofweek.astype("int8")
    frame["is_weekend"] = local_time.dt.dayofweek.isin([5, 6]).astype("int8")
    frame["hour_sin"] = np.sin(2.0 * np.pi * frame["local_hour"] / 24.0)
    frame["hour_cos"] = np.cos(2.0 * np.pi * frame["local_hour"] / 24.0)
    frame["dow_sin"] = np.sin(2.0 * np.pi * frame["day_of_week"] / 7.0)
    frame["dow_cos"] = np.cos(2.0 * np.pi * frame["day_of_week"] / 7.0)

    frame["alert_occurrence"] = frame["alert_minutes"].gt(0.0).astype("int8")
    for target in TARGET_COLUMNS:
        group = frame.groupby("region", sort=False)[target]
        for lag in sorted(set(lags_hours)):
            frame[f"{target}_lag_{lag}h"] = group.shift(lag)
        for window in sorted(set(rolling_windows_hours)):
            # Perform rolling operations within each region. The previous code
            # shifted by group but called ``Series.transform`` afterward, which
            # obscured the intended group boundary and was fragile to row-order
            # or warm-up configuration changes.
            frame[f"{target}_rollsum_{window}h"] = group.transform(
                lambda values, window=window: values.shift(1).rolling(
                    window=window, min_periods=window
                ).sum()
            )
            frame[f"{target}_rollmean_{window}h"] = group.transform(
                lambda values, window=window: values.shift(1).rolling(
                    window=window, min_periods=window
                ).mean()
            )

    # Build one timestamp x region matrix of binary prior alert activity.
    # All neighbor features are shifted/rolled before being joined back, so
    # contemporaneous alerts in other regions do not leak into target bucket t.
    active_wide = frame.pivot(
        index=TIMESTAMP_COLUMN, columns="region", values="alert_occurrence"
    ).sort_index()
    normalized_to_actual: dict[str, list[str]] = {}
    for actual_region in active_wide.columns:
        key = normalize_region_name(actual_region)
        normalized_to_actual.setdefault(key, []).append(actual_region)

    neighbor_feature_frames: list[pd.DataFrame] = []
    for actual_region in frame["region"].drop_duplicates().tolist():
        region_key = normalize_region_name(actual_region)
        configured_neighbors = graph.get(region_key, ())
        actual_neighbors = sorted(
            neighbor
            for key in configured_neighbors
            for neighbor in normalized_to_actual.get(key, [])
            if neighbor != actual_region
        )
        region_features = pd.DataFrame(index=active_wide.index)
        if actual_neighbors:
            neighbor_active_count = active_wide[actual_neighbors].sum(axis=1).astype(
                "float64"
            )
            neighbor_count = float(len(actual_neighbors))
            neighbor_active_fraction = neighbor_active_count / neighbor_count
        else:
            neighbor_active_count = pd.Series(0.0, index=active_wide.index)
            neighbor_active_fraction = pd.Series(0.0, index=active_wide.index)

        for lag in sorted(set(neighbor_lags_hours)):
            column = f"neighbor_active_count_lag_{lag}h"
            region_features[column] = neighbor_active_count.shift(lag)
            column = f"neighbor_active_fraction_lag_{lag}h"
            region_features[column] = neighbor_active_fraction.shift(lag)
        prior_count = neighbor_active_count.shift(1)
        prior_fraction = neighbor_active_fraction.shift(1)
        for window in sorted(set(neighbor_rolling_hours)):
            column = f"neighbor_active_count_rollsum_{window}h"
            region_features[column] = prior_count.rolling(
                window=window, min_periods=window
            ).sum()
            column = f"neighbor_active_fraction_rollmean_{window}h"
            region_features[column] = prior_fraction.rolling(
                window=window, min_periods=window
            ).mean()

        region_features["region"] = actual_region
        region_features[TIMESTAMP_COLUMN] = region_features.index
        neighbor_feature_frames.append(region_features.reset_index(drop=True))

    neighbor_features = pd.concat(neighbor_feature_frames, ignore_index=True)
    frame = frame.merge(
        neighbor_features,
        on=["region", TIMESTAMP_COLUMN],
        how="left",
        validate="one_to_one",
    )

    # This field preserves compatibility with data_loader.py: it records the
    # latest historical bucket *start* used by the predictors, not wall-clock
    # availability time (the prior bucket completes at the target's start).
    frame[FEATURE_CUTOFF_COLUMN] = frame[TIMESTAMP_COLUMN] - HOUR
    if not (frame[FEATURE_CUTOFF_COLUMN] < frame[TIMESTAMP_COLUMN]).all():
        raise AssertionError(
            "Leakage detected: feature cutoff is not strictly before target bucket."
        )

    feature_columns = get_model_feature_columns(frame)
    if not feature_columns:
        raise AssertionError("No model feature columns were generated.")
    if drop_incomplete:
        previous_length = len(frame)
        frame = frame.dropna(subset=feature_columns).copy()
        LOGGER.info(
            "Removed %d warm-up rows with incomplete historical windows.",
            previous_length - len(frame),
        )
    if frame.empty:
        raise ValueError(
            "No rows remain after feature engineering. "
            "Supply more history or reduce lag/windows."
        )

    has_missing_features = bool(frame[feature_columns].isna().any().any())
    if drop_incomplete and has_missing_features:
        raise AssertionError(
            "Model features still contain missing values after warm-up rows were dropped."
        )
    if not drop_incomplete and has_missing_features:
        LOGGER.warning(
            "Warm-up rows are retained with missing historical features; use only for "
            "inspection unless a fold-fitted imputer/native-missing model is configured."
        )
    numeric_columns = [column for column in feature_columns if column != "region"]
    numeric_values = frame[numeric_columns].to_numpy(dtype=np.float64)
    if np.isinf(numeric_values).any():
        raise AssertionError("Numeric model features contain infinite values.")
    if drop_incomplete and np.isnan(numeric_values).any():
        raise AssertionError(
            "Model-ready features contain missing values after warm-up rows were dropped."
        )
    if not drop_incomplete and np.isnan(numeric_values).any():
        LOGGER.warning(
            "Warm-up rows were retained with missing lag features. This frame is for "
            "inspection only; the current model training entry point intentionally "
            "rejects missing predictors."
        )
    forbidden_columns = set(TARGET_COLUMNS) | {
        "alert_occurrence", FEATURE_CUTOFF_COLUMN
    }
    if set(feature_columns) & forbidden_columns:
        raise AssertionError(
            "A target or metadata column was accidentally included as a feature."
        )

    frame = frame.sort_values(
        [TIMESTAMP_COLUMN, "region"], kind="stable"
    ).reset_index(drop=True)
    LOGGER.info(
        "Built %d rows with %d causal features. Target range: %s to %s UTC.",
        len(frame),
        len(feature_columns),
        frame[TIMESTAMP_COLUMN].min(),
        frame[TIMESTAMP_COLUMN].max(),
    )
    return frame


def _read_neighbors(path: Path | None) -> Mapping[str, Sequence[str]] | None:
    """Load a custom region-adjacency JSON file when supplied."""
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict) or not all(
        isinstance(key, str) and isinstance(value, list)
        and all(isinstance(item, str) for item in value)
        for key, value in data.items()
    ):
        raise ValueError(
            "Neighbor JSON must map region strings to lists of region strings."
        )
    return data


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create command-line parser for the feature engineering entry point."""
    parser = argparse.ArgumentParser(
        description="Build causal hourly alert forecasting features."
    )
    parser.add_argument(
        "hourly_input", type=Path, help="Hourly CSV created by data_loader.py."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/model_data")
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--split-time", default=None,
        help="Optional start timestamp for out-of-time test.",
    )
    parser.add_argument(
        "--neighbors-json", type=Path, default=None,
        help="Optional custom adjacency mapping JSON.",
    )
    parser.add_argument("--calendar-timezone", default="Europe/Kyiv")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Load hourly data, engineer features, split by time, and save CSV outputs."""
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if not args.hourly_input.is_file():
        raise FileNotFoundError(f"Hourly input not found: {args.hourly_input}")
    hourly = pd.read_csv(args.hourly_input)
    features = build_features(
        hourly,
        neighbors=_read_neighbors(args.neighbors_json),
        calendar_timezone=args.calendar_timezone,
        drop_incomplete=True,
    )
    train, test = time_based_split(
        features, test_fraction=args.test_fraction, split_time=args.split_time
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output_dir / "supervised.csv", index=False)
    train.to_csv(args.output_dir / "train.csv", index=False)
    test.to_csv(args.output_dir / "test.csv", index=False)
    LOGGER.info("Saved features/train/test to %s", args.output_dir.resolve())
    LOGGER.info("Model feature columns: %s", get_model_feature_columns(features))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
