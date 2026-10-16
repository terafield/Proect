"""Stress tests for zero periods, spike targets, panel structure, and causal features."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data_loader import aggregate_hourly, time_based_split
from src.features import build_features, get_model_feature_columns
from src.text_features import build_point_in_time_text_features
from src.evaluate import evaluate_artifacts
from src.model import _make_pipeline, train_and_evaluate


class PipelineStressTests(unittest.TestCase):
    @staticmethod
    def _hourly_panel(hours: int = 12) -> pd.DataFrame:
        timestamps = pd.date_range("2025-01-01T00:00:00Z", periods=hours, freq="h")
        rows: list[dict[str, object]] = []
        for region in ("Region A", "Region B"):
            for index, timestamp in enumerate(timestamps):
                # A single spike plus otherwise quiet hours. Region B has no alerts.
                minutes = 45.0 if region == "Region A" and index == 2 else 0.0
                starts = 1 if region == "Region A" and index == 2 else 0
                rows.append({
                    "bucket_start": timestamp,
                    "region": region,
                    "alert_start_count": starts,
                    "alert_minutes": minutes,
                })
        return pd.DataFrame(rows)

    @staticmethod
    def _features(hourly: pd.DataFrame) -> pd.DataFrame:
        return build_features(
            hourly,
            lags_hours=(1, 2, 3),
            rolling_windows_hours=(3,),
            neighbor_lags_hours=(1,),
            neighbor_rolling_hours=(3,),
            neighbors={"Region A": ["Region B"], "Region B": ["Region A"]},
            drop_incomplete=True,
        )

    def test_lags_align_to_previous_completed_hour(self) -> None:
        hourly = self._hourly_panel()
        features = self._features(hourly)
        timestamp = pd.Timestamp("2025-01-01T04:00:00Z")
        row = features.loc[
            features["region"].eq("Region A")
            & features["bucket_start"].eq(timestamp)
        ].iloc[0]
        self.assertEqual(row["alert_minutes_lag_1h"], 0.0)
        self.assertEqual(row["alert_minutes_lag_2h"], 45.0)
        # Rolling window at 04:00 uses buckets 01:00, 02:00, and 03:00,
        # not the target bucket 04:00.
        self.assertEqual(row["alert_minutes_rollsum_3h"], 45.0)

    def test_mutating_current_target_does_not_change_current_features(self) -> None:
        hourly = self._hourly_panel()
        first = self._features(hourly)
        changed = hourly.copy()
        target_time = pd.Timestamp("2025-01-01T05:00:00Z")
        mask = changed["region"].eq("Region A") & changed["bucket_start"].eq(target_time)
        changed.loc[mask, "alert_minutes"] = 999.0
        changed.loc[mask, "alert_start_count"] = 25
        second = self._features(changed)
        feature_columns = get_model_feature_columns(first)
        left = first.loc[first["bucket_start"].eq(target_time)].sort_values("region")
        right = second.loc[second["bucket_start"].eq(target_time)].sort_values("region")
        pd.testing.assert_frame_equal(
            left[feature_columns].reset_index(drop=True),
            right[feature_columns].reset_index(drop=True),
        )

    def test_region_rolling_windows_do_not_borrow_another_region_history(self) -> None:
        hourly = self._hourly_panel()
        features = self._features(hourly)
        row = features.loc[
            features["region"].eq("Region B")
            & features["bucket_start"].eq(pd.Timestamp("2025-01-01T04:00:00Z"))
        ].iloc[0]
        self.assertEqual(row["alert_minutes_rollsum_3h"], 0.0)
        self.assertEqual(row["alert_start_count_rollsum_3h"], 0.0)

    def test_incomplete_region_hour_panel_is_rejected(self) -> None:
        hourly = self._hourly_panel()
        drop_mask = hourly["region"].eq("Region B") & hourly["bucket_start"].eq(
            pd.Timestamp("2025-01-01T05:00:00Z")
        )
        with self.assertRaisesRegex(ValueError, "complete hourly sequence|same hourly range"):
            self._features(hourly.loc[~drop_mask].copy())

    def test_train_and_evaluate_accepts_documented_timestamp_and_target_aliases(self) -> None:
        # Exercise the exact public API path that previously failed with
        # KeyError("bucket_start") when a mock used timestamp/occurrence/duration.
        hours = 72
        timestamps = pd.date_range("2025-01-01T00:00:00Z", periods=hours, freq="h")
        occurrence = ([1, 0, 0, 0] * (hours // 4))
        duration = [20.0 if value else 0.0 for value in occurrence]
        frame = pd.DataFrame({
            "timestamp": timestamps,
            "feature_cutoff": timestamps - pd.Timedelta(hours=1),
            "region": ["Demo Region"] * hours,
            "occurrence": occurrence,
            "duration": duration,
        })
        metrics, predictions, _, _ = train_and_evaluate(
            frame.iloc[:56].copy(),
            frame.iloc[56:].copy(),
            n_splits=2,
            random_state=7,
        )
        self.assertIn("classification", metrics)
        self.assertIn("duration_regression", metrics)
        self.assertIn("alert_probability", predictions.columns)
        self.assertIn("alert_minutes_prediction", predictions.columns)

    def test_missing_timestamp_raises_helpful_value_error_not_key_error(self) -> None:
        frame = pd.DataFrame({
            "feature_cutoff": [pd.Timestamp("2025-01-01T00:00:00Z")],
            "region": ["Demo Region"],
            "occurrence": [0],
            "duration": [0.0],
        })
        with self.assertRaisesRegex(ValueError, "Timestamp must be named"):
            from src.model import _canonicalize_model_frame
            _canonicalize_model_frame(frame, label="test fixture")

    def test_mass_overlap_is_not_clipped_and_zero_only_region_is_kept(self) -> None:
        start = pd.Timestamp("2025-01-01T00:00:00Z")
        end = start + pd.Timedelta(hours=1)
        events = pd.DataFrame({
            "start_time": [start, start],
            "end_time": [end, end],
            "region": ["Region A", "Region A"],
        })
        hourly = aggregate_hourly(
            events,
            coverage_start=start,
            coverage_end=start,
            regions=["Region A", "Region B"],
        )
        a = hourly.loc[hourly["region"].eq("Region A")].iloc[0]
        b = hourly.loc[hourly["region"].eq("Region B")].iloc[0]
        self.assertEqual(a["alert_start_count"], 2)
        self.assertEqual(a["alert_minutes"], 120.0)
        self.assertEqual(b["alert_start_count"], 0)
        self.assertEqual(b["alert_minutes"], 0.0)

    def test_empty_event_table_can_represent_covered_zero_alert_period(self) -> None:
        start = pd.Timestamp("2025-01-01T00:00:00Z")
        end = start + pd.Timedelta(hours=2)
        empty = pd.DataFrame({
            "start_time": pd.Series(dtype="datetime64[ns, UTC]"),
            "end_time": pd.Series(dtype="datetime64[ns, UTC]"),
            "region": pd.Series(dtype="string"),
        })
        hourly = aggregate_hourly(
            empty,
            coverage_start=start,
            coverage_end=end,
            regions=["Region A", "Region B"],
        )
        self.assertEqual(len(hourly), 6)
        self.assertTrue((hourly["alert_start_count"] == 0).all())
        self.assertTrue((hourly["alert_minutes"] == 0.0).all())

    def test_earlier_forecast_reference_is_rejected_for_hourly_lag_schema(self) -> None:
        base = self._features(self._hourly_panel(hours=12))
        base["forecast_reference_timestamp"] = (
            pd.to_datetime(base["bucket_start"], utc=True) - pd.Timedelta(minutes=5)
        )
        reports = pd.DataFrame({
            "text_timestamp": ["2025-01-01T00:02:00Z"],
            "text": ["missile report"],
            "region": ["Region A"],
        })
        with self.assertRaisesRegex(ValueError, "must equal bucket_start"):
            build_point_in_time_text_features(base, reports, windows_hours=(1,))

    def test_text_features_are_in_model_whitelist_but_reports_remain_point_in_time(self) -> None:
        hourly = self._hourly_panel(hours=12)
        base = self._features(hourly)
        events = pd.DataFrame({
            "text_timestamp": ["2025-01-01T03:00:00Z", "2025-01-01T05:00:00Z"],
            "text": ["missile report", "drone report"],
            "region": ["Region A", "Region A"],
        })
        enriched = build_point_in_time_text_features(base, events, windows_hours=(1, 6))
        columns = get_model_feature_columns(enriched)
        self.assertIn("text_report_count_1h", columns)
        row = enriched.loc[
            enriched["region"].eq("Region A")
            & enriched["bucket_start"].eq(pd.Timestamp("2025-01-01T04:00:00Z"))
        ].iloc[0]
        self.assertEqual(row["text_report_count_1h"], 1.0)
        # Exact timestamp at 00:05 is not included in the 00:05 forecast row.
        row_at_exact = enriched.loc[
            enriched["region"].eq("Region A")
            & enriched["bucket_start"].eq(pd.Timestamp("2025-01-01T05:00:00Z"))
        ].iloc[0]
        self.assertEqual(row_at_exact["text_report_count_1h"], 0.0)

    def test_evaluation_writes_charts_metrics_and_importance(self) -> None:
        hourly = self._hourly_panel(hours=80)
        # Add recurring alert events so both train and test contain positive and
        # negative examples and the classifier can be fitted for the smoke test.
        hourly = hourly.copy()
        hours = pd.to_datetime(hourly["bucket_start"], utc=True)
        for region in ("Region A", "Region B"):
            region_mask = hourly["region"].eq(region)
            alert_hours = hours.dt.hour.isin([4, 5, 17, 18])
            hourly.loc[region_mask & alert_hours, "alert_minutes"] = 20.0
            hourly.loc[region_mask & alert_hours, "alert_start_count"] = 1

        supervised = build_features(
            hourly,
            lags_hours=(1, 2, 3),
            rolling_windows_hours=(3,),
            neighbor_lags_hours=(1,),
            neighbor_rolling_hours=(3,),
            neighbors={"Region A": ["Region B"], "Region B": ["Region A"]},
        )
        train, test = time_based_split(supervised, test_fraction=0.25)
        feature_columns = get_model_feature_columns(train)
        classifier = _make_pipeline(task="classification", feature_columns=feature_columns)
        regressor = _make_pipeline(task="regression", feature_columns=feature_columns)
        classifier.fit(train[feature_columns], train["alert_occurrence"].astype(int))
        regressor.fit(train[feature_columns], train["alert_minutes"].astype(float))

        with tempfile.TemporaryDirectory() as temp_dir:
            result = evaluate_artifacts(
                train, test, classifier, regressor,
                output_dir=temp_dir,
                top_n_features=3,
                n_repeats=1,
                max_importance_rows=40,
                max_regions=2,
            )
            self.assertIn("classification", result)
            self.assertTrue((Path(temp_dir) / "actual_vs_predicted_alert_minutes.png").is_file())
            self.assertTrue((Path(temp_dir) / "actual_vs_predicted_occurrence.png").is_file())
            self.assertTrue((Path(temp_dir) / "permutation_importance_duration.png").is_file())
            self.assertTrue((Path(temp_dir) / "target_drift_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
