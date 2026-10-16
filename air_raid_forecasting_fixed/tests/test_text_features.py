"""Small regression tests for point-in-time text feature joins."""

from __future__ import annotations

import unittest

import pandas as pd

from src.data_loader import FEATURE_CUTOFF_COLUMN, TIMESTAMP_COLUMN
from src.text_features import build_point_in_time_text_features


class TextFeaturePointInTimeTests(unittest.TestCase):
    """Verify strict cutoff, regional scope, empty feeds, and embeddings."""

    def setUp(self) -> None:
        self.times = pd.date_range(
            "2026-01-01T00:00:00Z", periods=180, freq="h"
        )
        rows = []
        for region in ("A", "B"):
            for timestamp in self.times:
                rows.append(
                    {
                        TIMESTAMP_COLUMN: timestamp,
                        "region": region,
                        FEATURE_CUTOFF_COLUMN: timestamp - pd.Timedelta(hours=1),
                        "alert_start_count": 0.0,
                        "alert_minutes": 0.0,
                        "alert_occurrence": 0,
                        "local_hour": timestamp.hour,
                        "day_of_week": timestamp.dayofweek,
                        "is_weekend": int(timestamp.dayofweek >= 5),
                        "hour_sin": 0.0,
                        "hour_cos": 1.0,
                        "dow_sin": 0.0,
                        "dow_cos": 1.0,
                        "alert_start_count_lag_1h": 0.0,
                        "alert_minutes_lag_1h": 0.0,
                    }
                )
        self.series = pd.DataFrame(rows)

    def test_exact_forecast_timestamp_is_excluded(self) -> None:
        timestamp = self.times[10]
        reports = pd.DataFrame(
            [
                {
                    "text_timestamp": timestamp - pd.Timedelta(minutes=5),
                    "region": "A",
                    "text": "missile report",
                },
                {
                    "text_timestamp": timestamp,
                    "region": "A",
                    "text": "drone report at forecast timestamp",
                },
            ]
        )
        result = build_point_in_time_text_features(
            self.series, reports, windows_hours=(1, 6)
        )
        row = result.loc[
            result["region"].eq("A") & result[TIMESTAMP_COLUMN].eq(timestamp)
        ].iloc[0]
        self.assertEqual(row["text_report_count_1h"], 1.0)
        self.assertEqual(row["text_missile_report_count_1h"], 1.0)
        self.assertEqual(row["text_drone_report_count_1h"], 0.0)
        self.assertLess(row[FEATURE_CUTOFF_COLUMN], row[TIMESTAMP_COLUMN])

    def test_global_reports_apply_to_all_regions_but_local_do_not(self) -> None:
        timestamp = self.times[10]
        reports = pd.DataFrame(
            [
                {
                    "text_timestamp": timestamp - pd.Timedelta(minutes=5),
                    "region": "A",
                    "text": "missile report",
                },
                {
                    "text_timestamp": timestamp - pd.Timedelta(minutes=10),
                    "region": None,
                    "text": "nationwide air alert",
                },
            ]
        )
        result = build_point_in_time_text_features(
            self.series, reports, windows_hours=(1,)
        )
        row_a = result.loc[
            result["region"].eq("A") & result[TIMESTAMP_COLUMN].eq(timestamp)
        ].iloc[0]
        row_b = result.loc[
            result["region"].eq("B") & result[TIMESTAMP_COLUMN].eq(timestamp)
        ].iloc[0]
        self.assertEqual(row_a["text_report_count_1h"], 2.0)
        self.assertEqual(row_b["text_report_count_1h"], 1.0)
        self.assertEqual(row_b["text_missile_report_count_1h"], 0.0)

    def test_empty_feed_generates_zeros(self) -> None:
        reports = pd.DataFrame(
            columns=["text_timestamp", "region", "text"]
        )
        result = build_point_in_time_text_features(
            self.series, reports, windows_hours=(1,)
        )
        self.assertTrue(result["text_report_count_1h"].eq(0.0).all())

    def test_embedding_features_are_aggregated_point_in_time(self) -> None:
        timestamp = self.times[10]
        reports = pd.DataFrame(
            [
                {
                    "text_timestamp": timestamp - pd.Timedelta(minutes=1),
                    "region": "A",
                    "text": "air alert",
                    "embedding_0": 0.25,
                    "embedding_1": 0.75,
                },
                {
                    "text_timestamp": timestamp,
                    "region": "A",
                    "text": "future-at-cutoff report",
                    "embedding_0": 9.0,
                    "embedding_1": 9.0,
                },
            ]
        )
        result = build_point_in_time_text_features(
            self.series, reports, windows_hours=(1,)
        )
        row = result.loc[
            result["region"].eq("A") & result[TIMESTAMP_COLUMN].eq(timestamp)
        ].iloc[0]
        self.assertAlmostEqual(row["text_embedding_mean_1h_0"], 0.25)
        self.assertAlmostEqual(row["text_embedding_mean_1h_1"], 0.75)


if __name__ == "__main__":
    unittest.main()
