"""Generate deterministic synthetic inputs for an end-to-end smoke test.

The generated events are artificial and must never be treated as historical
Ukraine alert data or used to claim real-world model performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def generate_demo_data(output_dir: Path, hours: int = 520) -> tuple[Path, Path, Path]:
    """Write synthetic alert events, public-text events, and region adjacency."""
    if hours < 240:
        raise ValueError("Use at least 240 hours for a minimally useful demo history.")
    output_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp("2025-01-01T00:00:00Z")
    regions = ("Demo North", "Demo South")
    alert_rows: list[dict[str, object]] = []

    for region_index, region in enumerate(regions):
        event_index = 0
        hour = 1 + 2 * region_index
        while hour < hours - 3:
            start_time = start + pd.Timedelta(
                hours=hour,
                minutes=(region_index * 7 + event_index * 3) % 45,
            )
            duration_seconds = 1800 + (
                hour * 17 + event_index * 29 + region_index * 13
            ) % 7200
            end_time = start_time + pd.Timedelta(seconds=duration_seconds)
            alert_rows.append({
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": duration_seconds,
                "region": region,
            })
            hour += 4 + ((event_index + region_index) % 5)
            event_index += 1

    alert_path = output_dir / "alerts.csv"
    pd.DataFrame(alert_rows).to_csv(alert_path, index=False)

    text_rows: list[dict[str, object]] = []
    messages = (
        "Public update about drone activity",
        "Official statement on elevated threat conditions",
        "Routine public safety information",
        "Report mentions possible missile activity",
    )
    index = 0
    timestamp = start + pd.Timedelta(minutes=5)
    end = start + pd.Timedelta(hours=hours - 1)
    while timestamp < end:
        text_rows.append({
            "text_timestamp": timestamp.isoformat(),
            "text": messages[index % len(messages)],
            "region": "GLOBAL",
        })
        timestamp += pd.Timedelta(hours=5)
        index += 1
    text_path = output_dir / "text_events.csv"
    pd.DataFrame(text_rows).to_csv(text_path, index=False)

    neighbors_path = output_dir / "neighbors.json"
    neighbors_path.write_text(
        json.dumps({regions[0]: [regions[1]], regions[1]: [regions[0]]}, indent=2),
        encoding="utf-8",
    )
    coverage_end = start + pd.Timedelta(hours=hours - 1)
    print(f"Synthetic demo inputs written to: {output_dir.resolve()}")
    print(f"Alerts: {len(alert_rows)}; text reports: {len(text_rows)}; hours: {hours}")
    print(f"Coverage: {start.isoformat()} through {coverage_end.isoformat()} (inclusive)")
    print("Reminder: all generated events and reports are synthetic demo data.")
    return alert_path, text_path, neighbors_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/demo"))
    parser.add_argument("--hours", type=int, default=520)
    args = parser.parse_args()
    generate_demo_data(args.output_dir, args.hours)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
