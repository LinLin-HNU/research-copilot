"""Read-only summary for request_metrics telemetry."""
import argparse
import sqlite3
from statistics import mean

from database import DB_PATH, init_db


def percentile(values: list[float], value: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, round((len(values) - 1) * value))
    return round(values[index], 2)


def metric_summary(rows: list[sqlite3.Row], field: str) -> dict:
    values = [float(row[field]) for row in rows if row[field] is not None]
    return {
        "count": len(values),
        "mean": round(mean(values), 2) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    init_db()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM request_metrics ORDER BY id DESC LIMIT ?", (args.limit,)
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        print("No request metrics recorded yet.")
        return

    print(f"Requests sampled: {len(rows)}")
    print(f"Success rate: {sum(bool(row['success']) for row in rows) / len(rows):.1%}")
    for field in ("total_ms", "download_ms", "parse_ms", "index_ms", "routing_ms", "retrieval_ms", "generation_ms"):
        print(f"{field}: {metric_summary(rows, field)}")
    print(f"evidence_chars: {metric_summary(rows, 'evidence_chars')}")
    print(f"output_chars: {metric_summary(rows, 'output_chars')}")
    print(f"input_tokens: {metric_summary(rows, 'input_tokens')}")
    print(f"output_tokens: {metric_summary(rows, 'output_tokens')}")


if __name__ == "__main__":
    main()
