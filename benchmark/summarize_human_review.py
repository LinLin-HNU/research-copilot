"""Summarize manually labelled citation and grounding review results."""
import argparse
import csv


def as_bool(value: str) -> bool | None:
    value = value.strip().lower()
    if value in {"1", "true", "yes"}:
        return True
    if value in {"0", "false", "no"}:
        return False
    return None


def rate(rows: list[dict], field: str, expected: bool = True) -> str:
    values = [as_bool(row.get(field, "")) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return "n/a"
    return f"{sum(value is expected for value in values) / len(values):.1%} ({len(values)} reviewed)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("review_csv")
    args = parser.parse_args()
    with open(args.review_csv, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    print(f"Citation correctness: {rate(rows, 'citation_correct')}")
    print(f"Grounded answers: {rate(rows, 'grounded')}")
    print(f"Hallucination rate: {rate(rows, 'hallucination')}")


if __name__ == "__main__":
    main()
