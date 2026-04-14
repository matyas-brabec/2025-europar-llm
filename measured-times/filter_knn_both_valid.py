#!/usr/bin/env python3

import argparse
import csv
import pathlib
import sys


ARCHITECTURES = ["ampere", "volta", "hopper", "blackwell"]
FIELDNAMES = ["experiment_id", "time", "std", "compiled", "verified", "runtime_err", "extra"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create filtered k-NN CSV pairs that keep only solution attempts valid for both k=32 and k=1024."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent,
        help="Directory containing knn-32-<arch>.csv and knn-1024-<arch>.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Directory where the filtered CSV files should be written.",
    )
    parser.add_argument(
        "--architectures",
        nargs="*",
        default=ARCHITECTURES,
        help="Architectures to process.",
    )
    return parser.parse_args()


def read_rows(csv_path: pathlib.Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if reader.fieldnames != FIELDNAMES:
            raise ValueError(f"{csv_path} has unexpected columns: {reader.fieldnames}")
        return list(reader)


def row_is_valid(row: dict[str, str]) -> bool:
    return row["compiled"] == "True" and row["verified"] == "True" and row["runtime_err"] == ""


def write_rows(csv_path: pathlib.Path, rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def filter_architecture(input_dir: pathlib.Path, output_dir: pathlib.Path, architecture: str) -> None:
    k32_path = input_dir / f"knn-32-{architecture}.csv"
    k1024_path = input_dir / f"knn-1024-{architecture}.csv"

    k32_rows = read_rows(k32_path)
    k1024_rows = read_rows(k1024_path)

    k32_by_id = {row["experiment_id"]: row for row in k32_rows}
    k1024_by_id = {row["experiment_id"]: row for row in k1024_rows}
    if k32_by_id.keys() != k1024_by_id.keys():
        missing = sorted(k32_by_id.keys() - k1024_by_id.keys())
        extra = sorted(k1024_by_id.keys() - k32_by_id.keys())
        raise ValueError(
            f"{architecture}: knn-32 and knn-1024 CSVs use different experiment IDs: "
            f"missing={missing}, extra={extra}"
        )

    valid_on_both = {
        experiment_id
        for experiment_id in k32_by_id
        if row_is_valid(k32_by_id[experiment_id]) and row_is_valid(k1024_by_id[experiment_id])
    }

    filtered_k32 = [row for row in k32_rows if row["experiment_id"] in valid_on_both]
    filtered_k1024 = [row for row in k1024_rows if row["experiment_id"] in valid_on_both]

    write_rows(output_dir / f"knn-32-{architecture}-both-valid.csv", filtered_k32)
    write_rows(output_dir / f"knn-1024-{architecture}-both-valid.csv", filtered_k1024)

    print(
        f"{architecture}: kept {len(valid_on_both)}/{len(k32_rows)} solution attempts "
        f"valid in both k settings"
    )


def main() -> int:
    args = parse_args()
    for architecture in args.architectures:
        filter_architecture(args.input_dir, args.output_dir, architecture)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
