#!/usr/bin/env python3

import argparse
import pathlib
import re
import shutil
import sys


JOB_CSV_RE = re.compile(r"job-(?P<architecture>[^-]+)-\d+\.csv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve job-ID CSVs from framework/__log_dir__ into measured-times/<target>.csv "
            "by sampling only the first N data rows."
        )
    )
    parser.add_argument(
        "--log-dir",
        type=pathlib.Path,
        default=pathlib.Path("framework/__log_dir__"),
        help="Directory containing job-*.csv and job-*.err files.",
    )
    parser.add_argument(
        "--target-dir",
        type=pathlib.Path,
        default=pathlib.Path("measured-times"),
        help="Directory where resolved <target>.csv and <target>.err files should end up.",
    )
    parser.add_argument(
        "-n",
        "--sample-size",
        type=int,
        default=5,
        help="Number of data rows to inspect per CSV.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Move the files instead of only printing the planned renames.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy the files instead of moving (implies --apply).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting existing destination files.",
    )
    return parser.parse_args()


def get_architecture(csv_path: pathlib.Path) -> str:
    match = JOB_CSV_RE.fullmatch(csv_path.name)
    if match is None:
        raise ValueError(f"{csv_path} does not look like a job CSV filename")
    return match.group("architecture")


def parse_data_row(line: str, line_number: int, csv_path: pathlib.Path) -> tuple[str, str]:
    fields = line.rstrip("\n").split(";")
    if len(fields) < 7:
        raise ValueError(
            f"{csv_path}:{line_number} is malformed: expected at least 7 semicolon-separated fields"
        )

    experiment_id = fields[0].strip()
    extra = fields[6].strip()
    if not experiment_id:
        raise ValueError(f"{csv_path}:{line_number} has an empty experiment_id")

    return experiment_id, extra


def resolve_target_name(architecture: str, experiment_id: str, extra: str) -> str:
    if experiment_id.startswith("game_of_life"):
        stem = "gol"
    elif experiment_id.startswith("histogram"):
        if extra == "DATA=hex":
            stem = "histogram-hexdump"
        elif extra == "":
            stem = "histogram-loremipsum"
        else:
            raise ValueError(
                f"cannot resolve histogram dataset from extra={extra!r} for {experiment_id}"
            )
    elif experiment_id.startswith("knn"):
        if extra == "K=32":
            stem = "knn-32"
        elif extra == "":
            stem = "knn-1024"
        else:
            raise ValueError(f"cannot resolve knn variant from extra={extra!r} for {experiment_id}")
    else:
        raise ValueError(f"cannot resolve experiment family from experiment_id={experiment_id!r}")

    return f"{stem}-{architecture}.csv"


def inspect_csv(csv_path: pathlib.Path, sample_size: int) -> tuple[str, list[tuple[int, str, str]]]:
    if sample_size <= 0:
        raise ValueError("--sample-size must be positive")

    architecture = get_architecture(csv_path)
    resolved_rows: list[tuple[int, str, str]] = []

    with csv_path.open("r", encoding="utf-8") as handle:
        header = next(handle, None)
        if header is None:
            raise ValueError(f"{csv_path} is empty")

        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue

            experiment_id, extra = parse_data_row(line, line_number, csv_path)
            target_name = resolve_target_name(architecture, experiment_id, extra)
            resolved_rows.append((line_number, experiment_id, target_name))

            if len(resolved_rows) >= sample_size:
                break

    if not resolved_rows:
        raise ValueError(f"{csv_path} contains no data rows")

    target_names = {target_name for _, _, target_name in resolved_rows}
    if len(target_names) != 1:
        details = ", ".join(
            f"line {line_number} ({experiment_id}) -> {target_name}"
            for line_number, experiment_id, target_name in resolved_rows
        )
        raise ValueError(f"{csv_path} resolves to multiple targets within the sampled rows: {details}")

    return resolved_rows[0][2], resolved_rows


def ensure_destination_free(
    destination: pathlib.Path, force: bool, pending_targets: set[pathlib.Path], apply_changes: bool
) -> None:
    if destination in pending_targets:
        raise ValueError(f"multiple source files would write to {destination}")

    if apply_changes and destination.exists() and not force:
        raise ValueError(
            f"destination already exists: {destination} (use --force to overwrite when applying)"
        )


def should_transfer_err(err_path: pathlib.Path) -> bool:
    return err_path.exists() and err_path.stat().st_size > 0


def main() -> int:
    args = parse_args()
    csv_paths = sorted(args.log_dir.glob("job-*.csv"))

    if args.copy:
        args.apply = True

    if not csv_paths:
        print(f"No job CSV files found in {args.log_dir}", file=sys.stderr)
        return 1

    plan: list[
        tuple[
            pathlib.Path,
            pathlib.Path,
            pathlib.Path,
            pathlib.Path | None,
            list[tuple[int, str, str]],
        ]
    ] = []
    pending_targets: set[pathlib.Path] = set()
    pending_err_targets: set[pathlib.Path] = set()

    for csv_path in csv_paths:
        target_name, resolved_rows = inspect_csv(csv_path, args.sample_size)
        source_err_path = csv_path.with_suffix(".err")

        target_csv_path = args.target_dir / target_name
        target_err_path = (
            target_csv_path.with_suffix(".err") if should_transfer_err(source_err_path) else None
        )

        ensure_destination_free(target_csv_path, args.force, pending_targets, args.apply)
        if target_err_path is not None:
            ensure_destination_free(target_err_path, args.force, pending_err_targets, args.apply)

        pending_targets.add(target_csv_path)
        if target_err_path is not None:
            pending_err_targets.add(target_err_path)
        plan.append((csv_path, target_csv_path, source_err_path, target_err_path, resolved_rows))

    for source_csv_path, target_csv_path, source_err_path, target_err_path, resolved_rows in plan:
        sample_description = ", ".join(
            f"line {line_number}:{experiment_id}"
            for line_number, experiment_id, _ in resolved_rows
        )
        print(f"{source_csv_path} -> {target_csv_path} [{sample_description}]")
        if target_err_path is not None:
            print(f"{source_err_path} -> {target_err_path}")
        else:
            print(f"{source_err_path} [skip: missing or empty]")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to move the files.")
        return 0

    args.target_dir.mkdir(parents=True, exist_ok=True)
    action_verb = "Copied" if args.copy else "Moved"
    err_count = 0

    for source_csv_path, target_csv_path, source_err_path, target_err_path, _ in plan:
        target_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if args.copy:
            shutil.copy2(source_csv_path, target_csv_path)
        else:
            source_csv_path.replace(target_csv_path)

        if target_err_path is not None:
            target_err_path.parent.mkdir(parents=True, exist_ok=True)
            if args.copy:
                shutil.copy2(source_err_path, target_err_path)
            else:
                source_err_path.replace(target_err_path)
            err_count += 1

    print(f"\n{action_verb} {len(plan)} CSV files and {err_count} non-empty ERR files.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
