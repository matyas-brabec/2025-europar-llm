#!/usr/bin/env python3

import argparse
import csv
import pathlib
import re
import sys
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from typing import Callable


STATUS_OK = "✅"
STATUS_WRONG = "❌"
STATUS_CRASH = "❌💥"
STATUS_COMPILE = "❌⚙️"
STATUS_ORDER = [STATUS_OK, STATUS_WRONG, STATUS_CRASH, STATUS_COMPILE]

SUMMARY_RE = re.compile(
    r"^(?P<symbol>❌💥|❌⚙️|✅|❌)\s+–.*:\s+(?P<count>\d+)/(?P<total>\d+)\s+\((?P<percent>\d+)%\)\s*$"
)
FIXED_MARKER = "/// @FIXED"
REVIEW_CELL = tuple[str, bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the review markdown tables and summary rates against the measured-times CSV files."
        )
    )
    parser.add_argument(
        "reviews",
        nargs="*",
        choices=["knn", "histogram", "game-of-life"],
        help="Subset of review files to validate. Defaults to all.",
    )
    return parser.parse_args()


def round_percent(count: int, total: int) -> int:
    if total == 0:
        return 0
    value = Decimal(count * 100) / Decimal(total)
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def csv_status(row: dict[str, str]) -> str:
    compiled = row["compiled"].strip() == "True"
    verified = row["verified"].strip() == "True"
    runtime_err = row["runtime_err"].strip()

    if not compiled:
        return STATUS_COMPILE
    if runtime_err:
        return STATUS_CRASH
    if verified:
        return STATUS_OK
    return STATUS_WRONG


def load_csv_matrix(csv_path: pathlib.Path) -> dict[str, list[str]]:
    matrix: dict[str, dict[str, str]] = {}

    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            experiment_id = row["experiment_id"].split("/")[0]
            testcase, attempt = experiment_id.rsplit("-", 1)
            matrix.setdefault(testcase, {})[attempt] = csv_status(row)

    normalized: dict[str, list[str]] = {}
    for testcase, attempts in sorted(matrix.items()):
        normalized[testcase] = [attempts[f"{index:02d}"] for index in range(1, 11)]
    return normalized


def matrices_match(
    reference: dict[str, list[str]], candidate: dict[str, list[str]]
) -> tuple[bool, str | None]:
    if reference.keys() != candidate.keys():
        missing = sorted(reference.keys() - candidate.keys())
        extra = sorted(candidate.keys() - reference.keys())
        return False, f"testcase mismatch: missing={missing}, extra={extra}"

    for testcase in sorted(reference):
        if reference[testcase] != candidate[testcase]:
            return (
                False,
                f"status mismatch for {testcase}: {reference[testcase]} != {candidate[testcase]}",
            )

    return True, None


def load_consistent_matrix(csv_paths: list[pathlib.Path]) -> dict[str, list[str]]:
    reference = load_csv_matrix(csv_paths[0])
    for csv_path in csv_paths[1:]:
        candidate = load_csv_matrix(csv_path)
        matches, reason = matrices_match(reference, candidate)
        if not matches:
            raise ValueError(f"{csv_path} is inconsistent with {csv_paths[0]}: {reason}")
    return reference


def normalize_cell(text: str) -> str:
    return text.replace("\u00a0", " ").strip()


def canonical_status(text: str) -> str:
    cleaned = normalize_cell(text).replace(" ", "").replace("🛠️", "").replace("🛠", "")
    if cleaned not in STATUS_ORDER:
        raise ValueError(f"unknown review status cell: {text!r}")
    return cleaned


def has_fixed_indicator(text: str) -> bool:
    cleaned = normalize_cell(text).replace(" ", "")
    return "🛠️" in cleaned or "🛠" in cleaned


def parse_summary_sections(markdown_path: pathlib.Path) -> dict[str, dict[str, tuple[int, int, int]]]:
    sections: dict[str, dict[str, tuple[int, int, int]]] = {}
    current_section = "default"

    for raw_line in markdown_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("**") and line.endswith("**"):
            current_section = line.strip("*")
            sections.setdefault(current_section, {})
            continue

        match = SUMMARY_RE.match(line)
        if match is None:
            continue

        sections.setdefault(current_section, {})
        sections[current_section][match.group("symbol")] = (
            int(match.group("count")),
            int(match.group("total")),
            int(match.group("percent")),
        )

    return {name: values for name, values in sections.items() if values}


def parse_review_tables(
    markdown_path: pathlib.Path, row_normalizer: Callable[[str], str]
) -> dict[str, dict[str, list[REVIEW_CELL]]]:
    sections: dict[str, dict[str, list[REVIEW_CELL]]] = {}
    current_section = "default"

    for raw_line in markdown_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("**") and line.endswith("**"):
            current_section = line.strip("*")
            sections.setdefault(current_section, {})
            continue

        if not line.startswith("|"):
            continue

        cells = [normalize_cell(cell) for cell in line.strip("|").split("|")]
        if not cells:
            continue

        first_cell = cells[0]
        if first_cell.lower().startswith("test case"):
            continue
        if all(set(cell) <= {"-", ":"} for cell in cells if cell):
            continue

        testcase = row_normalizer(first_cell)
        statuses = [(canonical_status(cell), has_fixed_indicator(cell)) for cell in cells[1:] if cell]
        if len(statuses) != 10:
            raise ValueError(
                f"{markdown_path}: expected 10 status columns for {first_cell}, got {len(statuses)}"
            )
        sections.setdefault(current_section, {})
        sections[current_section][testcase] = statuses

    return {name: rows for name, rows in sections.items() if rows}


def normalize_knn_row(label: str) -> str:
    return label.lower()


def normalize_histogram_row(label: str) -> str:
    return label.lower()


def normalize_gol_row(label: str) -> str:
    normalized = label.lower()
    match = re.fullmatch(r"gol_(\d{2})(?:_tiled)?", normalized)
    if match is None:
        raise ValueError(f"cannot normalize GoL row label: {label!r}")
    suffix = "_tiled" if normalized.endswith("_tiled") else ""
    return f"game_of_life{match.group(1)}{suffix}"


def compare_tables(
    reported: dict[str, list[REVIEW_CELL]],
    expected_statuses: dict[str, list[str]],
    expected_fixed: dict[str, list[bool]],
    context: str,
) -> list[str]:
    errors: list[str] = []

    if reported.keys() != expected_statuses.keys():
        missing = sorted(expected_statuses.keys() - reported.keys())
        extra = sorted(reported.keys() - expected_statuses.keys())
        errors.append(f"{context}: testcase mismatch: missing={missing}, extra={extra}")
        return errors

    for testcase in sorted(expected_statuses):
        reported_statuses = [status for status, _ in reported[testcase]]
        reported_fixed = [fixed for _, fixed in reported[testcase]]

        if reported_statuses != expected_statuses[testcase]:
            errors.append(
                f"{context}: {testcase} status differs: "
                f"reported={reported_statuses} expected={expected_statuses[testcase]}"
            )
        if reported_fixed != expected_fixed[testcase]:
            errors.append(
                f"{context}: {testcase} indicator differs: "
                f"reported={reported_fixed} expected={expected_fixed[testcase]}"
            )

    return errors


def aggregate_counts(sections: dict[str, dict[str, list[str]]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for matrix in sections.values():
        for statuses in matrix.values():
            counts.update(statuses)
    return counts


def combine_knn_sections(
    first: dict[str, list[str]], second: dict[str, list[str]]
) -> dict[str, list[str]]:
    if first.keys() != second.keys():
        missing = sorted(first.keys() - second.keys())
        extra = sorted(second.keys() - first.keys())
        raise ValueError(f"kNN sections use different testcase sets: missing={missing}, extra={extra}")

    combined: dict[str, list[str]] = {}
    for testcase in sorted(first):
        row: list[str] = []
        if len(first[testcase]) != len(second[testcase]):
            raise ValueError(f"kNN sections use different row widths for {testcase}")

        for status_a, status_b in zip(first[testcase], second[testcase]):
            statuses = {status_a, status_b}
            if STATUS_COMPILE in statuses:
                row.append(STATUS_COMPILE)
            elif STATUS_CRASH in statuses:
                row.append(STATUS_CRASH)
            elif STATUS_WRONG in statuses:
                row.append(STATUS_WRONG)
            else:
                row.append(STATUS_OK)

        combined[testcase] = row

    return combined


def load_fixed_indicator_matrix(
    results_root: pathlib.Path, source_filename: str, expected_sections: dict[str, dict[str, list[str]]]
) -> dict[str, dict[str, list[bool]]]:
    matrices: dict[str, dict[str, list[bool]]] = {}

    for section_name, section in expected_sections.items():
        matrices[section_name] = {}
        for testcase in sorted(section):
            flags: list[bool] = []
            for index in range(1, 11):
                source_path = results_root / f"{testcase}-{index:02d}" / source_filename
                if not source_path.exists():
                    raise ValueError(f"missing generated source file: {source_path}")
                flags.append(FIXED_MARKER in source_path.read_text(encoding="utf-8"))
            matrices[section_name][testcase] = flags

    return matrices


def validate_review(
    markdown_path: pathlib.Path,
    expected_sections: dict[str, dict[str, list[str]]],
    row_normalizer: Callable[[str], str],
    expected_fixed_sections: dict[str, dict[str, list[bool]]],
    expected_summary_sections: dict[str, Counter[str]],
) -> list[str]:
    errors: list[str] = []
    reported_sections = parse_review_tables(markdown_path, row_normalizer)

    if reported_sections.keys() != expected_sections.keys():
        missing = sorted(expected_sections.keys() - reported_sections.keys())
        extra = sorted(reported_sections.keys() - expected_sections.keys())
        errors.append(f"{markdown_path}: section mismatch: missing={missing}, extra={extra}")
    else:
        for section_name, expected_matrix in expected_sections.items():
            errors.extend(
                compare_tables(
                    reported_sections[section_name],
                    expected_matrix,
                    expected_fixed_sections[section_name],
                    f"{markdown_path}:{section_name}",
                )
            )

    reported_summary_sections = parse_summary_sections(markdown_path)
    if reported_summary_sections.keys() != expected_summary_sections.keys():
        missing = sorted(expected_summary_sections.keys() - reported_summary_sections.keys())
        extra = sorted(reported_summary_sections.keys() - expected_summary_sections.keys())
        errors.append(f"{markdown_path}: summary section mismatch: missing={missing}, extra={extra}")
    else:
        for section_name, counts in expected_summary_sections.items():
            reported_summary = reported_summary_sections[section_name]
            if set(reported_summary) != set(STATUS_ORDER):
                errors.append(f"{markdown_path}:{section_name}: missing one or more summary lines")
                continue

            total = sum(counts.values())
            for status in STATUS_ORDER:
                expected_count = counts[status]
                expected_percent = round_percent(expected_count, total)
                reported_count, reported_total, reported_percent = reported_summary[status]

                if (
                    reported_count != expected_count
                    or reported_total != total
                    or reported_percent != expected_percent
                ):
                    errors.append(
                        f"{markdown_path}:{section_name}: summary mismatch for {status}: "
                        f"reported={reported_count}/{reported_total} ({reported_percent}%) "
                        f"expected={expected_count}/{total} ({expected_percent}%)"
                    )

    return errors


def review_specs(
    repo_root: pathlib.Path,
) -> dict[
    str,
    tuple[
        pathlib.Path,
        dict[str, dict[str, list[str]]],
        dict[str, dict[str, list[bool]]],
        dict[str, Counter[str]],
        Callable[[str], str],
    ],
]:
    measured_times = repo_root / "measured-times"
    reviews = repo_root / "reviews"
    results = repo_root / "results"

    histogram_matrix = load_consistent_matrix(
        [
            measured_times / "histogram-hexdump-blackwell.csv",
            measured_times / "histogram-loremipsum-blackwell.csv",
        ]
    )
    knn_sections = {
        "k=1024, n=4'194'304, m=4'096, r=10": load_csv_matrix(measured_times / "knn-1024-blackwell.csv"),
        "k=32, n=4'194'304, m=4'096, r=10": load_csv_matrix(measured_times / "knn-32-blackwell.csv"),
    }
    knn_sections["Combined across both k choices"] = combine_knn_sections(
        knn_sections["k=1024, n=4'194'304, m=4'096, r=10"],
        knn_sections["k=32, n=4'194'304, m=4'096, r=10"],
    )
    histogram_sections = {"default": histogram_matrix}
    gol_sections = {"default": load_csv_matrix(measured_times / "gol-blackwell.csv")}

    return {
        "knn": (
            reviews / "knn.md",
            knn_sections,
            load_fixed_indicator_matrix(results / "knn", "code.cu", knn_sections),
            {
                "Summary for k=1024 and k=32": aggregate_counts(
                    {
                        "k=1024, n=4'194'304, m=4'096, r=10": knn_sections["k=1024, n=4'194'304, m=4'096, r=10"],
                        "k=32, n=4'194'304, m=4'096, r=10": knn_sections["k=32, n=4'194'304, m=4'096, r=10"],
                    }
                ),
                "Summary for combined across both k choices": aggregate_counts(
                    {"Combined across both k choices": knn_sections["Combined across both k choices"]}
                ),
            },
            normalize_knn_row,
        ),
        "histogram": (
            reviews / "histogram.md",
            histogram_sections,
            load_fixed_indicator_matrix(results / "histogram", "code.cu", histogram_sections),
            {"default": aggregate_counts(histogram_sections)},
            normalize_histogram_row,
        ),
        "game-of-life": (
            reviews / "game-of-life.md",
            gol_sections,
            load_fixed_indicator_matrix(results / "gol", "gol.cu", gol_sections),
            {"default": aggregate_counts(gol_sections)},
            normalize_gol_row,
        ),
    }


def main() -> int:
    args = parse_args()
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    specs = review_specs(repo_root)
    selected = args.reviews or list(specs)

    all_errors: list[str] = []
    for review_name in selected:
        markdown_path, expected_sections, expected_fixed_sections, expected_summary_sections, row_normalizer = specs[review_name]
        errors = validate_review(
            markdown_path,
            expected_sections,
            row_normalizer,
            expected_fixed_sections,
            expected_summary_sections,
        )
        if errors:
            all_errors.extend(errors)
        else:
            formatted_sections = []
            for section_name, counts in expected_summary_sections.items():
                total = sum(counts.values())
                formatted = ", ".join(
                    f"{status}={counts[status]}/{total} ({round_percent(counts[status], total)}%)"
                    for status in STATUS_ORDER
                )
                formatted_sections.append(f"{section_name} [{formatted}]")
            print(f"{markdown_path}: OK {'; '.join(formatted_sections)}")

    if all_errors:
        for error in all_errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
