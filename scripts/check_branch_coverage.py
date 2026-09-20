"""Fail-closed verifier for cargo llvm-cov source branch JSON.

Run after ``cargo llvm-cov --branch --locked --all-targets --json``. This
checks actual branch outcomes, not line or combined statement coverage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import NoReturn

MINIMUM_PERCENT = 90
MAX_REPORT_BYTES = 64 * 1024 * 1024
DECLARATION_ONLY_MODULES = frozenset({"lib.rs", "ciff/mod.rs"})


class CoverageError(ValueError):
    """The coverage report cannot establish the required source-branch gate."""


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise CoverageError(f"{label} must be a nonnegative integer")
    return value


def _counts(value: object, label: str) -> tuple[int, int]:
    if not isinstance(value, dict):
        raise CoverageError(f"{label} must be an object")
    total = _nonnegative_int(value.get("count"), f"{label}.count")
    covered = _nonnegative_int(value.get("covered"), f"{label}.covered")
    missing = _nonnegative_int(value.get("notcovered"), f"{label}.notcovered")
    if covered + missing != total:
        raise CoverageError(f"{label} counts do not add up")
    return covered, total


def measure(data: object, repository: Path) -> tuple[int, int]:
    """Validate a complete report and return covered/total source branches."""

    if not isinstance(data, dict) or data.get("type") != "llvm.coverage.json.export":
        raise CoverageError("expected an llvm-cov JSON export")
    if data.get("version") != "3.1.0":
        raise CoverageError("unexpected llvm-cov JSON schema version")
    reports = data.get("data")
    if (
        not isinstance(reports, list)
        or len(reports) != 1
        or not isinstance(reports[0], dict)
    ):
        raise CoverageError("expected exactly one coverage report")
    report = reports[0]
    totals = report.get("totals")
    if not isinstance(totals, dict):
        raise CoverageError("missing coverage totals")
    covered, total = _counts(totals.get("branches"), "totals.branches")
    if total == 0:
        raise CoverageError("report contains no source branches")
    files = report.get("files")
    if not isinstance(files, list) or not files:
        raise CoverageError("report contains no source files")
    source_root = (repository / "src").resolve()
    file_names: set[Path] = set()
    file_covered = 0
    file_total = 0
    for index, file in enumerate(files):
        if not isinstance(file, dict) or not isinstance(file.get("filename"), str):
            raise CoverageError(f"files[{index}] has no filename")
        path = Path(file["filename"]).resolve()
        if (
            not path.is_relative_to(source_root)
            or path.suffix != ".rs"
            or not path.is_file()
        ):
            raise CoverageError(f"files[{index}] is not a repository Rust source file")
        if path in file_names:
            raise CoverageError(f"duplicate coverage source file: {path}")
        file_names.add(path)
        summary = file.get("summary")
        if not isinstance(summary, dict):
            raise CoverageError(f"files[{index}] has no summary")
        part_covered, part_total = _counts(
            summary.get("branches"), f"files[{index}].branches"
        )
        file_covered += part_covered
        file_total += part_total
    if (file_covered, file_total) != (covered, total):
        raise CoverageError("file and report branch totals differ")
    expected = {
        source.resolve()
        for source in source_root.rglob("*.rs")
        if source.relative_to(source_root).as_posix() not in DECLARATION_ONLY_MODULES
    }
    missing = expected - file_names
    if missing:
        names = ", ".join(
            str(path.relative_to(source_root)) for path in sorted(missing)
        )
        raise CoverageError(f"source files missing from coverage report: {names}")
    return covered, total


def _reject_nonfinite(token: str) -> NoReturn:
    raise CoverageError(f"non-finite JSON number: {token}")


def meets_gate(covered: int, total: int) -> bool:
    """Compare integer counts without rounded display percentages."""

    return total > 0 and 100 * covered >= MINIMUM_PERCENT * total


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CoverageError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("usage: check_branch_coverage.py LLVM_COV_JSON", file=sys.stderr)
        return 2
    source = Path(arguments[0])
    try:
        if source.stat().st_size > MAX_REPORT_BYTES:
            raise CoverageError("coverage report exceeds 64 MiB")
        with source.open("r", encoding="utf-8") as stream:
            data = json.load(
                stream,
                parse_constant=_reject_nonfinite,
                object_pairs_hook=_unique_object,
            )
        covered, total = measure(data, Path(__file__).resolve().parents[1])
    except (OSError, UnicodeError, json.JSONDecodeError, CoverageError) as error:
        print(f"branch coverage unavailable: {error}", file=sys.stderr)
        return 2
    percent = 100.0 * covered / total
    print(f"source branch coverage: {covered}/{total} = {percent:.2f}%")
    if not meets_gate(covered, total):
        print(f"required: {MINIMUM_PERCENT:.2f}%", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
