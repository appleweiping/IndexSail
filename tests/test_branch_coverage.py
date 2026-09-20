"""Independent malformed-report checks for the source-branch gate."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from scripts.check_branch_coverage import CoverageError, main, measure, meets_gate


class BranchCoverageGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repository = Path(self.temporary.name)
        (self.repository / "src").mkdir()
        self.source = self.repository / "src" / "lib.rs"
        self.source.write_text("pub fn sample() {}\n", encoding="utf-8")
        self.report = {
            "type": "llvm.coverage.json.export",
            "version": "3.1.0",
            "data": [
                {
                    "totals": {
                        "branches": {"count": 10, "covered": 9, "notcovered": 1}
                    },
                    "files": [
                        {
                            "filename": str(self.source),
                            "summary": {
                                "branches": {"count": 10, "covered": 9, "notcovered": 1}
                            },
                        }
                    ],
                }
            ],
        }

    def test_exact_ninety_percent_is_valid(self) -> None:
        self.assertEqual(measure(self.report, self.repository), (9, 10))
        self.assertTrue(meets_gate(9, 10))
        self.assertFalse(meets_gate(89, 100))
        self.assertFalse(meets_gate(0, 0))

    def test_no_branches_fails_closed(self) -> None:
        report = copy.deepcopy(self.report)
        summaries = (
            report["data"][0]["totals"],
            report["data"][0]["files"][0]["summary"],
        )
        for summary in summaries:
            summary["branches"] = {"count": 0, "covered": 0, "notcovered": 0}
        with self.assertRaisesRegex(CoverageError, "no source branches"):
            measure(report, self.repository)

    def test_missing_or_mismatched_totals_fail_closed(self) -> None:
        report = copy.deepcopy(self.report)
        report["data"][0]["totals"]["branches"]["covered"] = 8
        with self.assertRaises(CoverageError):
            measure(report, self.repository)
        report = copy.deepcopy(self.report)
        report["data"][0]["files"][0]["summary"]["branches"]["covered"] = 8
        report["data"][0]["files"][0]["summary"]["branches"]["notcovered"] = 2
        with self.assertRaisesRegex(CoverageError, "file and report"):
            measure(report, self.repository)

    def test_external_duplicate_and_non_rust_sources_fail_closed(self) -> None:
        report = copy.deepcopy(self.report)
        report["data"][0]["files"][0]["filename"] = str(self.repository / "else.rs")
        with self.assertRaisesRegex(CoverageError, "not a repository"):
            measure(report, self.repository)
        report = copy.deepcopy(self.report)
        report["data"][0]["files"].append(copy.deepcopy(report["data"][0]["files"][0]))
        with self.assertRaisesRegex(CoverageError, "duplicate"):
            measure(report, self.repository)
        report = copy.deepcopy(self.report)
        report["data"][0]["files"][0]["filename"] = str(
            self.repository / "src" / "lib.txt"
        )
        with self.assertRaisesRegex(CoverageError, "not a repository"):
            measure(report, self.repository)

    def test_schema_and_bool_counts_fail_closed(self) -> None:
        report = copy.deepcopy(self.report)
        report["version"] = "unexpected"
        with self.assertRaisesRegex(CoverageError, "schema"):
            measure(report, self.repository)
        report = copy.deepcopy(self.report)
        report["data"][0]["totals"]["branches"]["covered"] = True
        with self.assertRaisesRegex(CoverageError, "nonnegative"):
            measure(report, self.repository)

    def test_new_source_file_missing_from_report_fails_closed(self) -> None:
        (self.repository / "src" / "new_algorithm.rs").write_text(
            "pub fn rank() {}\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(CoverageError, "source files missing"):
            measure(self.report, self.repository)

    def test_missing_file_and_invalid_json_do_not_pass(self) -> None:
        self.assertEqual(main([str(self.repository / "absent.json")]), 2)
        invalid = self.repository / "invalid.json"
        invalid.write_text("{", encoding="utf-8")
        self.assertEqual(main([str(invalid)]), 2)
        invalid.write_text('{"type": NaN}', encoding="utf-8")
        self.assertEqual(main([str(invalid)]), 2)
        invalid.write_text('{"type": "a", "type": "b"}', encoding="utf-8")
        self.assertEqual(main([str(invalid)]), 2)


if __name__ == "__main__":
    unittest.main()
