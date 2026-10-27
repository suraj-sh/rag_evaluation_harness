import json
import tempfile
import unittest
from pathlib import Path
from evaluation.score_answers import (
    VALID_REVIEW_STATUSES,
    find_dataset_mismatches,
    find_review_findings,
    load_review,
)


class TestScoreAnswers(unittest.TestCase):
    def test_review_statuses_are_supported(self):
        self.assertEqual(
            VALID_REVIEW_STATUSES,
            {"pass", "partial", "fail", "pending"},
        )

    def test_missing_review_status_defaults_to_pending(self):
        findings = find_review_findings(
            ["Required criterion"],
            {},
        )

        self.assertEqual(
            findings,
            [("pending", "Required criterion")],
        )

    def test_failed_and_partial_points_are_reported(self):
        findings = find_review_findings(
            ["Passed", "Partial", "Failed"],
            {
                "Passed": "pass",
                "Partial": "partial",
                "Failed": "fail",
            },
        )

        self.assertEqual(
            findings,
            [
                ("partial", "Partial"),
                ("fail", "Failed"),
            ],
        )

    def test_unknown_status_defaults_to_pending(self):
        findings = find_review_findings(
            ["Criterion"],
            {"Criterion": "unknown"},
        )

        self.assertEqual(
            findings,
            [("pending", "Criterion")],
        )

    def test_all_passed_points_produce_no_findings(self):
        findings = find_review_findings(
            ["First", "Second"],
            {"First": "pass", "Second": "pass"},
        )

        self.assertEqual(findings, [])

    def test_dataset_mismatch_detects_changed_expected_points(self):
        report = {
            "results": [
                {
                    "id": "A01",
                    "expected_answer_points": ["Old criterion"],
                }
            ]
        }
        current_points = {"A01": ["New criterion"]}

        mismatches = find_dataset_mismatches(report, current_points)

        self.assertEqual(
            mismatches,
            [
                (
                    "A01",
                    "Expected answer points differ from the current dataset",
                )
            ],
        )

    def test_dataset_mismatch_detects_missing_question_id(self):
        report = {
            "results": [
                {
                    "id": "A99",
                    "expected_answer_points": [],
                }
            ]
        }

        mismatches = find_dataset_mismatches(report, {"A01": []})

        self.assertEqual(
            mismatches,
            [
                (
                    "A99",
                    "Question ID is missing from the current dataset",
                )
            ],
        )

    def test_matching_dataset_produces_no_mismatches(self):
        report = {
            "results": [
                {
                    "id": "A01",
                    "expected_answer_points": ["Criterion one"],
                }
            ]
        }

        self.assertEqual(
            find_dataset_mismatches(
                report,
                {"A01": ["Criterion one"]},
            ),
            [],
        )

    def write_review(self, directory, review, *, bom=False):
        path = Path(directory) / "review.json"
        content = json.dumps(review, ensure_ascii=False)
        encoding = "utf-8-sig" if bom else "utf-8"
        path.write_text(content, encoding=encoding)
        return path

    def valid_review(self):
        return {
            "review_method": "manual",
            "results": [
                {
                    "id": "A01",
                    "answer_points": [
                        {
                            "point": "Criterion one",
                            "status": "pass",
                        }
                    ],
                }
            ],
        }

    def test_load_review_reads_utf8_bom_and_indexes_by_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_review(
                directory,
                self.valid_review(),
                bom=True,
            )

            review, review_by_id = load_review(path)

        self.assertEqual(review["review_method"], "manual")
        self.assertIn("A01", review_by_id)
        self.assertEqual(
            review_by_id["A01"]["answer_points"][0]["status"],
            "pass",
        )

    def test_load_review_rejects_non_manual_review(self):
        review_data = self.valid_review()
        review_data["review_method"] = "automatic"

        with tempfile.TemporaryDirectory() as directory:
            path = self.write_review(directory, review_data)

            with self.assertRaisesRegex(
                ValueError,
                "Review method must be 'manual'",
            ):
                load_review(path)

    def test_load_review_rejects_duplicate_question_ids(self):
        review_data = self.valid_review()
        review_data["results"].append(
            {
                "id": "A01",
                "answer_points": [],
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            path = self.write_review(directory, review_data)

            with self.assertRaisesRegex(
                ValueError,
                "Duplicate review question ID: A01",
            ):
                load_review(path)

    def test_load_review_rejects_invalid_status(self):
        review_data = self.valid_review()
        review_data["results"][0]["answer_points"][0]["status"] = "unknown"

        with tempfile.TemporaryDirectory() as directory:
            path = self.write_review(directory, review_data)

            with self.assertRaisesRegex(
                ValueError,
                "Invalid review status for A01: unknown",
            ):
                load_review(path)


if __name__ == "__main__":
    unittest.main()