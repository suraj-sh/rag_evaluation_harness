import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from evaluation import compare_answer_reports
from evaluation.compare_answer_reports import (
    compare_numeric,
    index_results,
    retrieved_pages,
    citation_pages,
)


class CompareAnswerReportsTests(unittest.TestCase):
    def test_identical_question_data_has_same_comparison_fields(self):
        result = {
            "id": "A01",
            "answer": "Supported answer [Page 26].",
            "retrieved_sources": [{"page": 25}, {"page": 26}],
        }

        self.assertEqual(result["answer"], result["answer"])
        self.assertEqual(retrieved_pages(result), [25, 26])
        self.assertEqual(citation_pages(result["answer"]), [26])

    def test_answer_citation_changes_are_detectable(self):
        baseline = "Answer supported by [Page 26]."
        candidate = "Answer supported by [Page 27]."

        self.assertNotEqual(baseline, candidate)
        self.assertEqual(citation_pages(baseline), [26])
        self.assertEqual(citation_pages(candidate), [27])

    def test_retrieved_page_order_is_preserved(self):
        first = {"retrieved_sources": [{"page": 25}, {"page": 26}]}
        second = {"retrieved_sources": [{"page": 26}, {"page": 25}]}

        self.assertEqual(retrieved_pages(first), [25, 26])
        self.assertNotEqual(
            retrieved_pages(first), retrieved_pages(second)
        )

    def test_missing_and_additional_question_ids(self):
        baseline = {"A01": {}, "A02": {}}
        candidate = {"A01": {}, "A03": {}}

        self.assertEqual(sorted(set(baseline) - set(candidate)), ["A02"])
        self.assertEqual(sorted(set(candidate) - set(baseline)), ["A03"])

    def test_missing_question_id_is_rejected(self):
        with self.assertRaises(ValueError):
            index_results({"results": [{"answer": "No ID"}]}, "Test report")

    def test_duplicate_question_id_is_rejected(self):
        with self.assertRaises(ValueError):
            index_results(
                {"results": [{"id": "A01"}, {"id": "A01"}]},
                "Test report",
            )

    def test_latency_delta(self):
        self.assertAlmostEqual(compare_numeric(0.590, 22.592), 22.002)

    def test_missing_or_non_numeric_latency(self):
        self.assertIsNone(compare_numeric(None, 2.0))
        self.assertIsNone(compare_numeric("unknown", 2.0))

    def test_main_detects_changed_answer_and_citations(self):
        baseline = {
            "dataset": "test",
            "model": "test-model",
            "question_count": 1,
            "completed_count": 1,
            "failed_count": 0,
            "results": [{
                "id": "A01",
                "question": "Test question?",
                "answer": "Baseline answer [Page 2].",
                "retrieved_sources": [{"page": 1}],
                "retrieval_latency_seconds": 0.5,
                "generation_latency_seconds": 1.0,
                "error": None,
            }],
        }

        candidate = {
            **baseline,
            "results": [{
                **baseline["results"][0],
                "answer": "Updated answer [Page 3].",
            }],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_path = Path(temp_dir) / "baseline.json"
            candidate_path = Path(temp_dir) / "candidate.json"

            baseline_path.write_text(
                json.dumps(baseline), encoding="utf-8"
            )
            candidate_path.write_text(
                json.dumps(candidate), encoding="utf-8"
            )

            with (
                patch(
                    "sys.argv",
                    [
                        "compare_answer_reports",
                        str(baseline_path),
                        str(candidate_path),
                    ],
                ),
                patch("sys.stdout") as mock_stdout,
            ):
                exit_code = compare_answer_reports.main()

            self.assertEqual(exit_code, 0)
            output = "".join(
                str(call.args[0])
                for call in mock_stdout.write.call_args_list
                if call.args
            )
            self.assertIn("[A01] CHANGED", output)
            self.assertIn("Answer text", output)
            self.assertIn("Citation sequence", output)
            self.assertIn("Questions with content/source/status changes: 1", output)


if __name__ == "__main__":
    unittest.main()