import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "experiments"
REPORT_PATTERN = "answer_evaluation_*.json"
REVIEW_PATH = PROJECT_ROOT / "evaluation" / "answer_review.json"
DEFAULT_QUESTIONS_PATH = (
    PROJECT_ROOT / "evaluation" / "answer_questions_hybrid.json"
)
VALID_REVIEW_STATUSES = {"pass", "partial", "fail", "pending"}


def extract_cited_pages(answer):
    """Extract 1-based PDF page numbers from citations, including ranges."""
    pages = set()

    for match in re.finditer(
        r"\[Pages?\s+([\d,\s-]+)\]",
        answer,
        flags=re.IGNORECASE,
    ):
        content = match.group(1)

        for part in content.split(","):
            part = part.strip()

            range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
            if range_match:
                start, end = map(int, range_match.groups())
                if start > end:
                    continue
                pages.update(range(start, end + 1))
                continue

            page_match = re.fullmatch(r"\d+", part)
            if page_match:
                pages.add(int(part))

    return pages


def load_report(report_path=None):
    if report_path:
        path = Path(report_path)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if not path.is_file():
            raise FileNotFoundError(
                f"Evaluation report not found: {path}"
            )
    else:
        report_files = sorted(
            REPORT_DIR.glob(REPORT_PATTERN),
            key=lambda path: path.stat().st_mtime,
        )

        if not report_files:
            raise FileNotFoundError(
                f"No answer evaluation reports found in {REPORT_DIR}"
            )

        path = report_files[-1]

    with path.open("r", encoding="utf-8") as file:
        return path, json.load(file)


def load_expected_citation_pages():
    with DEFAULT_QUESTIONS_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    return {
        item["id"]: set(item.get("citation_pages_pdf", []))
        for item in dataset.get("questions", [])
    }


def load_review():
    with REVIEW_PATH.open("r", encoding="utf-8") as file:
        review = json.load(file)

    if review.get("review_method") != "manual":
        raise ValueError("Review method must be 'manual'.")

    review_by_id = {}

    for result in review.get("results", []):
        question_id = result["id"]

        if question_id in review_by_id:
            raise ValueError(f"Duplicate review question ID: {question_id}")

        for point in result.get("answer_points", []):
            if point.get("status") not in VALID_REVIEW_STATUSES:
                raise ValueError(
                    f"Invalid review status for {question_id}: "
                    f"{point.get('status')}"
                )

        review_by_id[question_id] = result

    return review, review_by_id


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to a specific answer evaluation report JSON",
    )
    args = arg_parser.parse_args()

    report_path, report = load_report(args.report)
    expected_pages_by_id = load_expected_citation_pages()
    review, review_by_id = load_review()

    print(f"Report: {report_path.name}")
    print(f"Questions: {report['question_count']}")
    print(f"Completed: {report['completed_count']}")
    print(f"Failed: {report['failed_count']}\n")

    citation_checks = 0
    citation_matches = 0
    expected_page_coverage_sum = 0.0
    citation_precision_sum = 0.0
    exact_page_matches = 0
    abstention_checks = 0
    abstention_signals = 0

    review_counts = {status: 0 for status in VALID_REVIEW_STATUSES}
    reviewed_questions = 0
    questions_with_points = 0

    for result in report["results"]:
        question_id = result["id"]
        answer = result.get("answer")
        expected_pages = expected_pages_by_id.get(
            result["id"],
            set(result.get("expected_citation_pages_pdf", [])),
        )

        print(f"--- {question_id}: {result['type']} ---")

        if result.get("error") or not answer:
            print("Status: API/evaluation failure")
            print(result.get("error", "No answer recorded"), "\n")
            continue

        print("Answer:", answer.replace("\n", " "))

        review_result = review_by_id.get(question_id)
        point_statuses = {}

        if review_result:
            for point in review_result.get("answer_points", []):
                point_statuses[point["point"]] = point["status"]

        expected_points = result.get("expected_answer_points", [])

        if expected_points:
            questions_with_points += 1
            all_pass = True

            print("Manual answer-point review:")

            for point in expected_points:
                status = point_statuses.get(point, "pending")

                if status not in VALID_REVIEW_STATUSES:
                    status = "pending"

                review_counts[status] += 1

                if status != "pass":
                    all_pass = False

                print(f"  - [{status.upper()}] {point}")

            if all_pass:
                reviewed_questions += 1

            print(
                "Question review:",
                "ALL POINTS PASS" if all_pass
                else "INCOMPLETE OR NEEDS REVIEW",
            )

        cited_pages = extract_cited_pages(answer)

        if expected_pages:
            citation_checks += 1

            # Answer citations are 1-based; expected metadata pages are 0-based.
            cited_pages_pdf = {page - 1 for page in cited_pages if page >= 1}
            overlap = cited_pages_pdf & expected_pages

            # Retain the original metric for comparison.
            matched = bool(overlap)
            if matched:
                citation_matches += 1

            # Page-level proxies, not claim-level grounding verification.
            expected_coverage = len(overlap) / len(expected_pages)
            citation_precision = (
                len(overlap) / len(cited_pages_pdf)
                if cited_pages_pdf else 0.0
            )
            exact_match = cited_pages_pdf == expected_pages

            expected_page_coverage_sum += expected_coverage
            citation_precision_sum += citation_precision

            if exact_match:
                exact_page_matches += 1

            print("Expected PDF pages:", sorted(expected_pages))
            print("Cited pages:", sorted(cited_pages))
            print(
                "Citation overlap heuristic:",
                "YES" if matched else "NO",
            )
            print(f"Expected-page coverage: {expected_coverage:.1%}")
            print(f"Citation precision: {citation_precision:.1%}")
            print("Exact page match:", "YES" if exact_match else "NO")

        if result.get("abstention_expected"):
            abstention_checks += 1

            abstained = bool(
                re.search(
                    r"\b(cannot answer|do not contain|does not contain|"
                    r"do not state|does not state|not provided|no information|"
                    r"cannot determine|not enough information|no exact|"
                    r"no guaranteed|does not specify|not specified|"
                    r"cannot be determined|unable to determine)\b",
                    answer,
                    flags=re.IGNORECASE,
                )
            )

            if abstained:
                abstention_signals += 1

            print(
                "Abstention phrase signal:",
                "FOUND (manual review required)"
                if abstained
                else "NOT FOUND (manual review required)",
            )

        print()

    total_points = sum(review_counts.values())
    completed_points = (
        review_counts["pass"]
        + review_counts["partial"]
        + review_counts["fail"]
    )

    print("=== Evaluation summary ===")
    print(
        f"Citation overlap heuristic: {citation_matches}/{citation_checks}"
    )

    if citation_checks:
        print(
            "Mean expected-page coverage:",
            f"{expected_page_coverage_sum / citation_checks:.1%}",
        )
        print(
            "Mean citation precision:",
            f"{citation_precision_sum / citation_checks:.1%}",
        )
        print(
            f"Exact page-set matches: {exact_page_matches}/{citation_checks}"
        )

    print(
        f"Abstention phrase signals: {abstention_signals}/{abstention_checks}"
    )
    print(f"Manual review points: {completed_points}/{total_points} completed")
    print(
        "Manual statuses:",
        ", ".join(
            f"{status}={review_counts[status]}"
            for status in ("pass", "partial", "fail", "pending")
        ),
    )
    print(
        "Questions with all points marked pass:",
        f"{reviewed_questions}/{questions_with_points}",
    )
    print(
        "Note: Citation overlap and abstention phrase checks are heuristics, "
        "not correctness or groundedness scores."
    )


if __name__ == "__main__":
    main()