import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "answer_evaluation_20261022_154244.json"
)


def extract_cited_pages(answer):
    """Extract page numbers cited as [Page N] or [Pages N, M]."""
    pages = set()

    for match in re.finditer(
        r"\[Pages?\s+([\d,\s]+)\]",
        answer,
        flags=re.IGNORECASE,
    ):
        pages.update(
            int(number)
            for number in re.findall(r"\d+", match.group(1))
        )

    return pages


def main():
    with REPORT_PATH.open("r", encoding="utf-8") as file:
        report = json.load(file)

    print(f"Report: {REPORT_PATH.name}")
    print(f"Questions: {report['question_count']}")
    print(f"Completed: {report['completed_count']}")
    print(f"Failed: {report['failed_count']}\n")

    citation_checks = 0
    citation_matches = 0
    abstention_checks = 0
    abstention_passes = 0

    for result in report["results"]:
        question_id = result["id"]
        answer = result.get("answer")
        expected_pages = set(
            result.get("expected_citation_pages_pdf", [])
        )

        print(f"--- {question_id}: {result['type']} ---")

        if result.get("error") or not answer:
            print("Status: API/evaluation failure")
            print(result.get("error", "No answer recorded"), "\n")
            continue

        print("Answer:", answer.replace("\n", " "))
        print("Expected answer points:")
        for point in result.get("expected_answer_points", []):
            print(f"  - [ ] {point}")

        cited_pages = extract_cited_pages(answer)

        if expected_pages:
            citation_checks += 1

            # Citations must match one-based physical PDF page numbers exactly.
            matched = bool(cited_pages & expected_pages)

            if matched:
                citation_matches += 1

            print("Expected PDF pages:", sorted(expected_pages))
            print("Cited pages:", sorted(cited_pages))
            print("Citation overlap:", "YES" if matched else "NO")

        if result.get("abstention_expected"):
            abstention_checks += 1

            # This is a heuristic flag for manual review,
            # not an automated correctness judgment.
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
                abstention_passes += 1

            print(
                "Abstention signal:",
                "FOUND (review answer)" if abstained
                else "NOT FOUND (review answer)",
            )

        print()

    print("=== Preliminary summary ===")
    print(
        f"Citation overlap: {citation_matches}/{citation_checks} "
        "(heuristic; requires manual verification)"
    )
    print(
        f"Abstention signals: {abstention_passes}/{abstention_checks} "
        "(heuristic; not a correctness score)"
    )


if __name__ == "__main__":
    main()