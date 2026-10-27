import argparse
import json
import sys
from pathlib import Path
from statistics import mean


def load_report(path: Path) -> dict:
    """Load and validate an answer evaluation report."""
    try:
        with path.open("r", encoding="utf-8") as file:
            report = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc

    if not isinstance(report, dict):
        raise ValueError(f"{path} must contain a JSON object.")

    if not isinstance(report.get("results"), list):
        raise ValueError(f"{path} must contain a 'results' list.")

    return report


def index_results(report: dict, label: str) -> dict:
    """Index questions by ID and reject missing or duplicate IDs."""
    indexed = {}

    for result in report["results"]:
        if not isinstance(result, dict):
            raise ValueError(f"{label}: each result must be a JSON object.")

        question_id = result.get("id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"{label}: a result is missing a valid question ID.")

        if question_id in indexed:
            raise ValueError(f"{label}: duplicate question ID {question_id!r}.")

        indexed[question_id] = result

    return indexed


def retrieved_pages(result: dict) -> list:
    """Return retrieved PDF page indices in their original ranking order."""
    sources = result.get("retrieved_sources") or []
    pages = []

    if not isinstance(sources, list):
        return [f"<invalid retrieved_sources: {sources!r}>"]

    for source in sources:
        if isinstance(source, dict):
            pages.append(source.get("page"))
        else:
            pages.append(f"<invalid source: {source!r}>")

    return pages


def citation_pages(answer) -> list:
    """Extract one-based [Page N] citations from answer text."""
    import re

    if not isinstance(answer, str):
        return []

    return [int(number) for number in re.findall(
        r"\[Page\s+(\d+)\]", answer, flags=re.IGNORECASE
    )]


def format_value(value) -> str:
    if value is None:
        return "None"
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def compare_numeric(before, after):
    """Return a numeric delta when both values are valid numbers."""
    if (
        isinstance(before, (int, float))
        and not isinstance(before, bool)
        and isinstance(after, (int, float))
        and not isinstance(after, bool)
    ):
        return after - before
    return None


def summarize(report: dict, results: dict) -> dict:
    retrieval_times = [
        r["retrieval_latency_seconds"]
        for r in results.values()
        if isinstance(r.get("retrieval_latency_seconds"), (int, float))
        and not isinstance(r.get("retrieval_latency_seconds"), bool)
    ]
    generation_times = [
        r["generation_latency_seconds"]
        for r in results.values()
        if isinstance(r.get("generation_latency_seconds"), (int, float))
        and not isinstance(r.get("generation_latency_seconds"), bool)
    ]

    return {
        "dataset": report.get("dataset"),
        "model": report.get("model"),
        "question_count": report.get("question_count"),
        "completed_count": report.get("completed_count"),
        "failed_count": report.get("failed_count"),
        "actual_result_count": len(results),
        "mean_retrieval_latency": (
            mean(retrieval_times) if retrieval_times else None
        ),
        "mean_generation_latency": (
            mean(generation_times) if generation_times else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare two RAG answer-evaluation JSON reports."
    )
    parser.add_argument("baseline", type=Path, help="Baseline JSON report")
    parser.add_argument("candidate", type=Path, help="New JSON report")
    parser.add_argument(
        "--show-unchanged",
        action="store_true",
        help="Show questions that did not change too.",
    )
    parser.add_argument(
        "--answers",
        action="store_true",
        help="Print full baseline and candidate answers for changed questions.",
    )
    args = parser.parse_args()

    try:
        baseline_report = load_report(args.baseline)
        candidate_report = load_report(args.candidate)
        baseline = index_results(baseline_report, "Baseline")
        candidate = index_results(candidate_report, "Candidate")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("=" * 78)
    print("RAG ANSWER EVALUATION REPORT COMPARISON")
    print("=" * 78)
    print(f"Baseline : {args.baseline}")
    print(f"Candidate: {args.candidate}")

    if baseline_report.get("dataset") != candidate_report.get("dataset"):
        print("\nWARNING: Dataset names differ.")
    if baseline_report.get("model") != candidate_report.get("model"):
        print("WARNING: Model names differ.")

    baseline_summary = summarize(baseline_report, baseline)
    candidate_summary = summarize(candidate_report, candidate)

    print("\nSUMMARY")
    for key in (
        "dataset",
        "model",
        "question_count",
        "completed_count",
        "failed_count",
        "actual_result_count",
        "mean_retrieval_latency",
        "mean_generation_latency",
    ):
        old = baseline_summary[key]
        new = candidate_summary[key]
        if key.endswith("_latency"):
            old_text = f"{old:.3f}s" if old is not None else "N/A"
            new_text = f"{new:.3f}s" if new is not None else "N/A"
            delta = compare_numeric(old, new)
            delta_text = f"{delta:+.3f}s" if delta is not None else "N/A"
            print(f"  {key}: {old_text} -> {new_text} (delta {delta_text})")
        else:
            print(f"  {key}: {old!r} -> {new!r}")

    baseline_ids = set(baseline)
    candidate_ids = set(candidate)
    only_baseline = sorted(baseline_ids - candidate_ids)
    only_candidate = sorted(candidate_ids - baseline_ids)
    common_ids = sorted(baseline_ids & candidate_ids)

    print("\nQUESTION COVERAGE")
    print(f"  In both reports: {len(common_ids)}")
    print(f"  Only in baseline: {only_baseline or 'None'}")
    print(f"  Only in candidate: {only_candidate or 'None'}")

    changed_count = 0
    latency_differences = []

    for question_id in common_ids:
        old = baseline[question_id]
        new = candidate[question_id]

        changes = []
        fields = (
            ("question", "Question text"),
            ("expected_answer_points", "Expected answer points"),
            ("expected_citation_pages_pdf", "Expected citation pages"),
            ("abstention_expected", "Abstention expectation"),
            ("answer", "Answer text"),
            ("error", "Error/status"),
        )

        for field, label in fields:
            if old.get(field) != new.get(field):
                changes.append((label, old.get(field), new.get(field)))

        old_sources = old.get("retrieved_sources")
        new_sources = new.get("retrieved_sources")
        if old_sources != new_sources:
            changes.append((
                "Retrieved sources (including order)",
                old_sources,
                new_sources,
            ))

        old_citations = citation_pages(old.get("answer"))
        new_citations = citation_pages(new.get("answer"))
        if old_citations != new_citations:
            changes.append((
                "Citation sequence [Page N]",
                old_citations,
                new_citations,
            ))

        for field, label in (
            ("retrieval_latency_seconds", "Retrieval latency"),
            ("generation_latency_seconds", "Generation latency"),
        ):
            before = old.get(field)
            after = new.get(field)
            delta = compare_numeric(before, after)
            if delta is not None and delta != 0:
                latency_differences.append((question_id, label, before, after, delta))

        if changes:
            changed_count += 1
        elif args.show_unchanged:
            print(f"\n[{question_id}] UNCHANGED")

        if changes:
            print(f"\n[{question_id}] CHANGED ({len(changes)} difference(s))")
            for label, before, after in changes:
                print(f"  - {label}")
                if label == "Answer text" and not args.answers:
                    print(f"      baseline: {str(before)[:180]!r}")
                    print(f"      candidate: {str(after)[:180]!r}")
                    print("      (Use --answers to print full answers.)")
                else:
                    print(f"      baseline: {format_value(before)[:500]}")
                    print(f"      candidate: {format_value(after)[:500]}")

        if args.answers and any(label == "Answer text" for label, _, _ in changes):
            print("\n  FULL BASELINE ANSWER")
            print(old.get("answer"))
            print("\n  FULL CANDIDATE ANSWER")
            print(new.get("answer"))

    print("\nLATENCY CHANGES")
    if latency_differences:
        for question_id, label, before, after, delta in latency_differences:
            print(
                f"  {question_id} {label}: {before:.3f}s -> {after:.3f}s "
                f"({delta:+.3f}s)"
            )
    else:
        print("  No numeric latency differences.")

    print("\nFINAL COUNTS")
    print(f"  Questions with content/source/status changes: {changed_count}")
    print(f"  Questions compared: {len(common_ids)}")
    print("\nNote: This script compares report data; it does not judge answer correctness.")
    print("Retrieved page indices are zero-based; [Page N] answer citations are one-based.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
