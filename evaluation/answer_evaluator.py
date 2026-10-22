import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from app.rag.retrieval import get_retriever
from app.rag.generation import generate_answer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "answer_questions.json"
RESULTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--retry-failed",
            type=str,
            help="Path to a previous evaluation report JSON",
        )
        parser.add_argument(
            "--question-id",
            type=str,
            help="Evaluate only one question, e.g. A01",
        )
        args = parser.parse_args()

        previous_results = {}
        
        if args.retry_failed:
            with open(args.retry_failed, "r", encoding="utf-8") as file:
                previous_report = json.load(file)

            previous_results = {
                result["id"]: result
                for result in previous_report["results"]
            }

            dataset["questions"] = [
                item for item in dataset["questions"]
                if (
                    item["id"] not in previous_results
                    or previous_results[item["id"]]["error"] is not None
                )
            ]
        
        if args.question_id:
            dataset["questions"] = [
                item
                for item in dataset["questions"]
                if item["id"] == args.question_id
            ]

            if not dataset["questions"]:
                raise ValueError(
                    f"Question ID not found: {args.question_id}"
                )

    retriever = get_retriever()
    results = []

    for index, item in enumerate(dataset["questions"], start=1):
        question = item["question"]
        print(f"\n[{index}/{len(dataset['questions'])}] {question}")

        try:
            start = time.perf_counter()
            documents = retriever.invoke(question)
            retrieval_seconds = time.perf_counter() - start

            start = time.perf_counter()
            answer = generate_answer(question, documents)
            generation_seconds = time.perf_counter() - start

            sources = [
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get(
                        "pdf_page",
                        doc.metadata.get("page", "unknown"),
                    ),
                }
                for doc in documents
            ]

            result = {
                "id": item["id"],
                "type": item["type"],
                "question": question,
                "expected_answer_points": item["expected_answer_points"],
                "expected_citation_pages_pdf": item["citation_pages_pdf"],
                "abstention_expected": item["abstention_expected"],
                "answer": answer,
                "retrieved_sources": sources,
                "retrieval_latency_seconds": round(retrieval_seconds, 3),
                "generation_latency_seconds": round(generation_seconds, 3),
                "error": None,
            }

            print(f"Answer: {answer}")
            print(f"Retrieved pages: {[s['page'] for s in sources]}")
            print(
                f"Latency: retrieval={retrieval_seconds:.2f}s, "
                f"generation={generation_seconds:.2f}s"
            )

        except Exception as exc:
            result = {
                "id": item["id"],
                "type": item["type"],
                "question": question,
                "expected_answer_points": item["expected_answer_points"],
                "expected_citation_pages_pdf": item["citation_pages_pdf"],
                "abstention_expected": item["abstention_expected"],
                "answer": None,
                "retrieved_sources": [],
                "retrieval_latency_seconds": None,
                "generation_latency_seconds": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"ERROR: {result['error']}")

        results.append(result)

        if args.retry_failed:
            merged_results = previous_results.copy()

            for result in results:
                merged_results[result["id"]] = result

            with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
                all_questions = json.load(file)["questions"]

            results = [
                merged_results[item["id"]]
                for item in all_questions
                if item["id"] in merged_results
            ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"answer_evaluation_{timestamp}.json"

    report = {
        "dataset": dataset["dataset_name"],
        "model": "gemini-3.1-flash-lite",
        "question_count": len(results),
        "completed_count": sum(r["error"] is None for r in results),
        "failed_count": sum(r["error"] is not None for r in results),
        "results": results,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"\nEvaluation report saved to: {output_path}")


if __name__ == "__main__":
    main()