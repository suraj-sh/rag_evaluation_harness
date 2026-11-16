import argparse
import json
import statistics
import time
from pathlib import Path
from sentence_transformers import CrossEncoder
from app.rag.hybrid_retrieval import retrieve_hybrid
from evaluation.candidate_pool_experiment import deduplicate_by_page, get_page_numbers, rerank_documents


DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def summarize_latency(values):
    """Return descriptive statistics for latency measurements."""
    if not values:
        return {}

    return {
        "mean_ms": round(statistics.mean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "stdev_ms": round(statistics.stdev(values), 3)
        if len(values) > 1 else 0.0,
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Repeat latency comparison for Top 10 vs Full candidate pools."
    )

    parser.add_argument(
        "--questions",
        default="evaluation/questions.json",
    )

    parser.add_argument(
        "--output",
        default="experiments/candidate_pool_latency_repeat.json",
    )

    parser.add_argument(
        "--fetch-k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Number of additional benchmark runs.",
    )

    args = parser.parse_args()

    if args.fetch_k < 1:
        parser.error("--fetch-k must be at least 1.")

    if args.repeats < 1:
        parser.error("--repeats must be at least 1.")

    questions_path = Path(args.questions)

    if not questions_path.exists():
        raise FileNotFoundError(
            f"Questions file not found: {questions_path}"
        )

    questions = json.loads(
        questions_path.read_text(encoding="utf-8-sig")
    )

    if not questions:
        raise ValueError("The questions file is empty.")

    # --------------------------------------------------
    # 1. Load CrossEncoder
    # --------------------------------------------------

    print(f"Loading reranker: {args.model}")
    model = CrossEncoder(args.model)

    # --------------------------------------------------
    # 2. Warm up CrossEncoder
    # --------------------------------------------------

    first_question = (
        questions[0].get("retrieval_query")
        or questions[0]["question"]
    )

    print("\nRetrieving documents for warm-up...")

    warmup_components = retrieve_hybrid(
        first_question,
        top_k=args.fetch_k,
        fetch_k=args.fetch_k,
        return_components=True,
    )

    warmup_candidates = deduplicate_by_page(
        warmup_components["hybrid"]
    )

    warmup_sizes = sorted({
        min(10, len(warmup_candidates)),
        len(warmup_candidates),
    })

    print("Warming up CrossEncoder...")

    for size in warmup_sizes:
        if size == 0:
            continue

        rerank_documents(
            model,
            first_question,
            warmup_candidates[:size],
        )

        print(f"Warm-up completed: {size} candidates")

    # --------------------------------------------------
    # 3. Initialize repeat containers
    # --------------------------------------------------

    repeat_results = []

    # --------------------------------------------------
    # 4. Run additional benchmark repetitions
    # --------------------------------------------------

    for repeat_number in range(1, args.repeats + 1):

        print("\n" + "=" * 75)
        print(f"LATENCY REPEAT {repeat_number}/{args.repeats}")
        print("=" * 75)

        configurations = {
            "top_10": [],
            "full": [],
        }

        retrieval_latencies = []

        repeat_start = time.perf_counter()

        for index, item in enumerate(questions):

            question = (
                item.get("retrieval_query")
                or item["question"]
            )

            print(
                f"\n[{index + 1}/{len(questions)}] "
                f"Question ID: {item['id']}"
            )

            # Retrieve once; both configurations use
            # exactly the same candidate documents.
            components = retrieve_hybrid(
                question,
                top_k=args.fetch_k,
                fetch_k=args.fetch_k,
                return_components=True,
            )

            retrieval_ms = (
                components["timings"]["total_retrieval_ms"]
            )

            retrieval_latencies.append(retrieval_ms)

            candidates = deduplicate_by_page(
                components["hybrid"]
            )

            candidate_count = len(candidates)

            # Top 10 vs the complete available pool.
            pools = {
                "top_10": candidates[:10],
                "full": candidates,
            }

            # Alternate order across questions and repeats.
            # This avoids always running one configuration first.
            execution_order = (
                ["top_10", "full"]
                if (index + repeat_number) % 2 == 0
                else ["full", "top_10"]
            )

            for name in execution_order:

                selected_candidates = pools[name]

                rerank_start = time.perf_counter()

                rerank_documents(
                    model,
                    question,
                    selected_candidates,
                )

                reranker_ms = (
                    time.perf_counter() - rerank_start
                ) * 1000

                total_ms = retrieval_ms + reranker_ms

                result = {
                    "id": item["id"],
                    "candidate_count": len(selected_candidates),
                    "available_candidate_count": candidate_count,
                    "candidate_pages": get_page_numbers(
                        selected_candidates
                    ),
                    "retrieval_latency_ms": round(
                        retrieval_ms, 3
                    ),
                    "reranker_latency_ms": round(
                        reranker_ms, 3
                    ),
                    "total_latency_ms": round(
                        total_ms, 3
                    ),
                    "execution_order": execution_order,
                }

                configurations[name].append(result)

                print(
                    f"{name}: "
                    f"candidates={len(selected_candidates)}, "
                    f"reranker={reranker_ms:.2f} ms, "
                    f"total={total_ms:.2f} ms"
                )

        repeat_seconds = (
            time.perf_counter() - repeat_start
        )

        # --------------------------------------------------
        # 5. Summarize this repetition
        # --------------------------------------------------

        repeat_summary = {}

        for name, results in configurations.items():

            reranker_values = [
                result["reranker_latency_ms"]
                for result in results
            ]

            total_values = [
                result["total_latency_ms"]
                for result in results
            ]

            repeat_summary[name] = {
                "reranker_latency": summarize_latency(
                    reranker_values
                ),
                "total_latency": summarize_latency(
                    total_values
                ),
                "results": results,
            }

        top10_mean = repeat_summary["top_10"][
            "reranker_latency"
        ]["mean_ms"]

        full_mean = repeat_summary["full"][
            "reranker_latency"
        ]["mean_ms"]

        reduction = (
            (1 - top10_mean / full_mean) * 100
            if full_mean else 0.0
        )

        repeat_output = {
            "repeat": repeat_number,
            "questions_evaluated": len(questions),
            "duration_seconds": round(repeat_seconds, 3),
            "retrieval_latency": summarize_latency(
                retrieval_latencies
            ),
            "latency_reduction_percent": round(
                reduction, 2
            ),
            "configurations": repeat_summary,
        }

        repeat_results.append(repeat_output)

        print("\nREPEAT SUMMARY")
        print("-" * 55)

        for name in ("top_10", "full"):
            data = repeat_summary[name]

            print(f"\n{name.upper()}")
            print(
                "Reranker:",
                data["reranker_latency"],
            )
            print(
                "Total:",
                data["total_latency"],
            )

        print(
            f"\nReranker latency reduction: {reduction:.2f}%"
        )

    # --------------------------------------------------
    # 6. Save results
    # --------------------------------------------------

    output = {
        "experiment": "Repeated Top 10 vs Full candidate-pool latency comparison",
        "questions_evaluated": len(questions),
        "repeats": args.repeats,
        "retrieval_configuration": {
            "method": "Hybrid RRF",
            "fetch_k": args.fetch_k,
            "candidate_aggregation": "Page-level",
        },
        "reranker_model": args.model,
        "warmup_enabled": True,
        "latency_methodology": {
            "retrieval": (
                "Hybrid retrieval is executed once per question "
                "in each repeat and shared by both configurations."
            ),
            "reranker": (
                "Includes candidate pair construction, "
                "CrossEncoder prediction and sorting."
            ),
            "total": (
                "Measured retrieval latency plus reranker latency."
            ),
            "execution_order": (
                "Top 10 and Full execution order alternates "
                "across questions and repetitions."
            ),
            "warmup": (
                "CrossEncoder warm-up is excluded from measurements."
            ),
        },
        "results": repeat_results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 75)
    print("REPEATED LATENCY EXPERIMENT COMPLETED")
    print("=" * 75)
    print(f"Repeats: {args.repeats}")
    print(f"Questions per repeat: {len(questions)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()