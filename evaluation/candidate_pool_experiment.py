import argparse
import json
import statistics
import time
from pathlib import Path
from sentence_transformers import CrossEncoder
from app.rag.hybrid_retrieval import retrieve_hybrid
from evaluation.metrics import calculate_ranking_metrics


DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# None represents the complete available candidate pool.
POOL_CONFIGURATIONS = (10, 15, 20, None)


def deduplicate_by_page(documents):
    """Keep the first document for each PDF page."""
    unique_documents = []
    seen_pages = set()

    for doc in documents:
        page = doc.metadata.get("page")

        if page is None or page not in seen_pages:
            unique_documents.append(doc)

            if page is not None:
                seen_pages.add(page)

    return unique_documents


def get_page_numbers(documents):
    """Extract page numbers while preserving ranking order."""
    return [
        doc.metadata.get("page")
        for doc in documents
    ]


def rerank_documents(model, question, documents):
    """Score and rank all supplied candidate documents."""
    if not documents:
        return []

    pairs = [
        (question, doc.page_content)
        for doc in documents
    ]

    scores = model.predict(
        pairs,
        show_progress_bar=False,
    )

    ranked = sorted(
        zip(documents, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    return ranked


def calculate_candidate_recall_at_n(candidate_pages, relevant_pages):
    """Calculate recall over the actual candidate pool."""
    relevant = set(relevant_pages)

    if not relevant:
        return None

    retrieved = set(candidate_pages)

    return len(retrieved & relevant) / len(relevant)


def calculate_latency_summary(latencies):
    if not latencies:
        return {
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }

    return {
        "mean_ms": round(statistics.mean(latencies), 3),
        "median_ms": round(statistics.median(latencies), 3),
        "min_ms": round(min(latencies), 3),
        "max_ms": round(max(latencies), 3),
    }


def configuration_name(pool_size):
    return f"top_{pool_size}" if pool_size is not None else "full"


def warm_up(model, question, candidate_documents):
    """Warm up CrossEncoder inference outside measured evaluation."""
    print("\nWarming up CrossEncoder...")

    # Warm up with each distinct candidate count.
    warmup_sizes = sorted({
        min(size, len(candidate_documents))
        for size in (10, 15, 20)
    } | {len(candidate_documents)})

    for size in warmup_sizes:
        if size == 0:
            continue

        rerank_documents(
            model,
            question,
            candidate_documents[:size],
        )

        print(f"Warm-up completed: {size} candidates")

    print("Warm-up completed.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CrossEncoder candidate-pool sizes."
    )

    parser.add_argument(
        "--questions",
        default="evaluation/questions.json",
    )

    parser.add_argument(
        "--output",
        default="experiments/candidate_pool_experiment.json",
    )

    parser.add_argument(
        "--fetch-k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Warm up CrossEncoder inference before evaluation.",
    )

    args = parser.parse_args()

    if args.fetch_k < 1:
        parser.error("--fetch-k must be at least 1.")

    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")

    if args.top_k > 10:
        parser.error("--top-k cannot exceed the smallest pool size (10).")

    questions_path = Path(args.questions)
    output_path = Path(args.output)

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
    # 2. Initialize experiment containers
    # --------------------------------------------------

    configurations = {
        configuration_name(size): []
        for size in POOL_CONFIGURATIONS
    }

    reranker_latencies = {
        name: []
        for name in configurations
    }

    total_latencies = {
        name: []
        for name in configurations
    }

    retrieval_latencies = []

    # --------------------------------------------------
    # 3. Optional warm-up
    # --------------------------------------------------

    if args.warmup:
        first_question = (
            questions[0].get("retrieval_query")
            or questions[0]["question"]
        )

        warmup_components = retrieve_hybrid(
            first_question,
            top_k=args.fetch_k,
            fetch_k=args.fetch_k,
            return_components=True,
        )

        warmup_candidates = deduplicate_by_page(
            warmup_components["hybrid"]
        )

        warm_up(
            model,
            first_question,
            warmup_candidates,
        )

    # --------------------------------------------------
    # 4. Evaluate questions
    # --------------------------------------------------

    evaluation_start = time.perf_counter()

    for index, item in enumerate(questions, start=1):

        question = (
            item.get("retrieval_query")
            or item["question"]
        )

        relevant_pages = item["relevant_pages"]

        print(
            f"\n[{index}/{len(questions)}] {question}"
        )

        # ----------------------------------------------
        # Retrieve candidates exactly once per question.
        # ----------------------------------------------

        components = retrieve_hybrid(
            question,
            top_k=args.fetch_k,
            fetch_k=args.fetch_k,
            return_components=True,
        )

        timings = components["timings"]

        retrieval_ms = timings["total_retrieval_ms"]

        retrieval_latencies.append(retrieval_ms)

        hybrid_documents = deduplicate_by_page(
            components["hybrid"]
        )

        candidate_count = len(hybrid_documents)

        candidate_pages = get_page_numbers(
            hybrid_documents
        )

        # Rotate configuration execution order between
        # questions to reduce systematic order effects.
        execution_order = list(POOL_CONFIGURATIONS)

        rotation = (index - 1) % len(execution_order)

        execution_order = (
            execution_order[rotation:]
            + execution_order[:rotation]
        )

        question_results = {}

        # ----------------------------------------------
        # Evaluate each candidate-pool configuration.
        # ----------------------------------------------

        for pool_size in execution_order:

            name = configuration_name(pool_size)

            if pool_size is None:
                candidate_documents = hybrid_documents
            else:
                candidate_documents = hybrid_documents[:pool_size]

            selected_candidate_pages = get_page_numbers(
                candidate_documents
            )

            actual_candidate_count = len(candidate_documents)

            # Candidate generation recall.
            candidate_recall = calculate_candidate_recall_at_n(
                selected_candidate_pages,
                relevant_pages,
            )

            # Measure reranker stage.
            rerank_start = time.perf_counter()

            reranked_all = rerank_documents(
                model,
                question,
                candidate_documents,
            )

            reranker_ms = (
                time.perf_counter() - rerank_start
            ) * 1000

            reranked_documents = [
                doc
                for doc, score in reranked_all
            ]

            reranked_pages = get_page_numbers(
                reranked_documents
            )

            final_documents = reranked_documents[:args.top_k]

            retrieved_pages = get_page_numbers(
                final_documents
            )

            ranking_metrics = calculate_ranking_metrics(
                retrieved_pages,
                relevant_pages,
            )

            total_ms = retrieval_ms + reranker_ms

            result = {
                "id": item["id"],
                "question": item["question"],
                "relevant_pages": relevant_pages,

                "configured_pool_size": (
                    pool_size if pool_size is not None else "full"
                ),

                "available_candidate_count": candidate_count,
                "actual_candidate_count": actual_candidate_count,

                "candidate_pages": selected_candidate_pages,
                "candidate_recall": (
                    round(candidate_recall, 4)
                    if candidate_recall is not None
                    else None
                ),

                "reranked_pages": reranked_pages,
                "retrieved_pages": retrieved_pages,

                **ranking_metrics,

                "retrieval_latency_ms": round(retrieval_ms, 3),
                "reranker_latency_ms": round(reranker_ms, 3),
                "total_latency_ms": round(total_ms, 3),
            }

            question_results[name] = result

            configurations[name].append(result)

            reranker_latencies[name].append(reranker_ms)
            total_latencies[name].append(total_ms)

            print(
                f"{name}: "
                f"candidates={actual_candidate_count}, "
                f"recall={candidate_recall}, "
                f"retrieved={retrieved_pages}, "
                f"reranker={reranker_ms:.2f}ms"
            )

    evaluation_seconds = (
        time.perf_counter() - evaluation_start
    )

    # --------------------------------------------------
    # 5. Aggregate results
    # --------------------------------------------------

    metric_names = [
        "hit_at_1",
        "hit_at_2",
        "hit_at_4",
        "precision_at_1",
        "precision_at_2",
        "precision_at_4",
        "recall_at_1",
        "recall_at_2",
        "recall_at_4",
        "reciprocal_rank",
        "ndcg_at_4",
    ]

    summary = {}

    for name, results in configurations.items():

        count = len(results)

        metrics = {
            metric: round(
                statistics.mean(
                    result[metric]
                    for result in results
                ),
                4,
            )
            for metric in metric_names
        }

        valid_candidate_recalls = [
            result["candidate_recall"]
            for result in results
            if result["candidate_recall"] is not None
        ]

        candidate_recall_mean = (
            round(statistics.mean(valid_candidate_recalls), 4)
            if valid_candidate_recalls
            else None
        )

        candidate_counts = [
            result["actual_candidate_count"]
            for result in results
        ]

        summary[name] = {
            "configured_pool_size": (
                "full"
                if name == "full"
                else int(name.split("_")[1])
            ),

            "metrics": metrics,

            "candidate_recall": candidate_recall_mean,

            "questions_with_full_candidate_recall": sum(
                result["candidate_recall"] == 1.0
                for result in results
            ),

            "candidate_count_summary": {
                "mean": round(statistics.mean(candidate_counts), 2),
                "min": min(candidate_counts),
                "max": max(candidate_counts),
            },

            "reranker_latency": calculate_latency_summary(
                reranker_latencies[name]
            ),

            "total_latency": calculate_latency_summary(
                total_latencies[name]
            ),

            "results": results,
        }

    # --------------------------------------------------
    # 6. Prepare experiment artifact
    # --------------------------------------------------

    output = {
        "experiment": "CrossEncoder candidate-pool size comparison",

        "questions_evaluated": len(questions),

        "retrieval_configuration": {
            "fetch_k": args.fetch_k,
            "retrieval_method": "Hybrid RRF",
            "candidate_aggregation": "Page-level",
        },

        "reranker_model": args.model,

        "top_k": args.top_k,

        "candidate_pool_configurations": [
            10,
            15,
            20,
            "full",
        ],

        "warmup_enabled": args.warmup,

        "evaluation_duration_seconds": round(
            evaluation_seconds,
            3,
        ),

        "latency_methodology": {
            "retrieval": (
                "Hybrid retrieval is executed once per question. "
                "Its measured latency is shared across configurations."
            ),

            "reranker": (
                "Includes pair construction, CrossEncoder prediction "
                "and sorting for the selected candidate pool."
            ),

            "total": (
                "Measured hybrid retrieval latency plus the "
                "corresponding reranker stage latency."
            ),

            "execution_order": (
                "Candidate-pool execution order is rotated between "
                "questions to reduce systematic order effects."
            ),

            "warmup": (
                "Optional warm-up runs are excluded from measured "
                "evaluation results and latency statistics."
            ),
        },

        "retrieval_latency_summary": calculate_latency_summary(
            retrieval_latencies
        ),

        "configurations": summary,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # 7. Print comparison
    # --------------------------------------------------

    print("\n" + "=" * 85)
    print("CANDIDATE POOL SIZE COMPARISON")
    print("=" * 85)

    for name, data in summary.items():

        metrics = data["metrics"]

        print(f"\n{name.upper()}")
        print(f"Candidates: {data['candidate_count_summary']}")
        print(f"Candidate Recall: {data['candidate_recall']}")
        print(f"Hit@1: {metrics['hit_at_1']:.4f}")
        print(f"Hit@4: {metrics['hit_at_4']:.4f}")
        print(f"MRR: {metrics['reciprocal_rank']:.4f}")
        print(f"nDCG@4: {metrics['ndcg_at_4']:.4f}")
        print(
            f"Reranker Mean: "
            f"{data['reranker_latency']['mean_ms']:.2f}ms"
        )

        print(
            f"Total Mean: "
            f"{data['total_latency']['mean_ms']:.2f}ms"
        )

    print("\n" + "=" * 85)
    print(f"Questions evaluated: {len(questions)}")
    print(f"Evaluation duration: {evaluation_seconds:.2f}s")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()