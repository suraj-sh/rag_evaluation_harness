import argparse
import json
import statistics
import time
from pathlib import Path
from sentence_transformers import CrossEncoder
from app.rag.hybrid_retrieval import retrieve_hybrid
from evaluation.metrics import CANDIDATE_KS, calculate_ranking_metrics, calculate_candidate_recall


DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def deduplicate_by_page(documents):
    """Keep only the first result for each PDF page."""
    unique_documents = []
    seen_pages = set()

    for doc in documents:
        page = doc.metadata.get("page")

        if page is None or page not in seen_pages:
            unique_documents.append(doc)

            if page is not None:
                seen_pages.add(page)

    return unique_documents


def rerank_documents(model, question, documents, top_k=None):
    """Rerank candidate documents using a cross-encoder.

    If top_k is None, return the complete ranked candidate list.
    """
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

    return ranked if top_k is None else ranked[:top_k]

def evaluate_configuration(retrieved_pages, relevant_pages):
    """Calculate ranking metrics for one retrieval configuration."""
    return calculate_ranking_metrics(
        retrieved_pages,
        relevant_pages,
    )


def calculate_latency_summary(latencies):
    """Calculate mean and median latency."""
    if not latencies:
        return {
            "mean_ms": 0.0,
            "median_ms": 0.0,
        }

    return {
        "mean_ms": round(statistics.mean(latencies), 3),
        "median_ms": round(statistics.median(latencies), 3),
    }


def warm_up(retriever_query, model, warmup_runs, fetch_k, top_k):
    """
    Warm up retrieval and reranker before collecting measurements.

    Warm-up executions are excluded from evaluation results and
    latency statistics.
    """
    if warmup_runs <= 0:
        return

    print("\n" + "=" * 65)
    print(f"WARM-UP: {warmup_runs} RUN(S)")
    print("=" * 65)

    for run in range(1, warmup_runs + 1):
        print(f"\nWarm-up run {run}/{warmup_runs}")

        components = retrieve_hybrid(
            retriever_query,
            top_k=fetch_k,
            fetch_k=fetch_k,
            return_components=True,
        )

        hybrid_docs = deduplicate_by_page(
            components["hybrid"]
        )

        print(
            "[candidate-counts] "
            f"dense_chunks={len(components['dense'])}, "
            f"bm25_chunks={len(components['bm25'])}, "
            f"fused_pages={len(components['hybrid'])}, "
            f"deduplicated_pages={len(hybrid_docs)}"
        )

        # Warm up CrossEncoder inference.
        rerank_documents(
            model,
            retriever_query,
            hybrid_docs,
            top_k,
        )

    print("\nWarm-up completed. Starting measured evaluation.")


def main():
    parser = argparse.ArgumentParser(
        description="Compare four retrieval configurations."
    )

    parser.add_argument(
        "--questions",
        default="evaluation/questions.json",
    )

    parser.add_argument(
        "--output",
        default="experiments/retrieval_comparison.json",
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
        "--warmup-runs",
        type=int,
        default=2,
        help="Number of warm-up runs excluded from evaluation.",
    )

    args = parser.parse_args()

    if args.fetch_k < 1:
        parser.error("--fetch-k must be at least 1.")

    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")

    if args.warmup_runs < 0:
        parser.error("--warmup-runs cannot be negative.")

    questions_path = Path(args.questions)
    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
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
    # 2. Initialize result containers
    # --------------------------------------------------
    configurations = {
        "dense": [],
        "bm25": [],
        "hybrid": [],
        "hybrid_reranker": [],
    }

    latencies = {
        name: []
        for name in configurations
    }

    # Individual latency components across questions.
    latency_breakdown = {
        "embedding_ms": [],
        "dense_search_ms": [],
        "dense_conversion_ms": [],
        "bm25_search_ms": [],
        "bm25_conversion_ms": [],
        "rrf_ms": [],
        "total_retrieval_ms": [],
        "reranker_stage_ms": [],
    }

    # --------------------------------------------------
    # 3. Warm-up
    # --------------------------------------------------
    first_question = (
        questions[0].get("retrieval_query")
        or questions[0]["question"]
    )

    warm_up(
        retriever_query=first_question,
        model=model,
        warmup_runs=args.warmup_runs,
        fetch_k=args.fetch_k,
        top_k=args.top_k,
    )

    # --------------------------------------------------
    # 4. Evaluate all questions
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
        # Hybrid retrieval with component timings
        # ----------------------------------------------
        components = retrieve_hybrid(
            question,
            top_k=args.fetch_k,
            fetch_k=args.fetch_k,
            return_components=True,
        )

        timings = components.get("timings")

        if timings is None:
            raise RuntimeError(
                "retrieve_hybrid() did not return timing data. "
                "Check that return_components=True includes "
                "the 'timings' dictionary."
            )

        # Extract retrieval timings.
        embedding_ms = timings["embedding_ms"]
        dense_search_ms = timings["dense_search_ms"]
        dense_conversion_ms = timings["dense_conversion_ms"]

        bm25_search_ms = timings["bm25_search_ms"]
        bm25_conversion_ms = timings["bm25_conversion_ms"]

        rrf_ms = timings["rrf_ms"]
        total_retrieval_ms = timings["total_retrieval_ms"]

        # ----------------------------------------------
        # Deduplicate chunk-level results by PDF page
        # ----------------------------------------------
        dense_docs = deduplicate_by_page(
            components["dense"]
        )

        bm25_docs = deduplicate_by_page(
            components["bm25"]
        )

        hybrid_docs = deduplicate_by_page(
            components["hybrid"]
        )

        # Preserve every unique page candidate before reranking.
        candidate_pages = [
            doc.metadata.get("page")
            for doc in hybrid_docs
        ]

        candidate_count = len(candidate_pages)

        # ----------------------------------------------
        # Configuration A: Dense-only
        # ----------------------------------------------
        dense_top = dense_docs[:args.top_k]

        # ----------------------------------------------
        # Configuration B: BM25-only
        # ----------------------------------------------
        bm25_top = bm25_docs[:args.top_k]

        # ----------------------------------------------
        # Configuration C: Hybrid without reranking
        # ----------------------------------------------
        hybrid_top = hybrid_docs[:args.top_k]

        # ----------------------------------------------
        # Configuration D: Hybrid with CrossEncoder
        # ----------------------------------------------
        rerank_start = time.perf_counter()

        # Score and sort every candidate page.
        reranked_all = rerank_documents(
            model,
            question,
            hybrid_docs,
            top_k=None,
        )

        reranker_stage_ms = (
            time.perf_counter() - rerank_start
        ) * 1000

        # Preserve the complete reranked order for analysis.
        reranked_pages = [
            doc.metadata.get("page")
            for doc, _ in reranked_all
        ]

        # Only the top_k pages are used for final ranking metrics.
        reranked = reranked_all[:args.top_k]

        hybrid_reranker_top = [
            doc for doc, _ in reranked
        ]

        print(
            f"[reranker] candidates={candidate_count}, "
            f"scored={len(reranked_all)}, "
            f"returned={len(hybrid_reranker_top)}"
        )
        # ----------------------------------------------
        # Collect latency breakdown
        # ----------------------------------------------
        latency_breakdown["embedding_ms"].append(
            embedding_ms
        )

        latency_breakdown["dense_search_ms"].append(
            dense_search_ms
        )

        latency_breakdown["dense_conversion_ms"].append(
            dense_conversion_ms
        )

        latency_breakdown["bm25_search_ms"].append(
            bm25_search_ms
        )

        latency_breakdown["bm25_conversion_ms"].append(
            bm25_conversion_ms
        )

        latency_breakdown["rrf_ms"].append(
            rrf_ms
        )

        latency_breakdown["total_retrieval_ms"].append(
            total_retrieval_ms
        )

        latency_breakdown["reranker_stage_ms"].append(
            reranker_stage_ms
        )

        # ----------------------------------------------
        # Configuration-specific latency
        # ----------------------------------------------

        # Dense path includes embedding because dense search
        # requires a query embedding.
        dense_latency_ms = (
            embedding_ms
            + dense_search_ms
            + dense_conversion_ms
        )

        # BM25 does not require query embeddings.
        bm25_latency_ms = (
            bm25_search_ms
            + bm25_conversion_ms
        )

        # Hybrid uses both retrieval methods and RRF.
        hybrid_latency_ms = total_retrieval_ms

        # Hybrid + reranker includes retrieval and reranking.
        hybrid_reranker_latency_ms = (
            total_retrieval_ms
            + reranker_stage_ms
        )

        configuration_latencies = {
            "dense": dense_latency_ms,
            "bm25": bm25_latency_ms,
            "hybrid": hybrid_latency_ms,
            "hybrid_reranker": hybrid_reranker_latency_ms,
        }

        selected_results = {
            "dense": dense_top,
            "bm25": bm25_top,
            "hybrid": hybrid_top,
            "hybrid_reranker": hybrid_reranker_top,
        }

        # ----------------------------------------------
        # Calculate ranking metrics
        # ----------------------------------------------
        for name, documents in selected_results.items():
            retrieved_pages = [
                doc.metadata.get("page")
                for doc in documents
            ]

            metrics = evaluate_configuration(
                retrieved_pages,
                relevant_pages,
            )

            result_record = {
                "id": item["id"],
                "question": item["question"],
                "relevant_pages": relevant_pages,
                "retrieved_pages": retrieved_pages,
                **metrics,
            }

            # Candidate-generation diagnostics apply to the
            # hybrid + reranker configuration.
            if name == "hybrid_reranker":
                result_record.update({
                    "candidate_pages": candidate_pages,
                    "candidate_count": candidate_count,
                    "reranked_pages": reranked_pages,
                    "candidate_recall": calculate_candidate_recall(
                        candidate_pages,
                        relevant_pages,
                    ),
                })

            configurations[name].append(result_record)

            latencies[name].append(
                configuration_latencies[name]
            )

            print(
                f"{name}: {retrieved_pages}"
            )

        print(
            "[latency] "
            f"dense={dense_latency_ms:.2f}ms, "
            f"bm25={bm25_latency_ms:.2f}ms, "
            f"hybrid={hybrid_latency_ms:.2f}ms, "
            f"reranker={reranker_stage_ms:.2f}ms, "
            f"hybrid+reranker={hybrid_reranker_latency_ms:.2f}ms"
        )

    evaluation_seconds = (
        time.perf_counter() - evaluation_start
    )

    # --------------------------------------------------
    # 5. Aggregate ranking metrics
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

        averages = {
            metric: round(
                sum(result[metric] for result in results)
                / count,
                4,
            )
            for metric in metric_names
        }

        latency_stats = calculate_latency_summary(
            latencies[name]
        )

        averages["average_latency_ms"] = latency_stats["mean_ms"]
        averages["median_latency_ms"] = latency_stats["median_ms"]

        summary[name] = {
            "metrics": averages,
            "results": results,
        }
    # --------------------------------------------------
    # 5.1 Aggregate candidate-generation metrics
    # --------------------------------------------------
    candidate_results = configurations["hybrid_reranker"]

    candidate_metric_names = [
        f"candidate_recall_at_{k}"
        for k in CANDIDATE_KS
    ] + ["candidate_recall_all"]

    candidate_recall_summary = {}

    for metric in candidate_metric_names:
        values = [
            result["candidate_recall"][metric]
            for result in candidate_results
            if result["candidate_recall"][metric] is not None
        ]

        candidate_recall_summary[metric] = (
            round(statistics.mean(values), 4)
            if values
            else None
        )

    candidate_counts = [
        result["candidate_count"]
        for result in candidate_results
    ]

    candidate_pool_summary = {
        "mean_candidates": round(
            statistics.mean(candidate_counts), 2
        ),
        "min_candidates": min(candidate_counts),
        "max_candidates": max(candidate_counts),
    }
    # --------------------------------------------------
    # 6. Aggregate individual latency components
    # --------------------------------------------------
    latency_summary = {}

    for component, values in latency_breakdown.items():
        latency_summary[component] = (
            calculate_latency_summary(values)
        )

    # --------------------------------------------------
    # 7. Prepare output
    # --------------------------------------------------
    output = {
        "questions_evaluated": len(questions),
        "fetch_k": args.fetch_k,
        "top_k": args.top_k,
        "warmup_runs": args.warmup_runs,
        "reranker_model": args.model,
        "evaluation_duration_seconds": round(
            evaluation_seconds,
            3,
        ),
        "latency_methodology": {
            "dense": (
                "Embedding + dense search + dense document conversion. "
                "Component latency, not an independent execution."
            ),
            "bm25": (
                "BM25 scoring/ranking + BM25 document conversion. "
                "Component latency, not an independent execution."
            ),
            "hybrid": (
                "Total warm retrieval latency returned by "
                "retrieve_hybrid(), including embedding, dense search, "
                "BM25, conversions, and RRF."
            ),
            "hybrid_reranker": (
                "Hybrid retrieval latency + CrossEncoder reranker stage. "
                "The reranker stage includes pair construction, model prediction, "
                "and sorting of all unique page candidates."
            ),
            "warmup": (
                "Warm-up executions are excluded from evaluation "
                "metrics and latency statistics."
            ),
        },
        "latency_summary": latency_summary,
        "configurations": summary,
        "candidate_pool_summary": candidate_pool_summary,
        "candidate_recall_summary": candidate_recall_summary,
    }

    output_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # 8. Print final report
    # --------------------------------------------------
    print("\n" + "=" * 70)
    print("RETRIEVAL CONFIGURATION COMPARISON")
    print("=" * 70)

    for name, data in summary.items():
        print(f"\n{name.upper()}")

        for metric, value in data["metrics"].items():
            print(f"{metric}: {value}")

    print("\n" + "=" * 70)
    print("LATENCY BREAKDOWN")
    print("=" * 70)

    for component, stats in latency_summary.items():
        print(
            f"{component}: "
            f"mean={stats['mean_ms']:.3f}ms, "
            f"median={stats['median_ms']:.3f}ms"
        )

    print("\n" + "=" * 70)
    print(f"Warm-up runs: {args.warmup_runs}")
    print(f"Questions evaluated: {len(questions)}")
    print(f"Evaluation duration: {evaluation_seconds:.2f}s")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()