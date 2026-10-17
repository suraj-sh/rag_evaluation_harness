import json
import os
import argparse
import time

from app.rag.hybrid_retrieval import retrieve_hybrid

QUESTION_PATH = "evaluation/questions.json"
RESULTS_DIR = "experiments"


def load_questions():
    with open(QUESTION_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_question(question):
    retrieval_query = question["question"]
    start_time = time.perf_counter()
    docs = retrieve_hybrid(retrieval_query, top_k=4, fetch_k=10)
    latency_ms = (time.perf_counter() - start_time) * 1000

    retrieved_pages = [doc.metadata.get("page") for doc in docs]
    relevant_pages = question["relevant_pages"]

    return {
        "id": question["id"],
        "question": question["question"],
        "retrieval_query": retrieval_query,
        "relevant_pages": relevant_pages,
        "retrieved_pages": retrieved_pages,
        "latency_ms": round(latency_ms, 2),
        "hit_at_1": any(
            page in relevant_pages for page in retrieved_pages[:1]
        ),
        "hit_at_2": any(
            page in relevant_pages for page in retrieved_pages[:2]
        ),
        "hit_at_4": any(
            page in relevant_pages for page in retrieved_pages[:4]
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=os.path.join(
            RESULTS_DIR, "hybrid_1500_similarity_bm25.json"
        ),
        help="Path to save detailed evaluation results",
    )
    args = parser.parse_args()

    questions = load_questions()
    results = []

    for question in questions:
        result = evaluate_question(question)
        results.append(result)

        print(f"\nQuestion {result['id']}:")
        print(result["question"])
        print(f"Relevant pages: {result['relevant_pages']}")
        print(f"Retrieved pages: {result['retrieved_pages']}")
        print(f"Hit@1: {'YES' if result['hit_at_1'] else 'NO'}")
        print(f"Hit@2: {'YES' if result['hit_at_2'] else 'NO'}")
        print(f"Hit@4: {'YES' if result['hit_at_4'] else 'NO'}")
        print(f"Latency: {result['latency_ms']:.2f} ms")

    total = len(results)
    hits_at_1 = sum(r["hit_at_1"] for r in results)
    hits_at_2 = sum(r["hit_at_2"] for r in results)
    hits_at_4 = sum(r["hit_at_4"] for r in results)

    print("\n" + "=" * 50)
    print("HYBRID RETRIEVAL EVALUATION")
    print("=" * 50)
    print(f"Questions evaluated: {total}")

    if total:
        # Retrieval quality metrics
        print(
            f"Hit@1: {hits_at_1}/{total} = "
            f"{hits_at_1 / total:.2%}"
        )
        print(
            f"Hit@2: {hits_at_2}/{total} = "
            f"{hits_at_2 / total:.2%}"
        )
        print(
            f"Hit@4: {hits_at_4}/{total} = "
            f"{hits_at_4 / total:.2%}"
        )

        # Retrieval latency metrics
        latencies = [r["latency_ms"] for r in results]

        print(f"\nCold-start latency: {latencies[0]:.2f} ms")

        if total > 1:
            warm_latencies = latencies[1:]

            print(
                f"Average warm-query latency: "
                f"{sum(warm_latencies) / len(warm_latencies):.2f} ms"
            )
            print(
                f"Minimum warm-query latency: "
                f"{min(warm_latencies):.2f} ms"
            )
            print(
                f"Maximum warm-query latency: "
                f"{max(warm_latencies):.2f} ms"
            )
        
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()