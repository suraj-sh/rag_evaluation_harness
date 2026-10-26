
import json
import os
import time

from app.rag.hybrid_retrieval import retrieve_hybrid

QUESTION_PATH = "evaluation/map_query_experiment.json"
OUTPUT_PATH = "experiments/map_query_experiment.json"


def main():
    with open(QUESTION_PATH, "r", encoding="utf-8") as file:
        questions = json.load(file)

    results = []

    for item in questions:
        query = item["retrieval_query"]
        relevant_pages = item["relevant_pages"]

        start = time.perf_counter()
        docs = retrieve_hybrid(
            query,
            top_k=4,
            fetch_k=10,
            dense_weight=1.0,
            bm25_weight=1.0,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        retrieved_pages = [
            doc.metadata.get("page") for doc in docs
        ]

        result = {
            "id": item["id"],
            "question": item["question"],
            "retrieval_query": query,
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

        results.append(result)

        print(f"\nQ{item['id']}: {item['question']}")
        print(f"Retrieval query: {query}")
        print(f"Relevant pages: {relevant_pages}")
        print(f"Retrieved pages: {retrieved_pages}")
        print(f"Hit@1: {result['hit_at_1']}")
        print(f"Hit@2: {result['hit_at_2']}")
        print(f"Hit@4: {result['hit_at_4']}")
        print(f"Latency: {latency_ms:.2f} ms")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
