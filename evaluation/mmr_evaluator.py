
import json
import os
import time

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

QUESTION_PATH = "evaluation/questions.json"
RESULTS_DIR = "experiments"
CHROMA_PATH = "chroma_db"


def load_questions():
    with open(QUESTION_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def get_mmr_retriever():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model,
    )

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5,
        },
    )


def evaluate_question(question, retriever):
    query = question["question"]

    start_time = time.perf_counter()
    docs = retriever.invoke(query)
    latency_ms = (time.perf_counter() - start_time) * 1000

    retrieved_pages = [
        doc.metadata.get("page") for doc in docs
    ]
    relevant_pages = question["relevant_pages"]

    return {
        "id": question["id"],
        "question": query,
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


def main():
    questions = load_questions()
    retriever = get_mmr_retriever()
    results = []

    for question in questions:
        result = evaluate_question(question, retriever)
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
    print("DENSE MMR RETRIEVAL EVALUATION")
    print("=" * 50)
    print(f"Questions evaluated: {total}")

    if total:
        print(f"Hit@1: {hits_at_1}/{total} = {hits_at_1 / total:.2%}")
        print(f"Hit@2: {hits_at_2}/{total} = {hits_at_2 / total:.2%}")
        print(f"Hit@4: {hits_at_4}/{total} = {hits_at_4 / total:.2%}")

        latencies = [r["latency_ms"] for r in results]
        print(f"\nCold-start latency: {latencies[0]:.2f} ms")

        if total > 1:
            warm_latencies = latencies[1:]
            print(
                "Average warm-query latency: "
                f"{sum(warm_latencies) / len(warm_latencies):.2f} ms"
            )
            print(f"Minimum warm-query latency: {min(warm_latencies):.2f} ms")
            print(f"Maximum warm-query latency: {max(warm_latencies):.2f} ms")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "dense_mmr_fetch10_lambda05.json")

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
