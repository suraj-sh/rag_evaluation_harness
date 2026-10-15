import json
import os
import argparse
from app.rag.retrieval import get_retriever

QUESTION_PATH = "evaluation/questions.json"
RESULTS_DIR = "experiments"


def load_questions():
    with open(QUESTION_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def build_retrieval_query(question, query_mode):
    original = question["question"]

    if query_mode == "nist_prefix":
        # Add consistent context without inserting the answer.
        return (
            "According to the NIST AI RMF, "
            + original[0].lower()
            + original[1:]
        )

    return original


def evaluate_question(retriever, question, query_mode):
    retrieval_query = build_retrieval_query(
        question,
        query_mode
    )

    docs = retriever.invoke(retrieval_query)

    retrieved_pages = [
        doc.metadata.get("page")
        for doc in docs
    ]

    relevant_pages = question["relevant_pages"]

    hit_at_1 = any(
        page in relevant_pages
        for page in retrieved_pages[:1]
    )

    hit_at_2 = any(
        page in relevant_pages
        for page in retrieved_pages[:2]
    )

    hit_at_4 = any(
        page in relevant_pages
        for page in retrieved_pages[:4]
    )

    return {
        "id": question["id"],
        "question": question["question"],
        "retrieval_query": retrieval_query,
        "query_mode": query_mode,
        "relevant_pages": relevant_pages,
        "retrieved_pages": retrieved_pages,
        "hit_at_1": hit_at_1,
        "hit_at_2": hit_at_2,
        "hit_at_4": hit_at_4,
        "retrieved_docs": docs,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default=os.path.join(
            RESULTS_DIR,
            "latest_results.json"
        ),
        help="Path to save detailed evaluation results",
    )

    parser.add_argument(
        "--query-mode",
        choices=["original", "nist_prefix"],
        default="original",
        help="Use original questions or add a consistent NIST AI RMF prefix",
    )

    args = parser.parse_args()

    questions = load_questions()
    retriever = get_retriever()

    results = []

    print(f"Query mode: {args.query_mode}")

    for question in questions:
        result = evaluate_question(
            retriever,
            question,
            args.query_mode
        )

        results.append(result)

        print(f"\nQuestion {result['id']}:")
        print(result["question"])
        print(f"Retrieval query: {result['retrieval_query']}")
        print(f"Relevant pages: {result['relevant_pages']}")
        print(f"Retrieved pages: {result['retrieved_pages']}")

        print(f"Hit@1: {'YES' if result['hit_at_1'] else 'NO'}")
        print(f"Hit@2: {'YES' if result['hit_at_2'] else 'NO'}")
        print(f"Hit@4: {'YES' if result['hit_at_4'] else 'NO'}")

        if not result["hit_at_4"]:
            print("\nRetrieved content:")

            for doc in result["retrieved_docs"]:
                print("-" * 50)
                print(f"Page: {doc.metadata.get('page')}")
                print(doc.page_content[:1000])

    total = len(results)

    hits_at_1 = sum(r["hit_at_1"] for r in results)
    hits_at_2 = sum(r["hit_at_2"] for r in results)
    hits_at_4 = sum(r["hit_at_4"] for r in results)

    print("\n" + "=" * 50)
    print("EVALUATION")
    print("=" * 50)

    print(f"Questions evaluated: {total}")

    if total:
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

    os.makedirs(
        os.path.dirname(args.output) or ".",
        exist_ok=True
    )

    serializable_results = [
        {
            key: value
            for key, value in result.items()
            if key != "retrieved_docs"
        }
        for result in results
    ]

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(
            serializable_results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()