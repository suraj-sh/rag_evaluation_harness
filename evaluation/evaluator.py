import json
from app.rag.retrieval import get_retriever

QUESTION_PATH = "evaluation/questions.json"

def load_questions():
    with open(QUESTION_PATH, "r", encoding="utf-8") as file:
        return json.load(file)

def evaluate_question(retriever, question):
    docs = retriever.invoke(question["question"])

    retrieved_pages = [
        doc.metadata.get("page")
        for doc in docs
    ]

    relevant_pages = question["relevant_pages"]

    # hit = any(
    #     page in relevant_pages
    #     for page in retrieved_pages
    # )

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

    return{
        "id": question["id"],
        "question": question["question"],
        "relevant_pages": relevant_pages,
        "retrieved_pages": retrieved_pages,
        "retrieved_docs" : docs,
        # "hit": hit
        "hit_at_1": hit_at_1,
        "hit_at_2": hit_at_2,
        "hit_at_4": hit_at_4
    }


def main():
    questions = load_questions()

    retriever = get_retriever()

    results = []

    for question in questions:
        result = evaluate_question(
            retriever,
            question
        )

        results.append(result)

        print(f"\nQuestion {result['id']}:")
        print(result['question'])

        print(f"Relevant pages: {result['relevant_pages']}")

        print(f"Retrieved pages: {result['retrieved_pages']}")

        # print(f"Hit@4: {'YES' if result['hit'] else 'NO'}")
        print(f"Hit@1: {'YES' if result['hit_at_1'] else 'NO'}")
        print(f"Hit@2: {'YES' if result['hit_at_2'] else 'NO'}")
        print(f"Hit@4: {'YES' if result['hit_at_4'] else 'NO'}")

        if not result["hit_at_4"]:
            print("\nRetrieved content:")

            for doc in result["retrieved_docs"]:
                print("-" * 50)
                print(f"Page: {doc.metadata.get('page')}")
                print(doc.page_content[:1000])

    # hits = sum(
    #     result["hit"]
    #     for result in results
    # )

    # total = len(results)

    # hit_rate = hits / total

    total = len(results)

    hits_at_1 = sum(
        result["hit_at_1"]
        for result in results
    )

    hits_at_2 = sum(
        result["hit_at_2"]
        for result in results
    )

    hits_at_4 = sum(
        result["hit_at_4"]
        for result in results
    )

    # print("\n" + "=" * 50)
    # print("BASELINE EVALUATION")
    # print("=" * 50)

    # print(f"Questions evaluated: {total}")
    # print(f"Successful retrievals: {hits}")
    # print(f"Hit@4: {hit_rate:.2%}")

    print("\n" + "=" * 50)
    print("EVALUATION")
    print("=" * 50)

    print(f"Questions evaluated: {total}")
    print(f"Hit@1: {hits_at_1}/{total} = {hits_at_1 / total:.2%}")
    print(f"Hit@2: {hits_at_2}/{total} = {hits_at_2 / total:.2%}")
    print(f"Hit@4: {hits_at_4}/{total} = {hits_at_4 / total:.2%}")

if __name__ == "__main__":
    main()