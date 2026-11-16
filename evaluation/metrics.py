import math

RANKING_KS = (1, 2, 4)
CANDIDATE_KS = (10, 20, 30)


def calculate_ranking_metrics(retrieved_pages, relevant_pages):
    relevant = set(relevant_pages)
    metrics = {}

    for k in RANKING_KS:
        retrieved_at_k = retrieved_pages[:k]
        matches = len(set(retrieved_at_k) & relevant)

        metrics[f"hit_at_{k}"] = int(matches > 0)
        metrics[f"precision_at_{k}"] = matches / k
        metrics[f"recall_at_{k}"] = (
            matches / len(relevant) if relevant else 0.0
        )

    first_relevant_rank = next(
        (
            rank
            for rank, page in enumerate(retrieved_pages, start=1)
            if page in relevant
        ),
        None,
    )

    metrics["reciprocal_rank"] = (
        1.0 / first_relevant_rank if first_relevant_rank else 0.0
    )

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, page in enumerate(retrieved_pages[:4], start=1)
        if page in relevant
    )

    ideal_count = min(len(relevant), 4)
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )

    metrics["ndcg_at_4"] = dcg / idcg if idcg else 0.0

    return metrics


def calculate_candidate_recall(candidate_pages, relevant_pages):
    """Measure how many relevant pages survive the candidate-generation stage."""
    relevant = set(relevant_pages)

    if not relevant:
        return {
            **{
                f"candidate_recall_at_{k}": None
                for k in CANDIDATE_KS
            },
            "candidate_recall_all": None,
        }

    metrics = {}

    for k in CANDIDATE_KS:
        retrieved_at_k = candidate_pages[:k]

        metrics[f"candidate_recall_at_{k}"] = (
            len(set(retrieved_at_k) & relevant) / len(relevant)
        )

    metrics["candidate_recall_all"] = (
        len(set(candidate_pages) & relevant) / len(relevant)
    )

    return metrics

def calculate_micro_recall(results):
    total_relevant = sum(
        len(set(result["relevant_pages"]))
        for result in results
    )

    micro_recall = {}

    for k in RANKING_KS:
        found = sum(
            len(
                set(result["retrieved_pages"][:k])
                & set(result["relevant_pages"])
            )
            for result in results
        )

        micro_recall[f"micro_recall_at_{k}"] = (
            round(found / total_relevant, 4)
            if total_relevant
            else 0.0
        )

    return micro_recall