import re
import time
from rank_bm25 import BM25Okapi
from app.rag.embedding_model import get_embedding_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def tokenize(text):
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def get_hybrid_retriever():
    """Initialize hybrid retrieval components."""
    embedding_model = get_embedding_model()

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model,
    )

    stored_data = vectorstore.get(
        include=["documents", "metadatas"]
    )
    documents = stored_data["documents"]
    metadatas = stored_data["metadatas"]
    ids = stored_data["ids"]

    if not documents:
        raise ValueError(
            "Chroma contains no documents. Run ingestion first."
        )

    tokenized_documents = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_documents)

    return vectorstore, bm25, documents, metadatas, ids

# Cache the initialized components for reuse across queries.
_hybrid_components = None


def get_cached_hybrid_retriever():
    """Return cached components, initializing them on first use."""
    global _hybrid_components

    if _hybrid_components is None:
        _hybrid_components = get_hybrid_retriever()

    return _hybrid_components


def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    k=60,
    dense_weight=1.0,
    bm25_weight=1.0,
):
    """Fuse dense and BM25 rankings, aggregating scores by page."""
    page_scores = {}
    page_docs = {}
    source_page_scores = {}
    chunk_scores = {}
    chunk_docs = {}

    sources = [
        ("dense", dense_weight, dense_results),
        ("bm25", bm25_weight, bm25_results),
    ]

    for source, weight, results in sources:
        for rank, doc in enumerate(results, start=1):
            doc_id = doc.metadata["_id"]
            page = doc.metadata.get("page")

            page_key = (
                ("page", page)
                if page is not None
                else ("chunk", doc_id)
            )

            contribution = weight / (k + rank)

            chunk_scores[doc_id] = (
                chunk_scores.get(doc_id, 0.0) + contribution
            )
            chunk_docs[doc_id] = doc

            if page_key not in source_page_scores:
                source_page_scores[page_key] = {}
                page_scores[page_key] = 0.0

            previous = source_page_scores[page_key].get(source, 0.0)

            source_page_scores[page_key][source] = max(
                previous, contribution
            )

    for page_key, contributions in source_page_scores.items():
        page_scores[page_key] = sum(contributions.values())

    # Choose the highest-scoring chunk as each page's representative.
    for doc_id, score in chunk_scores.items():
        doc = chunk_docs[doc_id]
        page = doc.metadata.get("page")

        page_key = (
            ("page", page)
            if page is not None
            else ("chunk", doc_id)
        )

        current = page_docs.get(page_key)

        if current is None or score > current[0]:
            page_docs[page_key] = (score, doc)

    ranked_pages = sorted(
        page_scores,
        key=lambda page_key: page_scores[page_key],
        reverse=True,
    )

    return [page_docs[key][1] for key in ranked_pages]


def retrieve_hybrid(
    query,
    top_k=4,
    fetch_k=10,
    dense_weight=1.0,
    bm25_weight=1.0,
    rrf_k=60,
    return_components=False,
):
    (
        vectorstore,
        bm25,
        documents,
        metadatas,
        ids,
    ) = get_cached_hybrid_retriever()

    # Measures warm retrieval latency only.
    # Model, Chroma, and BM25 initialization are excluded.
    total_start = time.perf_counter()

    # --------------------------------------------------
    # 1. Query embedding
    # --------------------------------------------------
    start = time.perf_counter()

    query_embedding = vectorstore._embedding_function.embed_query(query)

    embedding_seconds = time.perf_counter() - start

    # --------------------------------------------------
    # 2. Dense vector search
    # --------------------------------------------------
    start = time.perf_counter()

    dense_results = vectorstore._collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_k,
        include=["documents", "metadatas"],
    )

    dense_seconds = time.perf_counter() - start

    # --------------------------------------------------
    # 3. Dense result conversion
    # --------------------------------------------------
    dense_conversion_start = time.perf_counter()

    dense_docs = []

    for i, doc_id in enumerate(dense_results["ids"][0]):
        dense_docs.append(
            Document(
                page_content=dense_results["documents"][0][i],
                metadata={
                    **(dense_results["metadatas"][0][i] or {}),
                    "_id": doc_id,
                },
            )
        )

    dense_conversion_seconds = (
        time.perf_counter() - dense_conversion_start
    )

    # --------------------------------------------------
    # 4. BM25 scoring and ranking
    # --------------------------------------------------
    start = time.perf_counter()

    query_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)

    bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:fetch_k]

    bm25_seconds = time.perf_counter() - start

    # --------------------------------------------------
    # 5. BM25 result conversion
    # --------------------------------------------------
    bm25_conversion_start = time.perf_counter()

    bm25_docs = [
        Document(
            page_content=documents[i],
            metadata={
                **(metadatas[i] or {}),
                "_id": ids[i],
            },
        )
        for i in bm25_indices
    ]

    bm25_conversion_seconds = (
        time.perf_counter() - bm25_conversion_start
    )

    # --------------------------------------------------
    # 6. Reciprocal Rank Fusion
    # --------------------------------------------------
    fusion_start = time.perf_counter()

    fused_docs = reciprocal_rank_fusion(
        dense_docs,
        bm25_docs,
        k=rrf_k,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
    )

    fusion_seconds = time.perf_counter() - fusion_start

    # Capture total before diagnostic operations and printing.
    total_seconds = time.perf_counter() - total_start

    # --------------------------------------------------
    # 9. Return results
    # --------------------------------------------------
    if return_components:
        return {
            "dense": dense_docs,
            "bm25": bm25_docs,
            "hybrid": fused_docs,
            "timings": {
                "embedding_ms": round(
                    embedding_seconds * 1000, 3
                ),
                "dense_search_ms": round(
                    dense_seconds * 1000, 3
                ),
                "dense_conversion_ms": round(
                    dense_conversion_seconds * 1000, 3
                ),
                "bm25_search_ms": round(
                    bm25_seconds * 1000, 3
                ),
                "bm25_conversion_ms": round(
                    bm25_conversion_seconds * 1000, 3
                ),
                "rrf_ms": round(
                    fusion_seconds * 1000, 3
                ),
                "total_retrieval_ms": round(
                    total_seconds * 1000, 3
                ),
            },
        }

    return fused_docs[:top_k]


if __name__ == "__main__":
    query = input("Question: ")
    results = retrieve_hybrid(query)

    for rank, doc in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Page: {doc.metadata.get('page')}")
        print(doc.page_content[:1000])