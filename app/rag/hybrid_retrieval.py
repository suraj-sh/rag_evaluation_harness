import re
from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEmbeddings
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
    """Initialize and cache the hybrid retrieval components."""
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model,
    )

    # Load all stored chunks from Chroma.
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

    # Build the BM25 keyword index once.
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
    """Combine dense and BM25 rankings using reciprocal rank fusion."""
    scores = {}
    result_docs = {}

    for weight, results in [
        (dense_weight, dense_results),
        (bm25_weight, bm25_results),
    ]:
        for rank, doc in enumerate(results, start=1):
            doc_id = doc.metadata["_id"]

            scores[doc_id] = scores.get(doc_id, 0.0) + (
                weight / (k + rank)
            )
            result_docs[doc_id] = doc

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    return [result_docs[doc_id] for doc_id in ranked_ids]


def retrieve_hybrid(query, top_k=4, fetch_k=10, dense_weight=1.0, bm25_weight=1.0,):
    (vectorstore, bm25, documents, metadatas, ids) = get_cached_hybrid_retriever()

   # Dense semantic search with original Chroma IDs.
    query_embedding = vectorstore._embedding_function.embed_query(query)

    dense_results = vectorstore._collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_k,
        include=["documents", "metadatas"],
    )

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

    # Sparse keyword search.
    query_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:fetch_k]

    bm25_docs = [
        Document(
            page_content=documents[i],
            metadata={**(metadatas[i] or {}), "_id": ids[i]}
        )
        for i in bm25_indices
    ]


    fused_docs = reciprocal_rank_fusion(
        dense_docs,
        bm25_docs,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
    )

    return fused_docs[:top_k]

if __name__ == "__main__":
    query = input("Question: ")
    results = retrieve_hybrid(query)

    for rank, doc in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Page: {doc.metadata.get('page')}")
        print(doc.page_content[:1000])