from langchain_core.documents import Document

from app.rag.hybrid_retrieval import reciprocal_rank_fusion


def make_doc(page, chunk_id):
    return Document(
        page_content=f"Content for page {page}, chunk {chunk_id}",
        metadata={"page": page, "_id": chunk_id},
    )


def test_rrf_returns_pages_from_both_sources():
    dense = [make_doc(1, "d1"), make_doc(2, "d2")]
    bm25 = [make_doc(3, "b3"), make_doc(1, "b1")]

    results = reciprocal_rank_fusion(dense, bm25)

    result_pages = [doc.metadata["page"] for doc in results]

    assert set(result_pages) == {1, 2, 3}
    assert len(result_pages) == 3


def test_rrf_aggregates_same_page_across_sources():
    dense = [make_doc(9, "d9")]
    bm25 = [make_doc(9, "b9"), make_doc(10, "b10")]

    results = reciprocal_rank_fusion(dense, bm25)

    result_pages = [doc.metadata["page"] for doc in results]

    assert result_pages[0] == 9
    assert result_pages.count(9) == 1


def test_rrf_weights_change_page_ranking():
    dense = [make_doc(1, "d1"), make_doc(2, "d2")]
    bm25 = [make_doc(2, "b2"), make_doc(2, "b2_second")]

    results = reciprocal_rank_fusion(
        dense,
        bm25,
        dense_weight=1.0,
        bm25_weight=2.0,
    )

    assert results[0].metadata["page"] == 2
