import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag.retrieval import get_retriever
from app.rag.generation import generate_answer

app = FastAPI(title="RAG Evaluation Harness")

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


@app.get("/")
def root():
    return {"message": "RAG Evaluation Harness"}


@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    try:
        # retriever = get_retriever()
        # documents = retriever.invoke(question)
        start = time.perf_counter()
        retriever = get_retriever()
        documents = retriever.invoke(question)
        retrieval_time = time.perf_counter() - start

        if not documents:
            return {
                "question": question,
                "answer": "I could not find relevant information in the documents.",
                "sources": [],
            }

        # answer = generate_answer(question, documents)
        start = time.perf_counter()
        answer = generate_answer(question, documents)
        generation_time = time.perf_counter() - start

        print(f"Retrieval time: {retrieval_time:.2f}s")
        print(f"Generation time: {generation_time:.2f}s")
        print(f"Total measured time: {retrieval_time + generation_time:.2f}s")

        sources = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "unknown"),
            }
            for doc in documents
        ]

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question: {str(exc)}",
        ) from exc