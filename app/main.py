import time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from app.rag.retrieval import get_retriever
from app.rag.generation import generate_answer


app = FastAPI(title="RAG Evaluation Harness")

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    try:
        start = time.perf_counter()
        retriever = get_retriever()
        documents = retriever.invoke(question)
        retrieval_time = time.perf_counter() - start

        if not documents:
            return {
                "question": question,
                "answer": "I could not find relevant information in the documents.",
                "sources": [],
                "retrieval_time_seconds": round(retrieval_time, 4),
                "generation_time_seconds": 0,
                "total_time_seconds": round(retrieval_time, 4),
            }

        start = time.perf_counter()
        answer = generate_answer(question, documents)
        generation_time = time.perf_counter() - start

        sources = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get(
                    "pdf_page",
                    doc.metadata.get("page", "unknown"),
                ),
            }
            for doc in documents
        ]

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieval_time_seconds": round(retrieval_time, 4),
            "generation_time_seconds": round(generation_time, 4),
            "total_time_seconds": round(
                retrieval_time + generation_time, 4
            ),
        }
    except HTTPException:
        raise

    except RuntimeError as exc:
        if str(exc) == (
            "The AI service is temporarily unavailable. "
            "Please wait a moment and try again."
        ):
            raise HTTPException(
                status_code=503,
                detail=str(exc),
            ) from exc

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again later.",
        ) from exc

    except Exception as exc:
        print(f"Question processing failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again later.",
        ) from exc