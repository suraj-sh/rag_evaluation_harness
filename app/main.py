from fastapi import FastAPI

app = FastAPI(title="RAG Evaluation Harness")

@app.get("/")
def root():
    return {"message", "RAG Evaluation Harness"}