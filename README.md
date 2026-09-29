# RAG Evaluation Harness

An evaluation framework for measuring and comparing Retrieval-Augmented Generation (RAG) retrieval strategies.

Instead of evaluating a RAG system only by whether the final answer "looks correct", this project uses a fixed question set and measurable retrieval metrics to evaluate how effectively relevant information is retrieved.

## Project Goals

This project is being developed to:

- Evaluate RAG retrieval quality using a fixed benchmark
- Compare different retrieval strategies
- Experiment with chunking strategies
- Measure retrieval performance using Hit@K
- Evaluate answer correctness
- Measure retrieval latency
- Experiment with reranking and hybrid retrieval
- Integrate evaluation into CI/CD
- Detect retrieval-quality regressions automatically

## Current Architecture

```text
PDF Document
     ↓
Document Loader
     ↓
Text Splitting
     ↓
Embeddings
     ↓
Chroma Vector Store
     ↓
Retriever
     ↓
Retrieved Documents
     ↓
Evaluation
     ↓
Hit@K Metrics
```

## Tech Stack

- Python
- LangChain
- Hugging Face Embeddings
- Chroma
- PyPDF
- NIST AI RMF 1.0

## Evaluation Dataset

The current benchmark uses the **NIST AI Risk Management Framework (AI RMF) 1.0** as the source document.

The evaluation dataset currently contains 10 questions covering concepts such as:

- AI RMF Core
- GOVERN
- MAP
- MEASURE
- MANAGE
- Trustworthy AI characteristics
- Risk tolerance
- AI bias

Each question contains manually defined relevant page numbers that serve as retrieval ground truth.

## Evaluation Metrics

### Hit@1

Checks whether at least one relevant page appears in the first retrieved result.

### Hit@2

Checks whether at least one relevant page appears within the first two retrieved results.

### Hit@4

Checks whether at least one relevant page appears within the first four retrieved results.


## Chunking and Retrieval Experiments

The experiment evaluates three chunking configurations with both Similarity Search and Maximum Marginal Relevance (MMR).

All experiments use the same NIST AI RMF 1.0 document, evaluation questions, embedding model, and `k=4`.

| Chunk Size | Overlap | Retrieval | Hit@1 | Hit@2 | Hit@4 |
|---:|---:|---|---:|---:|---:|
| 1000 | 200 | Similarity | 70% | 70% | 80% |
| 1000 | 200 | MMR | 70% | 70% | 90% |
| 500 | 100 | Similarity | 60% | 70% | 70% |
| 500 | 100 | MMR | 60% | 70% | 80% |
| 1500 | 300 | Similarity | 60% | 80% | 100% |
| 1500 | 300 | MMR | 60% | 90% | 100% |


### Similarity Search

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 4
    }
)
```

### MMR Configuration

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)
```

### Observations

- The 1500/300 configuration achieved the highest Hit@4 score among the tested configurations on the current benchmark.
- Both 1500/300 configurations achieved 100% Hit@4.
- MMR achieved a higher Hit@2 score than Similarity Search with 1500/300.
- The 500/100 configuration performed below the 1000/200 baseline on Hit@1 and Hit@4.
- Results are specific to the current 10-question benchmark.

## Current Configuration

```text
Document: NIST AI RMF 1.0
Chunk size: 1500
Chunk overlap: 300
Embeddings: Hugging Face Embeddings
Vector store: Chroma
Retrieval: MMR
k: 4
fetch_k: 10
lambda_mult: 0.5
```

## Experiments

- [x] Build retrieval evaluation harness
- [x] Create initial 10-question benchmark
- [x] Compare similarity search and MMR
- [x] Compare chunking strategies
- [ ] Expand evaluation dataset
- [ ] Evaluate answer correctness
- [ ] Measure retrieval latency
- [ ] Add reranking
- [ ] Experiment with hybrid retrieval
- [ ] Add CI evaluation
- [ ] Add regression thresholds

## Project Structure

```text
rag_evaluation_harness/
│
├── app/
│   ├── main.py
│   └── rag/
│       ├── ingestion.py
│       └── retrieval.py
│
├── data/
│   └── documents/
│       └── nist_ai_rmf_1.0.pdf
│
├── evaluation/
│   ├── questions.json
│   └── evaluator.py
│
├── experiments/
│
├── tests/
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

Install the project dependencies:

```bash
uv pip install -r requirements.txt
```

## Running the Project

### 1. Create the vector database

Run the ingestion pipeline:

```bash
python -m app.rag.ingestion
```

This loads the NIST AI RMF 1.0 PDF, splits it into chunks, generates embeddings, and stores them in Chroma.

### 2. Run the evaluation

Run the evaluation harness:

```bash
python -m evaluation.evaluator
```

The evaluator runs the benchmark questions against the configured retriever and reports Hit@1, Hit@2, and Hit@4.

## Future Experiments

The project will progressively evaluate additional RAG configurations, including:

- Different chunk sizes and overlap strategies
- Larger evaluation datasets
- Answer correctness
- Retrieval latency
- Reranking
- Hybrid retrieval
- Automated evaluation in CI/CD
- Retrieval-quality regression detection

## Why This Project?

Basic RAG implementations demonstrate that documents can be loaded, embedded, retrieved, and passed to an LLM.

This project focuses on a different problem:

> How do we know whether a RAG system is actually retrieving the right information?

The goal is to treat RAG retrieval as an engineering system that can be measured, compared, and tested rather than relying only on subjective inspection of generated answers.