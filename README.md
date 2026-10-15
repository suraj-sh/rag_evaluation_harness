# RAG Evaluation Harness

An evaluation framework for measuring and comparing Retrieval-Augmented Generation (RAG) retrieval strategies.

Instead of evaluating a RAG system only by whether the final answer looks correct, this project uses a fixed question set, manually defined retrieval ground truth, and measurable metrics to evaluate how effectively relevant information is retrieved.

The project currently benchmarks different chunking strategies and retrieval methods using the **NIST AI Risk Management Framework (AI RMF) 1.0** as its source document.

## Project Goals

This project aims to:

- Evaluate RAG retrieval quality using a fixed benchmark
- Compare similarity search and Maximum Marginal Relevance (MMR)
- Experiment with chunk sizes and overlap strategies
- Measure retrieval performance using Hit@K
- Evaluate answer correctness and groundedness
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
Evaluation Against Ground Truth
     ↓
Hit@1, Hit@2, Hit@4
```

## Tech Stack

- Python
- LangChain
- Hugging Face Embeddings
- Chroma
- PyPDF
- NIST AI RMF 1.0

## Evaluation Dataset

The benchmark uses the **NIST AI Risk Management Framework (AI RMF) 1.0** as its source document.

The current evaluation dataset contains **30 questions** covering topics such as:

- AI RMF Core and its four functions
- GOVERN, MAP, MEASURE, and MANAGE
- Trustworthy AI characteristics
- Risk tolerance and risk prioritization
- Categories of AI bias
- Governance, accountability, and monitoring
- AI system context and intended purpose
- Risk measurement and management

Each question contains manually defined relevant PDF page indices. These act as retrieval ground truth for determining whether the retriever returns relevant source material.

The benchmark is held constant across experiments so that the tested configurations can be compared using the same questions.

## Evaluation Metrics

### Hit@1

Measures whether at least one relevant page appears in the first retrieved result.

### Hit@2

Measures whether at least one relevant page appears among the first two retrieved results.

### Hit@4

Measures whether at least one relevant page appears among the first four retrieved results.

The evaluator reports each metric as both a count and a percentage.

For example, a Hit@4 score of 80% means that at least one relevant page was retrieved within the top four results for 24 out of 30 questions.

**Scope:** These metrics evaluate retrieval performance, not the correctness or faithfulness of a generated answer. Answer-level evaluation has not yet been implemented.

## Chunking and Retrieval Experiments

The benchmark compares three chunking configurations with two retrieval strategies:

- **Similarity Search:** Retrieves documents based on vector similarity to the query.
- **Maximum Marginal Relevance (MMR):** Selects relevant documents while also considering diversity among the retrieved results.

All six experiments use the same source document, 30 evaluation questions, embedding model, and `k=4`.

The MMR experiments additionally use `fetch_k=10` and `lambda_mult=0.5`.

### Results: 30-Question Benchmark

| Chunk Size | Overlap | Retrieval | Hit@1 | Hit@2 | Hit@4 |
|---:|---:|---|---:|---:|---:|
| 500 | 100 | Similarity | 50.00% | 66.67% | 73.33% |
| 500 | 100 | MMR | 50.00% | 66.67% | 73.33% |
| 1000 | 200 | Similarity | 56.67% | 63.33% | 76.67% |
| 1000 | 200 | MMR | 56.67% | 60.00% | 70.00% |
| 1500 | 300 | Similarity | 40.00% | 60.00% | 83.33% |
| 1500 | 300 | MMR | 40.00% | 63.33% | 86.67% |

### Observations

- **Best Hit@1:** Both 1000/200 configurations achieved 56.67%, retrieving a relevant page first for 17 of 30 questions.
- **Best Hit@2:** The 500/100 Similarity and MMR configurations achieved 66.67%, retrieving a relevant page within the top two results for 20 questions.
- **Best Hit@4:** The 1500/300 MMR configuration achieved 86.67%, retrieving a relevant page within the top four results for 26 questions.
- **Effect of MMR:** MMR produced identical aggregate scores to Similarity at 500/100. At 1000/200, Similarity performed better at Hit@2 and Hit@4. At 1500/300, MMR performed better at Hit@2 and Hit@4.
- **Chunking trade-off:** Larger chunks performed better at Hit@4 in this benchmark, but did not perform as well at Hit@1. Smaller chunks were more successful at Hit@2 but less successful at Hit@4 than the best-performing configuration.

No configuration performed best across every metric. The results demonstrate why retrieval configurations should be evaluated against a fixed benchmark instead of selecting a strategy based on assumptions alone.

These findings are specific to the current document, question set, embedding model, and retrieval parameters. They should not be interpreted as universal conclusions about chunking or retrieval methods.

## Historical Results: Initial 10-Question Benchmark

Before expanding the dataset, the project used an initial benchmark of 10 questions. The results below are retained for historical reference and are **not directly comparable** with the 30-question benchmark above.

| Chunk Size | Overlap | Retrieval | Hit@1 | Hit@2 | Hit@4 |
|---:|---:|---|---:|---:|---:|
| 1000 | 200 | Similarity | 70% | 70% | 80% |
| 1000 | 200 | MMR | 70% | 70% | 90% |
| 500 | 100 | Similarity | 60% | 70% | 70% |
| 500 | 100 | MMR | 60% | 70% | 80% |
| 1500 | 300 | Similarity | 60% | 80% | 100% |
| 1500 | 300 | MMR | 60% | 90% | 100% |

The benchmark was subsequently expanded to 30 questions to provide a broader evaluation of retrieval performance.

## Retrieval Configurations

### Similarity Search

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 4
    }
)
```

### Maximum Marginal Relevance (MMR)

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

The `k` parameter controls the number of documents returned. In MMR, `fetch_k` controls the initial candidate pool, while `lambda_mult` controls the balance between query relevance and diversity.

## Current Configuration

The latest evaluated configuration is:

```text
Document: NIST AI RMF 1.0
Chunk size: 1000
Chunk overlap: 200
Chunks created: 150
Embeddings: Hugging Face Embeddings
Vector store: Chroma
Retrieval: MMR
k: 4
fetch_k: 10
lambda_mult: 0.5
Evaluation questions: 30
```

The configuration remains experimental. The benchmark results do not establish one universally optimal configuration.

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
├── tests/
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

Install the project dependencies:

```bash
uv pip install -r requirements.txt
```

Ensure the project's virtual environment is activated and the dependencies are installed before running the commands below.

## Running the Project

### 1. Create the vector database

Run the ingestion pipeline:

```bash
python -m app.rag.ingestion
```

This loads the NIST AI RMF 1.0 PDF, splits it into chunks, generates embeddings, and stores the vectors in Chroma.

Rebuild the vector database when changing the chunk size or overlap.

### 2. Configure the retriever

Choose Similarity Search or MMR by updating the retriever configuration in `app/rag/retrieval.py`.

Switching retrieval methods does not require rebuilding the vector database when the underlying chunks and embeddings remain unchanged.

### 3. Run the evaluation

```bash
python -m evaluation.evaluator
```

The evaluator runs the 30 benchmark questions against the configured retriever and reports Hit@1, Hit@2, and Hit@4.

## Implementation Status

- [x] Build the retrieval evaluation harness
- [x] Create the initial 10-question benchmark
- [x] Expand the benchmark to 30 questions
- [x] Compare Similarity Search and MMR
- [x] Compare three chunking configurations
- [x] Run all six configurations against the 30-question benchmark
- [ ] Evaluate answer correctness and groundedness
- [ ] Measure retrieval latency
- [ ] Add reranking
- [ ] Experiment with hybrid retrieval
- [ ] Add automated evaluation to CI/CD
- [ ] Introduce retrieval-quality regression thresholds

## Future Experiments

Potential next steps include:

- Adding reference answers to evaluate answer correctness
- Measuring whether generated answers are grounded in retrieved context
- Recording retrieval latency across configurations
- Testing alternative embedding models
- Experimenting with reranking and hybrid retrieval
- Evaluating on additional documents and question sets
- Integrating evaluation into CI/CD
- Establishing regression thresholds to detect retrieval-quality degradation

## Why This Project?

Basic RAG implementations demonstrate that documents can be loaded, embedded, retrieved, and passed to an LLM.

This project focuses on a different engineering question:

> How do we know whether a RAG system is actually retrieving the right information?

By defining retrieval ground truth and comparing configurations using measurable metrics, the project treats retrieval as an evaluatable engineering component rather than relying solely on subjective inspection of generated answers.