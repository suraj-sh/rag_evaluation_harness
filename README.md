# RAG Evaluation Harness

An evaluation framework for measuring and comparing Retrieval-Augmented Generation (RAG) retrieval strategies.

Instead of evaluating a RAG system only by whether the final answer looks correct, this project uses a fixed question set, manually defined retrieval ground truth, and measurable metrics to evaluate how effectively relevant information is retrieved.

The project currently benchmarks different chunking strategies and retrieval methods using the **NIST AI Risk Management Framework (AI RMF) 1.0** as its source document.

## Project Goals

This project aims to:

- Evaluate RAG retrieval quality using a fixed benchmark
- Compare similarity search and Maximum Marginal Relevance (MMR)
- Experiment with chunk sizes and overlap strategies
- Compare dense retrieval, BM25, and hybrid retrieval
- Combine dense and lexical rankings using Reciprocal Rank Fusion (RRF)
- Measure retrieval performance using Hit@K
- Evaluate answer correctness and groundedness
- Measure retrieval latency
- Experiment with reranking
- Integrate evaluation into CI/CD
- Detect retrieval-quality regressions automatically

## Current Architecture

```text
PDF Document
     |
Document Loader
     |
Text Splitting
     |
Embeddings
     |
Chroma Vector Store
     |
     +--------------------------+
     |                          |
Dense Retrieval            BM25 Retrieval
     |                          |
     +------------+-------------+
                  |
       Reciprocal Rank Fusion
                  |
       Top-K Retrieved Chunks
                  |
    Evaluation Against Ground Truth
                  |
          Hit@1, Hit@2, Hit@4
```

The project supports both conventional vector-based retrieval and hybrid retrieval that combines dense semantic search with BM25 lexical search.

## Tech Stack

- Python
- FastAPI
- LangChain
- Hugging Face Sentence Transformers
- Chroma
- BM25 (`rank-bm25`)
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

The initial benchmark compares three chunking configurations with two retrieval strategies.

- **Similarity Search:** Retrieves documents based on vector similarity to the query.
- **Maximum Marginal Relevance (MMR):** Selects relevant documents while also considering diversity among the retrieved results.

All six experiments use the same source document, 30 evaluation questions, embedding model, and `k=4`.

The MMR experiments additionally use `fetch_k=10` and `lambda_mult=0.5`.

A subsequent experiment evaluates hybrid retrieval using dense vector search, BM25 lexical retrieval, and Reciprocal Rank Fusion.

### Results: 30-Question Benchmark

| Chunk Size | Overlap | Retrieval | Hit@1 | Hit@2 | Hit@4 |
|---:|---:|---|---:|---:|---:|
| 500 | 100 | Similarity | 50.00% | 66.67% | 73.33% |
| 500 | 100 | MMR | 50.00% | 66.67% | 73.33% |
| 1000 | 200 | Similarity | 56.67% | 63.33% | 76.67% |
| 1000 | 200 | MMR | 56.67% | 60.00% | 70.00% |
| 1500 | 300 | Similarity | 40.00% | 60.00% | 83.33% |
| 1500 | 300 | MMR | 40.00% | 63.33% | 86.67% |
| 1500 | 300 | Hybrid (Dense + BM25 + RRF) | **63.33%** | **86.67%** | **93.33%** |

The hybrid experiment used equal-weight Reciprocal Rank Fusion, `fetch_k=10`, and `top_k=4`, without page-level deduplication.

### Observations

- **Best Hit@1:** Hybrid retrieval achieved 63.33%, retrieving a relevant page first for 19 of 30 questions.
- **Best Hit@2:** Hybrid retrieval achieved 86.67%, retrieving a relevant page within the top two results for 26 of 30 questions.
- **Best Hit@4:** Hybrid retrieval achieved 93.33%, retrieving a relevant page within the top four results for 28 of 30 questions.
- **Similarity versus MMR:** MMR produced identical aggregate scores to Similarity at 500/100. At 1000/200, Similarity performed better at Hit@2 and Hit@4. At 1500/300, MMR performed better at Hit@2 and Hit@4.
- **Hybrid retrieval:** Combining dense and lexical retrieval rankings improved all three Hit@K metrics compared with the 1500/300 Similarity baseline.
- **Fusion strategy:** Equal-weight RRF outperformed the tested dense-weighted fusion configuration on Hit@1 and Hit@2. Page-level deduplication also reduced the measured scores in the tested experiment.
- **Chunking trade-off:** Larger chunks performed better at Hit@4 in the original similarity and MMR experiments, but did not perform as well at Hit@1. Smaller chunks were more successful at Hit@2 in some configurations but less successful at Hit@4 than the best-performing configuration.

No retrieval configuration should be considered universally optimal based on these results. The findings are specific to the current document, question set, embedding model, retrieval parameters, and evaluation methodology.

The hybrid result is the strongest measured configuration in the experiments completed so far. It does not establish that hybrid retrieval will outperform other strategies on different datasets.

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

Similarity search retrieves the top `k` documents based on vector similarity to the query.

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

### Hybrid Retrieval

Hybrid retrieval combines dense vector search with BM25 lexical search.

The rankings are combined using Reciprocal Rank Fusion (RRF), which aggregates the positions of documents in the individual ranked lists.

The fusion score follows the general form:

\[
\operatorname{RRF}(d) =
\sum_{r \in R}
\frac{w_r}{c + \operatorname{rank}_r(d)}
\]

where:

- \(d\) is a document.
- \(R\) is the set of retrieval rankings.
- \(w_r\) is the weight assigned to a ranking.
- \(\operatorname{rank}_r(d)\) is the document's rank in that ranking.
- \(c\) is a constant that reduces the influence of high-ranking positions.

The tested hybrid configuration uses equal weights for dense and BM25 retrieval, with an RRF constant of 60.

```text
Dense Retrieval ──┐
                  ├──> Reciprocal Rank Fusion ──> Top-K Results
BM25 Retrieval ───┘
```

## Current Configuration

The strongest measured configuration so far is:

```text
Document: NIST AI RMF 1.0
Chunk size: 1500
Chunk overlap: 300
Embeddings: sentence-transformers/all-mpnet-base-v2
Vector store: Chroma
Dense retrieval: Similarity search
Lexical retrieval: BM25
Fusion: Equal-weight Reciprocal Rank Fusion
RRF constant: 60
Candidate pool (fetch_k): 10
Final retrieved chunks (top_k): 4
Evaluation questions: 30
Hit@1: 63.33%
Hit@2: 86.67%
Hit@4: 93.33%
```

This describes the best measured hybrid experiment, not a universally optimal configuration. The baseline Similarity and MMR configurations remain available for comparison.

## Project Structure

```text
rag_evaluation_harness/
│
├── app/
│   ├── main.py
│   └── rag/
│       ├── ingestion.py
│       ├── retrieval.py
│       └── hybrid_retrieval.py
│
├── data/
│   └── documents/
│       └── nist_ai_rmf_1.0.pdf
│
├── evaluation/
│   ├── questions.json
│   ├── evaluator.py
│   └── hybrid_evaluator.py
│
├── experiments/
│   └── benchmark result JSON files
│
├── tests/
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

The `.env` file is for local configuration and should not be committed to version control.

## Setup

Create and activate a Python virtual environment, then install the project dependencies.

For example, using `uv`:

```bash
uv venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
uv pip install -r requirements.txt
```

Ensure the virtual environment is activated and all dependencies are installed before running the commands below.

## Running the Project

### 1. Create the Vector Database

Run the ingestion pipeline:

```bash
python -m app.rag.ingestion
```

This loads the NIST AI RMF 1.0 PDF, splits it into chunks, generates embeddings, and stores the vectors in Chroma.

Rebuild the vector database when changing the chunk size, overlap, embedding model, or other ingestion settings that affect the stored vectors.

### 2. Run the Baseline Evaluation

```bash
python -m evaluation.evaluator
```

The evaluator runs the 30 benchmark questions against the configured baseline retriever and reports Hit@1, Hit@2, and Hit@4.

Configure Similarity Search or MMR in `app/rag/retrieval.py` before running the baseline evaluation.

### 3. Run the Hybrid Evaluation

```bash
python -m evaluation.hybrid_evaluator --output experiments/hybrid_1500_similarity_bm25.json
```

The hybrid evaluator runs the benchmark using dense retrieval, BM25, and RRF fusion.

The `--output` argument saves detailed evaluation results to a JSON file for later inspection and comparison.

Switching between Similarity, MMR, and hybrid retrieval does not require rebuilding the vector database when the stored chunks and embeddings remain unchanged.

## Implementation Status

- [x] Build the retrieval evaluation harness
- [x] Create the initial 10-question benchmark
- [x] Expand the benchmark to 30 questions
- [x] Compare Similarity Search and MMR
- [x] Compare three chunking configurations
- [x] Run all six baseline configurations against the 30-question benchmark
- [x] Implement hybrid retrieval using dense search and BM25
- [x] Fuse retrieval rankings using Reciprocal Rank Fusion
- [x] Compare hybrid retrieval against baseline retrieval
- [x] Save experiment results as JSON
- [ ] Evaluate answer correctness and groundedness
- [ ] Measure retrieval latency
- [ ] Add reranking
- [ ] Add automated evaluation to CI/CD
- [ ] Introduce retrieval-quality regression thresholds
- [ ] Evaluate additional documents and question sets

## Future Experiments

Potential next steps include:

- Adding reference answers to evaluate answer correctness
- Measuring whether generated answers are grounded in retrieved context
- Recording retrieval latency across configurations
- Testing alternative embedding models
- Experimenting with reranking
- Evaluating on additional documents and question sets
- Integrating evaluation into CI/CD
- Establishing regression thresholds to detect retrieval-quality degradation

## Why This Project?

Basic RAG implementations demonstrate that documents can be loaded, embedded, retrieved, and passed to an LLM.

This project focuses on a different engineering question:

> How do we know whether a RAG system is actually retrieving the right information?

By defining retrieval ground truth and comparing configurations using measurable metrics, this project treats retrieval as an evaluatable engineering component rather than relying solely on subjective inspection of generated answers.

The long-term goal is to extend the harness from retrieval evaluation to end-to-end RAG evaluation, including answer correctness, groundedness, latency, and automated regression testing.