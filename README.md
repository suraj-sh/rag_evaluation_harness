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
- Google Gemini (answer generation)
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

**Scope:** Hit@K metrics evaluate retrieval performance, not generated-answer correctness or faithfulness. The project now also includes a separate, preliminary 10-question answer-evaluation dataset to inspect expected answer points, cited PDF pages, and abstention behavior. Citation overlap and abstention signals are partly heuristic and require manual verification; they should not be interpreted as a comprehensive answer-correctness score.

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

### Latency Comparison

Latency measures how long the retrieval system takes to return relevant document chunks for a query.

| Metric | Dense-only | Hybrid (Dense + BM25 + RRF) |
|---|---:|---:|
| Average warm-query latency | 73.73 ms | 74.20 ms |
| Minimum warm-query latency | 59.36 ms | 60.39 ms |
| Maximum warm-query latency | 155.72 ms | 151.86 ms |

**Understanding the latency metrics**

- **Cold-start latency:** The time taken by the first timed query when the system may still need to initialize components, such as the embedding model, vector store, or BM25 index. It can be significantly higher than subsequent queries.

- **Warm-query latency:** The time taken to process a query after the required components have been initialized and are available for reuse. It better represents the retrieval latency during normal, repeated use.

- **Average warm-query latency:** The arithmetic mean of the measured warm-query latencies. It gives an overall indication of typical retrieval performance.

- **Minimum warm-query latency:** The shortest measured warm-query latency. It shows the fastest observed retrieval under the test conditions.

- **Maximum warm-query latency:** The longest measured warm-query latency among the warm queries. It shows the slowest observed retrieval after initialization, excluding the first query.

**Measurement notes**

Latency was measured across 30 evaluation questions. The first query was treated separately from the remaining 29 queries when calculating warm-query statistics.

The cold-start measurements from the two evaluators are not directly comparable because their timers start at different stages. The hybrid evaluator includes lazy component initialization in its first timed query, whereas the dense-only evaluator initializes its retriever before timing individual queries.

The results indicate that hybrid retrieval substantially improved Hit@K scores while average warm-query latency remained nearly unchanged. These findings are preliminary and specific to the current dataset, configuration, and test environment.


## Answer-Level Evaluation

In addition to retrieval benchmarking, the project includes a separate 10-question dataset for preliminary evaluation of generated answers using the NIST AI RMF 1.0.

Each evaluation question defines expected answer points, expected physical PDF citation pages where applicable, and whether the system should abstain when the source does not support a requested claim.

The evaluation workflow:

- Generates answers using the configured language model.
- Records the generated answer and cited PDF pages.
- Checks answer content against predefined expected points.
- Measures citation-page overlap against expected pages.
- Uses heuristic signals to inspect expected abstention behavior.
- Saves the results as JSON for inspection and reproducibility.

The current answer-evaluation run completed all 10 questions using Gemini 3.1 Flash Lite. The report is stored in `experiments/answer_evaluation_20261022_154244.json`.

**Limitations:** This is an initial evaluation harness, not a validated answer-quality benchmark. Citation overlap does not establish that a citation supports a claim, and heuristic abstention detection does not establish factual correctness or groundedness. Manual review remains necessary.


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
│       ├── generation.py
│       ├── ingestion.py
│       ├── retrieval.py
│       └── hybrid_retrieval.py
│
├── data/
│   └── documents/
│       └── nist_ai_rmf_1.0.pdf
│
├── evaluation/
│    ├── questions.json
│    ├── evaluator.py
│    ├── hybrid_evaluator.py
│    ├── answer_questions.json
│    ├── answer_evaluator.py
│    └── score_answers.py
│
├── experiments/
│   └── benchmark result JSON files
│
├── scripts/
│   └── fix_page_metadata.py
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

### 4. Run Answer-Level Evaluation

```bash
python -m evaluation.answer_evaluator
```

### 5. Score an Existing Answer-Evaluation Report

```bash
python -m evaluation.score_answers
```

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
- [x] Measure retrieval latency
- [x] Implement preliminary answer-level evaluation for a separate 10-question dataset
- [x] Check expected answer points, citation-page overlap, and abstention signals
- [ ] Improve automated answer-correctness and groundedness scoring
- [ ] Manually validate answer-evaluation results
- [ ] Add reranking
- [ ] Add automated evaluation to CI/CD
- [ ] Introduce retrieval-quality regression thresholds
- [ ] Evaluate additional documents and question sets

## Future Experiments

Potential next steps include:

- Strengthening reference-answer criteria and automated answer-correctness evaluation
- Improving groundedness checks to verify that citations support generated claims
- Adding robust evaluation metrics beyond heuristic citation overlap and abstention signals
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