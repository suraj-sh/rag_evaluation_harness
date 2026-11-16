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
- Evaluate query-enhanced retrieval strategies
- Experiment with page-level candidate aggregation and deduplication
- Evaluate cross-encoder reranking
- Compare PyTorch and ONNX inference backends
- Evaluate FP32 and INT8 model quantization
- Measure retrieval and reranking latency
- Evaluate candidate-pool sizes and their impact on reranking quality and latency
- Evaluate answer correctness and citation behavior
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

The equal-weight hybrid configuration was the strongest performer in the original chunking and retrieval comparison. Subsequent experiments evaluated additional retrieval configurations, including query-enhanced retrieval, which achieved stronger measured Hit@K scores. These later experiments are reported separately below.

### Later Retrieval Experiments

After the original chunking benchmark, additional experiments evaluated alternative retrieval configurations using the 30-question retrieval benchmark.

| Retrieval configuration | Hit@1 | Hit@2 | Hit@4 |
|---|---:|---:|---:|
| Dense-only baseline | 40.00% | 60.00% | 83.33% |
| Hybrid, equal-weight RRF | 63.33% | 86.67% | 93.33% |
| Query-enhanced retrieval | **73.33%** | **96.67%** | **96.67%** |

Query-enhanced retrieval achieved the strongest recorded Hit@K results among these experiments. These results are specific to the current document, benchmark questions, and tested configurations; they do not establish performance on other datasets.

These experiments represent a later stage of evaluation than the original chunking comparison. Comparisons should be interpreted in the context of the retrieval configuration and evaluation methodology used.

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

## Advanced Hybrid Retrieval and Reranking

After the initial hybrid retrieval experiments, further work focused on improving candidate selection and ranking quality.

The advanced retrieval pipeline combines:

1. Dense vector retrieval for semantic matching.
2. BM25 lexical retrieval for keyword-based matching.
3. Reciprocal Rank Fusion (RRF) to combine the rankings.
4. Page-level aggregation and deduplication to reduce repeated results from the same PDF page.
5. Cross-encoder reranking to reorder candidates based on query-document relevance.

Unlike the initial retrieval methods, which rank documents using independently calculated retrieval scores, a cross-encoder jointly evaluates the query and candidate document.

The final retrieval comparison used a candidate pool of `fetch_k=20` and returned `top_k=4` results.

The cross-encoder used was:

`cross-encoder/ms-marco-MiniLM-L-6-v2`

## Final Retrieval Comparison

The final retrieval comparison evaluated four configurations using the same 30-question benchmark.

| Retrieval configuration | Hit@1 | Hit@2 | Hit@4 | MRR | nDCG@4 |
|---|---:|---:|---:|---:|---:|
| Dense-only | 46.67% | 63.33% | 83.33% | 0.6056 | 0.6494 |
| BM25 | 70.00% | 80.00% | 96.67% | 0.8056 | 0.8421 |
| Hybrid RRF | 76.67% | 93.33% | 100% | 0.8722 | 0.8946 |
| Hybrid RRF + Reranker | **86.67%** | **96.67%** | **100%** | **0.9278** | **0.9438** |

### Observations

- Dense retrieval achieved a Hit@1 of 46.67%.
- BM25 outperformed dense retrieval on this benchmark.
- Hybrid retrieval improved ranking quality by combining semantic and lexical retrieval.
- Cross-encoder reranking improved Hit@1 from 76.67% to 86.67%.
- The reranked configuration achieved an MRR of 0.9278 and nDCG@4 of 0.9438.
- All 30 questions retrieved at least one relevant page within the top four results.

These results are specific to the NIST AI RMF document, the current benchmark, and the tested configurations. The 30-question dataset is a development benchmark rather than an independent held-out test set.

## Retrieval and Reranking Latency

The final retrieval comparison measured the latency of individual stages in the hybrid retrieval pipeline.

| Component | Mean latency |
|---|---:|
| Embedding generation | 91.326 ms |
| Dense search | 6.493 ms |
| Dense document conversion | 0.226 ms |
| BM25 search | 1.136 ms |
| BM25 document conversion | 0.224 ms |
| RRF fusion | 0.179 ms |
| **Total hybrid retrieval** | **99.588 ms** |
| Cross-encoder reranking stage | 1,909.745 ms |

### Observations

- Embedding generation accounts for most of the initial hybrid retrieval latency.
- Dense and BM25 search operations are comparatively inexpensive.
- Cross-encoder reranking introduces a substantial additional inference cost.
- The reranking stage dominates the measured retrieval pipeline latency.

These are stage-level measurements from the retrieval comparison experiment, not a single end-to-end API latency measurement.


## Candidate-Pool Size Optimization

The previous experiments showed that cross-encoder reranking improved retrieval quality but introduced substantial inference latency.

A further experiment investigated whether reducing the number of candidates passed to the cross-encoder could reduce latency without sacrificing retrieval quality.

### Experimental Methodology

The experiment used the existing hybrid retrieval pipeline:

1. Retrieve candidates using dense retrieval and BM25.
2. Combine the rankings using Reciprocal Rank Fusion (RRF).
3. Aggregate and deduplicate candidates at the PDF-page level.
4. Truncate the ranked candidate list to the selected candidate-pool size.
5. Rerank the remaining candidates using the cross-encoder.
6. Evaluate the final top-four results against the existing 30-question benchmark.

The following candidate-pool configurations were compared:

- Top 10
- Top 15
- Top 20
- Full candidate pool

The hybrid retrieval configuration used `fetch_k=20` for each retriever. The full candidate pool therefore varied in size after page-level aggregation and deduplication.

### Candidate-Pool Evaluation Results

| Candidate Pool | Candidate Recall | Hit@1 | Hit@4 | MRR | nDCG@4 |
|---|---:|---:|---:|---:|---:|
| Top 10 | 100% | **90.00%** | 100% | **0.9444** | **0.9561** |
| Top 15 | 100% | 86.67% | 100% | 0.9278 | 0.9438 |
| Top 20 | 100% | 86.67% | 100% | 0.9278 | 0.9438 |
| Full | 100% | 86.67% | 100% | 0.9278 | 0.9438 |

All four configurations achieved 100% candidate recall across the 30 evaluation questions.

The actual number of unique page candidates varied after aggregation:

| Candidate Pool | Mean Candidates | Minimum | Maximum |
|---|---:|---:|---:|
| Top 10 | 10.00 | 10 | 10 |
| Top 15 | 15.00 | 15 | 15 |
| Top 20 | 19.87 | 17 | 20 |
| Full | 22.83 | 17 | 29 |

### Latency Comparison

The initial candidate-pool experiment was followed by two additional latency runs comparing Top 10 against the Full candidate pool.

The following values represent the average of the three run-level mean measurements.

| Metric | Top 10 | Full | Reduction |
|---|---:|---:|---:|
| Mean reranker latency | 1,193.58 ms | 2,763.39 ms | **56.8%** |
| Mean total latency | 1,375.60 ms | 2,945.41 ms | **53.3%** |

Total latency includes hybrid retrieval and the reranking stage.

The repeated measurements showed consistent reranker latency reductions:

- Initial experiment: 55.14%
- Repeat 1: 57.43%
- Repeat 2: 58.05%

### Observations

- Top 10 achieved the highest Hit@1, MRR, and nDCG@4 among the tested configurations.
- All candidate-pool configurations retained 100% candidate recall on the current benchmark.
- Hit@4 remained at 100% across all configurations.
- Top 10 reduced mean reranker latency by approximately 56.8% compared with the Full candidate pool.
- Mean total latency decreased by approximately 53.3%.
- The Top 10 configuration did not sacrifice measured retrieval quality on this benchmark.

### Selected Configuration

Top 10 was selected as the preferred candidate-pool size for the current benchmark.

It achieved the best measured ranking metrics while substantially reducing reranking latency.

These results are specific to the current NIST AI RMF document, 30-question development benchmark, retrieval configuration, cross-encoder, and local test environment. They do not establish that Top 10 will be optimal for other documents or datasets.

The latency measurements are experimental observations rather than universal performance guarantees.


## Cross-Encoder Backend Optimization

After evaluating the retrieval-quality improvements from cross-encoder reranking, the next experiment focused on reducing inference latency.

Three inference backends were evaluated using the same cross-encoder model:

`cross-encoder/ms-marco-MiniLM-L-6-v2`

The evaluated configurations were:

- PyTorch FP32
- ONNX FP32
- ONNX INT8

### Backend Comparison

| Metric | PyTorch FP32 | ONNX FP32 | ONNX INT8 |
|---|---:|---:|---:|
| Hit@1 | 86.67% | 86.67% | 83.33% |
| Hit@2 | 96.67% | 96.67% | 96.67% |
| Hit@4 | 100% | 100% | 100% |
| MRR | 0.9278 | 0.9278 | 0.9111 |
| nDCG@4 | 0.9438 | 0.9438 | 0.9314 |
| Mean inference latency | 2341.02 ms | 2814.89 ms | **2098.23 ms** |
| Median inference latency | 1981.72 ms | 2400.11 ms | **1785.45 ms** |
| P95 inference latency | 4156.74 ms | 4399.20 ms | 3971.15 ms |

### Model Size Comparison

| Backend | Model size |
|---|---:|
| ONNX FP32 | 86.80 MB |
| ONNX INT8 | 22.01 MB |

INT8 quantization reduced the model size by approximately 74.65%.

### Observations

- ONNX FP32 reproduced the PyTorch retrieval results, achieving 100% top-four page-set agreement across the 30 questions.
- ONNX FP32 was approximately 20.24% slower than PyTorch in this benchmark.
- ONNX INT8 reduced mean inference latency by approximately 10.38% compared with PyTorch FP32.
- INT8 maintained 100% Hit@4 but showed a small reduction in Hit@1, MRR, and nDCG@4.
- INT8 provided a useful model-size and inference-latency trade-off.

### Measurement Limitations

These results come from a single 30-question run. The backends were executed in a fixed order, with warm-up performed only on the first question. Latency differences should therefore be treated as experimental observations rather than definitive performance guarantees.

## INT8 Ranking Stability Analysis

A further diagnostic compared the relevant-page rankings produced by PyTorch FP32 and ONNX INT8.

Across the 30 evaluation questions:

| Comparison | Number of questions |
|---|---:|
| Same top-four set and same relevant-page rank | 22 |
| Same top-four set but different relevant-page rank | 3 |
| Different top-four set but same relevant-page rank | 5 |
| Different top-four set and different relevant-page rank | 0 |

Only three questions experienced a change in the relevant page's position:

- One question improved under INT8.
- Two questions regressed under INT8.

The aggregate MRR difference was -0.0167.

Most observed differences involved the selection or ordering of non-relevant candidates rather than a change in the relevant page's position.

This diagnostic provides additional insight into ranking stability but does not establish complete equivalence between FP32 and INT8.

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

The project saves answer-evaluation runs as JSON reports in `experiments/`. Multiple reports are retained to support comparison and reproducibility. The scoring utility can select a specific saved report, allowing an earlier run to be evaluated independently of the most recently generated report.

**Limitations:** This is an initial evaluation harness, not a validated answer-quality benchmark. Citation overlap does not establish that a citation supports a claim, and heuristic abstention detection does not establish factual correctness or groundedness. Manual review remains necessary.

Expected citation pages use zero-based PDF page indices in the evaluation data, while displayed citations such as `[Page 25]` use one-based numbering. The scoring utility converts citation numbers before comparing them with expected indices. PDF viewer page labels may differ from physical PDF page indices, so citation results should be checked against the source document.


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


## Configuration Used in the Final Retrieval Comparison

The following configuration corresponds to the four-way retrieval comparison reported earlier in this README. It used a candidate pool of 20 before the subsequent candidate-pool optimization experiment.

The optimized Top 10 candidate-pool configuration and its results are documented in the Candidate-Pool Size Optimization section.


```text
Source document: NIST AI RMF 1.0

Chunk size: 1500
Chunk overlap: 300

Embedding model:
sentence-transformers/all-mpnet-base-v2

Vector store: Chroma
Dense retrieval: Similarity search
Lexical retrieval: BM25

Fusion: Equal-weight Reciprocal Rank Fusion
RRF constant: 60

Candidate pool: 20
Final results: 4

Page-level aggregation: Enabled

Reranker:
cross-encoder/ms-marco-MiniLM-L-6-v2

Reference backend: PyTorch FP32

Evaluation questions: 30

Hit@1: 86.67%
Hit@2: 96.67%
Hit@4: 100%

MRR: 0.9278
nDCG@4: 0.9438
```

The subsequent candidate-pool experiment selected Top 10 as the preferred reranking configuration, achieving Hit@1 of 90.00%, Hit@4 of 100%, MRR of 0.9444, and nDCG@4 of 0.9561, with 100% candidate recall on the current benchmark.


## Project Structure

```text
rag_evaluation_harness/
├── app/
│   ├── main.py
│   └── rag/
│       ├── embedding_model.py
│       ├── generation.py
│       ├── hybrid_retrieval.py
│       ├── ingestion.py
│       └── retrieval.py
│
├── data/
│   └── documents/
│       └── nist_ai_rmf_1.0.pdf
│
├── evaluation/
│   ├── answer_evaluator.py
│   ├── answer_questions_hybrid.json
│   ├── answer_questions.json
│   ├── answer_review.json
│   ├── candidate_pool_experiment.py
│   ├── candidate_pool_latency_repeat.py
│   ├── compare_answer_reports.py
│   ├── dense_evaluator.py
│   ├── evaluator.py
│   ├── hybrid_evaluator_candidates.py
│   ├── hybrid_evaluator.py
│   ├── map_query_evaluator.py
│   ├── metrics.py
│   ├── mmr_evaluator.py
│   ├── questions_query_enhanced.json
│   ├── questions.json
│   ├── retrieval_comparison.py
│   └── score_answers.py
│
├── experiments/
│   └── Retrieval, candidate-pool, latency, and answer-evaluation JSON reports
│
├── scripts/
│   └── fix_page_metadata.py
│
├── static/
│   ├── app.js
│   ├── index.html
│   └── style.css
│
├── tests/
│   └── test_citation_parser.py
│
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

The `.env` file is for local configuration and should not be committed to version control.

## Setup

### Prerequisites

- Python 3.11
- `uv` for dependency management
- A Google Gemini API key for answer generation

### 1. Create a Virtual Environment

Create and activate a Python virtual environment:

```bash
uv venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```bash
uv pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
```

Keep your API key private. Do not commit `.env` to version control.

The NIST AI RMF 1.0 source PDF is included in the repository at `data/documents/nist_ai_rmf_1.0.pdf`.

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


### 4. Run the Candidate-Pool Experiment

```powershell
python -m evaluation.candidate_pool_experiment --fetch-k 20 --top-k 4 --warmup
```

This experiment compares candidate-pool sizes of 10, 15, 20, and Full before cross-encoder reranking.

It evaluates candidate recall, ranking quality, and latency, and saves the detailed results to:

`experiments/candidate_pool_experiment.json`

### 5. Repeat the Candidate-Pool Latency Comparison

```powershell
python -m evaluation.candidate_pool_latency_repeat --repeats 2 --fetch-k 20
```

This experiment performs two additional latency runs comparing Top 10 against the Full candidate pool.

The results are saved to:

`experiments/candidate_pool_latency_repeat.json`

### 6. Run Answer-Level Evaluation

```bash
python -m evaluation.answer_evaluator
```

### 7. Score an Existing Answer-Evaluation Report

```bash
python -m evaluation.score_answers
```
## Testing

Run the unit tests from the project root with the virtual environment activated:

```powershell
python -m pytest -v
```

The test suite covers:

- Citation-page parsing, including single pages, multiple pages, page ranges, duplicate citations, missing citations, and reverse ranges.
- Hybrid retrieval and Reciprocal Rank Fusion behavior.
- Answer-evaluation report comparison.
- Expected-answer-point scoring and validation.

The current test suite contains 31 tests.

Check for whitespace errors before committing:

```powershell
git diff --check
```

These tests validate implementation behavior. They do not independently establish retrieval quality or generated-answer correctness.

## Implementation Status

- [x] Build the retrieval evaluation harness
- [x] Create the initial 10-question benchmark
- [x] Expand the benchmark to 30 questions
- [x] Compare Similarity Search and MMR
- [x] Compare three chunking configurations
- [x] Run all six baseline configurations
- [x] Implement dense and BM25 hybrid retrieval
- [x] Implement Reciprocal Rank Fusion
- [x] Evaluate query-enhanced retrieval
- [x] Implement page-level candidate aggregation and deduplication
- [x] Implement cross-encoder reranking
- [x] Compare dense, BM25, hybrid, and reranked retrieval
- [x] Measure retrieval and reranking latency
- [x] Benchmark PyTorch FP32, ONNX FP32, and ONNX INT8
- [x] Analyze INT8 ranking differences against FP32
- [x] Save experiment results as JSON
- [x] Implement preliminary answer-level evaluation
- [x] Evaluate expected answer points, citation-page overlap, and abstention signals
- [x] Add unit tests for citation-page parsing
- [x] Support scoring saved answer-evaluation reports
- [x] Compare candidate-pool sizes of 10, 15, 20, and Full
- [x] Validate candidate recall and per-question ranking differences
- [x] Repeat candidate-pool latency measurements across three runs
- [x] Select Top 10 as the preferred candidate-pool size for the current benchmark
- [ ] Improve automated answer-correctness and groundedness scoring
- [ ] Manually validate answer-evaluation results
- [ ] Integrate evaluation into CI/CD
- [ ] Introduce retrieval-quality regression thresholds
- [ ] Evaluate additional documents and question sets

## Future Experiments

Potential future improvements include:

- Strengthening reference-answer criteria and answer-correctness evaluation.
- Implementing claim-level citation verification.
- Improving groundedness evaluation beyond citation-page overlap.
- Validating generated answers through manual review.
- Evaluating on independent held-out question sets.
- Testing additional source documents and embedding models.
- Integrating retrieval evaluation into CI/CD.
- Establishing retrieval-quality regression thresholds.
- Exploring production-oriented inference optimization.

## Why This Project?

Basic RAG implementations demonstrate that documents can be loaded, embedded, retrieved, and passed to an LLM.

This project focuses on a different engineering question:

> How do we know whether a RAG system is actually retrieving the right information?

By defining retrieval ground truth and comparing configurations using measurable metrics, this project treats retrieval as an evaluatable engineering component rather than relying solely on subjective inspection of generated answers.

The long-term goal is to extend the harness from retrieval evaluation to end-to-end RAG evaluation, including answer correctness, groundedness, latency, and automated regression testing.