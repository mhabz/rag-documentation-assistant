# RAG Documentation Assistant

A question-answering assistant over the **SEPA Direct Debit Core Rulebook** (EPC), built from scratch
in Python. It retrieves the relevant passages of the PDF and answers with the source page.

The whole stack is **local and free**: no paid API, no cloud service. Questions can be asked in
French or English over an English document.

> **Status: work in progress.** Ingestion, embeddings and vector storage are done and tested.
> Ingesting the 233-page rulebook produces 800 chunks.
> Retrieval evaluation, answer generation and the API are next (see the roadmap).

## How it works

```
PDF -> pages -> chunks -> embeddings -> Qdrant
                                          |
question -> embedding -> top-k search ----+--> passages -> prompt -> local LLM -> answer + pages
```

| Step | What it does | Code |
|---|---|---|
| Extraction | Reads a text-based PDF page by page and removes repeated headers and footers | `app/ingestion/pdf_extractor.py` |
| Chunking | Splits each page into overlapping chunks and keeps the page number for citations | `app/ingestion/chunker.py` |
| Embeddings | Turns text into vectors with a local multilingual model (`sentence-transformers`) | `app/embeddings/embedding_service.py` |
| Vector store | Stores chunks in Qdrant and searches the closest ones (cosine similarity) | `app/vectorstore/qdrant_repository.py` |
| Ingestion | Chains extraction, chunking, embedding and storage | `app/ingestion/pipeline.py`, `scripts/ingest_documents.py` |

## Stack

Python 3.12, [uv](https://docs.astral.sh/uv/), PyMuPDF, sentence-transformers, Qdrant (Docker),
pytest, ruff. Planned: Ollama for a local LLM, FastAPI.

## Getting started

```bash
# 1. Install the dependencies
uv sync

# 2. Start Qdrant (http://localhost:6333/dashboard)
docker compose up -d

# 3. Ingest the PDF (see below for the download), about 10 seconds
uv run python -m scripts.ingest_documents --reset

# 4. Run the tests
uv run pytest -m "not integration"   # fast, no Docker nor model download
uv run pytest                        # everything, needs Qdrant and downloads the embedding model
```

The rulebook PDF is **not** included in this repository. Download it from the
[European Payments Council](https://www.europeanpaymentscouncil.eu/index.cfm/sepa-direct-debit/sepa-direct-debit-core-scheme-sdd-core/)
and save it as `data/sdd_core_rulebook.pdf`. The version used is EPC016-06 (2025 version 1.2).

## Design choices

- **Evaluation first.** Retrieval quality decides the quality of a RAG, so retrieval is measured
  before any LLM is added (golden set of questions, recall@k, MRR).
- **Testable by design.** The embedding model and the Qdrant client are injected, so unit tests run
  without downloading a model or starting a server. Tests check properties (chunk size, overlap,
  no lost content) rather than exact outputs, so they survive parameter tuning.
- **Idempotent ingestion.** A chunk's id is its position in the document, so ingesting the same PDF
  twice replaces points instead of duplicating them.
- **No framework.** The pipeline is written by hand to keep every step visible.

## Known limitations

- Text-based PDFs only: there is no OCR for scanned documents.
- Tables are flattened by the extraction, so their structure is lost.
- The header and footer cleanup also removes body lines that differ only by a number.
- Chunking counts characters, not tokens, and does not follow titles or paragraphs.
- A chunk never spans two pages, so a sentence cut by a page break has no overlap.

## Roadmap

- [x] Project setup, typed configuration
- [x] PDF extraction and chunking with page numbers
- [x] Local multilingual embeddings
- [x] Qdrant repository (idempotent upserts, similarity search)
- [x] Ingestion pipeline and script
- [ ] Retrieval and its evaluation (golden set, recall@k, MRR)
- [ ] Answer generation with a local LLM (Ollama) and page citations
- [ ] REST API (FastAPI)
- [ ] Docker image and CI (GitHub Actions)
