"""Command-line entry point to ingest a PDF into Qdrant.

Usage:
    uv run python -m scripts.ingest_documents
    uv run python -m scripts.ingest_documents path/to/file.pdf --reset
"""

import argparse
import time
from pathlib import Path

from qdrant_client import QdrantClient

from app.core.config import get_settings
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.pipeline import ingest_pdf
from app.vectorstore.qdrant_repository import QdrantRepository

DEFAULT_PDF = Path("data/sdd_core_rulebook.pdf")


def main() -> None:
    """Parse the command line, run the ingestion and print a short summary."""
    parser = argparse.ArgumentParser(description="Ingest a PDF into the Qdrant collection.")
    parser.add_argument("pdf", nargs="?", type=Path, default=DEFAULT_PDF, help="PDF to ingest")
    parser.add_argument("--chunk-size", type=int, default=800, help="max characters per chunk")
    parser.add_argument("--overlap", type=int, default=100, help="characters shared by chunks")
    parser.add_argument("--reset", action="store_true", help="delete the collection first")
    args = parser.parse_args()

    settings = get_settings()
    embedder = EmbeddingService(settings.embedding_model)
    repository = QdrantRepository(QdrantClient(url=settings.qdrant_url), settings.qdrant_collection)

    started = time.perf_counter()
    stored = ingest_pdf(
        args.pdf, embedder, repository, args.chunk_size, args.overlap, reset=args.reset
    )
    elapsed = time.perf_counter() - started
    print(f"{stored} chunks stored in '{settings.qdrant_collection}' in {elapsed:.1f}s")
    print(f"Collection now holds {repository.count()} points.")


if __name__ == "__main__":
    main()
