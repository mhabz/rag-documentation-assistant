"""Ingest a PDF: extract, chunk, embed and store it in the vector database."""

from pathlib import Path

from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.chunker import chunk_pages
from app.ingestion.pdf_extractor import extract_pages
from app.vectorstore.qdrant_repository import QdrantRepository


def ingest_pdf(
    pdf_path: Path,
    embedder: EmbeddingService,
    repository: QdrantRepository,
    chunk_size: int = 800,
    overlap: int = 100,
    reset: bool = False,
) -> int:
    """Read a PDF and store its chunks, with their embeddings, in Qdrant.

    Ingesting the same PDF twice replaces the existing points instead of duplicating them.
    Use ``reset`` when the new run may produce fewer chunks than the previous one (for example
    after changing ``chunk_size``), otherwise the extra old points would remain.

    Args:
        pdf_path: Path to a text-based PDF.
        embedder: Service that turns chunk texts into vectors.
        repository: Vector store receiving the chunks.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters shared by two consecutive chunks.
        reset: If True, delete the collection before ingesting.

    Returns:
        The number of chunks stored. 0 if the PDF contains no text.

    Raises:
        pymupdf.FileNotFoundError: If ``pdf_path`` does not exist.
        pymupdf.FileDataError: If the file is not a valid PDF.
        ValueError: If ``chunk_size`` or ``overlap`` is invalid.
    """
    chunks = chunk_pages(extract_pages(pdf_path), chunk_size, overlap)
    if reset:
        repository.delete_collection()
    if not chunks:
        return 0

    vectors = embedder.embed_texts([chunk.text for chunk in chunks])
    repository.ensure_collection(vector_size=len(vectors[0]))
    repository.upsert_chunks(chunks, vectors)
    return len(chunks)
