"""Tests of the ingestion pipeline with a fake encoder and an in-memory Qdrant."""

import pymupdf
import pytest
from qdrant_client import QdrantClient

from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.pipeline import ingest_pdf
from app.vectorstore.qdrant_repository import QdrantRepository


class FakeEncoder:
    """Maps each text to a 2-dimensional vector, without any model."""

    def encode(self, sentences, **kwargs):
        return [[float(len(s)), 1.0] for s in sentences]


@pytest.fixture
def pdf_path(tmp_path):
    """A 2-page text PDF."""
    path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    for text in ("First page about mandates.", "Second page about refunds."):
        doc.new_page().insert_text((72, 72), text)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def repo():
    return QdrantRepository(QdrantClient(":memory:"), "test")


@pytest.fixture
def embedder():
    return EmbeddingService("any", encoder=FakeEncoder())


@pytest.mark.unit
def test_ingest_stores_one_chunk_per_short_page(pdf_path, embedder, repo):
    assert ingest_pdf(pdf_path, embedder, repo) == 2
    assert repo.count() == 2


@pytest.mark.unit
def test_ingested_chunks_keep_their_page(pdf_path, embedder, repo):
    ingest_pdf(pdf_path, embedder, repo)
    hits = repo.search([27.0, 1.0], limit=2)
    assert {h.page for h in hits} == {1, 2}


@pytest.mark.unit
def test_ingesting_twice_does_not_duplicate(pdf_path, embedder, repo):
    ingest_pdf(pdf_path, embedder, repo)
    ingest_pdf(pdf_path, embedder, repo)
    assert repo.count() == 2


@pytest.mark.unit
def test_reset_removes_points_of_a_previous_run(pdf_path, embedder, repo):
    ingest_pdf(pdf_path, embedder, repo, chunk_size=10, overlap=2)
    many = repo.count()
    ingest_pdf(pdf_path, embedder, repo, reset=True)
    assert many > repo.count() == 2


@pytest.mark.unit
def test_pdf_without_text_stores_nothing(tmp_path, embedder, repo):
    path = tmp_path / "blank.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(path)
    doc.close()
    assert ingest_pdf(path, embedder, repo) == 0
