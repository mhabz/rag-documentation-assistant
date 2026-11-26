"""Tests of the Qdrant repository, using Qdrant's in-memory mode (no Docker needed)."""

import pytest
from qdrant_client import QdrantClient

from app.ingestion.models import Chunk
from app.vectorstore.qdrant_repository import QdrantRepository


@pytest.fixture
def repo() -> QdrantRepository:
    """A repository on a fresh in-memory collection of 2-dimensional vectors."""
    repository = QdrantRepository(QdrantClient(":memory:"), "test")
    repository.ensure_collection(vector_size=2)
    return repository


def make_chunks() -> list[Chunk]:
    return [
        Chunk(text="east", page=1, index=0),
        Chunk(text="north", page=2, index=1),
        Chunk(text="west", page=3, index=2),
    ]


EAST, NORTH, WEST = [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]


@pytest.mark.unit
def test_search_returns_closest_chunk_first_with_its_page(repo):
    repo.upsert_chunks(make_chunks(), [EAST, NORTH, WEST])
    results = repo.search([0.9, 0.1], limit=2)
    assert [r.text for r in results] == ["east", "north"]
    assert results[0].page == 1
    assert results[0].score > results[1].score


@pytest.mark.unit
def test_search_respects_limit(repo):
    repo.upsert_chunks(make_chunks(), [EAST, NORTH, WEST])
    assert len(repo.search(EAST, limit=1)) == 1


@pytest.mark.unit
def test_ingesting_twice_does_not_duplicate(repo):
    repo.upsert_chunks(make_chunks(), [EAST, NORTH, WEST])
    repo.upsert_chunks(make_chunks(), [EAST, NORTH, WEST])
    assert repo.count() == 3


@pytest.mark.unit
def test_ensure_collection_is_idempotent(repo):
    repo.upsert_chunks(make_chunks(), [EAST, NORTH, WEST])
    repo.ensure_collection(vector_size=2)
    assert repo.count() == 3


@pytest.mark.unit
def test_batches_store_every_chunk(repo):
    repo.upsert_chunks(make_chunks(), [EAST, NORTH, WEST], batch_size=2)
    assert repo.count() == 3


@pytest.mark.unit
def test_mismatched_lengths_raise(repo):
    with pytest.raises(ValueError):
        repo.upsert_chunks(make_chunks(), [EAST])
