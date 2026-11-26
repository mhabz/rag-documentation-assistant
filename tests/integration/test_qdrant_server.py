"""Check the repository against the real Qdrant server (docker compose up -d)."""

import uuid

import pytest
from qdrant_client import QdrantClient

from app.core.config import Settings
from app.ingestion.models import Chunk
from app.vectorstore.qdrant_repository import QdrantRepository


@pytest.fixture
def repo():
    """A repository on a temporary collection, deleted after the test."""
    client = QdrantClient(url=Settings(_env_file=None).qdrant_url)
    repository = QdrantRepository(client, f"test_{uuid.uuid4().hex}")
    repository.ensure_collection(vector_size=2)
    yield repository
    repository.delete_collection()


@pytest.mark.integration
def test_store_and_search_on_real_server(repo):
    chunks = [Chunk("east", 1, 0), Chunk("north", 2, 1)]
    repo.upsert_chunks(chunks, [[1.0, 0.0], [0.0, 1.0]])
    results = repo.search([0.0, 1.0], limit=1)
    assert (results[0].text, results[0].page) == ("north", 2)
