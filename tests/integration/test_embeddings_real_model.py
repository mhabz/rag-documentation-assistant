"""Check that the real embedding model captures meaning, in English and in French."""

import pytest

from app.core.config import Settings
from app.embeddings.embedding_service import EmbeddingService


def dot(a: list[float], b: list[float]) -> float:
    """Dot product, equal to cosine similarity for normalised vectors."""
    return sum(x * y for x, y in zip(a, b, strict=True))


@pytest.fixture(scope="module")
def service() -> EmbeddingService:
    """Load the real model once for the whole module."""
    return EmbeddingService(Settings(_env_file=None).embedding_model)


@pytest.mark.integration
def test_french_question_is_closer_to_matching_english_passage(service):
    question = service.embed_query("Le débiteur peut-il demander un remboursement ?")
    related, unrelated = service.embed_texts(
        [
            "The debtor can ask for a refund of an authorised direct debit.",
            "The weather is sunny in Paris today.",
        ]
    )
    assert dot(question, related) > dot(question, unrelated) + 0.3
