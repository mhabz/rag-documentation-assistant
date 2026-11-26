"""Unit tests for the embedding service, using a fake encoder (no model download)."""

import pytest

from app.embeddings.embedding_service import EmbeddingService


class FakeEncoder:
    """Returns a vector [len(text), 1.0] for each text and records the calls."""

    def __init__(self):
        self.calls = []

    def encode(self, sentences, **kwargs):
        self.calls.append((sentences, kwargs))
        return [[float(len(s)), 1.0] for s in sentences]


@pytest.mark.unit
def test_embed_texts_returns_one_vector_per_text_in_order():
    service = EmbeddingService("any", encoder=FakeEncoder())
    assert service.embed_texts(["ab", "abcd"]) == [[2.0, 1.0], [4.0, 1.0]]


@pytest.mark.unit
def test_embed_texts_asks_for_normalised_vectors():
    encoder = FakeEncoder()
    EmbeddingService("any", encoder=encoder).embed_texts(["x"])
    assert encoder.calls[0][1]["normalize_embeddings"] is True


@pytest.mark.unit
def test_embed_texts_of_nothing_does_not_call_the_model():
    encoder = FakeEncoder()
    assert EmbeddingService("any", encoder=encoder).embed_texts([]) == []
    assert encoder.calls == []


@pytest.mark.unit
def test_embed_query_returns_a_single_vector():
    assert EmbeddingService("any", encoder=FakeEncoder()).embed_query("abc") == [3.0, 1.0]
