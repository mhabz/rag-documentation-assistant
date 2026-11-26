"""Turn texts into embedding vectors with a local sentence-transformers model."""

from typing import Protocol

from sentence_transformers import SentenceTransformer


class _Encoder(Protocol):
    """Minimal interface of an embedding model, so tests can replace the real one."""

    def encode(self, sentences: list[str], **kwargs) -> list[list[float]]:
        """Encode sentences into vectors."""


class EmbeddingService:
    """Compute normalised embeddings for documents and queries.

    The model is loaded on first use, because loading takes several seconds.
    Vectors are normalised to length 1, so cosine similarity equals a dot product.
    """

    def __init__(self, model_name: str, encoder: _Encoder | None = None) -> None:
        """Create the service.

        Args:
            model_name: Name of the sentence-transformers model to load.
            encoder: Optional ready-made encoder, mainly to inject a fake one in tests.
                When given, no model is downloaded or loaded.
        """
        self._model_name = model_name
        self._encoder = encoder

    @property
    def encoder(self) -> _Encoder:
        """The underlying model, loaded lazily."""
        if self._encoder is None:
            self._encoder = SentenceTransformer(self._model_name)
        return self._encoder

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed several texts at once.

        Args:
            texts: Texts to embed, for example the chunks of a document.

        Returns:
            One vector per text, in the same order. An empty list if ``texts`` is empty.
        """
        if not texts:
            return []
        vectors = self.encoder.encode(texts, normalize_embeddings=True)
        return [[float(x) for x in vector] for vector in vectors]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query.

        Args:
            query: The user's question.

        Returns:
            The vector of the query.
        """
        return self.embed_texts([query])[0]
