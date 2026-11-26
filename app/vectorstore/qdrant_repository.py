"""Store chunks and their embeddings in Qdrant, and search the closest ones."""

from itertools import batched

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.ingestion.models import Chunk
from app.vectorstore.models import SearchResult


class QdrantRepository:
    """Read and write the chunks of one Qdrant collection.

    Each chunk becomes a Qdrant point: its id is the chunk index, its vector is the embedding,
    and its payload holds the text and the page. Using the chunk index as id makes ingestion
    idempotent: ingesting the same document twice replaces the points instead of duplicating them.
    """

    def __init__(self, client: QdrantClient, collection: str) -> None:
        """Create the repository.

        Args:
            client: Qdrant client. Use ``QdrantClient(":memory:")`` in tests.
            collection: Name of the collection to work with.
        """
        self._client = client
        self._collection = collection

    def ensure_collection(self, vector_size: int) -> None:
        """Create the collection if it does not exist yet.

        Args:
            vector_size: Number of values in each embedding (384 for our model).
        """
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def upsert_chunks(
        self, chunks: list[Chunk], vectors: list[list[float]], batch_size: int = 100
    ) -> None:
        """Insert or replace chunks with their embeddings, by batches.

        Args:
            chunks: Chunks to store.
            vectors: Embedding of each chunk, in the same order as ``chunks``.
            batch_size: Number of points sent to Qdrant per request.

        Raises:
            ValueError: If ``chunks`` and ``vectors`` do not have the same length.
        """
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        points = [
            PointStruct(
                id=chunk.index,
                vector=vector,
                payload={"text": chunk.text, "page": chunk.page, "index": chunk.index},
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        for batch in batched(points, batch_size):
            self._client.upsert(collection_name=self._collection, points=list(batch))

    def search(self, vector: list[float], limit: int = 5) -> list[SearchResult]:
        """Find the chunks whose embedding is the closest to ``vector``.

        Args:
            vector: Embedding of the query.
            limit: Maximum number of chunks to return (the "k" of top-k).

        Returns:
            The closest chunks, most similar first.
        """
        response = self._client.query_points(
            collection_name=self._collection, query=vector, limit=limit, with_payload=True
        )
        return [
            SearchResult(
                text=point.payload["text"],
                page=point.payload["page"],
                index=point.payload["index"],
                score=point.score,
            )
            for point in response.points
        ]

    def count(self) -> int:
        """Return the number of chunks stored in the collection."""
        return self._client.count(collection_name=self._collection, exact=True).count

    def delete_collection(self) -> None:
        """Delete the collection and everything it contains, if it exists."""
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
