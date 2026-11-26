"""Data structures returned by the vector store."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    """A chunk found by a similarity search.

    Attributes:
        text: Text content of the chunk.
        page: Number of the page the chunk comes from.
        index: Position of the chunk in the whole document.
        score: Cosine similarity with the query, between -1 and 1 (higher is closer).
    """

    text: str
    page: int
    index: int
    score: float
