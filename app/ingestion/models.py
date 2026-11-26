"""Data structures passed between the ingestion steps."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    """A page of text extracted from a document.

    Attributes:
        number: Page number, starting at 1 as in a PDF viewer.
        text: Text content of the page.
    """

    number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    """A piece of text small enough to be indexed and retrieved on its own.

    Attributes:
        text: Text content of the chunk.
        page: Number of the page the chunk comes from, used to cite sources.
        index: Position of the chunk in the whole document, starting at 0.
    """

    text: str
    page: int
    index: int
