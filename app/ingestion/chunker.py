"""Split text into overlapping chunks, keeping track of the source page."""

from app.ingestion.models import Chunk, Page


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into chunks of at most ``chunk_size`` characters.

    Consecutive chunks share about ``overlap`` characters, so a sentence cut at a boundary
    still appears whole in one of the two chunks. Cuts prefer whitespace, so words are
    not split unless the text contains no space.

    Args:
        text: Text to split. Leading and trailing whitespace is ignored.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters shared by two consecutive chunks. Must be at least 0
            and smaller than ``chunk_size``.

    Returns:
        The chunks in order. An empty list if ``text`` is blank.

    Raises:
        ValueError: If ``chunk_size`` is not positive, or if ``overlap`` is negative or
            not smaller than ``chunk_size``.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = text.strip()
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + overlap + 1, end)
            if cut != -1:
                end = cut
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
        if text[start - 1] != " " and text[start] != " ":
            space = text.find(" ", start, end)
            if space != -1:
                start = space + 1
    return chunks


def chunk_pages(pages: list[Page], chunk_size: int = 800, overlap: int = 100) -> list[Chunk]:
    """Chunk each page with ``chunk_text``, keeping the source page number.

    A chunk never spans two pages, so every chunk can be cited with its page.

    Args:
        pages: Pages to split.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters shared by two consecutive chunks.

    Returns:
        The chunks of all pages, numbered from 0 across the whole document.

    Raises:
        ValueError: If ``chunk_size`` or ``overlap`` is invalid, see ``chunk_text``.
    """
    chunks: list[Chunk] = []
    for page in pages:
        for text in chunk_text(page.text, chunk_size, overlap):
            chunks.append(Chunk(text=text, page=page.number, index=len(chunks)))
    return chunks
