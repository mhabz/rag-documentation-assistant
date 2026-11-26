"""Extract the text of a PDF, page by page, and remove repeated headers and footers."""

from collections import Counter
from pathlib import Path

import pymupdf

from app.ingestion.models import Page


def extract_pages(pdf_path: Path) -> list[Page]:
    """Read a text-based PDF and return one ``Page`` per PDF page.

    Scanned PDFs (images) are not supported: there is no OCR.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        The pages in order, with repeated headers and footers removed.

    Raises:
        pymupdf.FileNotFoundError: If ``pdf_path`` does not exist.
        pymupdf.FileDataError: If the file is not a valid PDF.
    """
    with pymupdf.open(pdf_path) as doc:
        pages = [Page(number=i + 1, text=p.get_text()) for i, p in enumerate(doc)]
    return remove_repeated_lines(pages)


def remove_repeated_lines(pages: list[Page], min_ratio: float = 0.5) -> list[Page]:
    """Drop the lines that appear on many pages, such as headers and footers.

    Digits are replaced before counting, so "Page 3" and "Page 4" count as the same line.
    Known limit: body lines that differ only by a digit are removed too.

    Args:
        pages: Pages to clean.
        min_ratio: Minimum share of pages (between 0 and 1) on which a line must appear
            to be considered a header or footer.

    Returns:
        New pages without the repeated lines. Documents of fewer than 3 pages are
        returned unchanged, because repetition cannot be measured reliably.
    """
    if len(pages) < 3:
        return pages

    def key(line: str) -> str:
        return "".join("#" if c.isdigit() else c for c in line.strip())

    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({key(line) for line in page.text.splitlines() if line.strip()})

    threshold = len(pages) * min_ratio
    repeated = {k for k, n in counts.items() if n >= threshold}
    return [
        Page(
            number=page.number,
            text="\n".join(line for line in page.text.splitlines() if key(line) not in repeated),
        )
        for page in pages
    ]
