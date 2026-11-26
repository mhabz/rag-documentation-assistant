"""Unit tests for the PDF extractor."""

import pymupdf

from app.ingestion.models import Page
from app.ingestion.pdf_extractor import extract_pages, remove_repeated_lines


def test_remove_repeated_header_and_page_number():
    bodies = ["Mandat", "Prélèvement", "Rejet", "Remboursement"]
    pages = [Page(n, f"EPC Rulebook\n{body}\nPage {n}") for n, body in enumerate(bodies, 1)]
    cleaned = remove_repeated_lines(pages)
    assert [p.text for p in cleaned] == bodies


def test_short_documents_are_left_untouched():
    pages = [Page(1, "En-tête\nA"), Page(2, "En-tête\nB")]
    assert remove_repeated_lines(pages) == pages


def test_extract_pages_from_generated_pdf(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    for text in ("Premier mandat SEPA", "Deuxième page"):
        doc.new_page().insert_text((72, 72), text)
    doc.save(pdf_path)
    doc.close()

    pages = extract_pages(pdf_path)
    assert [p.number for p in pages] == [1, 2]
    assert "mandat SEPA" in pages[0].text
