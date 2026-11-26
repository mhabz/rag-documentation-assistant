"""Unit tests for the chunker."""

import itertools

import pytest

from app.ingestion.chunker import chunk_pages, chunk_text
from app.ingestion.models import Page


@pytest.mark.unit
def test_short_text_gives_single_chunk():
    assert chunk_text("hello world", chunk_size=100, overlap=10) == ["hello world"]


@pytest.mark.unit
def test_empty_text_gives_no_chunk():
    assert chunk_text("   ") == []


@pytest.mark.unit
def test_chunks_respect_max_size():
    text = "word " * 500
    assert all(len(c) <= 100 for c in chunk_text(text, chunk_size=100, overlap=20))


@pytest.mark.unit
def test_consecutive_chunks_overlap():
    text = " ".join(f"w{i}" for i in range(200))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    for a, b in itertools.pairwise(chunks):
        assert b.split()[0] in a.split()


@pytest.mark.unit
def test_no_content_lost():
    words = [f"w{i}" for i in range(200)]
    chunks = chunk_text(" ".join(words), chunk_size=100, overlap=20)
    assert set(" ".join(chunks).split()) == set(words)


@pytest.mark.unit
def test_text_without_spaces_is_cut_at_chunk_size():
    assert len(chunk_text("a" * 1000, chunk_size=300, overlap=0)) == 4


@pytest.mark.unit
@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_params_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=size, overlap=overlap)


@pytest.mark.unit
def test_chunk_pages_keeps_page_number():
    chunks = chunk_pages([Page(number=7, text="Un seul paragraphe.")])
    assert len(chunks) == 1
    assert chunks[0].page == 7


@pytest.mark.unit
def test_chunk_pages_keeps_their_own_page_and_global_index():
    pages = [Page(number=1, text="Page un."), Page(number=2, text="Page deux.")]
    chunks = chunk_pages(pages)
    assert [(c.page, c.index) for c in chunks] == [(1, 0), (2, 1)]


@pytest.mark.unit
def test_chunk_pages_never_mixes_pages():
    pages = [Page(number=1, text="alpha " * 100), Page(number=2, text="beta " * 100)]
    for chunk in chunk_pages(pages, chunk_size=100, overlap=20):
        assert ("alpha" in chunk.text) != ("beta" in chunk.text)
        assert chunk.page == (1 if "alpha" in chunk.text else 2)
