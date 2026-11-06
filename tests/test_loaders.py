"""Tests for document loaders."""

import pytest
from src.rag.loaders import is_supported, load_file, load_text, load_html


def test_supported_extensions():
    assert is_supported("document.pdf") is True
    assert is_supported("document.docx") is True
    assert is_supported("slides.pptx") is True
    assert is_supported("notes.txt") is True
    assert is_supported("page.html") is True
    assert is_supported("image.png") is False
    assert is_supported("archive.zip") is False


def test_load_text_file():
    content = b"This is a sample document for RAG ingestion."
    docs = load_text(content, "sample.txt")
    assert len(docs) == 1
    assert docs[0].page_content == "This is a sample document for RAG ingestion."
    assert docs[0].metadata["source"] == "sample.txt"
    assert docs[0].metadata["title"] == "sample"
    assert docs[0].metadata["page"] == 1


def test_load_html_file():
    html_content = b"""
    <html>
        <head><title>Test Page</title><style>.hidden{display:none;}</style></head>
        <body>
            <h1>Enterprise Architecture</h1>
            <p>LangGraph Agentic RAG system.</p>
            <script>console.log('strip me');</script>
        </body>
    </html>
    """
    docs = load_html(html_content, "page.html")
    assert len(docs) == 1
    assert "Enterprise Architecture" in docs[0].page_content
    assert "LangGraph Agentic RAG system." in docs[0].page_content
    assert "console.log" not in docs[0].page_content


def test_load_unsupported_file_raises():
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_file(b"data", "file.mp4")
