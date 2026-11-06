"""Turns files (PDF, DOCX, PPTX, TXT, HTML) into LangChain Documents.

Each Document carries metadata: title, source (file name), and page (page or slide number).
"""

import io
from collections.abc import Callable
from pathlib import PurePath

from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from langchain_core.documents import Document
from pptx import Presentation
from pypdf import PdfReader


def _meta(file_name: str, page: int) -> dict:
    return {"title": PurePath(file_name).stem, "source": file_name, "page": page}


def load_pdf(data: bytes, file_name: str) -> list[Document]:
    reader = PdfReader(io.BytesIO(data))
    documents = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            documents.append(Document(page_content=text, metadata=_meta(file_name, number)))
    return documents


def load_docx(data: bytes, file_name: str) -> list[Document]:
    doc = DocxDocument(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    text = "\n".join(parts).strip()
    return [Document(page_content=text, metadata=_meta(file_name, 1))] if text else []


def load_pptx(data: bytes, file_name: str) -> list[Document]:
    documents = []
    for number, slide in enumerate(Presentation(io.BytesIO(data)).slides, start=1):
        lines = [
            paragraph.text
            for shape in slide.shapes
            if shape.has_text_frame
            for paragraph in shape.text_frame.paragraphs
            if paragraph.text.strip()
        ]
        if lines:
            documents.append(Document(page_content="\n".join(lines), metadata=_meta(file_name, number)))
    return documents


def load_text(data: bytes, file_name: str) -> list[Document]:
    text = data.decode("utf-8", errors="replace").strip()
    return [Document(page_content=text, metadata=_meta(file_name, 1))] if text else []


def load_html(data: bytes, file_name: str) -> list[Document]:
    soup = BeautifulSoup(data, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return [Document(page_content=text, metadata=_meta(file_name, 1))] if text else []


LOADERS: dict[str, Callable[[bytes, str], list[Document]]] = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".pptx": load_pptx,
    ".txt": load_text,
    ".html": load_html,
    ".htm": load_html,
}


def is_supported(file_name: str) -> bool:
    return PurePath(file_name).suffix.lower() in LOADERS


def load_file(data: bytes, file_name: str) -> list[Document]:
    """Pick a loader by file extension. Raises ValueError for unsupported types."""
    suffix = PurePath(file_name).suffix.lower()
    if suffix not in LOADERS:
        raise ValueError(f"Unsupported file type: {suffix or file_name}")
    return LOADERS[suffix](data, file_name)
