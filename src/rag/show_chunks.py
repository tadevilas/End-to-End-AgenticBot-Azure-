"""Show how a document in Blob Storage becomes pages and chunks (with the overlap marked).

Run from the project root:
    uv run python -X utf8 src/rag/show_chunks.py              # every file in the container
    uv run python -X utf8 src/rag/show_chunks.py Policy.pdf   # one file
    uv run python -X utf8 src/rag/show_chunks.py Policy.pdf 400 50   # custom chunk size / overlap
"""

import asyncio
import sys
import warnings
from difflib import SequenceMatcher
from pathlib import Path

# Lets this file run directly (python src/rag/show_chunks.py) by making the project root importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.config import get_settings  # noqa: E402
from src.rag.blob_source import create_blob_source  # noqa: E402
from src.rag.ingest import split_documents  # noqa: E402
from src.rag.loaders import load_file  # noqa: E402

warnings.filterwarnings("ignore")
BAR = "=" * 78


def shared_overlap(previous: str, current: str) -> str:
    """Text that ends the previous chunk and starts the current one ('' if none)."""
    m = SequenceMatcher(None, previous, current, autojunk=False).find_longest_match(
        0, len(previous), 0, len(current)
    )
    touches_ends = m.a + m.size == len(previous) and m.b == 0
    return previous[m.a : m.a + m.size] if touches_ends else ""


async def main(only_file: str | None, chunk_size: int, chunk_overlap: int) -> None:
    async for name, data in create_blob_source(get_settings()).iter_files():
        if only_file and name != only_file:
            continue
        pages = load_file(data, name)
        print(f"{BAR}\nFILE: {name}  ({len(data)} bytes)  -> {len(pages)} page(s)\n{BAR}")
        for page in pages:
            print(f"\n--- PAGE {page.metadata['page']} ({len(page.page_content)} chars) ---")
            print(page.page_content)

        chunks = split_documents(pages, chunk_size, chunk_overlap)
        print(f"\n{BAR}\nCHUNKS: size={chunk_size}, overlap={chunk_overlap} -> {len(chunks)} chunk(s)\n{BAR}")
        for i, chunk in enumerate(chunks):
            print(f"\n--- CHUNK {i + 1}  (page {chunk.metadata['page']}, {len(chunk.page_content)} chars) ---")
            print(chunk.page_content)
            if i == 0:
                continue
            if chunks[i - 1].metadata["page"] != chunk.metadata["page"]:
                print("\n   >> new page: chunks never overlap across pages")
                continue
            shared = shared_overlap(chunks[i - 1].page_content, chunk.page_content)
            note = f"OVERLAP with chunk {i} ({len(shared)} chars): {shared!r}" if shared else f"no shared text with chunk {i}"
            print(f"\n   >> {note}")


if __name__ == "__main__":
    args = sys.argv[1:]
    asyncio.run(
        main(
            only_file=args[0] if len(args) > 0 else None,
            chunk_size=int(args[1]) if len(args) > 1 else 800,
            chunk_overlap=int(args[2]) if len(args) > 2 else 100,
        )
    )
