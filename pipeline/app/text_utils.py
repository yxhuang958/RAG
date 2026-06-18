from __future__ import annotations

from typing import List


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        chunks.append(text[start:end])
        if end == length:
            break
        start = max(end - overlap, 0)
    return chunks
