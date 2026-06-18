from __future__ import annotations

import re
import unicodedata
from typing import List


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_REPEATED_PUNCT = re.compile(r"([._\-*=·•])\1{2,}")
_PAGE_MARKER = re.compile(r"^\s*(?:page\s*)?\d+\s*(?:/|of|-)?\s*\d*\s*$", re.IGNORECASE)
_TOC_DOT_LINE = re.compile(r".*([._\-·])\1{2,}\s*\d+\s*$")
_BULLET_PREFIX = re.compile(r"^\s*[•·▪■◆◇▶►●○◦*]+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?，。；：！？、）】》])")
_SPACE_AFTER_OPEN = re.compile(r"([（【《])\s+")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_CJK_SPLIT = re.compile(r"(?<=[。！？；])")
_LATIN_SPLIT = re.compile(r"(?<=[.!?;])\s+")


def clean_pdf_text(text: str) -> str:
    """Normalize common PDF extraction noise before chunking."""
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = text.replace("\u2010", "-").replace("\u2011", "-")
    text = _CONTROL_CHARS.sub("", text)

    cleaned_lines: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            continue

        if _TOC_DOT_LINE.match(line):
            continue

        line = _REPEATED_PUNCT.sub(" ", line)
        line = _BULLET_PREFIX.sub("", line).strip()
        line = line.replace("**", "")
        line = _SPACE_BEFORE_PUNCT.sub(r"\1", line)
        line = _SPACE_AFTER_OPEN.sub(r"\1", line)
        line = _MULTI_SPACE.sub(" ", line)

        if _is_noise_line(line):
            continue
        cleaned_lines.append(line)

    paragraphs = _merge_wrapped_lines(cleaned_lines)
    return "\n\n".join(paragraphs).strip()


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    text = clean_pdf_text(text)
    if not text:
        return []

    size = max(size, 100)
    overlap = max(0, min(overlap, size // 2))

    units = _split_units(text, size)
    chunks: List[str] = []
    current = ""

    for unit in units:
        if not unit:
            continue

        if len(unit) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_split(unit, size, overlap))
            continue

        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if len(candidate) <= size:
            current = candidate
            continue

        if current:
            chunks.append(current)
        current = _tail_overlap(chunks[-1], overlap) + "\n\n" + unit if chunks and overlap else unit
        if len(current) > size:
            chunks.extend(_hard_split(current, size, overlap))
            current = ""

    if current:
        chunks.append(current)

    return [_compact_chunk(chunk) for chunk in chunks if _compact_chunk(chunk)]


def _is_noise_line(line: str) -> bool:
    if _PAGE_MARKER.match(line):
        return True

    meaningful = sum(ch.isalnum() for ch in line)
    if meaningful == 0:
        return True

    symbol_ratio = 1 - meaningful / max(len(line), 1)
    return len(line) <= 3 and meaningful <= 1 or symbol_ratio > 0.8


def _merge_wrapped_lines(lines: List[str]) -> List[str]:
    paragraphs: List[str] = []
    current = ""

    for line in lines:
        if not line:
            if current:
                paragraphs.append(current.strip())
                current = ""
            continue

        if not current:
            current = line
            continue

        if _should_join_without_space(current, line):
            current += line
        elif _looks_like_wrapped_line(current, line):
            current += " " + line
        else:
            paragraphs.append(current.strip())
            current = line

    if current:
        paragraphs.append(current.strip())

    return [paragraph for paragraph in paragraphs if paragraph]


def _should_join_without_space(previous: str, current: str) -> bool:
    return bool(previous and current and _is_cjk(previous[-1]) and _is_cjk(current[0]))


def _looks_like_wrapped_line(previous: str, current: str) -> bool:
    if previous.endswith((".", "!", "?", ";", "。", "！", "？", "；", "：", ":")):
        return False
    if re.match(r"^\d+[\).、]\s*", current):
        return False
    if re.match(r"^[A-Z][A-Z\s]{4,}$", current):
        return False
    return True


def _split_units(text: str, size: int) -> List[str]:
    units: List[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= size:
            units.append(paragraph)
        else:
            units.extend(_split_sentences(paragraph))
    return units


def _split_sentences(text: str) -> List[str]:
    pieces: List[str] = []
    for cjk_part in _CJK_SPLIT.split(text):
        cjk_part = cjk_part.strip()
        if not cjk_part:
            continue
        pieces.extend(piece.strip() for piece in _LATIN_SPLIT.split(cjk_part) if piece.strip())
    return pieces or [text]


def _hard_split(text: str, size: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + size, length)
        chunks.append(text[start:end].strip())
        if end == length:
            break
        start = max(end - overlap, start + 1)

    return chunks


def _tail_overlap(text: str, overlap: int) -> str:
    if overlap <= 0:
        return ""

    tail = text[-overlap:].strip()
    if tail and tail[0].isalnum() and len(text) > len(tail):
        previous = text[-len(tail) - 1]
        if previous.isalnum():
            boundary = re.search(r"\s+", tail)
            if boundary:
                tail = tail[boundary.end():].strip()

    return tail


def _compact_chunk(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    return _MULTI_SPACE.sub(" ", text)


def _is_cjk(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"
