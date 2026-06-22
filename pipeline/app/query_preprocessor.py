from __future__ import annotations

from dataclasses import dataclass
import re


_QUESTION_MARK_RE = re.compile(r"[?？]")
_QUESTION_CUE_RE = re.compile(
    r"(?i)\b(what|who|when|where|why|how|which|can|could|should|is|are|do|does|did)\b|"
    r"(什么|谁|哪|哪里|如何|怎么|为什么|是否|能否|请问|问题|区别|多少)"
)
_EXPLICIT_QUESTION_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?"
    r"(?:question|q|问题|我的问题|用户问题|请问)\s*[:：]\s*"
)
_CONTEXT_MARKER_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?"
    r"(?:context|background|reference|材料|背景|上下文|参考|临时上下文)\s*[:：]\s*"
)


@dataclass(frozen=True)
class ProcessedQuery:
    original: str
    question: str
    temporary_context: str
    retrieval_query: str
    was_split: bool
    strategy: str


def preprocess_query(
    text: str,
    *,
    long_threshold_chars: int = 1200,
    retrieval_context_chars: int = 400,
    prompt_context_chars: int = 6000,
) -> ProcessedQuery:
    original = text.strip()
    if not original:
        return ProcessedQuery("", "", "", "", False, "empty")

    question, temporary_context, strategy = _split_query(original, long_threshold_chars)
    question = question.strip() or original
    temporary_context = _clean_temporary_context(temporary_context, question)
    temporary_context = _middle_truncate(temporary_context, max(0, prompt_context_chars))
    retrieval_query = _build_retrieval_query(
        question,
        temporary_context,
        max(0, retrieval_context_chars),
    )

    return ProcessedQuery(
        original=original,
        question=question,
        temporary_context=temporary_context,
        retrieval_query=retrieval_query,
        was_split=bool(temporary_context and question != original),
        strategy=strategy,
    )


def _split_query(text: str, long_threshold_chars: int) -> tuple[str, str, str]:
    explicit = list(_EXPLICIT_QUESTION_RE.finditer(text))
    if explicit:
        marker = explicit[-1]
        question = text[marker.end() :].strip()
        if question:
            return question, text[: marker.start()].strip(), "explicit_question_marker"

    if len(text) < long_threshold_chars:
        return text, "", "short_query"

    paragraphs = [item.strip() for item in re.split(r"\n{2,}", text) if item.strip()]
    for index in range(len(paragraphs) - 1, -1, -1):
        paragraph = paragraphs[index]
        if _looks_like_question(paragraph):
            temporary_context = "\n\n".join(paragraphs[:index] + paragraphs[index + 1 :])
            return paragraph, temporary_context, "last_question_paragraph"

    lines = [item.strip() for item in text.splitlines() if item.strip()]
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index]
        if _looks_like_question(line):
            temporary_context = "\n".join(lines[:index] + lines[index + 1 :])
            return line, temporary_context, "last_question_line"

    sentences = _split_sentences(text)
    for index in range(len(sentences) - 1, -1, -1):
        sentence = sentences[index].strip()
        if _looks_like_question(sentence):
            temporary_context = " ".join(sentences[:index] + sentences[index + 1 :])
            return sentence, temporary_context, "last_question_sentence"

    return text, "", "long_query_no_split"


def _remove_context_marker_prefix(text: str) -> str:
    return _CONTEXT_MARKER_RE.sub("", text, count=1).strip()


def _looks_like_question(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return bool(_QUESTION_MARK_RE.search(stripped) or _QUESTION_CUE_RE.search(stripped[:160]))


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[?？.!。])\s+", text)
    return [part for part in parts if part.strip()]


def _clean_temporary_context(context: str, question: str) -> str:
    context = _remove_context_marker_prefix(context)
    if context.strip() == question.strip():
        return ""
    return context.strip()


def _build_retrieval_query(question: str, temporary_context: str, context_chars: int) -> str:
    question = question.strip()
    if not temporary_context or context_chars <= 0:
        return question

    hints = _edge_excerpt(temporary_context, context_chars)
    if not hints:
        return question
    return f"{question}\nTemporary context hints:\n{hints}"


def _edge_excerpt(text: str, limit: int) -> str:
    text = _collapse_blank_lines(text)
    if len(text) <= limit:
        return text
    head = max(1, limit // 2)
    tail = max(1, limit - head)
    return f"{text[:head].rstrip()}\n...\n{text[-tail:].lstrip()}"


def _middle_truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return _edge_excerpt(text, limit)


def _collapse_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.strip())
