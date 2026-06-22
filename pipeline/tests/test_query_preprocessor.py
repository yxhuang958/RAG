from app.query_preprocessor import preprocess_query


def test_explicit_question_marker_splits_temporary_context() -> None:
    text = """背景:
Milvus 的 collection 已经有文档 A 和文档 B。
用户还贴了一段很长的临时材料, 这段材料不应该整体拿去做检索。

问题: 文档 A 的核心结论是什么?"""

    result = preprocess_query(text)

    assert result.was_split
    assert result.strategy == "explicit_question_marker"
    assert result.question == "文档 A 的核心结论是什么?"
    assert "不应该整体拿去做检索" in result.temporary_context
    assert result.retrieval_query.startswith("文档 A 的核心结论是什么?")


def test_long_query_uses_last_question_paragraph() -> None:
    long_context = "这是一段临时上下文。" * 120
    text = f"{long_context}\n\n请问应该如何优化检索 query?"

    result = preprocess_query(text, long_threshold_chars=200)

    assert result.was_split
    assert result.strategy == "last_question_paragraph"
    assert result.question == "请问应该如何优化检索 query?"
    assert long_context[:20] in result.temporary_context


def test_short_query_is_not_split() -> None:
    result = preprocess_query("输入过长的时候是否也需要 chunk?")

    assert not result.was_split
    assert result.strategy == "short_query"
    assert result.question == "输入过长的时候是否也需要 chunk?"
    assert result.temporary_context == ""
    assert result.retrieval_query == result.question


def test_temporary_context_is_bounded_for_retrieval_and_prompt() -> None:
    text = "背景: " + ("abcdef" * 1000) + "\n问题: 结论是什么?"

    result = preprocess_query(text, retrieval_context_chars=80, prompt_context_chars=200)

    assert len(result.temporary_context) <= 205
    assert len(result.retrieval_query) < 140
    assert "Temporary context hints:" in result.retrieval_query
