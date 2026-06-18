from __future__ import annotations

from .llm import LLMClient
from .prompt import build_prompt
from .rerank import Reranker
from .retriever import Retriever


class QAPipeline:
    def __init__(self) -> None:
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.llm = LLMClient()

    def answer(self, collection, messages: list[dict], top_k: int) -> dict:
        question = messages[-1]["content"]
        hits = self.retriever.retrieve(collection, question, top_k)
        reranked = self.reranker.rerank(question, hits)
        prompt = build_prompt(messages, reranked)
        answer = self.llm.generate(prompt)
        return {
            "answer": answer,
            "contexts": reranked,
        }
