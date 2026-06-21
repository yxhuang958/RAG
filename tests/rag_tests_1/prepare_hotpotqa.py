#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import textwrap
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
QA_PATH = ROOT / "qa_eval" / "hotpotqa_long_multihop.json"
DEFAULT_DATASET_NAME = "hotpotqa/hotpot_qa"
DEFAULT_DATASET_CONFIG = "distractor"
DEFAULT_DATASET_SPLIT = "validation"


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare long-document multi-hop hallucination eval data.")
    parser.add_argument("--limit", type=int, default=80, help="Number of HotpotQA examples to keep.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--min-doc-words", type=int, default=450, help="Pad PDFs to at least this many words.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT)
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=2000,
        help="Load at most this many examples from the split before filtering/sampling. Use 0 for the full split.",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--qa-path", type=Path, default=QA_PATH)
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.qa_path.parent.mkdir(parents=True, exist_ok=True)

    examples = load_hotpotqa(args.dataset_name, args.dataset_config, args.dataset_split, args.scan_limit)
    selected = select_examples(examples, args.limit, args.seed)
    qa_items, docs = build_eval_set(selected)
    write_pdfs(docs, args.data_dir, args.min_doc_words)
    args.qa_path.write_text(json.dumps(qa_items, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Prepared {len(qa_items)} questions")
    print(f"Wrote {len(docs)} PDFs to {args.data_dir}")
    print(f"Wrote qa_eval to {args.qa_path}")
    return 0


def load_hotpotqa(
    dataset_name: str,
    dataset_config: str,
    dataset_split: str,
    scan_limit: int,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing datasets. Install with: python -m pip install -r tests/rag_tests_1/requirements.txt"
        ) from exc

    split = dataset_split if scan_limit <= 0 else f"{dataset_split}[:{scan_limit}]"
    print(f"Loading dataset {dataset_name}/{dataset_config} split={split}")
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    return [normalize_hotpot_item(dict(item)) for item in dataset]


def normalize_hotpot_item(item: dict[str, Any]) -> dict[str, Any]:
    context = item.get("context") or []
    if isinstance(context, dict):
        titles = context.get("title") or []
        sentences_list = context.get("sentences") or []
        context = list(zip(titles, sentences_list))

    supporting_facts = item.get("supporting_facts") or []
    if isinstance(supporting_facts, dict):
        titles = supporting_facts.get("title") or []
        sent_ids = supporting_facts.get("sent_id") or []
        supporting_facts = list(zip(titles, sent_ids))

    return {
        "_id": item.get("id") or item.get("_id"),
        "type": item.get("type", "multi-hop"),
        "level": item.get("level", "unknown"),
        "question": item.get("question"),
        "answer": item.get("answer"),
        "context": context,
        "supporting_facts": supporting_facts,
    }


def select_examples(examples: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    candidates = []
    for item in examples:
        support_titles = {title for title, _ in item.get("supporting_facts", [])}
        if len(support_titles) < 2:
            continue
        if not item.get("question") or not item.get("answer"):
            continue
        candidates.append(item)

    random.Random(seed).shuffle(candidates)
    return candidates[:limit]


def build_eval_set(examples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    docs: dict[str, dict[str, Any]] = {}
    qa_items = []

    for index, item in enumerate(examples, start=1):
        context_by_title = {title: sentences for title, sentences in item.get("context", [])}
        supporting_facts = [
            {"title": title, "sent_id": sent_id, "sentence": context_by_title.get(title, [""])[sent_id]}
            for title, sent_id in item.get("supporting_facts", [])
            if title in context_by_title and 0 <= sent_id < len(context_by_title[title])
        ]
        support_titles = sorted({fact["title"] for fact in supporting_facts})
        if len(support_titles) < 2:
            continue

        for title, sentences in context_by_title.items():
            key = doc_key(title)
            docs.setdefault(
                key,
                {
                    "title": title,
                    "filename": f"hotpot__{safe_filename(title)}.pdf",
                    "sentences": [],
                    "distractor_sentences": [],
                },
            )
            docs[key]["sentences"].extend(sentences)

        distractors = [
            sentence
            for title, sentences in context_by_title.items()
            if title not in support_titles
            for sentence in sentences
        ]
        for title in support_titles:
            docs[doc_key(title)]["distractor_sentences"].extend(distractors[:30])

        qa_items.append(
            {
                "id": item.get("_id") or f"HPQA_{index:04d}",
                "type": item.get("type", "multi-hop"),
                "difficulty": item.get("level", "unknown"),
                "question": item["question"],
                "reference_answer": item["answer"],
                "source_files": [f"hotpot/{safe_filename(title)}.md" for title in support_titles],
                "supporting_facts": supporting_facts,
                "dataset": "HotpotQA distractor dev",
            }
        )

    return qa_items, docs


def write_pdfs(docs: dict[str, dict[str, Any]], data_dir: Path, min_words: int) -> None:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError as exc:
        raise RuntimeError(
            "Missing reportlab. Install with: python -m pip install -r tests/rag_tests_1/requirements.txt"
        ) from exc

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    body_style = styles["BodyText"]

    for doc in docs.values():
        path = data_dir / doc["filename"]
        story = [Paragraph(escape_html(doc["title"]), title_style), Spacer(1, 12)]

        unique_sentences = dedupe(doc["sentences"])
        text = " ".join(unique_sentences)
        filler = dedupe(doc["distractor_sentences"])
        while len(text.split()) < min_words and filler:
            text += " " + " ".join(filler[:20])
            filler = filler[20:] + filler[:20]
            if len(filler) <= 20:
                break

        for paragraph in split_paragraphs(text):
            story.append(Paragraph(escape_html(paragraph), body_style))
            story.append(Spacer(1, 8))

        pdf = SimpleDocTemplate(str(path), pagesize=letter)
        pdf.build(story)


def dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        clean = " ".join(str(item).split())
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


def split_paragraphs(text: str) -> list[str]:
    return textwrap.wrap(text, width=1200, break_long_words=False, replace_whitespace=False) or [text]


def safe_filename(title: str) -> str:
    value = re.sub(r"[^0-9A-Za-z]+", "_", title).strip("_").lower()
    return value[:120] or "untitled"


def doc_key(title: str) -> str:
    return safe_filename(title)


def escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    raise SystemExit(main())
