#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import random
import sys
from typing import Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm


def load_sentences(path: str):
    sentences = []
    id_to_text = {}
    id_to_index = {}
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            sentence_id = row["sentence_id"]
            sentences.append(row)
            id_to_text[sentence_id] = row["text"]
            id_to_index[sentence_id] = idx
    return sentences, id_to_text, id_to_index


def extract_json(text: str):
    if not text:
        return None
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


def build_prompt(
    slice_sentences: List[dict],
    qas_per_run: int,
    min_context: int,
    max_context: int,
) -> str:
    lines = []
    for row in slice_sentences:
        lines.append(f"{row['sentence_id']}\t{row['text']}")
    slice_text = "\n".join(lines)
    return (
        "You are building a reading-comprehension dataset.\n"
        "You will receive a slice of sentences from a long document. Each line starts "
        "with a sentence ID, a tab, then the sentence text.\n\n"
        f"Generate {qas_per_run} question-answer pairs in JSON array format. "
        "Questions must require multi-sentence reasoning and an understanding of the "
        "overall slice. Avoid short factual questions, named-entity trivia, or single-sentence "
        "lookups.\n"
        "Each JSON item must include:\n"
        '- "question": string\n'
        '- "answer": string (2-4 sentences)\n'
        f'- "context_sentence_ids": array of {min_context}-{max_context} IDs drawn only '
        "from the provided slice\n"
        "Return JSON only, no extra text.\n\n"
        "Sentences:\n"
        f"{slice_text}"
    )


def validate_items(
    items: List[dict],
    slice_id_set: set,
    id_to_text: Dict[str, str],
    id_to_index: Dict[str, int],
    min_context: int,
    max_context: int,
) -> List[dict]:
    valid = []
    for item in items:
        question = item.get("question")
        answer = item.get("answer")
        ctx_ids = item.get("context_sentence_ids")
        if not question or not answer or not isinstance(ctx_ids, list):
            continue
        ctx_ids = [c for c in ctx_ids if isinstance(c, str)]
        if not (min_context <= len(ctx_ids) <= max_context):
            continue
        if any(c not in slice_id_set for c in ctx_ids):
            continue
        unique_ids = []
        seen = set()
        for cid in ctx_ids:
            if cid not in seen:
                unique_ids.append(cid)
                seen.add(cid)
        if not (min_context <= len(unique_ids) <= max_context):
            continue
        ctx_ids = unique_ids
        ctx_ids.sort(key=lambda x: id_to_index.get(x, -1))
        context_text = " ".join(id_to_text[cid] for cid in ctx_ids if cid in id_to_text)
        valid.append(
            {
                "question": question.strip(),
                "answer": answer.strip(),
                "context_sentence_ids": ctx_ids,
                "context_text": context_text.strip(),
            }
        )
    return valid


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate QA pairs from OpenAlex data.")
    parser.add_argument(
        "--input-root",
        default="data/holisqa",
        help="Root directory created by the downloader script.",
    )
    parser.add_argument(
        "--categories",
        default="all",
        help="Comma-separated category keys or 'all'.",
    )
    parser.add_argument("--runs", type=int, default=50, help="Number of slice runs.")
    parser.add_argument("--qas-per-run", type=int, default=10, help="QAs per run.")
    parser.add_argument(
        "--slice-ratio",
        type=float,
        default=0.2,
        help="Fraction of sentences to include in each slice.",
    )
    parser.add_argument(
        "--min-context-sentences", type=int, default=3, help="Min context sentences."
    )
    parser.add_argument(
        "--max-context-sentences", type=int, default=7, help="Max context sentences."
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash-lite",
        help="Model name for Gemini OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--base-url",
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
        help="Base URL for OpenAI-compatible Gemini API.",
    )
    parser.add_argument("--api-key", default=None, help="API key for Gemini.")
    parser.add_argument(
        "--max-retries", type=int, default=2, help="Max retries per run."
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get(
        "OPENAI_API_KEY"
    )
    if not api_key:
        raise SystemExit("Missing API key. Provide --api-key or set GEMINI_API_KEY.")

    if args.categories == "all":
        categories = [
            "computer_science",
            "medicine",
            "business",
            "biology",
            "psychology",
        ]
    else:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    rng = random.SystemRandom()

    for category in categories:
        category_dir = os.path.join(args.input_root, category)
        sentences_path = os.path.join(category_dir, "sentences.jsonl")
        if not os.path.exists(sentences_path):
            print(f"Missing sentences file: {sentences_path}", file=sys.stderr)
            continue

        sentences, id_to_text, id_to_index = load_sentences(sentences_path)
        if not sentences:
            print(f"No sentences found for {category}.", file=sys.stderr)
            continue

        qa_path = os.path.join(category_dir, "qa.jsonl")
        slices_path = os.path.join(category_dir, "slices.jsonl")

        qa_id_counter = 0
        slice_id_counter = 0

        slice_len = max(1, int(len(sentences) * args.slice_ratio))
        max_start = max(0, len(sentences) - slice_len)

        with open(qa_path, "w", encoding="utf-8") as qa_file, open(
            slices_path, "w", encoding="utf-8"
        ) as slices_file:
            for run_idx in tqdm(
                range(args.runs),
                desc=category,
                unit="slice",
                dynamic_ncols=True,
            ):
                start_idx = rng.randint(0, max_start) if max_start > 0 else 0
                end_idx = start_idx + slice_len
                slice_sentences = sentences[start_idx:end_idx]
                slice_id_counter += 1
                slice_id = f"SL{slice_id_counter:06d}"

                slice_id_set = {row["sentence_id"] for row in slice_sentences}
                prompt = build_prompt(
                    slice_sentences,
                    args.qas_per_run,
                    args.min_context_sentences,
                    args.max_context_sentences,
                )

                attempts = 0
                valid_items: List[dict] = []
                while attempts <= args.max_retries and len(valid_items) < args.qas_per_run:
                    attempts += 1
                    response = client.chat.completions.create(
                        model=args.model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    content = response.choices[0].message.content or ""
                    parsed = extract_json(content)
                    if not isinstance(parsed, list):
                        continue
                    valid_items = validate_items(
                        parsed,
                        slice_id_set,
                        id_to_text,
                        id_to_index,
                        args.min_context_sentences,
                        args.max_context_sentences,
                    )

                if len(valid_items) < args.qas_per_run:
                    print(
                        f"{category} run {run_idx + 1}: only {len(valid_items)} valid QAs.",
                        file=sys.stderr,
                    )

                for item in valid_items[: args.qas_per_run]:
                    qa_id_counter += 1
                    row = {
                        "qa_id": f"QA{qa_id_counter:06d}",
                        "category": category,
                        "slice_id": slice_id,
                        "question": item["question"],
                        "answer": item["answer"],
                        "context_sentence_ids": item["context_sentence_ids"],
                        "context_text": item["context_text"],
                        "created_at": dt.datetime.utcnow().isoformat() + "Z",
                    }
                    qa_file.write(json.dumps(row, ensure_ascii=True) + "\n")

                slice_row = {
                    "slice_id": slice_id,
                    "category": category,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "sentence_count": len(slice_sentences),
                    "start_sentence_id": slice_sentences[0]["sentence_id"],
                    "end_sentence_id": slice_sentences[-1]["sentence_id"],
                    "slice_ratio": args.slice_ratio,
                    "created_at": dt.datetime.utcnow().isoformat() + "Z",
                }
                slices_file.write(json.dumps(slice_row, ensure_ascii=True) + "\n")

                print(
                    f"{category} run {run_idx + 1}/{args.runs}: "
                    f"{len(valid_items)} QAs, slice {slice_id}."
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
