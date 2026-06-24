#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import re
import sys
from typing import Dict, Iterable, List, Optional

import requests
from tqdm import tqdm


CATEGORIES: Dict[str, str] = {
    "computer_science": "C41008148",
    "medicine": "C71924100",
    "business": "C144133560",
    "biology": "C86803240",
    "psychology": "C15744967",
}


def reconstruct_abstract(inv_index: Dict[str, List[int]]) -> str:
    if not inv_index:
        return ""
    max_pos = -1
    for positions in inv_index.values():
        if positions:
            max_pos = max(max_pos, max(positions))
    if max_pos < 0:
        return ""
    words = [""] * (max_pos + 1)
    for word, positions in inv_index.items():
        for pos in positions:
            if 0 <= pos <= max_pos:
                words[pos] = word
    return " ".join(w for w in words if w)


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in pieces if p and p.strip()]


def fetch_with_cursor(
    concept_id: str,
    year: int,
    per_category: int,
    to_date: Optional[str],
    progress_label: str,
) -> List[dict]:
    cursor = "*"
    collected: List[dict] = []
    seen_ids = set()
    headers = {"Accept-Encoding": "gzip"}

    with tqdm(
        total=per_category,
        desc=progress_label,
        unit="abs",
        dynamic_ncols=True,
    ) as pbar:
        while cursor and len(collected) < per_category:
            filters = [
                f"concepts.id:{concept_id}",
                "has_abstract:true",
                f"publication_year:{year}",
            ]
            if to_date:
                filters.append(f"to_publication_date:{to_date}")

            params = {
                "filter": ",".join(filters),
                "sort": "publication_date:desc",
                "per-page": 200,
                "cursor": cursor,
            }
            resp = requests.get(
                "https://api.openalex.org/works",
                params=params,
                headers=headers,
                timeout=60,
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"OpenAlex request failed: {resp.status_code} {resp.text[:200]}"
                )
            data = resp.json()
            for paper in data.get("results", []):
                openalex_id = paper.get("id")
                if not openalex_id or openalex_id in seen_ids:
                    continue
                seen_ids.add(openalex_id)

                abstract = reconstruct_abstract(
                    paper.get("abstract_inverted_index") or {}
                )
                if not abstract.strip():
                    continue
                paper["_abstract_text"] = abstract
                collected.append(paper)
                pbar.update(1)
                if len(collected) >= per_category:
                    break

            cursor = data.get("meta", {}).get("next_cursor")

    return collected


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download OpenAlex abstracts and build indices."
    )
    parser.add_argument(
        "--output-root",
        default="data/holisqa",
        help="Root output directory.",
    )
    parser.add_argument(
        "--per-category",
        type=int,
        default=1111,
        help="Number of abstracts per category.",
    )
    parser.add_argument("--year", type=int, default=2025, help="Publication year.")
    parser.add_argument(
        "--to-date",
        default=None,
        help="Latest publication date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--categories",
        default="all",
        help="Comma-separated category keys or 'all'.",
    )
    args = parser.parse_args()

    to_date = args.to_date or dt.date.today().isoformat()

    if args.categories == "all":
        categories = list(CATEGORIES.keys())
    else:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]
        missing = [c for c in categories if c not in CATEGORIES]
        if missing:
            raise SystemExit(f"Unknown categories: {missing}")

    for category in categories:
        concept_id = CATEGORIES[category]
        category_dir = os.path.join(args.output_root, category)
        individuals_dir = os.path.join(category_dir, "individuals")
        ensure_dir(individuals_dir)

        print(
            f"Fetching {args.per_category} abstracts for {category} (year {args.year})..."
        )
        papers = fetch_with_cursor(
            concept_id=concept_id,
            year=args.year,
            per_category=args.per_category,
            to_date=to_date,
            progress_label=category,
        )
        if len(papers) < args.per_category:
            print(
                f"Warning: only fetched {len(papers)} abstracts for {category}.",
                file=sys.stderr,
            )

        articles_path = os.path.join(category_dir, "articles.jsonl")
        sentences_path = os.path.join(category_dir, "sentences.jsonl")
        merged_path = os.path.join(category_dir, "all_abstracts.txt")

        article_rows = []
        sentence_rows = []
        sentence_id_counter = 0

        with open(merged_path, "w", encoding="utf-8") as merged_file:
            for idx, paper in enumerate(papers, start=1):
                article_id = f"A{idx:06d}"
                abstract_text = paper.get("_abstract_text", "").strip()
                if not abstract_text:
                    continue

                abstract_path = os.path.join(individuals_dir, f"{article_id}.txt")
                with open(abstract_path, "w", encoding="utf-8") as f:
                    f.write(abstract_text)

                if idx > 1:
                    merged_file.write("\n\n")
                merged_file.write(abstract_text)

                authorships = paper.get("authorships", [])
                authors = []
                for author in authorships:
                    name = (
                        author.get("author", {}) or {}
                    ).get("display_name")
                    if name:
                        authors.append(name)

                article_rows.append(
                    {
                        "article_id": article_id,
                        "openalex_id": paper.get("id"),
                        "title": paper.get("title"),
                        "publication_date": paper.get("publication_date"),
                        "publication_year": paper.get("publication_year"),
                        "authors": authors,
                        "abstract_path": abstract_path,
                        "abstract_char_len": len(abstract_text),
                        "category": category,
                    }
                )

                sentences = split_sentences(abstract_text)
                for sent_idx, sentence in enumerate(sentences):
                    sentence_id_counter += 1
                    sentence_rows.append(
                        {
                            "sentence_id": f"S{sentence_id_counter:08d}",
                            "article_id": article_id,
                            "sent_idx": sent_idx,
                            "text": sentence,
                        }
                    )

        write_jsonl(articles_path, article_rows)
        write_jsonl(sentences_path, sentence_rows)
        print(
            f"{category}: saved {len(article_rows)} articles, "
            f"{len(sentence_rows)} sentences."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
