# HolisQA

HolisQA is a holistic, multi-sentence reading-comprehension dataset built from
recent (2025) open-access paper abstracts on [OpenAlex](https://openalex.org).
Each question requires reasoning across several sentences of a document slice
rather than a single-fact lookup.

## Domains

`biology/`, `business/`, `computer_science/`, `medicine/`, `psychology/`
(about 2,200 QA pairs in total).

## Files (per domain)

| File | Contents |
|------|----------|
| `qa.jsonl` | the QA pairs (see schema below) |
| `sentences.jsonl` | source sentences with stable `sentence_id`s |
| `articles.jsonl` | article metadata (title, authors, OpenAlex id) |
| `all_abstracts.txt` | concatenated abstracts, ready to index |
| `slices.jsonl` | the document slices each QA batch was drawn from |

## `qa.jsonl` schema

```json
{
  "qa_id": "QA000001",
  "category": "biology",
  "slice_id": "SL000001",
  "question": "...",
  "answer": "...",
  "context_sentence_ids": ["S00004678", "S00004681"],
  "context_text": "..."
}
```

## Rebuilding from scratch

```bash
# 1. download abstracts + build sentence/article indices
python scripts/holisqa_1_download_openalex.py --per-category 1111 --year 2025

# 2. generate QA pairs from sentence slices
python scripts/holisqa_2_generate_qa.py --runs 50 --qas-per-run 10
```

The QA generator talks to any OpenAI-compatible endpoint; the released split was
generated with `gemini-2.5-flash-lite`. Set `GEMINI_API_KEY` (or `OPENAI_API_KEY`)
and adjust `--model` / `--base-url` as needed.
