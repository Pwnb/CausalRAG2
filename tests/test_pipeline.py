"""Offline end-to-end smoke test for CausalRAG2.

Runs the full text -> graph -> QA pipeline with a deterministic mock LLM, so it
needs no API key and no network. Run it with:

    python tests/test_pipeline.py        # or:  pytest -q

It verifies that the indexer produces a graph that ``run_single`` can load and
answer over.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class MockLLM:
    """Deterministic offline LLM used by the indexer (extraction + causal gate)."""

    TD, RD, CD = "<|>", "##", "<|COMPLETE|>"

    def chat(self, messages, json_mode: bool = False) -> str:
        user = messages[-1]["content"] if messages else ""
        if "Return exactly one token" in user:  # causal-gate verification (Fig. 9)
            return "yes"
        if "entity_description" in user:  # entity/relation extraction (Fig. 8)
            passage = user.split("Text:")[-1].split("######")[0]
            tokens: list[str] = []
            for word in re.findall(r"[A-Za-z]{4,}", passage.lower()):
                if word not in tokens:
                    tokens.append(word)
            tokens = tokens[:8]
            recs = [f'("entity"{self.TD}{w}{self.TD}concept{self.TD}concept {w})' for w in tokens]
            recs += [
                f'("relationship"{self.TD}{tokens[i]}{self.TD}{tokens[i + 1]}{self.TD}relates to{self.TD}7)'
                for i in range(len(tokens) - 1)
            ]
            return (self.RD + "\n").join(recs) + "\n" + self.CD
        return json.dumps({"title": "topic", "summary": "summary"})  # community report path


# --- Fake OpenAI client so run_single's _openai_generate runs offline --------- #
class _Resp:
    def __init__(self, text: str) -> None:
        self.output_text = text
        self.usage = None


class _Responses:
    def create(self, **kwargs):
        prompt = kwargs.get("input", "")
        if "precise" in prompt:  # causal reranker expects JSON
            return _Resp('{"precise": ["C1"], "p_answer": "A draft answer."}')
        return _Resp("Aspirin reduces inflammation, which causes pain.")  # final answer


class _Completions:
    def create(self, **kwargs):
        class _Msg:
            content = "Aspirin reduces inflammation, which causes pain."
            refusal = ""

        class _Choice:
            message = _Msg()

        class _Out:
            choices = [_Choice()]
            usage = None

        return _Out()


class FakeOpenAI:
    def __init__(self, **kwargs) -> None:
        self.responses = _Responses()
        self.chat = type("C", (), {"completions": _Completions()})()


def test_pipeline() -> None:
    # Inject a fake `openai` module so run_single runs offline (no real package
    # or API key needed). A real run uses the genuine openai client instead.
    import types

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeOpenAI
    sys.modules["openai"] = fake_openai

    from causalrag2 import build_graph, run_single

    passages = [
        "Aspirin reduces inflammation in tissues. Aspirin is a common medication.",
        "Inflammation causes pain and swelling. Chronic inflammation harms organs.",
        "Pain signals travel through nerves to the brain. Pain lowers quality of life.",
    ]

    with tempfile.TemporaryDirectory() as tmp:
        root = build_graph(
            passages,
            Path(tmp) / "graph",
            client=MockLLM(),
            build_causal=True,
            max_workers=2,
            cache=False,
        )

        # All parquet files the method needs must exist.
        out = Path(root) / "output_causal"
        for name in ["entities", "relationships", "text_units", "communities",
                     "community_reports", "community_causal"]:
            assert (out / f"{name}.parquet").exists(), f"missing {name}.parquet"

        result = run_single({"question": "What reduces inflammation?"}, str(root))

    assert isinstance(result, dict), type(result)
    assert result.get("answer", "").strip(), result
    print("ANSWER         :", result["answer"])
    print("seed entities  :", [e["title"] for e in result["meta"]["seed_entities"]])
    print("subgraph nodes :", result["meta"]["subgraph_nodes"])
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    test_pipeline()
