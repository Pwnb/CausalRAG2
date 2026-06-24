"""All LLM prompts used by CausalRAG2.

These match the prompts reported in the paper appendix:
  - IE_EXTRACTION_PROMPT   : Figure 8  (offline entity/relation extraction)
  - CAUSAL_GATE_PROMPT     : Figure 9  (offline binary causal-gate verification)
  - CAUSAL_RERANK_PROMPT   : Figure 6  (online causal path identification, standard)
  - CT_CAUSAL_RERANK_PROMPT: Figure 5  (online causal path identification, spurious-aware)
  - FINAL_ANSWER_PROMPT    : Figure 7  (online final answer generation)
"""

from __future__ import annotations

# Delimiters used by the GraphRAG-style extraction format (Figure 8).
TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"
COMPLETION_DELIMITER = "<|COMPLETE|>"
DEFAULT_ENTITY_TYPES = "concept,entity,event,method,finding,organization,person,location"


# --------------------------------------------------------------------------- #
# Offline: entity / relation extraction (Figure 8, modified from GraphRAG)
# --------------------------------------------------------------------------- #
IE_EXTRACTION_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:
Entity_types: ORGANIZATION,PERSON
Text:
The Verdantis's Central Institution is scheduled to meet on Monday and Thursday, with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT, followed by a press conference where Central Institution Chair Martin Smith will take questions.
######################
Output:
("entity"{tuple_delimiter}CENTRAL INSTITUTION{tuple_delimiter}ORGANIZATION{tuple_delimiter}The Central Institution is the Federal Reserve of Verdantis, which is setting interest rates on Monday and Thursday){record_delimiter}
("entity"{tuple_delimiter}MARTIN SMITH{tuple_delimiter}PERSON{tuple_delimiter}Martin Smith is the chair of the Central Institution){record_delimiter}
("relationship"{tuple_delimiter}MARTIN SMITH{tuple_delimiter}CENTRAL INSTITUTION{tuple_delimiter}Martin Smith is the Chair of the Central Institution and will answer questions at a press conference{tuple_delimiter}9)
{completion_delimiter}

######################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
""".strip()


# --------------------------------------------------------------------------- #
# Offline: binary causal-gate verification (Figure 9)
# --------------------------------------------------------------------------- #
CAUSAL_GATE_PROMPT = """
-Goal-
Given two text snippets A and B, decide whether there is any plausible causal relationship between them (either direction) under some reasonable context.

-Steps-
Read A and B, and consider whether one could plausibly influence the other (directly or indirectly).
Require a plausible mechanism; ignore mere correlation or co-occurrence.
If uncertain or only associative, choose "no".

-Output-
Return exactly one token: "yes" or "no". No extra text.

######################
-Real Data-
A: {a_text}
B: {b_text}
######################
Output:
""".strip()


# --------------------------------------------------------------------------- #
# Online: causal path identification, standard (Figure 6)
# --------------------------------------------------------------------------- #
CAUSAL_RERANK_PROMPT = """
---Role---

You are a careful causality analyst acting as a reranker for retrieval.


---Goal---

Given a query and a list of context items (short ID + content), select the most important items that best support answering the query as a causal graph.

You MUST:

- Use only the provided items.

- Rank the `precise` list from most important to least important.

- Output JSON only. Do not add markdown.

- Use the short IDs exactly as shown.

- Do NOT include any IDs in `p_answer`.

- If evidence is insufficient, say so in `p_answer` (e.g., "Unknown").


---Inputs---

Query:

{query}


Context Items (short ID | content):

{context_table}


---Output Format (JSON)---

{{

  "precise": ["C1", "N2", "E3"],

  "p_answer": "concise draft answer"

}}


---Constraints---

- `precise` length: at most {max_precise_items} items.

- `p_answer` length: at most {max_answer_words} words.
""".strip()


# --------------------------------------------------------------------------- #
# Online: causal path identification, spurious-aware (Figure 5, main setting)
# --------------------------------------------------------------------------- #
CT_CAUSAL_RERANK_PROMPT = """
---Role---

You are a careful causality analyst acting as a reranker for retrieval.


---Goal---

Given a query and a list of context items (short ID + content), select the most important items consisting the causal graph and output them in "precise".

Also output the least important items as the spurious information in "ct_precise".

You MUST:

- Use only the provided items.

- Rank `precise` from most important to least important.

- Rank `ct_precise` from least important to more important.

- Output JSON only. Do not add markdown.

- Use the short IDs exactly as shown.

- Do NOT include any IDs in `p_answer`.


---Inputs---

Query:

{query}

Context Items (short ID | content):

{context_table}


---Output Format (JSON)---

{{

  "precise": ["C1", "N2", "E3"],

  "ct_precise": ["T7", "N9"],

  "p_answer": "concise draft answer"

}}


---Constraints---

- `precise` length: at most {max_precise_items} items.

- `ct_precise` length: at most {max_ct_precise_items} items.

- `p_answer` length: at most {max_answer_words} words.
""".strip()


# --------------------------------------------------------------------------- #
# Online: final answer generation (Figure 7)
# --------------------------------------------------------------------------- #
FINAL_ANSWER_PROMPT = """
---Role---

You are a helpful assistant answering the user's question.


---Goal---

Answer the question using the provided evidence context. A draft answer may be provided; use it only if it is supported by the evidence.


---Evidence Context---

{report_context}


---Draft Answer (optional)---

{draft_answer}


---Question---

{query}


---Answer Format---

Concise, direct, and neutral.
""".strip()


# --------------------------------------------------------------------------- #
# Offline: community report
# (the paper builds community summaries in the GraphRAG style, Edge et al. 2024;
#  there is no dedicated figure for it, so this is a faithful condensed version)
# --------------------------------------------------------------------------- #
COMMUNITY_REPORT_PROMPT = """
-Goal-
Write a concise report that summarizes a community of related entities and
relationships extracted from a knowledge graph.

-Output-
Return JSON only: {{"title": "<a 5-7 word title naming the community>", "summary": "<2-4 sentences describing the community's main entities, themes, and how they relate>"}}.

######################
-Community Data-
{records}
######################
Output:
""".strip()
