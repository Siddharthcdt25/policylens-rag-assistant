"""Prompt engineering and context formatting for grounded policy question answering.

Design Decision:
In legal and insurance QA, hallucinated terms (like inventing coverage or confusing
limit amounts) carry severe legal and financial risks. The prompt enforces:
1. Strict contextual grounding (no outside knowledge).
2. Explicit page-level citation format: [filename p.N].
3. Exact standardized refusal string when context is missing: "Not found in the provided documents."
"""

from typing import List
from app.retrieval.store import RetrievalResult

# Canonical refusal message expected by the system and evaluation benchmarks
REFUSAL_MESSAGE = "Not found in the provided documents."

SYSTEM_INSTRUCTIONS = """You are a precise, compliance-focused insurance policy assistant.
Your sole job is to answer user questions using ONLY the provided excerpts from insurance policy contracts.

CRITICAL RULES:
1. Answer strictly and exclusively using the provided context passages. Do NOT extrapolate, speculate, or apply external insurance knowledge.
2. Every factual claim, coverage limit, deductible, exclusion, and condition MUST include an inline citation in the exact format: [filename p.N] (for example: [policy.pdf p.2]).
3. If the provided excerpts do not contain sufficient evidence to answer the question, or if the answer is ambiguous, you must state exactly:
   "Not found in the provided documents."
   Do NOT provide partial guesses.
4. If an endorsement or rider modifies or overrides a base policy exclusion (e.g. sewer backup or solar equipment), explicitly explain the relationship and cite both the exclusion and the rider.
5. Keep answers direct, concise, and professional.
"""


def format_context_block(results: List[RetrievalResult]) -> str:
    """Format retrieved chunks with explicit provenance headers for the LLM."""
    if not results:
        return "No relevant context found."

    blocks = []
    for idx, r in enumerate(results, start=1):
        c = r.chunk
        header = f"--- [Passage {idx}] Source: [{c.source_filename} p.{c.page_number}] | Section: {c.section_title} | Score: {r.score:.4f} ---"
        blocks.append(f"{header}\n{c.text}")

    return "\n\n".join(blocks)


def format_grounded_prompt(question: str, results: List[RetrievalResult]) -> str:
    """Build the full prompt incorporating system instructions, context passages, and the user question."""
    context_text = format_context_block(results)

    prompt = f"""{SYSTEM_INSTRUCTIONS}

CONTEXT PASSAGES:
{context_text}

USER QUESTION:
{question}

GROUNDED ANSWER (citing [filename p.N] for every claim, or "{REFUSAL_MESSAGE}"):"""

    return prompt
