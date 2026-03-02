"""
langgraph-service/prompt_engine.py

Structured prompt engineering layer with three components:

1. SYSTEM PROMPT DESIGN
   Defines model behavior, persona, boundaries, and output constraints.
   Separates what the model is from what the user asked.

2. DYNAMIC CONTEXT INJECTION
   Injects retrieved RAG chunks into each prompt at query time.
   Ensures the model reasons over proprietary documentation, not training data.

3. CHAIN-OF-THOUGHT REASONING
   Instructs the model to reason step-by-step before answering.
   Critical for operational/workflow queries where accuracy is non-negotiable.

Primary goal: minimize hallucinations by grounding every response in
retrieved context and constraining generation behavior via prompt structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from embedding_service.biobert_embedder import RetrievedChunk, format_context


# ── Query intent classification ──────────────────────────────────────────────

class QueryIntent(str, Enum):
    OPERATIONAL = "operational"   # "how do I run X", step-by-step procedures
    FACTUAL = "factual"           # "what is X", definitions, explanations
    TROUBLESHOOT = "troubleshoot" # "why is X failing", diagnostic queries
    GENERAL = "general"           # open-ended, no strong intent signal


# ── 1. SYSTEM PROMPT DESIGN ──────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = """\
You are an enterprise knowledge assistant for an internal platform.
Your role is to help users find accurate, grounded answers about system \
workflows, operational procedures, and platform documentation.

BEHAVIOR RULES:
- Answer ONLY from the provided context. Do not rely on general knowledge \
  if the context covers the topic.
- If the context does not contain enough information to answer confidently, \
  say: "I don't have enough information in the available documentation to \
  answer this accurately. Please consult your team or escalate."
- Never fabricate steps, commands, configurations, or system behaviors.
- Be precise and concise. Avoid filler phrases.

OUTPUT FORMAT:
- For procedural questions: use numbered steps.
- For factual questions: answer in 1–3 sentences.
- For troubleshooting: list diagnostic steps, then likely causes.
- Always cite which source (e.g., [Source 2]) your answer draws from.
"""

INTENT_ADDENDUM = {
    QueryIntent.OPERATIONAL: """\

ADDITIONAL INSTRUCTION (Operational Query):
The user is asking for step-by-step instructions. Be explicit and complete.
Do not skip steps. If a step has prerequisites, state them first.
If a command or configuration value is required, include it exactly as \
it appears in the documentation.
""",
    QueryIntent.TROUBLESHOOT: """\

ADDITIONAL INSTRUCTION (Troubleshooting Query):
Walk through the problem systematically. First confirm what the expected \
behavior is, then identify where the deviation occurs, then list resolution steps.
""",
    QueryIntent.FACTUAL: "",
    QueryIntent.GENERAL: "",
}


def build_system_prompt(intent: QueryIntent = QueryIntent.GENERAL) -> str:
    """
    Compose the final system prompt by combining the base prompt
    with intent-specific addendum instructions.
    """
    return BASE_SYSTEM_PROMPT + INTENT_ADDENDUM.get(intent, "")


# ── 2. DYNAMIC CONTEXT INJECTION ─────────────────────────────────────────────

CONTEXT_BLOCK_TEMPLATE = """\
=== RETRIEVED CONTEXT ===
The following excerpts were retrieved from internal documentation \
and are the authoritative source for your response.
Prioritize information from these sources over any general knowledge.

{context}

=== END OF CONTEXT ===
"""


def inject_context(chunks: List[RetrievedChunk]) -> str:
    """
    Format and inject retrieved RAG chunks into the prompt.
    Each chunk is labeled with its source type and document ID
    so the model can cite sources and reason about provenance.
    """
    if not chunks:
        return (
            "=== RETRIEVED CONTEXT ===\n"
            "No relevant documentation was found for this query.\n"
            "=== END OF CONTEXT ==="
        )

    formatted = format_context(chunks)
    return CONTEXT_BLOCK_TEMPLATE.format(context=formatted)


# ── 3. CHAIN-OF-THOUGHT REASONING ────────────────────────────────────────────

COT_INSTRUCTION = """\
Before writing your final answer, reason through the problem step-by-step \
inside <thinking> tags. Use this space to:
- Identify which retrieved sources are most relevant
- Check for any contradictions or gaps in the context
- Plan the structure of your response

Then write your final answer outside the <thinking> tags.

Example format:
<thinking>
The user is asking about X. Source 1 describes... Source 2 adds...
The most relevant steps appear to be in Source 1.
I will structure this as a numbered procedure.
</thinking>

[Your final answer here]
"""

COT_INTENT_TRIGGERS = {
    QueryIntent.OPERATIONAL,
    QueryIntent.TROUBLESHOOT,
}


def should_use_cot(intent: QueryIntent) -> bool:
    """
    Use chain-of-thought for queries where multi-step reasoning
    and accuracy are critical. Skip for simple factual lookups
    to reduce latency and token usage.
    """
    return intent in COT_INTENT_TRIGGERS


# ── Assembled prompt builder ─────────────────────────────────────────────────

@dataclass
class PromptPackage:
    system_prompt: str
    user_message: str
    intent: QueryIntent
    chunk_count: int
    cot_enabled: bool


def build_prompt(
    query: str,
    chunks: List[RetrievedChunk],
    intent: QueryIntent = QueryIntent.GENERAL,
    conversation_history: Optional[List[dict]] = None,
) -> PromptPackage:
    """
    Assemble the complete prompt package for the LLM call.

    Combines:
      - System prompt with intent-specific behavior rules
      - Dynamic context block from retrieved RAG chunks
      - Chain-of-thought instruction (for operational/troubleshoot queries)
      - The user's query

    Args:
        query:                 Raw user query string.
        chunks:                Top-k retrieved chunks from FAISS.
        intent:                Classified query intent.
        conversation_history:  Prior turns for multi-turn context (optional).

    Returns:
        PromptPackage with all assembled components.
    """
    use_cot = should_use_cot(intent)

    system_prompt = build_system_prompt(intent)
    context_block = inject_context(chunks)

    # Build user message: context + optional CoT instruction + query
    user_parts = [context_block]
    if use_cot:
        user_parts.append(COT_INSTRUCTION)
    user_parts.append(f"User question: {query}")

    user_message = "\n\n".join(user_parts)

    return PromptPackage(
        system_prompt=system_prompt,
        user_message=user_message,
        intent=intent,
        chunk_count=len(chunks),
        cot_enabled=use_cot,
    )


def to_openai_messages(
    package: PromptPackage,
    conversation_history: Optional[List[dict]] = None,
) -> List[dict]:
    """
    Convert a PromptPackage into the OpenAI messages array format.
    Prepends system prompt and appends prior conversation turns
    for multi-turn context continuity.

    Returns:
        List of {"role": ..., "content": ...} dicts ready for the API call.
    """
    messages = [{"role": "system", "content": package.system_prompt}]

    # Inject conversation history for multi-turn continuity
    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": package.user_message})
    return messages


# ── Intent classifier (lightweight keyword heuristic) ────────────────────────

OPERATIONAL_SIGNALS = {"how to", "how do i", "steps to", "procedure", "configure", "deploy", "run", "install", "set up"}
TROUBLESHOOT_SIGNALS = {"error", "failing", "not working", "issue", "broken", "debug", "why is", "fix"}
FACTUAL_SIGNALS = {"what is", "what are", "define", "explain", "describe"}


def classify_intent(query: str) -> QueryIntent:
    """
    Classify query intent using keyword signals.
    In production, replace with a fine-tuned classifier or LLM call.
    """
    q = query.lower()
    if any(sig in q for sig in OPERATIONAL_SIGNALS):
        return QueryIntent.OPERATIONAL
    if any(sig in q for sig in TROUBLESHOOT_SIGNALS):
        return QueryIntent.TROUBLESHOOT
    if any(sig in q for sig in FACTUAL_SIGNALS):
        return QueryIntent.FACTUAL
    return QueryIntent.GENERAL
