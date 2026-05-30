"""
Two cognitive surfaces on the same substrate:

    think(input)   — pure associative wave. Surfaces what activates,
                     without interpretation. No question/answer framing.
                     This is the brain's reflex; you "think about X" and
                     a constellation of related thoughts fires.

    ask(question)  — targeted retrieval. Predicate-aware structural
                     match. Returns one fact and a confidence, or
                     refuses honestly with "I don't know."

Same substrate, different recall policies. think is broader and unfiltered;
ask is narrower and disciplined. The MCP layer exposes both — Claude calls
think when it wants context, ask when it needs a specific fact.

Usage:
    surface = CognitiveSurface(space, cognitive, renderer=...)
    activation = surface.think("project aurora")
    answer     = surface.ask("what hull does aurora use?")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ada.memory.glyph_cognitive import GlyphCognitiveLoop, GlyphReasoningChain
from ada.memory.thought_space import StoredThought, ThoughtGlyphSpace


# ── Return types ─────────────────────────────────────────────────────

@dataclass
class Activation:
    """What think() returns — raw associative output, no synthesis."""
    input: str
    thoughts: list[StoredThought] = field(default_factory=list)
    similarities: list[float] = field(default_factory=list)
    pathway: Optional[str] = None
    pattern_boost: float = 0.0

    def __iter__(self):
        return iter(zip(self.thoughts, self.similarities))

    def __len__(self) -> int:
        return len(self.thoughts)


@dataclass
class Answer:
    """What ask() returns — one fact and a confidence, or 'I don't know'."""
    question: str
    fact: Optional[StoredThought] = None
    confidence: float = 0.0
    rendered: str = ""           # natural-language form (via renderer)
    refused: bool = False        # True when confidence < threshold


# ── The surface ──────────────────────────────────────────────────────

class CognitiveSurface:
    """The two-tool API on Ada's substrate."""

    def __init__(
        self,
        space: ThoughtGlyphSpace,
        cognitive: GlyphCognitiveLoop,
        renderer=None,
        min_confidence: float = 0.4,
    ):
        self.space = space
        self.cognitive = cognitive
        self.renderer = renderer
        self.min_confidence = min_confidence

    # ── think — associative wave ──────────────────────────────────────

    def think(self, input_text: str, top_k: int = 10) -> Activation:
        """The brain's reflex — what does this input activate?

        Surfaces the constellation of related thoughts without filtering
        for "the answer." No question is assumed. No refusal threshold.
        The output is what's resonant — interpretation is the caller's job.
        """
        chains = self.cognitive.reason(input_text, top_k=top_k)
        thoughts: list[StoredThought] = []
        sims: list[float] = []
        seen: set[str] = set()
        for chain in chains:
            for hop in chain.hops:
                tid = hop.result.thought.thought_id
                if tid in seen:
                    continue
                seen.add(tid)
                thoughts.append(hop.result.thought)
                sims.append(chain.confidence)
        return Activation(
            input=input_text,
            thoughts=thoughts,
            similarities=sims,
            pathway=chains[0].pathway_name if chains else None,
            pattern_boost=chains[0].pattern_boost if chains else 0.0,
        )

    # ── ask — targeted retrieval ──────────────────────────────────────

    def ask(self, question: str, top_chains: int = 5) -> Answer:
        """Find me a fact that answers this question.

        Uses the substrate's predicate-aware recall + the LLM renderer
        (when available) to produce a single grounded answer. Refuses
        with low confidence rather than inventing — the no-hallucinations
        property comes from this discipline.
        """
        chains = self.cognitive.reason(question, top_chains)
        if not chains or not chains[0].hops:
            return Answer(question=question, refused=True, rendered="I don't know.")

        top = chains[0]
        confidence = top.confidence

        if confidence < self.min_confidence:
            return Answer(
                question=question,
                confidence=confidence,
                refused=True,
                rendered="I don't know.",
            )

        fact = top.hops[0].result.thought

        # Render the answer through the LLM if one's wired in. The renderer
        # sees only the top facts — it can't invent because there's nothing
        # off-substrate to invent from.
        if self.renderer is not None:
            try:
                facts_for_render = "\n".join(
                    f"- {chain.hops[0].result.thought.content}"
                    for chain in chains[:top_chains]
                    if chain.hops
                )
                rendered = self.renderer(facts_for_render, question)
            except Exception:
                rendered = fact.content
        else:
            rendered = fact.content

        return Answer(
            question=question,
            fact=fact,
            confidence=confidence,
            rendered=rendered,
        )
