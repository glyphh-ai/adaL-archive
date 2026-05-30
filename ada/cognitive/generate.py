"""
GlyphGenerator — render text from the cognitive substrate.

The 10th stage of the cognitive loop. Takes a query, runs glyph-based
multi-hop recall, and emits two things:
  1. A "render context" — what the substrate believes is relevant
  2. Optionally, rendered text (if a language renderer is wired in)

The substrate does the cognition; the renderer does the surface form.
LLM is optional — without it you see the substrate's raw output, which
is the new bit. With it, the LLM only sees what Ada chose to expose.

Usage:
    from ada.memory.thought_space import ThoughtGlyphSpace
    from ada.memory.glyph_cognitive import GlyphCognitiveLoop
    from ada.cognitive.generate import GlyphGenerator, build_llm_renderer

    space = ThoughtGlyphSpace()
    cognitive = GlyphCognitiveLoop(space)
    gen = GlyphGenerator(space, cognitive, renderer=build_llm_renderer())

    space.absorb("My name is Chris.")
    result = gen.generate("What's my name?")
    print(result.context)     # substrate output
    print(result.response)    # LLM rendering (if renderer wired)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

from ada.memory.glyph_cognitive import GlyphCognitiveLoop, GlyphReasoningChain
from ada.memory.thought_space import ThoughtGlyphSpace

logger = logging.getLogger(__name__)


# ── Result type ──────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    """Output of one generate() call."""
    query: str
    chains: list[GlyphReasoningChain] = field(default_factory=list)
    context: str = ""
    response: Optional[str] = None
    used_renderer: bool = False

    @property
    def top_recall(self) -> Optional[str]:
        """The single most-relevant recalled memory, if any."""
        if not self.chains or not self.chains[0].hops:
            return None
        return self.chains[0].hops[0].result.thought.content


# A Renderer takes (context, query) and returns generated text.
Renderer = Callable[[str, str], str]


# ── Generator ────────────────────────────────────────────────────────────

DEFAULT_SYSTEM = (
    "You are Ada — a cognitive substrate that recalls memories via "
    "hyperdimensional vectors and renders them to natural language. "
    "Answer the user's query using ONLY the recalled memories provided. "
    "Be concise (1-2 sentences). If the memories don't contain the answer, "
    "say so directly — never invent facts."
)


class GlyphGenerator:
    """The GENERATE stage of the cognitive loop.

    Pipeline:
      1. cognitive.reason(query) → glyph chains with confidence + layer activations
      2. Build a render context from the top chains (memories + their activation pattern)
      3. Optionally call a renderer (LLM-as-renderer is the natural fit)
      4. Absorb the response back to memory (closes the learning loop)
      5. Hebbian-reinforce the winning chain
    """

    def __init__(
        self,
        space: ThoughtGlyphSpace,
        cognitive: GlyphCognitiveLoop,
        renderer: Optional[Renderer] = None,
    ):
        self.space = space
        self.cognitive = cognitive
        self.renderer = renderer

    def generate(
        self,
        query: str,
        top_chains: int = 5,
        absorb_response: bool = True,
        reinforce: bool = True,
    ) -> GenerationResult:
        # 1. Cognitive recall — multi-hop glyph reasoning.
        chains = self.cognitive.reason(query, top_k=top_chains)

        # 2. Assemble the render context.
        context = self._build_context(query, chains)

        # 3. Optional rendering via LLM.
        response = None
        used_renderer = False
        if self.renderer is not None:
            try:
                response = self.renderer(context, query)
                used_renderer = True
            except Exception as e:
                logger.warning("Renderer failed: %s", e)

        # 4. Close the loop — absorb Ada's own response as a thought.
        if absorb_response and response:
            self.space.absorb(response, speaker="outgoing")

        # 5. Reinforce the winning chain (Hebbian).
        if reinforce and chains:
            self.cognitive.confirm(chains[0])

        return GenerationResult(
            query=query,
            chains=chains,
            context=context,
            response=response,
            used_renderer=used_renderer,
        )

    # ── context assembly ────────────────────────────────────────────────

    def _build_context(self, query: str, chains: list[GlyphReasoningChain]) -> str:
        lines = [f"Query: {query!r}", ""]
        if not chains:
            lines.append("(substrate found no relevant memories)")
            return "\n".join(lines)
        lines.append(f"Substrate recalled {len(chains)} chain(s):")
        for i, chain in enumerate(chains, 1):
            tag = ""
            if chain.pathway_name:
                tag += f" via pathway:{chain.pathway_name}"
            if chain.pattern_boost > 0:
                tag += f" +{chain.pattern_boost:.2f}"
            lines.append(f"  chain {i}  conf={chain.confidence:.2f}{tag}")
            for j, hop in enumerate(chain.hops, 1):
                content = hop.result.thought.content
                top_layers = sorted(
                    hop.matching_layers.items(), key=lambda kv: -kv[1]
                )[:3]
                layer_str = ", ".join(f"{k}:{v:.2f}" for k, v in top_layers)
                lines.append(f"    hop {j}: {content!r}")
                if layer_str:
                    lines.append(f"            layers ─ {layer_str}")
        return "\n".join(lines)


# ── LLM renderer factory ─────────────────────────────────────────────────

def build_llm_renderer(
    model: str | None = None,
    system: str = DEFAULT_SYSTEM,
    max_tokens: int = 200,
) -> Optional[Renderer]:
    """Return an Anthropic-backed renderer, or None if the SDK / key isn't here.

    The substrate works without it — the renderer is just the surface form.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.info("No ANTHROPIC_API_KEY — generator will return substrate context only.")
        return None
    try:
        import anthropic  # noqa: F401
    except ImportError:
        logger.info("anthropic SDK not installed — generator will return substrate context only.")
        return None

    import anthropic as _a
    client = _a.Anthropic()
    chosen = model or os.environ.get("ADA_MODEL", "claude-haiku-4-5-20251001")

    def _render(context: str, query: str) -> str:
        resp = client.messages.create(
            model=chosen,
            max_tokens=max_tokens,
            system=system,
            messages=[{
                "role": "user",
                "content": (
                    f"{context}\n\n"
                    f"Answer this query using only the recalled memories: {query}"
                ),
            }],
        )
        return resp.content[0].text.strip()

    return _render
