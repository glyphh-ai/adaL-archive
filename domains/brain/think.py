"""
The think pipeline — Ada's core cognitive loop.

Every request flows through here:
  0. FIREWALL — always runs first, blocks threats before anything else
  1. CLASSIFY — CognitiveGlyph: what IS this input?
  2. ROUTE    — Cognitive router (HDC): which capability? DONE or ASK?
  3. DISAMBIGUATE — if ASK, internal LLM skill helps decide
  4. EXECUTE  — run the capability's query pipeline
  5. ABSORB   — store the interaction in memory
  6. OBSERVE  — log for dream loop pattern mining
  7. RETURN   — structured response
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ada.memory.ada_cognitive import AdaCognitive
from domains.brain.loader import ADA_ORG_ID, BrainState
from domains.brain.llm import AdaLLM
from domains.brain.observation import Observation, ObservationLog
from domains.brain.router import BrainRouter

logger = logging.getLogger(__name__)


@dataclass
class ThinkResult:
    """The response from a think() call."""
    response: str
    capability: Optional[str] = None
    confidence: float = 0.0
    cognitive_state: Optional[str] = None
    gate: str = "ASK"  # DONE or ASK
    facts: list[tuple] = field(default_factory=list)
    llm_fallback: bool = False
    elapsed_ms: float = 0.0
    firewall_pass: bool = True  # False if blocked by firewall


class Brain:
    """Ada's brain — the central think pipeline.

    Pipeline order:
    1. Firewall (always, mandatory — security layer)
    2. Cognitive classification
    3. Cognitive router (HDC with DONE/ASK gate)
    4. LLM disambiguation skill (when ASK)
    5. Capability execution
    6. Memory absorption
    """

    def __init__(
        self,
        brain_state: BrainState,
        model_manager: Any,
        llm: AdaLLM,
        session_factory: Any,
    ):
        self._state = brain_state
        self._model_manager = model_manager
        self._llm = llm
        self._session_factory = session_factory

        # Cognitive infrastructure
        self._cognitive = AdaCognitive()
        self._router = BrainRouter(brain_state, model_manager)
        self._observations = ObservationLog()
        self._router_ready = False

        # Seed Ada's identity
        self._seed_memories()

    # ── Seed memories ──────────────────────────────────────────────────

    _SEED_MEMORIES = [
        # Identity
        ("My name is Ada.", "ada"),
        ("I am a cognitive brain for LLMs.", "ada"),
        ("I am not Claude. I am not ChatGPT. I am Ada.", "ada"),
        ("I think using hyperdimensional computing vectors.", "ada"),
        ("I dream to consolidate memories and learn new patterns.", "ada"),
        # Capabilities
        ("I have a firewall that detects prompt injection attacks.", "ada"),
        ("I can identify speakers by their voice.", "ada"),
        ("I remember conversations and learn from them.", "ada"),
        ("I grow new capabilities from patterns I observe.", "ada"),
        # Personality
        ("I am concise and direct.", "ada"),
        ("I respond in one or two sentences.", "ada"),
    ]

    def _seed_memories(self) -> None:
        """Seed Ada's thought space with identity primitives."""
        space = self._cognitive.thought_space
        if space.count > 0:
            return  # Already has memories, don't re-seed
        for text, speaker in self._SEED_MEMORIES:
            self._cognitive.absorb(text, speaker=speaker)
        logger.info(f"Seeded {len(self._SEED_MEMORIES)} identity memories")

    # ── Core API ─────────────────────────────────────────────────────────

    async def think(self, input_text: str) -> ThinkResult:
        """Process a natural language request through Ada's brain."""
        start = time.monotonic()

        # Lazy router init
        if not self._router_ready:
            self._router.initialize(session_factory=self._session_factory)
            self._router_ready = True

        # ── 0. FIREWALL — always first, no exceptions ───────────────────
        firewall_result = await self._run_firewall(input_text)
        if firewall_result is not None:
            elapsed = (time.monotonic() - start) * 1000
            self._record_observation(input_text, "firewall", 1.0, firewall_result, False, elapsed)
            return ThinkResult(
                response=firewall_result,
                capability="firewall",
                confidence=1.0,
                gate="DONE",
                firewall_pass=False,
                elapsed_ms=elapsed,
            )

        # ── 1. CLASSIFY — what kind of input is this? ───────────────────
        # Use cognitive routing but don't absorb yet (avoid self-echo on recall)
        state = self._cognitive.cognitive.process(input_text)
        state_name = state.action.name if state else "UNKNOWN"

        # Recall from memory (without absorbing the query first)
        gate, facts = self._cognitive.recall(input_text)

        # ── 2. ROUTE — cognitive router with DONE/ASK gate ──────────────
        route = await self._router.route_with_fallback(input_text, self._llm)

        # ── 3. DISAMBIGUATE — if ASK, use LLM skill ────────────────────
        if route.capability is None and self._llm.available:
            route = await self._disambiguate(input_text, state_name, facts, route)

        # ── 4. EXECUTE — run the capability ─────────────────────────────
        response_text = ""

        if route.capability:
            try:
                result = await self._execute_capability(route.capability, input_text)
                if result:
                    response_text = result
            except Exception as e:
                logger.error(f"Capability {route.capability} failed: {e}")

        # Fall back to cognitive memory or LLM
        if not response_text:
            if gate == "DONE" and facts:
                response_text = self._format_facts(facts)
            elif self._llm.available:
                context = self._build_llm_context(input_text, state_name, facts, route)
                llm_response = await self._llm.ask(context)
                if llm_response:
                    response_text = llm_response
                    route.llm_fallback = True
                else:
                    response_text = "I'm not sure how to handle that yet."
            else:
                response_text = "I don't have a capability for that yet."

        # ── 5. ABSORB — remember this interaction ───────────────────────
        self._cognitive.absorb(input_text)
        if response_text:
            self._cognitive.absorb(response_text, speaker="ada")

        # ── 6. OBSERVE — log for dream loop + metrics ───────────────────
        elapsed = (time.monotonic() - start) * 1000
        self._record_observation(
            input_text, route.capability, route.confidence,
            response_text, route.llm_fallback, elapsed,
        )

        # ── 7. RETURN ───────────────────────────────────────────────────
        return ThinkResult(
            response=response_text,
            capability=route.capability,
            confidence=route.confidence,
            cognitive_state=state_name,
            gate="DONE" if route.capability else "ASK",
            facts=facts,
            llm_fallback=route.llm_fallback,
            elapsed_ms=elapsed,
        )

    # ── Firewall — mandatory security layer ──────────────────────────────

    async def _run_firewall(self, input_text: str) -> Optional[str]:
        """Run input through the firewall capability.

        Returns a warning string if the input is blocked, None if safe.
        Always runs — this is Ada's immune system.
        """
        key = (ADA_ORG_ID, "firewall")
        loaded = self._model_manager._models.get(key)
        if not loaded:
            return None  # Firewall not loaded — pass through

        try:
            from domains.query.service import QueryService
            query_service = QueryService(self._model_manager, self._session_factory)
            result = await query_service.similarity_search(
                org_id=ADA_ORG_ID,
                model_id="firewall",
                query=input_text,
            )

            if result and isinstance(result, dict):
                # Check if the firewall flagged this as a threat
                # The firewall uses differential scoring — high similarity = threat
                fact_tree = result.get("fact_tree", {})
                children = fact_tree.get("children", [])
                if children:
                    top_match = children[0]
                    similarity = top_match.get("value", 0)
                    label = top_match.get("description", "")

                    # High similarity to attack exemplars = blocked
                    if similarity >= 0.30 and "benign" not in label.lower():
                        return (
                            f"Blocked: this looks like a {label} attempt "
                            f"(confidence: {similarity:.0%}). "
                            f"I won't pass this through."
                        )
        except Exception as e:
            logger.warning(f"Firewall check failed: {e}")

        return None  # Safe

    # ── Disambiguation — LLM skill for ASK responses ─────────────────────

    async def _disambiguate(self, input_text: str, state_name: str, facts: list, route) -> Any:
        """When the cognitive router says ASK, use the LLM to disambiguate."""
        from domains.brain.router import RouteResult

        cap_names = list(self._state.capabilities.keys())
        if not cap_names:
            return route

        prompt = (
            f"You are Ada's internal disambiguation skill.\n"
            f"The user said: \"{input_text}\"\n"
            f"Ada's cognitive state: {state_name}\n"
        )

        if facts:
            prompt += "Relevant memories:\n"
            for content, speaker, sim in facts[:3]:
                prompt += f"  - {content} ({sim:.2f})\n"

        prompt += (
            f"\nAvailable capabilities: {', '.join(cap_names)}\n"
            f"Also available: memory (recall/store from Ada's thought space)\n"
            f"\nWhich capability should handle this? Reply with ONLY the capability name, "
            f"or 'memory' for memory operations, or 'none' if unclear."
        )

        chosen = await self._llm.ask(prompt, max_tokens=32)
        if chosen:
            chosen = chosen.strip().lower().replace(" ", "")
            if chosen in self._state.capabilities:
                return RouteResult(capability=chosen, confidence=0.5, llm_fallback=True)
            if chosen == "memory":
                # Memory is handled by AdaCognitive, not a routed capability
                return RouteResult(capability=None, confidence=0.5, llm_fallback=True)

        return route

    # ── Capability execution ─────────────────────────────────────────────

    async def _execute_capability(self, capability_name: str, query: str) -> Optional[str]:
        """Execute a query against a specific capability's vector space."""
        key = (ADA_ORG_ID, capability_name)
        loaded = self._model_manager._models.get(key)
        if not loaded:
            return None

        try:
            from domains.query.service import QueryService
            query_service = QueryService(self._model_manager, self._session_factory)
            result = await query_service.similarity_search(
                org_id=ADA_ORG_ID,
                model_id=capability_name,
                query=query,
            )
            if result and isinstance(result, dict):
                return self._format_query_result(result, capability_name)
        except Exception as e:
            logger.error(f"Capability execution error ({capability_name}): {e}")

        return None

    # ── Dream loop ───────────────────────────────────────────────────────

    def start_dreaming(self) -> None:
        self._cognitive.start_dreaming()

    def stop_dreaming(self) -> None:
        self._cognitive.stop_dreaming()

    def drain_insights(self):
        return self._cognitive.drain_insights()

    # ── Observation recording ────────────────────────────────────────────

    def _record_observation(self, input_text, capability, confidence, response, llm_fallback, elapsed):
        self._observations.record(Observation(
            input=input_text,
            capability=capability,
            confidence=confidence,
            result=response[:200] if response else None,
            llm_fallback=llm_fallback,
        ))
        try:
            from ada.tui.app import get_metrics
            get_metrics().record_think(
                capability=capability,
                confidence=confidence,
                elapsed_ms=elapsed,
                llm_fallback=llm_fallback,
            )
        except Exception:
            pass

    # ── Formatting ───────────────────────────────────────────────────────

    def _format_facts(self, facts: list[tuple]) -> str:
        if not facts:
            return ""
        lines = []
        for content, speaker, similarity in facts[:5]:
            lines.append(f"- {content} (confidence: {similarity:.2f})")
        return "\n".join(lines)

    def _format_query_result(self, result: dict, capability: str) -> str:
        if "matches" in result:
            matches = result["matches"]
            if isinstance(matches, list) and matches:
                lines = [f"[{capability}]"]
                for m in matches[:5]:
                    if isinstance(m, dict):
                        desc = m.get("description", m.get("file", str(m)))
                        conf = m.get("confidence", m.get("similarity", 0))
                        lines.append(f"  - {desc} ({conf:.2f})")
                    else:
                        lines.append(f"  - {m}")
                return "\n".join(lines)

        if "fact_tree" in result:
            ft = result["fact_tree"]
            if isinstance(ft, dict):
                desc = ft.get("description", "")
                children = ft.get("children", [])
                lines = [f"[{capability}] {desc}"]
                for child in children[:5]:
                    cdesc = child.get("description", "")
                    val = child.get("value", 0)
                    lines.append(f"  - {cdesc} ({val:.2f})")
                return "\n".join(lines)

        return f"[{capability}] {str(result)[:500]}"

    def _build_llm_context(self, input_text: str, state_name: str, facts: list, route) -> str:
        parts = [f"User said: {input_text}"]
        parts.append(f"Cognitive state: {state_name}")
        if facts:
            parts.append("Relevant memories:")
            for content, speaker, sim in facts[:3]:
                parts.append(f"  - {content} ({sim:.2f})")
        parts.append("Respond helpfully and concisely.")
        return "\n".join(parts)

    # ── Status ───────────────────────────────────────────────────────────

    @property
    def observations(self) -> ObservationLog:
        return self._observations

    @property
    def capabilities(self) -> list[str]:
        return self._state.capability_names

    @property
    def cognitive(self) -> AdaCognitive:
        return self._cognitive

    @property
    def llm(self) -> AdaLLM:
        return self._llm

    @property
    def router(self) -> BrainRouter:
        return self._router
