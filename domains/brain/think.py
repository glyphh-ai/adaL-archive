"""
The think pipeline — Ada's core cognitive loop.

Every request flows through here, whether it comes from the MCP endpoint
(LLM calling think()) or the TUI (human talking to Ada).

Pipeline:
  1. CLASSIFY — CognitiveGlyph: what IS this input?
  2. ROUTE   — BrainRouter: which capability handles it?
  3. EXECUTE — Run the capability's query pipeline
  4. ABSORB  — Store the interaction in memory
  5. OBSERVE — Log for dream loop pattern mining
  6. RETURN  — Structured response
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ada.memory.ada_cognitive import AdaCognitive, CognitiveResult
from domains.brain.loader import ADA_ORG_ID, BrainState
from domains.brain.llm import AdaLLM
from domains.brain.observation import Observation, ObservationLog
from domains.brain.router import BrainRouter

logger = logging.getLogger(__name__)


@dataclass
class ThinkResult:
    """The response from a think() call."""
    response: str  # Natural language response
    capability: Optional[str] = None  # Which capability handled it
    confidence: float = 0.0  # Routing confidence
    cognitive_state: Optional[str] = None  # RECALL, STORE, FEEL, etc.
    gate: str = "ASK"  # DONE or ASK
    facts: list[tuple] = field(default_factory=list)  # Retrieved facts
    llm_fallback: bool = False  # Whether Haiku helped with routing
    elapsed_ms: float = 0.0  # Processing time


class Brain:
    """Ada's brain — the central think pipeline.

    Owns:
    - AdaCognitive (memory, reasoning, dream loop)
    - BrainRouter (deterministic + LLM routing)
    - ObservationLog (for dream loop consumption)
    - AdaLLM (Haiku — Ada's internal language center)
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

        # Router initializes lazily on first think() call
        self._router_ready = False

    # ── Core API ─────────────────────────────────────────────────────────

    async def think(self, input_text: str) -> ThinkResult:
        """Process a natural language request through Ada's brain.

        This is the ONE entry point. Everything flows through here.
        """
        start = time.monotonic()

        # Lazy router init (avoids blocking server startup)
        if not self._router_ready:
            self._router.initialize()
            self._router_ready = True

        # 1. CLASSIFY — what kind of input is this?
        cognitive = self._cognitive.process(input_text)
        state_name = cognitive.state.action.name if cognitive.state else "UNKNOWN"

        # 2. ROUTE — which capability handles it?
        route = await self._router.route_with_fallback(input_text, self._llm)

        # 3. EXECUTE — run the capability's pipeline
        response_text = ""
        facts = cognitive.facts

        if route.capability:
            try:
                result = await self._execute_capability(
                    route.capability, input_text
                )
                if result:
                    response_text = result
            except Exception as e:
                logger.error(f"Capability {route.capability} failed: {e}")
                response_text = f"I tried to use my {route.capability} capability but encountered an error."

        # If no capability matched or execution failed, use cognitive result
        if not response_text:
            if cognitive.gate == "DONE" and cognitive.facts:
                # We have memories that match
                response_text = self._format_facts(cognitive.facts)
            elif self._llm.available:
                # Ask Haiku for help
                context = self._build_llm_context(input_text, cognitive, route)
                llm_response = await self._llm.ask(context)
                if llm_response:
                    response_text = llm_response
                    route.llm_fallback = True
                else:
                    response_text = "I'm not sure how to handle that yet."
            else:
                response_text = "I don't have a capability for that yet, and my language center is offline."

        # 4. ABSORB — remember this interaction
        self._cognitive.absorb(input_text)
        if response_text:
            self._cognitive.absorb(response_text, speaker="ada")

        # 5. OBSERVE — log for dream loop + TUI metrics
        elapsed = (time.monotonic() - start) * 1000
        self._observations.record(Observation(
            input=input_text,
            capability=route.capability,
            confidence=route.confidence,
            result=response_text[:200] if response_text else None,
            llm_fallback=route.llm_fallback,
        ))

        # Feed TUI dashboard metrics
        try:
            from ada.tui.app import get_metrics
            get_metrics().record_think(
                capability=route.capability,
                confidence=route.confidence,
                elapsed_ms=elapsed,
                llm_fallback=route.llm_fallback,
            )
        except Exception:
            pass  # TUI not running

        # 6. RETURN
        return ThinkResult(
            response=response_text,
            capability=route.capability,
            confidence=route.confidence,
            cognitive_state=state_name,
            gate=cognitive.gate,
            facts=facts,
            llm_fallback=route.llm_fallback,
            elapsed_ms=elapsed,
        )

    # ── Capability execution ─────────────────────────────────────────────

    async def _execute_capability(
        self, capability_name: str, query: str
    ) -> Optional[str]:
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
        """Start Ada's background reasoning."""
        self._cognitive.start_dreaming()

    def stop_dreaming(self) -> None:
        """Stop Ada's background reasoning."""
        self._cognitive.stop_dreaming()

    def drain_insights(self):
        """Get recent dream loop insights."""
        return self._cognitive.drain_insights()

    # ── Formatting helpers ───────────────────────────────────────────────

    def _format_facts(self, facts: list[tuple]) -> str:
        """Format recalled facts into a response."""
        if not facts:
            return ""
        lines = []
        for content, speaker, similarity in facts[:5]:
            lines.append(f"- {content} (confidence: {similarity:.2f})")
        return "\n".join(lines)

    def _format_query_result(self, result: dict, capability: str) -> str:
        """Format a capability query result into text."""
        # Handle different result shapes from the query service
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

        # Fallback: just stringify
        return f"[{capability}] {str(result)[:500]}"

    def _build_llm_context(self, input_text: str, cognitive: CognitiveResult, route) -> str:
        """Build context for Haiku when Ada needs language help."""
        parts = [f"User said: {input_text}"]
        if cognitive.state:
            parts.append(f"Cognitive state: {cognitive.state.action.name}")
        if cognitive.facts:
            parts.append("Relevant memories:")
            for content, speaker, sim in cognitive.facts[:3]:
                parts.append(f"  - {content} ({sim:.2f})")
        if route.capability:
            parts.append(f"Best capability match: {route.capability} ({route.confidence:.2f})")
        else:
            caps = list(self._state.capabilities.keys())
            parts.append(f"Available capabilities: {', '.join(caps)}")
            parts.append("No confident capability match found.")
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
