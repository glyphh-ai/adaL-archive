"""
Brain router — deterministic routing of think() requests to capabilities.

Uses the toolrouter capability's HDC encoder + cosine similarity
to match queries against capability descriptions. Sub-10ms after init.

Fallback: Ada calls Haiku to classify when confidence is low.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from domains.brain.loader import BrainState, Capability

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.40


@dataclass
class RouteResult:
    """Where the brain decided to send this request."""
    capability: Optional[str] = None
    confidence: float = 0.0
    llm_fallback: bool = False


class BrainRouter:
    """Routes natural language to the right brain capability.

    Uses the toolrouter's encoder to build a vector per capability,
    then cosine similarity at query time. After first-call init (~1s),
    routing is sub-10ms.
    """

    def __init__(self, brain_state: BrainState, model_manager=None):
        self._state = brain_state
        self._model_manager = model_manager
        self._capability_vectors: dict[str, np.ndarray] = {}
        self._encoder = None
        self._encode_fn = None
        self._initialized = False

    def initialize(self) -> None:
        """Build capability routing vectors from the toolrouter encoder."""
        if not self._state.router:
            logger.warning("No toolrouter — routing will use LLM only")
            return

        if not self._model_manager:
            logger.warning("No model manager — routing degraded")
            return

        from domains.brain.loader import ADA_ORG_ID

        router_key = (ADA_ORG_ID, self._state.router.name)
        loaded = self._model_manager._models.get(router_key)
        if not loaded:
            logger.warning("Toolrouter not in model registry — routing degraded")
            return

        self._encoder = loaded.encoder
        self._encode_fn = loaded.encode_query_fn

        if not self._encode_fn or not self._encoder:
            logger.warning("Toolrouter has no encoder — routing degraded")
            return

        from ada.core.types import Concept

        # Warm up the extractor cache (one-time ~16s cost)
        logger.info("Warming up intent extractor...")
        self._encode_fn("warmup")
        logger.info("Intent extractor ready")

        # Now encode each capability — should be fast
        for name, cap in self._state.capabilities.items():
            desc = self._build_routing_text(cap)
            try:
                raw = self._encode_fn(desc)
                if isinstance(raw, dict):
                    concept = Concept(
                        name=raw.get("name", name),
                        attributes=raw.get("attributes", {}),
                    )
                else:
                    concept = raw
                vector = self._encoder.encode(concept)
                self._capability_vectors[name] = vector
            except Exception as e:
                logger.warning(f"Failed to encode routing vector for {name}: {e}")

        self._initialized = True
        logger.info(
            f"Router initialized: {len(self._capability_vectors)} capability vectors"
        )

    def _build_routing_text(self, cap: Capability) -> str:
        parts = [cap.manifest.description]
        if cap.manifest.tags:
            parts.append(" ".join(cap.manifest.tags))
        if cap.manifest.category:
            parts.append(cap.manifest.category)
        return " ".join(parts)

    def route(self, query: str) -> RouteResult:
        """Route a query to the best matching capability. Sub-10ms."""
        if not self._initialized or not self._encoder or not self._encode_fn:
            return RouteResult()

        try:
            from ada.core.types import Concept
            raw = self._encode_fn(query)
            if isinstance(raw, dict):
                concept = Concept(
                    name=raw.get("name", "query"),
                    attributes=raw.get("attributes", {}),
                )
            else:
                concept = raw
            query_vector = self._encoder.encode(concept)
        except Exception as e:
            logger.warning(f"Failed to encode query: {e}")
            return RouteResult()

        best_name = None
        best_score = -1.0

        for name, cap_vector in self._capability_vectors.items():
            score = self._cosine_similarity(query_vector, cap_vector)
            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= CONFIDENCE_THRESHOLD and best_name:
            return RouteResult(capability=best_name, confidence=best_score)

        return RouteResult(confidence=best_score)

    async def route_with_fallback(self, query: str, llm=None) -> RouteResult:
        """Route with optional LLM fallback for low-confidence queries."""
        result = self.route(query)

        if result.capability is not None:
            return result

        if llm and llm.available:
            cap_names = list(self._state.capabilities.keys())
            if cap_names:
                chosen = await llm.classify(query, cap_names)
                if chosen and chosen in self._state.capabilities:
                    return RouteResult(
                        capability=chosen,
                        confidence=0.5,
                        llm_fallback=True,
                    )

        return result

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        """Cosine similarity — works with Glyph objects or numpy arrays."""
        try:
            # Glyph objects have a .cortex attribute (the global vector)
            va = getattr(a, 'cortex', a)
            vb = getattr(b, 'cortex', b)

            if isinstance(va, list):
                va = np.array(va)
            if isinstance(vb, list):
                vb = np.array(vb)

            if hasattr(va, 'flatten'):
                va = va.flatten()
            if hasattr(vb, 'flatten'):
                vb = vb.flatten()

            norm_a = np.linalg.norm(va)
            norm_b = np.linalg.norm(vb)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(va, vb) / (norm_a * norm_b))
        except Exception:
            return 0.0
