"""
Brain router — deterministic routing of think() requests to capabilities.

Primary: uses the toolrouter capability's HDC encoder + cosine similarity
to match queries against capability descriptions. Sub-10ms, zero tokens.

Fallback: Ada calls Haiku to classify when deterministic confidence is low.
Haiku's answer feeds back into the observation log → dream loop learns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from domains.brain.loader import BrainState, Capability

logger = logging.getLogger(__name__)

# Minimum cosine similarity for a confident deterministic route
CONFIDENCE_THRESHOLD = 0.40


@dataclass
class RouteResult:
    """Where the brain decided to send this request."""
    capability: Optional[str]  # Capability name or None
    confidence: float  # 0.0 to 1.0
    llm_fallback: bool = False  # True if Haiku had to help


class BrainRouter:
    """Routes natural language to the right brain capability.

    At boot, each capability's description + tags get encoded into
    a reference vector using the SDK's encoder. At query time, the
    query gets encoded the same way, and cosine similarity picks
    the best match.

    If no capability matches above threshold, Ada can optionally
    fall back to her internal LLM (Haiku) for classification.
    """

    def __init__(self, brain_state: BrainState, model_manager=None):
        self._state = brain_state
        self._model_manager = model_manager
        self._capability_vectors: dict[str, np.ndarray] = {}
        self._encoder = None
        self._initialized = False

    def initialize(self) -> None:
        """Build capability routing vectors from the loaded router.

        Must be called after capabilities are registered in the DB
        and the toolrouter's encoder is available.
        """
        if not self._state.router:
            logger.warning("No toolrouter capability loaded — routing will use LLM only")
            return

        if not self._model_manager:
            logger.warning("No model manager — cannot initialize router vectors")
            return

        from domains.brain.loader import ADA_ORG_ID

        # Get the toolrouter's runtime encoder from the model manager
        router_key = (ADA_ORG_ID, self._state.router.name)
        loaded = self._model_manager._models.get(router_key)
        if not loaded:
            logger.warning("Toolrouter not in model registry — routing degraded")
            return

        self._encoder = loaded.encoder
        encode_fn = loaded.encode_query_fn

        if not encode_fn or not self._encoder:
            logger.warning("Toolrouter has no encode_query_fn — routing degraded")
            return

        # Encode each capability's description as a routing reference
        from ada.core.types import Concept

        for name, cap in self._state.capabilities.items():
            desc = self._build_routing_text(cap)
            try:
                raw = encode_fn(desc)
                # encode_query returns a dict — wrap as a Concept for the encoder
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
        """Build a representative text for encoding a capability."""
        parts = [cap.manifest.description]
        if cap.manifest.tags:
            parts.append(" ".join(cap.manifest.tags))
        if cap.manifest.category:
            parts.append(cap.manifest.category)
        return " ".join(parts)

    def route(self, query: str) -> RouteResult:
        """Route a query to the best matching capability.

        Returns RouteResult with capability name and confidence.
        Returns None capability if nothing matches above threshold.
        """
        if not self._initialized or not self._encoder:
            return RouteResult(capability=None, confidence=0.0)

        encode_fn = None
        if self._state.router:
            from domains.brain.loader import ADA_ORG_ID
            loaded = self._model_manager._models.get(
                (ADA_ORG_ID, self._state.router.name)
            )
            if loaded:
                encode_fn = loaded.encode_query_fn

        if not encode_fn:
            return RouteResult(capability=None, confidence=0.0)

        try:
            from ada.core.types import Concept
            raw = encode_fn(query)
            if isinstance(raw, dict):
                concept = Concept(
                    name=raw.get("name", "query"),
                    attributes=raw.get("attributes", {}),
                )
            else:
                concept = raw
            query_vector = self._encoder.encode(concept)
        except Exception as e:
            logger.warning(f"Failed to encode query for routing: {e}")
            return RouteResult(capability=None, confidence=0.0)

        # Cosine similarity against all capability vectors
        best_name = None
        best_score = -1.0

        for name, cap_vector in self._capability_vectors.items():
            score = self._cosine_similarity(query_vector, cap_vector)
            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= CONFIDENCE_THRESHOLD and best_name:
            return RouteResult(capability=best_name, confidence=best_score)

        return RouteResult(capability=None, confidence=best_score)

    async def route_with_fallback(self, query: str, llm=None) -> RouteResult:
        """Route with optional LLM fallback for low-confidence queries."""
        result = self.route(query)

        if result.capability is not None:
            return result

        # LLM fallback — ask Haiku to classify
        if llm and llm.available:
            cap_names = list(self._state.capabilities.keys())
            if cap_names:
                chosen = await llm.classify(query, cap_names)
                if chosen and chosen in self._state.capabilities:
                    return RouteResult(
                        capability=chosen,
                        confidence=0.5,  # LLM-assisted routes get moderate confidence
                        llm_fallback=True,
                    )

        return result

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        if isinstance(a, list):
            a = np.array(a)
        if isinstance(b, list):
            b = np.array(b)

        # Handle multi-layer vectors (list of arrays)
        if isinstance(a, np.ndarray) and a.ndim == 1 and isinstance(b, np.ndarray) and b.ndim == 1:
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))

        # For complex encoder outputs, try to get a flat representation
        try:
            if hasattr(a, 'flatten'):
                a = a.flatten()
            if hasattr(b, 'flatten'):
                b = b.flatten()
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))
        except Exception:
            return 0.0
