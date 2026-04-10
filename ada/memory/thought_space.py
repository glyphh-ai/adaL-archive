"""
ThoughtGlyphSpace — Ada's long-term memory.

Orchestrates encoder + primitives + storage. Absorbs natural language
into structured Thought Glyphs, stores them, and recalls by layered
similarity search.

Two storage backends:
  - pgvector (production): HNSW-indexed, unlimited capacity
  - memory (dev/test): In-memory numpy, no external dependencies

Active recall always searches ALL stored thoughts (long-term memory).
The DreamLoop's working set is a separate in-memory subset.

Usage:
    from ada.memory.thought_space import ThoughtGlyphSpace

    space = ThoughtGlyphSpace()  # in-memory backend
    space.absorb("my name is chris", speaker="incoming")
    results = space.recall("what is my name", top_k=5)
    # → [("my name is chris", 0.37, glyph), ...]
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ada.core.ops import cosine_similarity
from ada.core.types import Glyph
from ada.memory.primitives import PrimitiveSpace
from ada.memory.thought_glyph import ThoughtGlyphEncoder

logger = logging.getLogger(__name__)


# ── Stored thought metadata ──────────────────────────────────────────────

@dataclass
class StoredThought:
    """A thought in long-term memory."""
    thought_id: str
    content: str
    speaker: str
    glyph: Glyph
    strength: float = 1.0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def reinforce(self, amount: float = 0.1) -> None:
        """Hebbian strengthening with diminishing returns.

        strength += amount / (1 + 0.1 * strength)
        Getting from 1.0→2.0 is easy, 2.0→3.0 is hard.
        """
        self.strength = min(3.0, self.strength + amount / (1.0 + 0.1 * self.strength))
        self.last_accessed = time.time()
        self.access_count += 1

    def decay(self, factor: float = 0.98) -> None:
        """Weaken unused thought. Called by dream loop Slow-Wave."""
        self.strength = max(0.0, self.strength * factor)

    @property
    def age_hours(self) -> float:
        """Hours since creation."""
        return (time.time() - self.created_at) / 3600.0

    @property
    def idle_hours(self) -> float:
        """Hours since last access."""
        return (time.time() - self.last_accessed) / 3600.0

    @property
    def is_archivable(self) -> bool:
        """Below threshold — candidate for cold storage."""
        return self.strength < 0.05


@dataclass
class RecallResult:
    """A recall result with similarity scores."""
    thought: StoredThought
    global_similarity: float
    layer_similarities: dict[str, float] = field(default_factory=dict)


# ── ThoughtGlyphSpace ────────────────────────────────────────────────────

class ThoughtGlyphSpace:
    """Ada's long-term memory — absorb, store, recall.

    In-memory backend for now. pgvector backend will be added when
    Docker is running — the interface stays the same.

    Args:
        primitives: PrimitiveSpace (loaded). If None, creates and loads one.
        encoder: ThoughtGlyphEncoder. If None, creates one from primitives.
    """

    def __init__(
        self,
        primitives: PrimitiveSpace | None = None,
        encoder: ThoughtGlyphEncoder | None = None,
    ) -> None:
        # Primitives
        if primitives is None:
            primitives = PrimitiveSpace()
            primitives.load()
        self._primitives = primitives

        # Encoder
        if encoder is None:
            encoder = ThoughtGlyphEncoder(primitives=primitives)
        self._encoder = encoder

        # Storage (in-memory for now)
        self._thoughts: dict[str, StoredThought] = {}
        self._absorbed_texts: set[str] = set()  # dedup
        self._archive: list[StoredThought] = []  # cold storage
        self._primitives_version: int = primitives.version  # track for re-encoding

    @property
    def primitives(self) -> PrimitiveSpace:
        return self._primitives

    @property
    def encoder(self) -> ThoughtGlyphEncoder:
        return self._encoder

    @property
    def count(self) -> int:
        return len(self._thoughts)

    # ── Absorb ────────────────────────────────────────────────────────────

    def absorb(
        self,
        text: str,
        speaker: str = "incoming",
        metadata: dict[str, Any] | None = None,
    ) -> StoredThought | None:
        """Encode a thought and store it in long-term memory.

        Deduplicates by exact text match. Returns None if already absorbed.

        Args:
            text: Natural language text.
            speaker: "incoming" (user) or "outgoing" (Ada).
            metadata: Optional metadata.

        Returns:
            StoredThought, or None if duplicate.
        """
        text_key = text.strip().lower()
        if not text_key:
            return None
        if text_key in self._absorbed_texts:
            return None

        glyph = self._encoder.encode_thought(text, speaker=speaker, metadata=metadata)
        thought_id = str(uuid.uuid4())[:12]

        stored = StoredThought(
            thought_id=thought_id,
            content=text,
            speaker=speaker,
            glyph=glyph,
            metadata=metadata or {},
        )

        self._thoughts[thought_id] = stored
        self._absorbed_texts.add(text_key)
        logger.debug("Absorbed thought %s: %s", thought_id, text[:50])
        return stored

    # ── Recall ────────────────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.01,
        speaker: str = "incoming",
    ) -> list[RecallResult]:
        """Search ALL long-term memory for thoughts matching a query.

        Three-signal search (following ada-code's layered re-ranking):
          1. Content cosine — pure content word match (primary signal, 0.50)
          2. Layer cosine — weighted per-layer comparison (structure, 0.35)
          3. Structural overlap — Jaccard of activated segments (bonus, 0.15)

        Content similarity is the dominant signal, like ada-code's
        content layer (0.50 weight). This avoids the global cortex noise
        problem where bundled vectors dilute specific word matches.

        Args:
            query: Natural language query.
            top_k: Max results to return.
            min_similarity: Minimum combined similarity threshold.
            speaker: Speaker context for the query encoding.

        Returns:
            List of RecallResult sorted by combined similarity.
        """
        if not self._thoughts:
            return []

        # Re-encode if primitives changed (crystallization added new compounds).
        # Keeps content vectors in sync so query↔stored similarity is symmetric.
        self._refresh_if_needed()

        query_glyph = self._encoder.encode_thought(query, speaker=speaker)

        # ── Signal 1: Content vector (primary — like ada-code content layer)
        query_content_vec = query_glyph.metadata.get("_content_vector")
        query_activated = set(query_glyph.metadata.get("_activated_attrs", []))

        _LAYER_WEIGHTS = {
            "perspective": 0.25,
            "semantic": 0.30,
            "relational": 0.25,
            "temporal": 0.10,
            "direction": 0.10,
        }

        # Normalize query for self-match exclusion
        query_normalized = query.strip().lower().rstrip("?.! ")

        results: list[RecallResult] = []
        for stored in self._thoughts.values():
            # Skip self-matches — don't recall the question itself
            stored_normalized = stored.content.strip().lower().rstrip("?.! ")
            if stored_normalized == query_normalized:
                continue

            # ── Content similarity ──────────────────────────────
            content_sim = 0.0
            stored_content_vec = stored.glyph.metadata.get("_content_vector")
            if query_content_vec is not None and stored_content_vec is not None:
                content_sim = float(cosine_similarity(
                    query_content_vec, stored_content_vec,
                ))
                content_sim = max(0.0, content_sim)

            # ── Layer similarity (weighted per-layer role cosine) ──
            stored_activated = set(
                stored.glyph.metadata.get("_activated_attrs", [])
            )

            layer_sims: dict[str, float] = {}
            total_weighted_sim = 0.0
            total_weight = 0.0

            for layer_name, query_layer in query_glyph.layers.items():
                if layer_name not in stored.glyph.layers:
                    continue
                stored_layer = stored.glyph.layers[layer_name]
                layer_weight = _LAYER_WEIGHTS.get(layer_name, 0.1)

                layer_sim_sum = 0.0
                layer_seg_count = 0

                for seg_name, query_seg in query_layer.segments.items():
                    attr_key = f"{layer_name}_{seg_name}"
                    if attr_key not in query_activated:
                        continue
                    layer_seg_count += 1

                    if attr_key not in stored_activated:
                        continue
                    if seg_name not in stored_layer.segments:
                        continue
                    stored_seg = stored_layer.segments[seg_name]

                    for role_name, query_role_vec in query_seg.roles.items():
                        if role_name not in stored_seg.roles:
                            continue
                        stored_role_vec = stored_seg.roles[role_name]
                        rsim = float(cosine_similarity(
                            query_role_vec.data,
                            stored_role_vec.data,
                        ))
                        layer_sim_sum += max(0.0, rsim)

                if layer_seg_count > 0:
                    layer_sim = layer_sim_sum / layer_seg_count
                    layer_sims[layer_name] = layer_sim
                    total_weighted_sim += layer_sim * layer_weight
                    total_weight += layer_weight

            role_sim = total_weighted_sim / total_weight if total_weight > 0 else 0.0

            # ── Structural overlap (Jaccard of activated segments) ──
            seg_overlap = len(query_activated & stored_activated)
            seg_total = len(query_activated | stored_activated) or 1
            structural = seg_overlap / seg_total

            # ── Combine: adaptive weights (like ada-code re-ranking) ──
            # When query has distinctive content words, content dominates.
            # When query is all primitives ("who am i?"), lean on structure.
            n_content = len(query_glyph.metadata.get("_content_words", []))
            if n_content >= 2:
                w_content, w_role, w_struct = 0.55, 0.30, 0.15
            elif n_content == 1:
                w_content, w_role, w_struct = 0.40, 0.40, 0.20
            else:
                w_content, w_role, w_struct = 0.10, 0.65, 0.25

            similarity = (
                content_sim * w_content
                + role_sim * w_role
                + structural * w_struct
            )

            # ── Strength boost (Hebbian) ─────────────────────────
            # range: 1.0 (fresh, strength=1.0) to ~2.4 (max strength=3.0)
            import math
            strength_boost = 1.0 + math.log(1.0 + stored.strength)

            # ── Recency boost (exponential decay with floor) ─────
            # Recent memories score higher; old ones keep 30% weight
            _RECENCY_FLOOR = 0.3
            _RECENCY_LAMBDA = 0.05  # per hour
            recency_boost = _RECENCY_FLOOR + (1.0 - _RECENCY_FLOOR) * math.exp(
                -_RECENCY_LAMBDA * stored.idle_hours
            )

            combined = similarity * strength_boost * recency_boost

            if combined >= min_similarity:
                results.append(RecallResult(
                    thought=stored,
                    global_similarity=combined,
                    layer_similarities=layer_sims,
                ))

        results.sort(key=lambda r: r.global_similarity, reverse=True)
        top_results = results[:top_k]

        # ── Hebbian reinforcement — recalled memories get stronger ──
        for r in top_results:
            r.thought.reinforce(amount=0.1)

        return top_results

    def recall_by_layer(
        self,
        query: str,
        layer_name: str,
        top_k: int = 5,
        speaker: str = "incoming",
    ) -> list[RecallResult]:
        """Search by a specific layer only.

        Useful for targeted queries: "search perspective layer for
        thoughts about self".
        """
        if not self._thoughts:
            return []

        query_glyph = self._encoder.encode_thought(query, speaker=speaker)
        if layer_name not in query_glyph.layers:
            return []

        query_layer_cortex = query_glyph.layers[layer_name].cortex.data
        results: list[RecallResult] = []

        for stored in self._thoughts.values():
            if layer_name not in stored.glyph.layers:
                continue
            sim = float(cosine_similarity(
                query_layer_cortex,
                stored.glyph.layers[layer_name].cortex.data,
            ))
            results.append(RecallResult(
                thought=stored,
                global_similarity=sim,
                layer_similarities={layer_name: sim},
            ))

        results.sort(key=lambda r: r.global_similarity, reverse=True)
        return results[:top_k]

    # ── Primitive version tracking ─────────────────────────────────────────

    def _refresh_if_needed(self) -> None:
        """Re-encode all stored thoughts if primitives have changed.

        Crystallization adds new compound primitives, which changes how
        words are classified (content vs. structural). Without re-encoding,
        stored thoughts have stale content vectors that don't match queries
        encoded with the new primitives.

        Cheap for in-memory storage (<50ms for ~100 thoughts at 2000D).
        """
        current_version = self._primitives.version
        if current_version == self._primitives_version:
            return

        logger.info(
            "Primitives changed (v%d → v%d), re-encoding %d thoughts",
            self._primitives_version, current_version, len(self._thoughts),
        )

        for stored in self._thoughts.values():
            new_glyph = self._encoder.encode_thought(
                stored.content, speaker=stored.speaker, metadata=stored.metadata,
            )
            stored.glyph = new_glyph

        self._primitives_version = current_version

    # ── Working set for DreamLoop ─────────────────────────────────────────

    def load_working_set(
        self,
        strategy: str = "recent",
        size: int = 50,
    ) -> list[StoredThought]:
        """Load a subset of thoughts for the DreamLoop.

        Strategies:
          - "recent": Most recently absorbed
          - "weak": Lowest strength (need reinforcement or pruning)
          - "random": Random sample for diversity

        Args:
            strategy: Selection strategy.
            size: Number of thoughts to load.

        Returns:
            List of StoredThought.
        """
        thoughts = list(self._thoughts.values())
        if not thoughts:
            return []

        if strategy == "recent":
            thoughts.sort(key=lambda t: t.created_at, reverse=True)
        elif strategy == "weak":
            thoughts.sort(key=lambda t: t.strength)
        elif strategy == "random":
            np.random.shuffle(thoughts)

        return thoughts[:size]

    # ── Maintenance ───────────────────────────────────────────────────────

    def reinforce(self, thought_id: str, amount: float = 0.1) -> None:
        """Strengthen a thought (Hebbian reinforcement)."""
        if thought_id in self._thoughts:
            self._thoughts[thought_id].reinforce(amount)

    def decay_all(self, factor: float = 0.98, idle_threshold_hours: float = 1.0) -> int:
        """Decay idle thoughts, archive weak ones.

        Only decays thoughts that haven't been accessed recently.
        Active memories are untouched — Hebbian reinforcement keeps them strong.

        Returns count of archived (moved to cold storage).
        """
        archived = 0
        to_archive = []
        for tid, thought in self._thoughts.items():
            # Only decay if idle — active memories survive
            if thought.idle_hours > idle_threshold_hours:
                thought.decay(factor)

            # Archive threshold — not deleted, moved to cold storage
            if thought.is_archivable:
                to_archive.append(tid)
                archived += 1

        for tid in to_archive:
            thought = self._thoughts[tid]
            # Store in archive before removing
            self._archive.append(thought)
            text_key = thought.content.strip().lower()
            self._absorbed_texts.discard(text_key)
            del self._thoughts[tid]

        return archived

    def restore_from_archive(self, content_fragment: str) -> StoredThought | None:
        """Restore an archived thought back to active memory.

        Used by dream loop when a pattern references an archived thought.
        Restored with partial strength (0.5).
        """
        for i, thought in enumerate(self._archive):
            if content_fragment.lower() in thought.content.lower():
                thought.strength = 0.5
                thought.last_accessed = time.time()
                self._thoughts[thought.thought_id] = thought
                self._absorbed_texts.add(thought.content.strip().lower())
                self._archive.pop(i)
                logger.info("Restored archived thought: %s", thought.content[:50])
                return thought
        return None

    @property
    def archive_count(self) -> int:
        return len(self._archive)

    def clear(self) -> None:
        """Clear all thoughts (hard reset)."""
        self._thoughts.clear()
        self._absorbed_texts.clear()

    # ── Introspection ─────────────────────────────────────────────────────

    def get_thought(self, thought_id: str) -> StoredThought | None:
        return self._thoughts.get(thought_id)

    def all_thoughts(self) -> list[StoredThought]:
        """All thoughts sorted by strength (strongest first)."""
        return sorted(self._thoughts.values(), key=lambda t: t.strength, reverse=True)

    def stats(self) -> dict:
        return {
            "count": len(self._thoughts),
            "primitives": self._primitives.stats(),
            "avg_strength": (
                sum(t.strength for t in self._thoughts.values()) / len(self._thoughts)
                if self._thoughts else 0.0
            ),
        }

    def format_recall(self, results: list[RecallResult], max_results: int = 5) -> str:
        """Format recall results for injection into the LLM prompt.

        Translates speaker tags into natural language so the LLM
        understands perspective: who said what.
        """
        if not results:
            return ""

        lines = []
        for r in results[:max_results]:
            t = r.thought
            who = "the user said" if t.speaker == "incoming" else "Ada said"
            lines.append(
                f"[{r.global_similarity:.2f}] {who}: \"{t.content}\""
            )

        return "Ada remembers:\n" + "\n".join(lines)
