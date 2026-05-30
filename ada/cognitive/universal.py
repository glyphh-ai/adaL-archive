"""
Universal schema prototype — the substrate done right.

Instead of growing ad-hoc predicates ("favorite_color", "has_dream_loop",
"quantity_of_bones") as Haiku invents them, every fact gets MAPPED into
a fixed universal shape. The substrate stays stable; queries steer by
adjusting layer/role weights at recall time.

LAYERS (fixed):
    entity        — what kind of thing is this
    perceptual    — color / size / shape / sound / texture / temperature
    spatial       — location / origin / direction
    temporal      — time / duration / age / era / frequency
    relational    — subject / predicate / object / possessor / agent
    quantitative  — count / magnitude / unit
    epistemic     — source / certainty / modality

Every fact fills SOME of these slots (the rest are absent). Two facts are
similar at the ROLE level if they fill the same role with similar values
(`perceptual.color = blue` matches `perceptual.color = blue` exactly).
They're similar at the LAYER level if many roles in the same layer
overlap. They're similar at the CORTEX level if their global bundles
match.

Same substrate, three abstraction scales of similarity, queryable
independently or in combination.

    ANTHROPIC_API_KEY=... PYTHONPATH=. python -m ada.cognitive.universal
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── The universal schema ─────────────────────────────────────────────

UNIVERSAL_SCHEMA: dict[str, list[str]] = {
    "entity":       ["name", "kind", "subkind"],
    "perceptual":   ["color", "size", "shape", "texture", "sound",
                     "smell", "taste", "temperature"],
    "spatial":      ["location", "origin", "direction"],
    "temporal":     ["time", "duration", "age", "era", "frequency"],
    "relational":   ["subject", "predicate", "object", "possessor",
                     "agent", "patient", "instrument"],
    "quantitative": ["count", "magnitude", "unit", "ratio"],
    "epistemic":    ["source", "certainty", "modality"],
}

ALL_LAYERS = list(UNIVERSAL_SCHEMA)
ALL_ROLES = [(layer, role) for layer, roles in UNIVERSAL_SCHEMA.items() for role in roles]


# ── Vector ops (kept self-contained for the prototype) ───────────────

DIM = 2048
RNG_SEED_BASE = 0xADA0


def _atom_for(token: str) -> np.ndarray:
    """Deterministic bipolar atom for a string token."""
    seed = int(hashlib.sha256(token.lower().encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed ^ RNG_SEED_BASE)
    return (rng.integers(0, 2, DIM) * 2 - 1).astype(np.int8)


def _role_vec(layer: str, role: str) -> np.ndarray:
    return _atom_for(f"role::{layer}.{role}")


def _bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).astype(np.int8)


def _bundle(vecs: list[np.ndarray]) -> np.ndarray:
    if not vecs:
        return np.zeros(DIM, dtype=np.int8)
    s = np.zeros(DIM, dtype=np.int32)
    for v in vecs:
        s += v
    return np.sign(s).astype(np.int8) + (s == 0)  # break ties → +1


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a.astype(np.float32)))
    nb = float(np.linalg.norm(b.astype(np.float32)))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a.astype(np.float32), b.astype(np.float32)) / (na * nb))


# ── Universal glyph ──────────────────────────────────────────────────

@dataclass
class UniversalGlyph:
    """A fact in the universal shape.

    slots[layer][role] = (value, value_vector)  or absent if unfilled.
    Cached: role-vectors (bind(role_vec, value_vec)), layer-vectors
    (bundle of role-vectors in that layer), and the global cortex
    (bundle of all role-vectors). All three are 2048-dim bipolar vectors
    usable for similarity at the appropriate abstraction.
    """
    text: str
    slots: dict[str, dict[str, str]] = field(default_factory=dict)
    role_vecs: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    layer_vecs: dict[str, np.ndarray] = field(default_factory=dict)
    cortex: np.ndarray = field(default_factory=lambda: np.zeros(DIM, dtype=np.int8))

    def filled_roles(self) -> list[tuple[str, str]]:
        return [(layer, role)
                for layer, roles in self.slots.items()
                for role in roles]

    def value_at(self, layer: str, role: str) -> Optional[str]:
        return self.slots.get(layer, {}).get(role)

    @staticmethod
    def encode(text: str, mapped: dict[str, dict[str, str]]) -> "UniversalGlyph":
        # Drop empties.
        slots: dict[str, dict[str, str]] = {}
        for layer, roles in mapped.items():
            if layer not in UNIVERSAL_SCHEMA:
                continue
            filled = {}
            for role, value in (roles or {}).items():
                if role not in UNIVERSAL_SCHEMA[layer]:
                    continue
                if value is None:
                    continue
                v = str(value).strip()
                if not v or v.lower() in ("none", "null", "n/a", ""):
                    continue
                filled[role] = v
            if filled:
                slots[layer] = filled

        # Build vectors: for each filled role, bind(role_vec, value_atom).
        role_vecs: dict[tuple[str, str], np.ndarray] = {}
        layer_vecs: dict[str, np.ndarray] = {}
        all_role_vecs: list[np.ndarray] = []
        for layer, roles in slots.items():
            layer_components = []
            for role, value in roles.items():
                rv = _bind(_role_vec(layer, role), _atom_for(value))
                role_vecs[(layer, role)] = rv
                layer_components.append(rv)
                all_role_vecs.append(rv)
            layer_vecs[layer] = _bundle(layer_components)
        cortex = _bundle(all_role_vecs)
        return UniversalGlyph(
            text=text, slots=slots,
            role_vecs=role_vecs, layer_vecs=layer_vecs, cortex=cortex,
        )


# ── Enricher: LLM maps a fact INTO the fixed shape ──────────────────

_ENRICHER_SYSTEM = (
    "You are a structured-data extractor. Given a fact, fill in the "
    "values of this fixed universal schema. Use ONLY the roles listed. "
    "Leave a role's value blank/null if the fact doesn't mention it. "
    "Do not invent roles. Return strict JSON only.\n\n"
    "SCHEMA:\n" +
    "\n".join(f"  {layer}: {', '.join(roles)}"
              for layer, roles in UNIVERSAL_SCHEMA.items())
)


class UniversalEnricher:
    """One Haiku call per fact, cached. Maps into the fixed schema."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        import anthropic
        self._client = anthropic.Anthropic()
        self._model = model
        self._cache: dict[str, dict] = {}

    def enrich(self, text: str) -> dict[str, dict[str, str]]:
        key = hashlib.sha256(text.encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        prompt = (
            f"Fact: {text}\n\n"
            "Return a JSON object with this exact structure (omit unfilled layers):\n"
            "{\n"
            '  "entity": {"name": "...", "kind": "..."},\n'
            '  "perceptual": {"color": "...", "size": "..."},\n'
            '  "temporal": {"age": "..."},\n'
            "  ...\n"
            "}\n"
        )
        try:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=400,
                system=_ENRICHER_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
            mapped = json.loads(raw)
        except Exception as e:
            logger.warning("Universal enrich failed for %r: %s", text[:40], e)
            mapped = {}
        self._cache[key] = mapped
        return mapped


# ── Universal space + weighted recall ────────────────────────────────

@dataclass
class UniversalMatch:
    glyph: UniversalGlyph
    score: float
    role_scores: dict[tuple[str, str], float]


class UniversalSpace:
    """A substrate of universally-shaped glyphs.

    Queries can adjust layer/role weights at recall time to steer what
    'similar' means for this query — perceptual.color heavy for "what
    color was X?", relational.predicate heavy for "what is X's job?",
    etc. The substrate is the same; the query weights differ.
    """

    def __init__(self, enricher: UniversalEnricher):
        self.enricher = enricher
        self.glyphs: list[UniversalGlyph] = []

    def tell(self, text: str) -> UniversalGlyph:
        mapped = self.enricher.enrich(text)
        g = UniversalGlyph.encode(text, mapped)
        self.glyphs.append(g)
        return g

    def recall(
        self,
        query_text: str,
        weights: Optional[dict[str, float]] = None,
        role_weights: Optional[dict[tuple[str, str], float]] = None,
        top_k: int = 5,
    ) -> list[UniversalMatch]:
        """Find glyphs similar to the query, weighted by layer/role.

        weights        — layer-name → multiplier (default 1.0 each)
        role_weights   — (layer, role) → multiplier (overrides layer weight)

        When the query enriches into roles, similarity is computed
        role-by-role and weighted. When the query is too sparse (a bare
        interrogative like 'how old?'), we fall back to PRESENCE-WEIGHTED
        matching: any fact that fills the role_weights' roles gets a
        proportional boost. This lets queries steer the substrate even
        when the query itself has no concrete values.
        """
        # Encode the query through the same enricher path.
        q_mapped = self.enricher.enrich(query_text)
        q = UniversalGlyph.encode(query_text, q_mapped)

        matches: list[UniversalMatch] = []
        for g in self.glyphs:
            if g.text == query_text:
                continue
            role_scores: dict[tuple[str, str], float] = {}
            total = 0.0
            total_weight = 0.0

            # 1. Value similarity for roles the query filled.
            for (layer, role), qrv in q.role_vecs.items():
                grv = g.role_vecs.get((layer, role))
                w = 1.0
                if weights and layer in weights:
                    w = weights[layer]
                if role_weights and (layer, role) in role_weights:
                    w = role_weights[(layer, role)]
                total_weight += w
                if grv is None:
                    continue
                sim = _cosine(qrv, grv)
                if sim <= 0:
                    continue
                role_scores[(layer, role)] = sim
                total += sim * w

            # 2. Presence-only contribution for role_weights the query
            #    didn't fill but the caller still asked us to steer by.
            #    This makes "how old?" with role_weights={temporal.age: 5}
            #    surface any fact that fills temporal.age, even if Haiku
            #    couldn't enrich the bare interrogative cleanly.
            if role_weights:
                for (layer, role), w in role_weights.items():
                    if (layer, role) in q.role_vecs:
                        continue        # already counted above
                    total_weight += w
                    if (layer, role) in g.role_vecs:
                        role_scores[(layer, role)] = 1.0
                        total += w

            score = total / total_weight if total_weight > 0 else 0.0
            matches.append(UniversalMatch(g, score, role_scores))

        matches.sort(key=lambda m: -m.score)
        return matches[:top_k]


# ── Tiny demo ────────────────────────────────────────────────────────

def _demo() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY required for the universal-schema demo.")
        return
    enricher = UniversalEnricher()
    space = UniversalSpace(enricher)

    facts = [
        "Chris's car is blue.",
        "The sky is blue.",
        "The ocean is blue.",
        "Alice's car is red.",
        "Bob is 45 years old.",
        "Chris is 31 years old.",
        "Diana lives in Denver.",
        "Tokyo is in Japan.",
        "The capital of Japan is Tokyo.",
        "Aurora's boat is 32 feet long.",
    ]
    print("─── telling Ada (universal-schema enriched) ───\n")
    for f in facts:
        g = space.tell(f)
        slots = ", ".join(f"{l}.{r}={v}"
                          for l, roles in g.slots.items()
                          for r, v in roles.items())
        print(f"  • {f}")
        print(f"      → {slots or '(empty)'}\n")

    print("─── query: 'something blue', perceptual.color weighted heavily ───\n")
    matches = space.recall(
        "anything blue",
        role_weights={("perceptual", "color"): 5.0},
        top_k=5,
    )
    for m in matches:
        print(f"  [{m.score:+.2f}]  {m.glyph.text}")

    print("\n─── query: 'how old?', temporal.age weighted heavily ───\n")
    matches = space.recall(
        "how old",
        role_weights={("temporal", "age"): 5.0},
        top_k=5,
    )
    for m in matches:
        print(f"  [{m.score:+.2f}]  {m.glyph.text}")

    print("\n─── query: 'where', spatial.location weighted heavily ───\n")
    matches = space.recall(
        "where",
        role_weights={("spatial", "location"): 5.0},
        top_k=5,
    )
    for m in matches:
        print(f"  [{m.score:+.2f}]  {m.glyph.text}")


if __name__ == "__main__":
    _demo()
