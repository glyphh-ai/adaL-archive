"""
The universal schema — every fact maps into one fixed shape.

Instead of growing ad-hoc predicates ("favorite_color", "has_dream_loop",
"quantity_of_bones") as an LLM invents them, every fact gets MAPPED into
a fixed 7-layer lattice. The substrate stays stable; structured queries
(count / distribution / intersection) address facts by layer.role slot.

LAYERS (fixed):
    entity        — what kind of thing is this
    perceptual    — color / size / shape / sound / texture / temperature
    spatial       — location / origin / direction
    temporal      — time / duration / age / era / frequency
    relational    — subject / predicate / object / possessor / agent
    quantitative  — count / magnitude / unit
    epistemic     — source / certainty / modality

Every fact fills SOME of these slots (the rest are absent). An absent
slot is a structural ∅ — the substrate refuses rather than confabulates.

``UniversalEnricher`` is the LLM-at-build-time extractor that maps free
text into the schema. It runs once per fact at write time and is cached;
the read path never calls an LLM.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re

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
    """One LLM call per fact, cached. Maps free text into the fixed schema."""

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
