"""
LLM-at-build-time fact enricher.

Parses incoming facts into a structured `{subject, predicate, object, topic}`
tuple so the HDC encoder can put each part of a fact into a discriminative
binding role. This is the "cortical interpretation" step — the brain doesn't
store raw sentences, it stores parsed structure.

Two enrichers:

  HaikuEnricher      — one Claude Haiku call per fact, cached by content hash.
                       Uses tool-style structured output for reliability.
                       Activates if ANTHROPIC_API_KEY is set and the SDK
                       is importable; falls through to HeuristicEnricher
                       otherwise.

  HeuristicEnricher  — regex patterns for common templates. Works offline;
                       narrower coverage but enough for demos and held-out
                       tests on similar-shaped corpora.

The enricher returns the SAME shape regardless of which backend is in use,
so the substrate stays agnostic. Failures degrade to topic="general" with
an empty triple, which is no worse than the un-enriched path.

Usage:
    from ada.encoder.llm_enricher import auto_enricher
    enr = auto_enricher()
    fact = enr.enrich("Bob's favorite color is red.")
    # → StructuredFact(subject="Bob", predicate="favorite_color",
    #                  object="red", topic="person.preference")
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


@dataclass
class StructuredFact:
    """Parsed representation of a single fact."""
    text: str
    subject: str = ""
    predicate: str = ""
    object: str = ""
    topic: str = "general"

    def is_structured(self) -> bool:
        return bool(self.subject or self.predicate or self.object)


# ── Protocol ─────────────────────────────────────────────────────────

class FactEnricher(Protocol):
    """Anything that turns text → StructuredFact."""
    def enrich(self, text: str) -> StructuredFact: ...


# ── Heuristic enricher (offline) ─────────────────────────────────────

def _strip_punct(s: str) -> str:
    return s.strip().strip("?.,!").lower()


def _try_facts(text: str) -> Optional[StructuredFact]:
    """Statement-shaped facts. Returns None if nothing matches."""

    # "X's favorite Y is Z."  ─ predicate = "favorite_Y"
    m = re.match(r"^(\w+)(?:'s|s)\s+favorite\s+(\w+)\s+is\s+(.+?)\.?$", text, re.I)
    if m:
        return StructuredFact(text=text,
            subject=_strip_punct(m.group(1)),
            predicate=f"favorite_{m.group(2).lower()}",
            object=_strip_punct(m.group(3)),
            topic="person.preference")

    # "X is N years old."  ─ predicate = "age"
    m = re.match(r"^(\w+)\s+is\s+(\d+)\s+years?\s+old\.?$", text, re.I)
    if m:
        return StructuredFact(text=text,
            subject=_strip_punct(m.group(1)), predicate="age",
            object=m.group(2), topic="person.age")

    # "X lives in Y."  ─ predicate = "residence"
    m = re.match(r"^(\w+)\s+lives?\s+in\s+(.+?)\.?$", text, re.I)
    if m:
        return StructuredFact(text=text,
            subject=_strip_punct(m.group(1)), predicate="residence",
            object=_strip_punct(m.group(2)), topic="person.location")

    # "X works as (a|an) Y."  ─ predicate = "job"
    m = re.match(r"^(\w+)\s+works?\s+as\s+(?:a|an)?\s*(.+?)\.?$", text, re.I)
    if m:
        return StructuredFact(text=text,
            subject=_strip_punct(m.group(1)), predicate="job",
            object=_strip_punct(m.group(2)), topic="person.job")

    # "The capital of X is Y."  ─ predicate = "capital"
    m = re.match(r"^The\s+capital\s+of\s+(.+?)\s+is\s+(.+?)\.?$", text, re.I)
    if m:
        return StructuredFact(text=text,
            subject=_strip_punct(m.group(1)), predicate="capital",
            object=_strip_punct(m.group(2)), topic="geography.capital")

    # "X stands for Y."  ─ predicate = "abbreviation_of"
    m = re.match(r"^(\w+)\s+stands\s+for\s+(.+?)\.?$", text, re.I)
    if m:
        return StructuredFact(text=text,
            subject=_strip_punct(m.group(1)), predicate="abbreviation_of",
            object=_strip_punct(m.group(2)), topic="knowledge.definition")

    # "The X has N Y."  ─ predicate = "quantity_of_Y"
    m = re.match(r"^The\s+(\w+(?:\s+\w+)?)\s+has\s+(\d+)\s+(\w+)\.?$", text, re.I)
    if m:
        return StructuredFact(text=text,
            subject=_strip_punct(m.group(1)),
            predicate=f"quantity_of_{m.group(3).lower()}",
            object=m.group(2), topic="knowledge.quantity")

    # "Ada uses N-dimensional Y."  ─ predicate = "dimension"
    m = re.match(r"^(\w+)\s+uses\s+(\d+)-dimensional\s+(.+?)\.?$", text, re.I)
    if m:
        return StructuredFact(text=text,
            subject=_strip_punct(m.group(1)), predicate="dimension",
            object=m.group(2), topic="ada.architecture")

    # "The Ada server runs on port N."
    m = re.match(r"^The\s+(\w+)\s+server\s+runs\s+on\s+port\s+(\d+)\.?$", text, re.I)
    if m:
        return StructuredFact(text=text,
            subject=_strip_punct(m.group(1)), predicate="port",
            object=m.group(2), topic="ada.architecture")

    return None


def _try_queries(text: str) -> Optional[StructuredFact]:
    """Question-shaped queries → same triple shape so they hash to the
    same structural keys as the facts that should answer them."""

    # "What is X's age?" / "How old is X?"  ─ predicate = "age"
    m = re.match(r"^(?:what\s+is\s+(\w+)(?:'s|s)\s+age|how\s+old\s+is\s+(\w+))\??$",
                 text, re.I)
    if m:
        subj = m.group(1) or m.group(2)
        return StructuredFact(text=text, subject=_strip_punct(subj),
            predicate="age", topic="person.age")

    # "Where does X live?" / "Where is X?"  ─ predicate = "residence"
    m = re.match(r"^where\s+does\s+(\w+)\s+live\??$", text, re.I)
    if m:
        return StructuredFact(text=text, subject=_strip_punct(m.group(1)),
            predicate="residence", topic="person.location")

    # "What job does X have?" / "What does X do?"  ─ predicate = "job"
    m = re.match(r"^(?:what\s+job\s+does\s+(\w+)\s+have|what\s+does\s+(\w+)\s+do)\??$",
                 text, re.I)
    if m:
        subj = m.group(1) or m.group(2)
        return StructuredFact(text=text, subject=_strip_punct(subj),
            predicate="job", topic="person.job")

    # "What is X's favorite Y?"  ─ predicate = "favorite_Y"
    m = re.match(r"^what\s+is\s+(\w+)(?:'s|s)\s+favorite\s+(\w+)\??$", text, re.I)
    if m:
        return StructuredFact(text=text, subject=_strip_punct(m.group(1)),
            predicate=f"favorite_{m.group(2).lower()}",
            topic="person.preference")
    # "What color does X like?"
    m = re.match(r"^what\s+color\s+does\s+(\w+)\s+like\??$", text, re.I)
    if m:
        return StructuredFact(text=text, subject=_strip_punct(m.group(1)),
            predicate="favorite_color", topic="person.preference")

    # "What is the capital of X?" / "Capital of X?" / "Capital city of X?"
    m = re.match(r"^(?:what\s+is\s+the\s+capital\s+of|where\s+is\s+the\s+capital\s+of|"
                 r"capital\s+(?:city\s+)?of|what\s+is\s+(\w+)(?:'s|s)\s+capital)\s*(\w+)?\??$",
                 text, re.I)
    if m:
        subj = m.group(2) or m.group(1)
        if subj:
            return StructuredFact(text=text, subject=_strip_punct(subj),
                predicate="capital", topic="geography.capital")

    # "What does X stand for?"
    m = re.match(r"^what\s+does\s+(\w+)\s+stand\s+for\??$", text, re.I)
    if m:
        return StructuredFact(text=text, subject=_strip_punct(m.group(1)),
            predicate="abbreviation_of", topic="knowledge.definition")

    # "What dimension are X's vectors?"
    m = re.match(r"^what\s+dimension\s+are\s+(\w+)(?:'s|s)\s+vectors\??$", text, re.I)
    if m:
        return StructuredFact(text=text, subject=_strip_punct(m.group(1)),
            predicate="dimension", topic="ada.architecture")

    # "What port does X run on?"
    m = re.match(r"^what\s+port\s+does\s+(\w+)\s+run\s+on\??$", text, re.I)
    if m:
        return StructuredFact(text=text, subject=_strip_punct(m.group(1)),
            predicate="port", topic="ada.architecture")

    return None


class HeuristicEnricher:
    """Regex-based fact + query parser. Cheap, offline, narrow coverage."""

    def enrich(self, text: str) -> StructuredFact:
        clean = text.strip()
        return _try_facts(clean) or _try_queries(clean) or StructuredFact(text=clean)


# ── Haiku enricher (online) ──────────────────────────────────────────

_HAIKU_SYSTEM = (
    "You parse a fact into a structured tuple. Return JSON with exactly "
    "these keys: subject, predicate, object, topic. "
    "predicate is a snake_case verb-like attribute name "
    "(age, residence, job, capital, favorite_color, abbreviation_of, etc.). "
    "topic is a dotted hierarchical tag "
    "(person.age, person.location, geography.capital, knowledge.definition, "
    "science.constant, etc.). "
    "Be terse — no explanation, just JSON."
)


class HaikuEnricher:
    """Claude Haiku as a build-time parser. One call per fact, cached."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        import anthropic
        self._client = anthropic.Anthropic()
        self._model = model
        self._cache: dict[str, StructuredFact] = {}

    def enrich(self, text: str) -> StructuredFact:
        key = hashlib.sha256(text.encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        try:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=120,
                system=_HAIKU_SYSTEM,
                messages=[{"role": "user", "content": f"Fact: {text}"}],
            )
            import json
            raw = resp.content[0].text.strip()
            # Strip markdown fences if present.
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
            data = json.loads(raw)
            sf = StructuredFact(
                text=text,
                subject=str(data.get("subject", "")).strip().lower(),
                predicate=str(data.get("predicate", "")).strip().lower().replace(" ", "_"),
                object=str(data.get("object", "")).strip().lower(),
                topic=str(data.get("topic", "general")).strip().lower(),
            )
        except Exception as e:
            logger.warning("Haiku enrich failed for %r: %s — falling back to heuristic", text[:40], e)
            sf = HeuristicEnricher().enrich(text)
        self._cache[key] = sf
        return sf


# ── Factory ──────────────────────────────────────────────────────────

def auto_enricher() -> FactEnricher:
    """Pick the best available enricher: Haiku if configured, else heuristic."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # noqa: F401
            logger.info("Enricher: HaikuEnricher (LLM-at-build-time, cached)")
            return HaikuEnricher()
        except ImportError:
            pass
    logger.info("Enricher: HeuristicEnricher (regex patterns, offline)")
    return HeuristicEnricher()
