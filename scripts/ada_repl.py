"""
Ada interactive REPL — talk to the substrate.

Three commands at the prompt:

    write <text>        — absorb a fact (use 'write key=foo your text' for versioned)
    ask <question>      — targeted retrieval, refuses with 'I don't know'
    think <input>       — pure associative wave, returns activations
    history <key>       — version chain for a stable key
    drift <key>         — cosine drift across the chain
    schema              — what predicates / topics Ada has learned
    aurora              — pre-load the Project Aurora corpus (77 facts)
    quit                — exit

If ANTHROPIC_API_KEY is set, Haiku enriches every absorb and renders
every `ask`. Otherwise a heuristic enricher + Qwen3 (local, if present)
render. Either way the substrate works.

    PYTHONPATH=. python scripts/ada_repl.py
"""

from __future__ import annotations

import sys

from ada.cognitive.generate import build_llm_renderer, build_qwen_renderer
from ada.cognitive.surface import CognitiveSurface
from ada.encoder.llm_enricher import auto_enricher
from ada.memory.glyph_cognitive import GlyphCognitiveLoop
from ada.memory.thought_space import ThoughtGlyphSpace
from ada.skill import SchemaTracker


# ── Banner ────────────────────────────────────────────────────────────

_GRAD = ["\033[38;5;204m", "\033[38;5;177m", "\033[38;5;141m", "\033[38;5;111m"]
_BANNER = [
    " ⣼⢻⡄ ⣿⠛⠛⢻⡄ ⣼⢻⡄",
    "⣼⠃ ⢻⡄⣿  ⢸⡇⣼⠃ ⢻⡄",
    "⣿⠛⠛⢻⡇⣿  ⢸⡇⣿⠛⠛⢻⡇",
    "⠛  ⠘⠃⠛⠛⠛⠛ ⠛  ⠘⠃",
]
D = "\033[38;5;244m"
P = "\033[38;5;141m"
C = "\033[38;5;116m"
R = "\033[0m"


def _banner() -> None:
    for i, line in enumerate(_BANNER):
        print(f"  {_GRAD[min(i, len(_GRAD)-1)]}\033[1m{line}{R}")
    print(f"  {D}cognitive substrate v0.6.0{R}\n")


# ── REPL ──────────────────────────────────────────────────────────────

HELP = """  commands:
    write <text>           — absorb a fact
    write key=K <text>     — absorb as next version of K
    ask <question>         — targeted retrieval (refuses 'I don't know')
    think <input>          — pure associative wave
    history <key>          — version chain
    drift <key>            — cosine drift across the chain
    schema                 — predicates/topics learned so far
    aurora                 — pre-load the Aurora corpus (77 facts)
    help / ?               — this help
    quit / exit            — leave
"""


def main() -> None:
    enricher = auto_enricher()
    space = ThoughtGlyphSpace(enricher=enricher)
    tracker = SchemaTracker(space, history_path="docs/repl_schema.md")
    cognitive = GlyphCognitiveLoop(space)
    renderer = build_llm_renderer() or build_qwen_renderer()
    surface = CognitiveSurface(space, cognitive, renderer=renderer)

    _banner()
    if renderer is None:
        print(f"  {D}(no LLM renderer — ask will return raw facts, not prose){R}")
    print(f"  {D}type 'help' for commands, 'quit' to exit{R}\n")

    while True:
        try:
            raw = input(f"  {P}ada{D}>{R} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        if raw in ("quit", "exit", "q"):
            break
        if raw in ("help", "?"):
            print(HELP)
            continue
        if raw == "aurora":
            _load_aurora(space, tracker)
            continue
        if raw == "schema":
            s = tracker.summary()
            print(f"  {D}predicates known: {s['predicates_known']}{R}")
            print(f"  {D}topics known    : {s['topics_known']}{R}")
            print(f"  {D}top predicates  : {s['top_predicates']}{R}")
            print(f"  {D}top topics      : {s['top_topics']}{R}")
            continue
        if raw.startswith("history "):
            key = raw[len("history "):].strip()
            hist = space.history(key)
            if not hist:
                print(f"  {D}no history for {key!r}{R}")
                continue
            for st in hist:
                v = st.metadata.get("_version")
                print(f"  {D}v{v}: {R}{st.content}")
            continue
        if raw.startswith("drift "):
            key = raw[len("drift "):].strip()
            d = space.drift(key)
            if d.get("drift_rate") is None:
                print(f"  {D}only {d.get('versions', 0)} version(s) — drift needs ≥2{R}")
                continue
            print(f"  {D}versions: {d['versions']}{R}")
            cons = ", ".join(f"{x:+.3f}" for x in d["consecutive_similarities"])
            print(f"  {D}consecutive sims:{R} [{cons}]")
            print(f"  {D}first→last sim  :{R} {d['first_to_last_similarity']:+.3f}")
            print(f"  {D}drift rate      :{R} {d['drift_rate']:+.3f}")
            continue
        if raw.startswith("write "):
            body = raw[len("write "):].strip()
            key = None
            if body.startswith("key="):
                parts = body.split(" ", 1)
                key = parts[0][len("key="):]
                body = parts[1] if len(parts) > 1 else ""
            if not body:
                print(f"  {D}usage: write [key=K] <text>{R}")
                continue
            stored = space.absorb(body, key=key)
            if stored is None:
                print(f"  {D}(duplicate, not absorbed){R}")
                continue
            tracker.observe(stored)
            v = stored.metadata.get("_version")
            tag = f" v{v}" if v else ""
            print(f"  {D}absorbed{tag}: {R}{stored.thought_id}")
            continue
        if raw.startswith("ask "):
            q = raw[len("ask "):].strip()
            a = surface.ask(q)
            if a.refused:
                print(f"  {C}{a.rendered}{R}  {D}(conf={a.confidence:.2f}){R}")
            else:
                print(f"  {C}{a.rendered}{R}")
                print(f"  {D}grounded in: {a.fact.content if a.fact else '—'}{R}")
            continue
        if raw.startswith("think "):
            inp = raw[len("think "):].strip()
            act = surface.think(inp, top_k=6)
            if not act.thoughts:
                print(f"  {D}(no activation){R}")
                continue
            print(f"  {D}{len(act)} thoughts activated"
                  f"{f' (pathway: {act.pathway})' if act.pathway else ''}{R}")
            for t, s in list(act)[:6]:
                print(f"    {D}[{s:+.2f}]{R}  {t.content}")
            continue
        # Default: treat as `ask`
        a = surface.ask(raw)
        if a.refused:
            print(f"  {C}{a.rendered}{R}  {D}(conf={a.confidence:.2f}){R}")
        else:
            print(f"  {C}{a.rendered}{R}")

    print(f"  {D}substrate held {space.count} thoughts at exit.{R}")


# ── Aurora preloader ─────────────────────────────────────────────────

def _load_aurora(space, tracker) -> None:
    """Load the 77-fact Aurora corpus."""
    sys.path.insert(0, "scripts")
    from hard_test import SESSION_1
    n_loaded = 0
    for text, key in SESSION_1:
        stored = space.absorb(text, key=key)
        if stored:
            tracker.observe(stored)
            n_loaded += 1
    print(f"  {D}loaded {n_loaded}/{len(SESSION_1)} Aurora facts.{R}")


if __name__ == "__main__":
    main()
