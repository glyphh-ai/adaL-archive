"""
Ada interactive REPL — talk to the substrate.

Commands at the prompt:

    write <text>        — absorb a fact (use 'write key=foo your text' for versioned)
    ask <question>      — targeted retrieval, refuses with 'I don't know'
    think <input>       — broad recall, returns matches
    count <l.r>=<v>     — structured count, e.g. count perceptual.color=blue
    top <layer.role>    — distribution, e.g. top spatial.location
    find <l.r>=<v> ...  — slot intersection (ALL conditions must match)
    history <key>       — version chain for a stable key
    schema              — what predicates / topics Ada has learned
    stats               — substrate vital signs
    aurora              — pre-load the Project Aurora corpus (77 facts)
    quit                — exit

If ANTHROPIC_API_KEY is set, the enricher maps every absorb into the
universal schema and an LLM renders every `ask`. Otherwise a heuristic
enricher runs offline. Either way the substrate works.

    PYTHONPATH=. python scripts/ada_repl.py
"""

from __future__ import annotations

import sys

from ada.cognitive.generate import build_llm_renderer
from ada.cognitive.surface import CognitiveSurface
from ada.encoder.llm_enricher import auto_enricher
from ada.memory.thought_space import ThoughtSpace
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
    print(f"  {D}structured memory substrate v0.7.0{R}\n")


# ── REPL ──────────────────────────────────────────────────────────────

HELP = """  commands:
    write <text>           — absorb a fact
    write key=K <text>     — absorb as next version of K
    ask <question>         — targeted retrieval (refuses 'I don't know')
    think <input>          — broad recall
    count <l.r>=<v>        — structured count (count perceptual.color=blue)
    top <layer.role>       — distribution (top spatial.location)
    find <l.r>=<v> ...     — slot intersection, ALL conditions must match
    history <key>          — version chain
    schema                 — predicates/topics learned so far
    stats                  — substrate vital signs
    aurora                 — pre-load the Aurora corpus (77 facts)
    help / ?               — this help
    quit / exit            — leave
"""


def _parse_conditions(body: str) -> dict[str, str] | None:
    """Parse 'layer.role=value layer.role=value' into a conditions dict."""
    conditions: dict[str, str] = {}
    for part in body.split():
        if "=" not in part or "." not in part.split("=", 1)[0]:
            return None
        lr, value = part.split("=", 1)
        conditions[lr] = value
    return conditions or None


def main() -> None:
    enricher = auto_enricher()
    space = ThoughtSpace(enricher=enricher)
    tracker = SchemaTracker(space, history_path="docs/repl_schema.md")
    renderer = build_llm_renderer()
    surface = CognitiveSurface(space, renderer=renderer)

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
        if raw == "stats":
            s = space.stats()
            print(f"  {D}thoughts          : {s['count']}{R}")
            print(f"  {D}versioned keys    : {s['versioned_keys']}{R}")
            print(f"  {D}multi-version keys: {s['multi_version_keys']}{R}")
            print(f"  {D}layer fill        : {s['layer_fill']}{R}")
            continue
        if raw.startswith("count "):
            conditions = _parse_conditions(raw[len("count "):].strip())
            if not conditions:
                print(f"  {D}usage: count layer.role=value{R}")
                continue
            (lr, value), = list(conditions.items())[:1]
            layer, _, role = lr.partition(".")
            n = space.count_where(layer, role, value)
            print(f"  {C}{n}{R}  {D}facts with {lr}={value}{R}")
            continue
        if raw.startswith("top "):
            lr = raw[len("top "):].strip()
            if "." not in lr:
                print(f"  {D}usage: top layer.role{R}")
                continue
            layer, _, role = lr.partition(".")
            dist = space.distribution(layer, role, top_k=5)
            if not dist:
                print(f"  {D}∅ — no facts fill {lr} (structural refusal){R}")
                continue
            for value, n in dist:
                print(f"    {C}{n:>5}{R}  {value}")
            continue
        if raw.startswith("find "):
            conditions = _parse_conditions(raw[len("find "):].strip())
            if not conditions:
                print(f"  {D}usage: find layer.role=value [layer.role=value ...]{R}")
                continue
            hits = space.find_where(conditions)
            if not hits:
                print(f"  {D}∅ — no facts match all conditions (structural refusal){R}")
                continue
            for t in hits[:10]:
                print(f"    {C}•{R} {t.content}")
            if len(hits) > 10:
                print(f"  {D}… and {len(hits) - 10} more{R}")
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
            print(f"  {D}{len(act)} thoughts matched{R}")
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
