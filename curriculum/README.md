# Curricula

A curriculum is a structured corpus of facts taught to Ada in a defined
order. Each fact is encoded into the universal-schema substrate via
`space.tell_raw()` — no LLM in the path. The substrate accumulates
knowledge that compounds across layers.

## Why structured curricula

LLMs absorb knowledge by training over text. Ada absorbs knowledge by
ingesting **explicit universal-schema bindings**, one fact at a time, in
an order chosen by the curriculum designer. The trade is:

- LLMs: cheap to feed text, expensive to verify what was learned,
  no audit trail for any individual fact.
- Ada: more careful to feed (each fact is a deliberate slot fill),
  every fact is auditable in `docs/`, verifiable via `ask` and `history`.

For a substrate intended to be a brain — not a parrot — the curriculum
shape is the right shape for input.

## Available curricula

| Path | What it teaches | Layers |
|---|---|---|
| [`agi/layer-0/`](agi/layer-0/) | NSM semantic primes — the irreducible cognitive primitives | 0 |
| `agi/layer-1/` | *(planned)* basic perceptual categories (colors, numbers, spatial, logical) | 1 |
| `agi/layer-2/` | *(planned)* physical objects (foods, animals, tools, vehicles) | 2 |
| `domain/aurora.yaml` | *(planned)* the Aurora project as a structured curriculum | n/a |

Each `layer-N` builds on the previous one. Each `domain/` is independent
and assumes layer-0 / layer-1 are already loaded.

## Loading a curriculum

```bash
# Load NSM primes into a fresh substrate, print verification
PYTHONPATH=. python curriculum/agi/layer-0/teach.py
PYTHONPATH=. python curriculum/agi/layer-0/verify.py
```

Each curriculum is two files:

- `*.yaml` — the data (a list of facts, each with universal-schema slots)
- `teach.py` — the loader (reads the yaml, calls `tell_raw` for each fact)
- `verify.py` — optional, queries Ada to confirm what landed

## Adding a new curriculum

1. Create `curriculum/<name>/`.
2. Write a `facts.yaml` (or any name) following the universal-schema
   shape — see `agi/layer-0/primes.yaml` for the canonical example.
3. Copy `agi/layer-0/teach.py` as a starting template; adjust the
   yaml path.
4. (Optional) Write a `verify.py` that asks Ada about what should now
   be there.

The substrate doesn't care what curriculum a fact came from — facts
compose freely across curricula at recall time, just like in a brain.
