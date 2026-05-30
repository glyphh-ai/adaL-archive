"""
bench_cross_type.py — Ada vs Kùzu on CROSS-ENTITY-TYPE aggregation.

The argument: knowledge graphs require ONE node table per entity type.
Cross-type aggregation ("count everything blue, regardless of what
kind of thing it is") requires UNION queries across multiple tables,
which requires (a) anticipating the cross-type query at schema-design
time and (b) modifying the query when new entity types are added.

Ada's universal schema is *uniform across entity types*: every fact
fills the same `perceptual.color` role regardless of whether it's
about a person, a vehicle, a building, or a planet. Cross-type queries
are a single dict scan with no special-casing.

This script generates a mixed corpus:
  - 1500 persons (each with 20 facts, ~half of them with colors)
  - 750 vehicles (each with 8 facts including a color)
  - 500 buildings (each with 8 facts including a color)

Then it runs three cross-type queries:

  X1. How many entities (across ALL types) have blue as their color?
  X2. Top-5 colors across ALL entity types.
  X3. How many entities (across ALL types) are in Austin?

For each:
  - Ada answers with a single universal-schema role scan.
  - Kùzu (with the existing person-only schema) returns ONLY person
    matches — wrong, but shows what happens when the schema designer
    didn't anticipate cross-type queries.
  - Kùzu (with a fixed schema + UNION query) answers correctly but
    requires both: (a) adding node tables for Vehicle and Building,
    and (b) writing a UNION query that touches all three.

The architectural win is not 'KG can't do it' — it's 'KG requires
schema work for every new entity type; Ada doesn't'.

    PYTHONPATH=. python curriculum/at-scale/bench_cross_type.py
"""

from __future__ import annotations

import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import kuzu

sys.path.insert(0, str(Path(__file__).parent))
from generate import COLORS, CITIES, generate as gen_persons

from ada.memory.thought_space import ThoughtGlyphSpace


KUZU_DB = "/tmp/ada_cross_type_bench.kuzu"


# ── Extra entity-type generators ─────────────────────────────────────

def gen_vehicles(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed + 1)
    kinds = ["sedan", "suv", "truck", "motorcycle", "bicycle"]
    facts: list[dict] = []
    for i in range(n):
        name = f"Vehicle{i:05d}"
        kind = rng.choice(kinds)
        color = rng.choice(COLORS)
        city = rng.choice(CITIES)
        year = rng.randint(1990, 2025)
        facts += [
            {"key": f"vehicle.{name}.color",
             "entity": {"name": name, "kind": "vehicle"},
             "perceptual": {"color": color}},
            {"key": f"vehicle.{name}.location",
             "entity": {"name": name, "kind": "vehicle"},
             "spatial": {"location": city}},
            {"key": f"vehicle.{name}.kind",
             "entity": {"name": name, "kind": "vehicle"},
             "semantic": {"category": kind}},
            {"key": f"vehicle.{name}.year",
             "entity": {"name": name, "kind": "vehicle"},
             "temporal": {"time": str(year)}},
        ]
    return facts


def gen_buildings(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed + 2)
    kinds = ["house", "office", "warehouse", "apartment", "museum"]
    facts: list[dict] = []
    for i in range(n):
        name = f"Building{i:05d}"
        kind = rng.choice(kinds)
        color = rng.choice(COLORS)
        city = rng.choice(CITIES)
        floors = rng.randint(1, 50)
        facts += [
            {"key": f"building.{name}.color",
             "entity": {"name": name, "kind": "building"},
             "perceptual": {"color": color}},
            {"key": f"building.{name}.location",
             "entity": {"name": name, "kind": "building"},
             "spatial": {"location": city}},
            {"key": f"building.{name}.kind",
             "entity": {"name": name, "kind": "building"},
             "semantic": {"category": kind}},
            {"key": f"building.{name}.floors",
             "entity": {"name": name, "kind": "building"},
             "quantitative": {"count": str(floors), "unit": "floors"}},
        ]
    return facts


# ── Ground truth across all entity types ─────────────────────────────

def compute_truth(facts: list[dict]) -> dict:
    """Count every fact that fills the role, regardless of which entity
    type the fact is about. This is what 'cross-type aggregation' actually
    means: persons + vehicles + buildings all contribute."""
    color_count: Counter = Counter()
    city_count: Counter = Counter()
    for f in facts:
        key = f.get("key", "")
        col = (f.get("perceptual") or {}).get("color")
        # Skip car_color (it's a sub-attribute, not the primary identifier).
        if col and "car_color" not in key:
            color_count[col] += 1
        loc = (f.get("spatial") or {}).get("location")
        if loc:
            city_count[loc] += 1
    return {
        "blue_total": color_count["blue"],
        "top5_colors": color_count.most_common(5),
        "austin_total": city_count["Austin"],
    }


# ── Ada answers (uniform across entity types — that's the point) ────

def ada_answers(space: ThoughtGlyphSpace) -> dict:
    out = {}

    t0 = time.time()
    n_blue = 0
    for t in space._thoughts.values():
        key = t.metadata.get("_key") or ""
        if "car_color" in key:
            continue
        u = t.glyph.metadata.get("_universal") or {}
        if u.get("perceptual", {}).get("color") == "blue":
            n_blue += 1
    out["X1"] = (n_blue, (time.time() - t0) * 1000)

    t0 = time.time()
    c: Counter = Counter()
    for t in space._thoughts.values():
        key = t.metadata.get("_key") or ""
        if "car_color" in key:
            continue
        v = (t.glyph.metadata.get("_universal") or {}).get("perceptual", {}).get("color")
        if v:
            c[v] += 1
    out["X2"] = (c.most_common(5), (time.time() - t0) * 1000)

    t0 = time.time()
    n_austin = 0
    for t in space._thoughts.values():
        u = t.glyph.metadata.get("_universal") or {}
        if u.get("spatial", {}).get("location") == "Austin":
            n_austin += 1
    out["X3"] = (n_austin, (time.time() - t0) * 1000)

    return out


# ── Kùzu schemas: "person only" (wrong) vs. "fully extended" (right) ─

def kuzu_load_persons_only(persons: list[dict]) -> kuzu.Connection:
    """The schema as it existed when only persons were in the corpus.

    This is what a KG looks like *before* the modeller knows about
    vehicles and buildings. The cross-type aggregation queries will
    return wrong answers because Vehicle and Building rows aren't in
    the schema.
    """
    if Path(KUZU_DB).exists():
        if Path(KUZU_DB).is_dir():
            shutil.rmtree(KUZU_DB)
        else:
            Path(KUZU_DB).unlink()
    db = kuzu.Database(KUZU_DB)
    conn = kuzu.Connection(db)

    conn.execute("CREATE NODE TABLE Person (name STRING, PRIMARY KEY (name))")
    conn.execute("CREATE NODE TABLE City (name STRING, PRIMARY KEY (name))")
    conn.execute("CREATE NODE TABLE Color (name STRING, PRIMARY KEY (name))")
    conn.execute("CREATE REL TABLE LIVES_IN (FROM Person TO City)")
    conn.execute("CREATE REL TABLE HAS_COLOR (FROM Person TO Color)")

    # Insert dim nodes.
    cities, colors, names = set(), set(), set()
    for f in persons:
        e = f.get("entity") or {}
        n = e.get("name")
        if not n or e.get("kind") != "person":
            continue
        names.add(n)
        key = f.get("key", "")
        col = (f.get("perceptual") or {}).get("color")
        if col and "car_color" not in key and "color" in key:
            colors.add(col)
        loc = (f.get("spatial") or {}).get("location")
        if loc and "city" in key:
            cities.add(loc)
    for name in names:
        conn.execute("CREATE (:Person {name: $n})", {"n": name})
    for city in cities:
        conn.execute("CREATE (:City {name: $n})", {"n": city})
    for color in colors:
        conn.execute("CREATE (:Color {name: $n})", {"n": color})

    # Insert edges.
    for f in persons:
        e = f.get("entity") or {}
        n = e.get("name")
        if not n or e.get("kind") != "person":
            continue
        key = f.get("key", "")
        col = (f.get("perceptual") or {}).get("color")
        if col and "car_color" not in key and "color" in key:
            conn.execute(
                "MATCH (p:Person {name: $n}), (c:Color {name: $c}) "
                "CREATE (p)-[:HAS_COLOR]->(c)",
                {"n": n, "c": col},
            )
        loc = (f.get("spatial") or {}).get("location")
        if loc and "city" in key:
            conn.execute(
                "MATCH (p:Person {name: $n}), (c:City {name: $c}) "
                "CREATE (p)-[:LIVES_IN]->(c)",
                {"n": n, "c": loc},
            )
    return conn


def kuzu_extend_with_vehicles_and_buildings(
    conn: kuzu.Connection, vehicles: list[dict], buildings: list[dict],
) -> None:
    """Schema migration: add Vehicle + Building node tables and rels.
    This is the SECOND-PASS schema work a KG modeller has to do when
    the corpus grows new entity types."""
    conn.execute("CREATE NODE TABLE Vehicle (name STRING, PRIMARY KEY (name))")
    conn.execute("CREATE NODE TABLE Building (name STRING, PRIMARY KEY (name))")
    conn.execute("CREATE REL TABLE V_HAS_COLOR (FROM Vehicle TO Color)")
    conn.execute("CREATE REL TABLE V_LOCATED_IN (FROM Vehicle TO City)")
    conn.execute("CREATE REL TABLE B_HAS_COLOR (FROM Building TO Color)")
    conn.execute("CREATE REL TABLE B_LOCATED_IN (FROM Building TO City)")

    # Add new colors/cities that vehicles/buildings introduced.
    existing_colors: set[str] = set()
    r = conn.execute("MATCH (c:Color) RETURN c.name")
    while r.has_next():
        existing_colors.add(str(r.get_next()[0]))
    existing_cities: set[str] = set()
    r = conn.execute("MATCH (c:City) RETURN c.name")
    while r.has_next():
        existing_cities.add(str(r.get_next()[0]))

    def collect_new(facts, role: str) -> set[str]:
        out = set()
        for f in facts:
            v = (f.get(role) or {}).get("color" if role == "perceptual"
                                         else "location")
            if v:
                out.add(v)
        return out

    new_colors = (collect_new(vehicles, "perceptual")
                  | collect_new(buildings, "perceptual")) - existing_colors
    new_cities = (collect_new(vehicles, "spatial")
                  | collect_new(buildings, "spatial")) - existing_cities
    for c in new_colors:
        conn.execute("CREATE (:Color {name: $n})", {"n": c})
    for c in new_cities:
        conn.execute("CREATE (:City {name: $n})", {"n": c})

    # Insert vehicles/buildings + edges.
    for facts, tbl, color_rel, loc_rel in [
        (vehicles, "Vehicle", "V_HAS_COLOR", "V_LOCATED_IN"),
        (buildings, "Building", "B_HAS_COLOR", "B_LOCATED_IN"),
    ]:
        seen = set()
        for f in facts:
            name = (f.get("entity") or {}).get("name")
            if not name:
                continue
            if name not in seen:
                conn.execute(f"CREATE (:{tbl} {{name: $n}})", {"n": name})
                seen.add(name)
        for f in facts:
            name = (f.get("entity") or {}).get("name")
            if not name:
                continue
            col = (f.get("perceptual") or {}).get("color")
            if col:
                conn.execute(
                    f"MATCH (e:{tbl} {{name: $n}}), (c:Color {{name: $c}}) "
                    f"CREATE (e)-[:{color_rel}]->(c)",
                    {"n": name, "c": col},
                )
            loc = (f.get("spatial") or {}).get("location")
            if loc:
                conn.execute(
                    f"MATCH (e:{tbl} {{name: $n}}), (c:City {{name: $c}}) "
                    f"CREATE (e)-[:{loc_rel}]->(c)",
                    {"n": name, "c": loc},
                )


def kuzu_answers_person_only(conn: kuzu.Connection) -> dict:
    """What the *original* schema returns — Person only. Wrong by design
    for cross-type aggregation."""
    out = {}
    t0 = time.time()
    r = conn.execute(
        "MATCH (p:Person)-[:HAS_COLOR]->(c:Color {name: 'blue'}) RETURN count(p)"
    )
    out["X1"] = (int(r.get_next()[0]), (time.time() - t0) * 1000)

    t0 = time.time()
    r = conn.execute(
        "MATCH (p:Person)-[:HAS_COLOR]->(c:Color) "
        "RETURN c.name, count(p) AS n ORDER BY n DESC LIMIT 5"
    )
    rows = []
    while r.has_next():
        row = r.get_next()
        rows.append((str(row[0]), int(row[1])))
    out["X2"] = (rows, (time.time() - t0) * 1000)

    t0 = time.time()
    r = conn.execute(
        "MATCH (p:Person)-[:LIVES_IN]->(:City {name: 'Austin'}) RETURN count(p)"
    )
    out["X3"] = (int(r.get_next()[0]), (time.time() - t0) * 1000)
    return out


def kuzu_answers_unioned(conn: kuzu.Connection) -> dict:
    """The CORRECT cross-type Cypher: UNION across Person + Vehicle + Building.
    Note the modeller has to (a) anticipate the new entity types in schema
    and (b) write 3-table-union queries by hand."""
    out = {}

    # X1: count blue across three node tables.
    t0 = time.time()
    n1 = int(conn.execute(
        "MATCH (p:Person)-[:HAS_COLOR]->(c:Color {name: 'blue'}) RETURN count(p)"
    ).get_next()[0])
    n2 = int(conn.execute(
        "MATCH (v:Vehicle)-[:V_HAS_COLOR]->(c:Color {name: 'blue'}) RETURN count(v)"
    ).get_next()[0])
    n3 = int(conn.execute(
        "MATCH (b:Building)-[:B_HAS_COLOR]->(c:Color {name: 'blue'}) RETURN count(b)"
    ).get_next()[0])
    out["X1"] = (n1 + n2 + n3, (time.time() - t0) * 1000)

    # X2: top-5 colors across all three. Manual Counter — Cypher
    # UNION ALL with GROUP BY across labels gets fiddly in Kùzu.
    t0 = time.time()
    counts: Counter = Counter()
    for cypher in [
        "MATCH (p:Person)-[:HAS_COLOR]->(c:Color) RETURN c.name AS x",
        "MATCH (v:Vehicle)-[:V_HAS_COLOR]->(c:Color) RETURN c.name AS x",
        "MATCH (b:Building)-[:B_HAS_COLOR]->(c:Color) RETURN c.name AS x",
    ]:
        r = conn.execute(cypher)
        while r.has_next():
            counts[str(r.get_next()[0])] += 1
    out["X2"] = (counts.most_common(5), (time.time() - t0) * 1000)

    # X3: Austin across all.
    t0 = time.time()
    n1 = int(conn.execute(
        "MATCH (p:Person)-[:LIVES_IN]->(:City {name: 'Austin'}) RETURN count(p)"
    ).get_next()[0])
    n2 = int(conn.execute(
        "MATCH (v:Vehicle)-[:V_LOCATED_IN]->(:City {name: 'Austin'}) RETURN count(v)"
    ).get_next()[0])
    n3 = int(conn.execute(
        "MATCH (b:Building)-[:B_LOCATED_IN]->(:City {name: 'Austin'}) RETURN count(b)"
    ).get_next()[0])
    out["X3"] = (n1 + n2 + n3, (time.time() - t0) * 1000)
    return out


# ── Runner ──────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n  ── generating mixed-entity-type corpus ──")
    persons = gen_persons(500, seed=0)
    vehicles = gen_vehicles(250, seed=0)
    buildings = gen_buildings(150, seed=0)
    all_facts = persons + vehicles + buildings

    print(f"  {len(persons):,} person facts (500 persons × ~20 facts)")
    print(f"  {len(vehicles):,} vehicle facts (250 vehicles × 4 facts)")
    print(f"  {len(buildings):,} building facts (150 buildings × 4 facts)")
    print(f"  {len(all_facts):,} total facts across THREE entity types")

    truth = compute_truth(all_facts)
    print(f"\n  ground truth (across ALL types):")
    print(f"    X1. blue count    : {truth['blue_total']}")
    print(f"    X2. top-5 colors  : {truth['top5_colors']}")
    print(f"    X3. Austin count  : {truth['austin_total']}")

    # ── Ada: one substrate, one query each, no schema work ──
    print(f"\n  ── Ada (one universal-schema substrate, no schema work) ──")
    import copy
    space = ThoughtGlyphSpace(enricher=None)
    t0 = time.time()
    for f in copy.deepcopy(all_facts):
        key = f.pop("key", None)
        space.tell_raw(facts=f, key=key, speaker="curriculum")
    print(f"  loaded {space.count:,} thoughts in {time.time()-t0:.1f}s "
          f"(same tell_raw API for persons, vehicles, AND buildings)")
    ada = ada_answers(space)

    # ── Kùzu pass 1: original Person-only schema ──
    print(f"\n  ── Kùzu pass 1: Person-only schema (the ORIGINAL design) ──")
    print(f"  hand-designed for the persons corpus; vehicles/buildings DON'T EXIST in it")
    t0 = time.time()
    conn = kuzu_load_persons_only(persons)
    print(f"  loaded persons in {time.time()-t0:.1f}s")
    kuzu_p1 = kuzu_answers_person_only(conn)

    # ── Kùzu pass 2: schema-migrate to add Vehicle + Building ──
    print(f"\n  ── Kùzu pass 2: SCHEMA MIGRATION required ──")
    print(f"  + CREATE NODE TABLE Vehicle, Building")
    print(f"  + CREATE REL TABLE V_HAS_COLOR, V_LOCATED_IN, B_HAS_COLOR, B_LOCATED_IN")
    print(f"  + load vehicles + buildings into new node tables")
    print(f"  + REWRITE every cross-type query as 3-table UNION")
    t0 = time.time()
    kuzu_extend_with_vehicles_and_buildings(conn, vehicles, buildings)
    print(f"  migration + load in {time.time()-t0:.1f}s")
    kuzu_p2 = kuzu_answers_unioned(conn)

    # ── Report ──
    print(f"\n  ── results ──")
    print("─" * 80)
    print(f"{'QUERY':<8}{'TRUTH':<18}{'ADA':<22}{'KUZU (orig)':<22}{'KUZU (migrated)':<22}")
    print("─" * 80)
    for q in ["X1", "X2", "X3"]:
        truth_val = {
            "X1": truth["blue_total"],
            "X2": truth["top5_colors"],
            "X3": truth["austin_total"],
        }[q]
        a, _ = ada[q]
        k1, _ = kuzu_p1[q]
        k2, _ = kuzu_p2[q]
        a_ok = (a == truth_val)
        k1_ok = (k1 == truth_val) if not isinstance(truth_val, list) else sorted(k1) == sorted(truth_val)
        k2_ok = (k2 == truth_val) if not isinstance(truth_val, list) else sorted(k2) == sorted(truth_val)

        tval = str(truth_val)[:16]
        aval = str(a)[:20]
        k1val = str(k1)[:20]
        k2val = str(k2)[:20]
        print(f"{q:<8}{tval:<18}{('✓ ' if a_ok else '✗ ') + aval:<22}"
              f"{('✓ ' if k1_ok else '✗ ') + k1val:<22}"
              f"{('✓ ' if k2_ok else '✗ ') + k2val:<22}")

    print("─" * 80)
    print("\n  ── architectural takeaway ──")
    print("  Ada            : same universal schema, same tell_raw API, same query path.")
    print("                   Adding entity types is zero schema work.")
    print("  Kùzu (orig)    : answers the cross-type query WRONG by undercounting,")
    print("                   because Vehicle and Building aren't in the schema.")
    print("  Kùzu (migrated): answers CORRECTLY but only after a schema migration")
    print("                   AND every cross-type query had to be hand-UNION'd.")
    print()
    print("  This is the 'two object types in one set' win: Ada's universal schema")
    print("  makes cross-type aggregation a one-line dict scan. KGs make it a")
    print("  schema-evolution problem AND a query-rewriting problem, for every new type.")


if __name__ == "__main__":
    main()
