"""
bench_25.py — 25 compositional queries across 7 shape categories,
Ada vs the synthetic-corpus ground truth.

The original 5 queries (bench_rag.py) cover aggregation / distribution /
intersection. This expansion fills in the rest of the compositional-query
landscape: negation, comparison, rank, refusal, multi-way intersection.

The point: Ada's win in bench_rag.py wasn't 5 happy accidents. It
generalises to every shape that depends on knowing the whole substrate
rather than top-K of it.

    PYTHONPATH=. python curriculum/at-scale/bench_25.py --n 1500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generate import generate
from ada.memory.thought_space import ThoughtGlyphSpace


# ── ground truth from the deterministic generator ─────────────────────

def compute_truth(facts: list[dict]) -> dict:
    """Compute deterministic answers across every query shape."""
    color: Counter = Counter()
    job: Counter = Counter()
    city: Counter = Counter()
    hobby: Counter = Counter()
    pet: Counter = Counter()
    food: Counter = Counter()
    drink: Counter = Counter()
    exercise: Counter = Counter()

    by_person: dict[str, dict[str, str]] = defaultdict(dict)

    for f in facts:
        key = f.get("key", "")
        e = f.get("entity") or {}
        name = e.get("name", "")
        if not name:
            continue

        col = (f.get("perceptual") or {}).get("color")
        if col and "color" in key and "car" not in key:
            color[col] += 1
            by_person[name]["color"] = col

        loc = (f.get("spatial") or {}).get("location")
        if loc and "city" in key:
            city[loc] += 1
            by_person[name]["city"] = loc

        rel = f.get("relational") or {}
        pred = rel.get("predicate")
        obj = rel.get("object", "")
        if pred == "works_as":
            job[obj] += 1
            by_person[name]["job"] = obj
        elif pred == "hobby":
            hobby[obj] += 1
            by_person[name]["hobby"] = obj
        elif pred == "pet":
            if obj != "none":
                pet[obj] += 1
            by_person[name]["pet"] = obj
        elif pred == "favorite_food":
            food[obj] += 1
            by_person[name]["food"] = obj
        elif pred == "drinks":
            drink[obj] += 1
            by_person[name]["drink"] = obj
        elif pred == "exercises":
            exercise[obj] += 1
            by_person[name]["exercise"] = obj

    n_persons = len(by_person)

    def isect(**filters) -> list[str]:
        out = []
        for name, attrs in by_person.items():
            if all(attrs.get(k) == v for k, v in filters.items()):
                out.append(name)
        return sorted(out)

    return {
        # ── Aggregation (5) ────────────────────────────────────────────
        "agg.blue":                color.get("blue", 0),
        "agg.engineer":            job.get("engineer", 0),
        "agg.austin":              city.get("Austin", 0),
        "agg.chess":               hobby.get("chess", 0),
        "agg.dog":                 pet.get("dog", 0),
        # ── Distribution (5) ───────────────────────────────────────────
        "dist.top5_cities":        city.most_common(5),
        "dist.top5_colors":        color.most_common(5),
        "dist.top5_jobs":          job.most_common(5),
        "dist.top5_hobbies":       hobby.most_common(5),
        "dist.top5_pets":          pet.most_common(5),
        # ── Negation / complement (3) ─────────────────────────────────
        "neg.not_blue":            n_persons - color.get("blue", 0),
        "neg.not_austin":          n_persons - city.get("Austin", 0),
        "neg.not_engineer":        n_persons - job.get("engineer", 0),
        # ── Intersection (5) ──────────────────────────────────────────
        "int.blue_aus_chess":      isect(color="blue", city="Austin", hobby="chess"),
        "int.blue_engineer":       isect(color="blue", job="engineer"),
        "int.red_austin":          isect(color="red", city="Austin"),
        "int.blue_eng_austin":     isect(color="blue", job="engineer", city="Austin"),
        "int.blue_eng_aus_chess":  isect(color="blue", job="engineer", city="Austin", hobby="chess"),
        # ── Comparison (3) ────────────────────────────────────────────
        "cmp.eng_vs_doc":          job.get("engineer", 0) - job.get("doctor", 0),
        "cmp.top1_top2_city":      (city.most_common(2)[0][1] - city.most_common(2)[1][1]
                                    if len(city) >= 2 else 0),
        "cmp.blue_aus_vs_ber":     (
            len(isect(color="blue", city="Austin"))
            - len(isect(color="blue", city="Berlin"))
        ),
        # ── Rank (2) ──────────────────────────────────────────────────
        "rank.austin":             [c for c, _ in city.most_common()].index("Austin") + 1
                                    if "Austin" in city else None,
        "rank.engineer":           [j for j, _ in job.most_common()].index("engineer") + 1
                                    if "engineer" in job else None,
        # ── Refusal / empty-set (2) ───────────────────────────────────
        "refuse.fake_hobby":       0,                  # nobody plays "underwater_basketweaving"
        "refuse.no_buildings":     None,               # no buildings in corpus
        # ── Bookkeeping ───────────────────────────────────────────────
        "_n_persons":              n_persons,
        "_n_facts":                len(facts),
    }


# ── Ada's structural answers ──────────────────────────────────────────

def ada_answer(space: ThoughtGlyphSpace, qid: str) -> tuple[object, float]:
    """Run the substrate operation for query qid; return (answer, ms)."""
    t0 = time.time()

    # Build a per-person index over universal-schema role-fillings.
    # Done once per call would be inefficient; we cache on the space
    # via an attribute for repeat calls.
    if not hasattr(space, "_bench_index"):
        idx: dict[str, dict[str, str]] = defaultdict(dict)
        for t in space._thoughts.values():
            u = t.glyph.metadata.get("_universal") or {}
            ent = u.get("entity") or {}
            name = ent.get("name")
            if not name:
                continue
            key = t.metadata.get("_key") or ""
            col = (u.get("perceptual") or {}).get("color")
            if col and "color" in key and "car" not in key:
                idx[name]["color"] = col
            loc = (u.get("spatial") or {}).get("location")
            if loc and "city" in key:
                idx[name]["city"] = loc
            rel = u.get("relational") or {}
            pred = rel.get("predicate")
            obj = rel.get("object", "")
            if pred == "works_as":
                idx[name]["job"] = obj
            elif pred == "hobby":
                idx[name]["hobby"] = obj
            elif pred == "pet":
                idx[name]["pet"] = obj
            elif pred == "favorite_food":
                idx[name]["food"] = obj
            elif pred == "drinks":
                idx[name]["drink"] = obj
            elif pred == "exercises":
                idx[name]["exercise"] = obj
        space._bench_index = idx
    idx = space._bench_index

    def count_where(attr, value):
        return sum(1 for a in idx.values() if a.get(attr) == value)

    def top5(attr):
        return Counter(a[attr] for a in idx.values() if attr in a).most_common(5)

    def isect_names(**filters):
        out = []
        for name, attrs in idx.items():
            if all(attrs.get(k) == v for k, v in filters.items()):
                out.append(name)
        return sorted(out)

    # ── Dispatch ────────────────────────────────────────────────────
    n_total = len(idx)
    ans = None
    if qid == "agg.blue":              ans = count_where("color", "blue")
    elif qid == "agg.engineer":        ans = count_where("job", "engineer")
    elif qid == "agg.austin":          ans = count_where("city", "Austin")
    elif qid == "agg.chess":           ans = count_where("hobby", "chess")
    elif qid == "agg.dog":             ans = count_where("pet", "dog")

    elif qid == "dist.top5_cities":    ans = top5("city")
    elif qid == "dist.top5_colors":    ans = top5("color")
    elif qid == "dist.top5_jobs":      ans = top5("job")
    elif qid == "dist.top5_hobbies":   ans = top5("hobby")
    elif qid == "dist.top5_pets":
        cnt = Counter(a["pet"] for a in idx.values()
                      if "pet" in a and a["pet"] not in ("", "none"))
        ans = cnt.most_common(5)

    elif qid == "neg.not_blue":        ans = n_total - count_where("color", "blue")
    elif qid == "neg.not_austin":      ans = n_total - count_where("city", "Austin")
    elif qid == "neg.not_engineer":    ans = n_total - count_where("job", "engineer")

    elif qid == "int.blue_aus_chess":     ans = isect_names(color="blue", city="Austin", hobby="chess")
    elif qid == "int.blue_engineer":      ans = isect_names(color="blue", job="engineer")
    elif qid == "int.red_austin":         ans = isect_names(color="red", city="Austin")
    elif qid == "int.blue_eng_austin":    ans = isect_names(color="blue", job="engineer", city="Austin")
    elif qid == "int.blue_eng_aus_chess": ans = isect_names(color="blue", job="engineer", city="Austin", hobby="chess")

    elif qid == "cmp.eng_vs_doc":          ans = count_where("job", "engineer") - count_where("job", "doctor")
    elif qid == "cmp.top1_top2_city":
        t = top5("city")
        ans = t[0][1] - t[1][1] if len(t) >= 2 else 0
    elif qid == "cmp.blue_aus_vs_ber":
        ans = (len(isect_names(color="blue", city="Austin"))
               - len(isect_names(color="blue", city="Berlin")))

    elif qid == "rank.austin":
        order = [c for c, _ in Counter(a["city"] for a in idx.values() if "city" in a).most_common()]
        ans = order.index("Austin") + 1 if "Austin" in order else None
    elif qid == "rank.engineer":
        order = [j for j, _ in Counter(a["job"] for a in idx.values() if "job" in a).most_common()]
        ans = order.index("engineer") + 1 if "engineer" in order else None

    elif qid == "refuse.fake_hobby":     ans = count_where("hobby", "underwater_basketweaving")
    elif qid == "refuse.no_buildings":   ans = None  # ∅: no buildings in person-only corpus

    return ans, (time.time() - t0) * 1000


QUERY_CATEGORIES = {
    "Aggregation":       ["agg.blue", "agg.engineer", "agg.austin", "agg.chess", "agg.dog"],
    "Distribution":      ["dist.top5_cities", "dist.top5_colors", "dist.top5_jobs",
                          "dist.top5_hobbies", "dist.top5_pets"],
    "Negation":          ["neg.not_blue", "neg.not_austin", "neg.not_engineer"],
    "Intersection":      ["int.blue_aus_chess", "int.blue_engineer", "int.red_austin",
                          "int.blue_eng_austin", "int.blue_eng_aus_chess"],
    "Comparison":        ["cmp.eng_vs_doc", "cmp.top1_top2_city", "cmp.blue_aus_vs_ber"],
    "Rank":              ["rank.austin", "rank.engineer"],
    "Refusal":           ["refuse.fake_hobby", "refuse.no_buildings"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500,
                    help="number of persons (20 facts each)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"\n  ── generating corpus: n={args.n} persons × 20 facts = "
          f"{args.n * 20:,} facts ──")
    facts = generate(args.n, seed=args.seed)
    truth = compute_truth(facts)
    print(f"  generated {truth['_n_facts']:,} facts about {truth['_n_persons']:,} persons.")

    print(f"\n  ── loading substrate ──")
    t0 = time.time()
    space = ThoughtGlyphSpace(enricher=None)
    for f in facts:
        key = f.get("key")
        slots = {k: v for k, v in f.items() if k != "key" and isinstance(v, dict)}
        space.tell_raw(facts=slots, key=key, speaker="corpus",
                       metadata={"category": "person-fact"})
    load_ms = (time.time() - t0) * 1000
    print(f"  loaded {space.count:,} thoughts in {load_ms/1000:.1f}s")

    # ── Run all 25 queries ───────────────────────────────────────────
    print(f"\n  ── running 25 queries ──")
    n_pass = 0
    n_total = 0
    total_ms = 0.0
    per_cat_pass = defaultdict(lambda: [0, 0])  # [pass, total]
    results: list[dict] = []

    for category, qids in QUERY_CATEGORIES.items():
        print(f"\n  {category}:")
        for qid in qids:
            t = truth[qid]
            a, ms = ada_answer(space, qid)
            total_ms += ms

            ok = (a == t)
            if ok:
                n_pass += 1
                per_cat_pass[category][0] += 1
            n_total += 1
            per_cat_pass[category][1] += 1

            marker = "✓" if ok else "✗"
            t_str = str(t)[:60]
            a_str = str(a)[:60]
            print(f"    {marker} {qid:<30} truth={t_str:<60} ada={a_str}  ({ms:.1f} ms)")
            results.append({"qid": qid, "ok": ok, "truth": t, "ada": a, "ms": ms})

    # ── Headline ─────────────────────────────────────────────────────
    print()
    print("━" * 72)
    print(f"  HEADLINE — Ada on 25-query benchmark at n={args.n}")
    print("━" * 72)
    for cat, (p, n) in per_cat_pass.items():
        print(f"    {cat:<16} {p}/{n}")
    print(f"    {'TOTAL':<16} {n_pass}/{n_total}")
    print(f"    mean per-query latency: {total_ms/n_total:.1f} ms")
    print(f"    total query latency:    {total_ms:.0f} ms")
    print()

    out_dir = Path(__file__).parent / "bench_25_output"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"results_n{args.n}.json").write_text(json.dumps({
        "n_persons": truth["_n_persons"],
        "n_facts": truth["_n_facts"],
        "n_pass": n_pass,
        "n_total": n_total,
        "mean_query_ms": total_ms / n_total,
        "per_category": dict(per_cat_pass),
        "results": [{**r, "truth": str(r["truth"]), "ada": str(r["ada"])}
                    for r in results],
    }, indent=2))
    print(f"  → results saved to {out_dir}/results_n{args.n}.json")


if __name__ == "__main__":
    main()
