"""
Wikipedia category crawler — produces a large chemistry corpus by
walking subcategories under chemistry-related roots.

    PYTHONPATH=. python curriculum/agi/chemistry/build_corpus.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path

HERE = Path(__file__).parent
CACHE = HERE / ".corpus_cache"
CACHE.mkdir(exist_ok=True)
# OUT is set after DOMAIN is read below; placeholder
OUT: Path  # set in __main__ block


_os = __import__("os")

# Roots to crawl per domain. Each tuple = (category, max_depth).
DOMAIN_ROOTS: dict[str, list[tuple[str, int]]] = {
    "chemistry": [
        ("Organic compounds", 3),
        ("Inorganic compounds", 3),
        ("Biomolecules", 3),
        ("Drugs", 2),
        ("Pharmaceuticals", 2),
        ("Functional groups", 2),
        ("Acids", 2),
        ("Bases (chemistry)", 2),
        ("Sugars", 2),
        ("Amino acids", 2),
        ("Lipids", 2),
        ("Nucleotides", 2),
        ("Vitamins", 2),
        ("Hormones", 2),
        ("Antibiotics", 2),
        ("Hydrocarbons", 3),
        ("Polymers", 2),
        ("Materials science", 2),
        ("Chemical elements", 1),
        ("Chemical reactions", 2),
        ("Chemical bonding", 2),
    ],
    "philosophy": [
        ("Philosophical concepts", 3),
        ("Philosophical schools and traditions", 3),
        ("Philosophy by topic", 2),
        ("Branches of philosophy", 2),
        ("Philosophers", 2),
        ("Epistemology", 2),
        ("Metaphysics", 2),
        ("Ethics", 3),
        ("Aesthetics", 2),
        ("Logic", 2),
        ("Political philosophy", 2),
        ("Philosophy of mind", 2),
        ("Philosophy of science", 2),
        ("Philosophy of language", 2),
        ("Existentialism", 2),
        ("Phenomenology", 2),
        ("Ancient philosophy", 2),
        ("Modern philosophy", 2),
        ("Continental philosophy", 2),
        ("Analytic philosophy", 2),
    ],
    "law": [
        ("Law", 3),
        ("Legal terminology", 2),
        ("Legal concepts", 3),
        ("Legal doctrines", 3),
        ("Branches of law", 2),
        ("Constitutional law", 2),
        ("Criminal law", 2),
        ("Contract law", 2),
        ("Tort law", 2),
        ("Property law", 2),
        ("Family law", 2),
        ("Corporate law", 2),
        ("Administrative law", 2),
        ("International law", 2),
        ("Intellectual property law", 2),
        ("Human rights", 2),
        ("Courts", 2),
        ("Legal procedure", 2),
        ("Common law", 2),
        ("Civil law (legal system)", 2),
    ],
}

DOMAIN = _os.environ.get("DOMAIN", "chemistry")
ROOTS = DOMAIN_ROOTS[DOMAIN]
MAX_TITLES = int(_os.environ.get("MAX_TITLES", "10000"))

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "AdaCurriculum/0.1 (research)",
    "Accept": "application/json",
}


def _fetch_json(params: dict) -> dict | None:
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(1.5 + 1.5 * attempt)
                continue
            return None
        except Exception:
            time.sleep(0.5)
    return None


def list_category(cat: str, cmtype: str) -> list[str]:
    """Returns titles in category, paginating through cmcontinue."""
    cache_path = CACHE / f"cat_{cmtype}_{cat.replace('/', '_')}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass
    out: list[str] = []
    cont: str | None = None
    while True:
        params = {
            "action": "query", "format": "json",
            "list": "categorymembers",
            "cmtitle": f"Category:{cat}",
            "cmlimit": "500",
            "cmtype": cmtype,
        }
        if cont:
            params["cmcontinue"] = cont
        data = _fetch_json(params)
        if not data:
            break
        for m in data.get("query", {}).get("categorymembers", []):
            t = m.get("title", "")
            if cmtype == "subcat" and t.startswith("Category:"):
                out.append(t[len("Category:"):])
            elif cmtype == "page":
                # Drop disambiguation/list pages.
                if t.startswith(("List of ", "Index of ")):
                    continue
                out.append(t)
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break
    cache_path.write_text(json.dumps(out))
    return out


def crawl(root: str, max_depth: int, seen_cats: set[str],
          all_titles: set[str], cap: int) -> None:
    queue: deque[tuple[str, int]] = deque([(root, 0)])
    while queue and len(all_titles) < cap:
        cat, depth = queue.popleft()
        if cat in seen_cats:
            continue
        seen_cats.add(cat)
        pages = list_category(cat, "page")
        for p in pages:
            all_titles.add(p)
            if len(all_titles) >= cap:
                return
        if depth < max_depth:
            for sub in list_category(cat, "subcat"):
                if sub not in seen_cats:
                    queue.append((sub, depth + 1))


def main() -> None:
    global OUT
    out_root = HERE.parent / DOMAIN / "phase1_output"
    out_root.mkdir(parents=True, exist_ok=True)
    OUT = out_root / "corpus_titles.txt"
    print(f"Building {DOMAIN} corpus (cap={MAX_TITLES}) → {OUT}")
    seen_cats: set[str] = set()
    titles: set[str] = set()
    for root, depth in ROOTS:
        if len(titles) >= MAX_TITLES:
            break
        print(f"  crawling: {root:<32} (depth {depth})  ", end="", flush=True)
        before = len(titles)
        crawl(root, depth, seen_cats, titles, MAX_TITLES)
        print(f"+{len(titles) - before}  total={len(titles)}")
    # Dedup against existing static corpus.
    sorted_titles = sorted(titles)
    OUT.write_text("\n".join(sorted_titles))
    print(f"\n  wrote {len(sorted_titles)} unique titles to {OUT}")


if __name__ == "__main__":
    main()
