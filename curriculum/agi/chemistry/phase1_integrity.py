"""
Phase 1: composition integrity test at scale.

Tests whether an LLM (Haiku 4.5), given the universal schema + the
existing substrate's key namespace, can convert real Wikipedia
chemistry text into tell_raw-compatible records whose composed_of
references resolve into the lattice.

If the answer is yes — at usable cost — the path from 84 entries to
50K (and 50M) entries is engineering, not research.

Cost-controlled. Uses Anthropic prompt caching to cut cost ~10× on
the schema/keys overhead. Hard kill switch: BUDGET_USD.

    PYTHONPATH=. ANTHROPIC_API_KEY=... python \\
        curriculum/agi/chemistry/phase1_integrity.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

HERE = Path(__file__).parent
ROOT = HERE.parent  # curriculum/agi
sys.path.insert(0, str(HERE))
from teach import (CHEM_FILES, LAYER_0, LAYER_1, _load_yaml,  # noqa
                   main as teach_main)

# ── knobs ───────────────────────────────────────────────────────────
N_COMPOUNDS    = int(os.environ.get("N_COMPOUNDS", "1500"))
BUDGET_USD     = float(os.environ.get("BUDGET_USD", "60"))
CONCURRENCY    = int(os.environ.get("CONCURRENCY", "8"))
CORPUS_FILE    = os.environ.get("CORPUS_FILE", "")
DO_RETRY       = os.environ.get("DO_RETRY", "1") == "1"
# When set (comma-sep prefixes), restrict the LLM's namespace to only
# keys with those prefixes. Default: include everything in the lattice.
NAMESPACE_PREFIXES = os.environ.get("NAMESPACE_PREFIXES", "")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "")  # absolute or relative; if blank uses HERE/phase1_output
MODEL          = "claude-haiku-4-5-20251001"
HAIKU_IN_PER_M = 1.00  # $/MTok input
HAIKU_OUT_PER_M = 5.00  # $/MTok output
# Cached input is 0.1× the base rate on Anthropic.
HAIKU_CACHED_IN_PER_M = 0.10
CACHE_DIR      = HERE / ".phase1_cache"
CACHE_DIR.mkdir(exist_ok=True)
EXTRACT_CACHE  = HERE / ".phase1_extract_cache"
EXTRACT_CACHE.mkdir(exist_ok=True)


# ── source corpus ──────────────────────────────────────────────────
# A breadth-first list of chemistry concepts from textbook organic +
# inorganic + biochem + drugs + materials. Each is a Wikipedia page
# title. We deliberately include some redundant ones to test dedup,
# some obscure ones to test refusal, and some that won't ground
# cleanly without the LLM declaring intermediate keys.

CORPUS = """
# Hydrocarbons
methane ethane propane butane pentane hexane heptane octane nonane decane
cyclopropane cyclobutane cyclopentane cyclohexane cycloheptane
ethene propene butene pentene 1-hexene cyclohexene
ethyne propyne butyne acetylene
benzene toluene xylene styrene naphthalene anthracene phenanthrene
pyridine pyrrole furan thiophene indole quinoline imidazole

# Alcohols, ethers, aldehydes, ketones
methanol ethanol 1-propanol 2-propanol 1-butanol tert-butanol
glycerol ethylene_glycol propylene_glycol phenol catechol resorcinol
dimethyl_ether diethyl_ether tetrahydrofuran 1,4-dioxane methyl_tert-butyl_ether
formaldehyde acetaldehyde propionaldehyde butyraldehyde benzaldehyde
acetone butanone methyl_ethyl_ketone cyclohexanone acetophenone benzophenone

# Carboxylic acids and derivatives
formic_acid acetic_acid propionic_acid butyric_acid valeric_acid
oxalic_acid malonic_acid succinic_acid glutaric_acid adipic_acid
citric_acid tartaric_acid lactic_acid pyruvic_acid malic_acid
benzoic_acid salicylic_acid phthalic_acid terephthalic_acid
acetyl_chloride acetic_anhydride methyl_acetate ethyl_acetate
methyl_benzoate aspirin ethyl_propionate acetamide

# Amines, amides, nitriles
methylamine ethylamine dimethylamine trimethylamine ethylenediamine
aniline pyridine_(compound) imidazole_(amine) histamine
acetamide urea formamide N,N-dimethylformamide
acetonitrile benzonitrile cyanide hydrogen_cyanide

# Inorganic compounds
water carbon_dioxide carbon_monoxide ozone hydrogen_peroxide
ammonia hydrazine nitrous_oxide nitric_oxide nitrogen_dioxide
sulfur_dioxide sulfur_trioxide hydrogen_sulfide carbon_disulfide
sodium_chloride potassium_chloride calcium_chloride magnesium_chloride
sodium_bicarbonate sodium_carbonate calcium_carbonate calcium_phosphate
sulfuric_acid hydrochloric_acid nitric_acid phosphoric_acid perchloric_acid
sodium_hydroxide potassium_hydroxide calcium_hydroxide ammonium_hydroxide
sodium_sulfate copper_sulfate iron_sulfate zinc_sulfate
silver_nitrate potassium_nitrate ammonium_nitrate sodium_nitrate
potassium_permanganate sodium_thiosulfate sodium_hypochlorite
iron(III)_oxide iron(II)_oxide aluminum_oxide silicon_dioxide titanium_dioxide
zinc_oxide magnesium_oxide calcium_oxide sodium_oxide
sulfur_hexafluoride boron_trifluoride boric_acid

# Amino acids
glycine alanine valine leucine isoleucine proline phenylalanine tryptophan
methionine cysteine serine threonine tyrosine asparagine glutamine
aspartic_acid glutamic_acid lysine arginine histidine

# Sugars and carbohydrates
glucose fructose galactose mannose ribose deoxyribose
sucrose lactose maltose cellobiose
starch cellulose glycogen amylose amylopectin chitin

# Lipids
palmitic_acid stearic_acid oleic_acid linoleic_acid
glycerol triglyceride phosphatidylcholine cholesterol steroid
arachidonic_acid prostaglandin

# Nucleotides and nucleobases
adenine guanine cytosine thymine uracil
adenosine guanosine cytidine thymidine uridine
adenosine_monophosphate adenosine_diphosphate adenosine_triphosphate
deoxyribonucleic_acid ribonucleic_acid messenger_RNA transfer_RNA

# Vitamins
ascorbic_acid retinol cholecalciferol tocopherol phylloquinone
thiamine riboflavin niacin pantothenic_acid pyridoxine biotin folate cobalamin

# Drugs / pharmaceuticals
aspirin ibuprofen acetaminophen naproxen
caffeine theobromine theophylline nicotine
morphine codeine heroin methadone
penicillin amoxicillin cephalexin tetracycline
insulin epinephrine norepinephrine dopamine serotonin melatonin acetylcholine
ethanol_(drug) THC cannabidiol
amphetamine methamphetamine MDMA
diazepam fluoxetine sertraline omeprazole atorvastatin metformin

# Polymers
polyethylene polypropylene polystyrene polyvinyl_chloride polyethylene_terephthalate
nylon Kevlar Teflon Bakelite
DNA RNA protein peptide

# Functional groups & motifs we may want to register
hydroxyl_group carbonyl_group carboxyl_group amino_group methyl_group
ester ether ketone aldehyde alcohol amine amide phosphate sulfonate nitro_group
phenyl_group benzyl_group vinyl_group allyl_group ethyl_group
alkene alkyne alkane arene aromatic_hydrocarbon
disulfide_bond peptide_bond glycosidic_bond ester_bond

# Bonds & general chem
covalent_bond ionic_bond hydrogen_bond metallic_bond van_der_Waals_force
dipole-dipole_interaction London_dispersion_force pi_bond sigma_bond
electronegativity polarity solubility melting_point boiling_point
acid base pH buffer_solution
oxidation reduction redox combustion hydrolysis esterification neutralization
photosynthesis cellular_respiration glycolysis Krebs_cycle fermentation

# Materials
graphite diamond fullerene graphene carbon_nanotube
glass steel bronze brass concrete cement gypsum
limestone marble quartz silica clay ceramic
rubber latex Vulcanization
copper_oxide bronze brass mercury(II)_chloride sulfur(disambiguation)

# Trick / refusal candidates — substrate has no taste/smell role for these
benzene methane octane gasoline naphthalene_(odor)
"""


def parse_corpus(text: str) -> list[str]:
    seen, out = set(), []
    for tok in text.split():
        if tok.startswith("#"):
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


# ── Wikipedia ───────────────────────────────────────────────────────

def _wp_cache_path(title: str) -> Path:
    safe = title.replace("/", "_")[:120]
    return CACHE_DIR / f"wp_{safe}.json"


def fetch_summary(title: str, timeout: float = 10,
                  max_retries: int = 4) -> dict | None:
    cache = _wp_cache_path(title)
    if cache.exists():
        try:
            cached = json.loads(cache.read_text())
            if cached.get("extract"):
                return cached
        except Exception:
            pass
    if os.environ.get("SKIP_NEW_FETCH") == "1":
        return None
    enc = urllib.parse.quote(title.replace("_", " "))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{enc}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "AdaCurriculum/0.1 (research; chris.timmerman@glyphh.ai)",
        "Accept": "application/json",
    })
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            out = {
                "title": data.get("title"),
                "extract": data.get("extract", ""),
                "description": data.get("description", ""),
                "type": data.get("type"),
            }
            cache.write_text(json.dumps(out))
            return out
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(1.5 + 1.5 * attempt)
                continue
            if e.code == 404:
                return {"_error": "404", "_title": title}
            time.sleep(1)
        except Exception:
            time.sleep(1)
    return {"_error": "max_retries", "_title": title}


# ── Build the schema + keys context for Haiku ───────────────────────

def build_substrate_index() -> tuple[dict[str, dict], list[str]]:
    """Returns (lattice, sorted_key_namespace)."""
    lattice: dict[str, dict] = {}
    for p in [LAYER_0, *LAYER_1, *CHEM_FILES]:
        for e in _load_yaml(p):
            k = e.get("key")
            if k:
                lattice[k] = e
    keys = sorted(lattice)
    return lattice, keys


def key_namespace_text(keys: list[str]) -> str:
    """Compact, grouped listing of available keys."""
    by_prefix: dict[str, list[str]] = {}
    for k in keys:
        prefix = k.split(".", 1)[0]
        by_prefix.setdefault(prefix, []).append(k)
    lines = []
    for prefix in ["prime", "color", "number", "spatial", "logic",
                   "time", "element", "bond", "group", "compound",
                   "property", "reaction", "composition"]:
        ks = by_prefix.get(prefix, [])
        if not ks:
            continue
        # Drop composition.* — they are derived.
        if prefix == "composition":
            continue
        lines.append(f"## {prefix}.* ({len(ks)})")
        # Pack with comma separation, wrap at ~100 chars.
        line = ""
        for k in ks:
            piece = k + ", "
            if len(line) + len(piece) > 100:
                lines.append(line.rstrip(", "))
                line = ""
            line += piece
        if line:
            lines.append(line.rstrip(", "))
    return "\n".join(lines)


SYSTEM_TMPL = """\
You map chemistry text into the Ada universal-schema lattice.

OUTPUT FORMAT: a single JSON object (no markdown, no commentary):
{{
  "key": "compound.<lowercase-hyphenated-name>",
  "category": "compound" | "element" | "group" | "property" | "reaction" \
| "material",
  "entity": {{"name": "<name>", "kind": "compound", "subkind": "<optional>"}},
  "perceptual": {{"color": "...", "temperature": "..."}}            (optional),
  "quantitative": {{"magnitude": "<num>", "unit": "<unit>"}}        (optional),
  "semantic":   {{"category": "<short-phrase, dashed>"}}            (optional),
  "relational": {{"predicate": "<rel>", "object": "<thing>"}}       (optional),
  "composed_of": ["<existing-key>", "<existing-key>", ...]
}}

RULES (strict):
1. `composed_of` MUST be a non-empty list of keys from the namespace below.
2. NEVER invent a key. If a needed atom doesn't exist, decompose into
   ones that do (e.g. "ammonium" → element.nitrogen + element.hydrogen
   + bond.covalent).
3. Use the most specific keys that apply. If the molecule has -OH groups,
   include group.hydroxyl. If it contains a C=O, include group.carbonyl.
   If it hydrogen-bonds, include bond.hydrogen.
4. For elements you may use composition with prime.* + color.* (matching
   the layer-1 pattern). For new compounds use elements/bonds/groups.
5. Output ONLY the JSON. No prose.

KEY NAMESPACE (the only keys you may reference):
{namespace}
"""


def build_user(title: str, extract: str, description: str) -> str:
    return (
        f"Title: {title}\n"
        f"Description: {description}\n"
        f"Wikipedia extract: {extract[:1200]}\n\n"
        "Emit JSON."
    )


# ── Validation ─────────────────────────────────────────────────────

def parse_response(text: str) -> dict | None:
    text = text.strip()
    # Strip code fences if present.
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except Exception:
        # Try to extract the first {...} blob.
        start = text.find("{")
        end = text.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                return None
        return None


def validate(entry: dict, keys: set[str]) -> tuple[bool, list[str]]:
    """Return (is_valid, list_of_unresolved_refs)."""
    if not isinstance(entry, dict):
        return False, ["not_a_dict"]
    if not entry.get("key"):
        return False, ["missing_key"]
    if not entry.get("entity"):
        return False, ["missing_entity"]
    refs = entry.get("composed_of") or []
    if not isinstance(refs, list) or not refs:
        return False, ["empty_composed_of"]
    unresolved = [r for r in refs if r not in keys]
    return (not unresolved), unresolved


def trace_to_primes(lattice: dict[str, dict], key: str,
                    max_depth: int = 8) -> tuple[bool, int, set[str]]:
    """BFS until everything is a prime.* or layer-1 atom.
    Returns (grounded, depth_reached, primes_set)."""
    visited: set[str] = set()
    primes: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(key, 0)])
    max_seen = 0
    while queue:
        cur, d = queue.popleft()
        if cur in visited:
            continue
        visited.add(cur)
        max_seen = max(max_seen, d)
        if d > max_depth:
            return False, max_seen, primes
        node = lattice.get(cur)
        if not node:
            return False, max_seen, primes
        parents = node.get("composed_of") or []
        if cur.startswith("prime."):
            primes.add(cur)
            continue
        if not parents:
            # Layer-1 atomic concept with empty composed_of: still grounded.
            continue
        for p in parents:
            queue.append((p, d + 1))
    return True, max_seen, primes


# ── Caller ─────────────────────────────────────────────────────────

def call_haiku(client, system_blocks: list[dict],
               messages: list[dict],
               max_retries: int = 3) -> tuple[dict | None, dict]:
    """Returns (parsed_json | None, usage_dict)."""
    import anthropic
    for attempt in range(max_retries + 1):
        try:
            r = client.messages.create(
                model=MODEL,
                max_tokens=900,
                system=system_blocks,
                messages=messages,
            )
            usage = {
                "in_tok": r.usage.input_tokens,
                "out_tok": r.usage.output_tokens,
                "cache_create_in_tok": getattr(
                    r.usage, "cache_creation_input_tokens", 0) or 0,
                "cache_read_in_tok": getattr(
                    r.usage, "cache_read_input_tokens", 0) or 0,
            }
            text = r.content[0].text if r.content else ""
            return parse_response(text), usage
        except anthropic.RateLimitError:
            time.sleep(2 ** attempt)
        except (anthropic.APITimeoutError, anthropic.APIConnectionError):
            time.sleep(1 + attempt)
        except Exception as e:
            if attempt == max_retries:
                return None, {"error": str(e)[:200]}
            time.sleep(1)
    return None, {}


def _extract_cache_path(title: str) -> Path:
    safe = title.replace("/", "_").replace(" ", "_")[:140]
    return EXTRACT_CACHE / f"ex_{safe}.json"


def cached_extract(title: str) -> dict | None:
    p = _extract_cache_path(title)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def save_extract(title: str, parsed: dict | None, usage: dict) -> None:
    _extract_cache_path(title).write_text(json.dumps(
        {"parsed": parsed, "usage": usage}))


def usage_cost(u: dict) -> float:
    in_uncached = u.get("in_tok", 0) - u.get("cache_read_in_tok", 0)
    in_cached = u.get("cache_read_in_tok", 0)
    in_cache_create = u.get("cache_create_in_tok", 0)
    # Cache creation is 1.25× base; cached read 0.1× base; uncached = base.
    return (
        in_uncached     * HAIKU_IN_PER_M / 1e6 +
        in_cache_create * (HAIKU_IN_PER_M * 1.25) / 1e6 +
        in_cached       * HAIKU_CACHED_IN_PER_M / 1e6 +
        u.get("out_tok", 0) * HAIKU_OUT_PER_M / 1e6
    )


def main() -> None:
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY.", file=sys.stderr)
        sys.exit(1)

    print()
    print("━" * 72)
    print(f"  PHASE 1 — composition integrity at scale")
    print(f"  N_COMPOUNDS={N_COMPOUNDS}  BUDGET=${BUDGET_USD}  "
          f"CONCURRENCY={CONCURRENCY}  MODEL={MODEL}")
    print("━" * 72)

    # 1. Substrate.
    space = teach_main()
    lattice, keys = build_substrate_index()
    initial_keys = set(keys)
    print(f"  baseline: {len(initial_keys)} keys in substrate.\n")

    # 2. Corpus.
    if CORPUS_FILE and Path(CORPUS_FILE).exists():
        titles = [t.strip() for t in Path(CORPUS_FILE).read_text().split("\n")
                  if t.strip()][:N_COMPOUNDS]
        print(f"  corpus: {len(titles)} chemistry titles from "
              f"{CORPUS_FILE}.\n")
    else:
        titles = parse_corpus(CORPUS)[:N_COMPOUNDS]
        print(f"  corpus: {len(titles)} chemistry titles queued.\n")

    # 3. Fetch Wikipedia in parallel (throttled — be a good neighbor).
    wp_conc = int(os.environ.get("WP_CONCURRENCY", "8"))
    print(f"  ── fetching Wikipedia summaries (concurrency={wp_conc}) ...")
    summaries: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=wp_conc) as ex:
        futs = {ex.submit(fetch_summary, t): t for t in titles}
        for i, fut in enumerate(as_completed(futs), 1):
            t = futs[fut]
            s = fut.result()
            if s and not s.get("_error") and s.get("extract"):
                summaries[t] = s
            if i % 1000 == 0:
                print(f"    {i}/{len(titles)} fetched, "
                      f"{len(summaries)} with extracts.")
    print(f"    fetched {len(summaries)}/{len(titles)} non-empty summaries.\n")

    # 4. Build prompt context.
    if NAMESPACE_PREFIXES:
        allowed = {p.strip() for p in NAMESPACE_PREFIXES.split(",") if p.strip()}
        filtered_keys = [k for k in keys
                         if k.split(".", 1)[0] in allowed]
        print(f"  ── namespace filter: {NAMESPACE_PREFIXES} → "
              f"{len(filtered_keys)}/{len(keys)} keys exposed to LLM")
        initial_keys = set(filtered_keys)  # rebind: refs only resolve in this slice
        ns_text = key_namespace_text(filtered_keys)
    else:
        ns_text = key_namespace_text(keys)
    system_blocks = [
        {"type": "text",
         "text": SYSTEM_TMPL.format(namespace=ns_text),
         "cache_control": {"type": "ephemeral"}},
    ]
    print(f"  ── system prompt: {len(SYSTEM_TMPL.format(namespace=ns_text))} chars "
          f"(prompt-caching enabled).\n")

    # 5. Call Haiku for each summary, with budget enforcement.
    client = anthropic.Anthropic()
    results: list[dict] = []
    total_cost = 0.0
    n_done = 0
    items = list(summaries.items())

    print(f"  ── extracting via Haiku (concurrency={CONCURRENCY}) ...")
    t0 = time.time()
    budget_hit = False
    n_cache_hits = 0

    # Reserve early cache hits so we don't pay for them.
    pending: list[tuple[str, dict]] = []
    for title, s in items:
        cached = cached_extract(title)
        if cached is not None:
            results.append({"title": title, "parsed": cached.get("parsed"),
                            "usage": cached.get("usage", {}),
                            "cost_usd": 0.0})
            n_done += 1
            n_cache_hits += 1
        else:
            pending.append((title, s))
    print(f"    {n_cache_hits} cache hits; {len(pending)} fresh calls needed.")

    def one(item):
        title, s = item
        user = build_user(title, s["extract"], s.get("description", ""))
        parsed, usage = call_haiku(
            client, system_blocks,
            [{"role": "user", "content": user}])
        save_extract(title, parsed, usage)
        return title, parsed, usage

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = [ex.submit(one, it) for it in pending]
        for fut in as_completed(futs):
            if budget_hit:
                continue
            title, parsed, usage = fut.result()
            cost = usage_cost(usage)
            total_cost += cost
            n_done += 1
            results.append({"title": title, "parsed": parsed, "usage": usage,
                            "cost_usd": cost})
            if n_done % 200 == 0:
                rate = n_done / max(1, time.time() - t0)
                print(f"    {n_done:>5}/{len(items)}  "
                      f"${total_cost:>6.2f} spent  "
                      f"{rate:.1f}/s")
            if total_cost > BUDGET_USD:
                print(f"    ✗ budget exceeded at {n_done} items "
                      f"(${total_cost:.2f}) — draining...")
                budget_hit = True

    print(f"\n    {n_done} extractions complete in {time.time() - t0:.1f}s, "
          f"fresh-call cost ${total_cost:.4f} "
          f"({n_cache_hits} from cache).\n")

    # 6. Validation pass.
    print(f"  ── validation ──")

    n_parseable = sum(1 for r in results if r["parsed"])
    n_schema_ok = 0
    n_refs_ok = 0
    unresolved_examples: list[tuple[str, list[str]]] = []
    new_entries: list[dict] = []

    # Pass 1: collect all proposed new keys, allow refs that point at them
    # IF they themselves are valid.
    proposed_keys = {r["parsed"]["key"] for r in results
                     if r["parsed"] and isinstance(r["parsed"], dict)
                     and isinstance(r["parsed"].get("key"), str)}
    all_known = initial_keys | proposed_keys

    for r in results:
        entry = r["parsed"]
        if not isinstance(entry, dict):
            continue
        ok, unresolved = validate(entry, all_known)
        if entry.get("key") and entry.get("entity") \
                and entry.get("composed_of"):
            n_schema_ok += 1
        if ok:
            n_refs_ok += 1
            new_entries.append(entry)
        elif unresolved:
            unresolved_examples.append((entry.get("key", "?"), unresolved))

    n_attempted = n_done
    print(f"    JSON parseable:           {n_parseable}/{n_attempted} = "
          f"{100 * n_parseable / max(1, n_attempted):.1f}%")
    print(f"    Schema fields present:    {n_schema_ok}/{n_attempted} = "
          f"{100 * n_schema_ok / max(1, n_attempted):.1f}%")
    print(f"    All composed_of resolve:  {n_refs_ok}/{n_attempted} = "
          f"{100 * n_refs_ok / max(1, n_attempted):.1f}%")
    if unresolved_examples:
        print(f"    Sample unresolved refs (first 5):")
        for k, urs in unresolved_examples[:5]:
            print(f"      • {k}  → unresolved: {urs[:5]}")
    print()

    # ── Retry-with-feedback pass ────────────────────────────────────
    if DO_RETRY and not budget_hit:
        # Collect failures with original summary text.
        failures: list[tuple[str, dict, dict, list[str]]] = []
        title_to_summary = dict(summaries)
        for r in results:
            entry = r.get("parsed")
            if not isinstance(entry, dict):
                continue
            ok, unresolved = validate(entry, all_known)
            if not ok and unresolved \
                    and unresolved != ["empty_composed_of"] \
                    and unresolved != ["missing_key"] \
                    and unresolved != ["missing_entity"] \
                    and r["title"] in title_to_summary:
                failures.append((r["title"], title_to_summary[r["title"]],
                                 entry, unresolved))
        print(f"  ── retry pass: {len(failures)} entries with unresolved refs")
        if failures:
            print(f"    sending corrective feedback to Haiku...")

            def retry_one(t):
                title, s, prev, unresolved = t
                user1 = build_user(title, s["extract"],
                                   s.get("description", ""))
                prev_json = json.dumps(prev, indent=None)
                feedback = (
                    f"Your previous response was:\n{prev_json}\n\n"
                    f"These composed_of keys do NOT exist in the namespace: "
                    f"{unresolved}. Re-emit a valid JSON for the SAME "
                    f"concept, but every composed_of key MUST appear in the "
                    f"namespace shown in the system prompt. Decompose unknown "
                    f"functional groups into element.* + bond.* + group.* keys "
                    f"that DO exist."
                )
                parsed, usage = call_haiku(
                    client, system_blocks,
                    [{"role": "user", "content": user1},
                     {"role": "assistant", "content": prev_json},
                     {"role": "user", "content": feedback}])
                return title, parsed, usage

            n_retry_fixed = 0
            with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
                futs = [ex.submit(retry_one, t) for t in failures]
                for i, fut in enumerate(as_completed(futs), 1):
                    if budget_hit:
                        continue
                    title, parsed, usage = fut.result()
                    cost = usage_cost(usage)
                    total_cost += cost
                    if parsed is not None:
                        ok, unresolved = validate(parsed, all_known)
                        if ok:
                            n_retry_fixed += 1
                            # Replace the original failed entry.
                            for r in results:
                                if r["title"] == title:
                                    r["parsed"] = parsed
                                    r["usage"] = usage
                                    r["cost_usd"] += cost
                                    r["retried"] = True
                                    break
                            save_extract(title, parsed, usage)
                    if i % 100 == 0:
                        print(f"    retry {i}/{len(failures)}  "
                              f"+{n_retry_fixed} fixed  "
                              f"${total_cost:.2f}")
                    if total_cost > BUDGET_USD:
                        budget_hit = True
            print(f"    retry pass fixed {n_retry_fixed}/{len(failures)} "
                  f"failures.\n")

            # Re-run validation post-retry.
            n_parseable = sum(1 for r in results if r["parsed"])
            n_schema_ok = 0
            n_refs_ok = 0
            unresolved_examples = []
            new_entries = []
            proposed_keys = {r["parsed"]["key"] for r in results
                             if r["parsed"]
                             and isinstance(r["parsed"], dict)
                             and isinstance(r["parsed"].get("key"), str)}
            all_known = initial_keys | proposed_keys
            for r in results:
                entry = r["parsed"]
                if not isinstance(entry, dict):
                    continue
                ok, unresolved = validate(entry, all_known)
                if entry.get("key") and entry.get("entity") \
                        and entry.get("composed_of"):
                    n_schema_ok += 1
                if ok:
                    n_refs_ok += 1
                    new_entries.append(entry)
                elif unresolved:
                    unresolved_examples.append((entry.get("key", "?"),
                                                unresolved))
            print(f"  ── validation (post-retry) ──")
            print(f"    JSON parseable:           {n_parseable}/"
                  f"{n_attempted} = "
                  f"{100 * n_parseable / max(1, n_attempted):.1f}%")
            print(f"    Schema fields present:    {n_schema_ok}/"
                  f"{n_attempted} = "
                  f"{100 * n_schema_ok / max(1, n_attempted):.1f}%")
            print(f"    All composed_of resolve:  {n_refs_ok}/"
                  f"{n_attempted} = "
                  f"{100 * n_refs_ok / max(1, n_attempted):.1f}%\n")

    # 7. Integrate clean entries into a new lattice; re-grounded?
    print(f"  ── prime-grounding on the scaled lattice ──")
    new_lattice = dict(lattice)
    for e in new_entries:
        new_lattice[e["key"]] = {
            "key": e["key"],
            "category": e.get("category", "compound"),
            "entity": e.get("entity", {}),
            "composed_of": e.get("composed_of", []),
        }
    n_grounded = 0
    depths: list[int] = []
    primes_seen: set[str] = set()
    for e in new_entries:
        grounded, depth, primes = trace_to_primes(new_lattice, e["key"])
        if grounded:
            n_grounded += 1
            depths.append(depth)
            primes_seen |= primes
    if depths:
        mean_d = sum(depths) / len(depths)
        max_d = max(depths)
    else:
        mean_d = max_d = 0
    print(f"    Of {len(new_entries)} clean entries, "
          f"{n_grounded} ground to layer-0 primes within depth-8.")
    print(f"    Depth-to-primes: mean {mean_d:.1f}, max {max_d}.")
    print(f"    Layer-0 primes reached across new entries: {len(primes_seen)}.")
    print()

    # 8. Save artefacts.
    out_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else HERE / "phase1_output"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "clean_entries.yaml").write_text(yaml.safe_dump(
        new_entries, sort_keys=False, default_flow_style=False))
    summary_payload = {
        "n_attempted": n_attempted,
        "n_parseable": n_parseable,
        "n_schema_ok": n_schema_ok,
        "n_refs_ok": n_refs_ok,
        "n_grounded": n_grounded,
        "mean_depth": mean_d,
        "max_depth": max_d,
        "n_primes_reached": len(primes_seen),
        "primes_reached": sorted(primes_seen),
        "cost_usd": total_cost,
        "model": MODEL,
        "baseline_keys": len(initial_keys),
        "new_keys": sum(1 for e in new_entries
                        if e["key"] not in initial_keys),
        "unresolved_examples": unresolved_examples[:30],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2))
    print(f"  ── artefacts saved to {out_dir}/")
    print(f"    results.json       ({len(results)} entries)")
    print(f"    clean_entries.yaml ({len(new_entries)} clean entries)")
    print(f"    summary.json       (the headline numbers)")
    print()

    # 9. Headline.
    print("━" * 72)
    print(f"  PHASE 1 HEADLINE")
    print("━" * 72)
    print(f"  Composition integrity rate: "
          f"{100 * n_refs_ok / max(1, n_attempted):.1f}%  "
          f"({n_refs_ok}/{n_attempted})")
    print(f"  Prime-grounding rate:       "
          f"{100 * n_grounded / max(1, len(new_entries)):.1f}%  "
          f"({n_grounded}/{len(new_entries)} of clean)")
    print(f"  Substrate growth:           "
          f"{len(initial_keys)} → "
          f"{len(initial_keys) + summary_payload['new_keys']} keys")
    print(f"  Cost:                       ${total_cost:.4f}")
    print(f"  Cost per clean fact:        "
          f"${total_cost / max(1, len(new_entries)):.5f}")
    print(f"  Projected cost / 50K facts: "
          f"${total_cost / max(1, len(new_entries)) * 50_000:.0f}")
    print(f"  Projected cost /  1M facts: "
          f"${total_cost / max(1, len(new_entries)) * 1_000_000:.0f}")
    print("━" * 72)


if __name__ == "__main__":
    main()
