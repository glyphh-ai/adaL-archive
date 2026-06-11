# Open-source launch checklist

*Repo is launch-ready except the items marked ☐ DECISION — those are
Chris's calls, not engineering tasks. Delete this file before or at
launch.*

## Done (verified in repo)

- ☑ Full-history secrets scan (see commit message for method/results)
- ☑ Test suite (19 tests, offline, <1s) + CI (ruff + pytest on
  3.11–3.13 + full import sweep)
- ☑ CONTRIBUTING.md with the pre-registration house rule
- ☑ Dead code: reachability-clean from all entry points
- ☑ Legacy curriculum: scripts removed, data kept as audit record with
  provenance README
- ☑ MCP auth enforceable (`ADA_AUTH_REQUIRED`), off by default with
  boot warning
- ☑ Paper, protocol, all phase reports + raw logs committed

## ☐ DECISION: license (BLOCKER)

`LICENSE-APACHE-2.0.draft` is staged. **Do not swap it in without
patent counsel**: the current Glyphh AI Community License cites
*Patent Pending App. No. 63/969,729*, and Apache-2.0 §3 grants
licensees a patent license to claims that read on the contributed
code. Options:
- Apache-2.0 (recommended for the open-core strategy) — accepts the
  grant for this codebase
- BUSL-1.1 (source-available, converts to Apache later) — if Ada
  itself is the revenue line
- Confirm whether 63/969,729's claims even read on the post-HDC
  codebase (the vectors the application likely covers were removed)

Also update: `pyproject.toml` classifier + `license` field, README
license section, paper footer.

## ☐ DECISION: name

"Ada" collides with the programming language and several AI products.
Options: keep `adaL` as the repo/product name, rename, or ship as
"Ada by Glyphh". Ten minutes of trademark search before HN, minimum.

## ☐ DECISION: repo history

History contains the full arc — HDC paper era, audit, rewrite. This is
a *strength* for the honesty narrative (the paper references it), but
review once more whether anything in old commits is unwanted in public
(code comments, internal names). No secrets found by scan.

## Remaining engineering (nice-to-have, not blockers)

- ☐ `make test` badge + repo description/topics
- ☐ Issue templates (bug / benchmark-dispute — the second one matters
  for this project's positioning)
- ☐ Tag v0.7.0 release at launch commit
- ☐ Decide public default for `ADA_AUTH_REQUIRED` in docker-compose
  (suggest: true, with token bootstrap in compose docs)
