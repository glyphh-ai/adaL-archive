"""Generate a clean Reddit/Twitter-friendly diagram of the Phase 1 result.
Light on mechanics, heavy on the headline numbers.

    python curriculum/agi/chemistry/make_diagram.py
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path(__file__).parent / "phase1_output" / "diagram.png"

# Color palette — calm, technical, no clown colors.
INK = "#1a1a1a"
MUTED = "#5a5a5a"
ACCENT = "#1f6feb"     # blue for the headline result
GREEN = "#1a7f37"      # green for the cost win
PALE = "#f6f8fa"       # box fill
LINE = "#d0d7de"
BG = "#ffffff"

fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_axis_off()


def box(x, y, w, h, fill=PALE, edge=LINE, radius=0.15):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=1.2, edgecolor=edge, facecolor=fill))


def text(x, y, s, size=12, weight="normal", color=INK, ha="center", va="center"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
            ha=ha, va=va, family="DejaVu Sans")


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=18,
        linewidth=1.4, color=MUTED))


# ── Title ──────────────────────────────────────────────────────────
text(8, 8.45, "LLM-assisted ontology extraction at scale",
     size=24, weight="bold")
text(8, 7.95,
     "12,431 Wikipedia chemistry summaries  →  grounded compositional lattice",
     size=14, color=MUTED)

# ── Row 1: pipeline ───────────────────────────────────────────────
# Input
box(0.5, 5.6, 3.8, 1.4)
text(2.4, 6.55, "INPUT", size=10, weight="bold", color=MUTED)
text(2.4, 6.15, "12,431", size=22, weight="bold", color=INK)
text(2.4, 5.80, "Wikipedia chemistry summaries", size=11, color=MUTED)

# Process
box(5.3, 5.6, 5.4, 1.4)
text(8.0, 6.55, "EXTRACTION", size=10, weight="bold", color=MUTED)
text(8.0, 6.15, "Haiku 4.5", size=20, weight="bold", color=INK)
text(8.0, 5.80,
     "schema + 206-key namespace (prompt-cached) + retry-on-failure",
     size=10.5, color=MUTED)

# Output
box(11.7, 5.6, 3.8, 1.4)
text(13.6, 6.55, "OUTPUT", size=10, weight="bold", color=MUTED)
text(13.6, 6.15, "12,178", size=22, weight="bold", color=INK)
text(13.6, 5.80, "grounded concept entries", size=11, color=MUTED)

arrow(4.4, 6.3, 5.2, 6.3)
arrow(10.8, 6.3, 11.6, 6.3)

# ── Row 2: headline numbers ───────────────────────────────────────
text(8, 4.85, "HEADLINE", size=10, weight="bold", color=MUTED)

box(0.5, 2.7, 4.9, 2.0, fill="#eef4fb", edge=ACCENT)
text(2.95, 4.20, "97.8%", size=44, weight="bold", color=ACCENT)
text(2.95, 3.50, "composition integrity", size=13, color=INK)
text(2.95, 3.10, "(every composed_of ref resolves into the lattice)",
     size=9.5, color=MUTED)

box(5.7, 2.7, 4.6, 2.0, fill="#eef7f0", edge=GREEN)
text(8.0, 4.20, "$0.0033", size=44, weight="bold", color=GREEN)
text(8.0, 3.50, "per clean fact", size=13, color=INK)
text(8.0, 3.10, "total run: $40.19 in 22 min on 12K facts",
     size=9.5, color=MUTED)

box(10.6, 2.7, 4.9, 2.0, fill="#eef4fb", edge=ACCENT)
text(13.05, 4.20, "99.9%", size=44, weight="bold", color=ACCENT)
text(13.05, 3.50, "prime-grounding rate", size=13, color=INK)
text(13.05, 3.10, "mean depth 3.1, max 5 (NSM primes)",
     size=9.5, color=MUTED)

# ── Row 3: scaling projection ────────────────────────────────────
text(8, 1.95, "SCALING PROJECTION  (at $0.0033/fact, empirically grounded)",
     size=10, weight="bold", color=MUTED)

cells = [
    ("12K facts",   "this run",        "$40"),
    ("50K facts",   "single domain",   "$165"),
    ("1M facts",    "mid-corpus",      "$3.3K"),
    ("50M facts",   "Wikidata-scale",  "$165K"),
]
col_w = 3.5
start_x = 1.0
y_top = 1.5
y_bot = 0.35
for i, (scale, label, cost) in enumerate(cells):
    cx = start_x + i * col_w + col_w / 2
    box(start_x + i * col_w + 0.15, y_bot, col_w - 0.3, y_top - y_bot)
    text(cx, 1.30, scale, size=13, weight="bold")
    text(cx, 1.00, label, size=10, color=MUTED)
    text(cx, 0.60, cost, size=18, weight="bold", color=GREEN)

# ── Footer ───────────────────────────────────────────────────────
text(0.4, 0.05,
     "github.com/timmetim/ada  ·  curriculum/agi/chemistry  ·  Phase 1 report",
     size=9, color=MUTED, ha="left")
text(15.6, 0.05, "claude-haiku-4-5-20251001",
     size=9, color=MUTED, ha="right")

plt.savefig(OUT, dpi=120, bbox_inches="tight", facecolor=BG)
print(f"wrote {OUT}")
