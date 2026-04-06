"""Braille character banner for Ada."""

from rich.text import Text
from rich.style import Style

# "ada" rendered in braille block characters
# Extracted from adaL banner3 figlet font, doubled vertically
BANNER_LINES = [
    "⠀⣼⢻⡄⠀⣿⠛⠛⢻⡄⠀⣼⢻⡄",
    "⣼⠃⠀⢻⡄⣿⠀⠀⢸⡇⣼⠃⠀⢻⡄",
    "⣿⠛⠛⢻⡇⣿⠀⠀⢸⡇⣿⠛⠛⢻⡇",
    "⠛⠀⠀⠘⠃⠛⠛⠛⠛⠀⠛⠀⠀⠘⠃",
]

# Gradient colors from warm to cool
GRADIENT = [
    "#ff6b6b",
    "#ee5a9f",
    "#cc5fd3",
    "#9b6bea",
    "#6b7bff",
]


def get_banner(version: str = "0.1.0") -> Text:
    """Return the Ada banner as styled Rich Text with gradient."""
    text = Text()
    total_width = max(len(line) for line in BANNER_LINES)

    for i, line in enumerate(BANNER_LINES):
        # Interpolate gradient color per row
        t = i / max(len(BANNER_LINES) - 1, 1)
        idx = t * (len(GRADIENT) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(GRADIENT) - 1)
        frac = idx - lo

        r1, g1, b1 = _hex_to_rgb(GRADIENT[lo])
        r2, g2, b2 = _hex_to_rgb(GRADIENT[hi])
        r = int(r1 + (r2 - r1) * frac)
        g = int(g1 + (g2 - g1) * frac)
        b = int(b1 + (b2 - b1) * frac)
        color = f"#{r:02x}{g:02x}{b:02x}"

        text.append(line, style=Style(color=color, bold=True))
        if i < len(BANNER_LINES) - 1:
            text.append("\n")

    # Version line, right-aligned under the banner
    version_str = f"v{version}"
    padding = total_width - len(version_str)
    text.append("\n")
    text.append(" " * padding + version_str, style=Style(color="#666666"))

    return text


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
