"""
Braille-character line graph for the terminal.

Renders a list of values as a sparkline graph using Unicode braille
characters. Each braille cell is 2 dots wide x 4 dots tall, giving
sub-character resolution for smooth curves.
"""

from __future__ import annotations

from rich.text import Text
from rich.style import Style


# Braille dot positions within a cell:
#   0  3
#   1  4
#   2  5
#   6  7
# Offset from U+2800
_DOT_OFFSETS = [0x01, 0x02, 0x04, 0x40, 0x08, 0x10, 0x20, 0x80]


def render_graph(
    values: list[float],
    width: int = 60,
    height: int = 12,
    color: str = "#6b7bff",
    axis_color: str = "#444444",
    label: str = "",
    fmt: str = ".1f",
) -> Text:
    """Render a braille line graph of values.

    Args:
        values: The data points to plot.
        width: Character width of the graph area.
        height: Character height of the graph area.
        color: Color for the line.
        axis_color: Color for axes and labels.
        label: Y-axis label.
        fmt: Format string for axis labels.

    Returns:
        Rich Text object ready to display.
    """
    if not values:
        return Text("No data.")

    # Canvas resolution: each char is 2 dots wide, 4 dots tall
    canvas_w = width * 2
    canvas_h = height * 4

    v_min = min(values)
    v_max = max(values)
    v_range = v_max - v_min if v_max != v_min else 1.0

    # Map values to canvas y coordinates (0 = top, canvas_h-1 = bottom)
    def to_canvas_y(v: float) -> int:
        normalized = (v - v_min) / v_range
        return max(0, min(canvas_h - 1, int((1 - normalized) * (canvas_h - 1))))

    # Map data indices to canvas x coordinates
    n = len(values)
    def to_canvas_x(i: int) -> int:
        if n == 1:
            return 0
        return int(i * (canvas_w - 1) / (n - 1))

    # Build canvas -- set of (x, y) dots that are lit
    dots: set[tuple[int, int]] = set()

    for i in range(n):
        x = to_canvas_x(i)
        y = to_canvas_y(values[i])
        dots.add((x, y))

        # Draw line segments between consecutive points
        if i > 0:
            x0 = to_canvas_x(i - 1)
            y0 = to_canvas_y(values[i - 1])
            _draw_line(dots, x0, y0, x, y)

    # Render canvas to braille characters
    text = Text()

    # Y-axis labels
    y_label_width = 8

    for row in range(height):
        # Y-axis label (top, middle, bottom)
        if row == 0:
            lbl = f"{v_max:{fmt}}"
        elif row == height - 1:
            lbl = f"{v_min:{fmt}}"
        elif row == height // 2:
            mid = (v_max + v_min) / 2
            lbl = f"{mid:{fmt}}"
        else:
            lbl = ""

        text.append(f"{lbl:>{y_label_width}} ", style=Style(color=axis_color))

        # Axis line
        text.append("\u2502", style=Style(color=axis_color))

        # Braille characters for this row
        for col in range(width):
            code = 0x2800
            for dy in range(4):
                for dx in range(2):
                    px = col * 2 + dx
                    py = row * 4 + dy
                    if (px, py) in dots:
                        dot_idx = dy + dx * 4 if dy < 3 else (6 + dx)
                        code |= _DOT_OFFSETS[dot_idx]
            text.append(chr(code), style=Style(color=color))

        text.append("\n")

    # X-axis
    text.append(" " * y_label_width + " \u2514" + "\u2500" * width, style=Style(color=axis_color))

    return text


def _draw_line(dots: set[tuple[int, int]], x0: int, y0: int, x1: int, y1: int) -> None:
    """Bresenham's line algorithm to fill dots between two points."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        dots.add((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
