"""Destination-shape generators for congestion experiments.

Design goals:
- Keep all destination layouts in one place (used by both experiments + visualization).
- Use a single scalable ASCII-art builder for text / drawings.

Coordinates are (col, row).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


def build_ascii_art_destination_cells(
    art: list[str],
    rows: int,
    cols: int,
    *,
    on_chars: Iterable[str] = ("#",),
) -> list[tuple[int, int]]:
    """Scale and center an ASCII-art mask into a (rows x cols) grid.

    - `art` is given top-to-bottom as human-readable lines.
    - Any character in `on_chars` becomes a destination cell.
    - The art is uniformly scaled in x/y by integer factors (sx, sy) and centered.

    If the grid is too small and the art would produce zero cells, we fall back to a
    single center destination cell so the MDP remains well-defined.
    """

    if rows <= 0 or cols <= 0:
        return []

    if not art:
        return [(cols // 2, rows // 2)]

    on = set(on_chars)

    art_h = len(art)
    art_w = max(len(line) for line in art)

    # Scale factors chosen to fit art with a small margin.
    sx = max(1, cols // (art_w + 2))
    sy = max(1, rows // (art_h + 2))

    draw_w = art_w * sx
    draw_h = art_h * sy
    x0 = max(0, (cols - draw_w) // 2)
    y0 = max(0, (rows - draw_h) // 2)

    dest: set[tuple[int, int]] = set()
    for r, line in enumerate(art):
        for c, ch in enumerate(line):
            if ch not in on:
                continue
            for dy in range(sy):
                for dx in range(sx):
                    x = x0 + c * sx + dx
                    # Flip vertically: art[0] is top row.
                    y = y0 + (art_h - 1 - r) * sy + dy
                    if 0 <= x < cols and 0 <= y < rows:
                        dest.add((x, y))

    if not dest:
        return [(cols // 2, rows // 2)]

    return sorted(dest)


_H_SHAPE = [
    " #       # ",
    " #       # ",
    " #       # ",
    " ######### ",
    " #       # ",
    " #       # ",
    " #       # ",
]

_IITB_ART = [
    "### ### ##### #### ",
    " #   #    #   #   #",
    " #   #    #   #   #",
    " #   #    #   #### ",
    " #   #    #   #   #",
    " #   #    #   #   #",
    "### ###   #   #### ",
]


# 7-row block-ish art for "HELLO WORLD".
# It intentionally favors readability over compactness.
_HELLO_WORLD_ART = [
    "#   # ##### #     #      ###        #   #  ###  ####  #     #### ",
    "#   # #     #     #     #   #       #   # #   # #   # #     #   #",
    "##### ####  #     #     #   #       #   # #   # ####  #     #   #",
    "#   # #     #     #     #   #       # # # #   # # #   #     #   #",
    "#   # #     #     #     #   #       ## ## #   # #  #  #     #   #",
    "#   # #     #     #     #   #       #   # #   # #   # #     #   #",
    "#   # ##### ##### #####  ###        #   #  ###  #   # ##### #### ",
]


# A corrected smiley-face using ASCII art:
# - Clear boundary
# - Two eyes
# - Curved smile
_SMILEY_ART = [
    "   #####   ",
    "  #     #  ",
    " #  # #  # ",
    "#         #",
    "#  #   #  #",
    "#   ###   #",
    " #       # ",
    "  #     #  ",
    "   #####   ",
]

def dest_h_shape(rows: int, cols: int) -> list[tuple[int, int]]:
    return build_ascii_art_destination_cells(_H_SHAPE, rows=rows, cols=cols, on_chars=("#",))

def dest_word_iitb(rows: int, cols: int) -> list[tuple[int, int]]:
    return build_ascii_art_destination_cells(_IITB_ART, rows=rows, cols=cols, on_chars=("#",))


def dest_word_hello_world(rows: int, cols: int) -> list[tuple[int, int]]:
    return build_ascii_art_destination_cells(_HELLO_WORLD_ART, rows=rows, cols=cols, on_chars=("#",))


def dest_smiley(rows: int, cols: int) -> list[tuple[int, int]]:
    return build_ascii_art_destination_cells(_SMILEY_ART, rows=rows, cols=cols, on_chars=("#",))


DESTINATION_SHAPES: dict[str, Callable[[int, int], list[tuple[int, int]]]] = {
    "h_center": dest_h_shape,
    "iitb": dest_word_iitb,
    "hello_world": dest_word_hello_world,
    "smiley": dest_smiley,
}
