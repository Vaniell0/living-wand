"""Synthetic symbol data generator for OCR training.

Generates 64x64 grayscale images of hand-drawn-style magic symbols.
Uses random perturbation to simulate imprecise mouse/pen drawing.
"""

import numpy as np
import math
import random

IMG_SIZE = 64
LINE_WIDTH_RANGE = (2, 4)

# All 30 primitives + noise class
SYMBOL_NAMES = [
    # Elements (6)
    "fire",      # 0 — upward triangle (flame)
    "water",     # 1 — wavy lines
    "earth",     # 2 — square
    "air",       # 3 — spiral
    "light",     # 4 — sun/star (circle + rays)
    "dark",      # 5 — crescent moon
    # Domains (6)
    "body",      # 6 — stick torso
    "mind",      # 7 — eye shape
    "matter",    # 8 — cube (3D box)
    "space",     # 9 — diamond
    "time",      # 10 — hourglass
    "life",      # 11 — circle with dot
    # Directions (6)
    "inward",    # 12 — arrow pointing to center
    "outward",   # 13 — arrow pointing away
    "up",        # 14 — chevron ^
    "down",      # 15 — chevron v
    "through",   # 16 — arrow with crossbar
    "reverse",   # 17 — U-turn arrow
    # Quantity (4)
    "point",     # 18 — single dot
    "flow",      # 19 — wavy arrow
    "wave",      # 20 — sine wave
    "all",       # 21 — circle with cross
    # Modifiers (4)
    "amplify",   # 22 — large up-triangle
    "weaken",    # 23 — small down-triangle
    "hold",      # 24 — two vertical bars (pause)
    "dispel",    # 25 — X cross
    # Meta (4)
    "here",      # 26 — dot in small circle
    "there",     # 27 — dot in dashed circle
    "self",      # 28 — vertical line with dot on top
    "target",    # 29 — crosshair
    # Noise
    "noise",     # 30 — random scribbles
]
NUM_SYMBOLS = len(SYMBOL_NAMES)


def _jitter(x: float, y: float, amount: float = 2.0) -> tuple[float, float]:
    """Add random jitter to a point."""
    return (
        x + random.gauss(0, amount),
        y + random.gauss(0, amount),
    )


class StrokeRecorder:
    """Drop-in for the 64×64 canvas that records geometry instead of pixels.

    Every draw_* function below draws through _draw_line (or writes dot
    pixels directly), so passing a StrokeRecorder turns all 30 rune painters
    into stroke generators for free. Consecutive segments are merged into
    polylines; direct pixel writes (dots) become one dot stroke.
    """

    shape = (IMG_SIZE, IMG_SIZE)

    def __init__(self):
        self.polylines: list[list[tuple[float, float]]] = []
        self._dot_pixels: list[tuple[int, int]] = []

    def add_segment(self, x0: float, y0: float, x1: float, y1: float):
        if self.polylines:
            lx, ly = self.polylines[-1][-1]
            if abs(lx - x0) < 1e-3 and abs(ly - y0) < 1e-3:
                self.polylines[-1].append((x1, y1))
                return
        self.polylines.append([(x0, y0), (x1, y1)])

    def __setitem__(self, key, value):
        py, px = key
        self._dot_pixels.append((px, py))

    def __getitem__(self, key):
        return 0.0

    def strokes(self) -> list[list[tuple[float, float]]]:
        out = [list(p) for p in self.polylines]
        if self._dot_pixels:
            xs, ys = zip(*self._dot_pixels)
            out.append([(float(np.mean(xs)), float(np.mean(ys)))])
        return out


def _draw_line(img: np.ndarray, x0: float, y0: float, x1: float, y1: float,
               thickness: int = 2, jitter: float = 1.5):
    """Draw a line with hand-drawn jitter using Bresenham-like approach."""
    if isinstance(img, StrokeRecorder):
        img.add_segment(x0, y0, x1, y1)
        return
    n_points = max(int(math.hypot(x1 - x0, y1 - y0)), 2)
    for i in range(n_points + 1):
        t = i / n_points
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        jx, jy = _jitter(x, y, jitter)
        ix, iy = int(round(jx)), int(round(jy))
        for dx in range(-thickness // 2, thickness // 2 + 1):
            for dy in range(-thickness // 2, thickness // 2 + 1):
                px, py = ix + dx, iy + dy
                if 0 <= px < IMG_SIZE and 0 <= py < IMG_SIZE:
                    img[py, px] = min(1.0, img[py, px] + 0.8)


def _draw_circle(img: np.ndarray, cx: float, cy: float, r: float,
                 thickness: int = 2, jitter: float = 1.5, n_points: int = 40):
    """Draw a circle with hand-drawn jitter."""
    for i in range(n_points):
        a0 = 2 * math.pi * i / n_points
        a1 = 2 * math.pi * (i + 1) / n_points
        x0, y0 = cx + r * math.cos(a0), cy + r * math.sin(a0)
        x1, y1 = cx + r * math.cos(a1), cy + r * math.sin(a1)
        _draw_line(img, x0, y0, x1, y1, thickness, jitter)


def _draw_arc(img: np.ndarray, cx: float, cy: float, r: float,
              start_angle: float, end_angle: float,
              thickness: int = 2, jitter: float = 1.5, n_points: int = 20):
    """Draw an arc."""
    for i in range(n_points):
        a0 = start_angle + (end_angle - start_angle) * i / n_points
        a1 = start_angle + (end_angle - start_angle) * (i + 1) / n_points
        x0, y0 = cx + r * math.cos(a0), cy + r * math.sin(a0)
        x1, y1 = cx + r * math.cos(a1), cy + r * math.sin(a1)
        _draw_line(img, x0, y0, x1, y1, thickness, jitter)


def _random_transform(cx: float = 32, cy: float = 32, scale_range: tuple = (0.4, 1.2)):
    """Random center offset + scale + rotation for variety."""
    scale = random.uniform(*scale_range)
    ox = random.gauss(0, 6)
    oy = random.gauss(0, 6)
    return cx + ox, cy + oy, scale


def _rotate_point(x, y, cx, cy, angle):
    """Rotate (x,y) around (cx,cy) by angle radians."""
    dx, dy = x - cx, y - cy
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a


def draw_fire(img: np.ndarray):
    """Upward triangle — flame. Many style variations."""
    cx, cy, s = _random_transform()
    rot = random.gauss(0, 0.3)  # slight random rotation
    h = random.uniform(16, 26) * s
    w = random.uniform(12, 22) * s
    top = _rotate_point(cx, cy - h * 0.6, cx, cy, rot)
    bl = _rotate_point(cx - w, cy + h * 0.4, cx, cy, rot)
    br = _rotate_point(cx + w, cy + h * 0.4, cx, cy, rot)
    th = random.randint(*LINE_WIDTH_RANGE)
    j = random.uniform(1.5, 4.0)
    # Sometimes skip one side (incomplete triangle)
    sides = [(bl, top), (top, br), (br, bl)]
    if random.random() < 0.15:
        sides.pop(random.randint(0, 2))
    for a, b in sides:
        _draw_line(img, *a, *b, th, j)


def draw_water(img: np.ndarray):
    """Wavy lines — 1 to 3, various orientations."""
    cx, cy, s = _random_transform()
    th = random.randint(*LINE_WIDTH_RANGE)
    j = random.uniform(1.5, 3.0)
    rot = random.gauss(0, 0.2)
    n_waves = random.randint(1, 3)
    freq = random.uniform(1.5, 3.0)
    amp = random.uniform(3, 8) * s
    wave_w = random.uniform(24, 36) * s
    for row in range(n_waves):
        y_off = cy + (row - n_waves / 2) * 10 * s
        n_pts = 20
        pts = []
        for i in range(n_pts + 1):
            t = i / n_pts
            x = cx - wave_w / 2 + wave_w * t
            dy = amp * math.sin(t * freq * math.pi + row * 0.5)
            rx, ry = _rotate_point(x, y_off + dy, cx, cy, rot)
            pts.append((rx, ry))
        for i in range(len(pts) - 1):
            _draw_line(img, *pts[i], *pts[i + 1], th, j)


def draw_earth(img: np.ndarray):
    """Square / rectangle — AXIS-ALIGNED (no rotation, distinguishes from diamond/space)."""
    cx, cy, s = _random_transform()
    rot = 0  # no rotation — key difference from space (diamond)
    hw = random.uniform(12, 20) * s
    hh = random.uniform(12, 20) * s
    corners = [
        _rotate_point(cx - hw, cy - hh, cx, cy, rot),
        _rotate_point(cx + hw, cy - hh, cx, cy, rot),
        _rotate_point(cx + hw, cy + hh, cx, cy, rot),
        _rotate_point(cx - hw, cy + hh, cx, cy, rot),
    ]
    th = random.randint(*LINE_WIDTH_RANGE)
    j = random.uniform(1.5, 4.0)
    sides = list(range(4))
    if random.random() < 0.1:
        sides.pop(random.randint(0, 3))
    for i in sides:
        _draw_line(img, *corners[i], *corners[(i + 1) % 4], th, j)


def draw_air(img: np.ndarray):
    """Spiral — various tightness, direction, completeness."""
    cx, cy, s = _random_transform()
    th = random.randint(*LINE_WIDTH_RANGE)
    j = random.uniform(1.0, 3.0)
    turns = random.uniform(1.5, 3.5)
    direction = random.choice([1, -1])
    n_pts = random.randint(30, 70)
    start_r = random.uniform(2, 6) * s
    end_r = random.uniform(14, 22) * s
    for i in range(n_pts):
        t0 = i / n_pts * turns * 2 * math.pi * direction
        t1 = (i + 1) / n_pts * turns * 2 * math.pi * direction
        r0 = start_r + (end_r - start_r) * (i / n_pts)
        r1 = start_r + (end_r - start_r) * ((i + 1) / n_pts)
        x0, y0 = cx + r0 * math.cos(t0), cy + r0 * math.sin(t0)
        x1, y1 = cx + r1 * math.cos(t1), cy + r1 * math.sin(t1)
        _draw_line(img, x0, y0, x1, y1, th, j)


def draw_life(img: np.ndarray):
    """Circle (with or without dot) — various sizes, elliptical."""
    cx, cy, s = _random_transform()
    th = random.randint(*LINE_WIDTH_RANGE)
    j = random.uniform(1.5, 3.0)
    r = random.uniform(14, 22) * s
    # Sometimes elliptical
    squash = random.uniform(0.7, 1.0)
    n_pts = random.randint(25, 45)
    for i in range(n_pts):
        a0 = 2 * math.pi * i / n_pts
        a1 = 2 * math.pi * (i + 1) / n_pts
        x0, y0 = cx + r * math.cos(a0), cy + r * squash * math.sin(a0)
        x1, y1 = cx + r * math.cos(a1), cy + r * squash * math.sin(a1)
        _draw_line(img, x0, y0, x1, y1, th, j)
    # Dot (sometimes absent)
    if random.random() < 0.7:
        dot_r = random.randint(1, 3)
        for dx in range(-dot_r, dot_r + 1):
            for dy in range(-dot_r, dot_r + 1):
                px, py = int(cx) + dx, int(cy) + dy
                if 0 <= px < IMG_SIZE and 0 <= py < IMG_SIZE:
                    img[py, px] = 1.0


def draw_mind(img: np.ndarray):
    """Eye shape — two arcs, various openness."""
    cx, cy, s = _random_transform()
    rot = random.gauss(0, 0.3)
    th = random.randint(*LINE_WIDTH_RANGE)
    j = random.uniform(1.5, 3.0)
    openness = random.uniform(6, 12) * s
    width = random.uniform(16, 24) * s
    _draw_arc(img, cx, cy + openness, width, -math.pi * 0.8, -math.pi * 0.2, th, j)
    _draw_arc(img, cx, cy - openness, width, math.pi * 0.2, math.pi * 0.8, th, j)
    if random.random() < 0.7:
        _draw_circle(img, cx, cy, random.uniform(3, 7) * s, th, j, n_points=12)


def draw_space(img: np.ndarray):
    """Diamond — various proportions and rotation."""
    cx, cy, s = _random_transform()
    rot = random.gauss(0, 0.4)
    hv = random.uniform(16, 24) * s  # vertical half
    hh = random.uniform(14, 22) * s  # horizontal half
    corners = [
        _rotate_point(cx, cy - hv, cx, cy, rot),
        _rotate_point(cx + hh, cy, cx, cy, rot),
        _rotate_point(cx, cy + hv, cx, cy, rot),
        _rotate_point(cx - hh, cy, cx, cy, rot),
    ]
    th = random.randint(*LINE_WIDTH_RANGE)
    j = random.uniform(1.5, 4.0)
    for i in range(4):
        _draw_line(img, *corners[i], *corners[(i + 1) % 4], th, j)


def draw_time(img: np.ndarray):
    """Hourglass — two triangles, various proportions."""
    cx, cy, s = _random_transform()
    rot = random.gauss(0, 0.3)
    h = random.uniform(10, 18) * s
    w = random.uniform(10, 18) * s
    th = random.randint(*LINE_WIDTH_RANGE)
    j = random.uniform(1.5, 4.0)
    pts_top = [
        _rotate_point(cx - w, cy - h, cx, cy, rot),
        _rotate_point(cx + w, cy - h, cx, cy, rot),
        _rotate_point(cx, cy, cx, cy, rot),
    ]
    pts_bot = [
        _rotate_point(cx - w, cy + h, cx, cy, rot),
        _rotate_point(cx + w, cy + h, cx, cy, rot),
        _rotate_point(cx, cy, cx, cy, rot),
    ]
    for i in range(3):
        _draw_line(img, *pts_top[i], *pts_top[(i + 1) % 3], th, j)
        _draw_line(img, *pts_bot[i], *pts_bot[(i + 1) % 3], th, j)


def draw_light(img: np.ndarray):
    """Sun/star — circle with 4-8 rays."""
    cx, cy, s = _random_transform()
    th = random.randint(*LINE_WIDTH_RANGE)
    j = random.uniform(1.0, 2.5)
    r = random.uniform(8, 14) * s
    _draw_circle(img, cx, cy, r, th, j, n_points=20)
    n_rays = random.randint(4, 8)
    ray_len = random.uniform(8, 14) * s
    rot = random.uniform(0, math.pi / n_rays)
    for i in range(n_rays):
        a = rot + 2 * math.pi * i / n_rays
        x0, y0 = cx + r * math.cos(a), cy + r * math.sin(a)
        x1, y1 = cx + (r + ray_len) * math.cos(a), cy + (r + ray_len) * math.sin(a)
        _draw_line(img, x0, y0, x1, y1, th, j)


def draw_dark(img: np.ndarray):
    """Crescent moon."""
    cx, cy, s = _random_transform()
    th = random.randint(*LINE_WIDTH_RANGE)
    j = random.uniform(1.0, 2.5)
    r = random.uniform(14, 22) * s
    rot = random.gauss(0, 0.3)
    # Outer arc
    _draw_arc(img, cx, cy, r, -math.pi * 0.6 + rot, math.pi * 0.6 + rot, th, j, 25)
    # Inner arc (shifted to create crescent)
    offset = r * random.uniform(0.3, 0.5)
    inner_r = r * random.uniform(0.75, 0.95)
    _draw_arc(img, cx + offset * math.cos(rot), cy + offset * math.sin(rot),
              inner_r, -math.pi * 0.5 + rot, math.pi * 0.5 + rot, th, j, 20)


def draw_body(img: np.ndarray):
    """Stick figure torso — vertical line + two arm lines."""
    cx, cy, s = _random_transform()
    th = random.randint(*LINE_WIDTH_RANGE)
    j = random.uniform(1.5, 3.0)
    h = random.uniform(18, 26) * s
    arm_w = random.uniform(10, 16) * s
    arm_y = cy - h * random.uniform(0.1, 0.3)
    # Spine
    _draw_line(img, cx, cy - h * 0.5, cx, cy + h * 0.5, th, j)
    # Arms
    _draw_line(img, cx - arm_w, arm_y, cx + arm_w, arm_y, th, j)
    # Head (small circle, sometimes)
    if random.random() < 0.6:
        _draw_circle(img, cx, cy - h * 0.5 - 5 * s, 4 * s, th, j, 12)


def draw_matter(img: np.ndarray):
    """Cube (isometric 3D box)."""
    cx, cy, s = _random_transform()
    th = random.randint(*LINE_WIDTH_RANGE)
    j = random.uniform(1.5, 3.0)
    size = random.uniform(12, 18) * s
    d = size * 0.5  # depth offset
    # Front face
    fl = (cx - size, cy)
    fr = (cx + size, cy)
    ft = (cx - size, cy - size)
    ftr = (cx + size, cy - size)
    _draw_line(img, *fl, *fr, th, j)
    _draw_line(img, *fr, *ftr, th, j)
    _draw_line(img, *ftr, *ft, th, j)
    _draw_line(img, *ft, *fl, th, j)
    # Back edges (depth)
    br = (fr[0] + d, fr[1] - d)
    btr = (ftr[0] + d, ftr[1] - d)
    bt = (ft[0] + d, ft[1] - d)
    _draw_line(img, *fr, *br, th, j)
    _draw_line(img, *ftr, *btr, th, j)
    _draw_line(img, *br, *btr, th, j)
    if random.random() < 0.5:
        _draw_line(img, *ft, *bt, th, j)
        _draw_line(img, *bt, *btr, th, j)


def draw_inward(img: np.ndarray):
    """Arrow pointing inward to center — ALWAYS from right side (distinguishes from outward)."""
    cx, cy, s = _random_transform()
    th = random.randint(*LINE_WIDTH_RANGE)
    j = random.uniform(1.5, 3.0)
    rot = random.gauss(0, 0.3)  # mostly from right, slight variation
    length = random.uniform(16, 24) * s
    # Arrow shaft from outside to center
    ox = cx + length * math.cos(rot)
    oy = cy + length * math.sin(rot)
    _draw_line(img, ox, oy, cx, cy, th, j)
    # Arrowhead
    head_len = 6 * s
    for da in [0.4, -0.4]:
        hx = cx + head_len * math.cos(rot + math.pi + da)
        hy = cy + head_len * math.sin(rot + math.pi + da)
        _draw_line(img, cx, cy, hx, hy, th, j)
    # Center dot
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            px, py = int(cx) + dx, int(cy) + dy
            if 0 <= px < IMG_SIZE and 0 <= py < IMG_SIZE:
                img[py, px] = 1.0


def draw_outward(img: np.ndarray):
    """Arrow pointing outward — ALWAYS to right side + extra tail mark (distinguishes from inward)."""
    cx, cy, s = _random_transform()
    th = random.randint(*LINE_WIDTH_RANGE)
    j = random.uniform(1.5, 3.0)
    rot = random.gauss(math.pi, 0.3)  # mostly to right (opposite of inward)
    length = random.uniform(16, 24) * s
    tip_x = cx + length * math.cos(rot)
    tip_y = cy + length * math.sin(rot)
    _draw_line(img, cx, cy, tip_x, tip_y, th, j)
    # Arrowhead at tip
    head_len = 6 * s
    for da in [2.7, -2.7]:
        hx = tip_x + head_len * math.cos(rot + da)
        hy = tip_y + head_len * math.sin(rot + da)
        _draw_line(img, tip_x, tip_y, hx, hy, th, j)
    # Tail circle at origin (distinguishes from inward)
    _draw_circle(img, cx, cy, 3 * s, max(1, th - 1), j, 10)


def draw_up(img: np.ndarray):
    """Chevron ^ pointing up."""
    cx, cy, s = _random_transform()
    th = random.randint(*LINE_WIDTH_RANGE)
    j = random.uniform(1.5, 3.5)
    w = random.uniform(14, 22) * s
    h = random.uniform(12, 20) * s
    _draw_line(img, cx - w, cy + h * 0.4, cx, cy - h * 0.4, th, j)
    _draw_line(img, cx, cy - h * 0.4, cx + w, cy + h * 0.4, th, j)


def draw_down(img: np.ndarray):
    """Chevron v pointing down."""
    cx, cy, s = _random_transform()
    th = random.randint(*LINE_WIDTH_RANGE)
    j = random.uniform(1.5, 3.5)
    w = random.uniform(14, 22) * s
    h = random.uniform(12, 20) * s
    _draw_line(img, cx - w, cy - h * 0.4, cx, cy + h * 0.4, th, j)
    _draw_line(img, cx, cy + h * 0.4, cx + w, cy - h * 0.4, th, j)


def draw_through(img: np.ndarray):
    """Horizontal arrow with vertical crossbar."""
    cx, cy, s = _random_transform()
    th = random.randint(*LINE_WIDTH_RANGE)
    j = random.uniform(1.5, 3.0)
    w = random.uniform(18, 26) * s
    # Shaft
    _draw_line(img, cx - w, cy, cx + w, cy, th, j)
    # Arrowhead
    head = 6 * s
    _draw_line(img, cx + w, cy, cx + w - head, cy - head * 0.6, th, j)
    _draw_line(img, cx + w, cy, cx + w - head, cy + head * 0.6, th, j)
    # Crossbar
    bar_h = random.uniform(8, 14) * s
    _draw_line(img, cx, cy - bar_h, cx, cy + bar_h, th, j)


def draw_reverse(img: np.ndarray):
    """U-turn / curved return arrow."""
    cx, cy, s = _random_transform()
    th = random.randint(*LINE_WIDTH_RANGE)
    j = random.uniform(1.5, 3.0)
    r = random.uniform(12, 18) * s
    # U-arc (top half)
    _draw_arc(img, cx, cy, r, math.pi, 2 * math.pi, th, j, 20)
    # Left leg down
    _draw_line(img, cx - r, cy, cx - r, cy + r * 0.6, th, j)
    # Right leg with arrowhead
    _draw_line(img, cx + r, cy, cx + r, cy + r * 0.6, th, j)
    head = 5 * s
    _draw_line(img, cx + r, cy + r * 0.6, cx + r - head, cy + r * 0.6 - head, th, j)
    _draw_line(img, cx + r, cy + r * 0.6, cx + r + head, cy + r * 0.6 - head, th, j)


def draw_point(img: np.ndarray):
    """Single dot — large filled circle."""
    cx, cy, s = _random_transform()
    r = int(random.uniform(4, 8) * s)
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                px, py = int(cx) + dx, int(cy) + dy
                if 0 <= px < IMG_SIZE and 0 <= py < IMG_SIZE:
                    img[py, px] = 1.0


def draw_flow(img: np.ndarray):
    """Wavy arrow → rightward — with straight baseline underneath (distinguishes from wave)."""
    cx, cy, s = _random_transform()
    th = random.randint(*LINE_WIDTH_RANGE)
    j = random.uniform(1.0, 2.5)
    w = random.uniform(22, 32) * s
    amp = random.uniform(4, 8) * s
    freq = random.uniform(1.5, 2.5)
    pts = []
    for i in range(25):
        t = i / 24
        x = cx - w / 2 + w * t
        y = cy + amp * math.sin(t * freq * math.pi * 2)
        pts.append((x, y))
    for i in range(len(pts) - 1):
        _draw_line(img, *pts[i], *pts[i + 1], th, j)
    # Large arrowhead at end
    head = 8 * s
    _draw_line(img, *pts[-1], pts[-1][0] - head, pts[-1][1] - head * 0.6, th + 1, j)
    _draw_line(img, *pts[-1], pts[-1][0] - head, pts[-1][1] + head * 0.6, th + 1, j)
    # Straight baseline (distinguishes from wave)
    _draw_line(img, cx - w / 2, cy + amp + 6 * s, cx + w / 2, cy + amp + 6 * s, max(1, th - 1), j)


def draw_wave(img: np.ndarray):
    """Sine wave ∿ (no arrow)."""
    cx, cy, s = _random_transform()
    th = random.randint(*LINE_WIDTH_RANGE)
    j = random.uniform(1.0, 2.5)
    w = random.uniform(24, 34) * s
    amp = random.uniform(6, 12) * s
    freq = random.uniform(1.5, 3.0)
    pts = []
    for i in range(30):
        t = i / 29
        x = cx - w / 2 + w * t
        y = cy + amp * math.sin(t * freq * math.pi * 2)
        pts.append((x, y))
    for i in range(len(pts) - 1):
        _draw_line(img, *pts[i], *pts[i + 1], th, j)


def draw_all(img: np.ndarray):
    """Circle with FULL cross extending beyond circle (⊕, distinguishes from air/life)."""
    cx, cy, s = _random_transform()
    th = random.randint(*LINE_WIDTH_RANGE)
    j = random.uniform(1.5, 3.0)
    r = random.uniform(12, 18) * s
    _draw_circle(img, cx, cy, r, th, j, 30)
    # Cross extends BEYOND circle (key visual difference)
    ext = r + 6 * s
    _draw_line(img, cx - ext, cy, cx + ext, cy, th, j)
    _draw_line(img, cx, cy - ext, cx, cy + ext, th, j)


def draw_amplify(img: np.ndarray):
    """Double upward triangle — always has inner triangle (distinguishes from fire)."""
    cx, cy, s = _random_transform()
    th = random.randint(2, 4)
    j = random.uniform(1.5, 3.5)
    h = random.uniform(22, 30) * s
    w = random.uniform(18, 26) * s
    top = (cx, cy - h * 0.5)
    bl = (cx - w, cy + h * 0.5)
    br = (cx + w, cy + h * 0.5)
    _draw_line(img, *bl, *top, th, j)
    _draw_line(img, *top, *br, th, j)
    _draw_line(img, *br, *bl, th, j)
    # Inner triangle — ALWAYS present (key difference from fire)
    s2 = random.uniform(0.4, 0.6)
    _draw_line(img, cx, cy - h * 0.5 * s2, cx - w * s2, cy + h * 0.5 * s2, max(1, th - 1), j)
    _draw_line(img, cx, cy - h * 0.5 * s2, cx + w * s2, cy + h * 0.5 * s2, max(1, th - 1), j)
    _draw_line(img, cx - w * s2, cy + h * 0.5 * s2, cx + w * s2, cy + h * 0.5 * s2, max(1, th - 1), j)


def draw_weaken(img: np.ndarray):
    """Small downward-pointing triangle."""
    cx, cy, s = _random_transform()
    th = random.randint(*LINE_WIDTH_RANGE)
    j = random.uniform(1.5, 3.5)
    h = random.uniform(12, 18) * s
    w = random.uniform(10, 16) * s
    bot = (cx, cy + h * 0.5)
    tl = (cx - w, cy - h * 0.5)
    tr = (cx + w, cy - h * 0.5)
    _draw_line(img, *tl, *bot, th, j)
    _draw_line(img, *bot, *tr, th, j)
    _draw_line(img, *tr, *tl, th, j)


def draw_hold(img: np.ndarray):
    """Two vertical bars ‖ (pause symbol)."""
    cx, cy, s = _random_transform()
    th = random.randint(2, 4)
    j = random.uniform(1.5, 3.0)
    h = random.uniform(18, 26) * s
    gap = random.uniform(8, 14) * s
    _draw_line(img, cx - gap, cy - h / 2, cx - gap, cy + h / 2, th, j)
    _draw_line(img, cx + gap, cy - h / 2, cx + gap, cy + h / 2, th, j)


def draw_dispel(img: np.ndarray):
    """X cross."""
    cx, cy, s = _random_transform()
    th = random.randint(*LINE_WIDTH_RANGE)
    j = random.uniform(1.5, 3.5)
    size = random.uniform(14, 22) * s
    _draw_line(img, cx - size, cy - size, cx + size, cy + size, th, j)
    _draw_line(img, cx + size, cy - size, cx - size, cy + size, th, j)


def draw_here(img: np.ndarray):
    """Dot inside small circle + crosshair lines (distinguishes from life)."""
    cx, cy, s = _random_transform()
    th = random.randint(*LINE_WIDTH_RANGE)
    j = random.uniform(1.5, 3.0)
    r = random.uniform(6, 10) * s
    _draw_circle(img, cx, cy, r, th, j, 18)
    # Small crosshair marks outside circle (distinguishes from life)
    mark = r + 4 * s
    _draw_line(img, cx - mark, cy, cx - r - 1, cy, max(1, th - 1), j)
    _draw_line(img, cx + r + 1, cy, cx + mark, cy, max(1, th - 1), j)
    # Dot
    dot_r = max(2, int(3 * s))
    for dx in range(-dot_r, dot_r + 1):
        for dy in range(-dot_r, dot_r + 1):
            if dx * dx + dy * dy <= dot_r * dot_r:
                px, py = int(cx) + dx, int(cy) + dy
                if 0 <= px < IMG_SIZE and 0 <= py < IMG_SIZE:
                    img[py, px] = 1.0


def draw_there(img: np.ndarray):
    """Dot inside dashed circle."""
    cx, cy, s = _random_transform()
    th = random.randint(*LINE_WIDTH_RANGE)
    j = random.uniform(1.5, 3.0)
    r = random.uniform(12, 18) * s
    # Dashed circle
    n_dashes = random.randint(6, 10)
    for i in range(n_dashes):
        a0 = 2 * math.pi * i / n_dashes
        a1 = a0 + math.pi / n_dashes * 0.7
        _draw_arc(img, cx, cy, r, a0, a1, th, j, 5)
    # Dot
    dot_r = max(2, int(3 * s))
    for dx in range(-dot_r, dot_r + 1):
        for dy in range(-dot_r, dot_r + 1):
            if dx * dx + dy * dy <= dot_r * dot_r:
                px, py = int(cx) + dx, int(cy) + dy
                if 0 <= px < IMG_SIZE and 0 <= py < IMG_SIZE:
                    img[py, px] = 1.0


def draw_self(img: np.ndarray):
    """Vertical line with dot/circle on top (person-like)."""
    cx, cy, s = _random_transform()
    th = random.randint(*LINE_WIDTH_RANGE)
    j = random.uniform(1.5, 3.0)
    h = random.uniform(20, 28) * s
    # Vertical line
    _draw_line(img, cx, cy - h * 0.2, cx, cy + h * 0.5, th, j)
    # Head dot/circle
    head_r = random.uniform(4, 7) * s
    _draw_circle(img, cx, cy - h * 0.2 - head_r, head_r, th, j, 14)


def draw_target(img: np.ndarray):
    """Crosshair (+) with small circle."""
    cx, cy, s = _random_transform()
    th = random.randint(*LINE_WIDTH_RANGE)
    j = random.uniform(1.5, 3.0)
    size = random.uniform(14, 22) * s
    # Cross
    _draw_line(img, cx - size, cy, cx + size, cy, th, j)
    _draw_line(img, cx, cy - size, cx, cy + size, th, j)
    # Center circle
    r = random.uniform(4, 8) * s
    _draw_circle(img, cx, cy, r, th, j, 14)


def draw_noise(img: np.ndarray):
    """Random scribbles — NOT a symbol. Lines, dots, scratches."""
    th = random.randint(*LINE_WIDTH_RANGE)
    j = random.uniform(1.0, 3.0)
    variant = random.randint(0, 4)

    if variant == 0:
        # Single straight line
        x0, y0 = random.uniform(5, 58), random.uniform(5, 58)
        x1, y1 = random.uniform(5, 58), random.uniform(5, 58)
        _draw_line(img, x0, y0, x1, y1, th, j)
    elif variant == 1:
        # Two crossing lines
        for _ in range(2):
            x0, y0 = random.uniform(5, 58), random.uniform(5, 58)
            x1, y1 = random.uniform(5, 58), random.uniform(5, 58)
            _draw_line(img, x0, y0, x1, y1, th, j)
    elif variant == 2:
        # Random zigzag
        pts = [(random.uniform(10, 54), random.uniform(10, 54)) for _ in range(random.randint(3, 6))]
        for i in range(len(pts) - 1):
            _draw_line(img, *pts[i], *pts[i + 1], th, j)
    elif variant == 3:
        # Short scratches
        for _ in range(random.randint(2, 5)):
            x0, y0 = random.uniform(10, 54), random.uniform(10, 54)
            angle = random.uniform(0, math.pi)
            length = random.uniform(5, 20)
            _draw_line(img, x0, y0, x0 + length * math.cos(angle),
                      y0 + length * math.sin(angle), th, j)
    else:
        # Curve/arc fragment
        cx, cy = random.uniform(20, 44), random.uniform(20, 44)
        r = random.uniform(10, 25)
        start = random.uniform(0, math.pi * 2)
        span = random.uniform(0.5, 1.5)
        _draw_arc(img, cx, cy, r, start, start + span, th, j, n_points=10)


DRAW_FUNCTIONS = [
    # Elements (6)
    draw_fire, draw_water, draw_earth, draw_air, draw_light, draw_dark,
    # Domains (6)
    draw_body, draw_mind, draw_matter, draw_space, draw_time, draw_life,
    # Directions (6)
    draw_inward, draw_outward, draw_up, draw_down, draw_through, draw_reverse,
    # Quantity (4)
    draw_point, draw_flow, draw_wave, draw_all,
    # Modifiers (4)
    draw_amplify, draw_weaken, draw_hold, draw_dispel,
    # Meta (4)
    draw_here, draw_there, draw_self, draw_target,
    # Noise
    draw_noise,
]


def generate_sample(label: int) -> np.ndarray:
    """Generate a single 64x64 symbol image."""
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    DRAW_FUNCTIONS[label](img)
    img = np.clip(img, 0.0, 1.0)
    img += np.random.normal(0, 0.02, img.shape).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img


def generate_dataset(n_per_class: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Generate full dataset. Returns (images, labels)."""
    images = []
    labels = []
    for label in range(NUM_SYMBOLS):
        for _ in range(n_per_class):
            img = generate_sample(label)
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)


def generate_circle(closed: bool = True, jitter: float = 2.0) -> np.ndarray:
    """Generate a circle image (for circle detection training)."""
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    cx, cy, s = _random_transform(scale_range=(0.8, 1.0))
    r = random.uniform(20, 28) * s
    th = random.randint(*LINE_WIDTH_RANGE)

    n_points = 40
    end = n_points if closed else random.randint(int(n_points * 0.5), int(n_points * 0.85))
    for i in range(end):
        a0 = 2 * math.pi * i / n_points
        a1 = 2 * math.pi * (i + 1) / n_points
        x0, y0 = cx + r * math.cos(a0), cy + r * math.sin(a0)
        x1, y1 = cx + r * math.cos(a1), cy + r * math.sin(a1)
        _draw_line(img, x0, y0, x1, y1, th, jitter)

    img = np.clip(img, 0.0, 1.0)
    return img


# ═══════════════════════════════════════════
# Stroke-based dataset (shared renderer with inference)
# ═══════════════════════════════════════════

def generate_strokes(label: int) -> list[list[tuple[float, float]]]:
    """Clean polylines (64-px space) for one random instance of a symbol."""
    rec = StrokeRecorder()
    DRAW_FUNCTIONS[label](rec)
    return rec.strokes()


def augment_strokes(strokes, rng: np.random.Generator, strength: float = 1.0,
                    irregular_sampling: bool = False):
    """Make clean polylines look hand-drawn with a mouse or pen.

    Smooth, low-frequency deformation instead of per-pixel noise: small
    rotation/shear/anisotropic scale, a sinusoidal warp field, endpoint
    over/undershoot and a correlated tremor along each stroke.
    Rotation stays small on purpose — up/down, inward/outward and
    fire/weaken differ only by orientation.
    """
    pts = [np.asarray(s, dtype=np.float32).reshape(-1, 2) for s in strokes if len(s)]
    if not pts:
        return []
    allp = np.concatenate(pts)
    center = (allp.min(0) + allp.max(0)) / 2.0
    size = float(max(np.ptp(allp, axis=0).max(), 4.0))

    rot = rng.normal(0, 0.12 * strength)
    shear = rng.normal(0, 0.10 * strength)
    sx, sy = rng.uniform(1 - 0.15 * strength, 1 + 0.15 * strength, size=2)
    c, s_ = math.cos(rot), math.sin(rot)
    A = np.array([[c, -s_], [s_, c]], dtype=np.float32) @ np.array(
        [[sx, shear], [0, sy]], dtype=np.float32)

    n_waves = 3
    amp = rng.uniform(0, 0.045 * size * strength, size=(n_waves, 2))
    freq = rng.uniform(0.5, 2.0, size=(n_waves, 2)) * (2 * math.pi / size)
    phase = rng.uniform(0, 2 * math.pi, size=(n_waves, 2))

    def warp(p: np.ndarray) -> np.ndarray:
        d = np.zeros_like(p)
        for k in range(n_waves):
            d[:, 0] += amp[k, 0] * np.sin(freq[k, 0] * p[:, 1] + phase[k, 0])
            d[:, 1] += amp[k, 1] * np.sin(freq[k, 1] * p[:, 0] + phase[k, 1])
        return d

    out = []
    for p in pts:
        if len(p) == 1:
            q = (p - center) @ A.T + center
            out.append([tuple(map(float, q[0]))])
            continue
        # over/undershoot at both ends
        ext = rng.normal(0, 0.04 * strength, size=2)
        p = p.copy()
        p[0] = p[0] + (p[0] - p[1]) * ext[0]
        p[-1] = p[-1] + (p[-1] - p[-2]) * ext[1]
        # resample along the stroke so tremor is smooth along arc length
        step = size / (rng.uniform(12, 40) if irregular_sampling else 30.0)
        dense = [p[0]]
        for a, b in zip(p[:-1], p[1:]):
            n = max(int(np.linalg.norm(b - a) / max(step, 0.5)), 1)
            for t in range(1, n + 1):
                dense.append(a + (b - a) * (t / n))
        q = np.asarray(dense, dtype=np.float32)
        noise = rng.normal(0, 0.012 * size * strength, size=q.shape).astype(np.float32)
        if len(noise) >= 5:
            kernel = np.ones(5, dtype=np.float32) / 5.0
            for ax in range(2):
                noise[:, ax] = np.convolve(noise[:, ax], kernel, mode="same")
        q = q + noise
        q = (q - center) @ A.T + center
        q = q + warp(q)
        out.append([tuple(map(float, xy)) for xy in q])
    return out


def generate_stroke_sample(label: int, rng: np.random.Generator | None = None,
                           strength: float = 1.0) -> np.ndarray:
    """One training image produced by the SAME renderer the game uses."""
    from .sketch import render_strokes

    rng = rng or np.random.default_rng()
    strokes = augment_strokes(generate_strokes(label), rng, strength)
    img = render_strokes(
        strokes,
        fill=float(rng.uniform(0.72, 0.88)),
        thickness=float(rng.uniform(1.8, 3.4)),
        min_extent=10.0,
    )
    img += rng.normal(0, 0.02, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def generate_stroke_dataset(n_per_class: int = 200, seed: int | None = None,
                            strength: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Stroke-rendered dataset. Same layout as generate_dataset()."""
    rng = np.random.default_rng(seed)
    if seed is not None:
        random.seed(seed)
    images, labels = [], []
    for label in range(NUM_SYMBOLS):
        for _ in range(n_per_class):
            images.append(generate_stroke_sample(label, rng, strength))
            labels.append(label)
    return np.array(images), np.array(labels)
