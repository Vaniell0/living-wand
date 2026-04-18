"""Synthetic symbol data generator for OCR training.

Generates 64x64 grayscale images of hand-drawn-style magic symbols.
Uses random perturbation to simulate imprecise mouse/pen drawing.
"""

import numpy as np
import math
import random

IMG_SIZE = 64
LINE_WIDTH_RANGE = (2, 4)

# Symbols we generate — 8 base primitives (visually distinct shapes)
SYMBOL_NAMES = [
    "fire",      # 0 — upward triangle (flame)
    "water",     # 1 — downward wavy lines
    "earth",     # 2 — square
    "air",       # 3 — spiral
    "life",      # 4 — circle with dot
    "mind",      # 5 — eye shape
    "space",     # 6 — diamond
    "time",      # 7 — hourglass (two triangles)
    "noise",     # 8 — random scribbles, lines, dots (NOT a symbol)
]
NUM_SYMBOLS = len(SYMBOL_NAMES)


def _jitter(x: float, y: float, amount: float = 2.0) -> tuple[float, float]:
    """Add random jitter to a point."""
    return (
        x + random.gauss(0, amount),
        y + random.gauss(0, amount),
    )


def _draw_line(img: np.ndarray, x0: float, y0: float, x1: float, y1: float,
               thickness: int = 2, jitter: float = 1.5):
    """Draw a line with hand-drawn jitter using Bresenham-like approach."""
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
    """Square / rectangle — various proportions and rotation."""
    cx, cy, s = _random_transform()
    rot = random.gauss(0, 0.3)
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
    draw_fire, draw_water, draw_earth, draw_air,
    draw_life, draw_mind, draw_space, draw_time,
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
