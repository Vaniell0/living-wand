"""Sketch → spell graph: the drawing half of the pipeline, UI-agnostic.

Everything between "the player moved the mouse" and "here is a SpellGraph"
lives here, as plain numpy, so it can be tested without pygame and reused by
any frontend (pygame prototype, engine plugin over the HTTP API, tablet app).

    strokes ──► split dots / connector candidates / glyph strokes
            ──► cluster glyph strokes into runes (gap + drawing-order aware)
            ──► connectors whose ends land on two runes become edges
            ──► render each rune with render_strokes()   ◄── SAME function
            ──► recognizer → top-k, uncertainty flag          builds the
            ──► SpellGraph (drawn edges, else grammar wiring)  training set

The single shared renderer is the main robustness fix: the old prototype
trained SymbolNet on per-pixel-jittered bitmaps but fed it clean 3×3-stamped
screen strokes normalised to 80% of the frame — two different distributions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .algebra import RUNES, EdgeType, SignalKind, SpellGraph

IMG_SIZE = 64

Point = tuple[float, float]
Recognizer = Callable[[np.ndarray], list[tuple[str, float]]]


# ═══════════════════════════════════════════
# Rendering (shared by training data and inference)
# ═══════════════════════════════════════════

def render_strokes(
    strokes: Sequence[Sequence[Point]],
    size: int = IMG_SIZE,
    fill: float = 0.8,
    thickness: float = 2.6,
    min_extent: float = 0.0,
) -> np.ndarray:
    """Rasterise polylines into a size×size float image in [0, 1].

    The glyph's bounding box is scaled to `fill` of the frame and centred, so
    absolute drawing size and position never reach the classifier. A stroke
    with a single point is a dot (rendered as a disc). `min_extent` (source
    units) caps the zoom so a lone click does not explode into a blob.
    Lines are anti-aliased via a distance field, so thickness is sub-pixel.
    """
    img = np.zeros((size, size), dtype=np.float32)
    pts = [np.asarray(s, dtype=np.float32).reshape(-1, 2) for s in strokes if len(s) > 0]
    if not pts:
        return img

    allp = np.concatenate(pts)
    lo, hi = allp.min(0), allp.max(0)
    extent = float(max(hi[0] - lo[0], hi[1] - lo[1], min_extent, 1e-6))
    scale = fill * size / extent
    center = (lo + hi) / 2.0
    offset = np.array([size / 2.0, size / 2.0], dtype=np.float32)

    seg_a, seg_b, dots = [], [], []
    for p in pts:
        q = (p - center) * scale + offset
        if len(q) == 1 or float(np.ptp(q, axis=0).max()) < 0.5:
            dots.append(q.mean(0))
        else:
            seg_a.append(q[:-1])
            seg_b.append(q[1:])

    half = thickness / 2.0
    flat = np.zeros(size * size, dtype=np.float32)
    if seg_a:
        _stamp_segments(flat, np.concatenate(seg_a), np.concatenate(seg_b), half, size)
    if dots:
        c = np.asarray(dots, dtype=np.float32)
        _stamp_segments(flat, c, c, half + thickness * 0.9, size)
    return flat.reshape(size, size)


def _stamp_segments(flat: np.ndarray, a: np.ndarray, b: np.ndarray,
                    half: float, size: int):
    """Anti-aliased capsule stamping, fully vectorised.

    Segments are first split into ≤1.5 px pieces so a fixed window around
    each piece's midpoint covers it; then every window pixel gets
    clip(half + 0.5 - distance, 0, 1), merged with max.
    """
    ab = b - a
    n = np.maximum(np.ceil(np.sqrt((ab ** 2).sum(1)) / 1.5), 1).astype(np.int64)
    idx = np.repeat(np.arange(len(a)), n)
    k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    t0 = (k / n[idx])[:, None].astype(np.float32)
    t1 = ((k + 1) / n[idx])[:, None].astype(np.float32)
    pa = a[idx] + ab[idx] * t0
    pb = a[idx] + ab[idx] * t1

    r = int(math.ceil(half + 2.0))
    off = np.arange(-r, r + 1)
    oy, ox = np.meshgrid(off, off, indexing="ij")
    ox, oy = ox.ravel(), oy.ravel()
    mid = np.floor((pa + pb) / 2.0).astype(np.int64)
    px = mid[:, 0:1] + ox[None, :]                       # (N, W)
    py = mid[:, 1:2] + oy[None, :]
    inside = (px >= 0) & (px < size) & (py >= 0) & (py < size)

    cx = px.astype(np.float32) + 0.5
    cy = py.astype(np.float32) + 0.5
    d = pb - pa
    denom = np.maximum((d ** 2).sum(1), 1e-6)[:, None]
    t = np.clip(((cx - pa[:, 0:1]) * d[:, 0:1] + (cy - pa[:, 1:2]) * d[:, 1:2]) / denom, 0, 1)
    qx = pa[:, 0:1] + t * d[:, 0:1]
    qy = pa[:, 1:2] + t * d[:, 1:2]
    dist = np.sqrt((cx - qx) ** 2 + (cy - qy) ** 2)
    val = np.clip(half + 0.5 - dist, 0.0, 1.0)

    keep = inside & (val > 0)
    np.maximum.at(flat, (py[keep] * size + px[keep]), val[keep].astype(np.float32))


# ═══════════════════════════════════════════
# Stroke geometry
# ═══════════════════════════════════════════

def _arr(stroke: Sequence[Point]) -> np.ndarray:
    return np.asarray(stroke, dtype=np.float32).reshape(-1, 2)


def path_length(stroke: Sequence[Point]) -> float:
    p = _arr(stroke)
    if len(p) < 2:
        return 0.0
    return float(np.sqrt((np.diff(p, axis=0) ** 2).sum(1)).sum())


def straightness(stroke: Sequence[Point]) -> float:
    """Endpoint distance / path length (1.0 = perfectly straight)."""
    p = _arr(stroke)
    L = path_length(stroke)
    if L < 1e-6:
        return 0.0
    return float(np.linalg.norm(p[-1] - p[0]) / L)


def bbox(points: np.ndarray) -> tuple[float, float, float, float]:
    lo, hi = points.min(0), points.max(0)
    return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


def _resample(stroke: Sequence[Point], step: float = 4.0) -> np.ndarray:
    p = _arr(stroke)
    if len(p) < 2:
        return p
    out = [p[0]]
    for a, b in zip(p[:-1], p[1:]):
        n = int(np.linalg.norm(b - a) // step)
        for k in range(1, n + 1):
            out.append(a + (b - a) * (k / (n + 1)))
        out.append(b)
    return np.asarray(out, dtype=np.float32)


def _min_dist(a: np.ndarray, b: np.ndarray) -> float:
    d = a[:, None, :] - b[None, :, :]
    return float(np.sqrt((d ** 2).sum(2)).min())


# ═══════════════════════════════════════════
# Result types
# ═══════════════════════════════════════════

@dataclass
class Glyph:
    """One rune as drawn: strokes, geometry, recognition result."""
    stroke_ids: list[int]
    bbox: tuple[float, float, float, float]
    image: np.ndarray | None = None
    label: str = "?"
    confidence: float = 0.0
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    uncertain: bool = False
    confirmed: bool = False     # the player picked the label explicitly
    node_id: int | None = None  # id in the produced SpellGraph

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0

    @property
    def size(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return max(x1 - x0, y1 - y0, 1.0)

    @property
    def is_rune(self) -> bool:
        return self.label in RUNES


@dataclass
class DrawnEdge:
    """A connector the player drew between two glyphs."""
    stroke_ids: list[int]
    src: int            # glyph index
    dst: int            # glyph index
    edge_type: EdgeType
    inferred: bool = False  # True when produced by auto-wiring, not drawn


@dataclass
class SketchResult:
    glyphs: list[Glyph]
    edges: list[DrawnEdge]
    graph: SpellGraph
    clarity: float                       # mean confidence of accepted runes
    uncertain: list[int]                 # glyph indices needing attention
    ignored: list[int]                   # glyphs read as noise / below threshold

    def summary(self) -> str:
        runes = [f"{g.label}{'?' if g.uncertain else ''}" for g in self.glyphs if g.is_rune]
        return f"{' '.join(runes)} | edges={len(self.edges)} clarity={self.clarity:.2f}"


# ═══════════════════════════════════════════
# Parser
# ═══════════════════════════════════════════

@dataclass
class SketchConfig:
    """Tunables in SCREEN pixels. Exposed so a frontend can calibrate them."""
    dot_max_extent: float = 8.0        # a stroke smaller than this is a dot
    gap_abs: float = 14.0              # strokes closer than this always merge
    gap_rel: float = 0.25              # … or closer than gap_rel × stroke size
    gap_rel_consecutive: float = 0.6   # looser for strokes drawn back-to-back
    overlap_merge: float = 0.3         # bbox overlap / smaller area
    connector_min_len: float = 40.0
    connector_straightness: float = 0.88
    connector_snap: float = 28.0       # endpoint → glyph bbox distance
    accept_conf: float = 0.35          # below this a glyph is ignored
    uncertain_conf: float = 0.6        # below this (or small margin) → flagged
    uncertain_margin: float = 0.2
    render_fill: float = 0.8
    render_thickness: float = 2.6
    render_min_extent: float = 30.0    # px; keeps a click a dot, not a blob
    mutual_rel_dist: float = 2.0       # auto-wiring: close FORCE/DOMAIN mix


class SketchParser:
    """Turns raw strokes into glyphs, edges and a SpellGraph."""

    def __init__(self, recognizer: Recognizer | None = None,
                 config: SketchConfig | None = None):
        self.recognizer = recognizer
        self.cfg = config or SketchConfig()

    # ── public API ──

    def parse(
        self,
        strokes: Sequence[Sequence[Point]],
        circle_closed: bool = True,
        overrides: dict | None = None,
        edge_overrides: dict | None = None,
        auto_wire: bool = True,
        leniency: Callable[[str], float] | None = None,
    ) -> SketchResult:
        """Full parse.

        overrides: label chosen by the player (trusted), keyed by glyph index
            or by frozenset of the glyph's stroke ids (stable while drawing).
        edge_overrides: edge type set by the player, keyed by (src, dst) glyph
            indices or by (frozenset, frozenset) of stroke ids.
        leniency: rune → [0, 1] familiarity; familiar runes are accepted and
            trusted at lower confidence (the wand knows your handwriting).
        """
        glyph_sets, connectors = self.segment(strokes)
        glyphs = [self._make_glyph(ids, strokes) for ids in glyph_sets]

        for i, g in enumerate(glyphs):
            self._recognize(g, leniency)
            if overrides:
                label = overrides.get(i, overrides.get(frozenset(g.stroke_ids)))
                if label is not None:
                    g.label, g.confidence = label, 1.0
                    g.uncertain, g.confirmed = False, True

        def edge_override(e: DrawnEdge) -> EdgeType | None:
            if not edge_overrides:
                return None
            key_sets = (frozenset(glyphs[e.src].stroke_ids), frozenset(glyphs[e.dst].stroke_ids))
            return edge_overrides.get((e.src, e.dst), edge_overrides.get(key_sets))

        edges = self._connector_edges(connectors, strokes, glyphs)
        for e in edges:
            e.edge_type = edge_override(e) or e.edge_type

        accepted = [i for i, g in enumerate(glyphs) if g.is_rune]
        ignored = [i for i, g in enumerate(glyphs) if not g.is_rune]
        drawn = [e for e in edges if e.src in accepted and e.dst in accepted]
        if not drawn and auto_wire:
            drawn = auto_wire_edges(glyphs, accepted, self.cfg.mutual_rel_dist)
            for e in drawn:
                e.edge_type = edge_override(e) or e.edge_type

        graph = SpellGraph(circle_closed=circle_closed)
        for i in accepted:
            glyphs[i].node_id = graph.add_rune(glyphs[i].label)
        for e in drawn:
            graph.connect(glyphs[e.src].node_id, glyphs[e.dst].node_id, e.edge_type)

        confs = [glyphs[i].confidence for i in accepted]
        clarity = float(np.mean(confs)) if confs else 0.0
        uncertain = [i for i in accepted if glyphs[i].uncertain]
        return SketchResult(glyphs, drawn, graph, clarity, uncertain, ignored)

    def segment(self, strokes: Sequence[Sequence[Point]]) -> tuple[list[list[int]], list[int]]:
        """Split strokes into glyph clusters and connector stroke ids.

        A connector is long, straight, does not cross any glyph stroke (the
        crossbar of `target`, `through`, `dispel` or `all` crosses its own
        rune), and its two ends land on two different glyphs. Rejected
        candidates rejoin the glyph strokes and the check repeats until
        nothing changes.
        """
        valid = [i for i, s in enumerate(strokes) if len(s) > 0]
        candidates = [i for i in valid if self._is_connector_candidate(strokes[i])]
        glyph_ids = [i for i in valid if i not in candidates]

        crossing = [ci for ci in candidates if self._crosses(strokes[ci], glyph_ids, strokes)]
        connectors = [ci for ci in candidates if ci not in crossing]
        glyph_ids = sorted(glyph_ids + crossing)

        while True:
            clusters = self._cluster(glyph_ids, strokes)
            rejected = [ci for ci in connectors
                        if self._endpoint_glyphs(strokes[ci], clusters, strokes) is None]
            if not rejected:
                return clusters, connectors
            connectors = [ci for ci in connectors if ci not in rejected]
            glyph_ids = sorted(glyph_ids + rejected)

    def _crosses(self, stroke, glyph_ids: list[int], strokes) -> bool:
        """Does the middle of this stroke run over another (glyph) stroke?"""
        p = _resample(stroke, 3.0)
        if len(p) < 6:
            return False
        cut = max(len(p) // 6, 1)
        middle = p[cut:-cut]
        for gi in glyph_ids:
            if _min_dist(middle, _resample(strokes[gi], 3.0)) < 4.0:
                return True
        return False

    # ── internals ──

    def _is_connector_candidate(self, stroke) -> bool:
        return (path_length(stroke) >= self.cfg.connector_min_len
                and straightness(stroke) >= self.cfg.connector_straightness)

    def _stroke_size(self, stroke) -> float:
        x0, y0, x1, y1 = bbox(_arr(stroke))
        return max(x1 - x0, y1 - y0)

    def _cluster(self, ids: list[int], strokes) -> list[list[int]]:
        if not ids:
            return []
        cfg = self.cfg
        pts = {i: _resample(strokes[i]) for i in ids}
        sizes = {i: self._stroke_size(strokes[i]) for i in ids}
        boxes = {i: bbox(_arr(strokes[i])) for i in ids}
        parent = {i: i for i in ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a_pos, a in enumerate(ids):
            for b in ids[a_pos + 1:]:
                consecutive = abs(a - b) == 1
                rel = cfg.gap_rel_consecutive if consecutive else cfg.gap_rel
                gap = max(cfg.gap_abs, rel * max(sizes[a], sizes[b]))
                linked = _min_dist(pts[a], pts[b]) < gap
                if not linked:
                    # a small mark inside a bigger stroke's frame belongs to it
                    # (the dot of `life`/`here`/`there`, the head of `self`)
                    small, big = (a, b) if sizes[a] < sizes[b] else (b, a)
                    sx0, sy0, sx1, sy1 = boxes[small]
                    bx0_, by0_, bx1_, by1_ = boxes[big]
                    scx, scy = (sx0 + sx1) / 2, (sy0 + sy1) / 2
                    linked = (sizes[small] < 0.5 * sizes[big]
                              and bx0_ < scx < bx1_ and by0_ < scy < by1_)
                if not linked:
                    ax0, ay0, ax1, ay1 = boxes[a]
                    bx0, by0, bx1, by1 = boxes[b]
                    ow = min(ax1, bx1) - max(ax0, bx0)
                    oh = min(ay1, by1) - max(ay0, by0)
                    if ow > 0 and oh > 0:
                        small = min(max(ax1 - ax0, 1) * max(ay1 - ay0, 1),
                                    max(bx1 - bx0, 1) * max(by1 - by0, 1))
                        linked = ow * oh / small > cfg.overlap_merge
                if linked:
                    parent[find(b)] = find(a)

        groups: dict[int, list[int]] = {}
        for i in ids:
            groups.setdefault(find(i), []).append(i)
        # stable order: by first stroke drawn
        return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])

    def _cluster_box(self, cluster: list[int], strokes) -> tuple[float, float, float, float]:
        return bbox(np.concatenate([_arr(strokes[i]) for i in cluster]))

    def _endpoint_glyphs(self, stroke, clusters, strokes) -> tuple[int, int] | None:
        """Glyph indices under the two ends of a connector, if distinct."""
        p = _arr(stroke)
        boxes = [self._cluster_box(c, strokes) for c in clusters]

        def nearest(q) -> int | None:
            best, best_d = None, self.cfg.connector_snap
            for k, (x0, y0, x1, y1) in enumerate(boxes):
                dx = max(x0 - q[0], 0.0, q[0] - x1)
                dy = max(y0 - q[1], 0.0, q[1] - y1)
                d = math.hypot(dx, dy)
                if d <= best_d:
                    best, best_d = k, d
            return best

        a, b = nearest(p[0]), nearest(p[-1])
        if a is None or b is None or a == b:
            return None
        # the middle of a connector must run through empty space
        mid = p[len(p) // 2]
        for x0, y0, x1, y1 in (boxes[a], boxes[b]):
            if x0 < mid[0] < x1 and y0 < mid[1] < y1:
                return None
        return a, b

    def _connector_edges(self, connectors, strokes, glyphs) -> list[DrawnEdge]:
        """A→B line = DIRECT; two lines between the same pair = MUTUAL."""
        boxes_ids = [g.stroke_ids for g in glyphs]
        pairs: dict[frozenset, list[tuple[int, int, int]]] = {}
        for ci in connectors:
            ends = self._endpoint_glyphs(strokes[ci], boxes_ids, strokes)
            if ends is None:
                continue
            pairs.setdefault(frozenset(ends), []).append((ends[0], ends[1], ci))
        edges = []
        for lines in pairs.values():
            src, dst, _ = lines[0]
            etype = EdgeType.MUTUAL if len(lines) >= 2 else EdgeType.DIRECT
            edges.append(DrawnEdge([c for _, _, c in lines], src, dst, etype))
        return edges

    def _make_glyph(self, ids: list[int], strokes) -> Glyph:
        cfg = self.cfg
        parts = []
        for i in ids:
            s = _arr(strokes[i])
            if self._stroke_size(strokes[i]) < cfg.dot_max_extent:
                s = s.mean(0, keepdims=True)   # a click/tiny scribble is a dot
            parts.append(s)
        box = bbox(np.concatenate(parts))
        img = render_strokes(parts, fill=cfg.render_fill, thickness=cfg.render_thickness,
                             min_extent=cfg.render_min_extent)
        return Glyph(stroke_ids=list(ids), bbox=box, image=img)

    def _recognize(self, g: Glyph, leniency: Callable[[str], float] | None = None):
        if self.recognizer is None or g.image is None:
            return
        top = self.recognizer(g.image)
        if not top:
            return
        g.alternatives = top
        g.label, g.confidence = top[0]
        ease = leniency(g.label) if (leniency and g.label in RUNES) else 0.0
        accept = self.cfg.accept_conf * (1.0 - 0.4 * ease)
        unsure = self.cfg.uncertain_conf * (1.0 - 0.3 * ease)
        if g.confidence < accept:
            g.label = "noise"
        margin = g.confidence - (top[1][1] if len(top) > 1 else 0.0)
        g.uncertain = g.is_rune and (g.confidence < unsure
                                     or margin < self.cfg.uncertain_margin * (1.0 - 0.5 * ease))


# ═══════════════════════════════════════════
# Grammar-aware auto-wiring (when the player drew no connectors)
# ═══════════════════════════════════════════

# Sources flow into cores, cores flow into references (matches the recipes:
# point→fire, hold→fire, outward→air, fire→here, air→target).
_SOURCE_KINDS = {SignalKind.MODIFIER, SignalKind.SCALAR, SignalKind.VECTOR}
_CORE_KINDS = {SignalKind.FORCE, SignalKind.DOMAIN}


def auto_wire_edges(glyphs: list[Glyph], accepted: list[int],
                    mutual_rel_dist: float = 2.0) -> list[DrawnEdge]:
    """Infer a sensible spell structure from rune kinds and layout.

    - cores (elements/domains) are joined by a minimum spanning tree;
      close neighbours MIX (MUTUAL), distant ones DIRECT element→domain;
    - each modifier/scalar/vector feeds its nearest core (DIRECT);
    - each reference receives from its nearest core (DIRECT).
    Deterministic and far less jumpy than the old distance thresholds,
    which produced MUTUAL cliques (and accidental triple combos).
    """
    if len(accepted) < 2:
        return []
    kind = {i: RUNES[glyphs[i].label][0] for i in accepted}
    cores = [i for i in accepted if kind[i] in _CORE_KINDS]

    def dist(a: int, b: int) -> float:
        (ax, ay), (bx, by) = glyphs[a].center, glyphs[b].center
        return math.hypot(ax - bx, ay - by)

    edges: list[DrawnEdge] = []
    if not cores:
        # nothing to anchor on — chain everything by an MST, DIRECT
        for a, b in _mst(accepted, dist):
            edges.append(DrawnEdge([], a, b, EdgeType.DIRECT, inferred=True))
        return edges

    for a, b in _mst(cores, dist):
        scale = (glyphs[a].size + glyphs[b].size) / 2.0
        if dist(a, b) <= mutual_rel_dist * scale:
            edges.append(DrawnEdge([], a, b, EdgeType.MUTUAL, inferred=True))
        else:
            # element colours a domain; otherwise drawing order decides
            if kind[a] == SignalKind.DOMAIN and kind[b] == SignalKind.FORCE:
                a, b = b, a
            edges.append(DrawnEdge([], a, b, EdgeType.DIRECT, inferred=True))

    for i in accepted:
        if kind[i] in _CORE_KINDS:
            continue
        core = min(cores, key=lambda c: (dist(i, c), c))
        if kind[i] == SignalKind.REFERENCE:
            edges.append(DrawnEdge([], core, i, EdgeType.DIRECT, inferred=True))
        else:
            edges.append(DrawnEdge([], i, core, EdgeType.DIRECT, inferred=True))
    return edges


def _mst(nodes: list[int], dist: Callable[[int, int], float]) -> list[tuple[int, int]]:
    """Prim's MST; ties broken by index so the result is deterministic."""
    if len(nodes) < 2:
        return []
    in_tree = {nodes[0]}
    out: list[tuple[int, int]] = []
    while len(in_tree) < len(nodes):
        best = None
        for a in sorted(in_tree):
            for b in nodes:
                if b in in_tree:
                    continue
                cand = (dist(a, b), a, b)
                if best is None or cand < best:
                    best = cand
        _, a, b = best
        out.append((min(a, b), max(a, b)))
        in_tree.add(b)
    return out


def make_recognizer(model, k: int = 3) -> Recognizer:
    """Wrap a SymbolNet (or AdaptiveOCR) into a top-k recognizer callable."""
    import torch
    import torch.nn.functional as F
    from .synth_data import SYMBOL_NAMES

    base = getattr(model, "base", model)

    def recognize(img: np.ndarray) -> list[tuple[str, float]]:
        base.eval()
        with torch.no_grad():
            x = torch.from_numpy(img).float()[None, None]
            probs = F.softmax(base(x), dim=-1)[0]
            top = torch.topk(probs, k)
        return [(SYMBOL_NAMES[i], float(p)) for p, i in zip(top.values.tolist(), top.indices.tolist())]

    return recognize
