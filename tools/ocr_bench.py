"""OCR robustness bench — how the rune reader behaves on human-like drawing.

Builds held-out "mouse-drawn" glyphs (stronger, differently-seeded
deformation than training, irregular point sampling, screen-sized 50–160 px,
integer mouse coordinates) and measures, per pipeline:

  acc / top3      — accuracy over the 31 classes
  accepted        — share of glyphs the game would accept (conf > 0.35, not noise)
  flip            — STABILITY: share of glyphs whose label changes when the same
                    drawing is re-traced with a slight tremor (lower = steadier)
  ece             — calibration error (is "80% sure" right 80% of the time?)
  confusions      — the most common mix-ups (candidates for alphabet redesign)

Pipelines:
  legacy  — the old game path: 3×3 pixel stamping, dots dropped, bbox+5px
  strokes — wand.sketch: shared renderer, dots kept, anti-aliased

    python tools/ocr_bench.py --model data/ocr_symbolnet_v2.pt --pipeline strokes
    python tools/ocr_bench.py --model data/ocr_symbolnet.pt --pipeline legacy
    python tools/ocr_bench.py --compare legacy:old.pt strokes:new.pt --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from wand.sketch import SketchConfig, render_strokes  # noqa: E402
from wand.synth_data import (  # noqa: E402
    IMG_SIZE, NUM_SYMBOLS, SYMBOL_NAMES, augment_strokes, generate_strokes,
)


# ═══════════════════════════════════════════
# Held-out "human" drawings
# ═══════════════════════════════════════════

def to_screen(strokes, rng: np.random.Generator) -> list[list[tuple[int, int]]]:
    """Scale a 64-px glyph to 50–160 screen px with integer mouse coords."""
    pts = np.concatenate([np.asarray(s, np.float32).reshape(-1, 2) for s in strokes])
    lo, hi = pts.min(0), pts.max(0)
    extent = max(float((hi - lo).max()), 1e-3)
    target = rng.uniform(50, 160)
    scale = target / extent
    origin = np.array([300.0, 300.0]) + rng.uniform(-40, 40, 2)
    out = []
    for s in strokes:
        q = (np.asarray(s, np.float32).reshape(-1, 2) - lo) * scale + origin
        q = np.round(q).astype(int)
        # drop consecutive duplicates the way mouse events would
        keep = [tuple(q[0])]
        for p in q[1:]:
            if tuple(p) != keep[-1]:
                keep.append(tuple(p))
        out.append(keep)
    return out


def human_glyph(label: int, rng: np.random.Generator, strength: float = 1.3):
    """(base strokes in 64-space, screen strokes) for one held-out drawing."""
    base = generate_strokes(label)
    drawn = augment_strokes(base, rng, strength=strength, irregular_sampling=True)
    return base, to_screen(drawn, rng)


def retrace(base, rng: np.random.Generator, strength: float):
    """Same intent drawn again — base deformation re-sampled at `strength`."""
    return to_screen(augment_strokes(base, rng, strength=strength, irregular_sampling=True), rng)


# ═══════════════════════════════════════════
# Renderers
# ═══════════════════════════════════════════

def render_legacy(strokes) -> np.ndarray:
    """Exact replica of the pre-sketch game path (_cluster_strokes + _render_to_image)."""
    kept = []
    for s in strokes:
        if len(s) < 2:
            continue
        xs = [p[0] for p in s]
        ys = [p[1] for p in s]
        if max(xs) - min(xs) < 5 and max(ys) - min(ys) < 5:
            continue  # the old clusterer silently dropped dots
        kept.append(s)
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    if not kept:
        return img
    all_x = [p[0] for s in kept for p in s]
    all_y = [p[1] for s in kept for p in s]
    bx, by = min(all_x) - 5, min(all_y) - 5
    bw, bh = max(all_x) - min(all_x) + 10, max(all_y) - min(all_y) + 10
    cx, cy = bx + bw // 2, by + bh // 2  # pygame.Rect.centerx semantics
    scale = min(IMG_SIZE / max(bw, 1), IMG_SIZE / max(bh, 1)) * 0.8
    ox = IMG_SIZE / 2 - (cx - bx) * scale
    oy = IMG_SIZE / 2 - (cy - by) * scale
    for s in kept:
        for i in range(len(s) - 1):
            x0 = int((s[i][0] - bx) * scale + ox)
            y0 = int((s[i][1] - by) * scale + oy)
            x1 = int((s[i + 1][0] - bx) * scale + ox)
            y1 = int((s[i + 1][1] - by) * scale + oy)
            n = max(abs(x1 - x0), abs(y1 - y0), 1)
            for t in range(n + 1):
                px = int(x0 + (x1 - x0) * t / n)
                py = int(y0 + (y1 - y0) * t / n)
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if 0 <= px + dx < IMG_SIZE and 0 <= py + dy < IMG_SIZE:
                            img[py + dy, px + dx] = 1.0
    return img


def render_sketch(strokes, cfg: SketchConfig | None = None) -> np.ndarray:
    """What wand.sketch.SketchParser feeds the classifier."""
    cfg = cfg or SketchConfig()
    parts = []
    for s in strokes:
        a = np.asarray(s, np.float32).reshape(-1, 2)
        if float(np.ptp(a, axis=0).max()) < cfg.dot_max_extent:
            a = a.mean(0, keepdims=True)
        parts.append(a)
    return render_strokes(parts, fill=cfg.render_fill, thickness=cfg.render_thickness,
                          min_extent=cfg.render_min_extent)


RENDERERS = {"legacy": render_legacy, "strokes": render_sketch}


# ═══════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════

def predict_probs(model, images: np.ndarray, batch: int = 256) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            x = torch.from_numpy(images[i:i + batch]).float().unsqueeze(1)
            out.append(F.softmax(model(x), dim=-1).numpy())
    return np.concatenate(out)


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    total = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def build_cases(n_per_class: int, seed: int, strength: float = 1.3):
    random.seed(seed)
    rng = np.random.default_rng(seed)
    cases = []
    for label in range(NUM_SYMBOLS):
        for _ in range(n_per_class):
            base, screen = human_glyph(label, rng, strength)
            cases.append((label, base, screen))
    return cases


def evaluate(model, pipeline: str, cases, retrace_strength: float = 0.35, seed: int = 1) -> dict:
    render = RENDERERS[pipeline]
    labels = np.array([c[0] for c in cases])
    imgs = np.stack([render(c[2]) for c in cases])
    probs = predict_probs(model, imgs)
    pred = probs.argmax(1)
    conf = probs.max(1)
    top3 = np.argsort(-probs, axis=1)[:, :3]

    rng = np.random.default_rng(seed)
    random.seed(seed)
    imgs2 = np.stack([render(retrace(c[1], rng, retrace_strength)) for c in cases])
    pred2 = predict_probs(model, imgs2).argmax(1)

    noise = SYMBOL_NAMES.index("noise")
    correct = pred == labels
    conf_pairs = Counter(
        (SYMBOL_NAMES[t], SYMBOL_NAMES[p]) for t, p in zip(labels, pred) if t != p)
    per_class = {SYMBOL_NAMES[k]: float(correct[labels == k].mean()) for k in range(NUM_SYMBOLS)}
    return {
        "pipeline": pipeline,
        "n": int(len(labels)),
        "acc": float(correct.mean()),
        "top3": float((top3 == labels[:, None]).any(1).mean()),
        "accepted": float(((conf > 0.35) & (pred != noise)).mean()),
        "flip": float((pred != pred2).mean()),
        "ece": ece(conf, correct.astype(np.float32)),
        "mean_conf_correct": float(conf[correct].mean()) if correct.any() else 0.0,
        "mean_conf_wrong": float(conf[~correct].mean()) if (~correct).any() else 0.0,
        "confusions": [f"{t}→{p} ×{n}" for (t, p), n in conf_pairs.most_common(10)],
        "worst_classes": sorted(per_class.items(), key=lambda kv: kv[1])[:8],
    }


def load_model(path: str):
    import torch
    from wand.ocr import SymbolNet
    m = SymbolNet()
    m.load_state_dict(torch.load(path, weights_only=True))
    m.eval()
    return m


def print_report(r: dict, title: str = ""):
    print(f"\n── {title or r['pipeline']} ({r['n']} glyphs) ──")
    print(f"  acc {r['acc']:.1%}   top3 {r['top3']:.1%}   accepted {r['accepted']:.1%}")
    print(f"  flip (label changes on re-trace) {r['flip']:.1%}   ECE {r['ece']:.3f}")
    print(f"  conf when right {r['mean_conf_correct']:.2f} / when wrong {r['mean_conf_wrong']:.2f}")
    print("  worst:", ", ".join(f"{k} {v:.0%}" for k, v in r["worst_classes"]))
    print("  confusions:", "; ".join(r["confusions"][:6]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--pipeline", choices=list(RENDERERS), default="strokes")
    ap.add_argument("--compare", nargs="+", metavar="PIPELINE:MODEL",
                    help="several pipeline:model pairs on the same drawings")
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--json", help="write results to this file")
    ap.add_argument("--samples-png", help="save a grid of what each pipeline feeds the model")
    args = ap.parse_args()

    runs = []
    if args.compare:
        for item in args.compare:
            pipe, path = item.split(":", 1)
            runs.append((pipe, path))
    elif args.model:
        runs.append((args.pipeline, args.model))
    else:
        ap.error("give --model or --compare")

    cases = build_cases(args.per_class, args.seed)
    results = []
    for pipe, path in runs:
        r = evaluate(load_model(path), pipe, cases)
        r["model"] = path
        print_report(r, f"{pipe} · {os.path.basename(path)}")
        results.append(r)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
    if args.samples_png:
        _save_grid(cases, [p for p, _ in runs], args.samples_png)


def _save_grid(cases, pipelines, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    picks = [c for i, c in enumerate(cases) if i % max(len(cases) // 16, 1) == 0][:16]
    fig, ax = plt.subplots(len(pipelines), len(picks), figsize=(len(picks), 1.15 * len(pipelines)))
    ax = np.atleast_2d(ax)
    for r, pipe in enumerate(pipelines):
        for c, (label, _, screen) in enumerate(picks):
            ax[r, c].imshow(RENDERERS[pipe](screen), cmap="magma", vmin=0, vmax=1)
            ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(SYMBOL_NAMES[label], fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=70)


if __name__ == "__main__":
    main()
