"""Magic explorer — automated playtesting of the spell space.

Runs quality-diversity search (MAP-Elites) over spells and writes a balance
report: the best spell in every niche, dominant runes, dead verbs, power-per-
rune outliers, stacking curves, catalog coverage, and a 2-D atlas of the
explored space. Use it like a test suite for magic design: run it after every
change to data/*.json or the algebra and diff the reports.

    python tools/explore.py                       # 6000 iterations, report in reports/explore/
    python tools/explore.py --iterations 20000 --objective power_per_rune
    python tools/explore.py --find "damage projectile piercing stable"
    python tools/explore.py --check               # exit 1 if balance invariants break (CI)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wand.explore import (OBJECTIVES, atlas_points, beam_search, exploit_report,  # noqa: E402
                          map_elites)

# Balance invariants: what "healthy magic" means for this project. Tune freely.
INVARIANTS = {
    "max_elite_runes_median": 5,       # elites should not be giant spaghetti spells
    "min_catalog_coverage": 0.6,       # most known combinations are reachable
    "max_dead_verbs": 2,               # nearly every verb is someone's main effect
    "max_power_per_rune_z": 6.0,       # no absurd outlier
}


def shaped_query(words: list[str]):
    """Turn keywords into a dense objective for beam search."""
    def score(p):
        s = p.power * 0.5
        verbs = {op.verb for op in p.ops}
        for w in words:
            if w in verbs:
                s += 1.0 if p.primary_verb == w else 0.4
            elif w == p.delivery.kind or w == p.target.selector or w == p.shape.kind:
                s += 0.8
            elif w == "piercing":
                s += 0.8 * p.delivery.piercing
            elif w == "stable":
                s += min(p.stability / 0.85, 1.0) * 0.6
            elif w == "cheap":
                s += max(0.0, 1.0 - p.cost / 30.0) * 0.6
            elif any(f.property == w and f.delta > 0 for f in p.fields):
                s += 0.6
        return s * (1.0 - p.risk.backfire_chance)
    return score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iterations", type=int, default=6000)
    ap.add_argument("--objective", choices=list(OBJECTIVES), default="efficiency")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/explore")
    ap.add_argument("--find", help="inverse design: keywords (verbs, delivery, fields, piercing, stable, cheap)")
    ap.add_argument("--check", action="store_true", help="exit non-zero if invariants fail")
    args = ap.parse_args()

    if args.find:
        res = beam_search(shaped_query(args.find.split()), width=16, depth=6, expansions=30,
                          seed=args.seed)
        for score, ev in res[:8]:
            p = ev.program
            print(f"{score:5.2f}  {ev.genome.describe():<60}  {p.title} · "
                  f"{p.primary_verb}/{p.delivery.kind} P={p.power:.2f} S={p.stability:.2f}")
        return

    arch = map_elites(iterations=args.iterations, objective=OBJECTIVES[args.objective], seed=args.seed)
    rep = exploit_report(arch)
    sizes = sorted(len(e.genome.runes) for e in arch.elites.values())
    rep["elite_runes_median"] = sizes[len(sizes) // 2] if sizes else 0
    rep["top_elites"] = [{"score": round(s, 4), "niche": list(e.niche), "spell": e.genome.describe(),
                          "title": e.program.title} for s, e in arch.top(15)]

    failures = []
    if rep["elite_runes_median"] > INVARIANTS["max_elite_runes_median"]:
        failures.append(f"elite spells too big (median {rep['elite_runes_median']} runes)")
    if rep["catalog_coverage"] < INVARIANTS["min_catalog_coverage"]:
        failures.append(f"catalog coverage {rep['catalog_coverage']:.0%}")
    if len(rep["dead_verbs"]) > INVARIANTS["max_dead_verbs"]:
        failures.append(f"dead verbs: {rep['dead_verbs']}")
    z = max((o["z"] for o in rep["power_per_rune_outliers"]), default=0)
    if z > INVARIANTS["max_power_per_rune_z"]:
        failures.append(f"power-per-rune outlier z={z}")
    rep["invariant_failures"] = failures

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    _atlas(arch, os.path.join(args.out, "atlas.png"))

    print(f"{rep['evaluations']} spells evaluated · {rep['niches_filled']} niches · "
          f"QD {rep['qd_score']} · coverage {rep['catalog_coverage']:.0%}")
    print("dominant runes:", rep["dominant_runes"][:5])
    print("dead verbs:", rep["dead_verbs"] or "none")
    print("stacking:", rep["stacking_curves"])
    for e in rep["top_elites"][:5]:
        print(f"  {e['score']:.3f} {e['niche']} {e['spell']}")
    print("invariants:", "OK" if not failures else "; ".join(failures))
    print(f"report → {args.out}/report.json, atlas → {args.out}/atlas.png")
    if args.check and failures:
        sys.exit(1)


def _atlas(arch, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    evs = arch.seen
    if len(evs) < 3:
        return
    pts, var = atlas_points(evs)
    verbs = sorted({e.program.primary_verb for e in evs})
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, v in enumerate(verbs):
        idx = [k for k, e in enumerate(evs) if e.program.primary_verb == v]
        ax.scatter(pts[idx, 0], pts[idx, 1], s=8, color=cmap(i % 20), label=v, alpha=0.7)
    elite_ids = {id(e) for e in arch.elites.values()}
    idx = [k for k, e in enumerate(evs) if id(e) in elite_ids]
    ax.scatter(pts[idx, 0], pts[idx, 1], s=30, facecolors="none", edgecolors="black", linewidths=0.6,
               label="elite")
    ax.set_title(f"Spell atlas — {len(evs)} spells, PCA {var[0]:.0%}+{var[1]:.0%} variance")
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
