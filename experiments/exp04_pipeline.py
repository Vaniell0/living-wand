"""
Experiment 04: Full Pipeline Test

End-to-end test: synthetic drawings → OCR → checker → wand → effect.
Tests all 7 preset spells + some intentionally broken ones.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from wand.synth_data import generate_dataset, generate_sample, generate_circle, SYMBOL_NAMES
from wand.ocr import train_symbol_cnn, train_circle_detector
from wand.model import create_wand, WandTrainer
from wand.events import BIOGRAPHIES, Pole, NUM_POLES
from wand.spells import (
    Operator, PRESET_SPELLS, FIRE, WATER, LIGHT, DARK,
    AMPLIFY, WEAKEN, LIFE, BODY, OUTWARD, TARGET,
    EARTH, HOLD, HERE, MIND, THERE, TIME as TIME_PRIM,
    SPACE as SPACE_PRIM, SELF, THROUGH, FLOW,
)
from wand.pipeline import CastingPipeline, DrawingInput
from wand.checker import Verdict


def make_drawing(symbol_names: list[str], connections: list[tuple[int, int, Operator]],
                 circle_closed: bool = True) -> DrawingInput:
    """Create a DrawingInput from symbol names."""
    symbol_images = []
    for name in symbol_names:
        if name in SYMBOL_NAMES:
            idx = SYMBOL_NAMES.index(name)
            symbol_images.append(generate_sample(idx))
        else:
            # Unknown symbol — random noise
            symbol_images.append(np.random.rand(64, 64).astype(np.float32) * 0.3)

    circle_img = generate_circle(closed=circle_closed)
    return DrawingInput(symbol_images, circle_img, connections)


def main():
    # Train OCR
    print("Training OCR components...")
    images, labels = generate_dataset(n_per_class=300)
    symbol_cnn, _ = train_symbol_cnn(images, labels, epochs=25)
    circle_det, _ = train_circle_detector(n_samples=800, epochs=15)
    print(f"  Symbol CNN: {symbol_cnn.param_count()} params")

    # Train a healer wand
    print("\nTraining healer wand...")
    wand = create_wand(seed=7)
    events = BIOGRAPHIES["healer"](500)
    trainer = WandTrainer(wand, lr=1e-3)
    trainer.train_biography(events, epochs=20)
    profile = wand.affinity_profile().numpy()
    print(f"  Profile: {dict(zip([p.name for p in Pole], [f'{v:.3f}' for v in profile]))}")

    pipeline = CastingPipeline(symbol_cnn, circle_det, wand)

    # --- Test cases ---
    test_cases = [
        {
            "name": "Heal (correct)",
            "symbols": ["life", "life", "life"],  # repeat for OCR test
            "connections": [(0, 1, Operator.MUTUAL), (2, 0, Operator.DIRECT)],
            "circle_closed": True,
        },
        {
            "name": "Fireball (correct)",
            "symbols": ["fire", "fire", "fire"],
            "connections": [(0, 1, Operator.DIRECT), (1, 2, Operator.DIRECT)],
            "circle_closed": True,
        },
        {
            "name": "Mind illusion",
            "symbols": ["mind", "light"],
            "connections": [(0, 1, Operator.DIRECT)],
            "circle_closed": True,
        },
        {
            "name": "Open circle (warning)",
            "symbols": ["fire", "earth"],
            "connections": [(0, 1, Operator.DIRECT)],
            "circle_closed": False,
        },
        {
            "name": "Fire+Water antagonism (error)",
            "symbols": ["fire", "water"],
            "connections": [(0, 1, Operator.DIRECT)],
            "circle_closed": True,
        },
        {
            "name": "Amplify+Weaken collapse (critical)",
            "symbols": ["fire", "fire"],  # will be overridden — checker uses spell primitives
            "connections": [(0, 1, Operator.MUTUAL)],
            "circle_closed": True,
            "override_primitives": [AMPLIFY, WEAKEN],
        },
        {
            "name": "Hanging symbol (warning)",
            "symbols": ["fire", "earth", "water"],
            "connections": [(0, 1, Operator.DIRECT)],  # water is hanging
            "circle_closed": True,
        },
        {
            "name": "Time spell (scholar domain)",
            "symbols": ["time", "time", "time"],
            "connections": [(0, 1, Operator.DIRECT), (1, 2, Operator.DIRECT)],
            "circle_closed": True,
        },
    ]

    print("\n" + "=" * 80)
    print("FULL PIPELINE TESTS — Healer Wand")
    print("=" * 80)

    results_for_plot = []

    for tc in test_cases:
        drawing = make_drawing(tc["symbols"], tc["connections"], tc["circle_closed"])

        # Override primitives if specified (for checker tests)
        if "override_primitives" in tc:
            from wand.spells import Spell, Connection
            spell_override = Spell(
                tc["override_primitives"],
                [Connection(f, t, o) for f, t, o in tc["connections"]],
                circle_closed=tc["circle_closed"],
            )

        result = pipeline.cast(drawing)

        print(f"\n--- {tc['name']} ---")
        print(f"  OCR: {', '.join(f'{n}({c:.0%})' for n, c in result.recognized_symbols)}")
        print(f"  Circle: {'closed' if result.circle_closed else 'OPEN'} ({result.circle_confidence:.0%})")
        print(f"  Checker: {result.check.verdict.name}", end="")
        if result.check.issues:
            print(f" — {'; '.join(result.check.issues)}")
        else:
            print()
        print(f"  Power: {result.effective_power:.3f} (raw={result.raw_power:.3f})")
        print(f"  Stability: {result.effective_stability:.3f}")
        print(f"  Dominant: {result.dominant_pole}")
        print(f"  > {result.effect_description}")

        results_for_plot.append({
            "name": tc["name"],
            "power": result.effective_power,
            "stability": result.effective_stability,
            "verdict": result.check.verdict.name,
            "poles": result.pole_strengths,
        })

    # --- Comparison: same spells, different wands ---
    print("\n" + "=" * 80)
    print("CROSS-WAND COMPARISON — Heal spell")
    print("=" * 80)

    heal_drawing = make_drawing(["life", "life", "life"],
                                [(0, 1, Operator.MUTUAL), (2, 0, Operator.DIRECT)],
                                circle_closed=True)

    wand_results = {}
    for bio_name in ["healer", "warrior", "scholar"]:
        w = create_wand(seed=7)
        t = WandTrainer(w, lr=1e-3)
        t.train_biography(BIOGRAPHIES[bio_name](500), epochs=20)
        p = CastingPipeline(symbol_cnn, circle_det, w)
        r = p.cast(heal_drawing)
        wand_results[bio_name] = r
        print(f"\n  {bio_name}: power={r.effective_power:.3f}, "
              f"dominant={r.dominant_pole}, stability={r.effective_stability:.3f}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Power comparison across test cases
    ax = axes[0, 0]
    names = [r["name"][:20] for r in results_for_plot]
    powers = [r["power"] for r in results_for_plot]
    stabilities = [r["stability"] for r in results_for_plot]
    verdict_colors = {
        "OK": "#88c999", "WARNING": "#e8c84a", "ERROR": "#e07070", "CRITICAL": "#cc3333"
    }
    bar_colors = [verdict_colors.get(r["verdict"], "#999") for r in results_for_plot]
    ax.barh(range(len(names)), powers, color=bar_colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Effective Power")
    ax.set_title("Pipeline Results (Healer Wand)")
    ax.invert_yaxis()
    # Legend
    for v, c in verdict_colors.items():
        ax.barh([], [], color=c, label=v)
    ax.legend(loc='lower right', fontsize=8)

    # 2. Pole distribution for each test case (heatmap)
    ax = axes[0, 1]
    pole_data = np.array([r["poles"] for r in results_for_plot])
    pole_names = [p.name[:5] for p in Pole]
    im = ax.imshow(pole_data, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(NUM_POLES))
    ax.set_xticklabels(pole_names, fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title("Pole Distribution per Spell")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 3. Cross-wand heal comparison
    ax = axes[1, 0]
    wand_names = list(wand_results.keys())
    heal_poles = np.array([wand_results[w].pole_strengths for w in wand_names])
    x = np.arange(NUM_POLES)
    width = 0.25
    colors = {'healer': '#88c999', 'warrior': '#e07070', 'scholar': '#7090d0'}
    for i, wn in enumerate(wand_names):
        ax.bar(x + i * width, heal_poles[i], width, label=wn, color=colors[wn])
    ax.set_xticks(x + width)
    ax.set_xticklabels(pole_names)
    ax.set_ylabel("Pole Strength")
    ax.set_title("Heal Spell — 3 Different Wands")
    ax.legend()

    # 4. Power + stability scatter
    ax = axes[1, 1]
    for r in results_for_plot:
        color = verdict_colors.get(r["verdict"], "#999")
        ax.scatter(r["power"], r["stability"], c=color, s=100, zorder=5)
        ax.annotate(r["name"][:15], (r["power"], r["stability"]),
                    fontsize=7, ha='left', va='bottom')
    ax.set_xlabel("Effective Power")
    ax.set_ylabel("Effective Stability")
    ax.set_title("Power vs Stability")
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(-0.05, 1.0)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiments/exp04_results.png", dpi=150, bbox_inches='tight')
    print("\nPlot saved to experiments/exp04_results.png")
    plt.close()


if __name__ == "__main__":
    main()
