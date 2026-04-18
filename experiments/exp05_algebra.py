"""
Experiment 05: Spell Algebra

Test the compositional spell evaluation system.
Show how different combinations produce emergent effects.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from wand.algebra import (
    ALGEBRA_PRESETS, evaluate_spell, SpellGraph, SpellEffect,
    SignalKind, Element, EdgeType, SYNERGY_PAIRS, CANCEL_PAIRS,
)


def main():
    print("=" * 80)
    print("SPELL ALGEBRA — Compositional Evaluation")
    print("=" * 80)

    results = {}
    for name, builder in ALGEBRA_PRESETS.items():
        graph = builder()
        effect = evaluate_spell(graph)
        results[name] = effect

        print(f"\n{'─'*60}")
        print(f"  {name.upper()}")
        print(f"  Nodes: {', '.join(n.name for n in graph.nodes.values())}")
        print(f"  Edges: {', '.join(f'{graph.nodes[e.from_id].name}→{graph.nodes[e.to_id].name}({e.edge_type.name})' for e in graph.edges)}")
        print(f"  {effect.describe()}")

    # --- Comparison experiments ---
    print(f"\n{'='*80}")
    print("COMBINATION EXPERIMENTS")
    print(f"{'='*80}")

    # Experiment A: same element reinforcement
    print("\n--- Reinforcement: Fire ═══ Fire ---")
    g = SpellGraph()
    f1 = g.add_node("fire", SignalKind.FORCE, Element.FIRE)
    f2 = g.add_node("fire2", SignalKind.FORCE, Element.FIRE)
    g.nodes[f2].element = Element.FIRE  # same element
    g.connect(f1, f2, EdgeType.MUTUAL)
    e = evaluate_spell(g)
    print(f"  {e.describe()}")

    # Experiment B: triple fire
    print("\n--- Triple Fire: Fire ═══ Fire ═══ Fire ---")
    g = SpellGraph()
    f1 = g.add_node("fire1", SignalKind.FORCE, Element.FIRE)
    f2 = g.add_node("fire2", SignalKind.FORCE, Element.FIRE)
    f3 = g.add_node("fire3", SignalKind.FORCE, Element.FIRE)
    g.nodes[f1].element = Element.FIRE
    g.nodes[f2].element = Element.FIRE
    g.nodes[f3].element = Element.FIRE
    g.connect(f1, f2, EdgeType.MUTUAL)
    g.connect(f2, f3, EdgeType.MUTUAL)
    e = evaluate_spell(g)
    print(f"  {e.describe()}")

    # Experiment C: modifier stacking
    print("\n--- Amplify chain: Amp → Amp → Fire ---")
    g = SpellGraph()
    a1 = g.add_node("amplify1", SignalKind.MODIFIER)
    a2 = g.add_node("amplify2", SignalKind.MODIFIER)
    fire = g.add_node("fire", SignalKind.FORCE, Element.FIRE)
    g.connect(a1, a2, EdgeType.DIRECT)
    g.connect(a2, fire, EdgeType.DIRECT)
    e = evaluate_spell(g)
    print(f"  {e.describe()}")

    # Experiment D: open circle penalty
    print("\n--- Open Circle: Fireball with broken circle ---")
    g = build_with_open_circle()
    e = evaluate_spell(g)
    print(f"  {e.describe()}")

    # Experiment E: all synergy pairs
    print(f"\n{'─'*60}")
    print("ALL SYNERGY PAIRS:")
    for pair, (name, mult) in sorted(SYNERGY_PAIRS.items(), key=lambda x: x[1][1], reverse=True):
        elems = list(pair)
        g = SpellGraph()
        n1 = g.add_node(elems[0].name, SignalKind.FORCE, elems[0])
        n2 = g.add_node(elems[1].name, SignalKind.FORCE, elems[1])
        g.connect(n1, n2, EdgeType.MUTUAL)
        e = evaluate_spell(g)
        print(f"  {elems[0].name:>6s} ═══ {elems[1].name:<6s} → {name:<8s} (×{mult}) "
              f"power={e.total_power:.2f}")

    print(f"\nALL CANCELLATION PAIRS:")
    for pair, factor in sorted(CANCEL_PAIRS.items(), key=lambda x: x[1]):
        elems = list(pair)
        g = SpellGraph()
        n1 = g.add_node(elems[0].name, SignalKind.FORCE, elems[0])
        n2 = g.add_node(elems[1].name, SignalKind.FORCE, elems[1])
        g.connect(n1, n2, EdgeType.MUTUAL)
        e = evaluate_spell(g)
        print(f"  {elems[0].name:>6s} ═══ {elems[1].name:<6s} → cancel (×{factor}) "
              f"power={e.total_power:.2f} stability={e.stability:.2f}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Power comparison
    ax = axes[0, 0]
    spell_names = list(results.keys())
    powers = [results[n].total_power for n in spell_names]
    stabilities = [results[n].stability for n in spell_names]
    colors = ['#E85D3A' if r.force_power else '#4CAF50' if r.domain_power else '#999'
              for r in results.values()]
    ax.barh(range(len(spell_names)), powers, color=colors, alpha=0.8)
    ax.set_yticks(range(len(spell_names)))
    ax.set_yticklabels(spell_names, fontsize=9)
    ax.set_xlabel("Total Power")
    ax.set_title("Spell Power (algebraic evaluation)")
    ax.invert_yaxis()

    # 2. Synergy / cancellation map
    ax = axes[0, 1]
    elements = list(Element)
    n_elem = len(elements)
    interaction_matrix = np.zeros((n_elem, n_elem))
    for i, e1 in enumerate(elements):
        for j, e2 in enumerate(elements):
            pair = frozenset({e1, e2})
            if i == j:
                interaction_matrix[i, j] = 1.8  # self-reinforcement
            elif pair in SYNERGY_PAIRS:
                interaction_matrix[i, j] = SYNERGY_PAIRS[pair][1]
            elif pair in CANCEL_PAIRS:
                interaction_matrix[i, j] = -CANCEL_PAIRS[pair]
            else:
                interaction_matrix[i, j] = 0
    im = ax.imshow(interaction_matrix, cmap='RdYlGn', vmin=-0.5, vmax=2.0)
    ax.set_xticks(range(n_elem))
    ax.set_xticklabels([e.name for e in elements], rotation=45, fontsize=9)
    ax.set_yticks(range(n_elem))
    ax.set_yticklabels([e.name for e in elements], fontsize=9)
    ax.set_title("Element Interaction Matrix")
    for i in range(n_elem):
        for j in range(n_elem):
            v = interaction_matrix[i, j]
            if v != 0:
                ax.text(j, i, f"{v:.1f}", ha='center', va='center', fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 3. Stability comparison
    ax = axes[1, 0]
    ax.scatter(powers, stabilities, s=100, c=colors, zorder=5)
    for i, name in enumerate(spell_names):
        ax.annotate(name, (powers[i], stabilities[i]),
                    fontsize=8, ha='left', va='bottom')
    ax.set_xlabel("Total Power")
    ax.set_ylabel("Stability")
    ax.set_title("Power vs Stability")
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)

    # 4. Tags/properties per spell
    ax = axes[1, 1]
    all_tags = sorted(set().union(*(r.tags for r in results.values())))
    tag_matrix = np.zeros((len(spell_names), len(all_tags)))
    for i, name in enumerate(spell_names):
        for j, tag in enumerate(all_tags):
            if tag in results[name].tags:
                tag_matrix[i, j] = 1.0
    if all_tags:
        im = ax.imshow(tag_matrix, aspect='auto', cmap='Blues')
        ax.set_xticks(range(len(all_tags)))
        ax.set_xticklabels(all_tags, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(len(spell_names)))
        ax.set_yticklabels(spell_names, fontsize=9)
        ax.set_title("Emergent Tags per Spell")
    else:
        ax.text(0.5, 0.5, "No tags", ha='center', va='center')
        ax.set_title("Emergent Tags per Spell")

    plt.tight_layout()
    plt.savefig("experiments/exp05_results.png", dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to experiments/exp05_results.png")
    plt.close()


def build_with_open_circle() -> SpellGraph:
    g = SpellGraph(circle_closed=False)
    fire = g.add_node("fire", SignalKind.FORCE, Element.FIRE)
    out = g.add_node("outward", SignalKind.VECTOR)
    tgt = g.add_node("target", SignalKind.REFERENCE)
    g.connect(fire, out, EdgeType.DIRECT)
    g.connect(out, tgt, EdgeType.DIRECT)
    return g


if __name__ == "__main__":
    main()
