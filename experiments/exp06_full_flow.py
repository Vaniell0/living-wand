"""
Experiment 06: Full Flow — How Everything Works Together

Shows the complete chain:
1. Player draws symbols → OCR recognizes them
2. Algebra evaluates the spell graph → deterministic base effect
3. Wand NN bends the effect based on biography → personalized result
4. Final effect = algebra × wand personality

This is the KEY experiment: algebra provides RULES,
wand provides PERSONALITY. Same spell, same rules,
different wands → different results.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from wand.algebra import (
    SpellGraph, SpellEffect, evaluate_spell,
    SignalKind, Element, EdgeType,
    ALGEBRA_PRESETS, build_fireball, build_heal, build_fire_storm,
    build_ice_wall, build_necromancy,
)
from wand.model import create_wand, WandTrainer, affinity_multiplier, hex_distance
from wand.events import BIOGRAPHIES, Pole, NUM_POLES
from wand.spells import SPELL_DIM


# ═══════════════════════════════════════════
# THE INTEGRATION: Algebra + Wand NN
# ═══════════════════════════════════════════

def spell_graph_to_wand_input(graph: SpellGraph) -> torch.Tensor:
    """Convert a SpellGraph into a tensor the wand NN understands.

    The wand doesn't re-evaluate the spell — it receives the
    ALGEBRAIC EVALUATION as context, plus the raw spell structure.

    Input vector layout (78 dims):
      [0:6]   — algebraic force power per element
      [6:12]  — node type histogram (how many FORCE/DOMAIN/VECTOR/etc)
      [12:19] — edge type histogram (how many DIRECT/MUTUAL/etc)
      [19:25] — algebraic pole signature (which poles are involved)
      [25]    — total algebraic power
      [26]    — algebraic stability
      [27]    — algebraic area
      [28]    — algebraic duration
      [29]    — circle closed (0/1)
      [30]    — number of nodes (normalized)
      [31]    — number of edges (normalized)
      [32:54] — emergent tag presence (22 known tags)
      [54:78] — padding to match SPELL_DIM or fixed size
    """
    DIM = max(SPELL_DIM, 78)
    vec = np.zeros(DIM, dtype=np.float32)

    # Run algebra first
    effect = evaluate_spell(graph)

    # Force power per element
    for elem in Element:
        vec[elem.value] = effect.force_power.get(elem, 0.0)

    # Node type histogram
    type_counts = np.zeros(6)
    for node in graph.nodes.values():
        idx = min(node.signal_kind.value, 5)
        type_counts[idx] += 1
    n_nodes = max(len(graph.nodes), 1)
    vec[6:12] = type_counts / n_nodes

    # Edge type histogram
    edge_counts = np.zeros(7)
    for edge in graph.edges:
        idx = min(edge.edge_type.value, 6)
        edge_counts[idx] += 1
    n_edges = max(len(graph.edges), 1)
    vec[12:19] = edge_counts / n_edges

    # Pole signature (from force + domain)
    pole_sig = np.zeros(NUM_POLES)
    for elem, power in effect.force_power.items():
        pole_sig[Pole.ELEMENT] += abs(power)
    for domain, power in effect.domain_power.items():
        domain_pole_map = {
            "body": Pole.BIOLOGY, "life": Pole.BIOLOGY,
            "mind": Pole.MENTAL, "matter": Pole.TRANSFORM,
            "space": Pole.SPACE, "time": Pole.TIME,
        }
        pole = domain_pole_map.get(domain, Pole.ELEMENT)
        pole_sig[pole] += abs(power)
    total = pole_sig.sum()
    if total > 0:
        pole_sig /= total
    vec[19:25] = pole_sig

    # Aggregate stats
    vec[25] = effect.total_power
    vec[26] = effect.stability
    vec[27] = effect.area
    vec[28] = effect.duration
    vec[29] = 1.0 if graph.circle_closed else 0.0
    vec[30] = len(graph.nodes) / 10.0
    vec[31] = len(graph.edges) / 10.0

    # Emergent tags
    known_tags = [
        'storm', 'solid', 'forge', 'flash', 'abyss', 'cave',
        'reinforced', 'amplified', 'weakened',
        'sustained', 'focused', 'negated', 'sequenced',
        'conditional', 'directed', 'multi-domain',
        'infused', 'ranged', 'local', 'self-cast',
        'domain-focused', 'elemental',
    ]
    for i, tag in enumerate(known_tags):
        if tag in effect.tags:
            vec[32 + i] = 1.0

    return torch.tensor(vec[:SPELL_DIM], dtype=torch.float32)


def cast_with_algebra(wand, graph: SpellGraph) -> dict:
    """Full cast: algebra evaluates, wand bends the result."""
    # Step 1: Algebra — deterministic base
    algebra_effect = evaluate_spell(graph)

    # Step 2: Wand NN — personality modifier
    wand.eval()
    with torch.no_grad():
        wand_input = spell_graph_to_wand_input(graph).unsqueeze(0)
        result = wand(wand_input)
        wand_poles = result["pole_strengths"][0].numpy()
        wand_power = result["power"][0, 0].item()
        wand_stability = result["stability"][0, 0].item()

    # Step 3: Combine — algebra provides base, wand provides modifier
    # Wand's affinity profile determines how much it boosts/reduces each pole
    profile = wand.affinity_profile().numpy()
    wand_dominant = int(np.argmax(profile))

    # Calculate pole-weighted power
    # Each element's power is scaled by wand's affinity to that pole
    final_force = {}
    for elem, base_power in algebra_effect.force_power.items():
        elem_pole = Pole.ELEMENT  # all elements are in ELEMENT pole
        aff = affinity_multiplier(wand_dominant, elem_pole)
        # Wand also redistributes — its personality biases the output
        final_force[elem] = base_power * (0.5 + 0.5 * aff) * (0.7 + 0.3 * wand_power)

    final_domain = {}
    domain_pole_map = {
        "body": Pole.BIOLOGY, "life": Pole.BIOLOGY,
        "mind": Pole.MENTAL, "matter": Pole.TRANSFORM,
        "space": Pole.SPACE, "time": Pole.TIME,
    }
    for domain, base_power in algebra_effect.domain_power.items():
        pole = domain_pole_map.get(domain, Pole.ELEMENT)
        aff = affinity_multiplier(wand_dominant, pole)
        final_domain[domain] = base_power * (0.5 + 0.5 * aff) * (0.7 + 0.3 * wand_power)

    total_power = sum(max(0, v) for v in final_force.values()) + \
                  sum(max(0, v) for v in final_domain.values())
    total_power *= algebra_effect.stability * (0.7 + 0.3 * wand_stability)

    return {
        "algebra": algebra_effect,
        "wand_poles": wand_poles,
        "wand_power": wand_power,
        "wand_stability": wand_stability,
        "wand_profile": profile,
        "final_force": final_force,
        "final_domain": final_domain,
        "total_power": total_power,
        "tags": algebra_effect.tags,
        "synergies": algebra_effect.synergies,
        "cancellations": algebra_effect.cancellations,
    }


def main():
    # Train 3 wands
    print("Training 3 wands on different biographies...")
    wands = {}
    for name, gen_fn in BIOGRAPHIES.items():
        wand = create_wand(seed=7)
        trainer = WandTrainer(wand, lr=1e-3)
        events = gen_fn(500)
        trainer.train_biography(events, epochs=20)
        wands[name] = wand
        profile = wand.affinity_profile().numpy()
        dom = Pole(np.argmax(profile)).name
        print(f"  {name}: dominant={dom} ({profile[np.argmax(profile)]:.3f})")

    # Test spells
    test_spells = {
        "Fireball": build_fireball(),
        "Heal": build_heal(),
        "Fire Storm": build_fire_storm(),
        "Ice Wall": build_ice_wall(),
        "Necromancy": build_necromancy(),
    }

    # Cast each spell through each wand
    print("\n" + "=" * 90)
    print("HOW THE SYSTEM WORKS — Step by Step")
    print("=" * 90)

    # Detailed walkthrough for ONE spell
    spell_name = "Fire Storm"
    graph = test_spells[spell_name]
    print(f"\n{'━'*90}")
    print(f"  DETAILED: {spell_name}")
    print(f"{'━'*90}")

    print(f"\n  STEP 1: Player draws symbols")
    print(f"  → Symbols: {', '.join(n.name for n in graph.nodes.values())}")
    print(f"  → Connections: {', '.join(f'{graph.nodes[e.from_id].name} {e.edge_type.name} {graph.nodes[e.to_id].name}' for e in graph.edges)}")

    algebra_effect = evaluate_spell(graph)
    print(f"\n  STEP 2: Algebra evaluates (deterministic rules)")
    print(f"  → Synergies found: {algebra_effect.synergies}")
    print(f"  → Force: {dict((e.name, f'{p:.2f}') for e, p in algebra_effect.force_power.items())}")
    print(f"  → Base power: {algebra_effect.total_power:.2f}")
    print(f"  → Stability: {algebra_effect.stability:.2f}")
    print(f"  → Tags: {algebra_effect.tags}")

    print(f"\n  STEP 3: Three different wands interpret the SAME spell")
    for wand_name, wand in wands.items():
        result = cast_with_algebra(wand, graph)
        profile = result["wand_profile"]
        dom = Pole(np.argmax(profile)).name
        print(f"\n  [{wand_name.upper()} WAND — dominant: {dom}]")
        print(f"    Wand modifier: power={result['wand_power']:.3f}, stability={result['wand_stability']:.3f}")
        print(f"    Final force: {dict((e.name, f'{p:.2f}') for e, p in result['final_force'].items())}")
        print(f"    TOTAL POWER: {result['total_power']:.2f}")

    # Full comparison table
    print(f"\n{'='*90}")
    print("FULL COMPARISON: 5 spells × 3 wands")
    print(f"{'='*90}")

    all_results = {}
    header = f"{'Spell':<15}"
    for wn in wands:
        header += f" | {wn:>10} pwr"
    print(header)
    print("-" * len(header))

    for spell_name, graph in test_spells.items():
        row = f"{spell_name:<15}"
        all_results[spell_name] = {}
        for wand_name, wand in wands.items():
            result = cast_with_algebra(wand, graph)
            all_results[spell_name][wand_name] = result
            row += f" | {result['total_power']:>10.2f}   "
        # Mark the best wand
        best = max(wands.keys(), key=lambda w: all_results[spell_name][w]["total_power"])
        row += f"  ← {best}"
        print(row)

    # --- Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    wand_names = list(wands.keys())
    colors = {'healer': '#88c999', 'warrior': '#e07070', 'scholar': '#7090d0'}
    spell_names = list(test_spells.keys())

    # 1. Architecture diagram (text)
    ax = axes[0, 0]
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    steps = [
        (5, 9.0, "Player draws symbols", "#FFE082", 12),
        (5, 7.5, "↓ OCR recognizes", "#999", 9),
        (5, 6.5, "Algebra evaluates\n(synergies, cancellations,\nreinforcement, tags)", "#90CAF9", 10),
        (5, 4.5, "↓ base effect", "#999", 9),
        (5, 3.5, "Wand NN bends result\n(biography → personality)", "#CE93D8", 10),
        (5, 1.8, "↓ final effect", "#999", 9),
        (5, 0.8, "Effect in game world", "#88c999", 11),
    ]
    for x, y, text, color, size in steps:
        if "↓" in text:
            ax.text(x, y, text, ha='center', va='center', fontsize=size, color='#666')
        else:
            ax.text(x, y, text, ha='center', va='center', fontsize=size,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.7))
    ax.set_title("System Architecture", fontsize=12, fontweight='bold')

    # 2. Power comparison heatmap
    ax = axes[0, 1]
    power_matrix = np.array([
        [all_results[sn][wn]["total_power"] for wn in wand_names]
        for sn in spell_names
    ])
    im = ax.imshow(power_matrix, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(wand_names)))
    ax.set_xticklabels(wand_names, fontsize=10)
    ax.set_yticks(range(len(spell_names)))
    ax.set_yticklabels(spell_names, fontsize=10)
    for i in range(len(spell_names)):
        for j in range(len(wand_names)):
            v = power_matrix[i, j]
            best_j = np.argmax(power_matrix[i])
            weight = 'bold' if j == best_j else 'normal'
            ax.text(j, i, f"{v:.2f}", ha='center', va='center',
                    fontsize=10, fontweight=weight)
    ax.set_title("Total Power (spell × wand)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 3. Wand profiles (radar)
    ax = axes[0, 2]
    pole_names = [p.name for p in Pole]
    angles = np.linspace(0, 2 * np.pi, NUM_POLES, endpoint=False).tolist()
    angles += angles[:1]
    for wn in wand_names:
        profile = wands[wn].affinity_profile().numpy()
        vals = profile.tolist() + [profile[0]]
        ax.plot(angles, vals, 'o-', label=wn, color=colors[wn], linewidth=2)
        ax.fill(angles, vals, alpha=0.1, color=colors[wn])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(pole_names, fontsize=9)
    ax.set_title("Wand Affinity Profiles")
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 0.45)

    # 4. Algebra vs Final comparison for Fire Storm
    ax = axes[1, 0]
    fs_algebra = evaluate_spell(test_spells["Fire Storm"])
    algebra_vals = [fs_algebra.force_power.get(Element.FIRE, 0),
                    fs_algebra.force_power.get(Element.AIR, 0)]
    labels = ["FIRE (algebra)", "AIR (algebra)"]
    x = np.arange(2)
    ax.bar(x - 0.3, algebra_vals, 0.2, label="Base (algebra)", color='#FFE082')
    for i, wn in enumerate(wand_names):
        r = all_results["Fire Storm"][wn]
        vals = [r["final_force"].get(Element.FIRE, 0),
                r["final_force"].get(Element.AIR, 0)]
        ax.bar(x - 0.1 + i * 0.2, vals, 0.2, label=f"{wn} wand", color=colors[wn])
    ax.set_xticks(x)
    ax.set_xticklabels(["FIRE", "AIR"])
    ax.set_ylabel("Power")
    ax.set_title("Fire Storm: Algebra Base vs Wand-Modified")
    ax.legend(fontsize=8)

    # 5. Tags per spell
    ax = axes[1, 1]
    all_tags = sorted(set().union(*(all_results[sn][wand_names[0]]["tags"] for sn in spell_names)))
    tag_matrix = np.zeros((len(spell_names), len(all_tags)))
    for i, sn in enumerate(spell_names):
        for j, tag in enumerate(all_tags):
            if tag in all_results[sn][wand_names[0]]["tags"]:
                tag_matrix[i, j] = 1.0
    if all_tags:
        im = ax.imshow(tag_matrix, aspect='auto', cmap='Purples')
        ax.set_xticks(range(len(all_tags)))
        ax.set_xticklabels(all_tags, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(spell_names)))
        ax.set_yticklabels(spell_names, fontsize=9)
    ax.set_title("Emergent Properties (from algebra rules)")

    # 6. Advantage chart — how much each wand gains/loses vs neutral
    ax = axes[1, 2]
    # Neutral = average of all wands
    for i, sn in enumerate(spell_names):
        avg_power = np.mean([all_results[sn][wn]["total_power"] for wn in wand_names])
        for j, wn in enumerate(wand_names):
            diff = all_results[sn][wn]["total_power"] - avg_power
            ax.barh(i * 4 + j, diff, color=colors[wn], alpha=0.8, height=0.8)
    yticks = []
    ylabels = []
    for i, sn in enumerate(spell_names):
        for j, wn in enumerate(wand_names):
            yticks.append(i * 4 + j)
            ylabels.append(f"{sn[:8]}/{wn[:3]}")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Advantage vs Average")
    ax.set_title("Wand Advantage per Spell")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig("experiments/exp06_results.png", dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to experiments/exp06_results.png")
    plt.close()


if __name__ == "__main__":
    main()
