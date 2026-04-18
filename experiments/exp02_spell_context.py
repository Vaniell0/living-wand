"""
Experiment 02: Spell-Dependent Behavior

Question: Do wands trained on different biographies produce meaningfully
different effects for the SAME spells?

Setup:
- 3 wands (healer, warrior, scholar) trained on biographies
- 7 preset spells cast through each wand
- Compare: healer's heal vs warrior's heal, etc.

Success criteria:
- Healer wand → strongest heal effect (biology pole)
- Warrior wand → strongest fireball effect (element pole)
- Scholar wand → strongest illusion/teleport (mental/space poles)
- Each wand is best at spells matching its biography
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from wand.events import BIOGRAPHIES, Pole, NUM_POLES
from wand.model import create_wand, WandTrainer, affinity_multiplier
from wand.spells import PRESET_SPELLS, Spell


def cast_spell(wand, spell: Spell) -> dict:
    """Cast a spell through a wand, return effect dict."""
    wand.eval()
    with torch.no_grad():
        encoded = spell.encode().unsqueeze(0)
        result = wand(encoded)
        poles = result["pole_strengths"][0].numpy()
        power = result["power"][0, 0].item()
        stability = result["stability"][0, 0].item()

    # Apply affinity multiplier
    profile = wand.affinity_profile().numpy()
    dominant_pole = int(np.argmax(profile))
    spell_sig = spell.pole_signature()
    spell_dominant = int(np.argmax(spell_sig)) if max(spell_sig) > 0 else 0

    aff_mult = affinity_multiplier(dominant_pole, spell_dominant)
    effective_power = power * aff_mult

    return {
        "poles": poles,
        "power": power,
        "effective_power": effective_power,
        "stability": stability,
        "affinity_mult": aff_mult,
    }


def main():
    init_seed = 7
    n_events = 500
    epochs = 20

    # Train wands
    print("Training wands...")
    wands = {}
    for name, gen_fn in BIOGRAPHIES.items():
        events = gen_fn(n_events)
        wand = create_wand(seed=init_seed)
        trainer = WandTrainer(wand, lr=1e-3)
        losses = trainer.train_biography(events, epochs=epochs)
        wands[name] = wand
        profile = wand.affinity_profile().numpy()
        pole_names = [p.name for p in Pole]
        dominant = pole_names[np.argmax(profile)]
        print(f"  {name}: loss={losses[-1]:.5f}, dominant={dominant} ({profile[np.argmax(profile)]:.3f})")

    # Cast all spells through all wands
    print("\n--- Spell Results ---\n")
    spell_names = list(PRESET_SPELLS.keys())
    results = {}  # results[wand_name][spell_name] = dict

    for wand_name, wand in wands.items():
        results[wand_name] = {}
        for spell_name, spell_fn in PRESET_SPELLS.items():
            spell = spell_fn()
            r = cast_spell(wand, spell)
            results[wand_name][spell_name] = r

    # Print table
    header = f"{'Spell':<14}" + "".join(f"  {n:>8s} pwr  aff" for n in wands.keys())
    print(header)
    print("-" * len(header))
    for spell_name in spell_names:
        row = f"{spell_name:<14}"
        for wand_name in wands.keys():
            r = results[wand_name][spell_name]
            row += f"  {r['effective_power']:>8.3f} {r['affinity_mult']:.1f}"
        print(row)

    # Find best wand per spell
    print("\n--- Best Wand per Spell ---")
    for spell_name in spell_names:
        best_wand = max(wands.keys(), key=lambda w: results[w][spell_name]["effective_power"])
        spell = PRESET_SPELLS[spell_name]()
        sig = spell.pole_signature()
        pole_names = [p.name for p in Pole]
        dominant_pole = pole_names[np.argmax(sig)] if max(sig) > 0 else "neutral"
        print(f"  {spell_name:<14} → {best_wand:<10} (spell pole: {dominant_pole})")

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    wand_names = list(wands.keys())
    colors = {'healer': '#88c999', 'warrior': '#e07070', 'scholar': '#7090d0'}

    # 1. Effective power comparison
    ax = axes[0, 0]
    x = np.arange(len(spell_names))
    width = 0.25
    for i, wn in enumerate(wand_names):
        vals = [results[wn][sn]["effective_power"] for sn in spell_names]
        ax.bar(x + i * width, vals, width, label=wn, color=colors[wn])
    ax.set_xticks(x + width)
    ax.set_xticklabels(spell_names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("Effective Power")
    ax.set_title("Effective Power per Spell (with affinity)")
    ax.legend()

    # 2. Pole output heatmap for heal spell
    ax = axes[0, 1]
    pole_names = [p.name[:5] for p in Pole]
    heal_data = np.array([results[wn]["heal"]["poles"] for wn in wand_names])
    im = ax.imshow(heal_data, aspect='auto', cmap='Greens')
    ax.set_xticks(range(NUM_POLES))
    ax.set_xticklabels(pole_names)
    ax.set_yticks(range(len(wand_names)))
    ax.set_yticklabels(wand_names)
    ax.set_title("Heal Spell — Pole Distribution")
    for i in range(len(wand_names)):
        for j in range(NUM_POLES):
            ax.text(j, i, f"{heal_data[i, j]:.2f}", ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 3. Pole output heatmap for fireball spell
    ax = axes[1, 0]
    fire_data = np.array([results[wn]["fireball"]["poles"] for wn in wand_names])
    im = ax.imshow(fire_data, aspect='auto', cmap='Reds')
    ax.set_xticks(range(NUM_POLES))
    ax.set_xticklabels(pole_names)
    ax.set_yticks(range(len(wand_names)))
    ax.set_yticklabels(wand_names)
    ax.set_title("Fireball Spell — Pole Distribution")
    for i in range(len(wand_names)):
        for j in range(NUM_POLES):
            ax.text(j, i, f"{fire_data[i, j]:.2f}", ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 4. Affinity profile comparison (radar)
    ax = axes[1, 1]
    pole_names_full = [p.name for p in Pole]
    angles = np.linspace(0, 2 * np.pi, NUM_POLES, endpoint=False).tolist()
    angles += angles[:1]
    for wn in wand_names:
        profile = wands[wn].affinity_profile().numpy()
        vals = profile.tolist() + [profile[0]]
        ax.plot(angles, vals, 'o-', label=wn, color=colors[wn], linewidth=2)
        ax.fill(angles, vals, alpha=0.1, color=colors[wn])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(pole_names_full, fontsize=9)
    ax.set_title("Wand Affinity Profiles")
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 0.5)

    plt.tight_layout()
    plt.savefig("experiments/exp02_results.png", dpi=150, bbox_inches='tight')
    print("\nPlot saved to experiments/exp02_results.png")
    plt.close()


if __name__ == "__main__":
    main()
