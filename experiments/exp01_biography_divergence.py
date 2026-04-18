"""
Experiment 01: Biography Divergence

Question: Does a small NN (~50K params) produce qualitatively different
outputs when trained on different synthetic biographies?

Setup:
- 3 wands with identical initialization
- Each trained on a different biography (healer, warrior, scholar)
- Same spell descriptors fed to all 3
- Compare output distributions

Success criteria:
- Pole strength distributions clearly differ between biographies
- Healer wand → biology/mental dominant
- Warrior wand → element/transform dominant
- Scholar wand → mental/space dominant
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from wand.events import BIOGRAPHIES, Pole, NUM_POLES
from wand.model import create_wand, WandTrainer


def make_test_spells(n: int = 50, dim: int = 12) -> torch.Tensor:
    """Generate random spell descriptors for evaluation."""
    torch.manual_seed(42)
    return torch.randn(n, dim)


def profile_wand(wand, spells: torch.Tensor) -> np.ndarray:
    """Get average pole distribution for a wand on given spells."""
    wand.eval()
    with torch.no_grad():
        poles, stability = wand(spells)
    return poles.mean(dim=0).numpy()


def main():
    init_seed = 7  # same init for all wands
    n_events = 500
    epochs = 15
    batch_size = 32

    # Generate biographies
    print("Generating biographies...")
    bios = {}
    for name, gen_fn in BIOGRAPHIES.items():
        bios[name] = gen_fn(n_events)
        print(f"  {name}: {len(bios[name])} events")

    # Create wands with identical init, train each on different biography
    print(f"\nTraining wands ({epochs} epochs each)...")
    wands = {}
    training_losses = {}

    for name, events in bios.items():
        wand = create_wand(seed=init_seed)
        trainer = WandTrainer(wand, lr=1e-3)
        losses = trainer.train_biography(events, batch_size=batch_size, epochs=epochs)
        wands[name] = wand
        training_losses[name] = losses
        print(f"  {name}: params={wand.param_count()}, final_loss={losses[-1]:.4f}")

    # Also keep an untrained wand for comparison
    untrained = create_wand(seed=init_seed)
    wands["untrained"] = untrained

    # Test
    test_spells = make_test_spells()
    print(f"\nPole profiles (avg over {len(test_spells)} spells):")

    pole_names = [p.name for p in Pole]
    profiles = {}

    for name, wand in wands.items():
        profile = profile_wand(wand, test_spells)
        profiles[name] = profile
        bars = " ".join(f"{pole_names[i][:4]}={profile[i]:.3f}" for i in range(NUM_POLES))
        print(f"  {name:10s}: {bars}")

    # Divergence metrics
    print("\nDivergence (L2 distance between profiles):")
    names = list(profiles.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = np.linalg.norm(profiles[names[i]] - profiles[names[j]])
            print(f"  {names[i]:10s} vs {names[j]:10s}: {d:.4f}")

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Radar chart of pole profiles
    ax = axes[0, 0]
    angles = np.linspace(0, 2 * np.pi, NUM_POLES, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    colors = {'healer': '#88c999', 'warrior': '#e07070', 'scholar': '#7090d0', 'untrained': '#999999'}
    for name, profile in profiles.items():
        vals = profile.tolist() + [profile[0]]
        ax.plot(angles, vals, 'o-', label=name, color=colors[name], linewidth=2)
        ax.fill(angles, vals, alpha=0.1, color=colors[name])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(pole_names, fontsize=9)
    ax.set_title("Affinity Hexagon — Pole Profiles")
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 0.5)

    # 2. Training loss curves
    ax = axes[0, 1]
    for name, losses in training_losses.items():
        ax.plot(losses, label=name, color=colors[name], linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted MSE Loss")
    ax.set_title("Training Loss")
    ax.legend()

    # 3. Bar chart comparison
    ax = axes[1, 0]
    x = np.arange(NUM_POLES)
    width = 0.2
    trained_names = [n for n in profiles if n != 'untrained']
    for i, name in enumerate(trained_names):
        ax.bar(x + i * width, profiles[name], width, label=name, color=colors[name])
    ax.set_xticks(x + width)
    ax.set_xticklabels(pole_names, fontsize=9)
    ax.set_ylabel("Average Pole Strength")
    ax.set_title("Pole Strength Comparison")
    ax.legend()

    # 4. Per-spell heatmap (first 20 spells)
    ax = axes[1, 1]
    n_show = 20
    heatmap_data = []
    labels = []
    for name in trained_names:
        wands[name].eval()
        with torch.no_grad():
            poles, _ = wands[name](test_spells[:n_show])
        heatmap_data.append(poles.numpy())
        labels.extend([f"{name[:3]}-{i}" for i in range(n_show)])

    combined = np.vstack(heatmap_data)
    im = ax.imshow(combined, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(NUM_POLES))
    ax.set_xticklabels(pole_names, fontsize=8)
    ax.set_ylabel("Wand-Spell pair")
    ax.set_title("Per-Spell Pole Outputs (healer|warrior|scholar)")
    # Add divider lines
    for k in range(1, len(trained_names)):
        ax.axhline(y=k * n_show - 0.5, color='white', linewidth=2)
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig("experiments/exp01_results.png", dpi=150, bbox_inches='tight')
    print("\nPlot saved to experiments/exp01_results.png")
    plt.close()


if __name__ == "__main__":
    main()
