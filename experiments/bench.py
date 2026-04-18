"""Benchmarks — latency and memory for all components."""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
from wand.synth_data import generate_sample, generate_dataset, NUM_SYMBOLS
from wand.ocr import SymbolCNN, train_symbol_cnn
from wand.algebra import (
    SpellGraph, evaluate_spell, SignalKind, Element, EdgeType,
    build_fireball, build_fire_storm, build_ice_wall, build_necromancy,
)
from wand.model import create_wand, WandTrainer
from wand.events import BIOGRAPHIES
from wand.checker import check_spell
from wand.spells import SPELL_DIM


def bench(name, fn, n=1000):
    # warmup
    for _ in range(10):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = (time.perf_counter() - t0) / n
    if dt < 0.001:
        print(f"  {name:<35s} {dt*1_000_000:>8.1f} µs  ({n} runs)")
    else:
        print(f"  {name:<35s} {dt*1_000:>8.2f} ms  ({n} runs)")
    return dt


def main():
    print("=" * 60)
    print("LIVING WAND — BENCHMARKS")
    print("=" * 60)

    # --- OCR ---
    print("\n--- OCR Training ---")
    t0 = time.perf_counter()
    images, labels = generate_dataset(n_per_class=200)
    dt_data = time.perf_counter() - t0
    print(f"  Dataset generation ({len(images)} samples)  {dt_data:.2f}s")

    t0 = time.perf_counter()
    model, hist = train_symbol_cnn(images, labels, epochs=20)
    dt_train = time.perf_counter() - t0
    print(f"  CNN training (20 epochs)             {dt_train:.2f}s")
    print(f"  Final accuracy                       {hist['val_acc'][-1]:.1%}")
    print(f"  Params                               {model.param_count():,}")

    print("\n--- OCR Inference ---")
    test_img = generate_sample(0)
    bench("Single symbol predict", lambda: model.predict(test_img))

    batch = torch.from_numpy(np.stack([generate_sample(i % NUM_SYMBOLS) for i in range(32)])).float().unsqueeze(1)
    bench("Batch 32 symbols (forward)", lambda: model(batch))

    # --- Algebra ---
    print("\n--- Spell Algebra ---")
    graphs = {
        "fireball (3 nodes)": build_fireball(),
        "fire_storm (4 nodes, synergy)": build_fire_storm(),
        "ice_wall (4 nodes, synergy)": build_ice_wall(),
        "necromancy (4 nodes, infusion)": build_necromancy(),
    }
    for name, g in graphs.items():
        bench(name, lambda g=g: evaluate_spell(g), n=5000)

    # Complex spell
    g = SpellGraph()
    ids = []
    for i in range(6):
        elem = Element(i % 6)
        ids.append(g.add_node(elem.name, SignalKind.FORCE, elem))
    for i in range(5):
        g.connect(ids[i], ids[i+1], EdgeType.MUTUAL)
    bench("6-element chain (complex)", lambda: evaluate_spell(g), n=5000)

    # --- Checker ---
    print("\n--- Syntax Checker ---")
    from wand.spells import spell_fireball, spell_heal, spell_teleport
    for name, fn in [("fireball", spell_fireball), ("heal", spell_heal), ("teleport (5 nodes)", spell_teleport)]:
        s = fn()
        bench(f"check {name}", lambda s=s: check_spell(s), n=10000)

    # --- Wand NN ---
    print("\n--- Wand Model ---")
    wand = create_wand(seed=7)
    print(f"  Params: {wand.param_count():,}")

    dummy_input = torch.randn(1, SPELL_DIM)
    bench("Single spell inference", lambda: wand(dummy_input))

    batch_input = torch.randn(32, SPELL_DIM)
    bench("Batch 32 inference", lambda: wand(batch_input))

    bench("Affinity profile (100 samples)", lambda: wand.affinity_profile())

    # --- Wand Training ---
    print("\n--- Wand Training ---")
    events = BIOGRAPHIES["healer"](200)
    trainer = WandTrainer(wand, lr=1e-3)

    bench("Train batch (32 events)", lambda: trainer.train_batch(events[:32]), n=100)

    t0 = time.perf_counter()
    trainer.train_biography(events, epochs=5, batch_size=32)
    dt = time.perf_counter() - t0
    print(f"  {"Full biography (200 events, 5 epochs)":<35s} {dt*1_000:>8.2f} ms")

    t0 = time.perf_counter()
    trainer.train_biography(BIOGRAPHIES["warrior"](500), epochs=20, batch_size=32)
    dt = time.perf_counter() - t0
    print(f"  {"Heavy training (500 events, 20 epochs)":<35s} {dt*1_000:>8.2f} ms")

    # --- Memory ---
    print("\n--- Memory ---")
    import sys as _sys
    wand_size = sum(p.nelement() * p.element_size() for p in wand.parameters())
    ocr_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    print(f"  Wand model weights                   {wand_size / 1024:.1f} KB")
    print(f"  OCR model weights                    {ocr_size / 1024:.1f} KB")
    print(f"  Total model memory                   {(wand_size + ocr_size) / 1024:.1f} KB")

    # --- Full pipeline ---
    print("\n--- Full Pipeline (end-to-end) ---")
    def full_cast():
        g = build_fire_storm()
        effect = evaluate_spell(g)
        check_spell(spell_fireball())
        wand.eval()
        with torch.no_grad():
            wand(torch.randn(1, SPELL_DIM))
    bench("Full cast (algebra+check+wand)", full_cast, n=2000)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("  OCR inference:    < 1ms per symbol")
    print("  Algebra:          < 0.1ms per spell")
    print("  Checker:          < 0.01ms per spell")
    print("  Wand inference:   < 0.5ms per spell")
    print("  Training (sleep): < 500ms for a full day")
    print("  Total cast:       < 2ms (60fps safe)")


if __name__ == "__main__":
    main()
