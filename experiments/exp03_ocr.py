"""
Experiment 03: OCR — Symbol Recognition

Train a small CNN on synthetic hand-drawn symbols.
Test accuracy, show confusion matrix, visualize samples.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from wand.synth_data import generate_dataset, generate_sample, SYMBOL_NAMES, NUM_SYMBOLS
from wand.ocr import train_symbol_cnn, train_circle_detector
from wand.synth_data import generate_circle


def main():
    print("Generating synthetic symbol dataset...")
    images, labels = generate_dataset(n_per_class=300)
    print(f"  {len(images)} samples, {NUM_SYMBOLS} classes")

    print("\nTraining Symbol CNN...")
    model, history = train_symbol_cnn(images, labels, epochs=25, batch_size=64)
    final_acc = history["val_acc"][-1]
    print(f"  Params: {model.param_count()}")
    print(f"  Final val accuracy: {final_acc:.1%}")

    # Per-class accuracy
    print("\nPer-class accuracy:")
    import torch
    model.eval()
    correct = np.zeros(NUM_SYMBOLS)
    total = np.zeros(NUM_SYMBOLS)
    confusion = np.zeros((NUM_SYMBOLS, NUM_SYMBOLS), dtype=int)

    for i in range(len(images)):
        name, conf = model.predict(images[i])
        pred_idx = SYMBOL_NAMES.index(name)
        true_idx = labels[i]
        confusion[true_idx, pred_idx] += 1
        total[true_idx] += 1
        if pred_idx == true_idx:
            correct[true_idx] += 1

    for i, name in enumerate(SYMBOL_NAMES):
        acc = correct[i] / total[i] if total[i] > 0 else 0
        print(f"  {name:<8s}: {acc:.1%}")

    # Train circle detector
    print("\nTraining Circle Detector...")
    circle_model, circle_hist = train_circle_detector(n_samples=800, epochs=15)
    circle_acc = circle_hist["val_acc"][-1]
    print(f"  Final val accuracy: {circle_acc:.1%}")

    # --- Plots ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Sample symbols
    ax = axes[0, 0]
    grid = np.zeros((64 * 2, 64 * 4))
    for i in range(NUM_SYMBOLS):
        row, col = i // 4, i % 4
        grid[row * 64:(row + 1) * 64, col * 64:(col + 1) * 64] = generate_sample(i)
    ax.imshow(grid, cmap='gray_r')
    for i in range(NUM_SYMBOLS):
        row, col = i // 4, i % 4
        ax.text(col * 64 + 32, row * 64 + 58, SYMBOL_NAMES[i],
                ha='center', va='bottom', fontsize=8, color='blue')
    ax.set_title("Symbol Samples")
    ax.axis('off')

    # 2. Training curve
    ax = axes[0, 1]
    ax.plot(history["train_loss"], label="Train Loss", color='#e07070')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax2 = ax.twinx()
    ax2.plot(history["val_acc"], label="Val Acc", color='#7090d0')
    ax2.set_ylabel("Accuracy")
    ax.set_title("Symbol CNN Training")
    ax.legend(loc='upper left')
    ax2.legend(loc='lower right')

    # 3. Confusion matrix
    ax = axes[0, 2]
    im = ax.imshow(confusion, cmap='Blues')
    ax.set_xticks(range(NUM_SYMBOLS))
    ax.set_xticklabels([s[:4] for s in SYMBOL_NAMES], rotation=45, fontsize=8)
    ax.set_yticks(range(NUM_SYMBOLS))
    ax.set_yticklabels([s[:4] for s in SYMBOL_NAMES], fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(NUM_SYMBOLS):
        for j in range(NUM_SYMBOLS):
            if confusion[i, j] > 0:
                ax.text(j, i, str(confusion[i, j]), ha='center', va='center', fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 4. Circle detector samples
    ax = axes[1, 0]
    grid2 = np.zeros((64 * 2, 64 * 4))
    circle_labels = []
    for i in range(8):
        closed = i < 4
        img = generate_circle(closed=closed)
        row, col = i // 4, i % 4
        grid2[row * 64:(row + 1) * 64, col * 64:(col + 1) * 64] = img
        detected, prob = circle_model.is_closed(img)
        circle_labels.append(f"{'C' if closed else 'O'} p={prob:.2f}")
    ax.imshow(grid2, cmap='gray_r')
    for i in range(8):
        row, col = i // 4, i % 4
        ax.text(col * 64 + 32, row * 64 + 58, circle_labels[i],
                ha='center', va='bottom', fontsize=8, color='green' if i < 4 else 'red')
    ax.set_title("Circle Detector (C=closed, O=open)")
    ax.axis('off')

    # 5. Circle training curve
    ax = axes[1, 1]
    ax.plot(circle_hist["train_loss"], label="Train Loss", color='#e07070')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax2 = ax.twinx()
    ax2.plot(circle_hist["val_acc"], label="Val Acc", color='#7090d0')
    ax2.set_ylabel("Accuracy")
    ax.set_title("Circle Detector Training")
    ax.legend(loc='upper left')
    ax2.legend(loc='lower right')

    # 6. Per-class accuracy bars
    ax = axes[1, 2]
    accs = [correct[i] / total[i] if total[i] > 0 else 0 for i in range(NUM_SYMBOLS)]
    bar_colors = ['#E85D3A', '#2196F3', '#8B6914', '#90CAF9',
                  '#4CAF50', '#9C27B0', '#00BCD4', '#FFC107']
    ax.bar(range(NUM_SYMBOLS), accs, color=bar_colors)
    ax.set_xticks(range(NUM_SYMBOLS))
    ax.set_xticklabels(SYMBOL_NAMES, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiments/exp03_results.png", dpi=150, bbox_inches='tight')
    print("\nPlot saved to experiments/exp03_results.png")
    plt.close()


if __name__ == "__main__":
    main()
