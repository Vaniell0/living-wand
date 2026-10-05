"""Train SymbolNet (rune OCR).

    python tools/train_ocr.py                      # stroke-rendered data (default)
    python tools/train_ocr.py --data legacy        # old per-pixel-jitter bitmaps

The default writes data/ocr_symbolnet_v2.pt, which game.py loads. The stroke
dataset is rendered by wand.sketch.render_strokes — the same function the game
uses on the player's drawing — so train and play see the same distribution.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from wand.ocr import train_symbol_net  # noqa: E402
from wand.synth_data import generate_dataset, generate_stroke_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", choices=["strokes", "legacy"], default="strokes")
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out = args.out or ("data/ocr_symbolnet_v2.pt" if args.data == "strokes" else "data/ocr_symbolnet.pt")

    t0 = time.time()
    if args.data == "strokes":
        images, labels = generate_stroke_dataset(args.per_class, seed=args.seed)
    else:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        images, labels = generate_dataset(args.per_class)
    print(f"{args.data}: {len(images)} images in {time.time() - t0:.1f}s")

    model, history = train_symbol_net(images, labels, epochs=args.epochs)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save(model.state_dict(), out)
    with open(out.replace(".pt", ".history.json"), "w") as f:
        json.dump({"data": args.data, "per_class": args.per_class, **history}, f)
    print(f"saved {out}  (val acc {history['val_acc'][-1]:.1%}, {time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
