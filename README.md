# living-wand

**A magic system that learns.**

Every magic system in games is a rules table: `fire + outward + target → fireball, damage = X`. The programmer wrote the result. The system is dead the moment it ships — every combination either exists in the table or returns an error. The designer controls the space of what is possible.

living-wand does not have a rules table. It has a Transformer.

---

## How it works

You draw runes. OCR converts the drawing to a spell graph — nodes (runes), edges (operators), an optional closed circle. A tokenizer serialises the graph into a sequence that preserves structure: not a bag of symbols, but a sequence that knows which symbol connects to which, and how.

SpellFormer — a 4-layer Transformer encoder, ~738K parameters, <15ms on CPU — reads the token sequence and predicts a 38-dimensional effect vector: six affinity poles, power, stability, element intensities, domain intensities, a 16D semantic embedding.

```
Strokes
  → Sketch       (sketch.py: glyph clustering, drawn links, grammar wiring — one renderer for train & play)
  → OCR          (SymbolNet trained on stroke-rendered runes; uncertain glyphs flagged, player confirms)
  → SpellGraph   (algebra.py: nodes, edge types, circle — evaluated in fixed operator precedence)
  → Tokenizer    (canonical graph → sequence ≤64 tokens, role + degree embeddings)
  → SpellFormer  (4-layer encoder, attention pooling, 38D output)
  → WandAdapter  (per-player drift, identity when fresh; XP from engine outcomes)
  → SpellProgram (actions.py: target · delivery · shape · property fields · verb ops · timeline ·
                  visual spec · backfire risk — engine-agnostic JSON, served by tools/serve.py)
  → Engine       (vfx.py is the reference consumer; Godot/Unity talk to the HTTP/stdio API)
  ← Outcome      (what really happened → wand experience, lexicon notes)
```

The model is trained on algebraically generated data (`algebra.py` + `datagen.py`). The algebra defines the physics of the world — synergy pairs, cancel pairs, element interactions — and generates the ground truth. The model learns to approximate it. Because it is a statistical model and not a lookup table:

- **Unseen combinations produce interpolated effects**, not errors. A rune sequence that never appeared in training still gets a prediction.
- **The boundary between "valid" and "invalid" spells is soft.** `checker.py` computes antagonism penalties and stability costs — a fire+water combination casts, at a cost, not a crash.
- **Frequently cast sequences crystallize into new vocabulary tokens** (`crystal.py`). After 5 000 identical casts, the sequence collapses into a single token whose embedding is the pooled encoder state of the original sequence. The wand invents abbreviations.

---

## Personalization

WandAdapter sits on top of SpellFormer. It is not a fine-tune of the world model — it is a separate, small overlay (6 per-pole MLP heads) trained only on a specific player's cast history.

Training is per-pole isolated: a BIOLOGY event trains exactly the BIOLOGY head. The other five heads are untouched. Specialisation in one pole slows learning in others (`multiclass_penalty`). Mastery scales logarithmically from XP.

The result: the same spell cast by a healer and a warrior produces different effects, not because the rules say so, but because their adapter weights are different.

```
AffinityHexagon
  Element — Biology — Mental
     |                  |
  Transform — Space — (centre)
```

XP banks are permanent. The wand accumulates a biography.

---

## What `algebra.py` is

`algebra.py` is the ground truth the model was trained to approximate — the formal physics of the world: `Element`, `SignalKind`, `EdgeType`, `SYNERGY_PAIRS`, `CANCEL_PAIRS`, `evaluate_spell`. It is always available for comparison: every cast in the v2 pipeline returns both the model's prediction and the algebra evaluation side by side (`CastResultV2.algebra_effect`). When the model disagrees with the algebra, that is a data point, not a bug.

`data/combos_*.json` is the human-readable expansion of the same algebra: element×element, element×domain, triple combinations, edge modifiers — with multipliers, cancel factors, and Russian-language descriptions for in-game display. Balance edits go in the JSON. Algebra logic stays in the code.

---

## Versions

| Version | What changed |
|---------|-------------|
| v0.1–v0.5 | WandCore (MLP on flat 52D spell vector), original pipeline, experiments exp01–exp06 |
| **v0.6** | **SpellFormer** (Transformer + graph tokenizer), WandAdapter (per-pole heads), crystal registry |
| v0.7 | AdaptiveOCR, all 30 runes, Grimoire, autosave |
| v0.8 | SymbolNet replaces SymbolCNN (CoordConv + SE + multi-scale + BatchNorm, 81.7% on 31 classes) |
| v0.9 | Particle and colour effects per element/domain/tag |
| **v1.0** | **Magic 2.0** — see [`docs/magic-2.md`](docs/magic-2.md): stroke pipeline (OCR 56.9% → 96.2% on human-like drawings, label flips 35.9% → 3.7%), SpellProgram API with property fields, world context, circle-as-vessel, wand signatures, dreams & lexicon, alchemy & enchanting on the same substrate, explorer / automated playtesting, new VFX |

`src/wand/model.py` and `pipeline.py` are the v0.1–v0.5 baseline — fully validated across six experiments, preserved as reference. Active development uses `pipeline_v2.py` and `transformer.py`.

---

## Running

```bash
python tools/train_ocr.py        # rune reader → data/ocr_symbolnet_v2.pt (game trains a smaller one on first run)
python train.py                  # retrain SpellFormer (the algebra changed in v1.0)

python game.py                   # draw, cast, dream (D), lexicon (Tab), world (E)
python sandbox.py                # text REPL: cast · dream/try/hint · brew · enchant · world · explore
python tools/serve.py            # engine API on :7777 (or --stdio)
python tools/explore.py --check  # automated playtest: balance report + invariants
python tools/ocr_bench.py --model data/ocr_symbolnet_v2.pt   # OCR on human-like drawings
python -m pytest tests
```

**Dependencies**: Python 3.11+, PyTorch, pygame-ce, numpy (matplotlib for reports). No GPU required.

---

## Where to read next

| If you want to … | Read this |
|------------------|-----------|
| Understand the spell algebra and world physics | `src/wand/algebra.py` |
| See how the graph becomes tokens | `src/wand/tokenizer.py` |
| Read the model architecture | `src/wand/transformer.py` |
| Understand per-player drift | `src/wand/wand_adapter.py` |
| See how crystallization works | `src/wand/crystal.py` |
| Read the magic lore and world canon | `world/00_index.md` |
| Track the roadmap and design intent | `living-wand.md` (Obsidian) |
| Read the v1.0 design (ideas, diagnosis, roadmap — Russian) | `docs/magic-2.md` |
| See what an engine receives | `src/wand/actions.py`, `GET /schema` |
| Turn strokes into a spell graph | `src/wand/sketch.py` |
| See how the player learns runes in dreams | `src/wand/dream.py` |
| Brew potions / enchant items on the same algebra | `src/wand/craft.py` |
| Stress-test the balance with agents | `src/wand/explore.py`, `tools/explore.py` |
| See the v0.1–v0.5 experiment history | `experiments/` |
