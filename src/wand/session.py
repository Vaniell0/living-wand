"""WandSession — one object that owns the whole casting pipeline.

game.py, sandbox.py and the HTTP API used to each re-implement inference
(adapter call, activation slicing, clarity, crystals, grimoire). Now they
all go through here:

    session = WandSession()                       # loads models + player save
    sketch, report = session.cast_strokes(strokes)  # drawing → SpellProgram
    report = session.cast(graph)                    # graph   → SpellProgram
    session.report_outcome(Outcome(...))            # engine  → experience
    session.sleep(); session.save()

Experience model (replaces "press 1-6 to live"):
  - events come from engine OUTCOMES of casts, with diminishing returns for
    repeating the same program;
  - Mastery tracks familiarity per rune and per program; familiarity makes
    casting steadier (stability bonus) and makes the OCR more forgiving for
    runes you draw often — the wand learns your handwriting;
  - the WandAdapter "dreams" (sleep) on the real SpellFormer outputs of the
    casts that produced the events, not on a zero vector.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .actions import Outcome, SpellProgram, apply_signatures, compile_program, outcome_to_event
from .algebra import SpellEffect, SpellGraph, evaluate_spell
from .checker import CheckResult, check_graph
from .crystal import CrystalRegistry
from .dream import AttemptResult, DreamWeaver, Lexicon, Vision
from .effect import EffectDescription, interpret_effect
from .events import Event, Pole
from .sketch import SketchParser, SketchResult, make_recognizer
from .tokenizer import SpellTokenizer, TokenizedSpell
from .transformer import SpellFormer
from .wand_adapter import AffinityHexagon, WandAdapter, WandAdapterTrainer

ROOT = Path(__file__).resolve().parent.parent.parent


def decode_effect(raw: torch.Tensor) -> dict[str, Any]:
    """Raw 38-D logits → named, activated values (the one place this lives)."""
    r = raw[0] if raw.dim() == 2 else raw
    return {
        "poles": torch.softmax(r[:6], dim=-1).numpy(),
        "power": torch.sigmoid(r[6]).item(),
        "stability": torch.sigmoid(r[7]).item(),
        "area": F.softplus(r[8]).item(),
        "duration": F.softplus(r[9]).item(),
        "element_intensity": torch.sigmoid(r[10:16]).numpy(),
        "domain_intensity": torch.sigmoid(r[16:22]).numpy(),
        "semantic": torch.tanh(r[22:38]).numpy(),
    }


# ═══════════════════════════════════════════
# Mastery — legible, mechanical experience
# ═══════════════════════════════════════════

@dataclass
class Mastery:
    rune_casts: Counter = field(default_factory=Counter)
    program_casts: Counter = field(default_factory=Counter)
    confirmed_glyphs: Counter = field(default_factory=Counter)  # player corrections per rune

    def familiarity(self, rune: str) -> float:
        """0 → never used, → 1 with practice (half-way at ~28 casts)."""
        n = self.rune_casts.get(rune, 0) + 2 * self.confirmed_glyphs.get(rune, 0)
        return 1.0 - math.exp(-n / 40.0)

    def stability_bonus(self, runes: Sequence[str]) -> float:
        if not runes:
            return 0.0
        return 0.08 * float(np.mean([self.familiarity(r) for r in runes]))

    def record(self, program: SpellProgram):
        self.rune_casts.update(program.runes)
        self.program_casts[program.id] += 1

    def to_dict(self) -> dict:
        return {"rune_casts": dict(self.rune_casts), "program_casts": dict(self.program_casts),
                "confirmed_glyphs": dict(self.confirmed_glyphs)}

    @classmethod
    def from_dict(cls, d: dict) -> "Mastery":
        return cls(Counter(d.get("rune_casts", {})), Counter(d.get("program_casts", {})),
                   Counter(d.get("confirmed_glyphs", {})))


@dataclass
class CastReport:
    program: SpellProgram
    algebra: SpellEffect
    check: CheckResult
    tokens: TokenizedSpell
    model: EffectDescription | None
    crystal_progress: float
    newly_crystallized: bool
    crystal_name: str | None = None
    discoveries: list = field(default_factory=list)


class WandSession:
    def __init__(
        self,
        root: str | Path = ROOT,
        save_dir: str | Path | None = None,
        authority: str = "blend",
        load_ocr: bool = True,
        use_grimoire: bool = True,
        crystal_threshold: int = 500,
        ocr_self_training: bool = False,
    ):
        self.root = Path(root)
        self.data_dir = self.root / "data"
        self.save_dir = Path(save_dir) if save_dir else self.data_dir / "save"
        self.authority = authority
        self.context: dict | None = None   # world at the caster (weather/time/place)
        # vocabulary_mode: unknown runes cast weakly and are learned through use
        # and dreams (the game). Off for plain API use: every rune works fully.
        self.vocabulary_mode = False
        self.lexicon = Lexicon()
        self.vision: Vision | None = None

        # Shared world model
        self.spell_former = SpellFormer()
        sf_path = self.data_dir / "spellformer.pt"
        self.model_trained = sf_path.exists()
        if self.model_trained:
            self.spell_former.load_state_dict(torch.load(sf_path, weights_only=True))
        self.spell_former.eval()
        if not self.model_trained:
            # an untrained transformer is noise — let the algebra speak alone
            self.authority = "algebra"

        # Personal wand
        self.adapter = WandAdapter()
        self.hexagon = AffinityHexagon()
        self.trainer = WandAdapterTrainer(self.adapter, self.hexagon)
        self.tokenizer = SpellTokenizer()
        self.crystal = CrystalRegistry(threshold=crystal_threshold,
                                       save_dir=str(self.save_dir / "crystals"))
        self.mastery = Mastery()
        self.pending: list[tuple[Event, torch.Tensor | None]] = []
        self._programs: dict[str, SpellProgram] = {}
        self._base_raw: dict[str, torch.Tensor] = {}

        self.grimoire = None
        if use_grimoire:
            from .grimoire import Grimoire
            self.grimoire = Grimoire(str(self.data_dir / "grimoire.db"))

        self.ocr = None
        self.parser = SketchParser()
        if load_ocr:
            self._load_ocr(ocr_self_training)
        self.load()

    # ── loading / saving ──

    def _load_ocr(self, self_training: bool):
        from .ocr import AdaptiveOCR, SymbolNet
        candidates = [self.save_dir / "ocr_adapted_v2.pt", self.data_dir / "ocr_symbolnet_v2.pt"]
        for path in candidates:
            if path.exists():
                net = SymbolNet()
                net.load_state_dict(torch.load(path, weights_only=True))
                net.eval()
                self.ocr = AdaptiveOCR(net, self_training=self_training)
                buf = self.save_dir / "ocr_buffer_v2.pt"
                if buf.exists():
                    self.ocr.load_buffer(str(buf))
                self.parser = SketchParser(make_recognizer(self.ocr))
                self.ocr_path = path
                return
        self.ocr_path = None  # run tools/train_ocr.py (or game.py trains on first run)

    def attach_ocr(self, symbol_net, self_training: bool = False):
        from .ocr import AdaptiveOCR
        self.ocr = AdaptiveOCR(symbol_net, self_training=self_training)
        self.parser = SketchParser(make_recognizer(self.ocr), self.parser.cfg)

    def load(self):
        adapter_path = self.save_dir / "adapter.pt"
        if adapter_path.exists():
            self.adapter.load_state_dict(torch.load(adapter_path, weights_only=True))
        self.hexagon.load(str(self.save_dir / "hexagon.json"))
        self.crystal.load()
        mastery_path = self.save_dir / "mastery.json"
        if mastery_path.exists():
            self.mastery = Mastery.from_dict(json.loads(mastery_path.read_text()))
        lexicon_path = self.save_dir / "lexicon.json"
        if lexicon_path.exists():
            self.lexicon = Lexicon.from_dict(json.loads(lexicon_path.read_text()))

    def save(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.adapter.state_dict(), self.save_dir / "adapter.pt")
        self.hexagon.save(str(self.save_dir / "hexagon.json"))
        self.crystal.save()
        (self.save_dir / "mastery.json").write_text(json.dumps(self.mastery.to_dict()))
        (self.save_dir / "lexicon.json").write_text(
            json.dumps(self.lexicon.to_dict(), ensure_ascii=False))
        if self.ocr is not None and self.ocr.corrections > 0:
            torch.save(self.ocr.base.state_dict(), self.save_dir / "ocr_adapted_v2.pt")
            self.ocr.save_buffer(str(self.save_dir / "ocr_buffer_v2.pt"))

    def close(self):
        if self.grimoire is not None:
            self.grimoire.close()

    # ── inference ──

    def infer(self, tokens: TokenizedSpell) -> tuple[EffectDescription | None, torch.Tensor]:
        ids = torch.tensor([tokens.token_ids])
        roles = torch.tensor([tokens.role_ids])
        degs = torch.tensor([tokens.degree])
        with torch.no_grad():
            out = self.spell_former(ids, roles, degs)
            base_raw = out["raw"]
            modified = self.adapter(base_raw, out["pole_distribution"])
        if not self.model_trained:
            return None, base_raw
        v = decode_effect(modified)
        desc = interpret_effect(v["poles"], v["power"], v["stability"], v["area"], v["duration"],
                                v["element_intensity"], v["domain_intensity"], v["semantic"])
        return desc, base_raw

    def pole_levels(self) -> dict[str, int]:
        return {pole.name: lvl for pole, lvl in zip(Pole, self.hexagon.levels())}

    def cast(self, graph: SpellGraph, clarity: float = 1.0, record: bool = True,
             context: dict | None = None) -> CastReport:
        tokens = self.tokenizer.tokenize(graph)
        algebra = evaluate_spell(graph)
        check = check_graph(graph)
        desc, base_raw = self.infer(tokens)
        runes = [n.name for n in graph.nodes.values()]
        if self.vocabulary_mode:
            # an unfamiliar word "sounds muffled": the wand half-understands it
            clarity *= 0.6 + 0.4 * self.lexicon.familiarity(runes)
        program = compile_program(
            graph, algebra, desc, check=check, clarity=clarity, authority=self.authority,
            stability_bonus=self.mastery.stability_bonus(runes),
            context=context if context is not None else self.context,
        )
        apply_signatures(program, self.pole_levels())

        progress, crystallized, crystal_name = 0.0, False, None
        discoveries = []
        if record:
            canon, crystallized = self.crystal.record_cast(tokens)
            progress = self.crystal.approaching_crystal(tokens)
            if crystallized:
                ids = torch.tensor([tokens.token_ids])
                with torch.no_grad():
                    h = self.spell_former.encode(ids, torch.tensor([tokens.role_ids]),
                                                 torch.tensor([tokens.degree]))
                    pooled = self.spell_former.pool(h, ids)
                self.crystal.set_crystal_embedding(canon, pooled[0])
                entry = self.crystal.get_crystal_by_hash(canon)
                if entry is not None:
                    self.spell_former.set_crystal_embedding(entry.crystal_id, pooled[0])
                    crystal_name = entry.name
            self.mastery.record(program)
            learned = self.lexicon.observe(program)
            if learned:
                program.provenance["learned"] = learned
            self._programs[program.id] = program
            self._base_raw[program.id] = base_raw
            if self.grimoire is not None:
                discoveries = self.grimoire.record_cast(
                    runes=sorted(runes), edges=[list(e) for e in program.edges],
                    circle_closed=graph.circle_closed, power=program.power,
                    stability=program.stability, title=program.title,
                    tags=[op.verb for op in program.ops],
                    crystal_hash=canon if crystallized else None,
                )
        return CastReport(program, algebra, check, tokens, desc, progress, crystallized,
                          crystal_name, discoveries)

    def parse_strokes(self, strokes, circle_closed: bool = True,
                      overrides: dict[int, str] | None = None,
                      edge_overrides=None) -> SketchResult:
        return self.parser.parse(strokes, circle_closed, overrides, edge_overrides,
                                 leniency=self.mastery.familiarity)

    def cast_strokes(self, strokes, circle_closed: bool = True,
                     overrides: dict[int, str] | None = None,
                     edge_overrides=None, record: bool = True) -> tuple[SketchResult, CastReport | None]:
        sketch = self.parse_strokes(strokes, circle_closed, overrides, edge_overrides)
        if not sketch.graph.nodes:
            return sketch, None
        report = self.cast(sketch.graph, clarity=sketch.clarity, record=record)
        # Player-confirmed glyphs are trusted labels for handwriting adaptation.
        if self.ocr is not None:
            for g in sketch.glyphs:
                if g.confirmed and g.image is not None:
                    self.ocr.confirm(g.image, g.label)
                    self.mastery.confirmed_glyphs[g.label] += 1
        return sketch, report

    # ── experience ──

    def program(self, program_id: str) -> SpellProgram | None:
        return self._programs.get(program_id)

    def report_outcome(self, outcome: Outcome) -> Event | None:
        program = self._programs.get(outcome.program_id)
        if program is None:
            return None
        repeats = max(self.mastery.program_casts.get(program.id, 1) - 1, 0)
        event = outcome_to_event(program, outcome, repeat_count=repeats)
        self.pending.append((event, self._base_raw.get(program.id)))
        return event

    def add_event(self, event: Event, base_raw: torch.Tensor | None = None):
        self.pending.append((event, base_raw))

    def sleep(self) -> int:
        """Train the wand on pending events (the wand 'dreams')."""
        n = len(self.pending)
        for event, base_raw in self.pending:
            be = base_raw if base_raw is not None else torch.zeros(1, self.adapter.effect_dim)
            self.trainer.train_event(event, be)
        self.pending = []
        return n

    # ── dreams: the space where the player learns the language ──

    def _dream_eval(self, genome):
        from .explore import Evaluated, features
        program = self.cast(genome.graph(), record=False).program
        return Evaluated(genome, program, features(program))

    def dream(self, seed: int | None = None) -> Vision | None:
        """Fall asleep: the wand trains on the day, then shows a vision."""
        import random as _random
        self.sleep()
        weaver = DreamWeaver(self.lexicon, self._dream_eval)
        self.vision = weaver.conjure(_random.Random(seed))
        return self.vision

    def dream_attempt(self, graph: SpellGraph) -> AttemptResult | None:
        """Cast inside the dream: free, safe, judged against the vision."""
        if self.vision is None:
            return None
        program = self.cast(graph, record=False).program
        result = DreamWeaver(self.lexicon, self._dream_eval).attempt(self.vision, program)
        if result.solved:
            from .combo_data import match_cast
            self.lexicon.discovered.update(d.key for d in match_cast(list(program.runes)))
            self.vision = None
        return result
