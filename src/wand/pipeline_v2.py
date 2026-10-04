"""SpellFormer casting pipeline v2.

Drawing → OCR → SpellGraph → Tokenize → SpellFormer → WandAdapter → Effect → Description

Replaces pipeline.py's WandCore-based flow with the Transformer model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .algebra import EdgeType, Element, SignalKind, SpellGraph, evaluate_spell
from .checker import CheckResult, Verdict, check_spell, spell_from_graph
from .crystal import CrystalRegistry
from .effect import EffectDescription, interpret_effect
from .events import Pole
from .ocr import CircleDetector, SymbolNet
from .spells import Connection, Operator, Spell, PRIMITIVE_BY_NAME
from .tags import nearest_tags
from .tokenizer import SpellTokenizer, TokenizedSpell
from .transformer import SpellFormer
from .wand_adapter import AffinityHexagon, WandAdapter


@dataclass
class CastResultV2:
    """Full result of a spell cast through SpellFormer pipeline."""
    # OCR stage
    recognized_symbols: list[tuple[str, float]]  # (name, confidence)
    circle_closed: bool
    circle_confidence: float
    clarity: float  # average OCR confidence → modulates power

    # Syntax
    check: CheckResult
    spell: Spell

    # Tokenized
    tokens: TokenizedSpell

    # SpellFormer output (raw)
    pole_distribution: np.ndarray   # (6,)
    power: float
    stability: float
    area: float
    duration: float
    element_intensity: np.ndarray   # (6,)
    domain_intensity: np.ndarray    # (6,)
    semantic_embedding: np.ndarray  # (16,)

    # Interpreted
    effect: EffectDescription

    # Crystallization
    crystal_progress: float  # [0, 1]
    newly_crystallized: bool

    # Algebra ground truth (for comparison / debug)
    algebra_effect: object | None = None


# Symbol name → (SignalKind, Element|None) for graph construction
SYMBOL_MAP: dict[str, tuple[SignalKind, Element | None]] = {
    "fire": (SignalKind.FORCE, Element.FIRE),
    "water": (SignalKind.FORCE, Element.WATER),
    "air": (SignalKind.FORCE, Element.AIR),
    "earth": (SignalKind.FORCE, Element.EARTH),
    "light": (SignalKind.FORCE, Element.LIGHT),
    "dark": (SignalKind.FORCE, Element.DARK),
    "body": (SignalKind.DOMAIN, None),
    "mind": (SignalKind.DOMAIN, None),
    "matter": (SignalKind.DOMAIN, None),
    "space": (SignalKind.DOMAIN, None),
    "time": (SignalKind.DOMAIN, None),
    "life": (SignalKind.DOMAIN, None),
    "inward": (SignalKind.VECTOR, None),
    "outward": (SignalKind.VECTOR, None),
    "up": (SignalKind.VECTOR, None),
    "down": (SignalKind.VECTOR, None),
    "through": (SignalKind.VECTOR, None),
    "reverse": (SignalKind.VECTOR, None),
    "point": (SignalKind.SCALAR, None),
    "flow": (SignalKind.SCALAR, None),
    "wave": (SignalKind.SCALAR, None),
    "all": (SignalKind.SCALAR, None),
    "amplify": (SignalKind.MODIFIER, None),
    "weaken": (SignalKind.MODIFIER, None),
    "hold": (SignalKind.MODIFIER, None),
    "dispel": (SignalKind.MODIFIER, None),
    "here": (SignalKind.REFERENCE, None),
    "there": (SignalKind.REFERENCE, None),
    "self": (SignalKind.REFERENCE, None),
    "target": (SignalKind.REFERENCE, None),
}

# Operator name → EdgeType
EDGE_MAP = {
    "direct": EdgeType.DIRECT,
    "mutual": EdgeType.MUTUAL,
    "seq": EdgeType.SEQUENCE,
    "sequence": EdgeType.SEQUENCE,
    "cond": EdgeType.CONDITION,
    "condition": EdgeType.CONDITION,
    "neg": EdgeType.NEGATE,
    "negate": EdgeType.NEGATE,
    "iter": EdgeType.ITERATE,
    "iterate": EdgeType.ITERATE,
    "limit": EdgeType.LIMIT,
}


class CastingPipelineV2:
    """SpellFormer-based casting pipeline."""

    def __init__(
        self,
        spell_former: SpellFormer,
        wand_adapter: WandAdapter,
        hexagon: AffinityHexagon,
        crystal_registry: CrystalRegistry,
        symbol_cnn: SymbolNet | None = None,
        circle_detector: CircleDetector | None = None,
    ):
        self.spell_former = spell_former
        self.wand_adapter = wand_adapter
        self.hexagon = hexagon
        self.crystal = crystal_registry
        self.symbol_cnn = symbol_cnn
        self.circle_detector = circle_detector
        self.tokenizer = SpellTokenizer()

    def cast_from_graph(
        self,
        graph: SpellGraph,
        spell: Spell | None = None,
    ) -> CastResultV2:
        """Cast from a pre-built SpellGraph (used by sandbox.py and game.py).

        Args:
            graph: the spell graph
            spell: optional Spell object for checker compatibility
        """
        # Tokenize
        tokens = self.tokenizer.tokenize(graph)

        # Syntax check
        if spell is None:
            spell = spell_from_graph(graph)

        check = check_spell(spell)

        # --- SpellFormer inference ---
        self.spell_former.eval()
        ids = torch.tensor([tokens.token_ids])
        roles = torch.tensor([tokens.role_ids])
        degs = torch.tensor([tokens.degree])

        with torch.no_grad():
            output = self.spell_former(ids, roles, degs)

            # Apply wand adapter
            base_raw = output["raw"]
            spell_poles = output["pole_distribution"]
            modified_raw = self.wand_adapter(base_raw, spell_poles)

            # Re-apply activations to modified raw
            poles = torch.softmax(modified_raw[:, :6], dim=-1)[0].numpy()
            power = torch.sigmoid(modified_raw[:, 6:7])[0, 0].item()
            stability = torch.sigmoid(modified_raw[:, 7:8])[0, 0].item()
            area = torch.nn.functional.softplus(modified_raw[:, 8:9])[0, 0].item()
            duration = torch.nn.functional.softplus(modified_raw[:, 9:10])[0, 0].item()
            elem_int = torch.sigmoid(modified_raw[:, 10:16])[0].numpy()
            dom_int = torch.sigmoid(modified_raw[:, 16:22])[0].numpy()
            sem_emb = torch.tanh(modified_raw[:, 22:38])[0].numpy()

        # Apply checker penalty
        power *= check.stability_penalty
        stability *= check.stability_penalty

        # Interpret
        effect_desc = interpret_effect(
            poles, power, stability, area, duration, elem_int, dom_int, sem_emb,
        )

        # Crystallization
        canon, newly_crystallized = self.crystal.record_cast(tokens)
        crystal_progress = self.crystal.approaching_crystal(tokens)

        if newly_crystallized:
            # Compute crystal embedding from SpellFormer encoder
            with torch.no_grad():
                h = self.spell_former.encode(ids, roles, degs)
                pooled = self.spell_former.pool(h, ids)
                self.crystal.set_crystal_embedding(canon, pooled[0])
                entry = self.crystal.get_crystal_by_hash(canon)
                if entry is not None:
                    self.spell_former.set_crystal_embedding(entry.crystal_id, pooled[0])

        # Algebra ground truth (for debug/comparison)
        algebra_effect = evaluate_spell(graph)

        return CastResultV2(
            recognized_symbols=[],
            circle_closed=graph.circle_closed,
            circle_confidence=1.0,
            clarity=1.0,
            check=check,
            spell=spell,
            tokens=tokens,
            pole_distribution=poles,
            power=power,
            stability=stability,
            area=area,
            duration=duration,
            element_intensity=elem_int,
            domain_intensity=dom_int,
            semantic_embedding=sem_emb,
            effect=effect_desc,
            crystal_progress=crystal_progress,
            newly_crystallized=newly_crystallized,
            algebra_effect=algebra_effect,
        )

    def cast_from_ocr(
        self,
        symbol_images: list[np.ndarray],
        circle_image: np.ndarray | None,
        connections: list[tuple[int, int, str]],
    ) -> CastResultV2:
        """Cast from raw drawings (OCR → graph → SpellFormer).

        Args:
            symbol_images: list of 64x64 grayscale images
            circle_image: 64x64 image for circle detection
            connections: (from_idx, to_idx, operator_name) triples
        """
        if self.symbol_cnn is None:
            raise RuntimeError("No SymbolNet loaded — use cast_from_graph for text mode")

        # OCR
        recognized = []
        for img in symbol_images:
            name, conf = self.symbol_cnn.predict(img)
            recognized.append((name, conf))

        # Circle
        circle_closed = False
        circle_conf = 0.0
        if circle_image is not None and self.circle_detector is not None:
            circle_closed, circle_conf = self.circle_detector.is_closed(circle_image)

        # Clarity = average confidence
        clarity = np.mean([c for _, c in recognized]) if recognized else 0.5

        # Build graph
        graph = SpellGraph(circle_closed=circle_closed)
        node_ids = []
        for name, conf in recognized:
            if name in SYMBOL_MAP:
                kind, elem = SYMBOL_MAP[name]
                nid = graph.add_node(name, kind, elem)
                node_ids.append(nid)

        for from_idx, to_idx, op_name in connections:
            if from_idx < len(node_ids) and to_idx < len(node_ids):
                edge_type = EDGE_MAP.get(op_name.lower(), EdgeType.DIRECT)
                graph.connect(node_ids[from_idx], node_ids[to_idx], edge_type)

        result = self.cast_from_graph(graph)

        # Override OCR-specific fields
        result.recognized_symbols = recognized
        result.circle_confidence = circle_conf
        result.clarity = clarity

        # Apply clarity modifier
        result.power *= clarity
        result.stability *= 0.5 + 0.5 * clarity
        result.effect.power *= clarity

        return result
