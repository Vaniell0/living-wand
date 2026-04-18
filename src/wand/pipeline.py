"""End-to-end pipeline: drawing → OCR → checker → wand → effect.

This module connects all components into a single casting flow.
"""

from dataclasses import dataclass
import numpy as np
import torch

from .spells import Spell, Primitive, Connection, Operator, PRIMITIVE_BY_NAME, SPELL_DIM
from .ocr import SymbolCNN, CircleDetector
from .checker import check_spell, CheckResult, Verdict
from .model import WandCore, affinity_multiplier
from .events import Pole, NUM_POLES


@dataclass
class DrawingInput:
    """Simulated drawing input — list of symbol images + circle image."""
    symbol_images: list[np.ndarray]       # list of 64x64 images
    circle_image: np.ndarray | None       # 64x64 circle image, or None
    connections: list[tuple[int, int, Operator]]  # (from_idx, to_idx, operator)


@dataclass
class CastResult:
    """Full result of a spell cast."""
    # OCR stage
    recognized_symbols: list[tuple[str, float]]  # (name, confidence)
    circle_closed: bool
    circle_confidence: float

    # Checker stage
    check: CheckResult
    spell: Spell

    # Wand stage
    pole_strengths: np.ndarray  # 6 floats
    raw_power: float
    stability: float

    # Final effect (after affinity + checker penalties)
    effective_power: float
    effective_stability: float
    dominant_pole: str
    effect_description: str


# Effect descriptions based on dominant pole
EFFECT_TEMPLATES = {
    Pole.ELEMENT: [
        "A burst of elemental energy erupts from the wand",
        "Raw elemental force surges forward",
        "The air crackles with elemental power",
    ],
    Pole.TRANSFORM: [
        "The target shimmers and shifts, its properties changing",
        "Matter bends and reshapes under the wand's influence",
        "A silver ripple passes through the target, transforming it",
    ],
    Pole.BIOLOGY: [
        "Warm green light flows from the wand, mending and growing",
        "Life energy pulses gently into the target",
        "A soft glow surrounds the target, healing and restoring",
    ],
    Pole.MENTAL: [
        "A violet wave ripples through the air, touching minds",
        "Reality shimmers — is it real or illusion?",
        "The wand hums with psychic resonance",
    ],
    Pole.SPACE: [
        "Space folds and twists around the wand tip",
        "A rift opens briefly, connecting distant points",
        "The fabric of space warps at the wand's command",
    ],
    Pole.TIME: [
        "Time stutters — moments stretch and compress",
        "Golden light bends the flow of time itself",
        "The world slows to a crawl around the wand",
    ],
}


class CastingPipeline:
    """Connects OCR → Checker → Wand into a single casting flow."""

    def __init__(self, symbol_cnn: SymbolCNN, circle_detector: CircleDetector, wand: WandCore):
        self.symbol_cnn = symbol_cnn
        self.circle_detector = circle_detector
        self.wand = wand

    def cast(self, drawing: DrawingInput) -> CastResult:
        """Execute the full casting pipeline."""
        import random

        # --- Stage 1: OCR ---
        recognized = []
        primitives = []
        for img in drawing.symbol_images:
            name, conf = self.symbol_cnn.predict(img)
            recognized.append((name, conf))
            if name in PRIMITIVE_BY_NAME:
                primitives.append(PRIMITIVE_BY_NAME[name])

        # Circle detection
        circle_closed = False
        circle_conf = 0.0
        if drawing.circle_image is not None:
            circle_closed, circle_conf = self.circle_detector.is_closed(drawing.circle_image)

        # Build connections
        connections = []
        for from_idx, to_idx, op in drawing.connections:
            if from_idx < len(primitives) and to_idx < len(primitives):
                connections.append(Connection(from_idx, to_idx, op))

        spell = Spell(primitives, connections, circle_closed=circle_closed)

        # --- Stage 2: Syntax Check ---
        check = check_spell(spell)

        # --- Stage 3: Wand inference ---
        self.wand.eval()
        with torch.no_grad():
            encoded = spell.encode().unsqueeze(0)
            result = self.wand(encoded)
            poles = result["pole_strengths"][0].numpy()
            raw_power = result["power"][0, 0].item()
            raw_stability = result["stability"][0, 0].item()

        # Apply affinity multiplier
        profile = self.wand.affinity_profile().numpy()
        wand_dominant = int(np.argmax(profile))
        spell_sig = spell.pole_signature()
        spell_dominant = int(np.argmax(spell_sig)) if max(spell_sig) > 0 else 0
        aff_mult = affinity_multiplier(wand_dominant, spell_dominant)

        # Apply checker penalty
        effective_power = raw_power * aff_mult * check.stability_penalty
        effective_stability = raw_stability * check.stability_penalty

        # Determine dominant pole
        dominant_idx = int(np.argmax(poles))
        dominant_pole = Pole(dominant_idx)

        # Generate description
        templates = EFFECT_TEMPLATES[dominant_pole]
        base_desc = random.choice(templates)

        if check.verdict == Verdict.CRITICAL:
            effect_desc = f"COLLAPSE! {base_desc}, but the spell shatters — uncontrolled energy erupts!"
        elif check.verdict == Verdict.ERROR:
            effect_desc = f"The spell fizzles... {base_desc.lower()}, but weak and unstable."
        elif check.verdict == Verdict.WARNING:
            effect_desc = f"{base_desc}, though the effect wavers slightly."
        else:
            effect_desc = f"{base_desc}."

        # Add power descriptor
        if effective_power > 0.7:
            effect_desc += " The effect is powerful and decisive."
        elif effective_power > 0.4:
            effect_desc += " A solid, workable effect."
        elif effective_power > 0.2:
            effect_desc += " The effect is faint but present."
        else:
            effect_desc += " Barely a whisper of magic."

        return CastResult(
            recognized_symbols=recognized,
            circle_closed=circle_closed,
            circle_confidence=circle_conf,
            check=check,
            spell=spell,
            pole_strengths=poles,
            raw_power=raw_power,
            stability=raw_stability,
            effective_power=effective_power,
            effective_stability=effective_stability,
            dominant_pole=dominant_pole.name,
            effect_description=effect_desc,
        )
