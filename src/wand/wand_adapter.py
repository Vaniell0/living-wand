"""WandAdapter — per-wand personality modifier on top of SpellFormer.

The SpellFormer is shared "world magic knowledge". The WandAdapter is the
player's personal wand that biases effects based on accumulated experience.

Architecture: 6 independent per-pole heads (training one pole never affects
another) + affinity gate + global bias. ~15K parameters.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .events import Event, Pole, NUM_POLES
from .transformer import EFFECT_DIM


class WandAdapter(nn.Module):
    """Per-wand personality that biases the shared magic model's output.

    Uses per-pole heads to prevent catastrophic forgetting: training
    Biology affinity never degrades Element mastery.
    """

    def __init__(self, effect_dim: int = EFFECT_DIM, hidden: int = 32, n_poles: int = NUM_POLES):
        super().__init__()
        self.n_poles = n_poles
        self.effect_dim = effect_dim

        # Per-pole bias heads — independent, each only trained by its pole's events
        self.pole_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(effect_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, effect_dim),
            )
            for _ in range(n_poles)
        ])

        # Affinity weights: how much each pole influences the output
        # Starts uniform, shaped by experience
        self.affinity = nn.Parameter(torch.ones(n_poles) / n_poles)

        # Global bias (small residual shift across all effects)
        self.bias = nn.Parameter(torch.zeros(effect_dim))

        # Gating strength: controls how much the adapter modifies the base
        # Starts small, grows with experience
        self.gate_scale = nn.Parameter(torch.tensor(0.1))

        # A fresh wand must be an exact identity over SpellFormer: zero the
        # output layer of every head so personality only appears through
        # training (gradients still flow — the hidden layer is random).
        for head in self.pole_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, base_effect: torch.Tensor,
                spell_pole_dist: torch.Tensor) -> torch.Tensor:
        """Apply wand personality to base SpellFormer output.

        Args:
            base_effect:     (batch, effect_dim) from SpellFormer
            spell_pole_dist: (batch, 6) which poles this spell activates

        Returns:
            modified_effect: (batch, effect_dim)
        """
        # Weight each pole head by (learned affinity * spell relevance)
        aff = torch.softmax(self.affinity, dim=0)  # (6,)
        weights = aff.unsqueeze(0) * spell_pole_dist  # (batch, 6)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Compute per-pole modifications
        modifications = torch.zeros_like(base_effect)
        for i, head in enumerate(self.pole_heads):
            pole_mod = head(base_effect)  # (batch, effect_dim)
            modifications = modifications + weights[:, i:i + 1] * pole_mod

        # Apply as gated residual with tanh (prevents runaway)
        gate = torch.sigmoid(self.gate_scale)
        return base_effect + gate * torch.tanh(modifications) + self.bias

    def affinity_profile(self) -> torch.Tensor:
        """Return the wand's current affinity as a 6D softmax vector."""
        return torch.softmax(self.affinity, dim=0).detach()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════
# Per-pole Experience Banks
# ═══════════════════════════════════════════

@dataclass
class ExperienceBank:
    """Per-pole accumulated experience. Only grows, never shrinks."""
    pole: Pole
    total_xp: float = 0.0
    event_count: int = 0

    @property
    def level(self) -> int:
        """Logarithmic level from XP."""
        return int(math.log2(self.total_xp + 1))

    @property
    def mastery_mult(self) -> float:
        """Multiplier for this pole's spells. +10% per level."""
        return 1.0 + 0.1 * self.level

    def add_xp(self, amount: float):
        self.total_xp += abs(amount)
        self.event_count += 1


class AffinityHexagon:
    """The 6-pole experience profile. Each pole is independent."""

    def __init__(self):
        self.banks = {pole: ExperienceBank(pole) for pole in Pole}

    def save(self, path: str):
        """Save hexagon state to JSON."""
        import json, os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {pole.name: {"total_xp": b.total_xp, "event_count": b.event_count}
                for pole, b in self.banks.items()}
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> bool:
        """Load hexagon state from JSON. Returns True if loaded."""
        import json
        if not os.path.exists(path):
            return False
        with open(path) as f:
            data = json.load(f)
        for pole in Pole:
            if pole.name in data:
                self.banks[pole].total_xp = data[pole.name]["total_xp"]
                self.banks[pole].event_count = data[pole.name]["event_count"]
        return True

    def add_experience(self, pole: Pole, amount: float):
        self.banks[pole].add_xp(amount)

    def multiclass_penalty(self, target_pole: Pole) -> float:
        """Learning a new pole is slower if you're already specialized.
        But it NEVER reduces existing poles.

        Returns: learning rate multiplier [0.46, 1.0]
        """
        total_xp = sum(b.total_xp for b in self.banks.values())
        if total_xp == 0:
            return 1.0
        target_xp = self.banks[target_pole].total_xp
        specialization = target_xp / (total_xp + 1)
        return 0.4 + 0.6 * (specialization + 0.1)

    def profile(self) -> list[float]:
        """Return 6D profile: XP per pole, normalized."""
        xps = [b.total_xp for b in self.banks.values()]
        total = sum(xps)
        if total == 0:
            return [1.0 / 6] * 6
        return [x / total for x in xps]

    def levels(self) -> list[int]:
        """Return level per pole."""
        return [b.level for b in self.banks.values()]


# ═══════════════════════════════════════════
# Wand Trainer (per-pole isolation)
# ═══════════════════════════════════════════

class WandAdapterTrainer:
    """Trains a WandAdapter from life events with per-pole isolation.

    Key invariant: training events for pole X only update pole_head[X].
    Other pole heads are frozen during that training step.
    """

    def __init__(self, adapter: WandAdapter, hexagon: AffinityHexagon, lr: float = 5e-4):
        self.adapter = adapter
        self.hexagon = hexagon
        self.lr = lr
        # Per-pole optimizers (each only manages its own head's params)
        self.pole_optimizers = [
            torch.optim.Adam(head.parameters(), lr=lr)
            for head in adapter.pole_heads
        ]
        # Global optimizer for affinity + bias + gate
        self.global_optimizer = torch.optim.Adam(
            [adapter.affinity, adapter.bias, adapter.gate_scale], lr=lr
        )
        self.loss_fn = nn.MSELoss()

    def train_event(self, event: Event, base_effect: torch.Tensor):
        """Train on a single event. Only updates the dominant pole's head.

        Args:
            event: the life/cast event
            base_effect: (1, effect_dim) SpellFormer output for context
        """
        # Determine which pole to train
        dominant_pole = int(max(range(NUM_POLES), key=lambda i: event.pole_affinity[i]))

        # Add XP (with multiclass penalty)
        xp = event.weight * self.hexagon.multiclass_penalty(Pole(dominant_pole))
        self.hexagon.add_experience(Pole(dominant_pole), xp)

        # Create training target: shift effect toward event's pole distribution
        target_poles = torch.tensor([event.pole_affinity], dtype=torch.float32)
        target_poles = target_poles / (target_poles.sum() + 1e-8)

        # Construct spell_pole_dist from event
        spell_pole_dist = target_poles

        self.adapter.train()
        modified = self.adapter(base_effect, spell_pole_dist)

        # Loss: modified pole distribution should lean toward event poles
        # Only affect the pole_distribution slice (dims 0:6)
        pred_poles = torch.softmax(modified[:, :6], dim=-1)
        loss = self.loss_fn(pred_poles, target_poles)

        # Only update the dominant pole's head
        self.pole_optimizers[dominant_pole].zero_grad()
        self.global_optimizer.zero_grad()
        loss.backward()
        self.pole_optimizers[dominant_pole].step()
        self.global_optimizer.step()

        return loss.item()

    def train_events(self, events: list[Event], base_effects: torch.Tensor) -> float:
        """Train on a batch of events sequentially.

        Args:
            events: list of events
            base_effects: (N, effect_dim) or (1, effect_dim) broadcast

        Returns:
            average loss
        """
        total_loss = 0.0
        for i, event in enumerate(events):
            if base_effects.shape[0] > 1:
                be = base_effects[i:i + 1]
            else:
                be = base_effects
            total_loss += self.train_event(event, be)
        return total_loss / max(len(events), 1)
