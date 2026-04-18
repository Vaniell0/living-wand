"""Wand neural network — the core model that accumulates experience."""

import random
import torch
import torch.nn as nn
from .events import Event, NUM_POLES
from .spells import SPELL_DIM


class WandCore(nn.Module):
    """
    Spell-aware wand model.

    Input: spell descriptor (encoded primitives, operators, circle)
    Output: effect parameters (strength per pole + power + stability)

    The weights are shaped by training on biography events,
    making each wand respond differently to the same spell.
    """

    def __init__(self, spell_dim: int = SPELL_DIM, hidden: int = 96, n_poles: int = NUM_POLES):
        super().__init__()
        self.n_poles = n_poles

        # Spell encoder — understands spell structure
        self.spell_enc = nn.Sequential(
            nn.Linear(spell_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )

        # Experience core — this is where the "personality" lives
        self.core = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        # Effect decoder — per-pole strength + power + stability
        self.effect_head = nn.Linear(hidden, n_poles + 2)  # 6 poles + power + stability

    def forward(self, spell: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            spell: (batch, spell_dim) encoded spell
        Returns:
            dict with:
              pole_strengths: (batch, n_poles) softmaxed
              power: (batch, 1) overall effect power [0, 1]
              stability: (batch, 1) how predictable [0, 1]
        """
        h = self.spell_enc(spell)
        h = self.core(h)
        out = self.effect_head(h)
        return {
            "pole_strengths": torch.softmax(out[:, :self.n_poles], dim=-1),
            "power": torch.sigmoid(out[:, self.n_poles:self.n_poles + 1]),
            "stability": torch.sigmoid(out[:, self.n_poles + 1:]),
        }

    def affinity_profile(self, n_samples: int = 100) -> torch.Tensor:
        """Compute wand's affinity profile by averaging over random spell inputs."""
        self.eval()
        with torch.no_grad():
            torch.manual_seed(0)
            dummy = torch.randn(n_samples, SPELL_DIM)
            result = self(dummy)
            return result["pole_strengths"].mean(dim=0)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def hex_distance(a: int, b: int, n: int = NUM_POLES) -> int:
    """Distance on the hexagon between two poles."""
    d = abs(a - b)
    return min(d, n - d)


def affinity_multiplier(home_pole: int, target_pole: int) -> float:
    """HxH-style affinity: 1.0 at home, 0.8/0.6/0.4 at distance 1/2/3."""
    return 1.0 - 0.2 * hex_distance(home_pole, target_pole)


class WandTrainer:
    """Trains a WandCore on biography events."""

    def __init__(self, wand: WandCore, lr: float = 1e-3):
        self.wand = wand
        self.optimizer = torch.optim.Adam(wand.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss(reduction='none')

    def _events_to_tensors(self, events: list[Event]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert events to (inputs, targets, weights) tensors."""
        features = torch.tensor(
            [e.to_tensor_components() for e in events], dtype=torch.float32
        )
        targets = torch.tensor(
            [e.pole_affinity for e in events], dtype=torch.float32
        )
        weights = torch.tensor(
            [e.weight for e in events], dtype=torch.float32
        )

        # Pad/truncate to SPELL_DIM
        if features.shape[1] >= SPELL_DIM:
            inputs = features[:, :SPELL_DIM]
        else:
            inputs = torch.zeros(len(events), SPELL_DIM)
            inputs[:, :features.shape[1]] = features

        # Normalize targets to distribution
        target_sum = targets.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        target_dist = targets / target_sum

        return inputs, target_dist, weights

    def train_batch(self, events: list[Event]) -> float:
        if not events:
            return 0.0

        self.wand.train()
        inputs, target_dist, weights = self._events_to_tensors(events)

        result = self.wand(inputs)
        pole_pred = result["pole_strengths"]

        loss_per_sample = self.loss_fn(pole_pred, target_dist).mean(dim=-1)
        weighted_loss = (loss_per_sample * weights).mean()

        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()

        return weighted_loss.item()

    def train_biography(self, events: list[Event], batch_size: int = 32, epochs: int = 5) -> list[float]:
        losses = []
        for _ in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            shuffled = events.copy()
            random.shuffle(shuffled)

            for i in range(0, len(shuffled), batch_size):
                batch = shuffled[i:i + batch_size]
                epoch_loss += self.train_batch(batch)
                n_batches += 1

            losses.append(epoch_loss / max(n_batches, 1))
        return losses


def create_wand(seed: int | None = None) -> WandCore:
    """Create a fresh wand with optional reproducible init."""
    if seed is not None:
        torch.manual_seed(seed)
    return WandCore()
