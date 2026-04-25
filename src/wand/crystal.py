"""Crystallization — frequent spell patterns compress into new tokens.

When a spell is cast thousands of times, it "crystallizes" into a single
token that can be used as a component in larger spells. Like how math
invented shorthand notation for common expressions.

The crystal token's embedding is the attention-pooled hidden state of
the original spell's encoding in SpellFormer.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .tokenizer import (
    CRYSTAL_OFFSET,
    MAX_CRYSTALS,
    MAX_SEQ_LEN,
    SpecialToken,
    TokenizedSpell,
)


@dataclass
class CrystalEntry:
    """A crystallized spell — a compressed token for a frequent combination."""
    crystal_id: int                # 0-based index (token_id = CRYSTAL_OFFSET + crystal_id)
    original_tokens: list[int]     # the original token sequence
    embedding: torch.Tensor | None # (d_model,) frozen embedding vector
    cast_count: int                # total casts that contributed
    name: str                      # auto-generated or player-named
    effect_cache: dict | None = None  # cached effect for quick lookup

    @property
    def token_id(self) -> int:
        return CRYSTAL_OFFSET + self.crystal_id


def _canonical_hash(token_ids: list[int]) -> str:
    """Produce a stable hash of a spell's token sequence.

    Ignores PAD tokens and normalizes order-independent parts.
    """
    # Strip PAD and EOS
    stripped = [t for t in token_ids if t not in (SpecialToken.PAD, SpecialToken.EOS)]
    raw = json.dumps(stripped, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class CrystalRegistry:
    """Tracks spell frequencies and manages crystallized tokens."""

    def __init__(self, threshold: int = 5000, save_dir: str = "data/crystals"):
        self.threshold = threshold
        self.save_dir = Path(save_dir)
        self.cast_counts: dict[str, int] = {}
        self.cast_tokens: dict[str, list[int]] = {}  # hash → original tokens
        self.crystals: dict[str, CrystalEntry] = {}
        self._next_id = 0

    @property
    def num_crystals(self) -> int:
        return len(self.crystals)

    def record_cast(self, tokenized: TokenizedSpell) -> tuple[str, bool]:
        """Record a cast and check if crystallization threshold is reached.

        Returns:
            (canonical_hash, newly_crystallized)
        """
        canon = _canonical_hash(tokenized.token_ids)
        self.cast_counts[canon] = self.cast_counts.get(canon, 0) + 1
        self.cast_tokens[canon] = tokenized.token_ids[:tokenized.length]

        newly_crystallized = False
        if (
            self.cast_counts[canon] >= self.threshold
            and canon not in self.crystals
            and self._next_id < MAX_CRYSTALS
        ):
            self._crystallize(canon)
            newly_crystallized = True

        return canon, newly_crystallized

    def _crystallize(self, canon: str):
        """Create a new crystal entry (embedding set later by SpellFormer)."""
        original = self.cast_tokens.get(canon, [])
        crystal = CrystalEntry(
            crystal_id=self._next_id,
            original_tokens=original,
            embedding=None,
            cast_count=self.cast_counts[canon],
            name=self._auto_name(original),
        )
        self.crystals[canon] = crystal
        self._next_id += 1

    def _auto_name(self, token_ids: list[int]) -> str:
        """Generate a name from the rune tokens in the sequence."""
        from .tokenizer import RUNE_NAMES, NUM_RUNES
        rune_names = []
        for tid in token_ids:
            if 0 <= tid < NUM_RUNES:
                rune_names.append(RUNE_NAMES[tid])
        if rune_names:
            return "+".join(rune_names[:4])
        return f"crystal_{self._next_id}"

    def set_crystal_embedding(self, canon: str, embedding: torch.Tensor):
        """Set the frozen embedding for a crystallized spell.

        Called by the inference pipeline after encoding the original sequence.
        """
        if canon in self.crystals:
            self.crystals[canon].embedding = embedding.detach().clone()

    def get_crystal_by_hash(self, canon: str) -> CrystalEntry | None:
        return self.crystals.get(canon)

    def get_crystal_by_id(self, crystal_id: int) -> CrystalEntry | None:
        for c in self.crystals.values():
            if c.crystal_id == crystal_id:
                return c
        return None

    def get_cast_count(self, tokenized: TokenizedSpell) -> int:
        """Get how many times this exact spell has been cast."""
        canon = _canonical_hash(tokenized.token_ids)
        return self.cast_counts.get(canon, 0)

    def approaching_crystal(self, tokenized: TokenizedSpell) -> float:
        """Return progress toward crystallization [0, 1]."""
        canon = _canonical_hash(tokenized.token_ids)
        count = self.cast_counts.get(canon, 0)
        if canon in self.crystals:
            return 1.0
        return min(count / self.threshold, 1.0)

    def list_crystals(self) -> list[CrystalEntry]:
        """Return all crystallized spells sorted by cast count."""
        return sorted(self.crystals.values(), key=lambda c: c.cast_count, reverse=True)

    # ── Persistence ──

    def save(self):
        """Save registry to disk."""
        self.save_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "threshold": self.threshold,
            "cast_counts": self.cast_counts,
            "cast_tokens": {k: v for k, v in self.cast_tokens.items()},
            "crystals": {
                canon: {
                    "crystal_id": c.crystal_id,
                    "original_tokens": c.original_tokens,
                    "cast_count": c.cast_count,
                    "name": c.name,
                }
                for canon, c in self.crystals.items()
            },
            "next_id": self._next_id,
        }
        torch.save(state, Path(self.save_dir, "registry.pt"))

        # Save embeddings separately
        for canon, crystal in self.crystals.items():
            if crystal.embedding is not None:
                torch.save(crystal.embedding, self.save_dir / f"emb_{crystal.crystal_id}.pt")

    def load(self) -> bool:
        """Load registry from disk. Returns True if successful."""
        path = Path(self.save_dir, "registry.pt")
        if not path.exists():
            return False

        state = torch.load(path, weights_only=False)
        self.threshold = state["threshold"]
        self.cast_counts = state["cast_counts"]
        self.cast_tokens = state["cast_tokens"]
        self._next_id = state["next_id"]

        self.crystals = {}
        for canon, data in state["crystals"].items():
            emb_path = self.save_dir / f"emb_{data['crystal_id']}.pt"
            embedding = torch.load(emb_path, weights_only=True) if emb_path.exists() else None
            self.crystals[canon] = CrystalEntry(
                crystal_id=data["crystal_id"],
                original_tokens=data["original_tokens"],
                embedding=embedding,
                cast_count=data["cast_count"],
                name=data["name"],
            )
        return True
