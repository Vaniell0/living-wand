"""SpellFormer — small Transformer that learns magic from examples.

Architecture: 4-layer Transformer encoder with hybrid positional encoding
(absolute + structural role + graph degree). Attention-pooled output maps
to a 38-dim effect vector.

~738K parameters. Inference <15ms on i5 CPU.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import (
    BASE_VOCAB_SIZE,
    CRYSTAL_OFFSET,
    MAX_CRYSTALS,
    MAX_SEQ_LEN,
    SpecialToken,
    VOCAB_SIZE,
)
from .tags import SEMANTIC_DIM

# Output dimensions
NUM_POLES = 6
NUM_ELEMENTS = 6
NUM_DOMAINS = 6
EFFECT_DIM = NUM_POLES + 2 + 2 + NUM_ELEMENTS + NUM_DOMAINS + SEMANTIC_DIM
# = 6 + 2 + 2 + 6 + 6 + 16 = 38


class SinusoidalPosEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = MAX_SEQ_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pe[:, :seq_len]


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-norm attention
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + attn_out
        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


class SpellFormer(nn.Module):
    """Transformer model that learns magic as mathematics.

    Input: tokenized spell sequence (token_ids, role_ids, degree)
    Output: 38-dim effect vector

    The model is the shared "world's magic knowledge" — same weights
    for all players. Personal wand bias is applied by WandAdapter on top.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 384,
        dropout: float = 0.1,
        n_roles: int = 4,
        max_degree: int = 9,
        role_dim: int = 16,
        degree_dim: int = 16,
    ):
        super().__init__()
        self.d_model = d_model

        # Token embedding (base vocab + crystal slots)
        self.token_embed = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=SpecialToken.PAD)

        # Positional encoding (sinusoidal — not learned, saves params)
        self.pos_enc = SinusoidalPosEncoding(d_model, MAX_SEQ_LEN)

        # Structural role embedding (meta / node / separator / edge)
        self.role_embed = nn.Embedding(n_roles, role_dim)

        # Graph degree embedding (how many connections a rune has)
        self.degree_embed = nn.Embedding(max_degree, degree_dim)

        # Project concatenated embeddings to d_model
        self.input_proj = nn.Linear(d_model + role_dim + degree_dim, d_model)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final norm
        self.final_norm = nn.LayerNorm(d_model)

        # Attention pooling (learned weights over sequence → single vector)
        self.pool_query = nn.Linear(d_model, 1)

        # Output head: d_model → EFFECT_DIM
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, EFFECT_DIM),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for linear layers, normal for embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])

    def encode(self, token_ids: torch.Tensor, role_ids: torch.Tensor,
               degree: torch.Tensor) -> torch.Tensor:
        """Encode token sequence to hidden representations.

        Args:
            token_ids: (batch, seq_len) int
            role_ids:  (batch, seq_len) int
            degree:    (batch, seq_len) int

        Returns:
            (batch, seq_len, d_model) hidden states
        """
        batch_size, seq_len = token_ids.shape

        # Embeddings
        tok_emb = self.token_embed(token_ids)             # (B, S, d_model)
        pos_emb = self.pos_enc(seq_len)                    # (1, S, d_model)
        role_emb = self.role_embed(role_ids)                # (B, S, role_dim)
        deg_emb = self.degree_embed(degree.clamp(0, 8))    # (B, S, degree_dim)

        # Combine: token + position, then concatenate role + degree
        combined = tok_emb + pos_emb
        combined = torch.cat([combined, role_emb, deg_emb], dim=-1)  # (B, S, d+r+d)
        h = self.input_proj(combined)  # (B, S, d_model)

        # Padding mask (True = ignore)
        pad_mask = token_ids == SpecialToken.PAD  # (B, S)

        # Transformer blocks
        for block in self.blocks:
            h = block(h, key_padding_mask=pad_mask)

        h = self.final_norm(h)
        return h

    def pool(self, h: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """Attention-weighted pooling over non-pad tokens.

        Args:
            h: (batch, seq_len, d_model) hidden states
            token_ids: (batch, seq_len) for padding mask

        Returns:
            (batch, d_model) pooled representation
        """
        # Compute attention weights
        scores = self.pool_query(h).squeeze(-1)  # (B, S)

        # Mask padding positions
        pad_mask = token_ids == SpecialToken.PAD
        scores = scores.masked_fill(pad_mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)  # (B, S)
        pooled = torch.bmm(weights.unsqueeze(1), h).squeeze(1)  # (B, d_model)
        return pooled

    def forward(self, token_ids: torch.Tensor, role_ids: torch.Tensor,
                degree: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full forward pass: tokens → effect vector.

        Args:
            token_ids: (batch, seq_len) int
            role_ids:  (batch, seq_len) int
            degree:    (batch, seq_len) int

        Returns:
            dict with:
              raw:                (batch, EFFECT_DIM) raw logits
              pole_distribution:  (batch, 6) softmax
              power:              (batch, 1) sigmoid [0,1]
              stability:          (batch, 1) sigmoid [0,1]
              area:               (batch, 1) softplus
              duration:           (batch, 1) softplus
              element_intensity:  (batch, 6) sigmoid [0,1]
              domain_intensity:   (batch, 6) sigmoid [0,1]
              semantic_embedding: (batch, 16) tanh [-1,1]
        """
        h = self.encode(token_ids, role_ids, degree)
        pooled = self.pool(h, token_ids)
        raw = self.output_head(pooled)  # (B, 38)

        # Split and activate
        poles = F.softmax(raw[:, :6], dim=-1)
        power = torch.sigmoid(raw[:, 6:7])
        stability = torch.sigmoid(raw[:, 7:8])
        area = F.softplus(raw[:, 8:9])
        duration = F.softplus(raw[:, 9:10])
        elem_int = torch.sigmoid(raw[:, 10:16])
        dom_int = torch.sigmoid(raw[:, 16:22])
        semantic = torch.tanh(raw[:, 22:38])

        return {
            "raw": raw,
            "pole_distribution": poles,
            "power": power,
            "stability": stability,
            "area": area,
            "duration": duration,
            "element_intensity": elem_int,
            "domain_intensity": dom_int,
            "semantic_embedding": semantic,
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def set_crystal_embedding(self, crystal_id: int, embedding: torch.Tensor):
        """Set the embedding for a crystallized spell token.

        Args:
            crystal_id: 0-based crystal index (token_id = CRYSTAL_OFFSET + crystal_id)
            embedding: (d_model,) vector from attention-pooled encoding
        """
        token_id = CRYSTAL_OFFSET + crystal_id
        with torch.no_grad():
            self.token_embed.weight[token_id] = embedding
