"""Multi-task loss for SpellFormer training.

Loss components:
  1. Pole distribution: KL divergence (it's a probability distribution)
  2. Power + stability: MSE (bounded [0,1])
  3. Area + duration: Huber/SmoothL1 (can have outliers)
  4. Element/domain intensities: BCE (independent [0,1])
  5. Semantic embedding: cosine similarity
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpellLoss(nn.Module):
    """Weighted multi-task loss for SpellFormer."""

    def __init__(
        self,
        w_pole: float = 2.0,
        w_scalar: float = 1.5,
        w_extent: float = 1.0,
        w_intensity: float = 1.0,
        w_semantic: float = 0.5,
    ):
        super().__init__()
        self.w_pole = w_pole
        self.w_scalar = w_scalar
        self.w_extent = w_extent
        self.w_intensity = w_intensity
        self.w_semantic = w_semantic

    def forward(self, pred_raw: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute loss from raw model output and target effect vector.

        Args:
            pred_raw: (batch, 38) raw logits from SpellFormer
            target:   (batch, 38) ground truth effect vector

        Returns:
            dict with 'total' and per-component losses
        """
        # 1. Pole distribution (KL divergence)
        pred_poles = F.log_softmax(pred_raw[:, :6], dim=-1)
        tgt_poles = target[:, :6].clamp(min=1e-8)  # avoid log(0)
        tgt_poles = tgt_poles / tgt_poles.sum(dim=-1, keepdim=True)  # ensure normalized
        pole_loss = F.kl_div(pred_poles, tgt_poles, reduction="batchmean")

        # 2. Power + stability (MSE, bounded [0,1])
        pred_scalar = torch.sigmoid(pred_raw[:, 6:8])
        tgt_scalar = target[:, 6:8]
        scalar_loss = F.mse_loss(pred_scalar, tgt_scalar)

        # 3. Area + duration (Huber loss for robustness)
        pred_extent = F.softplus(pred_raw[:, 8:10])
        tgt_extent = target[:, 8:10]
        extent_loss = F.smooth_l1_loss(pred_extent, tgt_extent)

        # 4. Element + domain intensities (BCE)
        pred_intensity = pred_raw[:, 10:22]  # logits, BCE applies sigmoid
        tgt_intensity = target[:, 10:22].clamp(0, 1)
        intensity_loss = F.binary_cross_entropy_with_logits(pred_intensity, tgt_intensity)

        # 5. Semantic embedding (cosine similarity)
        pred_sem = torch.tanh(pred_raw[:, 22:38])
        tgt_sem = target[:, 22:38]
        cos_sim = F.cosine_similarity(pred_sem, tgt_sem, dim=-1)
        semantic_loss = (1 - cos_sim).mean()

        # Weighted total
        total = (
            self.w_pole * pole_loss
            + self.w_scalar * scalar_loss
            + self.w_extent * extent_loss
            + self.w_intensity * intensity_loss
            + self.w_semantic * semantic_loss
        )

        return {
            "total": total,
            "pole": pole_loss,
            "scalar": scalar_loss,
            "extent": extent_loss,
            "intensity": intensity_loss,
            "semantic": semantic_loss,
        }
