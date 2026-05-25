"""
BAUD Learnable: Meta-trainable deviation weighting.

The key contribution: instead of hand-crafted AU weights,
learn which AU deviations predict pain across subjects,
then apply those weights to new patients with zero labels.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class DeviationWeightNet(nn.Module):
    """
    Learns per-AU deviation importance from z-score vectors.
    Replaces hand-crafted prior weights with data-driven ones.
    """
    def __init__(self, num_aus=41, hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_aus, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_scores: torch.Tensor) -> torch.Tensor:
        """z_scores: (batch, num_aus) → pain_logits: (batch, 1)"""
        return self.net(z_scores)


class TemporalGRU(nn.Module):
    """GRU temporal aggregation over z-score sequences."""
    def __init__(self, num_aus=41, hidden=32):
        super().__init__()
        self.gru = nn.GRU(num_aus, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, num_aus)

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        """z_seq: (batch, seq_len, num_aus) → (batch, seq_len, num_aus)"""
        out, _ = self.gru(z_seq)
        return self.proj(out)


class BAUDLearnable(nn.Module):
    """
    Full meta-learnable BAUD model.
    
    Architecture:
        z-scores → [TemporalGRU] → [DeviationWeightNet] → pain_score
    
    Training:
        Meta-train on labeled subjects: learn which AU deviations predict pain.
    
    Inference on new patient:
        1. Calibrate: compute baseline mean/std from neutral frames (zero labels)
        2. Score: compute z-scores → apply learned weights → pain score
    """
    def __init__(self, num_aus=41, hidden=64, gru_hidden=32, dropout=0.2):
        super().__init__()
        self.num_aus = num_aus
        self.temporal = TemporalGRU(num_aus, gru_hidden)
        self.weight_net = DeviationWeightNet(num_aus, hidden, dropout)

    def compute_z_scores(self, au_seq, baseline_mean, baseline_std, eps=1e-4):
        """
        Compute per-AU z-scores from AU sequence and patient baseline.
        
        Args:
            au_seq: (batch, seq_len, num_aus)
            baseline_mean: (batch, num_aus)
            baseline_std: (batch, num_aus)
        Returns:
            z_positive: (batch, seq_len, num_aus)
        """
        mean = baseline_mean.unsqueeze(1)
        std = baseline_std.unsqueeze(1)
        z = (au_seq - mean) / (std + eps)
        return torch.clamp(z, min=0)

    def forward(self, z_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_scores: (batch, seq_len, num_aus)
        Returns:
            pain_scores: (batch, seq_len) in [0, 1]
        """
        # Temporal aggregation
        aggregated = self.temporal(z_scores)
        
        # Per-frame deviation weighting
        B, T, N = aggregated.shape
        flat = aggregated.reshape(B * T, N)
        logits = self.weight_net(flat)
        scores = torch.sigmoid(logits).reshape(B, T)
        return scores

    def score_subject(self, neutral_aus, pain_aus, eps=1e-4):
        """
        End-to-end: calibrate on neutral, score pain.
        Both inputs are tensors: (seq_len, num_aus)
        Returns pain_scores: (pain_seq_len,)
        """
        # Calibrate from neutral (no labels)
        baseline_mean = neutral_aus.mean(dim=0, keepdim=True)  # (1, num_aus)
        baseline_std = neutral_aus.std(dim=0, keepdim=True).clamp(min=eps)

        # Compute z-scores for pain frames
        pain_batch = pain_aus.unsqueeze(0)  # (1, seq_len, num_aus)
        z = self.compute_z_scores(pain_batch, baseline_mean, baseline_std)

        # Score
        scores = self.forward(z)  # (1, seq_len)
        return scores.squeeze(0)


class MetaBAUDLoss(nn.Module):
    """
    Combined loss for meta-training BAUD.
    
    L = L_pain + λ_rank * L_rank + λ_temporal * L_temporal
    
    - L_pain: BCE between predicted and true pain labels
    - L_rank: margin ranking loss (pain scores > neutral scores)
    - L_temporal: smoothness penalty on adjacent frame scores
    """
    def __init__(self, lambda_rank=0.5, lambda_temporal=0.1):
        super().__init__()
        self.lambda_rank = lambda_rank
        self.lambda_temporal = lambda_temporal
        self.bce = nn.BCELoss()

    def forward(self, pain_scores, neutral_scores, pain_labels=None):
        """
        Args:
            pain_scores: (n_pain_frames,) predicted pain scores for pain frames
            neutral_scores: (n_neutral_frames,) predicted scores for neutral frames
            pain_labels: optional (n_pain_frames,) ground truth intensity
        """
        # BCE loss: pain frames should score high, neutral should score low
        pain_target = torch.ones_like(pain_scores)
        neutral_target = torch.zeros_like(neutral_scores)
        all_scores = torch.cat([pain_scores, neutral_scores])
        all_targets = torch.cat([pain_target, neutral_target])
        l_pain = self.bce(all_scores, all_targets)

        # Ranking loss: mean pain score should exceed mean neutral score
        pain_mean = pain_scores.mean()
        neutral_mean = neutral_scores.mean()
        margin = 0.3
        l_rank = F.relu(neutral_mean - pain_mean + margin)

        # Temporal smoothness on pain scores
        if len(pain_scores) > 1:
            diffs = (pain_scores[1:] - pain_scores[:-1]).pow(2)
            l_temporal = diffs.mean()
        else:
            l_temporal = torch.tensor(0.0, device=pain_scores.device)

        total = l_pain + self.lambda_rank * l_rank + self.lambda_temporal * l_temporal
        return total, {
            "total": total.item(),
            "bce": l_pain.item(),
            "rank": l_rank.item(),
            "temporal": l_temporal.item(),
        }
