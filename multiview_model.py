"""
multiview_model.py

Contains:
  1. MultiViewDataset -- PyTorch Dataset that returns all 4 feature views
     per audio sample, ready for batching.
  2. Four small encoder networks (Magnitude, Phase, Residual: CNNs;
     Prosody: MLP) -- each outputs a fixed-size embedding.
  3. ReliabilityGate -- learns how much to trust each of the 4 embeddings
     for a given input.
  4. Three full models for comparison:
       - FixedFusionModel   (Model 2: naive concatenation)
       - AdaptiveFusionModel (Model 3: the actual novelty)

Note: Model 1 (baseline) is your EXISTING AASIST model -- no new code
needed for that, reuse models/AASIST.py directly.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from feature_extractors import extract_all_features

EMBED_DIM = 64  # each branch outputs a 64-dim embedding


class MultiViewDataset(Dataset):
    """
    Reads a CSV with columns: filepath, label (bonafide/spoof)
    Returns all 4 feature views + label for each sample.

    WARNING: this extracts features LIVE (slow, ~15-30 sec/file due to
    prosody extraction). For training, use CachedMultiViewDataset instead,
    which loads pre-computed features from disk (fast).
    """
    def __init__(self, csv_path, label_column="label"):
        self.df = pd.read_csv(csv_path)
        self.label_column = label_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = extract_all_features(row["filepath"])

        label_str = row[self.label_column]
        label = 1 if label_str == "bonafide" else 0

        return {
            "magnitude": torch.tensor(features["magnitude"]),
            "phase": torch.tensor(features["phase"]),
            "residual": torch.tensor(features["residual"]),
            "prosody": torch.tensor(features["prosody"]),
            "label": torch.tensor(label, dtype=torch.long),
        }


class CachedMultiViewDataset(Dataset):
    """
    FAST dataset for training: loads pre-computed features from .npz
    cache files (built by cache_features.py) instead of extracting them
    live. Use this for all real training runs.
    """
    def __init__(self, manifest_csv):
        self.df = pd.read_csv(manifest_csv)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = np.load(row["cache_path"])

        label = 1 if row["label"] == "bonafide" else 0

        return {
            "magnitude": torch.tensor(data["magnitude"]),
            "phase": torch.tensor(data["phase"]),
            "residual": torch.tensor(data["residual"]),
            "prosody": torch.tensor(data["prosody"]),
            "label": torch.tensor(label, dtype=torch.long),
        }


class SpectrogramEncoder(nn.Module):
    """Small CNN for 2D spectrogram-like inputs (magnitude, phase, residual)."""
    def __init__(self, out_dim=EMBED_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),  # fixed output size regardless of input size
        )
        self.fc = nn.Linear(32 * 4 * 4, out_dim)

    def forward(self, x):
        # x shape: (batch, freq, time) -> add channel dim
        x = x.unsqueeze(1)  # (batch, 1, freq, time)
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


class ProsodyEncoder(nn.Module):
    """Small MLP for prosody features (3, N_TIME_FRAMES)."""
    def __init__(self, in_channels=3, time_frames=126, out_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * time_frames, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReliabilityGate(nn.Module):
    """
    Takes the 4 branch embeddings, concatenates them, and outputs 4
    weights (summing to 1 via softmax) indicating how much to trust
    each branch for THIS specific input.
    """
    def __init__(self, embed_dim=EMBED_DIM, num_branches=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * num_branches, 32),
            nn.ReLU(),
            nn.Linear(32, num_branches),
        )

    def forward(self, embeddings_concat):
        logits = self.net(embeddings_concat)
        raw_weights = torch.softmax(logits, dim=1)  # shape: (batch, 4)

        # FLOOR: guarantee every branch keeps at least ~10% influence,
        # preventing full collapse onto a single branch. The remaining
        # 60% is distributed according to what the gate actually learns,
        # so genuine input-dependent preference can still emerge within
        # this range. Math: each branch gets floor_per_branch as a base,
        # plus a share of the remaining budget proportional to what the
        # gate wants. This still sums to 1.0 across the 4 branches.
        num_branches = raw_weights.size(1)
        floor_per_branch = 0.10
        remaining_budget = 1.0 - (floor_per_branch * num_branches)  # 1.0 - 0.4 = 0.6
        weights = floor_per_branch + remaining_budget * raw_weights
        return weights


class MultiViewBackbone(nn.Module):
    """Shared backbone: the 4 encoders. Used by both Model 2 and Model 3."""
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.mag_encoder = SpectrogramEncoder(embed_dim)
        self.phase_encoder = SpectrogramEncoder(embed_dim)
        self.residual_encoder = SpectrogramEncoder(embed_dim)
        self.prosody_encoder = ProsodyEncoder(out_dim=embed_dim)

    def forward(self, batch):
        mag_emb = self.mag_encoder(batch["magnitude"])
        phase_emb = self.phase_encoder(batch["phase"])
        res_emb = self.residual_encoder(batch["residual"])
        pros_emb = self.prosody_encoder(batch["prosody"])
        return mag_emb, phase_emb, res_emb, pros_emb


class FixedFusionModel(nn.Module):
    """MODEL 2: naive fusion -- just concatenate all 4 embeddings equally."""
    def __init__(self, embed_dim=EMBED_DIM, num_classes=2):
        super().__init__()
        self.backbone = MultiViewBackbone(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, batch):
        mag_emb, phase_emb, res_emb, pros_emb = self.backbone(batch)
        fused = torch.cat([mag_emb, phase_emb, res_emb, pros_emb], dim=1)
        return self.classifier(fused)


class AdaptiveFusionModel(nn.Module):
    """MODEL 3 (the proposed novelty): reliability-gated adaptive fusion."""
    def __init__(self, embed_dim=EMBED_DIM, num_classes=2):
        super().__init__()
        self.backbone = MultiViewBackbone(embed_dim)
        self.gate = ReliabilityGate(embed_dim, num_branches=4)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, batch, return_weights=False):
        mag_emb, phase_emb, res_emb, pros_emb = self.backbone(batch)
        concat = torch.cat([mag_emb, phase_emb, res_emb, pros_emb], dim=1)
        weights = self.gate(concat)  # (batch, 4)

        # weighted sum of the 4 embeddings using the learned reliability weights
        stacked = torch.stack([mag_emb, phase_emb, res_emb, pros_emb], dim=1)  # (batch, 4, embed_dim)
        weights_expanded = weights.unsqueeze(-1)  # (batch, 4, 1)
        fused = (stacked * weights_expanded).sum(dim=1)  # (batch, embed_dim)

        output = self.classifier(fused)

        if return_weights:
            return output, weights
        return output
