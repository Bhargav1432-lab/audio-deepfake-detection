"""
models/advanced_multiview.py

Advanced Multi-View Audio Forensics Architecture:
1. SE-ResNet Spectrogram Encoders:
   - Residual convolutions with Squeeze-and-Excitation (SE) channel attention
     for Magnitude, Phase, and Residual views.
2. 1D Dilated ConvNet for Prosody features with SE attention.
3. Frequency & Cross-Modal Reliability Gating (CrossModalAttentionGate):
   - Dynamic cross-modal attention replacing simple scalar softmax gating.
   - Learns input-dependent and channel-dependent reliability weights.
4. AdvancedAdaptiveFusionModel:
   - Combines SE-ResNet representations through Cross-Modal Attention Gating
     for high-performance anti-spoofing and deepfake detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 64


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel recalibration."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced_channels = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, reduced_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduced_channels, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, _, _ = x.size()
        y = x.mean(dim=(2, 3))  # Global average pooling -> (B, C)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class SEResBlock(nn.Module):
    """Residual block with Squeeze-and-Excitation attention."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return self.relu(out)


class SEResNetSpectrogramEncoder(nn.Module):
    """
    SE-ResNet Encoder for 2D representations (Magnitude, Phase, LPC Residual).
    Replaces basic 2-layer CNN with deep residual representations and SE attention.
    """
    def __init__(self, out_dim=EMBED_DIM):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),  # (B, 16, 513, 63)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.layer1 = SEResBlock(16, 32, stride=2)   # (B, 32, 257, 32)
        self.layer2 = SEResBlock(32, 64, stride=2)   # (B, 64, 129, 16)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))     # (B, 64, 2, 2)
        self.fc = nn.Linear(64 * 2 * 2, out_dim)

    def forward(self, x):
        # x: (batch, freq, time)
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


class ProsodySEEncoder(nn.Module):
    """
    1D ConvNet with SE channel recalibration for prosody contours (F0, RMS, voiced prob).
    """
    def __init__(self, in_channels=3, out_dim=EMBED_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(4)
        )
        self.fc = nn.Linear(64 * 4, out_dim)

    def forward(self, x):
        # x: (batch, 3, time)
        feat = self.conv(x).flatten(1)
        return self.fc(feat)


class CrossModalAttentionGate(nn.Module):
    """
    Cross-Modal Attention Gate:
    Instead of a single scalar softmax weight, each branch acts as query, key,
    and value to attend to other forensic views, computing dynamic inter-modal
    trust while ensuring a 10% floor to prevent branch collapse.
    """
    def __init__(self, embed_dim=EMBED_DIM, num_branches=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_branches = num_branches

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.gate_scoring = nn.Sequential(
            nn.Linear(embed_dim * num_branches, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_branches)
        )

    def forward(self, branch_embeddings):
        # branch_embeddings: list of 4 tensors, each (B, embed_dim)
        stacked = torch.stack(branch_embeddings, dim=1)  # (B, 4, embed_dim)
        concat = torch.cat(branch_embeddings, dim=1)     # (B, 4 * embed_dim)

        # 1. Global reliability distribution with 10% floor
        raw_logits = self.gate_scoring(concat)
        soft_weights = torch.softmax(raw_logits, dim=1)  # (B, 4)
        floor = 0.10
        rem = 1.0 - (floor * self.num_branches)
        branch_weights = floor + rem * soft_weights      # (B, 4)

        # 2. Cross-modal self-attention refinement
        q = self.q_proj(stacked)  # (B, 4, D)
        k = self.k_proj(stacked)  # (B, 4, D)
        v = self.v_proj(stacked)  # (B, 4, D)

        scores = torch.bmm(q, k.transpose(1, 2)) / (self.embed_dim ** 0.5)  # (B, 4, 4)
        attn = torch.softmax(scores, dim=-1)
        refined_v = torch.bmm(attn, v)  # (B, 4, D)

        # 3. Weight-guided aggregation
        weights_exp = branch_weights.unsqueeze(-1)  # (B, 4, 1)
        fused = (refined_v * weights_exp).sum(dim=1)  # (B, D)

        return fused, branch_weights


class AdvancedAdaptiveFusionModel(nn.Module):
    """
    Proposed Next-Gen Architecture:
    - 3x SE-ResNet Encoders (Magnitude, Phase, Residual)
    - 1x Prosody SE-Conv Encoder
    - Cross-Modal Attention Reliability Gating
    """
    def __init__(self, embed_dim=EMBED_DIM, num_classes=2):
        super().__init__()
        self.mag_encoder = SEResNetSpectrogramEncoder(embed_dim)
        self.phase_encoder = SEResNetSpectrogramEncoder(embed_dim)
        self.residual_encoder = SEResNetSpectrogramEncoder(embed_dim)
        self.prosody_encoder = ProsodySEEncoder(in_channels=3, out_dim=embed_dim)

        self.gate = CrossModalAttentionGate(embed_dim=embed_dim, num_branches=4)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, batch, return_weights=False):
        mag_emb = self.mag_encoder(batch["magnitude"])
        phase_emb = self.phase_encoder(batch["phase"])
        res_emb = self.residual_encoder(batch["residual"])
        pros_emb = self.prosody_encoder(batch["prosody"])

        branches = [mag_emb, phase_emb, res_emb, pros_emb]
        fused, weights = self.gate(branches)
        out = self.classifier(fused)

        if return_weights:
            return out, weights
        return out
