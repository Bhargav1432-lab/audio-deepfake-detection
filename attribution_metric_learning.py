"""
attribution_metric_learning.py

Open-Set Generator Attribution with Metric Learning (ArcFace & CosFace).
Allows forensic models to:
1. Accurately attribute known generator systems (bonafide + A01 - A06).
2. Tightly cluster known generators in hyperspherical angular space.
3. Reliably detect Out-of-Distribution / Unseen generators (A07 - A19) as 'Unknown Fake'
   via cosine similarity thresholding.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score


class ArcMarginProduct(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss head.
    Reference: Deng et al., 'ArcFace: Additive Angular Margin Loss for Deep Face Recognition', CVPR 2019.
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.35, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # 1. Cosine similarity: cos(theta) = <x_norm, w_norm>
        cosine = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

        if label is None:
            # Inference mode: return scaled cosine similarity logits
            return cosine * self.s, cosine

        # 2. Angular margin penalty during training
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7, max=1.0))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot mask for ground-truth class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output, cosine


class CosMarginProduct(nn.Module):
    """
    CosFace: Large Margin Cosine Loss head.
    Reference: Wang et al., 'CosFace: Large Margin Cosine Loss for Deep Face Recognition', CVPR 2018.
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.30):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label=None):
        cosine = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if label is None:
            return cosine * self.s, cosine

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
        output = self.s * (cosine - one_hot * self.m)
        return output, cosine


class OpenSetAttributionHead(nn.Module):
    """
    Forensic metric learning head wrapping an encoder embedding with ArcFace/CosFace.
    """
    def __init__(self, embed_dim=160, num_known_classes=7, metric_type='arcface', s=30.0, m=0.35):
        super().__init__()
        self.embed_proj = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Linear(128, 64)
        )
        if metric_type == 'arcface':
            self.head = ArcMarginProduct(64, num_known_classes, s=s, m=m)
        else:
            self.head = CosMarginProduct(64, num_known_classes, s=s, m=m)

    def forward(self, x, labels=None):
        feat = self.embed_proj(x)
        logits, raw_cos = self.head(feat, labels)
        return logits, raw_cos, feat

    @torch.no_grad()
    def predict_open_set(self, x, threshold=0.30):
        """
        Classifies known classes or rejects as 'Unseen / Novel Deepfake Generator'.
        Returns:
            pred_class: integer index in [0..num_known-1] or -1 (rejected as unseen OOD)
            max_sim: maximum cosine similarity to any known generator class
        """
        feat = self.embed_proj(x)
        norm_w = F.normalize(self.head.weight, p=2, dim=1)
        norm_f = F.normalize(feat, p=2, dim=1)
        sim = torch.mm(norm_f, norm_w.t())  # (B, num_known)
        max_sim, best_class = torch.max(sim, dim=1)

        # If similarity is below threshold, flag as unknown/novel generator (-1)
        is_known = max_sim >= threshold
        pred_class = torch.where(is_known, best_class, torch.tensor(-1, device=x.device))
        return pred_class, max_sim
