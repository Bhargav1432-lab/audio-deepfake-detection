"""
train_augmented_multiview.py

Domain-Invariant Acoustic Augmented Training:
Injects real-world acoustic variations directly into training:
  1. Additive Spectral Noise Floor (simulating noisy room mics / Kathbath noise floor)
  2. Bandwidth & Telephony Roll-off (simulating mobile phone 3.4 kHz cutoff)
  3. Dynamic Range Jitter (simulating varied recording gain and microphone sensitivity)
  4. Prosody Energy Jitter (preventing overfitting to studio pitch tracking)

Objective:
  Bridge the acoustic domain gap so that English-trained models generalize
  to noisy real-world recordings (Tamil, Telugu, and in-the-wild mobile speech)
  without mistaking background noise for synthetic artifacts.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

from multiview_model import AdaptiveFusionModel, FixedFusionModel

TRAIN_MANIFEST = "feature_cache_manifest_train.csv"
DEV_MANIFEST = "feature_cache_manifest_dev.csv"
EVAL_MANIFEST = "feature_cache_manifest_eval.csv"
INDIC_MANIFEST = "feature_cache_manifest_indic.csv"

BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3
ENTROPY_WEIGHT = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "checkpoints"
os.makedirs(OUT_DIR, exist_ok=True)


def compute_eer(y, s):
    y, s = np.asarray(y), np.asarray(s)
    if len(set(y)) < 2 or not np.isfinite(s).all():
        return np.nan
    fpr, tpr, _ = roc_curve(y, s, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return 100.0 * (fpr[idx] + fnr[idx]) / 2.0


class AugmentedCachedDataset(Dataset):
    """
    Loads pre-computed features from .npz and applies domain-shift augmentations on the fly.
    """
    def __init__(self, manifest_csv, augment=False, p_aug=0.6):
        self.df = pd.read_csv(manifest_csv)
        self.augment = augment
        self.p_aug = p_aug

    def __len__(self):
        return len(self.df)

    def apply_domain_augmentations(self, mag, phase, res, pros):
        if not self.augment or np.random.rand() > self.p_aug:
            return mag, phase, res, pros

        # 1. Additive Spectral Noise Floor:
        # ASVspoof clean audio has silence floors at -80 dB, while mobile phones have floors at -50 to -35 dB.
        # Adding noise floor in linear power then converting back prevents noise from being treated as spoof!
        noise_level_db = np.random.uniform(-55.0, -35.0)
        mag_power = 10.0 ** (mag / 10.0)
        noise_power = 10.0 ** (noise_level_db / 10.0)
        mag_power_noisy = mag_power + noise_power * np.random.exponential(scale=1.0, size=mag.shape)
        mag_aug = 10.0 * np.log10(np.maximum(mag_power_noisy, 1e-8))
        mag_aug = np.clip(mag_aug, -80.0, 0.0)

        # Apply similar floor to LPC residual
        res_power = 10.0 ** (res / 10.0)
        res_power_noisy = res_power + noise_power * np.random.exponential(scale=1.0, size=res.shape)
        res_aug = 10.0 * np.log10(np.maximum(res_power_noisy, 1e-8))
        res_aug = np.clip(res_aug, -80.0, 0.0)

        # 2. Bandwidth / High-Frequency Roll-off:
        # Simulate mobile / telephony bandwidth limiting (cut above ~3.4-4 kHz, bins > 430 out of 1025)
        if np.random.rand() < 0.5:
            cutoff_bin = np.random.randint(380, 512)
            mag_aug[cutoff_bin:, :] -= np.random.uniform(20.0, 40.0)
            res_aug[cutoff_bin:, :] -= np.random.uniform(20.0, 40.0)

        # 3. Prosody Energy Jitter (varied mic gain)
        pros_aug = pros.copy()
        gain_scale = np.random.uniform(0.7, 1.3)
        pros_aug[1, :] *= gain_scale  # RMS channel

        # Phase stays as unwrapped diff, with slight random phase noise
        phase_aug = phase + np.random.normal(0.0, 0.05, size=phase.shape).astype(np.float32)

        return mag_aug.astype(np.float32), phase_aug.astype(np.float32), res_aug.astype(np.float32), pros_aug.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = np.load(row["cache_path"])

        mag = data["magnitude"]
        phase = data["phase"]
        res = data["residual"]
        pros = data["prosody"]

        if self.augment:
            mag, phase, res, pros = self.apply_domain_augmentations(mag, phase, res, pros)

        label = 1 if row["label"] == "bonafide" else 0

        return {
            "magnitude": torch.tensor(mag),
            "phase": torch.tensor(phase),
            "residual": torch.tensor(res),
            "prosody": torch.tensor(pros),
            "label": torch.tensor(label, dtype=torch.long),
        }


def evaluate(model, loader):
    model.eval()
    all_scores = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "label"}
            labels = batch["label"].cpu().numpy()
            out = model(inputs)
            if isinstance(out, tuple):
                out = out[0]
            scores = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()

            all_scores.extend(scores)
            all_labels.extend(labels)
            all_preds.extend(preds)

    y = np.array(all_labels)
    s = np.array(all_scores)
    p = np.array(all_preds)

    acc = accuracy_score(y, p) * 100.0
    eer = compute_eer(y, s)
    auc = roc_auc_score(y, s) if len(set(y)) > 1 else 0.5
    return acc, eer, auc, s, y


def train():
    print("=" * 75)
    print("      DOMAIN-INVARIANT ACOUSTIC AUGMENTED TRAINING PIPELINE      ")
    print("=" * 75)
    print(f"Device: {DEVICE}")

    train_ds = AugmentedCachedDataset(TRAIN_MANIFEST, augment=True, p_aug=0.7)
    dev_ds = AugmentedCachedDataset(DEV_MANIFEST, augment=False)
    eval_ds = AugmentedCachedDataset(EVAL_MANIFEST, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = AdaptiveFusionModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    ce_loss = nn.CrossEntropyLoss()

    best_dev_eer = 999.0
    best_weights_path = os.path.join(OUT_DIR, "adaptive_fusion_augmented.pth")

    print("\nStarting Training (10 Epochs with Acoustic Domain Augmentations)...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "label"}
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits, weights = model(inputs, return_weights=True)
            loss_cls = ce_loss(logits, labels)

            # Entropy regularization to keep gate balanced
            entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()
            loss = loss_cls - ENTROPY_WEIGHT * entropy

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        dev_acc, dev_eer, dev_auc, _, _ = evaluate(model, dev_loader)
        eval_acc, eval_eer, eval_auc, _, _ = evaluate(model, eval_loader)

        print(f"Epoch [{epoch:2d}/{EPOCHS}] Loss: {train_loss/len(train_loader):.4f} | "
              f"Dev EER: {dev_eer:5.2f}% (Acc: {dev_acc:5.2f}%) | "
              f"Eval 2021 Acc: {eval_acc:5.2f}% (EER: {eval_eer:5.2f}%)")

        if dev_eer < best_dev_eer:
            best_dev_eer = dev_eer
            torch.save(model.state_dict(), best_weights_path)

    print(f"\n[+] Best checkpoint saved to: {best_weights_path} (Dev EER: {best_dev_eer:.2f}%)")

    # Evaluate on Indic Dataset
    print("\n" + "=" * 75)
    print("          EVALUATING ZERO-SHOT CROSS-LINGUAL GENERALIZATION (INDIC)          ")
    print("=" * 75)

    model.load_state_dict(torch.load(best_weights_path, map_location=DEVICE))

    indic_df = pd.read_csv(INDIC_MANIFEST)
    print(f"Loaded {len(indic_df)} Indic test samples (Tamil & Telugu).")

    indic_ds = AugmentedCachedDataset(INDIC_MANIFEST, augment=False)
    indic_loader = DataLoader(indic_ds, batch_size=BATCH_SIZE, shuffle=False)

    acc, eer, auc, scores, labels = evaluate(model, indic_loader)
    indic_df["pred_score"] = scores
    indic_df["true_label"] = labels

    print(f"\n--- OVERALL ZERO-SHOT INDIC RESULTS (AFTER ACOUSTIC AUGMENTATION) ---")
    print(f"  Overall Accuracy: {acc:.2f}%")
    print(f"  Overall EER:      {eer:.2f}%")
    print(f"  Overall ROC-AUC:  {auc:.4f}")

    # Breakdown by Language
    results_summary = []
    for lang in ["Tamil", "Telugu"]:
        sub = indic_df[indic_df["language"] == lang]
        sub_y = sub["true_label"].values
        sub_s = sub["pred_score"].values
        sub_acc = accuracy_score(sub_y, (sub_s >= 0.5).astype(int)) * 100.0
        sub_eer = compute_eer(sub_y, sub_s)
        sub_auc = roc_auc_score(sub_y, sub_s)

        print(f"\n  [{lang.upper()}]:")
        print(f"    - Accuracy: {sub_acc:.2f}%")
        print(f"    - EER:      {sub_eer:.2f}%")
        print(f"    - ROC-AUC:  {sub_auc:.4f}")

        results_summary.append({
            "Language": lang,
            "Accuracy": sub_acc,
            "EER": sub_eer,
            "AUC": sub_auc
        })

    # Save results
    out_csv = "augmented_indic_results.csv"
    pd.DataFrame(results_summary).to_csv(out_csv, index=False)
    print(f"\n[+] Results saved to: {out_csv}")
    print("=" * 75)


if __name__ == "__main__":
    train()
