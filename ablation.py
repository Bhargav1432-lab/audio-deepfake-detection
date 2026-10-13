"""
ablation.py

Ablation study for the multi-view fusion model. Trains variants with the
SAME recipe as train_multiview.py (10 epochs, lr 1e-3, batch 8, seed 1234,
entropy weight 0.5, gate floor 0.10) and evaluates each on the 585
ASVspoof2021 LA samples.

Variants:
  full_adaptive_retrain  all 4 branches + gate (control: checks run-to-run noise)
  drop_<branch>          3 branches + gate  (what is lost when one branch is removed)
  single_<branch>        1 branch, no gate  (how good is each branch alone)

Checkpoints go to ablation/<variant>.pth, so your existing fixed_fusion.pth
and adaptive_fusion.pth are NOT touched. Each finished variant is appended to
ablation_results.csv; variants already in that file are skipped, so you can
stop (Ctrl+C) and resume later.

Usage:
    python ablation.py --variant drop_phase     (one variant)
    python ablation.py --all                    (everything, resumable)
    python ablation.py --summary                (print the results table)
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score

from multiview_model import (
    CachedMultiViewDataset, SpectrogramEncoder, ProsodyEncoder,
    ReliabilityGate, EMBED_DIM,
)

TRAIN_MANIFEST = "feature_cache_manifest_train.csv"
DEV_MANIFEST = "feature_cache_manifest_dev.csv"
EVAL_MANIFEST = "feature_cache_manifest_eval.csv"
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-3
SEED = 1234
ENTROPY_WEIGHT = 0.5
OUT_DIR = "ablation"
RESULTS_CSV = "ablation_results.csv"

ALL_BRANCHES = ["magnitude", "phase", "residual", "prosody"]

VARIANTS = {"full_adaptive_retrain": (ALL_BRANCHES, True)}
for b in ALL_BRANCHES:
    VARIANTS[f"drop_{b}"] = ([x for x in ALL_BRANCHES if x != b], True)
for b in ALL_BRANCHES:
    VARIANTS[f"single_{b}"] = ([b], False)


class AblationModel(nn.Module):
    def __init__(self, branches, gated, embed_dim=EMBED_DIM, num_classes=2):
        super().__init__()
        self.branches = branches
        self.gated = gated and len(branches) > 1
        self.encoders = nn.ModuleDict()
        for b in branches:
            if b == "prosody":
                self.encoders[b] = ProsodyEncoder(out_dim=embed_dim)
            else:
                self.encoders[b] = SpectrogramEncoder(embed_dim)
        if self.gated:
            self.gate = ReliabilityGate(embed_dim, num_branches=len(branches))
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, batch, return_weights=False):
        embs = [self.encoders[b](batch[b]) for b in self.branches]
        weights = None
        if self.gated:
            weights = self.gate(torch.cat(embs, dim=1))
            fused = (torch.stack(embs, dim=1) * weights.unsqueeze(-1)).sum(dim=1)
        else:
            fused = embs[0]
        out = self.classifier(fused)
        if return_weights:
            return out, weights
        return out


def compute_eer(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return 100 * (fpr[i] + fnr[i]) / 2


def to_device(batch, device):
    inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
    return inputs, batch["label"].to(device)


def evaluate(model, loader, device):
    model.eval()
    scores, labels, all_w = [], [], []
    with torch.no_grad():
        for batch in loader:
            inputs, y = to_device(batch, device)
            out, w = model(inputs, return_weights=True)
            scores.extend((out[:, 1] - out[:, 0]).cpu().numpy())
            labels.extend(y.cpu().numpy())
            if w is not None:
                all_w.append(w.cpu().numpy())
    weights = np.concatenate(all_w).mean(axis=0) if all_w else None
    return np.array(scores), np.array(labels), weights


def run_variant(name, device):
    branches, gated = VARIANTS[name]
    print(f"\n{'=' * 64}\nVariant: {name}   branches={branches}   gated={gated}\n{'=' * 64}")
    os.makedirs(OUT_DIR, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_loader = DataLoader(CachedMultiViewDataset(TRAIN_MANIFEST),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dev_loader = DataLoader(CachedMultiViewDataset(DEV_MANIFEST),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    eval_loader = DataLoader(CachedMultiViewDataset(EVAL_MANIFEST),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = AblationModel(branches, gated).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    ckpt = os.path.join(OUT_DIR, f"{name}.pth")
    best_dev = 0.0
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_epoch = time.time()
        total_loss, correct, seen = 0.0, 0, 0
        for batch in train_loader:
            inputs, y = to_device(batch, device)
            if model.gated:
                out, w = model(inputs, return_weights=True)
                entropy = -(w * torch.log(w + 1e-8)).sum(dim=1).mean()
                loss = criterion(out, y) - ENTROPY_WEIGHT * entropy
            else:
                out = model(inputs)
                loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            seen += y.size(0)

        dev_scores, dev_labels, _ = evaluate(model, dev_loader, device)
        dev_acc = float(((dev_scores > 0).astype(int) == dev_labels).mean())
        print(f"Epoch {epoch}/{EPOCHS} | train_loss={total_loss / seen:.4f} "
              f"train_acc={correct / seen:.4f} | dev_acc={dev_acc:.4f} | "
              f"{time.time() - t_epoch:.0f}s")
        if dev_acc > best_dev:
            best_dev = dev_acc
            torch.save(model.state_dict(), ckpt)
            print("  -> new best, saved")

    # evaluate the BEST checkpoint (by dev accuracy) on the 2021 eval set
    model.load_state_dict(torch.load(ckpt, map_location=device))
    scores, labels, weights = evaluate(model, eval_loader, device)
    if not np.isfinite(scores).all():
        print("WARNING: non-finite scores found on eval set!")

    eval_acc = float(((scores > 0).astype(int) == labels).mean())
    eval_eer = compute_eer(labels, scores)
    eval_auc = roc_auc_score(labels, scores)

    manifest = pd.read_csv(EVAL_MANIFEST).reset_index(drop=True)
    manifest["score"] = scores
    manifest["true_label"] = labels
    manifest.to_csv(os.path.join(OUT_DIR, f"{name}_scores.csv"), index=False)

    row = {
        "variant": name,
        "branches": "+".join(branches),
        "gated": gated,
        "best_dev_acc": best_dev,
        "eval_acc": eval_acc,
        "eval_eer": eval_eer,
        "eval_auc": eval_auc,
        "train_minutes": (time.time() - t_start) / 60,
    }
    for b in ALL_BRANCHES:
        row[f"w_{b}"] = float(weights[branches.index(b)]) if (weights is not None and b in branches) else np.nan

    df_row = pd.DataFrame([row])
    df_row.to_csv(RESULTS_CSV, mode="a", header=not os.path.exists(RESULTS_CSV), index=False)

    print(f"\n>>> {name}: eval_acc={eval_acc:.4f}  EER={eval_eer:.2f}%  AUC={eval_auc:.4f}")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def done_variants():
    if os.path.exists(RESULTS_CSV):
        return set(pd.read_csv(RESULTS_CSV)["variant"])
    return set()


def summary():
    rows = []
    if os.path.exists("eer_merged_scores.csv"):
        d = pd.read_csv("eer_merged_scores.csv")
        y = (d["label"] == "bonafide").astype(int).values
        for name, col in [("[ref] Baseline AASIST", "baseline_score"),
                          ("[ref] Fixed Fusion", "fixed_score"),
                          ("[ref] Adaptive Fusion", "adaptive_score")]:
            if col in d.columns:
                s = d[col].values
                rows.append({"variant": name,
                             "eval_acc": ((s > 0).astype(int) == y).mean(),
                             "eval_eer": compute_eer(y, s),
                             "eval_auc": roc_auc_score(y, s)})
    if os.path.exists(RESULTS_CSV):
        res = pd.read_csv(RESULTS_CSV)
        order = {v: i for i, v in enumerate(VARIANTS)}
        res = res.sort_values("variant", key=lambda c: c.map(order))
        rows.extend(res.to_dict("records"))
    if not rows:
        print("No results yet.")
        return
    cols = ["variant", "eval_acc", "eval_eer", "eval_auc", "best_dev_acc",
            "w_magnitude", "w_phase", "w_residual", "w_prosody"]
    out = pd.DataFrame(rows)
    out = out[[c for c in cols if c in out.columns]]
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n(eval_eer is in %, lower is better; w_* = average gate weight on the eval set)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    if args.summary:
        summary()
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.variant:
        run_variant(args.variant, device)
    elif args.all:
        for name in VARIANTS:
            if name in done_variants():
                print(f"Skipping {name} (already in {RESULTS_CSV})")
                continue
            run_variant(name, device)
    else:
        parser.print_help()
        return

    print("\n")
    summary()


if __name__ == "__main__":
    main()
