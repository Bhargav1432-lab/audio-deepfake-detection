"""
ablation_v2.py

Second, more reliable version of the ablation study. Changes vs ablation.py:

  1. Several random seeds per variant (default 3), reported as mean / std /
     min / max. One run per variant turned out to be too noisy to trust.
  2. The best checkpoint is chosen by DEV EER (lower is better), not dev
     accuracy. Dev accuracy barely changed between epochs, so it could not
     tell a good checkpoint from a bad one.
  3. A "full_fixed" variant (all 4 branches, plain concatenation, no gate) is
     included, so the core question "does the reliability gate help?" is
     answered with the same training recipe and several seeds.
  4. Class balance of the train / dev / eval manifests is printed at the start.

Everything else follows train_multiview.py (10 epochs, lr 1e-3, batch 8,
entropy weight 0.5 for gated models, gate floor 0.10).

Checkpoints and score files go to ablation_v2/. Your existing .pth files and
ablation results are not touched. Finished (variant, seed) runs are appended to
ablation_v2_results.csv and skipped on the next run, so you can stop with
Ctrl+C and resume.

Usage:
    python ablation_v2.py --all                     (all variants, 3 seeds)
    python ablation_v2.py --all --seeds 1234 42     (choose seeds)
    python ablation_v2.py --variant full_fixed --seed 42
    python ablation_v2.py --summary
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
ENTROPY_WEIGHT = 0.5
DEFAULT_SEEDS = [1234, 42, 7]
OUT_DIR = "ablation_v2"
RESULTS_CSV = "ablation_v2_results.csv"

ALL_BRANCHES = ["magnitude", "phase", "residual", "prosody"]

# name -> (branches, mode); mode is "gated", "concat" or "single"
VARIANTS = {
    "full_adaptive": (ALL_BRANCHES, "gated"),
    "full_fixed": (ALL_BRANCHES, "concat"),
}
for b in ALL_BRANCHES:
    VARIANTS[f"drop_{b}"] = ([x for x in ALL_BRANCHES if x != b], "gated")
for b in ALL_BRANCHES:
    VARIANTS[f"single_{b}"] = ([b], "single")


class AblationModel(nn.Module):
    def __init__(self, branches, mode, embed_dim=EMBED_DIM, num_classes=2):
        super().__init__()
        self.branches = branches
        self.mode = mode if len(branches) > 1 else "single"
        self.encoders = nn.ModuleDict()
        for b in branches:
            if b == "prosody":
                self.encoders[b] = ProsodyEncoder(out_dim=embed_dim)
            else:
                self.encoders[b] = SpectrogramEncoder(embed_dim)
        if self.mode == "gated":
            self.gate = ReliabilityGate(embed_dim, num_branches=len(branches))
        in_dim = embed_dim * len(branches) if self.mode == "concat" else embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    @property
    def gated(self):
        return self.mode == "gated"

    def forward(self, batch, return_weights=False):
        embs = [self.encoders[b](batch[b]) for b in self.branches]
        weights = None
        if self.mode == "gated":
            weights = self.gate(torch.cat(embs, dim=1))
            fused = (torch.stack(embs, dim=1) * weights.unsqueeze(-1)).sum(dim=1)
        elif self.mode == "concat":
            fused = torch.cat(embs, dim=1)
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


def safe_eer(y_true, scores):
    if not np.isfinite(scores).all() or len(set(y_true)) < 2:
        return 100.0
    return compute_eer(y_true, scores)


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


def print_balance():
    print("Class balance of the manifests:")
    for name, path in [("train", TRAIN_MANIFEST), ("dev", DEV_MANIFEST), ("eval2021", EVAL_MANIFEST)]:
        d = pd.read_csv(path)
        counts = d["label"].value_counts().to_dict()
        print(f"  {name:9s} {len(d):5d} files   {counts}")
    print()


def run(name, seed, device):
    branches, mode = VARIANTS[name]
    print(f"\n{'=' * 66}\n{name}  seed={seed}  branches={branches}  mode={mode}\n{'=' * 66}")
    os.makedirs(OUT_DIR, exist_ok=True)

    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = DataLoader(CachedMultiViewDataset(TRAIN_MANIFEST), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, generator=gen)
    dev_loader = DataLoader(CachedMultiViewDataset(DEV_MANIFEST), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)
    eval_loader = DataLoader(CachedMultiViewDataset(EVAL_MANIFEST), batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AblationModel(branches, mode).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    ckpt = os.path.join(OUT_DIR, f"{name}_seed{seed}.pth")
    best_dev_eer, best_epoch = float("inf"), 0
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
        dev_eer = safe_eer(dev_labels, dev_scores)
        dev_acc = float(((dev_scores > 0).astype(int) == dev_labels).mean())
        print(f"Epoch {epoch}/{EPOCHS} | train_loss={total_loss / seen:.4f} "
              f"train_acc={correct / seen:.4f} | dev_acc={dev_acc:.4f} dev_EER={dev_eer:.2f}% | "
              f"{time.time() - t_epoch:.0f}s")
        if dev_eer < best_dev_eer:
            best_dev_eer, best_epoch = dev_eer, epoch
            torch.save(model.state_dict(), ckpt)
            print("  -> new best dev EER, saved")

    model.load_state_dict(torch.load(ckpt, map_location=device))
    scores, labels, weights = evaluate(model, eval_loader, device)
    if not np.isfinite(scores).all():
        print("WARNING: non-finite scores on eval set!")

    eval_acc = float(((scores > 0).astype(int) == labels).mean())
    eval_eer = safe_eer(labels, scores)
    eval_auc = roc_auc_score(labels, scores) if np.isfinite(scores).all() else float("nan")

    manifest = pd.read_csv(EVAL_MANIFEST).reset_index(drop=True)
    manifest["score"] = scores
    manifest["true_label"] = labels
    manifest.to_csv(os.path.join(OUT_DIR, f"{name}_seed{seed}_scores.csv"), index=False)

    row = {
        "variant": name, "seed": seed, "mode": mode,
        "best_epoch": best_epoch, "best_dev_eer": best_dev_eer,
        "eval_acc": eval_acc, "eval_eer": eval_eer, "eval_auc": eval_auc,
        "minutes": (time.time() - t_start) / 60,
    }
    for b in ALL_BRANCHES:
        row[f"w_{b}"] = float(weights[branches.index(b)]) if (weights is not None and b in branches) else np.nan
    pd.DataFrame([row]).to_csv(RESULTS_CSV, mode="a", header=not os.path.exists(RESULTS_CSV), index=False)

    print(f"\n>>> {name} seed={seed}: best epoch {best_epoch}  eval_acc={eval_acc:.4f}  "
          f"EER={eval_eer:.2f}%  AUC={eval_auc:.4f}")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def done_runs():
    if os.path.exists(RESULTS_CSV):
        d = pd.read_csv(RESULTS_CSV)
        return set(zip(d["variant"], d["seed"]))
    return set()


def summary():
    if not os.path.exists(RESULTS_CSV):
        print("No results yet.")
        return
    res = pd.read_csv(RESULTS_CSV)
    g = res.groupby("variant")
    table = g.agg(
        runs=("seed", "count"),
        eer_mean=("eval_eer", "mean"), eer_std=("eval_eer", "std"),
        eer_min=("eval_eer", "min"), eer_max=("eval_eer", "max"),
        acc_mean=("eval_acc", "mean"), auc_mean=("eval_auc", "mean"),
        w_mag=("w_magnitude", "mean"), w_pha=("w_phase", "mean"),
        w_res=("w_residual", "mean"), w_pro=("w_prosody", "mean"),
    )
    table = table.reindex([v for v in VARIANTS if v in table.index])
    print(table.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nEER in %, lower is better. Reference (single official model): Baseline AASIST EER 9.57%.")
    print("\nPer-run EER (%) so you can see the spread:")
    piv = res.pivot_table(index="variant", columns="seed", values="eval_eer")
    piv = piv.reindex([v for v in VARIANTS if v in piv.index])
    print(piv.to_string(float_format=lambda x: f"{x:.2f}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    if args.summary:
        summary()
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print_balance()

    if args.variant:
        run(args.variant, args.seed if args.seed is not None else args.seeds[0], device)
    elif args.all:
        for seed in args.seeds:  # seed-major: every variant gets seed 1 before any gets seed 2
            for name in VARIANTS:
                if (name, seed) in done_runs():
                    print(f"Skipping {name} seed={seed} (already done)")
                    continue
                run(name, seed, device)
    else:
        parser.print_help()
        return

    print("\n")
    summary()


if __name__ == "__main__":
    main()
