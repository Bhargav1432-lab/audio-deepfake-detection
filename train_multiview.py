"""
train_multiview.py

Trains BOTH Model 2 (Fixed Fusion) and Model 3 (Adaptive Fusion) using
pre-cached features (built by cache_features.py). Reports accuracy for
both, so you can directly compare whether the reliability gate actually
helps -- this is your core experimental result.

Usage:
    python train_multiview.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

from multiview_model import CachedMultiViewDataset, FixedFusionModel, AdaptiveFusionModel

# ---------------- CONFIG ----------------
TRAIN_MANIFEST = "feature_cache_manifest_train.csv"
DEV_MANIFEST = "feature_cache_manifest_dev.csv"
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-3
SEED = 1234
# -----------------------------------------


def move_batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items() if k != "label"}, batch["label"].to(device)


def train_one_model(model, model_name, train_loader, dev_loader, device, use_entropy_reg=False):
    print(f"\n{'=' * 60}")
    print(f"Training {model_name}")
    print(f"{'=' * 60}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    ENTROPY_WEIGHT = 0.5  # increased from 0.1 -- previous value was too weak to prevent collapse

    best_dev_acc = 0.0
    save_path = f"{model_name.lower().replace(' ', '_')}.pth"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_correct, total_seen = 0.0, 0, 0
        start = time.time()

        for batch in train_loader:
            inputs, labels = move_batch_to_device(batch, device)

            if use_entropy_reg:
                output, weights = model(inputs, return_weights=True)
                # entropy of the weight distribution -- HIGH entropy means
                # weights are spread across branches (good, diverse).
                # LOW entropy means collapsed onto one branch (bad).
                # We SUBTRACT entropy from loss (i.e. reward high entropy)
                # to discourage collapse.
                eps = 1e-8
                entropy = -(weights * torch.log(weights + eps)).sum(dim=1).mean()
                classification_loss = criterion(output, labels)
                loss = classification_loss - ENTROPY_WEIGHT * entropy
            else:
                output = model(inputs)
                if isinstance(output, tuple):
                    output = output[0]
                loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if isinstance(output, tuple):
                pred_output = output[0]
            else:
                pred_output = output

            total_loss += loss.item() * labels.size(0)
            total_correct += (pred_output.argmax(dim=1) == labels).sum().item()
            total_seen += labels.size(0)

        train_loss = total_loss / total_seen
        train_acc = total_correct / total_seen

        # validation
        model.eval()
        dev_correct, dev_seen = 0, 0
        with torch.no_grad():
            for batch in dev_loader:
                inputs, labels = move_batch_to_device(batch, device)
                output = model(inputs)
                if isinstance(output, tuple):
                    output = output[0]
                dev_correct += (output.argmax(dim=1) == labels).sum().item()
                dev_seen += labels.size(0)
        dev_acc = dev_correct / dev_seen if dev_seen > 0 else 0

        elapsed = time.time() - start
        print(f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} | dev_acc={dev_acc:.4f} | {elapsed:.0f}s")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best. Saved to {save_path}")

    print(f"\n{model_name} best dev accuracy: {best_dev_acc:.4f}")
    return best_dev_acc


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading cached datasets...")
    train_dataset = CachedMultiViewDataset(TRAIN_MANIFEST)
    dev_dataset = CachedMultiViewDataset(DEV_MANIFEST)
    print(f"Train samples: {len(train_dataset)} | Dev samples: {len(dev_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ---- Model 2: Fixed Fusion ----
    model2 = FixedFusionModel().to(device)
    acc2 = train_one_model(model2, "Fixed Fusion", train_loader, dev_loader, device)

    # ---- Model 3: Adaptive Fusion (the proposed novelty) ----
    model3 = AdaptiveFusionModel().to(device)
    acc3 = train_one_model(model3, "Adaptive Fusion", train_loader, dev_loader, device,
                            use_entropy_reg=True)

    print(f"\n{'=' * 60}")
    print("FINAL COMPARISON")
    print(f"{'=' * 60}")
    print(f"Fixed Fusion (Model 2):    {acc2:.4f}")
    print(f"Adaptive Fusion (Model 3): {acc3:.4f}")
    print(f"Improvement from adaptive gating: {acc3 - acc2:+.4f}")


if __name__ == "__main__":
    main()
