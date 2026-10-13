"""
train_attribution.py

NOVELTY 1: Generator Attribution Classifier

Loads the pretrained, frozen AASIST model as a feature extractor (we never
update its weights), and trains a small new classifier head on top of its
160-dim embeddings to predict WHICH system generated a given audio clip:

    bonafide, A01, A02, A03, A04, A05, A06   (7 classes)

Requires:
    - attribution_dataset.csv (built earlier by build_attribution_dataset.py)
    - models/weights/AASIST.pth (pretrained weights)
    - config/AASIST.conf (for model architecture hyperparameters)

Usage (run from your aasist folder):
    python train_attribution.py
"""

import json
import random
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from importlib import import_module

# ---------------- CONFIG ----------------
CONFIG_PATH = "config/AASIST.conf"
AASIST_WEIGHTS = "models/weights/AASIST.pth"
CSV_PATH = "attribution_dataset.csv"
NB_SAMP = 64600          # same fixed-length audio window AASIST expects
BATCH_SIZE = 16          # smaller than default 24 to be safe on 4GB VRAM
EPOCHS = 10
LR = 1e-3
EMBED_DIM = 160          # 5 * gat_dims[1] (5 * 32) from AASIST's last_hidden
HIDDEN_DIM = 64
SEED = 1234
MODEL_SAVE_PATH = "attribution_head.pth"
# -----------------------------------------

LABELS = ["bonafide", "A01", "A02", "A03", "A04", "A05", "A06"]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_or_trim(audio, target_len=NB_SAMP):
    """AASIST expects a fixed-length input. Repeat-pad short clips, trim long ones."""
    if len(audio) >= target_len:
        return audio[:target_len]
    num_repeats = int(target_len / len(audio)) + 1
    padded = np.tile(audio, num_repeats)[:target_len]
    return padded


class AttributionDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio, _ = sf.read(row["filepath"])
        audio = pad_or_trim(audio.astype(np.float32))
        label_idx = LABEL_TO_IDX[row["label"]]
        return torch.tensor(audio, dtype=torch.float32), label_idx


class AttributionHead(nn.Module):
    """Small trainable classifier that sits on top of frozen AASIST embeddings."""
    def __init__(self, in_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_classes=len(LABELS)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_frozen_aasist(model_config, device):
    module = import_module("models.AASIST")
    Model = getattr(module, "Model")
    model = Model(model_config).to(device)
    model.load_state_dict(torch.load(AASIST_WEIGHTS, map_location=device))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    model_config = config["model_config"]

    print("Loading frozen AASIST encoder...")
    aasist = load_frozen_aasist(model_config, device)

    print("Loading attribution dataset...")
    df = pd.read_csv(CSV_PATH)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    dev_df = df[df["split"] == "dev"].reset_index(drop=True)
    print(f"Train samples: {len(train_df)} | Dev samples: {len(dev_df)}")

    train_loader = DataLoader(AttributionDataset(train_df), batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=2)
    dev_loader = DataLoader(AttributionDataset(dev_df), batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2)

    head = AttributionHead().to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_dev_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        head.train()
        total_loss, total_correct, total_seen = 0.0, 0, 0

        for batch_idx, (audio, labels) in enumerate(train_loader):
            audio, labels = audio.to(device), labels.to(device)

            with torch.no_grad():
                embeddings, _ = aasist(audio)

            logits = head(embeddings)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * audio.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_seen += audio.size(0)

            if batch_idx % 100 == 0:
                print(f"  epoch {epoch} batch {batch_idx}/{len(train_loader)} "
                      f"loss={loss.item():.4f}")

        train_loss = total_loss / total_seen
        train_acc = total_correct / total_seen

        # ---- validation ----
        head.eval()
        dev_correct, dev_seen = 0, 0
        with torch.no_grad():
            for audio, labels in dev_loader:
                audio, labels = audio.to(device), labels.to(device)
                embeddings, _ = aasist(audio)
                logits = head(embeddings)
                dev_correct += (logits.argmax(dim=1) == labels).sum().item()
                dev_seen += audio.size(0)
        dev_acc = dev_correct / dev_seen

        print(f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} | dev_acc={dev_acc:.4f}")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(head.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> New best dev accuracy ({dev_acc:.4f}). Saved to {MODEL_SAVE_PATH}")

    print(f"\nTraining complete. Best dev accuracy: {best_dev_acc:.4f}")
    print(f"Best model saved at: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
