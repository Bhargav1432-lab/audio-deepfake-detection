"""
evaluate_attribution.py

Two things:
1. Confusion matrix on the DEV set (A01-A06 + bonafide) -- shows exactly
   which known systems get confused with which.
2. Generalization test on the EVAL set (A07-A19 -- systems the classifier
   has NEVER seen). It obviously can't predict these labels correctly since
   they're not in its output space, but we can measure:
     - Is it at least confident the clip is NOT bonafide (i.e. correctly
       flags it as some kind of fake)?
     - Which of the 6 known classes does it end up guessing for each
       unseen system? (reveals which known technique the unseen one
       "resembles" most, from the model's point of view)

Requires:
    - attribution_head.pth      (trained by train_attribution.py)
    - models/weights/AASIST.pth (frozen encoder weights)
    - config/AASIST.conf
    - LA/ dataset (dev + eval protocol files)

Usage:
    python evaluate_attribution.py
"""

import json
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from importlib import import_module
from pathlib import Path

# ---------------- CONFIG ----------------
CONFIG_PATH = "config/AASIST.conf"
AASIST_WEIGHTS = "models/weights/AASIST.pth"
ATTRIBUTION_WEIGHTS = "attribution_head.pth"
NB_SAMP = 64600
BATCH_SIZE = 16
EMBED_DIM = 160
HIDDEN_DIM = 64

EVAL_PROTOCOL = Path("LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt")
EVAL_AUDIO_DIR = Path("LA/ASVspoof2019_LA_eval/flac")
# -----------------------------------------

LABELS = ["bonafide", "A01", "A02", "A03", "A04", "A05", "A06"]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
IDX_TO_LABEL = {i: label for label, i in LABEL_TO_IDX.items()}

UNSEEN_SYSTEM_INFO = {
    "A07": "TTS - vocoder+GAN", "A08": "TTS - neural waveform",
    "A09": "TTS - vocoder", "A10": "TTS - neural waveform",
    "A11": "TTS - griffin lim", "A12": "TTS - neural waveform",
    "A13": "TTS_VC - concat+filter", "A14": "TTS_VC - vocoder",
    "A15": "TTS_VC - neural waveform", "A16": "TTS - concatenation",
    "A17": "VC - waveform filtering", "A18": "VC - vocoder",
    "A19": "VC - spectral filtering",
}


def pad_or_trim(audio, target_len=NB_SAMP):
    if len(audio) >= target_len:
        return audio[:target_len]
    num_repeats = int(target_len / len(audio)) + 1
    return np.tile(audio, num_repeats)[:target_len]


class SimpleAudioDataset(Dataset):
    """Generic dataset: filepath + arbitrary string label (not necessarily in LABELS)."""
    def __init__(self, filepaths, raw_labels):
        self.filepaths = filepaths
        self.raw_labels = raw_labels

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        audio, _ = sf.read(self.filepaths[idx])
        audio = pad_or_trim(audio.astype(np.float32))
        return torch.tensor(audio, dtype=torch.float32), self.raw_labels[idx]


class AttributionHead(nn.Module):
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


def load_models(device):
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    model_config = config["model_config"]

    module = import_module("models.AASIST")
    Model = getattr(module, "Model")
    aasist = Model(model_config).to(device)
    aasist.load_state_dict(torch.load(AASIST_WEIGHTS, map_location=device))
    aasist.eval()
    for p in aasist.parameters():
        p.requires_grad = False

    head = AttributionHead().to(device)
    head.load_state_dict(torch.load(ATTRIBUTION_WEIGHTS, map_location=device))
    head.eval()

    return aasist, head


def run_dev_confusion_matrix(aasist, head, device):
    print("\n" + "=" * 60)
    print("PART 1: Confusion matrix on DEV set (known systems A01-A06)")
    print("=" * 60)

    df = pd.read_csv("attribution_dataset.csv")
    dev_df = df[df["split"] == "dev"].reset_index(drop=True)

    filepaths = dev_df["filepath"].tolist()
    raw_labels = [LABEL_TO_IDX[l] for l in dev_df["label"].tolist()]

    loader = DataLoader(SimpleAudioDataset(filepaths, raw_labels),
                         batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    num_classes = len(LABELS)
    confusion = np.zeros((num_classes, num_classes), dtype=int)

    with torch.no_grad():
        for audio, true_idx in loader:
            audio = audio.to(device)
            embeddings, _ = aasist(audio)
            logits = head(embeddings)
            pred_idx = logits.argmax(dim=1).cpu().numpy()
            true_idx = true_idx.numpy()
            for t, p in zip(true_idx, pred_idx):
                confusion[t, p] += 1

    # Print confusion matrix
    print("\nRows = true label, Columns = predicted label\n")
    header = "        " + "".join(f"{l:>8}" for l in LABELS)
    print(header)
    for i, label in enumerate(LABELS):
        row = "".join(f"{confusion[i, j]:>8}" for j in range(num_classes))
        print(f"{label:>8}{row}")

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for i, label in enumerate(LABELS):
        total = confusion[i].sum()
        correct = confusion[i, i]
        acc = correct / total if total > 0 else 0
        print(f"  {label}: {acc:.4f} ({correct}/{total})")

    overall_acc = np.trace(confusion) / confusion.sum()
    print(f"\nOverall dev accuracy: {overall_acc:.4f}")


def run_unseen_system_analysis(aasist, head, device, max_per_system=200):
    print("\n" + "=" * 60)
    print("PART 2: Generalization test on UNSEEN systems (A07-A19)")
    print("=" * 60)

    if not EVAL_PROTOCOL.exists():
        print(f"WARNING: {EVAL_PROTOCOL} not found, skipping this part.")
        return

    # Parse eval protocol, group by system, sample up to max_per_system each
    system_to_files = {}
    with open(EVAL_PROTOCOL, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, filename, _, system_id, key = parts
            if key == "bonafide":
                continue  # skip real speech here, we only care about unseen fakes
            system_to_files.setdefault(system_id, []).append(filename)

    results = {}

    for system_id in sorted(system_to_files.keys()):
        files = system_to_files[system_id][:max_per_system]
        filepaths = [str(EVAL_AUDIO_DIR / f"{fn}.flac") for fn in files]
        dummy_labels = [0] * len(filepaths)  # not used for unseen systems

        loader = DataLoader(SimpleAudioDataset(filepaths, dummy_labels),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        pred_counts = np.zeros(len(LABELS), dtype=int)
        not_bonafide_count = 0
        total = 0

        with torch.no_grad():
            for audio, _ in loader:
                audio = audio.to(device)
                embeddings, _ = aasist(audio)
                logits = head(embeddings)
                probs = torch.softmax(logits, dim=1)
                pred_idx = logits.argmax(dim=1).cpu().numpy()

                for p in pred_idx:
                    pred_counts[p] += 1
                    if p != LABEL_TO_IDX["bonafide"]:
                        not_bonafide_count += 1
                total += audio.size(0)

        correctly_flagged_fake_rate = not_bonafide_count / total if total > 0 else 0
        most_confused_with = LABELS[pred_counts.argmax()]

        results[system_id] = {
            "total": total,
            "flagged_as_fake_rate": correctly_flagged_fake_rate,
            "most_predicted_as": most_confused_with,
            "pred_distribution": pred_counts,
        }

        info = UNSEEN_SYSTEM_INFO.get(system_id, "unknown")
        print(f"\n{system_id} ({info}) -- {total} samples tested")
        print(f"  Correctly flagged as FAKE (any non-bonafide class): {correctly_flagged_fake_rate:.4f}")
        print(f"  Most often predicted as: {most_confused_with}")
        dist_str = ", ".join(f"{LABELS[i]}={pred_counts[i]}" for i in range(len(LABELS)))
        print(f"  Full distribution: {dist_str}")

    avg_flagged_rate = np.mean([r["flagged_as_fake_rate"] for r in results.values()])
    print(f"\n{'=' * 60}")
    print(f"Average 'flagged as fake' rate across all unseen systems: {avg_flagged_rate:.4f}")
    print("(This tells you: even though the model can't name these new systems,")
    print(" how often does it still correctly sense something is fake?)")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    aasist, head = load_models(device)

    run_dev_confusion_matrix(aasist, head, device)
    run_unseen_system_analysis(aasist, head, device, max_per_system=200)


if __name__ == "__main__":
    main()
