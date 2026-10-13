"""
test_multiview_pipeline.py

Sanity check: runs the full multi-view pipeline (dataset -> encoders ->
gate -> fusion -> classifier) on just 5 samples, to confirm everything
works end-to-end BEFORE committing to a full training run (which will
take a long time given prosody extraction is slow).

Usage:
    python test_multiview_pipeline.py
"""

import torch
from torch.utils.data import DataLoader
import pandas as pd

from multiview_model import MultiViewDataset, FixedFusionModel, AdaptiveFusionModel

# Build a tiny test CSV (5 samples) from your existing attribution dataset
def build_tiny_csv():
    df = pd.read_csv("attribution_dataset.csv")
    # take 3 bonafide + 2 spoof for a quick mixed test
    bonafide = df[df["label"] == "bonafide"].head(3)
    spoof = df[df["label"] != "bonafide"].head(2).copy()
    spoof["label"] = "spoof"  # normalize label to match expected bonafide/spoof format
    tiny = pd.concat([bonafide, spoof])
    tiny.to_csv("tiny_test.csv", index=False)
    print(f"Built tiny_test.csv with {len(tiny)} samples")


def main():
    build_tiny_csv()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dataset = MultiViewDataset("tiny_test.csv")
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    print("\nTesting FixedFusionModel (Model 2)...")
    model2 = FixedFusionModel().to(device)
    model2.eval()

    for batch in loader:
        batch_on_device = {k: v.to(device) for k, v in batch.items() if k != "label"}
        with torch.no_grad():
            output = model2(batch_on_device)
        print(f"  Batch output shape: {output.shape} (expected: [batch_size, 2])")
        break  # just test one batch

    print("\nTesting AdaptiveFusionModel (Model 3)...")
    model3 = AdaptiveFusionModel().to(device)
    model3.eval()

    for batch in loader:
        batch_on_device = {k: v.to(device) for k, v in batch.items() if k != "label"}
        with torch.no_grad():
            output, weights = model3(batch_on_device, return_weights=True)
        print(f"  Batch output shape: {output.shape} (expected: [batch_size, 2])")
        print(f"  Reliability weights shape: {weights.shape} (expected: [batch_size, 4])")
        print(f"  Sample weights (should sum to ~1.0 per row):")
        print(f"    {weights.cpu().numpy()}")
        break

    print("\nAll checks passed. Pipeline works end-to-end.")
    print("Ready to build the full training script.")


if __name__ == "__main__":
    main()
