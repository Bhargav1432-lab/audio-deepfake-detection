"""
evaluate_on_2021.py

THE CORE EXPERIMENT: evaluates both trained models (Fixed Fusion vs
Adaptive Fusion) on ASVspoof2021 LA data -- which has REAL codec and
transmission-channel variation, unlike the clean 2019 LA data they were
trained on.

This is where we test the actual hypothesis: does adaptive reliability
gating hold up better than fixed fusion when the acoustic channel changes?

Reports:
  1. Overall accuracy for both models on 2021 LA
  2. Breakdown by codec (alaw, opus, gsm, etc.)
  3. Breakdown by channel (ita_tx, sin_tx, loc_tx, etc.)

Usage:
    python evaluate_on_2021.py
"""

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from multiview_model import CachedMultiViewDataset, FixedFusionModel, AdaptiveFusionModel

MANIFEST_PATH = "feature_cache_manifest_eval.csv"
BATCH_SIZE = 8


def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            labels = batch["label"]

            output = model(inputs)
            if isinstance(output, tuple):
                output = output[0]

            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading manifest...")
    manifest_df = pd.read_csv(MANIFEST_PATH)
    print(f"Total 2021 LA samples: {len(manifest_df)}")

    dataset = CachedMultiViewDataset(MANIFEST_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("\nLoading trained models...")
    fixed_model = FixedFusionModel().to(device)
    fixed_model.load_state_dict(torch.load("fixed_fusion.pth", map_location=device))

    adaptive_model = AdaptiveFusionModel().to(device)
    adaptive_model.load_state_dict(torch.load("adaptive_fusion.pth", map_location=device))

    print("\nEvaluating Fixed Fusion on 2021 LA...")
    fixed_preds, labels = evaluate_model(fixed_model, loader, device)
    fixed_acc = (fixed_preds == labels).mean()

    print("Evaluating Adaptive Fusion on 2021 LA...")
    adaptive_preds, _ = evaluate_model(adaptive_model, loader, device)
    adaptive_acc = (adaptive_preds == labels).mean()

    print(f"\n{'=' * 60}")
    print("OVERALL RESULTS ON ASVSPOOF2021 LA (channel-degraded audio)")
    print(f"{'=' * 60}")
    print(f"Fixed Fusion accuracy:    {fixed_acc:.4f}")
    print(f"Adaptive Fusion accuracy: {adaptive_acc:.4f}")
    print(f"Difference (Adaptive - Fixed): {adaptive_acc - fixed_acc:+.4f}")

    # add predictions to manifest for breakdown analysis
    manifest_df = manifest_df.reset_index(drop=True)
    manifest_df["true_label"] = labels
    manifest_df["fixed_pred"] = fixed_preds
    manifest_df["adaptive_pred"] = adaptive_preds
    manifest_df["fixed_correct"] = (fixed_preds == labels)
    manifest_df["adaptive_correct"] = (adaptive_preds == labels)

    if "codec" in manifest_df.columns:
        print(f"\n{'=' * 60}")
        print("BREAKDOWN BY CODEC")
        print(f"{'=' * 60}")
        codec_summary = manifest_df.groupby("codec").agg(
            n=("fixed_correct", "size"),
            fixed_acc=("fixed_correct", "mean"),
            adaptive_acc=("adaptive_correct", "mean"),
        )
        codec_summary["difference"] = codec_summary["adaptive_acc"] - codec_summary["fixed_acc"]
        print(codec_summary.to_string(float_format=lambda x: f"{x:.4f}"))

    if "channel" in manifest_df.columns:
        print(f"\n{'=' * 60}")
        print("BREAKDOWN BY CHANNEL")
        print(f"{'=' * 60}")
        channel_summary = manifest_df.groupby("channel").agg(
            n=("fixed_correct", "size"),
            fixed_acc=("fixed_correct", "mean"),
            adaptive_acc=("adaptive_correct", "mean"),
        )
        channel_summary["difference"] = channel_summary["adaptive_acc"] - channel_summary["fixed_acc"]
        print(channel_summary.to_string(float_format=lambda x: f"{x:.4f}"))

    manifest_df.to_csv("evaluation_2021_results.csv", index=False)
    print(f"\nDetailed results saved to: evaluation_2021_results.csv")


if __name__ == "__main__":
    main()
