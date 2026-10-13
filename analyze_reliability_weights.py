"""
analyze_reliability_weights.py

Extracts the reliability gate's learned weights (how much it trusts
magnitude/phase/residual/prosody) for every sample in the 2021 LA eval
set, then breaks this down by codec and channel condition.

This answers: "does the gate systematically shift trust toward certain
branches under degraded conditions?" -- the key scientific insight behind
your adaptive fusion result.

Usage:
    python analyze_reliability_weights.py
"""

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from multiview_model import CachedMultiViewDataset, AdaptiveFusionModel

MANIFEST_PATH = "feature_cache_manifest_eval.csv"
BATCH_SIZE = 8
BRANCH_NAMES = ["magnitude", "phase", "residual", "prosody"]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    manifest_df = pd.read_csv(MANIFEST_PATH).reset_index(drop=True)
    dataset = CachedMultiViewDataset(MANIFEST_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("Loading trained adaptive fusion model...")
    model = AdaptiveFusionModel().to(device)
    model.load_state_dict(torch.load("adaptive_fusion.pth", map_location=device))
    model.eval()

    all_weights = []

    print("Extracting reliability weights for all samples...")
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            _, weights = model(inputs, return_weights=True)
            all_weights.append(weights.cpu().numpy())

    all_weights = np.concatenate(all_weights, axis=0)  # shape: (N, 4)

    # attach weights to manifest
    for i, branch in enumerate(BRANCH_NAMES):
        manifest_df[f"weight_{branch}"] = all_weights[:, i]

    print(f"\n{'=' * 70}")
    print("OVERALL AVERAGE RELIABILITY WEIGHTS (across all 2021 LA samples)")
    print(f"{'=' * 70}")
    overall_avg = manifest_df[[f"weight_{b}" for b in BRANCH_NAMES]].mean()
    for branch in BRANCH_NAMES:
        print(f"  {branch:12s}: {overall_avg[f'weight_{branch}']:.4f}")

    if "codec" in manifest_df.columns:
        print(f"\n{'=' * 70}")
        print("AVERAGE RELIABILITY WEIGHTS BY CODEC")
        print(f"{'=' * 70}")
        codec_weights = manifest_df.groupby("codec")[
            [f"weight_{b}" for b in BRANCH_NAMES]
        ].mean()
        codec_weights.columns = BRANCH_NAMES
        print(codec_weights.to_string(float_format=lambda x: f"{x:.4f}"))

        print("\nWhich branch is MOST trusted per codec:")
        most_trusted = codec_weights.idxmax(axis=1)
        for codec, branch in most_trusted.items():
            print(f"  {codec:8s} -> {branch}")

    if "channel" in manifest_df.columns:
        print(f"\n{'=' * 70}")
        print("AVERAGE RELIABILITY WEIGHTS BY CHANNEL")
        print(f"{'=' * 70}")
        channel_weights = manifest_df.groupby("channel")[
            [f"weight_{b}" for b in BRANCH_NAMES]
        ].mean()
        channel_weights.columns = BRANCH_NAMES
        print(channel_weights.to_string(float_format=lambda x: f"{x:.4f}"))

        print("\nWhich branch is MOST trusted per channel:")
        most_trusted = channel_weights.idxmax(axis=1)
        for channel, branch in most_trusted.items():
            print(f"  {channel:8s} -> {branch}")

    # also compare bonafide vs spoof
    print(f"\n{'=' * 70}")
    print("AVERAGE RELIABILITY WEIGHTS: BONAFIDE vs SPOOF")
    print(f"{'=' * 70}")
    label_weights = manifest_df.groupby("label")[
        [f"weight_{b}" for b in BRANCH_NAMES]
    ].mean()
    label_weights.columns = BRANCH_NAMES
    print(label_weights.to_string(float_format=lambda x: f"{x:.4f}"))

    manifest_df.to_csv("reliability_weights_analysis.csv", index=False)
    print(f"\nFull per-sample weights saved to: reliability_weights_analysis.csv")


if __name__ == "__main__":
    main()
