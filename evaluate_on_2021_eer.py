"""
evaluate_on_2021_eer.py

Same as evaluate_on_2021.py, but ALSO saves a per-sample score for
Fixed Fusion and Adaptive Fusion so EER can be computed.
Does not overwrite your old results: writes evaluation_2021_results_scores.csv

Usage:
    python evaluate_on_2021_eer.py
"""

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from multiview_model import CachedMultiViewDataset, FixedFusionModel, AdaptiveFusionModel

MANIFEST_PATH = "feature_cache_manifest_eval.csv"
OUTPUT_CSV = "evaluation_2021_results_scores.csv"
BATCH_SIZE = 8


def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels, all_scores = [], [], []

    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            labels = batch["label"]

            output = model(inputs)
            if isinstance(output, tuple):
                output = output[0]

            preds = output.argmax(dim=1).cpu().numpy()
            # score = how much the model favours class 1 over class 0
            scores = (output[:, 1] - output[:, 0]).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_scores.extend(scores)

    return np.array(all_preds), np.array(all_labels), np.array(all_scores)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    manifest_df = pd.read_csv(MANIFEST_PATH).reset_index(drop=True)
    print(f"Total 2021 LA samples: {len(manifest_df)}")

    dataset = CachedMultiViewDataset(MANIFEST_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    fixed_model = FixedFusionModel().to(device)
    fixed_model.load_state_dict(torch.load("fixed_fusion.pth", map_location=device))

    adaptive_model = AdaptiveFusionModel().to(device)
    adaptive_model.load_state_dict(torch.load("adaptive_fusion.pth", map_location=device))

    print("Evaluating Fixed Fusion...")
    fixed_preds, labels, fixed_scores = evaluate_model(fixed_model, loader, device)

    print("Evaluating Adaptive Fusion...")
    adaptive_preds, _, adaptive_scores = evaluate_model(adaptive_model, loader, device)

    fixed_acc = (fixed_preds == labels).mean()
    adaptive_acc = (adaptive_preds == labels).mean()
    print(f"\nFixed Fusion accuracy:    {fixed_acc:.4f}")
    print(f"Adaptive Fusion accuracy: {adaptive_acc:.4f}")
    print("(These should match your earlier numbers: ~0.8650 and ~0.9556)")

    manifest_df["true_label"] = labels
    manifest_df["fixed_pred"] = fixed_preds
    manifest_df["adaptive_pred"] = adaptive_preds
    manifest_df["fixed_correct"] = (fixed_preds == labels)
    manifest_df["adaptive_correct"] = (adaptive_preds == labels)
    manifest_df["fixed_score"] = fixed_scores
    manifest_df["adaptive_score"] = adaptive_scores

    manifest_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
