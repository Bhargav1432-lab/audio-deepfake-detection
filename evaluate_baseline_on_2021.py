"""
evaluate_baseline_on_2021.py

Completes the 3-way comparison by evaluating the ORIGINAL AASIST baseline
(Model 1 -- single view, no fusion) on the same ASVspoof2021 LA samples
used to test Fixed Fusion and Adaptive Fusion. Uses the same sample set
(from feature_cache_manifest_eval.csv) for a fair, apples-to-apples
comparison across all 3 models.

Usage:
    python evaluate_baseline_on_2021.py
"""

import json
import numpy as np
import pandas as pd
import librosa
import torch
from importlib import import_module

CONFIG_PATH = "config/AASIST.conf"
AASIST_WEIGHTS = "models/weights/AASIST.pth"
MANIFEST_PATH = "feature_cache_manifest_eval.csv"
NB_SAMP = 64600


def pad_or_trim(audio, target_len=NB_SAMP):
    if len(audio) >= target_len:
        return audio[:target_len]
    num_repeats = int(target_len / len(audio)) + 1
    return np.tile(audio, num_repeats)[:target_len]


def load_aasist(device):
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    model_config = config["model_config"]

    module = import_module("models.AASIST")
    Model = getattr(module, "Model")
    model = Model(model_config).to(device)
    model.load_state_dict(torch.load(AASIST_WEIGHTS, map_location=device))
    model.eval()
    return model


def predict_single(model, filepath, device):
    audio, _ = librosa.load(filepath, sr=16000)
    audio = pad_or_trim(audio.astype(np.float32))
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        _, output = model(audio_tensor)
        pred = output.argmax(dim=1).item()
    return pred  # 1 = bonafide, 0 = spoof


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading baseline AASIST model...")
    model = load_aasist(device)

    print("Loading manifest (same samples used for Fixed/Adaptive fusion eval)...")
    manifest_df = pd.read_csv(MANIFEST_PATH)
    print(f"Total samples: {len(manifest_df)}")

    correct = 0
    total = 0
    errors = 0
    results = []

    print("\nEvaluating baseline on ASVspoof2021 LA...")
    for idx, row in manifest_df.iterrows():
        true_is_bonafide = 1 if row["label"] == "bonafide" else 0

        try:
            pred = predict_single(model, row["original_filepath"], device)
        except Exception as e:
            errors += 1
            if errors <= 5:  # only print first few to avoid spam
                print(f"  ERROR on {row['original_filepath']}: {e}")
            continue

        is_correct = (pred == true_is_bonafide)
        correct += int(is_correct)
        total += 1

        result_row = {
            "filepath": row["original_filepath"],
            "label": row["label"],
            "baseline_pred": pred,
            "baseline_correct": is_correct,
        }
        if "codec" in row:
            result_row["codec"] = row["codec"]
            result_row["channel"] = row["channel"]
        results.append(result_row)

        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(manifest_df)} done")

    accuracy = correct / total if total > 0 else 0

    print(f"\n{'=' * 60}")
    print("BASELINE (Model 1) RESULTS ON ASVSPOOF2021 LA")
    print(f"{'=' * 60}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total}, {errors} errors)")

    results_df = pd.DataFrame(results)

    if "codec" in results_df.columns:
        print(f"\nBreakdown by codec:")
        codec_acc = results_df.groupby("codec")["baseline_correct"].mean()
        for codec, acc in codec_acc.items():
            print(f"  {codec}: {acc:.4f}")

    if "channel" in results_df.columns:
        print(f"\nBreakdown by channel:")
        channel_acc = results_df.groupby("channel")["baseline_correct"].mean()
        for channel, acc in channel_acc.items():
            print(f"  {channel}: {acc:.4f}")

    results_df.to_csv("baseline_2021_results.csv", index=False)
    print(f"\nSaved to baseline_2021_results.csv")

    print(f"\n{'=' * 60}")
    print("FULL 3-WAY COMPARISON (combine with your earlier results)")
    print(f"{'=' * 60}")
    print(f"Model 1 (Baseline AASIST): {accuracy:.4f}")
    print(f"Model 2 (Fixed Fusion):    0.8650  (from your earlier run)")
    print(f"Model 3 (Adaptive Fusion): 0.9556  (from your earlier run)")


if __name__ == "__main__":
    main()
